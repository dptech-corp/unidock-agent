#!/usr/bin/env python3
"""Screen a library of degraders against one fixed target + E3 pair.

The pipeline docks a single PROTAC per invocation: it reads
`SDMolSupplier(...)[0]` and silently ignores the rest of the file, so handing it
a multi-molecule library docks only the first entry with no warning. It also
writes its prep artifacts (protac_ligand.dms, protac_receptor.dms,
protac_temp_cursor.sdf) into the *input* directory under fixed names, so runs
sharing a directory overwrite each other.

This script handles both: it splits the library, gives every degrader its own
working directory with the shared target/E3 files linked in, runs them one at a
time, and collects the results into a summary table.

Usage
-----
    screen_degraders.py --library degraders.sdf \
        --tp-sdf target_warhead.sdf --ep-sdf e3_warhead.sdf \
        --receptor-pdb target.pdb --e3-pdb CRBN.pdb \
        --outdir screen_out \
        --n-runs 8 --exhaustiveness 1024 --mc-steps 40
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DRIVER = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "examples", "protac_docking",
    "run_protac_docking.py"))


def safe_name(mol, idx):
    raw = (mol.GetProp("_Name") if mol.HasProp("_Name") else "").strip()
    if not raw:
        raw = f"degrader_{idx:04d}"
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)[:64]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True,
                    help="multi-molecule SDF of degraders")
    ap.add_argument("--tp-sdf", required=True, help="target warhead SDF")
    ap.add_argument("--ep-sdf", required=True, help="E3 warhead SDF")
    ap.add_argument("--receptor-pdb", required=True)
    ap.add_argument("--e3-pdb", required=True)
    ap.add_argument("--outdir", default="screen_out")
    ap.add_argument("--driver", default=DEFAULT_DRIVER,
                    help="path to run_protac_docking.py")
    ap.add_argument("--python", default=sys.executable)
    # sampling — defaults are the settings that actually work, not the
    # pipeline's own defaults, which undersample badly for a PROTAC
    ap.add_argument("--n-runs", type=int, default=8)
    ap.add_argument("--exhaustiveness", type=int, default=1024)
    ap.add_argument("--mc-steps", type=int, default=40)
    ap.add_argument("--num-pose", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0,
                    help="only screen the first N degraders (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="set up the directories and print commands, run nothing")
    args = ap.parse_args()

    if not os.path.isfile(args.driver):
        sys.exit(f"driver not found: {args.driver}\nPass --driver explicitly.")

    shared = {
        "tp": args.tp_sdf, "ep": args.ep_sdf,
        "rec": args.receptor_pdb, "e3": args.e3_pdb,
    }
    for k, v in shared.items():
        if not os.path.isfile(v):
            sys.exit(f"missing shared input ({k}): {v}")

    mols = [m for m in Chem.SDMolSupplier(args.library, removeHs=False) if m]
    if not mols:
        sys.exit(f"no readable molecules in {args.library}")
    if args.limit:
        mols = mols[:args.limit]
    print(f"screening {len(mols)} degraders from {args.library}\n")

    os.makedirs(args.outdir, exist_ok=True)
    rows = []

    for i, mol in enumerate(mols, 1):
        name = safe_name(mol, i)
        work = os.path.join(args.outdir, name)
        os.makedirs(work, exist_ok=True)

        # one degrader per directory, so the fixed-name prep artifacts cannot
        # collide between library members
        lig = os.path.join(work, "degrader.sdf")
        w = Chem.SDWriter(lig)
        w.write(mol)
        w.close()
        for key, src in shared.items():
            dst = os.path.join(work, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        cmd = [
            args.python, args.driver,
            "--input-dir", work,
            "--protac-sdf", "degrader.sdf",
            "--tp-sdf", os.path.basename(shared["tp"]),
            "--ep-sdf", os.path.basename(shared["ep"]),
            "--receptor-pdb", os.path.basename(shared["rec"]),
            "--e3-pdb", os.path.basename(shared["e3"]),
            "--output-dir", os.path.join(work, "out"),
            "--n-runs", str(args.n_runs),
            "--exhaustiveness", str(args.exhaustiveness),
            "--mc-steps", str(args.mc_steps),
            "--num-pose", str(args.num_pose),
        ]

        if args.dry_run:
            print(f"[{i}/{len(mols)}] {name}\n    {' '.join(cmd)}")
            continue

        print(f"[{i}/{len(mols)}] {name} ... ", end="", flush=True)
        t0 = time.time()
        log = os.path.join(work, "run.log")
        with open(log, "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0

        rep = os.path.join(work, "out", "poses", "representatives_protac.sdf")
        best, n_clusters = "", 0
        if r.returncode == 0 and os.path.isfile(rep):
            reps = [m for m in Chem.SDMolSupplier(rep, removeHs=False) if m]
            n_clusters = len(reps)
            if reps and reps[0].HasProp("vina_binding_free_energy"):
                best = reps[0].GetProp("vina_binding_free_energy")
            print(f"ok  best {best}  clusters {n_clusters}  ({dt:.0f}s)")
        else:
            print(f"FAILED rc={r.returncode}  see {log}")

        rows.append(dict(name=name, status="ok" if r.returncode == 0 else "failed",
                         best_score=best, n_clusters=n_clusters,
                         seconds=round(dt, 1),
                         representatives=rep if os.path.isfile(rep) else ""))

    if args.dry_run or not rows:
        return

    summary = os.path.join(args.outdir, "screen_summary.csv")
    with open(summary, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok" and r["best_score"]]
    ok.sort(key=lambda r: float(r["best_score"]))
    print(f"\nwrote {summary}")
    print(f"{len(ok)}/{len(rows)} succeeded\n")
    print(f"{'rank':>5} {'degrader':<28} {'best score':>11} {'clusters':>9}")
    print("-" * 56)
    for k, r in enumerate(ok[:15], 1):
        print(f"{k:>5} {r['name']:<28} {r['best_score']:>11} "
              f"{r['n_clusters']:>9}")
    if ok:
        print("\nScores rank degraders only loosely — the margin between a "
              "correct and an incorrect pose was ~0.1 kcal/mol on the one "
              "system with a known answer. Inspect the top poses rather than "
              "trusting this ordering.")


if __name__ == "__main__":
    main()
