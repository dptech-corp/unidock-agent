---
name: protac-docking
description: Run PROTAC / degrader ternary-complex docking with Uni-Dock2 — screening degraders against a target and E3 ligase, sampling linker conformations, and predicting where the E3 lands. Use this whenever the user mentions a PROTAC, degrader, ternary complex, E3 ligase (VHL, CRBN, cereblon, DCAF), warhead, or linker, or asks to dock/model/screen/rank bifunctional molecules that bridge two proteins — even if they don't name Uni-Dock2. Also use when interpreting such a run's output (representatives, cluster_info.csv), choosing exhaustiveness or sampling settings, or validating a predicted ternary pose against a crystal structure.
---

# PROTAC ternary-complex docking (Uni-Dock2)

## What the method predicts

A PROTAC bridges a target protein and an E3 ligase. This workflow takes **both
warhead poses as given** — each warhead already positioned in its own binary
complex — and searches only the **linker conformation**. Where the E3 ligase
ends up is the prediction.

Keep that straight when reporting. "We reproduced the ternary complex" overstates
it; "given both warhead poses, the linker search placed the E3 within X Å of
crystallographic" is what the method supports.

One command runs four stages: prep (MCS-constrained alignment → torsion tree →
DMS), docking, post-processing, clustering.

## Screening a degrader series

The usual situation is a fixed target and a fixed E3 — the E3 side is drawn from
a small set (VHL, CRBN) and reused — while the degraders vary. Two properties of
the pipeline make that harder than it looks:

- **It docks one PROTAC per invocation.** It reads `SDMolSupplier(...)[0]` and
  ignores the rest of the file, with no warning. Hand it a 20-compound library
  and you get one result and no indication the other 19 were dropped.
- **Prep artifacts have fixed names and land in the input directory**
  (`protac_ligand.dms`, `protac_receptor.dms`, `protac_temp_cursor.sdf`), so
  runs sharing a directory silently overwrite each other's inputs.

Use the bundled driver, which splits the library, isolates each degrader in its
own directory with the shared target/E3 files copied in, and collects a summary:

```bash
python <skill>/scripts/screen_degraders.py \
    --library degraders.sdf \
    --tp-sdf target_warhead.sdf --ep-sdf e3_warhead.sdf \
    --receptor-pdb target.pdb --e3-pdb CRBN.pdb \
    --outdir screen_out
```

It defaults to the sampling settings that actually work (below), not the
pipeline's own. `--dry-run` prints the plan without running; `--limit N` tries a
few first. Output is `screen_out/screen_summary.csv` plus a per-degrader
directory holding the full result.

**Comparing scores across degraders is weak.** The score is a Vina affinity
whose margin between a correct and an incorrect pose was ~0.1 kcal/mol on the one
system with a known answer — far smaller than the differences you would want to
act on. Treat the ranking as triage for which poses to inspect, not as an
affinity prediction. Chemically different degraders also carry different torsion
counts, which shifts the score's conformational-entropy term independently of
binding.

## The input contract

Five files. Most failures trace back here rather than to the docking.

| File | Requirement |
|---|---|
| target warhead SDF | same coordinate frame as the receptor PDB |
| E3 warhead SDF | same coordinate frame as the E3 PDB |
| PROTAC SDF | 3D conformer, contains both warheads as substructures |
| receptor PDB | protonated, complete |
| E3 PDB | protonated, complete |

Everything is read with explicit hydrogens. The PDBs must already be prepared by
whatever protein-prep tool you use; nothing here protonates or repairs them, and
protonation drives the donor/acceptor typing that scoring depends on.

The two frames are independent — computing the transform between them is the
point — but the alignment is not invariant to large rigid motions of the inputs
(see `references/method.md`). If your target and E3 complexes come from different
PDB entries, translate them near each other first.

**Warhead matching is element-agnostic but bond-order-exact.** The MCS uses
`CompareAnyHeavyAtom` with `CompareOrderExact`, so a wrong bond order silently
shrinks the match while a wrong element does not. If the mapped atom count in the
log is below the warhead's heavy-atom count, that is what to check — and in a
screen, check it for every compound, since one bad SDF entry fails quietly.

## Sampling decides success

The shipped defaults undersample badly for a PROTAC. On the MZ1-BRD4-VHL
benchmark, where the answer is known:

| settings | poses | best pose vs crystal |
|---|---|---|
| `--exhaustiveness 128 --mc-steps 20` (default) | 1024 | 1.34 Å |
| `--exhaustiveness 1024 --mc-steps 40` | 7760 | **0.67 Å** |

~8× the sampling, about 6 minutes per compound on 20 cores. Budget accordingly
for a series. Raise `--exhaustiveness` before `--mc-steps`: it buys independent
starting points, whereas mc-steps only lengthens each walk.

The search space is the linker torsions (11 for MZ1) with the whole E3 protein
swinging on the end, so small torsion changes near a warhead are hugely
amplified — hence the need for many starts.

## Reading the results

In `<output-dir>/poses/`:

```
representatives_complex.pdb    MODEL 1 = top-ranked ternary complex
representatives_protac.sdf     one pose per cluster, best score first
representatives_<e3>.pdb       named after your E3 PDB
docked_protac_clustered.sdf    clustered top-500, each tagged cluster_id
cluster_info.csv               cluster id, size, representative pose, binding free energy, pose score
```

Each cluster contributes one pose — its best-scoring member — and clusters are
ordered by that score. The score is `vina_pose_score`, the engine's pose-pass score.
`vina_binding_free_energy` is still written but is not a ranking score: it
subtracts the intra energy of the first pose of each run, so it rewards poses
whose intra is lower than their run's top pose.

**Look at the top 2–3 representatives, not only the first.** On MZ1, over 5
seed sets of 8 runs each, the crystal-like pose (0.68–0.93 Å) was found every
time and ranked first by `vina_pose_score` in 5 of 5. Ranked by
`vina_binding_free_energy`, the same clusters put it first in only 3 of 5 (rank 5
and rank 19 in the others). One system does not make a scoring guarantee.

**Cluster size is not evidence.** A wide shallow basin collects more hits than a
narrow deep one. On the benchmark the correct cluster had 5 members and a wrong
one had 127.

Per-pose dumps are off by default; `--dump-all-poses` enables them but costs
several GB per run and nothing downstream reads them.

## Validating against a known structure

When a reference exists, measure rather than trust the score:

```bash
python <skill>/scripts/compare_to_reference.py \
    --reference crystal_ligand.sdf \
    --poses out/poses/representatives_protac.sdf
```

Reports heavy-atom RMSD **without superposition** for each representative, which
is the meaningful quantity — the receptor is common to both frames, so superposing
erases the displacement being measured. In PyMOL use `rms_cur`, never `align`.
It also accepts a raw PDB reference (`--reference 5t35.pdb --ligand-resname 759
--chain D`) and handles symmetry-equivalent mappings.

If the result is poor, `references/validation.md` gives the experiments that
separate "sampling never got there" from "scoring ranked it badly" — different
causes needing different fixes, in the order worth testing them.

## Alignment only

Steps 1–3 (embedding the PROTAC onto both warheads, plus the rigid-body
transform carrying the E3) need **no docking engine, no msys, no compiled code**
— just rdkit, numpy and networkx. Useful for checking warhead mapping across a
library before committing compute. Snippet in `references/method.md`.

The full pipeline re-randomises every linker torsion before sampling, so this
alignment is a scaffold, not a prediction.

## Known traps

- **`--opt-steps` does nothing.** It is overwritten inside the engine; every
  value gives identical results. Don't tune it or report it.
- **Import order crashes the engine.** Importing rdkit or scipy before the
  `pipeline` extension segfaults (duplicate OpenMP runtime), and calling
  `run_docking_pipeline` twice in one process also segfaults. The driver
  subprocesses each docking job for this reason — keep that pattern.
- **The box must enclose the E3 protein too**, not just the ligand. Shrinking
  `--box-size` silently discards poses.
- **Rebuilding the engine needs a nudge.** `build_cpu` has stale dependency
  scanning and reports "up to date" with objects older than the headers; touch
  the source first. And `cmake --build` does not copy the extension into
  `api/python/` — that copy is manual, and skipping it means running the old
  binary while believing otherwise.

## Reference material

- `references/method.md` — alignment mechanics, frame sensitivity, why warhead
  matching fails, the alignment-only snippet
- `references/validation.md` — rescoring a reference pose, the reachability
  floor, the convergence-radius experiment, and how to report honestly
