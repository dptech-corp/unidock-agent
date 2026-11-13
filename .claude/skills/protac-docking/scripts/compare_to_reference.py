#!/usr/bin/env python3
"""Compare docked PROTAC poses against a reference structure.

Reports heavy-atom RMSD *without superposition*. That is deliberate: the
receptor is common to both coordinate frames, so the deviation you care about is
the absolute one. Superposing (PyMOL's `align`, RDKit's `AlignMol`) would fit the
two ligands onto each other and erase exactly the displacement being measured.

Atom correspondence comes from a substructure match on the heavy-atom skeleton
with bond orders ignored, so a reference taken from a PDB (proximity bonding)
still matches a docked SDF. Symmetry-equivalent mappings are enumerated and the
best one used, so a flipped phenyl does not inflate the number.

Usage
-----
    compare_to_reference.py --reference crystal.sdf \
                            --poses out/poses/representatives_protac.sdf

    # reference taken straight from a PDB ligand
    compare_to_reference.py --reference 5t35.pdb --ligand-resname 759 \
                            --chain D --poses representatives_protac.sdf
"""
import argparse
import sys

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def load_reference(path, resname=None, chain=None):
    """Reference may be an SDF, or a PDB from which a ligand is extracted."""
    if path.lower().endswith((".sdf", ".mol")):
        mol = Chem.SDMolSupplier(path, removeHs=False)[0]
        if mol is None:
            sys.exit(f"could not parse {path}")
        return mol

    lines = [
        l for l in open(path)
        if l.startswith("HETATM")
        and (resname is None or l[17:20].strip() == resname)
        and (chain is None or l[21] == chain)
    ]
    if not lines:
        sys.exit(f"no HETATM records in {path} matching "
                 f"resname={resname} chain={chain}")
    mol = Chem.MolFromPDBBlock("".join(lines) + "END\n",
                               removeHs=False, sanitize=False)
    if mol is None:
        sys.exit(f"could not build a molecule from {path}")
    return mol


def skeleton(mol):
    """Elements + connectivity only.

    A PDB-derived reference is built by proximity bonding and will not agree
    with an SDF on bond orders or aromaticity, so matching on the bare skeleton
    is what lets the two correspond.
    """
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    for b in rw.GetBonds():
        b.SetIsAromatic(False)
        b.SetBondType(Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_SYMMRINGS |
                     Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    return out


def best_rmsd(pose, ref_heavy_xyz, matches, pose_heavy_idx):
    """Minimum over symmetry-equivalent atom mappings, no superposition."""
    P = pose.GetConformer().GetPositions()
    best = None
    for m in matches:
        d = P[[pose_heavy_idx[i] for i in range(len(m))]] - ref_heavy_xyz[list(m)]
        # `m` maps pose-skeleton atom i -> reference atom m[i]
        v = float(np.sqrt((d ** 2).sum(1).mean()))
        best = v if best is None else min(best, v)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True,
                    help="reference ligand: .sdf, or a .pdb plus --ligand-resname")
    ap.add_argument("--poses", required=True,
                    help="SDF of docked poses (e.g. representatives_protac.sdf)")
    ap.add_argument("--ligand-resname", default=None,
                    help="HETATM residue name, when --reference is a PDB")
    ap.add_argument("--chain", default=None,
                    help="chain id, when --reference is a PDB with several copies")
    ap.add_argument("--top", type=int, default=10,
                    help="how many poses to print (default 10; 0 = all)")
    ap.add_argument("--good", type=float, default=2.0,
                    help="RMSD below which a pose is counted as near-reference")
    args = ap.parse_args()

    ref = load_reference(args.reference, args.ligand_resname, args.chain)
    poses = [m for m in Chem.SDMolSupplier(args.poses, removeHs=False) if m]
    if not poses:
        sys.exit(f"no readable poses in {args.poses}")

    ref_sk = skeleton(Chem.RemoveHs(ref))
    ref_heavy_xyz = Chem.RemoveHs(ref).GetConformer().GetPositions()

    probe = Chem.RemoveHs(poses[0])
    matches = skeleton(probe).GetSubstructMatches(ref_sk, uniquify=False,
                                                  maxMatches=2000)
    if not matches:
        sys.exit("reference does not match the docked poses — check that they "
                 "are the same molecule and that bonds were perceived sensibly")

    # heavy-atom indices in the full (with-H) pose, in RemoveHs order
    pose_heavy_idx = [a.GetIdx() for a in poses[0].GetAtoms()
                      if a.GetAtomicNum() > 1]

    rows = []
    for i, p in enumerate(poses, 1):
        # the score the pipeline ranks by (vina_pose_score); older outputs only carry the binding free energy
        key = "vina_pose_score" if p.HasProp("vina_pose_score") else "vina_binding_free_energy"
        score = p.GetProp(key) if p.HasProp(key) else ""
        cid = p.GetProp("cluster_id") if p.HasProp("cluster_id") else ""
        rows.append((i, cid, score,
                     best_rmsd(p, ref_heavy_xyz, matches, pose_heavy_idx)))

    n = len(rows) if args.top == 0 else min(args.top, len(rows))
    print(f"reference : {args.reference}")
    print(f"poses     : {args.poses}  ({len(rows)} poses, "
          f"{len(matches)} symmetry mappings)")
    print("RMSD is heavy-atom, no superposition\n")
    print(f"{'rank':>5} {'cluster':>8} {'score':>10} {'RMSD':>8}")
    print("-" * 35)
    for r, cid, sc, v in rows[:n]:
        print(f"{r:>5} {cid:>8} {sc:>10} {v:>8.3f}")

    vals = np.array([r[3] for r in rows])
    best_i = int(vals.argmin())
    print("-" * 35)
    print(f"best     : {vals.min():.3f} A at rank {best_i + 1}")
    print(f"under {args.good:g} A: {(vals < args.good).sum()} of {len(vals)}")
    if best_i != 0:
        print(f"NOTE: rank 1 ({vals[0]:.3f} A) is not the closest pose. "
              f"Scoring put a worse pose first — worth reporting.")


if __name__ == "__main__":
    main()
