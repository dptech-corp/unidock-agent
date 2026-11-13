"""Comprehensive validation of per-residue energy decomposition correctness.

Tests:
  1. Sum consistency: sum(residue) == sum(atom) for all 5 terms + total
  2. Component additivity: residue_total == sum of 5 residue components
  3. Non-negativity of repulsion term
  4. HBond sign: hbond contributions must be <= 0
  5. Cross-pose residue label stability: all poses share same residue set
  6. Energy conservation across poses: different poses, same total energy formula
  7. Sparsity check: most residues should have zero energy (far from ligand)
  8. Label format validation: chain:resname:resid with valid types

Usage:
  python validate_residue_decomp.py <sdf_file>
"""
import sys
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ATOM_PROPS = ["atom_inter_energy", "atom_gauss1", "atom_gauss2",
              "atom_repulsion", "atom_hydrophobic", "atom_hbond"]
RES_PROPS = ["residue_inter_energy", "residue_gauss1", "residue_gauss2",
             "residue_repulsion", "residue_hydrophobic", "residue_hbond"]
VALID_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "ACE", "NME",  # protonation variants + caps
}
TOL = 0.02  # kcal/mol tolerance


def parse_prop(mol, prop):
    return [float(x) for x in mol.GetProp(prop).split()]


def run_tests(sdf_path):
    poses = [m for m in Chem.SDMolSupplier(sdf_path, removeHs=False) if m is not None]
    if not poses:
        print("FAIL: No poses loaded")
        return False

    print(f"Loaded {len(poses)} poses from {sdf_path}")
    print("=" * 70)

    n_pass = 0
    n_fail = 0
    n_skip = 0

    def check(name, condition, detail=""):
        nonlocal n_pass, n_fail
        if condition:
            n_pass += 1
            print(f"  PASS: {name}")
        else:
            n_fail += 1
            print(f"  FAIL: {name} — {detail}")

    def skip(name, reason):
        nonlocal n_skip
        n_skip += 1
        print(f"  SKIP: {name} — {reason}")

    # Check props exist
    mol0 = poses[0]
    has_atom = all(mol0.HasProp(p) for p in ATOM_PROPS)
    has_res = all(mol0.HasProp(p) for p in RES_PROPS) and mol0.HasProp("residue_labels")

    if not has_atom:
        print("FAIL: Missing atom decomposition properties")
        return False
    if not has_res:
        print("FAIL: Missing residue decomposition properties")
        return False

    # ======================================================================
    # Test 1: Sum consistency — sum(residue) == sum(atom) for each term
    # ======================================================================
    print("\n[Test 1] Sum consistency: sum(residue) == sum(atom) per term")
    term_pairs = list(zip(ATOM_PROPS, RES_PROPS))
    for pose_idx, mol in enumerate(poses):
        for atom_prop, res_prop in term_pairs:
            atom_vals = parse_prop(mol, atom_prop)
            res_vals = parse_prop(mol, res_prop)
            atom_sum = sum(atom_vals)
            rest_prop = "decomp_unwritten_" + atom_prop[len("atom_"):]
            if mol.HasProp(rest_prop):  # flex atoms beyond the written molecule (sdf_natom)
                atom_sum += float(mol.GetProp(rest_prop))
            res_sum = sum(res_vals)
            diff = abs(atom_sum - res_sum)
            term_name = atom_prop.replace("atom_", "")
            check(
                f"pose {pose_idx} {term_name}: atom_sum={atom_sum:.4f} res_sum={res_sum:.4f}",
                diff < TOL,
                f"diff={diff:.6f}"
            )

    # ======================================================================
    # Test 2: Component additivity — total == sum of 5 components per residue
    # ======================================================================
    print("\n[Test 2] Component additivity: residue_total == g1+g2+rep+hydro+hb")
    for pose_idx, mol in enumerate(poses):
        total = parse_prop(mol, "residue_inter_energy")
        g1 = parse_prop(mol, "residue_gauss1")
        g2 = parse_prop(mol, "residue_gauss2")
        rep = parse_prop(mol, "residue_repulsion")
        hyd = parse_prop(mol, "residue_hydrophobic")
        hb = parse_prop(mol, "residue_hbond")

        max_err = 0
        worst_res = -1
        for k in range(len(total)):
            expected = g1[k] + g2[k] + rep[k] + hyd[k] + hb[k]
            err = abs(total[k] - expected)
            if err > max_err:
                max_err = err
                worst_res = k

        check(
            f"pose {pose_idx}: max component error = {max_err:.6f}",
            max_err < 0.001,
            f"worst at residue index {worst_res}"
        )

    # ======================================================================
    # Test 3: Repulsion is non-negative
    # ======================================================================
    print("\n[Test 3] Repulsion non-negativity")
    for pose_idx, mol in enumerate(poses):
        rep = parse_prop(mol, "residue_repulsion")
        min_rep = min(rep)
        check(
            f"pose {pose_idx}: min repulsion = {min_rep:.6f}",
            min_rep >= -0.001,
            f"negative repulsion found"
        )

    # ======================================================================
    # Test 4: HBond sign — must be <= 0
    # ======================================================================
    print("\n[Test 4] HBond sign: all contributions <= 0")
    for pose_idx, mol in enumerate(poses):
        hb = parse_prop(mol, "residue_hbond")
        max_hb = max(hb)
        check(
            f"pose {pose_idx}: max hbond = {max_hb:.6f}",
            max_hb <= 0.001,
            f"positive hbond found"
        )

    # ======================================================================
    # Test 5: Cross-pose residue label stability
    # ======================================================================
    print("\n[Test 5] Cross-pose residue label stability")
    ref_labels = poses[0].GetProp("residue_labels")
    for pose_idx in range(1, len(poses)):
        labels = poses[pose_idx].GetProp("residue_labels")
        check(
            f"pose {pose_idx} labels match pose 0",
            labels == ref_labels,
            f"label mismatch"
        )

    # ======================================================================
    # Test 6: Array length consistency
    # ======================================================================
    print("\n[Test 6] Array length consistency")
    for pose_idx, mol in enumerate(poses):
        labels = mol.GetProp("residue_labels").split()
        nres = len(labels)
        for prop in RES_PROPS:
            vals = parse_prop(mol, prop)
            check(
                f"pose {pose_idx} {prop}: len={len(vals)} == nres={nres}",
                len(vals) == nres,
                f"length mismatch"
            )

    # ======================================================================
    # Test 7: Label format validation
    # ======================================================================
    print("\n[Test 7] Label format: chain:resname:resid")
    labels = poses[0].GetProp("residue_labels").split()
    for label in labels:
        parts = label.split(":")
        ok = len(parts) == 3
        if ok:
            chain, resname, resid_str = parts
            ok = len(chain) > 0 and len(resname) > 0 and resid_str.lstrip("-").isdigit()
        check(f"label '{label}'", ok, "bad format")

    # ======================================================================
    # Test 8: Sparsity — most residues should have near-zero energy
    # ======================================================================
    print("\n[Test 8] Sparsity check")
    labels = poses[0].GetProp("residue_labels").split()
    nres = len(labels)
    total = parse_prop(poses[0], "residue_inter_energy")
    nonzero = sum(1 for e in total if abs(e) > 0.001)
    ratio = nonzero / nres
    check(
        f"{nonzero}/{nres} residues nonzero ({ratio:.1%})",
        ratio < 0.5,
        "too many nonzero residues — ligand shouldn't contact >50% of box residues"
    )

    # ======================================================================
    # Test 9: Gauss1 and Gauss2 are non-positive (attractive)
    # ======================================================================
    print("\n[Test 9] Gauss1/Gauss2 sign: attractive (non-positive)")
    for pose_idx, mol in enumerate(poses):
        g1 = parse_prop(mol, "residue_gauss1")
        g2 = parse_prop(mol, "residue_gauss2")
        max_g1 = max(g1)
        max_g2 = max(g2)
        check(
            f"pose {pose_idx}: max gauss1={max_g1:.6f}, max gauss2={max_g2:.6f}",
            max_g1 <= 0.001 and max_g2 <= 0.001,
            "positive gaussian found"
        )

    # ======================================================================
    # Test 10: Hydrophobic is non-positive (attractive)
    # ======================================================================
    print("\n[Test 10] Hydrophobic sign: attractive (non-positive)")
    for pose_idx, mol in enumerate(poses):
        hyd = parse_prop(mol, "residue_hydrophobic")
        max_hyd = max(hyd)
        check(
            f"pose {pose_idx}: max hydrophobic={max_hyd:.6f}",
            max_hyd <= 0.001,
            "positive hydrophobic found"
        )

    # ======================================================================
    # Test 11: Per-term sum(residue) == sum(atom) consistency for hbond
    #          (verifies angular HBond mask propagation)
    # ======================================================================
    print("\n[Test 11] HBond mask propagation: residue hbond sum == atom hbond sum")
    for pose_idx, mol in enumerate(poses):
        atom_hb = parse_prop(mol, "atom_hbond")
        res_hb = parse_prop(mol, "residue_hbond")
        atom_sum = sum(atom_hb)
        res_sum = sum(res_hb)
        diff = abs(atom_sum - res_sum)
        check(
            f"pose {pose_idx}: atom_hb_sum={atom_sum:.4f} res_hb_sum={res_sum:.4f}",
            diff < TOL,
            f"diff={diff:.6f} — hbond mask may not propagate correctly"
        )

    # ======================================================================
    # Test 12: Ranking stability — top residues should be consistent
    # ======================================================================
    print("\n[Test 12] Top-5 residue ranking across poses")
    labels = poses[0].GetProp("residue_labels").split()
    top5_sets = []
    for pose_idx, mol in enumerate(poses):
        total = parse_prop(mol, "residue_inter_energy")
        ranked = sorted(zip(labels, total), key=lambda x: x[1])
        top5 = set(r[0] for r in ranked[:5])
        top5_sets.append(top5)
        print(f"    pose {pose_idx} top-5: {', '.join(r[0] for r in ranked[:5])}")

    # Check overlap between pose 0 and others
    ref_top5 = top5_sets[0]
    for pose_idx in range(1, len(poses)):
        overlap = len(ref_top5 & top5_sets[pose_idx])
        # Not a strict pass/fail — just informational
        print(f"    pose 0 vs {pose_idx}: {overlap}/5 overlap")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print(f"RESULTS: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print("=" * 70)
    return n_fail == 0


if __name__ == "__main__":
    sdf = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_resdecomp/output_10pose/ligand_ejm42_enriched_1.sdf"
    success = run_tests(sdf)
    sys.exit(0 if success else 1)
