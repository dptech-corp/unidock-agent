"""Per-atom energy decomposition tests.

Converted from: test_decomp_e2e.py
"""
import os

import numpy as np
import pytest

from tests.helpers import load_dms_ligand, read_sdf_poses, run_docking_protocol


# ---------------------------------------------------------------------------
# Shared fixture: run docking once with decomp=True + ff_minimize_steps>0
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def decomp_poses(tyk2_ligand_dms, tyk2_protein_dms, tmp_path_factory):
    """Docking with decomp=True and ff_minimize_steps=5."""
    _, center = load_dms_ligand(tyk2_ligand_dms)
    tmpdir = str(tmp_path_factory.mktemp("decomp"))
    out = os.path.join(tmpdir, "docked_decomp.sdf")
    run_docking_protocol(
        receptor=tyk2_protein_dms,
        ligand_list=[tyk2_ligand_dms],
        center=center,
        output_sdf=out,
        work_dir=tmpdir,
        decomp=True,
        ff_minimize_steps=5,
        exhaustiveness=8,
        num_pose=3,
        seed=42,
    )
    return read_sdf_poses(out)


# ---------------------------------------------------------------------------
# Vina decomposition properties
# ---------------------------------------------------------------------------
VINA_DECOMP_PROPS = [
    "atom_inter_energy",
    "atom_gauss1",
    "atom_gauss2",
    "atom_repulsion",
    "atom_hydrophobic",
    "atom_hbond",
]

FF_DECOMP_PROPS = [
    "atom_ff_total",
    "atom_ff_inter_lj",
    "atom_ff_inter_coulomb",
    "atom_ff_intra_lj",
    "atom_ff_intra_coulomb",
    "atom_ff_torsion",
]


@pytest.mark.slow
def test_decomp_produces_poses(decomp_poses):
    """At least one pose should be produced."""
    assert len(decomp_poses) > 0


@pytest.mark.slow
def test_decomp_vina_properties_present(decomp_poses):
    """All Vina per-atom decomposition properties should be present."""
    mol = decomp_poses[0]
    for prop in VINA_DECOMP_PROPS:
        assert mol.HasProp(prop), f"Missing Vina decomp property: {prop}"


@pytest.mark.slow
def test_decomp_ff_properties_present(decomp_poses):
    """All FF per-atom decomposition properties should be present."""
    mol = decomp_poses[0]
    for prop in FF_DECOMP_PROPS:
        assert mol.HasProp(prop), f"Missing FF decomp property: {prop}"


@pytest.mark.slow
def test_decomp_ff_components_sum_to_total(decomp_poses):
    """Per-atom ff_total should equal sum of ff components within tolerance."""
    mol = decomp_poses[0]
    if not all(mol.HasProp(p) for p in FF_DECOMP_PROPS):
        pytest.skip("FF decomposition properties not present")

    ff_total = [float(x) for x in mol.GetProp("atom_ff_total").split()]
    ff_ilj = [float(x) for x in mol.GetProp("atom_ff_inter_lj").split()]
    ff_icol = [float(x) for x in mol.GetProp("atom_ff_inter_coulomb").split()]
    ff_alj = [float(x) for x in mol.GetProp("atom_ff_intra_lj").split()]
    ff_acol = [float(x) for x in mol.GetProp("atom_ff_intra_coulomb").split()]
    ff_tor = [float(x) for x in mol.GetProp("atom_ff_torsion").split()]
    # harmonic bond / angle parts, written when the ligand DMS has stretch_harm / angle_harm
    ff_bonded = [0.0] * len(ff_total)
    for prop in ("atom_ff_bond", "atom_ff_angle"):
        if mol.HasProp(prop):
            ff_bonded = [b + float(x) for b, x in zip(ff_bonded, mol.GetProp(prop).split())]

    max_err = 0.0
    for i in range(len(ff_total)):
        expected = ff_ilj[i] + ff_icol[i] + ff_alj[i] + ff_acol[i] + ff_tor[i] + ff_bonded[i]
        err = abs(ff_total[i] - expected)
        max_err = max(max_err, err)

    assert max_err < 0.01, (
        f"FF per-atom total vs component sum error too large: {max_err:.6f}"
    )


@pytest.mark.slow
def test_ff_total_is_inter_plus_intra(decomp_poses):
    """ff_intra_ligand_energy includes bond + angle, so the pose-level FF parts add up."""
    for mol in decomp_poses:
        if not mol.HasProp("ff_total_energy"):
            pytest.skip("FF energies not present")
        total = float(mol.GetProp("ff_total_energy"))
        inter = float(mol.GetProp("ff_interaction_energy"))
        intra = float(mol.GetProp("ff_intra_ligand_energy"))
        assert abs(total - inter - intra) < 0.002, (total, inter, intra)


@pytest.mark.slow
def test_decomp_reconciles_with_pose_score(decomp_poses):
    """The atoms decompose the inter part of vina_pose_score (print rounding only)."""
    for mol in decomp_poses:
        parts = [float(mol.GetProp(p)) for p in ("vina_pose_inter", "vina_pose_intra", "vina_pose_penalty")]
        assert abs(sum(parts) - float(mol.GetProp("vina_pose_score"))) < 0.002
        atoms = [float(x) for x in mol.GetProp("atom_inter_energy").split()]
        tol = 5e-4 + 5e-5 * len(atoms) + 1e-3
        assert abs(sum(atoms) - parts[0]) < tol, (sum(atoms), parts[0])


# ---------------------------------------------------------------------------
# Per-residue Vina decomposition properties
# ---------------------------------------------------------------------------
RESIDUE_DECOMP_PROPS = [
    "residue_labels",
    "residue_inter_energy",
    "residue_gauss1",
    "residue_gauss2",
    "residue_repulsion",
    "residue_hydrophobic",
    "residue_hbond",
]


@pytest.mark.slow
def test_residue_decomp_properties_present(decomp_poses):
    """All per-residue decomposition properties should be present."""
    mol = decomp_poses[0]
    for prop in RESIDUE_DECOMP_PROPS:
        assert mol.HasProp(prop), f"Missing residue decomp property: {prop}"


@pytest.mark.slow
def test_residue_labels_format(decomp_poses):
    """Residue labels should be chain:resname:resid format."""
    mol = decomp_poses[0]
    if not mol.HasProp("residue_labels"):
        pytest.skip("residue_labels not present")

    labels = mol.GetProp("residue_labels").split()
    assert len(labels) > 0, "No residue labels found"

    for label in labels:
        parts = label.split(":")
        assert len(parts) == 3, f"Bad label format '{label}', expected chain:resname:resid"
        chain, resname, resid = parts
        assert len(chain) > 0, f"Empty chain in label '{label}'"
        assert len(resname) > 0, f"Empty resname in label '{label}'"
        assert resid.lstrip("-").isdigit(), f"Non-integer resid in label '{label}'"


@pytest.mark.slow
def test_residue_decomp_consistent_lengths(decomp_poses):
    """All residue decomp arrays should have the same length as residue_labels."""
    mol = decomp_poses[0]
    if not mol.HasProp("residue_labels"):
        pytest.skip("residue_labels not present")

    nres = len(mol.GetProp("residue_labels").split())
    for prop in RESIDUE_DECOMP_PROPS[1:]:  # skip residue_labels itself
        if not mol.HasProp(prop):
            pytest.skip(f"{prop} not present")
        values = mol.GetProp(prop).split()
        assert len(values) == nres, (
            f"{prop} has {len(values)} values but expected {nres} (matching residue_labels)"
        )


@pytest.mark.slow
def test_residue_components_sum_to_total(decomp_poses):
    """Per-residue total should equal sum of components within tolerance."""
    mol = decomp_poses[0]
    if not mol.HasProp("residue_inter_energy"):
        pytest.skip("residue decomp not present")

    total = [float(x) for x in mol.GetProp("residue_inter_energy").split()]
    g1 = [float(x) for x in mol.GetProp("residue_gauss1").split()]
    g2 = [float(x) for x in mol.GetProp("residue_gauss2").split()]
    rep = [float(x) for x in mol.GetProp("residue_repulsion").split()]
    hyd = [float(x) for x in mol.GetProp("residue_hydrophobic").split()]
    hb = [float(x) for x in mol.GetProp("residue_hbond").split()]

    max_err = 0.0
    for i in range(len(total)):
        expected = g1[i] + g2[i] + rep[i] + hyd[i] + hb[i]
        err = abs(total[i] - expected)
        max_err = max(max_err, err)

    assert max_err < 0.01, (
        f"Residue total vs component sum error too large: {max_err:.6f}"
    )


@pytest.mark.slow
def test_residue_vs_atom_sum_consistent(decomp_poses):
    """Sum of residue_inter_energy should match sum of atom_inter_energy."""
    mol = decomp_poses[0]
    if not mol.HasProp("residue_inter_energy") or not mol.HasProp("atom_inter_energy"):
        pytest.skip("residue or atom decomp not present")

    res_sum = sum(float(x) for x in mol.GetProp("residue_inter_energy").split())
    atom_sum = sum(float(x) for x in mol.GetProp("atom_inter_energy").split())

    diff = abs(res_sum - atom_sum)
    assert diff < 0.01, (
        f"Residue sum ({res_sum:.4f}) vs atom sum ({atom_sum:.4f}) "
        f"differ by {diff:.6f}"
    )
