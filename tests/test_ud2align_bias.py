"""ud2align bias-alignment integration tests.

Converted from:
  - test_plk4_align.py    → test_plk4_hbond_*
  - test_ud2align_e2e.py  → test_7sj3_hbond_*
"""
import glob
import os
import sqlite3
import subprocess

import numpy as np
import pytest
from rdkit import Chem

from tests.conftest import UD2ALIGN_DATA
from tests.helpers import find_ud2_binary, prepare_enriched_dms

pytest.importorskip("ud2align", reason="external ud2align package is not installed")


# ---------------------------------------------------------------------------
# Helpers specific to ud2align bias workflow
# ---------------------------------------------------------------------------
def _inject_bias_and_translate(lig_dms, mol_q, mol_t, matrix):
    """Inject bias into DMS and translate ligand centroid to template centroid."""
    from ud2align.bias_injector import inject_bias_into_dms

    n_bias = inject_bias_into_dms(
        dms_path=lig_dms,
        mol_query=mol_q,
        mol_template=mol_t,
        similarity_matrix=matrix,
        r_param=0.0,
    )

    coords_q = mol_q.GetConformer().GetPositions()
    coords_t = mol_t.GetConformer().GetPositions()
    translation = np.mean(coords_t, axis=0) - np.mean(coords_q, axis=0)

    conn = sqlite3.connect(lig_dms)
    cur = conn.cursor()
    cur.execute(
        "UPDATE particle SET x = x + ?, y = y + ?, z = z + ?",
        (float(translation[0]), float(translation[1]), float(translation[2])),
    )
    conn.commit()
    conn.close()
    return n_bias


def _run_bias_engine(ud2_bin, center, lig_dms, rec_dms, work_dir, bias_k=0.3):
    """Run ud2 engine with bias=align and return path to merged output SDF."""
    engine_output = os.path.join(work_dir, "engine_output")
    os.makedirs(engine_output, exist_ok=True)

    yaml_content = f"""\
Advanced:
  seed: 181129
  exhaustiveness: 128
  randomize: true
  mc_steps: 40
  opt_steps: 15
  refine_steps: 5
  num_pose: 1
  tor_lib: true
  bias: align
  bias_k: {bias_k}
  ff_minimize_steps: 0
  ligand_dms: {lig_dms}
  protein_dms: {rec_dms}

Hardware:
  ncpu: {os.cpu_count() or 4}

Settings:
  task: screen
  search_mode: free
  constraint_docking: false
  decomp: true
  center_x: {center[0]}
  center_y: {center[1]}
  center_z: {center[2]}
  size_x: 30.0
  size_y: 30.0
  size_z: 30.0

Outputs:
  dir: {engine_output}

Inputs:
  ligand_dms_full: {lig_dms}
  receptor_dms_full: {rec_dms}
"""
    config_path = os.path.join(work_dir, "engine.yaml")
    with open(config_path, "w") as f:
        f.write(yaml_content)

    result = subprocess.run(
        [ud2_bin, config_path], capture_output=True, text=True, cwd=work_dir
    )
    assert result.returncode == 0, f"ud2 failed:\n{result.stderr[-500:]}"

    sdf_files = sorted(
        f for f in os.listdir(engine_output) if f.endswith(".sdf")
    )
    assert sdf_files, "No output SDF files produced by engine"
    merged_sdf = os.path.join(work_dir, "merged_output.sdf")
    with open(merged_sdf, "w") as out:
        for fn in sdf_files:
            with open(os.path.join(engine_output, fn)) as inp:
                out.write(inp.read())
    return merged_sdf


def _best_hbond_distances(mol_q, mol_t, docked_coords):
    """Compute the best distances per HBond template atom.

    Returns dict: {template_atom_idx: (best_query_idx, distance, role_str)}.
    """
    from ud2align.properties import AtomProperties

    props_q = AtomProperties(mol_q)
    props_t = AtomProperties(mol_t)
    coords_t = mol_t.GetConformer().GetPositions()

    hbond_pairs = []
    for iq in props_q.hbond_acceptors:
        for it in props_t.hbond_acceptors:
            hbond_pairs.append((iq, it, "acc"))
    for iq in props_q.hbond_donors:
        for it in props_t.hbond_donors:
            hbond_pairs.append((iq, it, "don"))

    all_t = sorted(set(props_t.hbond_acceptors + props_t.hbond_donors))
    results = {}
    for it in all_t:
        role = []
        if it in props_t.hbond_acceptors:
            role.append("acc")
        if it in props_t.hbond_donors:
            role.append("don")
        role_str = "+".join(role)

        matched_q = [iq for (iq, jt, _) in hbond_pairs if jt == it]
        if not matched_q:
            continue
        dists = [
            (iq, float(np.linalg.norm(docked_coords[iq] - coords_t[it])))
            for iq in matched_q
        ]
        best_iq, best_d = min(dists, key=lambda x: x[1])
        results[it] = (best_iq, best_d, role_str)
    return results


def _run_full_bias_workflow(data_dir, query_name, template_name, receptor_name, work_dir):
    """Execute the full bias-alignment workflow, return (best_dists, mol_q, mol_t).

    best_dists : dict from _best_hbond_distances
    """
    from ud2align.aligner import HBondAligner

    ud2_bin = find_ud2_binary()
    if ud2_bin is None:
        pytest.skip("ud2 binary not found")

    fp_query = os.path.join(data_dir, query_name)
    fp_template = os.path.join(data_dir, template_name)
    fp_receptor = os.path.join(data_dir, receptor_name)
    for fp in (fp_query, fp_template, fp_receptor):
        if not os.path.isfile(fp):
            pytest.skip(f"Missing test data: {fp}")

    mol_q = Chem.SDMolSupplier(fp_query, removeHs=False)[0]
    mol_t = Chem.SDMolSupplier(fp_template, removeHs=False)[0]

    # Similarity matrix
    aligner = HBondAligner()
    matrix = aligner.compute_matrix(mol_q, mol_t)

    center = mol_t.GetConformer().GetPositions().mean(axis=0).tolist()

    # Enriched DMS
    prep_dir = os.path.join(work_dir, "prep")
    os.makedirs(prep_dir, exist_ok=True)
    lig_dms, rec_dms = prepare_enriched_dms(fp_receptor, fp_query, center, prep_dir)

    # Inject bias + translate
    n_bias = _inject_bias_and_translate(lig_dms, mol_q, mol_t, matrix)
    assert n_bias > 0, "No bias entries injected"

    # Run engine (returns merged SDF path)
    merged_sdf = _run_bias_engine(ud2_bin, center, lig_dms, rec_dms, work_dir)

    # Parse docked coordinates from SDF output
    mol_docked = Chem.SDMolSupplier(merged_sdf, removeHs=False)[0]
    assert mol_docked is not None, "Failed to parse docked pose"
    docked_coords = mol_docked.GetConformer().GetPositions()

    # Validate
    best_dists = _best_hbond_distances(mol_q, mol_t, docked_coords)
    return best_dists, mol_q, mol_t


# ---------------------------------------------------------------------------
# PLK4 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plk4_results(tmp_path_factory):
    if not UD2ALIGN_DATA:
        pytest.skip("set UD2ALIGN_DATA to the external ud2align test-data directory")
    data_dir = os.path.join(UD2ALIGN_DATA, "PLK4")
    work = str(tmp_path_factory.mktemp("plk4_align"))
    return _run_full_bias_workflow(data_dir, "query.sdf", "ref.sdf", "4yur.pdb", work)


# ---------------------------------------------------------------------------
# 7SJ3 fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sj3_results(tmp_path_factory):
    if not UD2ALIGN_DATA:
        pytest.skip("set UD2ALIGN_DATA to the external ud2align test-data directory")
    data_dir = os.path.join(UD2ALIGN_DATA, "7SJ3")
    work = str(tmp_path_factory.mktemp("7sj3_align"))
    return _run_full_bias_workflow(
        data_dir, "ds_19494215.sdf", "7SJ3_ligand_bias_ref.sdf", "7SJ3_receptor.pdb", work
    )


# ---------------------------------------------------------------------------
# PLK4 tests (from test_plk4_align.py)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_plk4_hbond_acceptors_within_0_5A(plk4_results):
    """PLK4: all HBond-acceptor template atoms should have a docked match < 0.5 A."""
    best_dists, mol_q, mol_t = plk4_results
    for it, (q_idx, dist, role) in best_dists.items():
        if "acc" in role:
            assert dist < 0.5, (
                f"PLK4 acceptor t[{it}] best match q[{q_idx}] at {dist:.2f} A > 0.5 A"
            )


@pytest.mark.slow
def test_plk4_hbond_donors_within_0_5A(plk4_results):
    """PLK4: all HBond-donor template atoms should have a docked match < 0.5 A."""
    best_dists, mol_q, mol_t = plk4_results
    from ud2align.properties import AtomProperties

    props_t = AtomProperties(mol_t)
    if not props_t.hbond_donors:
        pytest.skip("Template has no HBond donors")
    for it, (q_idx, dist, role) in best_dists.items():
        if "don" in role:
            assert dist < 0.5, (
                f"PLK4 donor t[{it}] best match q[{q_idx}] at {dist:.2f} A > 0.5 A"
            )


# ---------------------------------------------------------------------------
# 7SJ3 tests (from test_ud2align_e2e.py)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_7sj3_hbond_acceptors_within_0_7A(sj3_results):
    """7SJ3: all HBond-acceptor template atoms should have a docked match < 0.7 A."""
    best_dists, mol_q, mol_t = sj3_results
    for it, (q_idx, dist, role) in best_dists.items():
        if "acc" in role:
            assert dist < 0.7, (
                f"7SJ3 acceptor t[{it}] best match q[{q_idx}] at {dist:.2f} A > 0.7 A"
            )


@pytest.mark.slow
def test_7sj3_hbond_donors_within_0_7A(sj3_results):
    """7SJ3: all HBond-donor template atoms should have a docked match < 0.7 A."""
    best_dists, mol_q, mol_t = sj3_results
    from ud2align.properties import AtomProperties

    props_t = AtomProperties(mol_t)
    if not props_t.hbond_donors:
        pytest.skip("Template has no HBond donors")
    for it, (q_idx, dist, role) in best_dists.items():
        if "don" in role:
            assert dist < 0.7, (
                f"7SJ3 donor t[{it}] best match q[{q_idx}] at {dist:.2f} A > 0.7 A"
            )
