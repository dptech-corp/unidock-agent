"""FF minimization gating tests.

DMS is always built from SDF (required by engine).
When ff_minimize_steps=0, the DMS carries no FF parameters: no FF scoring, no FF minimization.
When ff_minimize_steps>0, FF parameters are loaded: FF scoring and FF minimization run.

Converted from: test_ff_minimize_gating.py
"""
import os
import sqlite3

import pytest

from tests.conftest import BATCH_LIGAND_DIR, FREE_DOCK_DIR, RESCORE_TEST_DIR

# ---------------------------------------------------------------------------
# Test data paths
# ---------------------------------------------------------------------------
RECEPTOR_1G9V = os.path.join(FREE_DOCK_DIR, "1G9V_protein_water_cleaned.pdb")
LIGAND_1G9V = os.path.join(FREE_DOCK_DIR, "ligand_prepared.sdf")
POCKET_CENTER_1G9V = (5.122, 18.327, 37.332)

CDK4_RECEPTOR = os.path.join(RESCORE_TEST_DIR, "7sj3_prep.pdb")
CDK4_LIGAND = os.path.join(BATCH_LIGAND_DIR, "cdk4_3lig.sdf")
CDK4_CENTER = (11.7, -37.8, 11.6)

FAST_PARAMS = dict(
    search_mode="free",
    exhaustiveness=4,
    mc_steps=1,
    opt_steps=0,
    refine_steps=1,
    num_pose=1,
    seed=42,
)


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------
def _skip_if_missing(*paths: str) -> None:
    for p in paths:
        if not os.path.isfile(p):
            pytest.skip(f"Missing test data: {p}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_default_ff_minimize_steps_is_zero(tmp_path):
    """Verify the default value of ff_minimize_steps is 0."""
    _skip_if_missing(RECEPTOR_1G9V, LIGAND_1G9V)

    from unidock_processing.unidocktools.unidock_protocol_runner import (
        UnidockProtocolRunner,
    )

    output_sdf = str(tmp_path / "poses.sdf")
    runner = UnidockProtocolRunner(
        receptor_file_name=RECEPTOR_1G9V,
        ligand_sdf_file_name_list=[LIGAND_1G9V],
        target_center=POCKET_CENTER_1G9V,
        working_dir_name=str(tmp_path),
        docking_pose_sdf_file_name=output_sdf,
        **FAST_PARAMS,
    )
    assert runner.ff_minimize_steps == 0


@pytest.mark.slow
def test_ff_off_still_builds_dms(tmp_path):
    """ff_minimize_steps=0: DMS is still built (required by engine), docking succeeds."""
    _skip_if_missing(RECEPTOR_1G9V, LIGAND_1G9V)

    from unidock_processing.unidocktools.unidock_protocol_runner import (
        UnidockProtocolRunner,
    )

    output_sdf = str(tmp_path / "poses.sdf")
    runner = UnidockProtocolRunner(
        receptor_file_name=RECEPTOR_1G9V,
        ligand_sdf_file_name_list=[LIGAND_1G9V],
        target_center=POCKET_CENTER_1G9V,
        working_dir_name=str(tmp_path),
        docking_pose_sdf_file_name=output_sdf,
        ff_minimize_steps=0,
        **FAST_PARAMS,
    )
    runner.run_unidock_protocol()

    assert os.path.exists(output_sdf)
    assert os.path.getsize(output_sdf) > 0

    # Minimal DMS built (no FF params when ff_minimize_steps=0)
    assert runner.ligand_has_ff is False
    assert len(runner.ligand_dms_dict) > 0
    assert runner.ligand_dms is not None
    assert runner.enriched_ligand_dms is not None


@pytest.mark.slow
def test_ff_on_dms_built_and_used(tmp_path):
    """ff_minimize_steps>0: DMS built, FF params loaded, FF energies in output."""
    _skip_if_missing(RECEPTOR_1G9V, LIGAND_1G9V)

    from unidock_processing.unidocktools.unidock_protocol_runner import (
        UnidockProtocolRunner,
    )

    output_sdf = str(tmp_path / "poses.sdf")
    runner = UnidockProtocolRunner(
        receptor_file_name=RECEPTOR_1G9V,
        ligand_sdf_file_name_list=[LIGAND_1G9V],
        target_center=POCKET_CENTER_1G9V,
        working_dir_name=str(tmp_path),
        docking_pose_sdf_file_name=output_sdf,
        ff_minimize_steps=100,
        **FAST_PARAMS,
    )
    runner.run_unidock_protocol()

    assert os.path.exists(output_sdf)
    assert os.path.getsize(output_sdf) > 0

    assert runner.ligand_has_ff is True
    assert len(runner.ligand_dms_dict) > 0
    assert runner.ligand_dms is not None

    assert runner.enriched_ligand_dms is not None
    assert os.path.exists(runner.enriched_ligand_dms)

    with open(output_sdf) as f:
        content = f.read()
    assert "ff_total_energy" in content
    assert "ff_interaction_energy" in content


@pytest.mark.slow
def test_ff_on_multi_ligand_merged_dms(tmp_path):
    """ff_minimize_steps>0 with multi-ligand SDF: merged DMS with ud2_molecules table."""
    _skip_if_missing(CDK4_RECEPTOR, CDK4_LIGAND)

    from unidock_processing.unidocktools.unidock_protocol_runner import (
        UnidockProtocolRunner,
    )

    output_sdf = str(tmp_path / "poses.sdf")
    runner = UnidockProtocolRunner(
        receptor_file_name=CDK4_RECEPTOR,
        ligand_sdf_file_name_list=[CDK4_LIGAND],
        target_center=CDK4_CENTER,
        working_dir_name=str(tmp_path),
        docking_pose_sdf_file_name=output_sdf,
        ff_minimize_steps=100,
        num_pose=1,
        search_mode="free",
        exhaustiveness=4,
        mc_steps=1,
        opt_steps=0,
        refine_steps=1,
        seed=42,
    )
    runner.run_unidock_protocol()

    assert os.path.exists(output_sdf)
    with open(output_sdf) as f:
        content = f.read()
    assert content.count("$$$$") == 3

    assert runner.enriched_ligand_dms is not None
    conn = sqlite3.connect(runner.enriched_ligand_dms)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ud2_molecules")
    assert cur.fetchone()[0] == 3
    conn.close()

    assert content.count("ff_total_energy") == 3
