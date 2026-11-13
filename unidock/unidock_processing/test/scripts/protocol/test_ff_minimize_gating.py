"""Tests for FF minimization gating logic.

When ff_minimize_steps=0 (default), no DMS building or FF param fetching occurs.
When ff_minimize_steps>0, DMS is built from SDF and FF params are loaded.
"""
import os

from context import TEST_DATA_DIR

from unidock_processing.io.get_temp_dir_prefix import get_temp_dir_prefix
from unidock_processing.io.tempfile import TemporaryDirectory
from unidock_processing.unidocktools.unidock_protocol_runner import (
    UnidockProtocolRunner,
)

# Use the small 1G9V test case (single ligand, fast on CPU)
RECEPTOR = os.path.join(
    TEST_DATA_DIR, 'free_docking', 'molecular_docking',
    '1G9V_protein_water_cleaned.pdb',
)
LIGAND = os.path.join(
    TEST_DATA_DIR, 'free_docking', 'molecular_docking',
    'ligand_prepared.sdf',
)
POCKET_CENTER = (5.122, 18.327, 37.332)
BOX_SIZE = (30.0, 30.0, 30.0)

# Fast docking params (minimal search)
FAST_PARAMS = dict(
    search_mode='free',
    exhaustiveness=4,
    mc_steps=1,
    opt_steps=0,
    refine_steps=1,
    num_pose=1,
    seed=42,
)


def test_ff_off_no_ff_minimization():
    """ff_minimize_steps=0 (default): DMS is still built (required for engine),
    but it carries no FF parameters, so neither FF minimization nor FF scoring runs."""
    temp_prefix = os.path.join('/tmp', get_temp_dir_prefix('test_ff_off'))
    with TemporaryDirectory(prefix=temp_prefix, delete=True) as tmpdir:
        output_sdf = os.path.join(tmpdir, 'poses.sdf')
        runner = UnidockProtocolRunner(
            receptor_file_name=RECEPTOR,
            ligand_sdf_file_name_list=[LIGAND],
            target_center=POCKET_CENTER,
            box_size=BOX_SIZE,
            working_dir_name=tmpdir,
            docking_pose_sdf_file_name=output_sdf,
            ff_minimize_steps=0,
            **FAST_PARAMS,
        )
        runner.run_unidock_protocol()

        # Output SDF should exist (docking still works)
        assert os.path.exists(output_sdf)
        assert os.path.getsize(output_sdf) > 0

        # DMS is always built now (required for DMS-only engine path),
        # but ff_minimize_steps=0 means no FF minimization occurs.
        assert runner.ff_minimize_steps == 0

        # The minimal DMS has an empty nonbonded_param table: the engine loads no FF,
        # so no FF energies are written (they used to read 0.000 from empty tables)
        with open(output_sdf) as f:
            content = f.read()
        assert 'ff_total_energy' not in content


def test_ff_on_dms_built_and_used():
    """ff_minimize_steps>0: DMS should be built, FF params loaded, FF energies in output."""
    temp_prefix = os.path.join('/tmp', get_temp_dir_prefix('test_ff_on'))
    with TemporaryDirectory(prefix=temp_prefix, delete=True) as tmpdir:
        output_sdf = os.path.join(tmpdir, 'poses.sdf')
        runner = UnidockProtocolRunner(
            receptor_file_name=RECEPTOR,
            ligand_sdf_file_name_list=[LIGAND],
            target_center=POCKET_CENTER,
            box_size=BOX_SIZE,
            working_dir_name=tmpdir,
            docking_pose_sdf_file_name=output_sdf,
            ff_minimize_steps=100,
            **FAST_PARAMS,
        )
        runner.run_unidock_protocol()

        # Output SDF should exist
        assert os.path.exists(output_sdf)
        assert os.path.getsize(output_sdf) > 0

        # DMS should have been built
        assert runner.ligand_has_ff is True
        assert len(runner.ligand_dms_dict) > 0
        assert runner.ligand_dms is not None

        # Enriched DMS should exist
        assert runner.enriched_ligand_dms is not None
        assert os.path.exists(runner.enriched_ligand_dms)

        # Output should contain ff_total_energy
        with open(output_sdf) as f:
            content = f.read()
        assert 'ff_total_energy' in content
        assert 'ff_interaction_energy' in content


def test_default_ff_minimize_steps_is_zero():
    """Verify the default value of ff_minimize_steps is 0."""
    temp_prefix = os.path.join('/tmp', get_temp_dir_prefix('test_ff_default'))
    with TemporaryDirectory(prefix=temp_prefix, delete=True) as tmpdir:
        output_sdf = os.path.join(tmpdir, 'poses.sdf')
        runner = UnidockProtocolRunner(
            receptor_file_name=RECEPTOR,
            ligand_sdf_file_name_list=[LIGAND],
            target_center=POCKET_CENTER,
            box_size=BOX_SIZE,
            working_dir_name=tmpdir,
            docking_pose_sdf_file_name=output_sdf,
            # Do NOT pass ff_minimize_steps — use default
            **FAST_PARAMS,
        )
        assert runner.ff_minimize_steps == 0
