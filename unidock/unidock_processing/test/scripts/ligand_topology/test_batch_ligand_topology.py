"""Regression test for batch ligand topology building with different atom counts.

The original bug: when a multi-ligand SDF (e.g., cdk4_3lig.sdf with 3 ligands
of atom counts 39, 53, 39) was processed with FF enabled, a single DMS built
from ligand 0 was shared with all ligands. Ligand 1 (53 atoms) would crash at
build_molecular_graph line 235 with IndexError because atom_type_list had only
39 entries.

The fix passes a per-ligand DMS dict so each ligand gets its own DMS path.
"""

import os
import sys
import tempfile
import importlib

import pytest
from unittest.mock import MagicMock, patch

from rdkit import Chem

from context import TEST_DATA_DIR


@pytest.fixture(autouse=True)
def _mock_msys():
    """Temporarily mock msys for this module's tests, then fully restore it."""
    msys_was_absent = 'msys' not in sys.modules
    original_msys = sys.modules.get('msys')

    if msys_was_absent:
        sys.modules['msys'] = MagicMock()

    # Import modules that depend on msys while mock is active
    import unidock_processing.unidocktools.unidock_protocol_runner  # noqa: F401
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (  # noqa: F811
        UnidockLigandTopologyBuilder,
    )

    yield UnidockLigandTopologyBuilder

    # Restore sys.modules
    if msys_was_absent:
        sys.modules.pop('msys', None)
    elif original_msys is not None:
        sys.modules['msys'] = original_msys

    # Reload modules that cached the mock so they pick up the real msys (or fail gracefully)
    for mod_name in list(sys.modules):
        if 'unidock_processing.unidocktools' in mod_name:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass


BATCH_LIGAND_DATA_DIR = os.path.join(TEST_DATA_DIR, 'batch_ligand')
CDK4_3LIG_SDF = os.path.join(BATCH_LIGAND_DATA_DIR, 'cdk4_3lig.sdf')

# Atom counts for cdk4_3lig.sdf: MOL_0=39 (ds_3738236), MOL_1=53 (ds_11682086), MOL_2=39 (ds_153366)
EXPECTED_ATOM_COUNTS = {'MOL_0': 39, 'MOL_1': 53, 'MOL_2': 39}


def _mock_read_gaff2_types_from_dms(dms_path):
    """Return dummy GAFF2 types matching the atom count encoded in the DMS filename."""
    for mol_name, n_atoms in EXPECTED_ATOM_COUNTS.items():
        if mol_name in os.path.basename(dms_path):
            return ['ca'] * n_atoms, [0.0] * n_atoms
    raise ValueError(f"Unexpected DMS path in mock: {dms_path}")


def test_batch_ligand_topology_different_atom_counts(_mock_msys):
    """Multi-ligand SDF with different atom counts must not crash (regression)."""
    UnidockLigandTopologyBuilder = _mock_msys
    with tempfile.TemporaryDirectory(prefix='test_batch_lig_') as tmpdir:
        # Create dummy per-ligand DMS files (content irrelevant; mock intercepts read)
        dms_dict = {}
        for name in ['MOL_0', 'MOL_1', 'MOL_2']:
            dms_path = os.path.join(tmpdir, f'mock_{name}.dms')
            open(dms_path, 'w').close()
            dms_dict[name] = dms_path

        with patch(
            'unidock_processing.core.dms.read_gaff2_types_from_dms',
            side_effect=_mock_read_gaff2_types_from_dms,
        ):
            builder = UnidockLigandTopologyBuilder(
                [CDK4_3LIG_SDF],
                has_ff=True,
                ligand_dms_path_dict=dms_dict,
                n_cpu=1,
                working_dir_name=tmpdir,
            )

            assert builder.num_ligands == 3

            builder.generate_batch_ligand_topology()
            builder.get_summary_ligand_info_dict()

        # Verify all 3 ligands were processed
        summary = builder.total_ligand_info_summary_dict
        assert set(summary.keys()) == {'MOL_0', 'MOL_1', 'MOL_2'}

        # Verify atom counts match each ligand (not shared from first)
        assert len(summary['MOL_0']['atoms']) == 39
        assert len(summary['MOL_1']['atoms']) == 53  # would be 39 with old bug
        assert len(summary['MOL_2']['atoms']) == 39


def test_batch_ligand_topology_single_ligand(_mock_msys):
    """Single-ligand SDF with FF should work (backward compat)."""
    UnidockLigandTopologyBuilder = _mock_msys
    with tempfile.TemporaryDirectory(prefix='test_batch_single_') as tmpdir:
        mol = Chem.SDMolSupplier(CDK4_3LIG_SDF, removeHs=False)[0]
        single_sdf = os.path.join(tmpdir, 'single.sdf')
        writer = Chem.SDWriter(single_sdf)
        writer.write(mol)
        writer.close()

        dms_path = os.path.join(tmpdir, 'mock_MOL_0.dms')
        open(dms_path, 'w').close()
        dms_dict = {'MOL_0': dms_path}

        with patch(
            'unidock_processing.core.dms.read_gaff2_types_from_dms',
            side_effect=lambda p: (['ca'] * 39, [0.0] * 39),
        ):
            builder = UnidockLigandTopologyBuilder(
                [single_sdf],
                has_ff=True,
                ligand_dms_path_dict=dms_dict,
                n_cpu=1,
                working_dir_name=tmpdir,
            )

            assert builder.num_ligands == 1

            builder.generate_batch_ligand_topology()
            builder.get_summary_ligand_info_dict()

        summary = builder.total_ligand_info_summary_dict
        assert list(summary.keys()) == ['MOL_0']
        assert len(summary['MOL_0']['atoms']) == 39


def test_batch_ligand_topology_no_ff(_mock_msys):
    """With has_ff=False and empty dict, builder construction works."""
    UnidockLigandTopologyBuilder = _mock_msys
    with tempfile.TemporaryDirectory(prefix='test_batch_noff_') as tmpdir:
        builder = UnidockLigandTopologyBuilder(
            [CDK4_3LIG_SDF],
            has_ff=False,
            ligand_dms_path_dict={},
            n_cpu=1,
            working_dir_name=tmpdir,
        )

        assert builder.num_ligands == 3
        assert builder.ligand_dms_path_dict == {}
        assert builder.has_ff is False
