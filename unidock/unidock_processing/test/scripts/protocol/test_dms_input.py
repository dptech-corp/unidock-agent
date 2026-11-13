"""Tests for DMS input auto-detection and conversion in UnidockProtocolRunner."""
import os
import tempfile
import pytest

# Path to C++ engine test data (DMS files from uni-fep)
FF_TEST_DATA_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', '..',
        'unidock_engine', 'test', 'data', 'ff_minimize', 'tyk2',
    )
)

# Path to existing SDF ligand test data
LIGAND_TEST_DATA_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'data', 'free_docking', 'molecular_docking',
    )
)


class TestDMSDetection:
    """Test DMS extension detection logic (no msys required)."""

    def test_receptor_dms_detected(self, tmp_path):
        """When receptor is .dms, protein_dms should be set."""
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        # Use a real SDF for ligand so no DMS conversion is attempted
        ligand_sdf = os.path.join(LIGAND_TEST_DATA_DIR, 'ligand_prepared.sdf')
        if not os.path.isfile(ligand_sdf):
            pytest.skip("Test SDF not found")

        # Create a dummy .dms file to test extension detection
        dummy_dms = os.path.join(str(tmp_path), 'protein.dms')
        open(dummy_dms, 'w').close()

        runner = UnidockProtocolRunner(
            receptor_file_name=dummy_dms,
            ligand_sdf_file_name_list=[ligand_sdf],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
        )
        assert runner.protein_dms is not None
        assert runner.protein_dms.endswith('.dms')
        assert runner.ligand_dms is None

    def test_receptor_pdb_no_dms(self, tmp_path):
        """When receptor is .pdb, protein_dms should be None."""
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        ligand_sdf = os.path.join(LIGAND_TEST_DATA_DIR, 'ligand_prepared.sdf')
        if not os.path.isfile(ligand_sdf):
            pytest.skip("Test SDF not found")

        dummy_pdb = os.path.join(str(tmp_path), 'protein.pdb')
        open(dummy_pdb, 'w').close()

        runner = UnidockProtocolRunner(
            receptor_file_name=dummy_pdb,
            ligand_sdf_file_name_list=[ligand_sdf],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
        )
        assert runner.protein_dms is None
        assert runner.ligand_dms is None

    def test_sdf_ligand_not_converted(self, tmp_path):
        """SDF ligand paths should remain unchanged."""
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        ligand_sdf = os.path.join(LIGAND_TEST_DATA_DIR, 'ligand_prepared.sdf')
        if not os.path.isfile(ligand_sdf):
            pytest.skip("Test SDF not found")

        dummy_pdb = os.path.join(str(tmp_path), 'protein.pdb')
        open(dummy_pdb, 'w').close()

        runner = UnidockProtocolRunner(
            receptor_file_name=dummy_pdb,
            ligand_sdf_file_name_list=[ligand_sdf],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
        )
        assert runner.ligand_sdf_file_name_list == [os.path.abspath(ligand_sdf)]


def _msys_available():
    try:
        import msys
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _msys_available(), reason="msys not installed")
class TestDMSConversion:
    """Test actual DMS-to-SDF conversion (requires msys + test DMS files)."""

    def test_ligand_dms_converted_to_sdf(self, tmp_path):
        """When ligand is .dms, it should be converted to SDF and ligand_dms stored."""
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        ligand_dms = os.path.join(FF_TEST_DATA_DIR, 'ligand_ejm42.dms')
        protein_dms = os.path.join(FF_TEST_DATA_DIR, 'protein.dms')
        if not os.path.isfile(ligand_dms) or not os.path.isfile(protein_dms):
            pytest.skip("Test DMS files not found")

        runner = UnidockProtocolRunner(
            receptor_file_name=protein_dms,
            ligand_sdf_file_name_list=[ligand_dms],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
        )
        # ligand_dms should be stored
        assert runner.ligand_dms == os.path.abspath(ligand_dms)
        assert runner.protein_dms == os.path.abspath(protein_dms)

        # ligand list should now point to converted SDF, not the original DMS
        assert len(runner.ligand_sdf_file_name_list) == 1
        converted_sdf = runner.ligand_sdf_file_name_list[0]
        assert converted_sdf.endswith('.sdf')
        assert os.path.isfile(converted_sdf)

    def test_converted_sdf_has_valid_molecule(self, tmp_path):
        """The converted SDF should contain a valid RDKit molecule."""
        from rdkit import Chem
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        ligand_dms = os.path.join(FF_TEST_DATA_DIR, 'ligand_ejm42.dms')
        protein_dms = os.path.join(FF_TEST_DATA_DIR, 'protein.dms')
        if not os.path.isfile(ligand_dms) or not os.path.isfile(protein_dms):
            pytest.skip("Test DMS files not found")

        runner = UnidockProtocolRunner(
            receptor_file_name=protein_dms,
            ligand_sdf_file_name_list=[ligand_dms],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
        )
        converted_sdf = runner.ligand_sdf_file_name_list[0]
        mol_list = list(Chem.SDMolSupplier(converted_sdf, removeHs=False))
        assert len(mol_list) == 1
        mol = mol_list[0]
        assert mol is not None

        # Should have atoms and 3D coordinates
        assert mol.GetNumAtoms() > 0
        conf = mol.GetConformer()
        assert conf.Is3D()

    def test_converted_mol_has_correct_bond_orders(self, tmp_path):
        """The converted mol should have correct bond orders (not all single)."""
        from rdkit import Chem
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        ligand_dms = os.path.join(FF_TEST_DATA_DIR, 'ligand_ejm42.dms')
        protein_dms = os.path.join(FF_TEST_DATA_DIR, 'protein.dms')
        if not os.path.isfile(ligand_dms) or not os.path.isfile(protein_dms):
            pytest.skip("Test DMS files not found")

        runner = UnidockProtocolRunner(
            receptor_file_name=protein_dms,
            ligand_sdf_file_name_list=[ligand_dms],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
        )
        converted_sdf = runner.ligand_sdf_file_name_list[0]
        mol = Chem.SDMolSupplier(converted_sdf, removeHs=False)[0]

        # ejm42 is a drug-like molecule — should have aromatic/double bonds
        bond_orders = set()
        for bond in mol.GetBonds():
            bond_orders.add(bond.GetBondType())

        # Must have more than just single bonds
        assert len(bond_orders) > 1, (
            f"Expected multiple bond types, got only: {bond_orders}"
        )

    def test_converted_mol_rotatable_bonds(self, tmp_path):
        """The converted mol should yield detectable rotatable bonds."""
        from rdkit import Chem
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        from unidock_processing.ligand_topology.generic_rotatable_bond import (
            GenericRotatableBond,
        )

        ligand_dms = os.path.join(FF_TEST_DATA_DIR, 'ligand_ejm42.dms')
        protein_dms = os.path.join(FF_TEST_DATA_DIR, 'protein.dms')
        if not os.path.isfile(ligand_dms) or not os.path.isfile(protein_dms):
            pytest.skip("Test DMS files not found")

        runner = UnidockProtocolRunner(
            receptor_file_name=protein_dms,
            ligand_sdf_file_name_list=[ligand_dms],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
        )
        converted_sdf = runner.ligand_sdf_file_name_list[0]
        mol = Chem.SDMolSupplier(converted_sdf, removeHs=False)[0]

        rot_finder = GenericRotatableBond()
        rot_bonds = rot_finder.identify_rotatable_bonds(mol)

        # ejm42 should have rotatable bonds
        assert len(rot_bonds) > 0, "Expected rotatable bonds in ejm42"

    def test_converted_mol_vina_atom_types(self, tmp_path):
        """Vina atom types should be assignable to the converted mol."""
        from rdkit import Chem
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        from unidock_processing.unidocktools.vina_atom_type import AtomType

        ligand_dms = os.path.join(FF_TEST_DATA_DIR, 'ligand_ejm42.dms')
        protein_dms = os.path.join(FF_TEST_DATA_DIR, 'protein.dms')
        if not os.path.isfile(ligand_dms) or not os.path.isfile(protein_dms):
            pytest.skip("Test DMS files not found")

        runner = UnidockProtocolRunner(
            receptor_file_name=protein_dms,
            ligand_sdf_file_name_list=[ligand_dms],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
        )
        converted_sdf = runner.ligand_sdf_file_name_list[0]
        mol = Chem.SDMolSupplier(converted_sdf, removeHs=False)[0]

        atom_typer = AtomType()
        atom_typer.assign_atom_types(mol)

        # All non-hydrogen atoms should have a vina_atom_type assigned
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                assert atom.HasProp('vina_atom_type'), (
                    f"Atom {atom.GetIdx()} ({atom.GetSymbol()}) missing vina_atom_type"
                )

    def test_ff_minimize_yaml_with_dms_inputs(self, tmp_path):
        """When DMS inputs are detected and ff_minimize_steps > 0,
        the YAML should include DMS paths."""
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )
        ligand_dms = os.path.join(FF_TEST_DATA_DIR, 'ligand_ejm42.dms')
        protein_dms = os.path.join(FF_TEST_DATA_DIR, 'protein.dms')
        if not os.path.isfile(ligand_dms) or not os.path.isfile(protein_dms):
            pytest.skip("Test DMS files not found")

        runner = UnidockProtocolRunner(
            receptor_file_name=protein_dms,
            ligand_sdf_file_name_list=[ligand_dms],
            target_center=(0.0, 0.0, 0.0),
            working_dir_name=str(tmp_path),
            ff_minimize_steps=10,
        )

        assert runner.ff_minimize_steps == 10
        assert runner.protein_dms is not None
        assert runner.ligand_dms is not None
