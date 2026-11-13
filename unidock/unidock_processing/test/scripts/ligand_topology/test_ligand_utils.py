import os
import tempfile

import numpy as np
import pytest

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock_processing.ligand_topology import utils


def _make_embedded_mol(smiles):
    """Create a molecule with Hs and a 3D conformer."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


# ---- Tests for calculate_center_of_mass ----

class TestCalculateCenterOfMass:
    def test_single_atom(self):
        """A single carbon atom -- CoM equals the atom position."""
        mol = Chem.MolFromSmiles("[CH4]")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        com = utils.calculate_center_of_mass(mol)
        # For a single heavy atom + 4 symmetric H, CoM should be close
        # to the carbon position.
        conformer = mol.GetConformer()
        c_pos = np.array(conformer.GetAtomPosition(0))
        assert com.shape == (3,)
        # The CoM should be very close to carbon (slight offset from Hs)
        assert np.linalg.norm(com - c_pos) < 0.1

    def test_return_shape(self):
        """CoM should always be a 3D vector."""
        mol = _make_embedded_mol("CCO")
        com = utils.calculate_center_of_mass(mol)
        assert com.shape == (3,)

    def test_symmetric_molecule_com_near_center(self):
        """Ethane (CC) -- CoM should lie near the midpoint of the two carbons."""
        mol = _make_embedded_mol("CC")
        conformer = mol.GetConformer()
        c0 = np.array(conformer.GetAtomPosition(0))
        c1 = np.array(conformer.GetAtomPosition(1))
        midpoint = (c0 + c1) / 2.0
        com = utils.calculate_center_of_mass(mol)
        # CoM should be near the midpoint (H atoms are symmetric)
        assert np.linalg.norm(com - midpoint) < 0.2

    def test_heavy_atom_dominates(self):
        """HBr -- CoM should be much closer to Br than to H."""
        mol = Chem.MolFromSmiles("[H]Br")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        conformer = mol.GetConformer()
        positions = conformer.GetPositions()
        # Find Br atom index
        br_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == "Br":
                br_idx = i
                break
        br_pos = positions[br_idx]
        com = utils.calculate_center_of_mass(mol)
        # Br is ~80 Da, H is ~1 Da, so CoM is almost at Br
        assert np.linalg.norm(com - br_pos) < 0.05


# ---- Tests for assign_atom_properties ----

class TestAssignAtomProperties:
    def test_gasteiger_charges_set(self):
        """After assign_atom_properties, each atom should have a 'charge' property."""
        mol = _make_embedded_mol("CCO")
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            assert atom.HasProp("charge")

    def test_internal_and_sdf_indices_set(self):
        """Each atom should get internal_atom_idx (0-based) and sdf_atom_idx (1-based)."""
        mol = _make_embedded_mol("CC")
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            assert atom.GetIntProp("internal_atom_idx") == atom_idx
            assert atom.GetIntProp("sdf_atom_idx") == atom_idx + 1

    def test_atom_name_assigned(self):
        """Atoms without existing atom_name get auto-generated names (e.g., C1, H2)."""
        mol = _make_embedded_mol("C")
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        c_atom = mol.GetAtomWithIdx(0)
        assert c_atom.HasProp("atom_name")
        assert c_atom.GetProp("atom_name").startswith("C")

    def test_residue_properties_assigned(self):
        """Atoms should get residue_name='MOL', residue_idx=1, chain_idx='A'."""
        mol = _make_embedded_mol("CC")
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        atom = mol.GetAtomWithIdx(0)
        assert atom.GetProp("residue_name") == "MOL"
        assert atom.GetIntProp("residue_idx") == 1
        assert atom.GetProp("chain_idx") == "A"

    def test_xyz_coordinates_set(self):
        """Each atom should have x, y, z double properties matching the conformer."""
        mol = _make_embedded_mol("CCO")
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        positions = mol.GetConformer().GetPositions()
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            assert abs(atom.GetDoubleProp("x") - positions[atom_idx, 0]) < 1e-6
            assert abs(atom.GetDoubleProp("y") - positions[atom_idx, 1]) < 1e-6
            assert abs(atom.GetDoubleProp("z") - positions[atom_idx, 2]) < 1e-6

    def test_existing_atom_name_preserved(self):
        """If an atom already has atom_name, it should not be overwritten."""
        mol = _make_embedded_mol("C")
        ComputeGasteigerCharges(mol)
        mol.GetAtomWithIdx(0).SetProp("atom_name", "CA")
        utils.assign_atom_properties(mol)
        assert mol.GetAtomWithIdx(0).GetProp("atom_name") == "CA"


# ---- Tests for convert_v3000_mol_to_v2000_sdf ----

class TestConvertV3000MolToV2000Sdf:
    def test_writes_valid_sdf_file(self):
        """Writing a mol to V2000 SDF should produce a readable file."""
        mol = _make_embedded_mol("CCO")
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            utils.convert_v3000_mol_to_v2000_sdf(mol, tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
            # The file should contain the V2000 mol block
            with open(tmp_path) as f:
                content = f.read()
            assert "V2000" in content
        finally:
            os.unlink(tmp_path)

    def test_mol_properties_included(self):
        """Molecule-level properties should appear as SDF data fields."""
        mol = _make_embedded_mol("CCO")
        mol.SetProp("_Name", "ethanol")
        mol.SetProp("test_key", "test_value")
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            utils.convert_v3000_mol_to_v2000_sdf(mol, tmp_path)
            with open(tmp_path) as f:
                content = f.read()
            assert "test_key" in content
            assert "test_value" in content
        finally:
            os.unlink(tmp_path)


# ---- Tests for check_manual_atom_mapping_connection ----

class TestCheckManualAtomMappingConnection:
    def test_valid_connected_mapping_returns_true(self):
        """Mapping between all atoms of two identical molecules should return True."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        # Map all heavy atoms to themselves (identity mapping)
        mapping = {i: i for i in range(mol.GetNumAtoms())}
        result = utils.check_manual_atom_mapping_connection(mol, mol, mapping)
        assert result is True

    def test_disconnected_mapping_returns_false(self):
        """Mapping that selects non-contiguous heavy atoms should return False."""
        # Use a non-aromatic chain so that SanitizeMol won't fail on
        # broken aromatic rings: hexane C-C-C-C-C-C (indices 0..5 heavy)
        mol = _make_embedded_mol("CCCCCC")
        # Select atoms 0,1 and 4,5 (skip 2,3), creating two disconnected pieces
        disconnected_indices = [0, 1, 4, 5]
        mapping = {idx: idx for idx in disconnected_indices}
        result = utils.check_manual_atom_mapping_connection(mol, mol, mapping)
        assert result is False

    def test_single_atom_mapping_returns_true(self):
        """Mapping a single atom is trivially connected."""
        mol = _make_embedded_mol("c1ccccc1")
        mapping = {0: 0}
        result = utils.check_manual_atom_mapping_connection(mol, mol, mapping)
        assert result is True
