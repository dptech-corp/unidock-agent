import pytest

from rdkit import Chem

from unidock_processing.utils.molecule_processing import (
    get_mol_without_indices,
    get_mol_with_indices,
)


@pytest.fixture
def ethanol_mol():
    """Ethanol (CCO) with explicit Hs, plus custom properties on each atom."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    Chem.EmbedMolecule(mol)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("original_idx", i)
        atom.SetProp("label", f"atom_{i}")
    return mol


@pytest.fixture
def propane_mol():
    """Propane (CCC) without explicit Hs, with index properties."""
    mol = Chem.MolFromSmiles("CCC")
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("original_idx", i)
        atom.SetProp("label", f"atom_{i}")
    return mol


@pytest.fixture
def methylamine_mol():
    """Methylamine (CN) to test N-H adjustment on atom removal."""
    mol = Chem.MolFromSmiles("CN")
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("original_idx", i)
    return mol


class TestGetMolWithoutIndices:
    def test_remove_nothing(self, propane_mol):
        result = get_mol_without_indices(propane_mol, remove_indices=[])
        assert result.GetNumAtoms() == propane_mol.GetNumAtoms()

    def test_remove_single_atom(self, propane_mol):
        # Remove middle carbon (index 1) from CCC
        result = get_mol_without_indices(propane_mol, remove_indices=[1])
        assert result.GetNumAtoms() == 2
        # The two remaining atoms should not be bonded (bond was through index 1)
        assert result.GetNumBonds() == 0

    def test_remove_terminal_atom(self, propane_mol):
        # Remove last carbon (index 2)
        result = get_mol_without_indices(propane_mol, remove_indices=[2])
        assert result.GetNumAtoms() == 2
        assert result.GetNumBonds() == 1

    def test_properties_preserved(self, propane_mol):
        result = get_mol_without_indices(
            propane_mol,
            remove_indices=[2],
            keep_properties=["original_idx", "label"],
        )
        atom0 = result.GetAtomWithIdx(0)
        assert atom0.GetIntProp("original_idx") == 0
        assert atom0.GetProp("label") == "atom_0"

    def test_properties_not_copied_if_not_requested(self, propane_mol):
        result = get_mol_without_indices(
            propane_mol, remove_indices=[2], keep_properties=[]
        )
        atom0 = result.GetAtomWithIdx(0)
        assert "original_idx" not in atom0.GetPropsAsDict()

    def test_nitrogen_gets_extra_h_on_neighbor_removal(self, methylamine_mol):
        """When removing a neighbor of N, the N atom gets an extra explicit H."""
        n_idx = None
        c_idx = None
        for atom in methylamine_mol.GetAtoms():
            if atom.GetSymbol() == "N":
                n_idx = atom.GetIdx()
            elif atom.GetSymbol() == "C":
                c_idx = atom.GetIdx()

        original_n_explicit_h = methylamine_mol.GetAtomWithIdx(n_idx).GetNumExplicitHs()
        result = get_mol_without_indices(methylamine_mol, remove_indices=[c_idx])
        new_n = result.GetAtomWithIdx(0)
        assert new_n.GetNumExplicitHs() == original_n_explicit_h + 1


class TestGetMolWithIndices:
    def test_keep_all(self, propane_mol):
        all_indices = list(range(propane_mol.GetNumAtoms()))
        result = get_mol_with_indices(propane_mol, selected_indices=all_indices)
        assert result.GetNumAtoms() == propane_mol.GetNumAtoms()
        assert result.GetNumBonds() == propane_mol.GetNumBonds()

    def test_keep_single_atom(self, propane_mol):
        result = get_mol_with_indices(propane_mol, selected_indices=[0])
        assert result.GetNumAtoms() == 1
        assert result.GetNumBonds() == 0

    def test_keep_two_bonded_atoms(self, propane_mol):
        result = get_mol_with_indices(propane_mol, selected_indices=[0, 1])
        assert result.GetNumAtoms() == 2
        assert result.GetNumBonds() == 1

    def test_keep_two_non_bonded_atoms(self, propane_mol):
        # Atoms 0 and 2 are terminal carbons not directly bonded
        result = get_mol_with_indices(propane_mol, selected_indices=[0, 2])
        assert result.GetNumAtoms() == 2
        assert result.GetNumBonds() == 0

    def test_properties_preserved(self, propane_mol):
        result = get_mol_with_indices(
            propane_mol,
            selected_indices=[1, 2],
            keep_properties=["original_idx", "label"],
        )
        atom0 = result.GetAtomWithIdx(0)
        # Index 0 in result corresponds to original index 1
        assert atom0.GetIntProp("original_idx") == 1
        assert atom0.GetProp("label") == "atom_1"

    def test_mol_properties_preserved(self):
        mol = Chem.MolFromSmiles("CC")
        mol.SetProp("mol_name", "ethane")
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetIntProp("idx", i)
        result = get_mol_with_indices(
            mol,
            selected_indices=[0, 1],
            keep_mol_properties=["mol_name"],
        )
        assert result.GetProp("mol_name") == "ethane"

    def test_nitrogen_gets_extra_h_on_neighbor_removal(self, methylamine_mol):
        """When keeping N but removing its C neighbor, N gets extra explicit H."""
        n_idx = None
        for atom in methylamine_mol.GetAtoms():
            if atom.GetSymbol() == "N":
                n_idx = atom.GetIdx()

        original_n_explicit_h = methylamine_mol.GetAtomWithIdx(n_idx).GetNumExplicitHs()
        result = get_mol_with_indices(methylamine_mol, selected_indices=[n_idx])
        new_n = result.GetAtomWithIdx(0)
        assert new_n.GetNumExplicitHs() == original_n_explicit_h + 1

    def test_empty_selection(self, propane_mol):
        result = get_mol_with_indices(propane_mol, selected_indices=[])
        assert result.GetNumAtoms() == 0


class TestGetMolWithAndWithoutIndicesConsistency:
    def test_complement_produces_same_split(self, propane_mol):
        """Removing indices [1] and keeping indices [0, 2] should give same result
        as keeping [0, 2] directly."""
        result_without = get_mol_without_indices(propane_mol, remove_indices=[1])
        result_with = get_mol_with_indices(propane_mol, selected_indices=[0, 2])
        assert result_without.GetNumAtoms() == result_with.GetNumAtoms()
        assert result_without.GetNumBonds() == result_with.GetNumBonds()
