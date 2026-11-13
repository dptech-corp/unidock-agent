import pytest

from rdkit import Chem

from unidock_processing.unidocktools.protein_topology import (
    is_peptide_bond,
    split_mol_by_residues,
)


def _build_two_residue_mol():
    """Build a small molecule with two residues (4 atoms: C-C~C-C) where
    atoms 0,1 belong to residue 0 and atoms 2,3 belong to residue 1.
    The bond between atoms 1 and 2 is the inter-residue (peptide) bond.
    """
    mol = Chem.RWMol(Chem.Mol())

    for i in range(4):
        atom = Chem.Atom(6)  # carbon
        res_idx = 0 if i < 2 else 1
        chain = "A"
        atom.SetIntProp("internal_residue_idx", res_idx)
        atom.SetProp("chain_idx", chain)
        atom.SetIntProp("atom_idx", i + 1)
        atom.SetProp("atom_name", f"C{i}")
        atom.SetDoubleProp("atom_charge", 0.0)
        atom.SetProp("ff_atom_type", "C")
        atom.SetIntProp("residue_idx", res_idx)
        atom.SetProp("residue_name", "ALA")
        atom.SetIntProp("internal_atom_idx", i)
        atom.SetDoubleProp("x", float(i))
        atom.SetDoubleProp("y", 0.0)
        atom.SetDoubleProp("z", 0.0)
        mol.AddAtom(atom)

    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    mol.AddBond(1, 2, Chem.BondType.SINGLE)  # inter-residue bond
    mol.AddBond(2, 3, Chem.BondType.SINGLE)

    mol = Chem.Mol(mol)
    Chem.GetSymmSSSR(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol


class TestIsPeptideBond:
    def test_intra_residue_bond_returns_false(self):
        mol = _build_two_residue_mol()
        bond_0_1 = mol.GetBondBetweenAtoms(0, 1)
        assert is_peptide_bond(bond_0_1) is False

    def test_inter_residue_bond_returns_true(self):
        mol = _build_two_residue_mol()
        bond_1_2 = mol.GetBondBetweenAtoms(1, 2)
        assert is_peptide_bond(bond_1_2) is True

    def test_different_chains_returns_true(self):
        """Bond between atoms on different chains is also inter-residue."""
        mol = Chem.RWMol(Chem.Mol())
        for i in range(2):
            atom = Chem.Atom(6)
            atom.SetIntProp("internal_residue_idx", 0)
            atom.SetProp("chain_idx", "A" if i == 0 else "B")
            mol.AddAtom(atom)
        mol.AddBond(0, 1, Chem.BondType.SINGLE)
        mol = Chem.Mol(mol)

        bond = mol.GetBondBetweenAtoms(0, 1)
        assert is_peptide_bond(bond) is True


class TestSplitMolByResidues:
    def test_two_residues_split_into_two_mols(self):
        mol = _build_two_residue_mol()
        residues = split_mol_by_residues(mol)
        assert len(residues) == 2

    def test_residue_atom_counts(self):
        mol = _build_two_residue_mol()
        residues = split_mol_by_residues(mol)
        assert residues[0].GetNumAtoms() == 2
        assert residues[1].GetNumAtoms() == 2

    def test_properties_preserved(self):
        mol = _build_two_residue_mol()
        residues = split_mol_by_residues(mol)
        atom0 = residues[0].GetAtomWithIdx(0)
        assert atom0.GetIntProp("internal_residue_idx") == 0
        assert atom0.GetProp("chain_idx") == "A"

    def test_single_residue_returns_one_mol(self):
        mol = Chem.RWMol(Chem.Mol())
        for i in range(3):
            atom = Chem.Atom(6)
            atom.SetIntProp("internal_residue_idx", 0)
            atom.SetProp("chain_idx", "A")
            atom.SetIntProp("atom_idx", i + 1)
            atom.SetProp("atom_name", f"C{i}")
            atom.SetDoubleProp("atom_charge", 0.0)
            atom.SetProp("ff_atom_type", "C")
            atom.SetIntProp("residue_idx", 0)
            atom.SetProp("residue_name", "GLY")
            atom.SetIntProp("internal_atom_idx", i)
            atom.SetDoubleProp("x", float(i))
            atom.SetDoubleProp("y", 0.0)
            atom.SetDoubleProp("z", 0.0)
            mol.AddAtom(atom)
        mol.AddBond(0, 1, Chem.BondType.SINGLE)
        mol.AddBond(1, 2, Chem.BondType.SINGLE)
        mol = Chem.Mol(mol)
        Chem.GetSymmSSSR(mol)
        mol.UpdatePropertyCache(strict=False)

        residues = split_mol_by_residues(mol)
        assert len(residues) == 1
        assert residues[0].GetNumAtoms() == 3

    def test_hydrogen_only_residue_is_skipped(self):
        """A residue containing only hydrogens should be skipped."""
        mol = Chem.RWMol(Chem.Mol())
        # Residue 0: one carbon (heavy atom)
        atom_c = Chem.Atom(6)
        atom_c.SetIntProp("internal_residue_idx", 0)
        atom_c.SetProp("chain_idx", "A")
        atom_c.SetIntProp("atom_idx", 1)
        atom_c.SetProp("atom_name", "C0")
        atom_c.SetDoubleProp("atom_charge", 0.0)
        atom_c.SetProp("ff_atom_type", "C")
        atom_c.SetIntProp("residue_idx", 0)
        atom_c.SetProp("residue_name", "ALA")
        atom_c.SetIntProp("internal_atom_idx", 0)
        atom_c.SetDoubleProp("x", 0.0)
        atom_c.SetDoubleProp("y", 0.0)
        atom_c.SetDoubleProp("z", 0.0)
        mol.AddAtom(atom_c)

        # Residue 1: only hydrogen
        atom_h = Chem.Atom(1)
        atom_h.SetIntProp("internal_residue_idx", 1)
        atom_h.SetProp("chain_idx", "A")
        atom_h.SetIntProp("atom_idx", 2)
        atom_h.SetProp("atom_name", "H1")
        atom_h.SetDoubleProp("atom_charge", 0.0)
        atom_h.SetProp("ff_atom_type", "H")
        atom_h.SetIntProp("residue_idx", 1)
        atom_h.SetProp("residue_name", "HOH")
        atom_h.SetIntProp("internal_atom_idx", 1)
        atom_h.SetDoubleProp("x", 1.0)
        atom_h.SetDoubleProp("y", 0.0)
        atom_h.SetDoubleProp("z", 0.0)
        mol.AddAtom(atom_h)

        mol = Chem.Mol(mol)
        Chem.GetSymmSSSR(mol)
        mol.UpdatePropertyCache(strict=False)

        residues = split_mol_by_residues(mol)
        assert len(residues) == 1

    def test_intra_residue_bonds_preserved(self):
        mol = _build_two_residue_mol()
        residues = split_mol_by_residues(mol)
        # Each residue has 2 atoms with 1 bond between them
        assert residues[0].GetNumBonds() == 1
        assert residues[1].GetNumBonds() == 1

    def test_inter_residue_bond_not_in_fragments(self):
        mol = _build_two_residue_mol()
        residues = split_mol_by_residues(mol)
        # Original has 3 bonds, each fragment has 1 -> inter-residue bond excluded
        total_bonds = sum(r.GetNumBonds() for r in residues)
        assert total_bonds == 2  # 3 original - 1 inter-residue = 2
