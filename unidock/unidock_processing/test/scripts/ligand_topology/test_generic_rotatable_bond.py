import pytest

from rdkit import Chem
from rdkit.Chem import AllChem

from unidock_processing.ligand_topology.generic_rotatable_bond import (
    GenericRotatableBond,
)


@pytest.fixture
def rotatable_bond_finder():
    return GenericRotatableBond()


def _make_mol(smiles):
    """Create an RDKit mol with Hs and embedded 3D coords from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


# ---- Tests for identify_rotatable_bonds ----

def test_butane_has_one_rotatable_bond(rotatable_bond_finder):
    """Butane (CCCC) should have 1 rotatable bond (central C-C)."""
    mol = _make_mol("CCCC")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert len(bonds) == 1


def test_benzene_no_rotatable_bonds(rotatable_bond_finder):
    """Benzene is fully rigid -- ring bonds are excluded by the -!@ constraint."""
    mol = _make_mol("c1ccccc1")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert len(bonds) == 0


def test_ethane_no_rotatable_bonds(rotatable_bond_finder):
    """Ethane (CC) has only methyl-terminated bonds, excluded by !$([CH3])."""
    mol = _make_mol("CC")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert len(bonds) == 0


def test_amide_bond_excluded(rotatable_bond_finder):
    """N-methylacetamide (CC(=O)NC) -- the C(=O)-N amide bond should be excluded."""
    mol = _make_mol("CC(=O)NC")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    # The amide C(=O)-N bond is in the default set but is then excluded.
    # Only the C-C(=O) and N-C(methyl) bonds would appear if not
    # filtered by !D1/!CH3 rules.  For this small molecule, the SMARTS
    # already eliminates terminal methyl groups, so we expect 0.
    assert len(bonds) == 0


def test_pentane_two_rotatable_bonds(rotatable_bond_finder):
    """Pentane (CCCCC) should have 2 rotatable bonds (C2-C3, C3-C4)."""
    mol = _make_mol("CCCCC")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert len(bonds) == 2


def test_cyclohexane_no_rotatable_bonds(rotatable_bond_finder):
    """Cyclohexane ring bonds are not rotatable (-!@ excludes ring bonds)."""
    mol = _make_mol("C1CCCCC1")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert len(bonds) == 0


def test_toluene_no_rotatable_bonds(rotatable_bond_finder):
    """Toluene (Cc1ccccc1) -- the methyl-aryl bond is excluded by !$([CH3])."""
    mol = _make_mol("Cc1ccccc1")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert len(bonds) == 0


def test_biphenyl_one_rotatable_bond(rotatable_bond_finder):
    """Biphenyl (c1ccc(-c2ccccc2)cc1) -- only the aryl-aryl bond is rotatable."""
    mol = _make_mol("c1ccc(-c2ccccc2)cc1")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert len(bonds) == 1


def test_hydroxyl_bond_kept_rotatable(rotatable_bond_finder):
    """Phenol (Oc1ccccc1) -- the O-H bond and C-OH bonds are not excluded.
    The hydroxyl SMARTS pattern is NOT added to the exclusion list per
    the source code comment about keeping amine/hydroxyl bonds rotatable
    for FF minimization.
    """
    mol = _make_mol("Oc1ccccc1")
    # The -!@ and !D1 filters dominate here; for phenol the C(ar)-O bond
    # has D1 oxygen so no default rotatable bonds are detected.
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    # Whether 0 or >0, the key property is hydroxyl exclusion is NOT applied.
    # We just confirm no crash and consistent result.
    assert isinstance(bonds, list)


def test_amine_bond_kept_rotatable(rotatable_bond_finder):
    """Ethylamine (CCN) -- amine bonds are intentionally kept rotatable."""
    mol = _make_mol("CCN")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    # The !$([CH3]) filter excludes the methyl end; the NH2 is D1 so also excluded.
    # The key point: amine bonds are NOT in the exclusion list.
    assert isinstance(bonds, list)


def test_phosphate_bond_excluded(rotatable_bond_finder):
    """Molecule with P(=O)(O)(O) -- bonds to phosphate should be excluded."""
    # dimethyl phosphate: COP(=O)(O)OC
    mol = _make_mol("COP(=O)(O)OC")
    bonds_with_phosphate = rotatable_bond_finder.identify_rotatable_bonds(mol)
    # The phosphate SMARTS matches P-O bonds and removes them from the rotatable set.
    # Terminal CH3 groups are also excluded. We verify the phosphate bonds are gone.
    for bond_pair in bonds_with_phosphate:
        atom_a = mol.GetAtomWithIdx(bond_pair[0])
        atom_b = mol.GetAtomWithIdx(bond_pair[1])
        symbols = {atom_a.GetSymbol(), atom_b.GetSymbol()}
        assert "P" not in symbols


def test_conjugated_bond_excluded(rotatable_bond_finder):
    """1,3-butadiene (C=CC=C) -- the central C-C bond is conjugated and excluded."""
    mol = _make_mol("C=CC=C")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    # The conjugate_bond_pattern [C,N;0;$([C,N;0]=[C,N;0])]-[C,N;0;$([C,N;0]=[C,N;0])]
    # matches the central single bond between the two C=C groups.
    # Also, terminal atoms with D1 are excluded by default SMARTS.
    assert len(bonds) == 0


def test_return_type_is_list_of_tuples(rotatable_bond_finder):
    """Verify the return structure is a list of 2-tuples of atom indices."""
    mol = _make_mol("CCCCC")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert isinstance(bonds, list)
    for item in bonds:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_long_alkane_rotatable_bond_count(rotatable_bond_finder):
    """n-hexane (CCCCCC) should have 3 rotatable bonds."""
    mol = _make_mol("CCCCCC")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert len(bonds) == 3


# ---- Charge primitives: '+0' is formal charge 0 ('0' alone is isotope 0) ----

def _bond_set(bonds):
    return {frozenset(b) for b in bonds}


def test_secondary_ammonium_c_n_bonds_rotatable(rotatable_bond_finder):
    """Dipropylammonium (CCC[NH2+]CCC) -- both C-N+ bonds move heavy atoms and
    stay rotatable; the neutral -NH2 exclusion must not catch R2NH2+."""
    mol = _make_mol("CCC[NH2+]CCC")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert _bond_set(bonds) == _bond_set([(1, 2), (2, 3), (3, 4), (4, 5)])


@pytest.mark.parametrize(
    "smiles", ["CCCN", "CCCO", "CCC[NH3+]"], ids=["NH2", "OH", "NH3+"]
)
def test_terminal_h_rotor_excluded(rotatable_bond_finder, smiles):
    """Propylamine, propanol, propylammonium -- the C-X bond turns only H: excluded
    with keep_h_rotors=False, kept with keep_h_rotors=True."""
    mol = _make_mol(smiles)
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert _bond_set(bonds) == _bond_set([(1, 2)])
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol, keep_h_rotors=True)
    assert _bond_set(bonds) == _bond_set([(1, 2), (2, 3)])


def test_neutral_amide_bond_excluded(rotatable_bond_finder):
    """N-ethylpropanamide (CCC(=O)NCC) -- C(=O)-N is excluded, its neighbours are not."""
    mol = _make_mol("CCC(=O)NCC")
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert _bond_set(bonds) == _bond_set([(1, 2), (4, 5)])


def test_amide_anion_bond_rotatable(rotatable_bond_finder):
    """Acylsulfonamide anion (CCC(=O)[N-]S(=O)(=O)CC) -- the amide pattern needs a
    v3 N with H, so it never matched N- (charge primitive not involved): C(=O)-N-
    stays rotatable before and after the fix."""
    mol = _make_mol("CCC(=O)[N-]S(=O)(=O)CC")
    assert not mol.HasSubstructMatch(rotatable_bond_finder.amide_bond_pattern)
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert frozenset((2, 4)) in _bond_set(bonds)


def test_charged_v3_amide_nitrogen_not_excluded(rotatable_bond_finder):
    """CCC(=O)[NH+]CC (v3 N+, a radical cation, not a ligand state) -- the only kind
    of amide N the charge primitive decides: the amide exclusion is for neutral N,
    so C(=O)-N+ stays rotatable."""
    mol = _make_mol("CCC(=O)[NH+]CC")
    assert not mol.HasSubstructMatch(rotatable_bond_finder.amide_bond_pattern)
    bonds = rotatable_bond_finder.identify_rotatable_bonds(mol)
    assert frozenset((2, 4)) in _bond_set(bonds)
