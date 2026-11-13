import numpy as np
import pytest
import xml.etree.ElementTree as ET

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry.rdGeometry import Point3D
from MDAnalysis.lib.distances import calc_dihedrals

from unidock_processing.torsion_library.utils import (
    canonicalize_angle,
    get_detailed_torsion_ranges,
    get_merged_torsion_ranges,
    get_pattern_atom_mapping,
    analyze_torsion_rule,
    get_torsion_atom_idx_tuple,
    get_torsion_mobile_atom_idx_list,
    rotate_torsion_angle,
)


def _prepare_mol(smiles):
    """Create an embedded molecule with internal_atom_idx set on every atom."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    for atom_idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(atom_idx).SetIntProp("internal_atom_idx", atom_idx)
    return mol


# ---------------------------------------------------------------------------
# canonicalize_angle
# ---------------------------------------------------------------------------

class TestCanonicalizeAngle:
    @pytest.mark.parametrize(
        "angle, expected",
        [
            (0.0, 0.0),
            (90.0, 90.0),
            (-90.0, -90.0),
            (180.0, 180.0),
            (-180.0, 180.0),
            (270.0, -90.0),
            (360.0, 0.0),
            (-270.0, 90.0),
            (181.0, -179.0),
            (-181.0, 179.0),
            (45.5, 45.5),
            (-45.5, -45.5),
        ],
    )
    def test_various_angles(self, angle, expected):
        result = canonicalize_angle(angle)
        assert result == pytest.approx(expected)

    def test_boundary_minus_180_maps_to_positive(self):
        assert canonicalize_angle(-180.0) == pytest.approx(180.0)

    def test_boundary_180_stays(self):
        assert canonicalize_angle(180.0) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# get_detailed_torsion_ranges
# ---------------------------------------------------------------------------

class TestGetDetailedTorsionRanges:
    def test_normal_range_unchanged(self):
        result = get_detailed_torsion_ranges([(-60.0, 60.0)])
        assert result == [(-60.0, 60.0)]

    def test_wrap_around_range_splits(self):
        result = get_detailed_torsion_ranges([(170.0, -170.0)])
        assert result == [(170.0, 180.0), (-180.0, -170.0)]

    def test_multiple_ranges_mixed(self):
        result = get_detailed_torsion_ranges([(-30.0, 30.0), (160.0, -160.0)])
        assert result == [(-30.0, 30.0), (160.0, 180.0), (-180.0, -160.0)]

    def test_empty_list(self):
        result = get_detailed_torsion_ranges([])
        assert result == []

    def test_single_point_range(self):
        result = get_detailed_torsion_ranges([(60.0, 60.0)])
        assert result == [(60.0, 60.0)]

    def test_full_range_wraps(self):
        result = get_detailed_torsion_ranges([(100.0, -100.0)])
        assert result == [(100.0, 180.0), (-180.0, -100.0)]


# ---------------------------------------------------------------------------
# get_merged_torsion_ranges
# ---------------------------------------------------------------------------

class TestGetMergedTorsionRanges:
    def test_non_overlapping_ranges(self):
        result = get_merged_torsion_ranges([(-60.0, -30.0), (30.0, 60.0)])
        assert result == [(-60.0, -30.0), (30.0, 60.0)]

    def test_overlapping_ranges_merge(self):
        result = get_merged_torsion_ranges([(-60.0, 10.0), (-10.0, 60.0)])
        assert result == [(-60.0, 60.0)]

    def test_adjacent_ranges_remain_separate(self):
        result = get_merged_torsion_ranges([(-60.0, -30.0), (-29.0, 0.0)])
        assert result == [(-60.0, -30.0), (-29.0, 0.0)]

    def test_contained_range_merges(self):
        result = get_merged_torsion_ranges([(-90.0, 90.0), (-30.0, 30.0)])
        assert result == [(-90.0, 90.0)]

    def test_empty_list(self):
        result = get_merged_torsion_ranges([])
        assert result == []

    def test_single_range(self):
        result = get_merged_torsion_ranges([(10.0, 20.0)])
        assert result == [(10.0, 20.0)]

    def test_three_overlapping_ranges(self):
        result = get_merged_torsion_ranges(
            [(-60.0, 0.0), (-10.0, 50.0), (40.0, 90.0)]
        )
        assert result == [(-60.0, 90.0)]

    def test_unsorted_input_gets_sorted(self):
        result = get_merged_torsion_ranges([(30.0, 60.0), (-60.0, -30.0)])
        assert result == [(-60.0, -30.0), (30.0, 60.0)]


# ---------------------------------------------------------------------------
# get_pattern_atom_mapping
# ---------------------------------------------------------------------------

class TestGetPatternAtomMapping:
    def test_smarts_with_atom_maps(self):
        pattern = Chem.MolFromSmarts("[*:1]~[CX4:2]!@[CX4:3]~[*:4]")
        result = get_pattern_atom_mapping(pattern)
        assert 1 in result
        assert 2 in result
        assert 3 in result
        assert 4 in result
        assert len(result) == 4

    def test_smarts_without_atom_maps(self):
        pattern = Chem.MolFromSmarts("[CX4]~[CX4]")
        result = get_pattern_atom_mapping(pattern)
        assert result == {}

    def test_partial_atom_maps(self):
        pattern = Chem.MolFromSmarts("[*:1]~[CX4]!@[CX4:3]~[*]")
        result = get_pattern_atom_mapping(pattern)
        assert len(result) == 2
        assert 1 in result
        assert 3 in result

    def test_atom_map_indices_correct(self):
        pattern = Chem.MolFromSmarts("[*:1]~[CX4:2]!@[n:3]~[*:4]")
        result = get_pattern_atom_mapping(pattern)
        assert result[1] == 0
        assert result[2] == 1
        assert result[3] == 2
        assert result[4] == 3


# ---------------------------------------------------------------------------
# analyze_torsion_rule
# ---------------------------------------------------------------------------

class TestAnalyzeTorsionRule:
    @staticmethod
    def _make_torsion_rule_xml(
        smarts,
        angle_values,
        angle_tolerances,
        histogram_counts=None,
        node_idx="test_0",
    ):
        rule = ET.Element("torsionRule", smarts=smarts, node_idx=node_idx)
        angle_list = ET.SubElement(rule, "angleList")
        for val, tol in zip(angle_values, angle_tolerances):
            ET.SubElement(
                angle_list,
                "angle",
                value=str(val),
                tolerance1=str(tol),
                score="1.0",
            )
        if histogram_counts is not None:
            hist = ET.SubElement(rule, "histogram")
            for count in histogram_counts:
                ET.SubElement(hist, "bin", count=str(count))
        return rule

    def test_basic_torsion_rule_parsing(self):
        node = self._make_torsion_rule_xml(
            smarts="[*:1]~[CX4:2]!@[CX4:3]~[*:4]",
            angle_values=[60.0, 180.0, -60.0],
            angle_tolerances=[30.0, 30.0, 30.0],
            histogram_counts=[100, 200, 300],
        )
        result = analyze_torsion_rule(node)
        assert result is not None
        assert result["type"] == "torsion_rule"
        assert result["smarts"] == "[*:1]~[CX4:2]!@[CX4:3]~[*:4]"
        assert result["node_idx"] == "test_0"
        assert len(result["angle_value"]) == 3
        assert result["angle_value"] == [60.0, 180.0, -60.0]
        assert result["histogram_weight"] == [100.0, 200.0, 300.0]

    def test_returns_none_for_N_lp_smarts(self):
        node = self._make_torsion_rule_xml(
            smarts="[*:1]~[N_lp:2]!@[CX4:3]~[*:4]",
            angle_values=[60.0],
            angle_tolerances=[30.0],
        )
        result = analyze_torsion_rule(node)
        assert result is None

    def test_returns_none_for_invalid_smarts(self):
        node = self._make_torsion_rule_xml(
            smarts="[invalid_smarts_that_rdkit_cannot_parse",
            angle_values=[60.0],
            angle_tolerances=[30.0],
        )
        result = analyze_torsion_rule(node)
        assert result is None

    def test_evenly_distributed_detection(self):
        values = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0,
                  30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
        tolerances = [15.0] * 12
        node = self._make_torsion_rule_xml(
            smarts="[*:1]~[CX4:2]!@[n:3]~[*:4]",
            angle_values=values,
            angle_tolerances=tolerances,
            histogram_counts=[100] * 36,
        )
        result = analyze_torsion_rule(node)
        assert result is not None
        assert result["angle_range"] == "evenly_distributed"

    def test_atom_map_dict_populated(self):
        node = self._make_torsion_rule_xml(
            smarts="[*:1]~[CX4:2]!@[CX4:3]~[*:4]",
            angle_values=[60.0],
            angle_tolerances=[30.0],
        )
        result = analyze_torsion_rule(node)
        assert len(result["atom_map_dict"]) == 4

    def test_no_histogram_returns_none_weight(self):
        node = self._make_torsion_rule_xml(
            smarts="[*:1]~[CX4:2]!@[CX4:3]~[*:4]",
            angle_values=[60.0],
            angle_tolerances=[30.0],
            histogram_counts=None,
        )
        result = analyze_torsion_rule(node)
        assert result["histogram_weight"] is None


# ---------------------------------------------------------------------------
# get_torsion_atom_idx_tuple
# ---------------------------------------------------------------------------

class TestGetTorsionAtomIdxTuple:
    def test_ethane_rotatable_bond(self):
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        rotatable_bond = (0, 1)
        result = get_torsion_atom_idx_tuple(mol, rotatable_bond)
        assert len(result) == 4
        assert result[1] == 0
        assert result[2] == 1
        assert result[0] != 1
        assert result[3] != 0

    def test_butane_middle_bond(self):
        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        rotatable_bond = (1, 2)
        result = get_torsion_atom_idx_tuple(mol, rotatable_bond)
        assert result[1] == 1
        assert result[2] == 2
        atom_0 = result[0]
        atom_3 = result[3]
        assert atom_0 in [
            n.GetIdx() for n in mol.GetAtomWithIdx(1).GetNeighbors() if n.GetIdx() != 2
        ]
        assert atom_3 in [
            n.GetIdx() for n in mol.GetAtomWithIdx(2).GetNeighbors() if n.GetIdx() != 1
        ]


# ---------------------------------------------------------------------------
# get_torsion_mobile_atom_idx_list
# ---------------------------------------------------------------------------

class TestGetTorsionMobileAtomIdxList:
    def test_butane_mobile_atoms(self):
        mol = _prepare_mol("CCCC")
        rotatable_bond = (1, 2)
        root_atom_idx = 0
        mobile_atoms = get_torsion_mobile_atom_idx_list(
            mol, rotatable_bond, root_atom_idx
        )
        assert 0 not in mobile_atoms
        assert 1 not in mobile_atoms
        assert 2 in mobile_atoms
        assert 3 in mobile_atoms

    def test_butane_mobile_atoms_reversed_root(self):
        mol = _prepare_mol("CCCC")
        rotatable_bond = (1, 2)
        root_atom_idx = 3
        mobile_atoms = get_torsion_mobile_atom_idx_list(
            mol, rotatable_bond, root_atom_idx
        )
        assert 0 in mobile_atoms
        assert 1 in mobile_atoms
        assert 2 not in mobile_atoms
        assert 3 not in mobile_atoms

    def test_propane_with_hydrogens(self):
        mol = _prepare_mol("CCC")
        rotatable_bond = (0, 1)
        root_atom_idx = 2
        mobile_atoms = get_torsion_mobile_atom_idx_list(
            mol, rotatable_bond, root_atom_idx
        )
        assert 0 in mobile_atoms
        h_on_c0 = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(0).GetNeighbors()
            if n.GetAtomicNum() == 1
        ]
        for h_idx in h_on_c0:
            assert h_idx in mobile_atoms


# ---------------------------------------------------------------------------
# rotate_torsion_angle
# ---------------------------------------------------------------------------

class TestRotateTorsionAngle:
    @staticmethod
    def _measure_dihedral(mol, idx0, idx1, idx2, idx3):
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        angle_rad = calc_dihedrals(
            pos[idx0], pos[idx1], pos[idx2], pos[idx3]
        )
        return float(np.degrees(angle_rad))

    def test_rotate_to_target_angle(self):
        mol = _prepare_mol("CCCC")
        rotatable_bond = (1, 2)
        torsion_idx = get_torsion_atom_idx_tuple(mol, rotatable_bond)
        mobile_atoms = get_torsion_mobile_atom_idx_list(mol, rotatable_bond, 0)
        target = 120.0
        rotate_torsion_angle(mol, torsion_idx, mobile_atoms, target)
        measured = self._measure_dihedral(
            mol, torsion_idx[0], torsion_idx[1], torsion_idx[2], torsion_idx[3]
        )
        assert measured == pytest.approx(target, abs=1.0)

    def test_rotate_to_zero(self):
        mol = _prepare_mol("CCCC")
        rotatable_bond = (1, 2)
        torsion_idx = get_torsion_atom_idx_tuple(mol, rotatable_bond)
        mobile_atoms = get_torsion_mobile_atom_idx_list(mol, rotatable_bond, 0)
        target = 0.0
        rotate_torsion_angle(mol, torsion_idx, mobile_atoms, target)
        measured = self._measure_dihedral(
            mol, torsion_idx[0], torsion_idx[1], torsion_idx[2], torsion_idx[3]
        )
        assert measured == pytest.approx(target, abs=1.0)

    def test_rotate_to_negative_angle(self):
        mol = _prepare_mol("CCCC")
        rotatable_bond = (1, 2)
        torsion_idx = get_torsion_atom_idx_tuple(mol, rotatable_bond)
        mobile_atoms = get_torsion_mobile_atom_idx_list(mol, rotatable_bond, 0)
        target = -60.0
        rotate_torsion_angle(mol, torsion_idx, mobile_atoms, target)
        measured = self._measure_dihedral(
            mol, torsion_idx[0], torsion_idx[1], torsion_idx[2], torsion_idx[3]
        )
        assert measured == pytest.approx(target, abs=1.0)

    def test_rotate_preserves_bond_lengths(self):
        mol = _prepare_mol("CCCC")
        conf = mol.GetConformer()
        pos_before = conf.GetPositions().copy()
        dist_01_before = np.linalg.norm(pos_before[0] - pos_before[1])
        dist_12_before = np.linalg.norm(pos_before[1] - pos_before[2])
        dist_23_before = np.linalg.norm(pos_before[2] - pos_before[3])

        rotatable_bond = (1, 2)
        torsion_idx = get_torsion_atom_idx_tuple(mol, rotatable_bond)
        mobile_atoms = get_torsion_mobile_atom_idx_list(mol, rotatable_bond, 0)
        rotate_torsion_angle(mol, torsion_idx, mobile_atoms, 90.0)

        pos_after = mol.GetConformer().GetPositions()
        dist_01_after = np.linalg.norm(pos_after[0] - pos_after[1])
        dist_12_after = np.linalg.norm(pos_after[1] - pos_after[2])
        dist_23_after = np.linalg.norm(pos_after[2] - pos_after[3])

        assert dist_01_after == pytest.approx(dist_01_before, abs=0.01)
        assert dist_12_after == pytest.approx(dist_12_before, abs=0.01)
        assert dist_23_after == pytest.approx(dist_23_before, abs=0.01)

    def test_successive_rotations(self):
        mol = _prepare_mol("CCCC")
        rotatable_bond = (1, 2)
        torsion_idx = get_torsion_atom_idx_tuple(mol, rotatable_bond)
        mobile_atoms = get_torsion_mobile_atom_idx_list(mol, rotatable_bond, 0)

        for target in [60.0, -120.0, 180.0, 0.0]:
            rotate_torsion_angle(mol, torsion_idx, mobile_atoms, target)
            measured = self._measure_dihedral(
                mol, torsion_idx[0], torsion_idx[1], torsion_idx[2], torsion_idx[3]
            )
            assert measured == pytest.approx(target, abs=1.0)
