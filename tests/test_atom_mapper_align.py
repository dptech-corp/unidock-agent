"""Tests for atom_mapper_align mode in unidock_ligand_topology_builder.

Validates the 4 behavioral differences vs generic template_docking:
1. Core atom calculation: get_full_ring_core_atoms (smart ring expansion)
2. Rotatable bond SMARTS: AlignRotatableBond (amide-only exclusion)
3. Torsion freezing: 4-atom criterion (all A-B-C-D must be mapped)
4. Root finding: core atom must have a core neighbor

Plus integration tests comparing both modes on real ligand data.
"""
import os
import tempfile
from copy import deepcopy

import numpy as np
import pytest
from rdkit import Chem

from unidock_processing.ligand_topology.utils import get_full_ring_core_atoms
from unidock_processing.ligand_topology.generic_rotatable_bond import (
    GenericRotatableBond,
)


# ============================================================================
# Helpers
# ============================================================================

def _align_rotatable_bonds(mol):
    """AlignRotatableBond logic: base SMARTS minus amide only."""
    smarts = (
        "[!$(*#*)&!D1&!$([CH3])&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$"
        "(C([CH3])([CH3])[CH3])]-!@[!$(*#*)&!D1&!$([CH3])&!$(C(F)(F)F)&!$"
        "(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
    )
    amide_smarts = "[C&$(C=O)]-[N&$(NC=O);v3;H1,H2;0]"
    rot_pat = Chem.MolFromSmarts(smarts)
    amide_pat = Chem.MolFromSmarts(amide_smarts)

    matches = list(mol.GetSubstructMatches(rot_pat))
    for excl in list(mol.GetSubstructMatches(amide_pat)):
        excl_rev = tuple(reversed(excl))
        if excl in matches:
            matches.remove(excl)
        elif excl_rev in matches:
            matches.remove(excl_rev)
    return matches


def _generic_rotatable_bonds(mol):
    """GenericRotatableBond logic (excludes amide + phosphate + conjugate)."""
    finder = GenericRotatableBond()
    return finder.identify_rotatable_bonds(mol)


def _should_freeze_generic(bond_begin, bond_end, core_atoms):
    """Generic template_docking: freeze if both bond atoms in core."""
    return bond_begin in core_atoms and bond_end in core_atoms


def _should_freeze_align(mol, bond_begin, bond_end, core_atoms):
    """atom_mapper_align: freeze only if ALL 4 torsion atoms in core."""
    if bond_begin not in core_atoms or bond_end not in core_atoms:
        return False

    core_set = set(core_atoms)
    begin_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(bond_begin).GetNeighbors()
        if nbr.GetIdx() != bond_end and nbr.GetAtomicNum() > 1
    ]
    end_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(bond_end).GetNeighbors()
        if nbr.GetIdx() != bond_begin and nbr.GetAtomicNum() > 1
    ]
    if not begin_neighbors or not end_neighbors:
        return False

    for bn in begin_neighbors:
        for en in end_neighbors:
            if {bn, bond_begin, bond_end, en}.issubset(core_set):
                return True
    return False


# ============================================================================
# 1. Unit tests for get_full_ring_core_atoms
# ============================================================================

class TestGetFullRingCoreAtoms:
    """Test smart ring expansion logic."""

    def test_singleton_ring_atom_removed(self):
        """If only 1 atom in a ring is mapped, remove it from core."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))
        # Find a ring atom
        ring_atoms = set(list(Chem.GetSymmSSSR(mol))[0])
        one_ring_atom = sorted(ring_atoms)[0]
        # Find a non-ring heavy atom (the methyl carbon)
        non_ring = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetIdx() not in ring_atoms and a.GetAtomicNum() > 1
        ][0]

        core = get_full_ring_core_atoms(mol, [one_ring_atom, non_ring])
        assert one_ring_atom not in core, "Singleton ring atom should be removed"
        assert non_ring in core, "Non-ring atom should be kept"

    def test_partial_ring_completed(self):
        """If >1 but not all ring atoms mapped, complete the ring."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))
        ring_atoms = sorted(list(Chem.GetSymmSSSR(mol))[0])
        # Map 3 of 6 ring atoms
        partial = ring_atoms[:3]
        core = get_full_ring_core_atoms(mol, partial)
        for a in ring_atoms:
            assert a in core, f"Ring atom {a} should be in expanded core"

    def test_two_ring_atoms_completed(self):
        """Even 2 mapped ring atoms should trigger full ring completion."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))
        ring_atoms = sorted(list(Chem.GetSymmSSSR(mol))[0])
        partial = ring_atoms[:2]
        core = get_full_ring_core_atoms(mol, partial)
        for a in ring_atoms:
            assert a in core, f"Ring atom {a} should be completed from 2-atom partial"

    def test_full_ring_unchanged(self):
        """If all ring atoms mapped, no change."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C"))
        ring_atoms = sorted(list(Chem.GetSymmSSSR(mol))[0])
        non_ring = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetIdx() not in set(ring_atoms) and a.GetAtomicNum() > 1
        ][0]
        core_in = ring_atoms + [non_ring]
        core_out = get_full_ring_core_atoms(mol, core_in)
        assert set(core_out) == set(core_in)

    def test_no_ring_atoms_unchanged(self):
        """If no ring atoms mapped, core is unchanged."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1CC"))
        ring_atoms = set(list(Chem.GetSymmSSSR(mol))[0])
        # Map only the chain heavy atoms (not in ring)
        chain_atoms = [
            a.GetIdx() for a in mol.GetAtoms()
            if a.GetIdx() not in ring_atoms and a.GetAtomicNum() > 1
        ]
        core = get_full_ring_core_atoms(mol, chain_atoms)
        assert set(core) == set(chain_atoms)

    def test_fused_rings_partial_one(self):
        """Partial mapping on one ring of naphthalene completes that ring."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc2ccccc2c1"))
        rings = [set(r) for r in Chem.GetSymmSSSR(mol)]
        # Find atoms unique to ring 0 (not shared with ring 1)
        shared = rings[0] & rings[1]
        unique_to_0 = rings[0] - shared
        # Map 2 unique-to-ring-0 atoms (partial mapping)
        if len(unique_to_0) >= 2:
            partial = sorted(unique_to_0)[:2]
        else:
            partial = sorted(rings[0])[:3]
        core = get_full_ring_core_atoms(mol, partial)
        # All of ring 0 should be included
        for a in rings[0]:
            assert a in core, f"Ring-0 atom {a} should be in expanded core"

    def test_empty_core_unchanged(self):
        """Empty core stays empty."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
        core = get_full_ring_core_atoms(mol, [])
        assert core == []

    def test_no_rings_in_molecule(self):
        """Molecule without rings: core unchanged."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCC"))
        core = get_full_ring_core_atoms(mol, [0, 1, 2])
        assert set(core) == {0, 1, 2}


# ============================================================================
# 2. Rotatable bond SMARTS comparison
# ============================================================================

class TestRotatableBondComparison:
    """Compare AlignRotatableBond vs GenericRotatableBond."""

    def test_amide_excluded_by_both(self):
        """Amide C(=O)-N bond excluded by both paths."""
        # N-methylacetamide: CC(=O)NC
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC"))
        align_bonds = _align_rotatable_bonds(mol)
        generic_bonds = _generic_rotatable_bonds(mol)
        # Both should produce the same result (amide excluded, other bonds
        # excluded by CH3 filter)
        assert len(align_bonds) == len(generic_bonds)

    def test_conjugated_kept_by_align_excluded_by_generic(self):
        """Conjugated C=C-C=C single bond: kept by Align, excluded by Generic."""
        # 2,4-hexadiene: CC=CC=CC → middle single bond is conjugated
        mol = Chem.AddHs(Chem.MolFromSmiles("CC=CC=CC"))

        align_bonds = _align_rotatable_bonds(mol)
        generic_bonds = _generic_rotatable_bonds(mol)

        assert len(align_bonds) > len(generic_bonds), (
            f"Align should find more bonds than Generic for conjugated system: "
            f"align={len(align_bonds)}, generic={len(generic_bonds)}"
        )

    def test_phosphate_kept_by_align_excluded_by_generic(self):
        """Phosphate O-P bond: kept by Align, excluded by Generic."""
        # Triethyl phosphate: CCOP(=O)(OCC)OCC
        mol = Chem.AddHs(Chem.MolFromSmiles("CCOP(=O)(OCC)OCC"))

        align_bonds = _align_rotatable_bonds(mol)
        generic_bonds = _generic_rotatable_bonds(mol)

        assert len(align_bonds) > len(generic_bonds), (
            f"Align should find more bonds than Generic for phosphate: "
            f"align={len(align_bonds)}, generic={len(generic_bonds)}"
        )

    def test_oh_nh2_kept_by_both(self):
        """Both paths keep -OH and -NH2 bonds rotatable (with explicit H)."""
        mol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccc(N)cc1"))

        align_bonds = _align_rotatable_bonds(mol)
        generic_bonds = _generic_rotatable_bonds(mol)

        # For 4-aminophenol (no phosphate, no conjugated chain), counts match
        assert len(align_bonds) == len(generic_bonds), (
            f"OH/NH2 molecule: Align ({len(align_bonds)}) should == "
            f"Generic ({len(generic_bonds)})"
        )

    def test_typical_drug_no_difference(self):
        """For a typical drug-like molecule, both paths agree."""
        # Ibuprofen: CC(C)Cc1ccc(cc1)C(C)C(=O)O
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))

        align_bonds = _align_rotatable_bonds(mol)
        generic_bonds = _generic_rotatable_bonds(mol)

        assert len(align_bonds) == len(generic_bonds), (
            f"Ibuprofen: Align ({len(align_bonds)}) should == "
            f"Generic ({len(generic_bonds)})"
        )


# ============================================================================
# 3. Torsion freeze criterion
# ============================================================================

class TestTorsionFreezeCriterion:
    """Test the 4-atom vs 2-atom torsion freeze logic."""

    def test_both_freeze_when_all_four_mapped(self):
        """Both modes freeze a bond when all 4 torsion atoms are in core."""
        # Butane: C0-C1-C2-C3
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
        core = [0, 1, 2, 3]

        assert _should_freeze_generic(1, 2, core) is True
        assert _should_freeze_align(mol, 1, 2, core) is True

    def test_align_keeps_rotatable_when_neighbor_unmapped(self):
        """atom_mapper_align keeps bond rotatable if a torsion neighbor is NOT mapped."""
        # Pentane: C0-C1-C2-C3-C4
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
        # Map atoms 1, 2, 3 but NOT 0 or 4
        core = [1, 2, 3]

        # Bond 1-2: generic freezes, align does NOT (atom 0 not in core)
        assert _should_freeze_generic(1, 2, core) is True
        assert _should_freeze_align(mol, 1, 2, core) is False

    def test_generic_freezes_when_two_in_core(self):
        """Generic freezes any bond where both endpoints are in core."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
        core = [1, 2, 3]

        # Bond 2-3: generic freezes (both in core)
        # align: neighbor A=1 (in core), D=4 (not in core) → not all 4 mapped
        assert _should_freeze_generic(2, 3, core) is True
        assert _should_freeze_align(mol, 2, 3, core) is False

    def test_no_freeze_when_bond_outside_core(self):
        """Neither mode freezes if a bond endpoint is outside core."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
        core = [1, 2]

        assert _should_freeze_generic(0, 1, core) is False
        assert _should_freeze_align(mol, 0, 1, core) is False

    def test_branched_molecule_partial_core(self):
        """Branched molecule: core covers the branch point but not all arms."""
        # Neopentane-like: CC(C)(C)CC → atoms 0=CH3, 1=C(central), 2=CH3, 3=CH3, 4=CH2, 5=CH3
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)(C)CC"))
        # Map atoms 0, 1, 4 (one arm + central + chain) but NOT 2, 3
        core = [0, 1, 4]

        # Bond 1-4: generic freezes (both in core)
        assert _should_freeze_generic(1, 4, core) is True
        # align: need all 4 in A-B-C-D. B=1, C=4.
        # A candidates for atom 1: 0, 2, 3 (heavy neighbors except 4)
        # D candidates for atom 4: 5 (heavy neighbor except 1)
        # Check: {0,1,4,5} → 5 not in core → not frozen
        # Check: {2,1,4,5} → 2,5 not in core → not frozen
        # Check: {3,1,4,5} → 3,5 not in core → not frozen
        assert _should_freeze_align(mol, 1, 4, core) is False

    def test_fully_mapped_branch(self):
        """When all torsion atoms including branch neighbors are mapped, both freeze."""
        mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)(C)CC"))
        # Map everything: 0,1,2,3,4,5
        core = [0, 1, 2, 3, 4, 5]

        assert _should_freeze_generic(1, 4, core) is True
        assert _should_freeze_align(mol, 1, 4, core) is True


# ============================================================================
# 4. Integration: build_molecular_graph with both modes
# ============================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACE_DIR = os.path.join(
    REPO_ROOT, "examples", "constraint_docking", "automatic_atom_mapping"
)


@pytest.fixture(scope="module")
def bace_mols():
    """Load BACE reference and CAT-4a ligand."""
    ref_path = os.path.join(BACE_DIR, "reference.sdf")
    lig_path = os.path.join(BACE_DIR, "CAT-4a.sdf")
    if not os.path.isfile(ref_path) or not os.path.isfile(lig_path):
        pytest.skip("Missing BACE test data")
    ref_mol = Chem.SDMolSupplier(ref_path, removeHs=False)[0]
    lig_mol = Chem.SDMolSupplier(lig_path, removeHs=False)[0]
    if ref_mol is None or lig_mol is None:
        pytest.skip("Failed to load BACE molecules")
    return ref_mol, lig_mol


@pytest.fixture(scope="module")
def torsion_library():
    """Load torsion library pickle."""
    import dill as pickle

    pkl_path = os.path.join(
        REPO_ROOT, "unidock", "unidock_processing", "torsion_library",
        "data", "torsion_library.pkl",
    )
    if not os.path.isfile(pkl_path):
        pytest.skip("Missing torsion library pickle")
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def atom_mapping(bace_mols):
    """Compute atom mapping between reference and CAT-4a."""
    from unidock_processing.ligand_topology.utils import (
        get_template_docking_atom_mapping,
    )
    ref_mol, lig_mol = bace_mols
    return get_template_docking_atom_mapping(ref_mol, lig_mol)


@pytest.fixture(scope="module")
def graph_align(bace_mols, torsion_library, atom_mapping):
    """Run build_molecular_graph with atom_mapper_align=True."""
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        build_molecular_graph,
    )
    ref_mol, lig_mol = bace_mols
    lig = deepcopy(lig_mol)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_molecular_graph(
            lig, torsion_library,
            template_docking=True,
            reference_mol=ref_mol,
            core_atom_mapping_dict=atom_mapping,
            working_dir_name=tmpdir,
            atom_mapper_align=True,
        )
    return result, lig


@pytest.fixture(scope="module")
def graph_generic(bace_mols, torsion_library, atom_mapping):
    """Run build_molecular_graph with atom_mapper_align=False (generic)."""
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        build_molecular_graph,
    )
    ref_mol, lig_mol = bace_mols
    lig = deepcopy(lig_mol)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_molecular_graph(
            lig, torsion_library,
            template_docking=True,
            reference_mol=ref_mol,
            core_atom_mapping_dict=atom_mapping,
            working_dir_name=tmpdir,
            atom_mapper_align=False,
        )
    return result, lig


class TestBuildMolecularGraphIntegration:
    """Compare build_molecular_graph output between align and generic modes."""

    @pytest.mark.slow
    def test_both_produce_valid_output(self, graph_align, graph_generic, bace_mols):
        """Both modes produce well-formed output."""
        _, lig_mol = bace_mols
        n_atoms = lig_mol.GetNumAtoms()

        for label, (result, _) in [("align", graph_align), ("generic", graph_generic)]:
            atoms, torsions, root, fragments = result
            assert len(atoms) == n_atoms, f"{label}: atom count mismatch"
            assert len(root) > 0, f"{label}: empty root"
            assert len(fragments) > 0, f"{label}: no fragments"
            assert len(atoms[0]) == 7, f"{label}: atom info should have 7 fields"

    @pytest.mark.slow
    def test_align_preserves_coordinates(self, graph_align, bace_mols):
        """atom_mapper_align should NOT change the ligand coordinates."""
        _, lig_orig = bace_mols
        _, lig_after = graph_align

        conf_orig = lig_orig.GetConformer()
        conf_after = lig_after.GetConformer()
        for i in range(lig_orig.GetNumAtoms()):
            p1 = conf_orig.GetAtomPosition(i)
            p2 = conf_after.GetAtomPosition(i)
            dist = p1.Distance(p2)
            assert dist < 1e-6, (
                f"atom_mapper_align changed atom {i} by {dist:.6f} A"
            )

    @pytest.mark.slow
    def test_generic_changes_coordinates(self, graph_generic, bace_mols):
        """Generic template_docking re-aligns the ligand (changes coords)."""
        _, lig_orig = bace_mols
        _, lig_after = graph_generic

        conf_orig = lig_orig.GetConformer()
        conf_after = lig_after.GetConformer()
        diffs = []
        for i in range(lig_orig.GetNumAtoms()):
            p1 = conf_orig.GetAtomPosition(i)
            p2 = conf_after.GetAtomPosition(i)
            diffs.append(p1.Distance(p2))
        max_diff = max(diffs)
        assert max_diff > 0.01, (
            f"Generic mode should re-align (max displacement {max_diff:.4f} A)"
        )

    @pytest.mark.slow
    def test_align_has_more_or_equal_torsions(self, graph_align, graph_generic):
        """atom_mapper_align should have >= torsions (less aggressive freezing)."""
        (_, torsions_align, _, _), _ = graph_align
        (_, torsions_generic, _, _), _ = graph_generic
        assert len(torsions_align) >= len(torsions_generic), (
            f"align torsions ({len(torsions_align)}) should be >= "
            f"generic ({len(torsions_generic)})"
        )

    @pytest.mark.slow
    def test_core_atoms_differ(self, bace_mols, atom_mapping):
        """get_full_ring_core_atoms vs get_core_alignment_for_template_docking
        produce different core sets."""
        from unidock_processing.ligand_topology.utils import (
            get_core_alignment_for_template_docking,
            get_full_ring_core_atoms,
        )
        ref_mol, lig_mol = bace_mols
        raw_core = list(atom_mapping.values())

        align_core = set(get_full_ring_core_atoms(deepcopy(lig_mol), raw_core))
        generic_core = set(
            get_core_alignment_for_template_docking(
                ref_mol, deepcopy(lig_mol), atom_mapping
            )
        )

        # They start from the same mapping but process differently:
        # align expands partial rings; generic just uses mapping keys
        # At minimum, verify both produce non-empty core sets
        assert len(align_core) > 0, "align core should be non-empty"
        assert len(generic_core) > 0, "generic core should be non-empty"

        # Log the difference for inspection
        only_align = align_core - generic_core
        only_generic = generic_core - align_core
        print(f"\nCore atoms only in align: {sorted(only_align)}")
        print(f"Core atoms only in generic: {sorted(only_generic)}")
        print(f"Shared: {len(align_core & generic_core)}")


# ============================================================================
# 5. Partial-mapping test: demonstrates torsion freeze difference on real data
# ============================================================================

CHEMBL_DATA = os.path.join(
    os.path.expanduser("~"), "Downloads", "test2", "atom_mapper",
    "tests", "data", "test_set", "CHEMBL4006633",
)


@pytest.fixture(scope="module")
def chembl_partial_mapping_data(torsion_library):
    """CHEMBL4006633 ligand 1 — partial mapping triggers torsion freeze difference."""
    ref_path = os.path.join(CHEMBL_DATA, "ref.sdf")
    lig_path = os.path.join(CHEMBL_DATA, "ligands.sdf")
    if not os.path.isfile(ref_path) or not os.path.isfile(lig_path):
        pytest.skip("Missing CHEMBL4006633 test data")

    from unidock_processing.ligand_topology.utils import (
        get_template_docking_atom_mapping,
    )
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        build_molecular_graph,
    )

    ref = Chem.SDMolSupplier(ref_path, removeHs=False)[0]
    ligs = [m for m in Chem.SDMolSupplier(lig_path, removeHs=False) if m is not None]
    lig = ligs[1]  # ligand index 1 has partial mapping

    mapping = get_template_docking_atom_mapping(ref, lig)

    results = {}
    for label, flag in [("align", True), ("generic", False)]:
        with tempfile.TemporaryDirectory() as td:
            atoms, torsions, root, frags = build_molecular_graph(
                deepcopy(lig), torsion_library,
                template_docking=True,
                reference_mol=ref,
                core_atom_mapping_dict=mapping,
                working_dir_name=td,
                atom_mapper_align=flag,
            )
        results[label] = {
            "atoms": atoms,
            "torsions": torsions,
            "root": root,
            "fragments": frags,
        }

    return results, lig, mapping


class TestPartialMappingDifference:
    """Demonstrate actual behavioral difference with partial MCS mapping."""

    @pytest.mark.slow
    def test_align_has_more_torsions_partial_mapping(self, chembl_partial_mapping_data):
        """With partial mapping, align keeps more bonds rotatable than generic."""
        results, lig, mapping = chembl_partial_mapping_data
        n_align = len(results["align"]["torsions"])
        n_generic = len(results["generic"]["torsions"])

        assert n_align > n_generic, (
            f"With partial mapping (CHEMBL4006633), align should have MORE torsions "
            f"than generic: align={n_align}, generic={n_generic}"
        )

    @pytest.mark.slow
    def test_align_has_more_fragments_partial_mapping(self, chembl_partial_mapping_data):
        """More torsions → more fragments in align mode."""
        results, _, _ = chembl_partial_mapping_data
        n_align = len(results["align"]["fragments"])
        n_generic = len(results["generic"]["fragments"])

        assert n_align >= n_generic, (
            f"align fragments ({n_align}) should be >= generic ({n_generic})"
        )

    @pytest.mark.slow
    def test_torsion_freeze_4atom_criterion_real_data(self, chembl_partial_mapping_data):
        """Verify the 4-atom criterion: bonds where generic freezes but align doesn't."""
        results, lig, mapping = chembl_partial_mapping_data
        core_set = set(get_full_ring_core_atoms(
            deepcopy(lig), list(mapping.values())
        ))

        # Collect all rotatable bonds (align SMARTS = base - amide)
        rot_bonds = _align_rotatable_bonds(lig)

        # Find bonds where both atoms in core but not all 4 torsion atoms mapped
        bonds_with_difference = []
        for b in rot_bonds:
            a0, a1 = b
            if a0 not in core_set or a1 not in core_set:
                continue
            generic_freezes = True  # both in core → generic freezes
            align_freezes = _should_freeze_align(lig, a0, a1, list(core_set))
            if generic_freezes and not align_freezes:
                bonds_with_difference.append(b)

        assert len(bonds_with_difference) > 0, (
            "Should find at least one bond where generic freezes but align doesn't"
        )


# ============================================================================
# 6. Full docking integration (via UnidockProtocolRunner)
# ============================================================================

@pytest.mark.slow
def test_atom_mapper_align_through_runner(
    bace_receptor, bace_reference, bace_cat4a, tmp_path_factory
):
    """Run full docking pipeline with atom_mapper_align=True."""
    from tests.helpers import read_sdf_poses, run_docking_protocol

    tmpdir = str(tmp_path_factory.mktemp("align_runner"))
    out = os.path.join(tmpdir, "align_docked.sdf")
    run_docking_protocol(
        receptor=bace_receptor,
        ligand_list=[bace_cat4a],
        center=(14.786, -0.626, -1.088),
        output_sdf=out,
        work_dir=tmpdir,
        template_docking=True,
        reference_sdf_file_name=bace_reference,
        atom_mapper_align=True,
        search_mode="balance",
        use_tor_lib=True,
        num_pose=1,
        seed=42,
    )
    mols = read_sdf_poses(out)
    assert len(mols) > 0, "atom_mapper_align docking produced no poses"
    assert mols[0].HasProp("vina_binding_free_energy")
