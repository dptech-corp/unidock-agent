"""
Cross-validation test: Python VINA_ATOM_TYPE_PROPERTIES vs C++ init_default_atom_type_props.

Ensures that build_type_property_arrays() produces the EXACT same boolean arrays
as the C++ fallback defaults in constants.h::init_default_atom_type_props().
Any mismatch means the Python-generated JSON would cause different scoring behavior
than the C++ hardcoded fallback.
"""
import pytest

from unidock_processing.unidocktools.unidock_vina_atom_types import (
    VINA_ATOM_TYPE_DICT,
    VINA_ATOM_TYPE_PROPERTIES,
    VINA_VDW_RADII,
    build_type_property_arrays,
)


# ---------------------------------------------------------------
# Ground truth: C++ init_default_atom_type_props() from constants.h
# Transcribed verbatim from the C++ source.
# ---------------------------------------------------------------
CPP_NUM_TYPES = 21

# atp.is_hydrogen[0] = true;   // H
CPP_IS_HYDROGEN = {0}

# atp.is_hydrophobic: C_H(2), F_H(12), Cl_H(16), Br_H(17), I_H(18)
CPP_IS_HYDROPHOBIC = {2, 12, 16, 17, 18}

# atp.is_donor: N_D(5), N_DA(7), O_D(9), O_DA(11), Met_D(20)
CPP_IS_DONOR = {5, 7, 9, 11, 20}

# atp.is_acceptor: N_A(6), N_DA(7), O_A(10), O_DA(11)
CPP_IS_ACCEPTOR = {6, 7, 10, 11}

# vdW radii from VN_VDW_RADII[] in constants.h
CPP_VDW_RADII = [
    1.1,   # 0: H
    1.92,  # 1: B
    1.9,   # 2: C_H
    1.9,   # 3: C_P
    1.8,   # 4: N_P
    1.8,   # 5: N_D
    1.8,   # 6: N_A
    1.8,   # 7: N_DA
    1.7,   # 8: O_P
    1.7,   # 9: O_D
    1.7,   # 10: O_A
    1.7,   # 11: O_DA
    1.5,   # 12: F_H
    2.2,   # 13: Si
    2.1,   # 14: P_P
    2.0,   # 15: S_P
    1.8,   # 16: Cl_H
    2.0,   # 17: Br_H
    2.2,   # 18: I_H
    2.3,   # 19: At
    1.2,   # 20: Met_D
]


class TestAtomTypePropsConsistency:
    """Verify Python atom type properties match C++ hardcoded defaults."""

    def test_type_count(self):
        """Number of atom types must match."""
        assert len(VINA_ATOM_TYPE_DICT) == CPP_NUM_TYPES

    def test_all_types_have_properties(self):
        """Every type in VINA_ATOM_TYPE_DICT must have an entry in VINA_ATOM_TYPE_PROPERTIES."""
        for name in VINA_ATOM_TYPE_DICT:
            assert name in VINA_ATOM_TYPE_PROPERTIES, (
                f"Atom type '{name}' (id={VINA_ATOM_TYPE_DICT[name]}) missing from "
                f"VINA_ATOM_TYPE_PROPERTIES"
            )

    def test_all_properties_have_required_keys(self):
        """Each property dict must have exactly the 4 required boolean keys."""
        required = {"hydrophobic", "donor", "acceptor", "hydrogen"}
        for name, props in VINA_ATOM_TYPE_PROPERTIES.items():
            keys = set(props.keys())
            assert keys == required, (
                f"Atom type '{name}' has keys {keys}, expected {required}"
            )

    def test_is_hydrogen_matches_cpp(self):
        """is_hydrogen array must match C++ defaults."""
        is_hydrogen, _, _, _ = build_type_property_arrays()
        py_set = {i for i, v in enumerate(is_hydrogen) if v}
        assert py_set == CPP_IS_HYDROGEN, (
            f"is_hydrogen mismatch:\n"
            f"  Python: {py_set}\n"
            f"  C++:    {CPP_IS_HYDROGEN}\n"
            f"  Diff:   Python-only={py_set - CPP_IS_HYDROGEN}, "
            f"C++-only={CPP_IS_HYDROGEN - py_set}"
        )

    def test_is_hydrophobic_matches_cpp(self):
        """is_hydrophobic array must match C++ defaults."""
        _, is_hydrophobic, _, _ = build_type_property_arrays()
        py_set = {i for i, v in enumerate(is_hydrophobic) if v}
        assert py_set == CPP_IS_HYDROPHOBIC, (
            f"is_hydrophobic mismatch:\n"
            f"  Python: {py_set}\n"
            f"  C++:    {CPP_IS_HYDROPHOBIC}\n"
            f"  Diff:   Python-only={py_set - CPP_IS_HYDROPHOBIC}, "
            f"C++-only={CPP_IS_HYDROPHOBIC - py_set}"
        )

    def test_is_donor_matches_cpp(self):
        """is_donor array must match C++ defaults."""
        _, _, is_donor, _ = build_type_property_arrays()
        py_set = {i for i, v in enumerate(is_donor) if v}
        assert py_set == CPP_IS_DONOR, (
            f"is_donor mismatch:\n"
            f"  Python: {py_set}\n"
            f"  C++:    {CPP_IS_DONOR}\n"
            f"  Diff:   Python-only={py_set - CPP_IS_DONOR}, "
            f"C++-only={CPP_IS_DONOR - py_set}"
        )

    def test_is_acceptor_matches_cpp(self):
        """is_acceptor array must match C++ defaults."""
        _, _, _, is_acceptor = build_type_property_arrays()
        py_set = {i for i, v in enumerate(is_acceptor) if v}
        assert py_set == CPP_IS_ACCEPTOR, (
            f"is_acceptor mismatch:\n"
            f"  Python: {py_set}\n"
            f"  C++:    {CPP_IS_ACCEPTOR}\n"
            f"  Diff:   Python-only={py_set - CPP_IS_ACCEPTOR}, "
            f"C++-only={CPP_IS_ACCEPTOR - py_set}"
        )

    def test_vdw_radii_matches_cpp(self):
        """vdW radii list must match C++ VN_VDW_RADII[]."""
        assert len(VINA_VDW_RADII) == len(CPP_VDW_RADII), (
            f"Length mismatch: Python={len(VINA_VDW_RADII)}, C++={len(CPP_VDW_RADII)}"
        )
        for i, (py_r, cpp_r) in enumerate(zip(VINA_VDW_RADII, CPP_VDW_RADII)):
            assert py_r == pytest.approx(cpp_r, abs=1e-6), (
                f"vdW radius mismatch at index {i}: Python={py_r}, C++={cpp_r}"
            )

    def test_o_d_is_not_donor_in_original_vina(self):
        """O_D (index 9) IS a donor in C++ — verify Python agrees.

        This is a subtle case: the name 'O_D' suggests donor, and the C++
        code explicitly sets is_donor[9]=true. Catch any regression.
        """
        _, _, is_donor, _ = build_type_property_arrays()
        # O_D is index 9 — originally IS a donor in Vina
        assert is_donor[9] is False or is_donor[9] is True  # sanity
        # Check actual value matches C++
        expected = 9 in CPP_IS_DONOR
        assert is_donor[9] == expected, (
            f"O_D (index 9) donor status: Python={is_donor[9]}, C++={expected}"
        )

    def test_n_da_is_both_donor_and_acceptor(self):
        """N_DA (index 7) must be both donor AND acceptor."""
        _, _, is_donor, is_acceptor = build_type_property_arrays()
        assert is_donor[7] is True, "N_DA must be donor"
        assert is_acceptor[7] is True, "N_DA must be acceptor"

    def test_o_da_is_both_donor_and_acceptor(self):
        """O_DA (index 11) must be both donor AND acceptor."""
        _, _, is_donor, is_acceptor = build_type_property_arrays()
        assert is_donor[11] is True, "O_DA must be donor"
        assert is_acceptor[11] is True, "O_DA must be acceptor"

    def test_only_h_is_hydrogen(self):
        """Only type 0 (H) should be marked as hydrogen."""
        is_hydrogen, _, _, _ = build_type_property_arrays()
        for i, v in enumerate(is_hydrogen):
            if i == 0:
                assert v is True, "H (index 0) must be hydrogen"
            else:
                assert v is False, f"Index {i} should NOT be hydrogen, but is_hydrogen={v}"

    def test_met_d_is_donor_only(self):
        """Met_D (index 20) is donor but not acceptor, not hydrophobic, not hydrogen."""
        is_h, is_hp, is_d, is_a = build_type_property_arrays()
        assert is_d[20] is True, "Met_D must be donor"
        assert is_a[20] is False, "Met_D must NOT be acceptor"
        assert is_hp[20] is False, "Met_D must NOT be hydrophobic"
        assert is_h[20] is False, "Met_D must NOT be hydrogen"
