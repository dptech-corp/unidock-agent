VINA_ATOM_TYPE_DICT = {
    "H": 0,
    "B": 1,
    "C_H": 2,
    "C_P": 3,
    "N_P": 4,
    "N_D": 5,
    "N_A": 6,
    "N_DA": 7,
    "O_P": 8,
    "O_D": 9,
    "O_A": 10,
    "O_DA": 11,
    "F_H": 12,
    "Si": 13,
    "P_P": 14,
    "S_P": 15,
    "Cl_H": 16,
    "Br_H": 17,
    "I_H": 18,
    "At": 19,
    "Met_D": 20,
}

# vdW radii for Vina scoring function (indexed by atom type integer)
# Source: AutoDock Vina, H and B from 10.1021/jp8111556
VINA_VDW_RADII = [
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

# Per-type scoring properties: hydrophobic, donor, acceptor, hydrogen
# These are serialized to JSON so the C++ engine can be fully data-driven
# (no hardcoded type-to-property mapping in C++).
VINA_ATOM_TYPE_PROPERTIES = {
    "H":     {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": True},
    "B":     {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": False},
    "C_H":   {"hydrophobic": True,  "donor": False, "acceptor": False, "hydrogen": False},
    "C_P":   {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": False},
    "N_P":   {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": False},
    "N_D":   {"hydrophobic": False, "donor": True,  "acceptor": False, "hydrogen": False},
    "N_A":   {"hydrophobic": False, "donor": False, "acceptor": True,  "hydrogen": False},
    "N_DA":  {"hydrophobic": False, "donor": True,  "acceptor": True,  "hydrogen": False},
    "O_P":   {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": False},
    "O_D":   {"hydrophobic": False, "donor": True,  "acceptor": False, "hydrogen": False},
    "O_A":   {"hydrophobic": False, "donor": False, "acceptor": True,  "hydrogen": False},
    "O_DA":  {"hydrophobic": False, "donor": True,  "acceptor": True,  "hydrogen": False},
    "F_H":   {"hydrophobic": True,  "donor": False, "acceptor": False, "hydrogen": False},
    "Si":    {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": False},
    "P_P":   {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": False},
    "S_P":   {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": False},
    "Cl_H":  {"hydrophobic": True,  "donor": False, "acceptor": False, "hydrogen": False},
    "Br_H":  {"hydrophobic": True,  "donor": False, "acceptor": False, "hydrogen": False},
    "I_H":   {"hydrophobic": True,  "donor": False, "acceptor": False, "hydrogen": False},
    "At":    {"hydrophobic": False, "donor": False, "acceptor": False, "hydrogen": False},
    "Met_D": {"hydrophobic": False, "donor": True,  "acceptor": False, "hydrogen": False},
}


def build_type_property_arrays():
    """Build boolean arrays indexed by Vina atom type integer for JSON serialization.
    Returns 4 lists: type_is_hydrogen, type_is_hydrophobic, type_is_donor, type_is_acceptor.
    """
    n = len(VINA_ATOM_TYPE_DICT)
    is_hydrogen = [False] * n
    is_hydrophobic = [False] * n
    is_donor = [False] * n
    is_acceptor = [False] * n

    for name, idx in VINA_ATOM_TYPE_DICT.items():
        props = VINA_ATOM_TYPE_PROPERTIES.get(name, {})
        is_hydrogen[idx] = props.get("hydrogen", False)
        is_hydrophobic[idx] = props.get("hydrophobic", False)
        is_donor[idx] = props.get("donor", False)
        is_acceptor[idx] = props.get("acceptor", False)

    return is_hydrogen, is_hydrophobic, is_donor, is_acceptor
