import os
from copy import deepcopy
import warnings

import numpy as np
import networkx as nx

from rdkit import Chem

def convert_v3000_mol_to_v2000_sdf(v3000_mol, v2000_sdf_file_name):
    mol_props_dict = v3000_mol.GetPropsAsDict()
    v2000_str = Chem.MolToV2KMolBlock(v3000_mol)
    for prop_key, prop_value in mol_props_dict.items():
        v2000_str += f">  <{prop_key}>  (1) \n"
        v2000_str += f"{prop_value}\n\n"

    with open(v2000_sdf_file_name, "w") as f:
        f.write(v2000_str)

def recover_full_atom_mapping_from_heavy_atoms(
    reference_mol,
    reference_mol_no_H,
    query_mol,
    query_mol_no_H,
    heavy_atom_mapping_dict,
):
    full_atom_mapping_dict = {}
    for reference_no_H_atom_idx in heavy_atom_mapping_dict:
        query_no_H_atom_idx = heavy_atom_mapping_dict[reference_no_H_atom_idx]

        reference_no_H_atom = reference_mol_no_H.GetAtomWithIdx(reference_no_H_atom_idx)
        reference_atom_idx = reference_no_H_atom.GetIntProp("internal_atom_idx")
        reference_atom = reference_mol.GetAtomWithIdx(reference_atom_idx)

        query_no_H_atom = query_mol_no_H.GetAtomWithIdx(query_no_H_atom_idx)
        query_atom_idx = query_no_H_atom.GetIntProp("internal_atom_idx")
        query_atom = query_mol.GetAtomWithIdx(query_atom_idx)

        full_atom_mapping_dict[reference_atom_idx] = query_atom_idx

        if reference_atom.GetSymbol() == query_atom.GetSymbol():
            reference_neighbor_atom_list = list(reference_atom.GetNeighbors())
            query_neighbor_atom_list = list(query_atom.GetNeighbors())

            if len(reference_neighbor_atom_list) != len(query_neighbor_atom_list):
                warnings.warn(
                    "Number of neighbor atoms does not match "
                    "between query and reference atoms! "
                    "Please look at this case carefully!"
                )

            reference_neighbor_h_atom_idx_list = []
            query_neighbor_h_atom_idx_list = []

            for reference_neighbor_atom in reference_neighbor_atom_list:
                if reference_neighbor_atom.GetSymbol() == "H":
                    reference_neighbor_h_atom_idx_list.append(
                        reference_neighbor_atom.GetIdx()
                    )

            for query_neighbor_atom in query_neighbor_atom_list:
                if query_neighbor_atom.GetSymbol() == "H":
                    query_neighbor_h_atom_idx_list.append(query_neighbor_atom.GetIdx())

            num_reference_neighbor_h_atoms = len(reference_neighbor_h_atom_idx_list)
            num_query_neighbor_h_atoms = len(query_neighbor_h_atom_idx_list)

            if num_reference_neighbor_h_atoms <= num_query_neighbor_h_atoms:
                num_neighbor_h_atoms = num_reference_neighbor_h_atoms
                query_neighbor_h_atom_idx_list = query_neighbor_h_atom_idx_list[
                    :num_reference_neighbor_h_atoms
                ]
            else:
                num_neighbor_h_atoms = num_query_neighbor_h_atoms
                reference_neighbor_h_atom_idx_list = reference_neighbor_h_atom_idx_list[
                    :num_query_neighbor_h_atoms
                ]

            for neighbor_h_idx in range(num_neighbor_h_atoms):
                reference_neighbor_h_atom_idx = reference_neighbor_h_atom_idx_list[
                    neighbor_h_idx
                ]
                query_neighbor_h_atom_idx = query_neighbor_h_atom_idx_list[
                    neighbor_h_idx
                ]
                full_atom_mapping_dict[reference_neighbor_h_atom_idx] = (
                    query_neighbor_h_atom_idx
                )

    return full_atom_mapping_dict


def calculate_center_of_mass(mol):
    positions_array = mol.GetConformer().GetPositions()
    num_atoms = mol.GetNumAtoms()

    total_mass = 0.0
    coords_mass_product = 0.0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        mass = atom.GetMass()
        coords = positions_array[atom_idx, :]
        coords_mass_product += coords * mass
        total_mass += mass

    return coords_mass_product / total_mass


def assign_atom_properties(mol):
    atom_positions = mol.GetConformer().GetPositions()
    num_atoms = mol.GetNumAtoms()

    internal_atom_idx = 0
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetIntProp("internal_atom_idx", atom_idx)
        atom.SetIntProp("sdf_atom_idx", atom_idx + 1)
        if not atom.HasProp("atom_name"):
            atom_element = atom.GetSymbol()
            atom_name = atom_element + str(internal_atom_idx + 1)
            atom.SetProp("atom_name", atom_name)
            atom.SetProp("residue_name", "MOL")
            atom.SetIntProp("residue_idx", 1)
            atom.SetProp("chain_idx", "A")
            internal_atom_idx += 1

        atom.SetDoubleProp("charge", atom.GetDoubleProp("_GasteigerCharge"))
        atom.SetDoubleProp("x", atom_positions[atom_idx, 0])
        atom.SetDoubleProp("y", atom_positions[atom_idx, 1])
        atom.SetDoubleProp("z", atom_positions[atom_idx, 2])


def calculate_nonbonded_atom_pairs(mol):
    num_atoms = mol.GetNumAtoms()
    atom_pair_12_13_nested_list = [None] * num_atoms
    atom_pair_14_nested_list = [None] * num_atoms

    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_pair_12_list = []
        atom_pair_13_set = set()
        atom_pair_14_set = set()

        neighbor_atom_12_list = list(atom.GetNeighbors())
        for neighbor_atom_12 in neighbor_atom_12_list:
            neighbor_atom_12_idx = neighbor_atom_12.GetIdx()
            atom_pair_12_list.append(neighbor_atom_12_idx)
            neighbor_atom_13_list = list(neighbor_atom_12.GetNeighbors())
            for neighbor_atom_13 in neighbor_atom_13_list:
                neighbor_atom_13_idx = neighbor_atom_13.GetIdx()
                if neighbor_atom_13_idx == atom_idx:
                    continue
                else:
                    atom_pair_13_set.add(neighbor_atom_13_idx)
                    neighbor_atom_14_list = list(neighbor_atom_13.GetNeighbors())
                    for neighbor_atom_14 in neighbor_atom_14_list:
                        neighbor_atom_14_idx = neighbor_atom_14.GetIdx()
                        if neighbor_atom_14_idx == atom_idx:
                            continue
                        else:
                            atom_pair_14_set.add(neighbor_atom_14_idx)

        atom_pair_13_raw_list = list(atom_pair_13_set)
        atom_pair_13_list = []
        for atom_idx_13 in atom_pair_13_raw_list:
            if atom_idx_13 not in atom_pair_12_list:
                atom_pair_13_list.append(atom_idx_13)

        atom_pair_14_raw_list = list(atom_pair_14_set)
        atom_pair_14_list = []
        for atom_idx_14 in atom_pair_14_raw_list:
            if (
                atom_idx_14 not in atom_pair_12_list
                and atom_idx_14 not in atom_pair_13_list
            ):
                atom_pair_14_list.append(atom_idx_14)

        atom_pair_12_13_list = atom_pair_12_list + atom_pair_13_list
        atom_pair_12_13_list.sort()
        atom_pair_14_list.sort()

        atom_pair_12_13_nested_list[atom_idx] = atom_pair_12_13_list
        atom_pair_14_nested_list[atom_idx] = atom_pair_14_list

    return atom_pair_12_13_nested_list, atom_pair_14_nested_list


def record_gaff2_atom_types_and_parameters(
    ligand_sdf_file_name, ligand_charge_method, working_dir_name
):
    ## Deal with sulfonamide with negative charge on Nitrogen atom cases
    ##############################################################################
    ##############################################################################
    mol = Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)[0]
    num_atoms = mol.GetNumAtoms()

    mol_copy = deepcopy(mol)
    sulfonamide_pattern = Chem.MolFromSmarts("[$(NS=O),$(NP=O);-1]")
    sulfonamide_N_tuple_list = list(mol_copy.GetSubstructMatches(sulfonamide_pattern))
    sulfonamide_N_atom_idx_list = [
        sulfonamide_N_tuple[0] for sulfonamide_N_tuple in sulfonamide_N_tuple_list
    ]

    for atom_idx in sulfonamide_N_atom_idx_list:
        atom = mol_copy.GetAtomWithIdx(atom_idx)
        num_implicit_Hs = atom.GetNumImplicitHs()
        num_explicit_Hs = atom.GetNumExplicitHs()
        total_current_num_Hs = num_implicit_Hs + num_explicit_Hs
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(total_current_num_Hs + 1)

    tetazole_pattern = Chem.MolFromSmarts("[n;H0;-1]")
    tetazole_N_tuple_list = list(mol_copy.GetSubstructMatches(tetazole_pattern))
    tetazole_N_atom_idx_list = [
        tetazole_N_tuple[0] for tetazole_N_tuple in tetazole_N_tuple_list
    ]

    for atom_idx in tetazole_N_atom_idx_list:
        atom = mol_copy.GetAtomWithIdx(atom_idx)
        num_implicit_Hs = atom.GetNumImplicitHs()
        num_explicit_Hs = atom.GetNumExplicitHs()
        total_current_num_Hs = num_implicit_Hs + num_explicit_Hs
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(total_current_num_Hs + 1)

    Chem.GetSymmSSSR(mol_copy)
    mol_copy.UpdatePropertyCache(strict=False)

    # AddHs for both sulfonamide tetrazole cases and covalent ligand dummy hydrogens.
    mol_copy_h = Chem.AddHs(mol_copy, addCoords=True)
    formal_charge = Chem.GetFormalCharge(mol_copy_h)

    temp_ligand_sdf_file_name = os.path.join(working_dir_name, "ligand_temp.sdf")
    temp_ligand_mol2_file_name = os.path.join(working_dir_name, "ligand_temp.mol2")
    temp_ligand_frcmod_file_name = os.path.join(working_dir_name, "ligand_temp.frcmod")
    temp_antechamber_log_file_name = os.path.join(
        working_dir_name, "ligand_temp_antechamber.log"
    )
    temp_antechamber_frcmod_file_name = os.path.join(
        working_dir_name, "ANTECHAMBER.FRCMOD"
    )

    convert_v3000_mol_to_v2000_sdf(mol_copy_h, temp_ligand_sdf_file_name)
    ##############################################################################
    ##############################################################################

    ##############################################################################
    ## Execute ambertools
    antechamber_command = (
        f"cd {working_dir_name}; "
        f"antechamber -i {temp_ligand_sdf_file_name} -fi sdf "
        f"-o {temp_ligand_mol2_file_name} -fo mol2 "
        f"-at gaff2 -c {ligand_charge_method} -nc {formal_charge} "
        f"-eq 2 -pf y >> {temp_antechamber_log_file_name}"
    )
    parmchk_command = (
        f"cd {working_dir_name}; "
        f"parmchk2 -i {temp_ligand_mol2_file_name} -f mol2 "
        f"-a Y -s 2 -o {temp_ligand_frcmod_file_name}"
    )
    remove_files_command = (
        f"cd {working_dir_name}; rm {temp_antechamber_frcmod_file_name}"
    )
    os.system(antechamber_command)
    os.system(parmchk_command)
    os.system(remove_files_command)
    ##############################################################################

    ##############################################################################
    ## Record atom types and parameters
    ## mol2 file parsing
    atom_type_list = [None] * num_atoms
    partial_charge_list = [None] * num_atoms
    atom_parameter_dict = {}
    torsion_parameter_dict = {}

    with open(temp_ligand_mol2_file_name) as mol2_file:
        line_list = mol2_file.readlines()

    for line_idx, line in enumerate(line_list):
        if line.startswith("@<TRIPOS>ATOM"):
            atom_header_line_idx = line_idx
        elif line.startswith("@<TRIPOS>BOND"):
            bond_header_line_idx = line_idx

    atom_idx = 0
    for line_idx in range(atom_header_line_idx + 1, bond_header_line_idx):
        line = line_list[line_idx]
        line_split_list = line.strip().split()
        atom_type = line_split_list[5]
        partial_charge = float(line_split_list[8])

        atom_type_list[atom_idx] = atom_type
        partial_charge_list[atom_idx] = partial_charge
        atom_idx += 1

        if atom_idx == num_atoms:
            break

    ## frcmod file parsing
    ## ambertools frcmod format refer to https://ambermd.org/FileFormats.php
    with open(temp_ligand_frcmod_file_name) as frcmod_file:
        line_list = frcmod_file.readlines()

    num_frcmod_lines = len(line_list)

    for line_idx, line in enumerate(line_list):
        if line.startswith("MASS"):
            atom_header_line_idx = line_idx
        elif line.startswith("BOND"):
            bond_header_line_idx = line_idx

    for line_idx in range(atom_header_line_idx + 1, bond_header_line_idx):
        line = line_list[line_idx]
        if len(line) < 4:
            continue

        line_split_list = line.strip().split()
        atom_type = line_split_list[0]
        mass = float(line_split_list[1])
        atom_parameter_dict[atom_type] = {}
        atom_parameter_dict[atom_type]["mass"] = mass

    for line_idx, line in enumerate(line_list):
        if line.startswith("NONBON"):
            nonbond_header_line_idx = line_idx

    for line_idx in range(nonbond_header_line_idx + 1, num_frcmod_lines):
        line = line_list[line_idx]
        if len(line) < 4:
            continue

        line_split_list = line.strip().split()
        atom_type = line_split_list[0]
        sigma = float(line_split_list[1])
        epsilon = float(line_split_list[2])
        atom_parameter_dict[atom_type]["sigma"] = sigma
        atom_parameter_dict[atom_type]["epsilon"] = epsilon

    for line_idx, line in enumerate(line_list):
        if line.startswith("DIHE"):
            torsion_header_line_idx = line_idx
        elif line.startswith("IMPROPER"):
            improper_header_line_idx = line_idx

    for line_idx in range(torsion_header_line_idx + 1, improper_header_line_idx):
        line = line_list[line_idx]
        if len(line) < 4:
            continue

        torsion_type_str = line[:11]
        torsion_type_split_list = torsion_type_str.split("-")
        torsion_type_i = torsion_type_split_list[0].strip()
        torsion_type_j = torsion_type_split_list[1].strip()
        torsion_type_k = torsion_type_split_list[2].strip()
        torsion_type_l = torsion_type_split_list[3].strip()
        torsion_type_tuple = (
            torsion_type_i,
            torsion_type_j,
            torsion_type_k,
            torsion_type_l,
        )

        torsion_parameter_str = line[14:54]
        torsion_parameter_split_list = torsion_parameter_str.strip().split()

        fps_dict = {}
        fps_dict["barrier_factor"] = int(torsion_parameter_split_list[0])
        fps_dict["barrier_height"] = float(torsion_parameter_split_list[1])
        fps_dict["periodicity"] = int(abs(float(torsion_parameter_split_list[3])))
        fps_dict["phase"] = float(torsion_parameter_split_list[2])

        if torsion_type_tuple in torsion_parameter_dict.keys():
            torsion_parameter_dict[torsion_type_tuple].append(fps_dict)
        else:
            torsion_parameter_dict[torsion_type_tuple] = [fps_dict]
    ##############################################################################

    return (
        atom_type_list,
        partial_charge_list,
        atom_parameter_dict,
        torsion_parameter_dict,
    )


def root_finding_strategy(fragment_mol_list, rotatable_bond_info_list):
    num_fragments = len(fragment_mol_list)
    num_torsions = len(rotatable_bond_info_list)
    edge_info_list = [None] * num_torsions

    for torsion_idx in range(num_torsions):
        rotatable_bond_info = rotatable_bond_info_list[torsion_idx]
        begin_atom_idx = rotatable_bond_info[0]
        end_atom_idx = rotatable_bond_info[1]
        begin_edge_fragment_idx = None
        end_edge_fragment_idx = None

        for fragment_idx in range(num_fragments):
            fragment_mol = fragment_mol_list[fragment_idx]
            for atom in fragment_mol.GetAtoms():
                if atom.GetIntProp("internal_atom_idx") == begin_atom_idx:
                    begin_edge_fragment_idx = fragment_idx
                    break

            if begin_edge_fragment_idx is not None:
                break

        for fragment_idx in range(num_fragments):
            fragment_mol = fragment_mol_list[fragment_idx]
            for atom in fragment_mol.GetAtoms():
                if atom.GetIntProp("internal_atom_idx") == end_atom_idx:
                    end_edge_fragment_idx = fragment_idx
                    break

            if end_edge_fragment_idx is not None:
                break

        edge_info_list[torsion_idx] = (begin_edge_fragment_idx, end_edge_fragment_idx)

    molecular_graph = nx.Graph()
    molecular_graph.add_nodes_from(list(range(num_fragments)))
    molecular_graph.add_edges_from(edge_info_list)

    fragment_level_list = [None] * num_fragments
    for fragment_idx in range(num_fragments):
        fragment_node_levels = nx.shortest_path_length(
            molecular_graph, fragment_idx
        ).values()
        max_fragment_node_level = np.max(list(fragment_node_levels))
        fragment_level_list[fragment_idx] = max_fragment_node_level

    overall_min_level = np.min(fragment_level_list)
    selected_fragment_idx_list = np.where(
        fragment_level_list <= (overall_min_level + 2)
    )[0].tolist()

    num_selected_fragments = len(selected_fragment_idx_list)
    selected_fragment_num_atoms_list = [None] * num_selected_fragments

    for fragment_idx in range(num_selected_fragments):
        selected_fragment_idx = selected_fragment_idx_list[fragment_idx]
        fragment_mol = fragment_mol_list[selected_fragment_idx]
        selected_fragment_num_atoms_list[fragment_idx] = fragment_mol.GetNumAtoms()

    selected_fragment_level_list = np.array(fragment_level_list)[
        selected_fragment_idx_list
    ].tolist()
    top_size_fragment_num_atoms = np.max(selected_fragment_num_atoms_list)
    top_size_selected_idx_array = np.where(
        selected_fragment_num_atoms_list == top_size_fragment_num_atoms
    )[0]
    min_level_top_size_selected_idx = top_size_selected_idx_array[
        np.argmin(np.array(selected_fragment_level_list)[top_size_selected_idx_array])
    ]
    min_level_top_size_selected_fragment_idx = np.array(selected_fragment_idx_list)[
        min_level_top_size_selected_idx
    ]

    return min_level_top_size_selected_fragment_idx


_TEMPLATE_MAPPING_NAMES = (
    'get_template_docking_atom_mapping',
    'get_core_alignment_for_template_docking',
    'check_manual_atom_mapping_connection',
    'get_full_ring_core_atoms',
)


_COVALENT_LIGAND_NAMES = (
    'prepare_covalent_ligand_mol',
)


def __getattr__(name):
    """Re-exports (one release) of the template mapping functions, moved to plugins.template.mapping, and of the
    covalent ligand preparation, moved to plugins.covalent.ligand."""
    if name in _TEMPLATE_MAPPING_NAMES:
        from unidock_processing.plugins.template import mapping
        return getattr(mapping, name)
    if name in _COVALENT_LIGAND_NAMES:
        from unidock_processing.plugins.covalent import ligand
        return getattr(ligand, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
