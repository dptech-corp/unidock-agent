"""The covalent-ligand branches of the ligand topology builder as a ``LigandHook``: ``prepare_covalent_ligand_mol``
is moved verbatim from ``ligand_topology/utils.py`` (that module keeps a re-export for one release), and the two
builder branches (the rebuild before atom typing, the anchor-atom root fragment) are the per-molecule hook.
"""
from copy import deepcopy

from rdkit import Chem

from unidock_processing.plugins.base import LigandHook, LigandMolHook
from unidock_processing.utils.molecule_processing import get_mol_without_indices


def prepare_covalent_ligand_mol(mol):
    covalent_atom_idx_string = mol.GetProp("covalent_atom_indices")
    covalent_atom_idx_string_list = covalent_atom_idx_string.split(",")
    covalent_atom_idx_list = [
        int(covalent_atom_idx_string)
        for covalent_atom_idx_string in covalent_atom_idx_string_list
    ]

    covalent_atom_name_string = mol.GetProp("covalent_atom_names")
    covalent_atom_name_list = covalent_atom_name_string.split(",")

    covalent_residue_name_string = mol.GetProp("covalent_residue_names")
    covalent_residue_name_list = covalent_residue_name_string.split(",")

    covalent_residue_idx_string = mol.GetProp("covalent_residue_indices")
    covalent_residue_idx_string_list = covalent_residue_idx_string.split(",")
    covalent_residue_idx_list = [
        int(covalent_residue_idx_string)
        for covalent_residue_idx_string in covalent_residue_idx_string_list
    ]

    covalent_chain_idx_string = mol.GetProp("covalent_chain_indices")
    covalent_chain_idx_list = covalent_chain_idx_string.split(",")

    num_covalent_residue_atoms = len(covalent_atom_name_list)
    covalent_atom_info_list = [None] * num_covalent_residue_atoms

    for covalent_residue_atom_idx in range(num_covalent_residue_atoms):
        covalent_atom_info_tuple = (
            covalent_chain_idx_list[covalent_residue_atom_idx],
            covalent_residue_name_list[covalent_residue_atom_idx],
            covalent_residue_idx_list[covalent_residue_atom_idx],
            covalent_atom_name_list[covalent_residue_atom_idx],
        )

        covalent_atom_info_list[covalent_residue_atom_idx] = covalent_atom_info_tuple

    covalent_anchor_atom_info = (
        covalent_chain_idx_list[0],
        covalent_residue_name_list[0],
        covalent_residue_idx_list[0],
        covalent_atom_name_list[0],
    )

    removed_atom_idx_list = []

    num_atoms = mol.GetNumAtoms()
    internal_atom_idx = 0
    atom_coords_dict = {}
    conformer = mol.GetConformer()
    for atom_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 0:
            atom.SetProp("atom_name", "None")
            removed_atom_idx_list.append(atom_idx)
        else:
            if atom_idx in covalent_atom_idx_list:
                atom_name = covalent_atom_name_list[
                    covalent_atom_idx_list.index(atom_idx)
                ]
                residue_name = covalent_residue_name_list[
                    covalent_atom_idx_list.index(atom_idx)
                ]
                residue_idx = covalent_residue_idx_list[
                    covalent_atom_idx_list.index(atom_idx)
                ]
                chain_idx = covalent_chain_idx_list[
                    covalent_atom_idx_list.index(atom_idx)
                ]
                atom.SetProp("atom_name", atom_name)
                atom.SetProp("residue_name", residue_name)
                atom.SetIntProp("residue_idx", residue_idx)
                atom.SetProp("chain_idx", chain_idx)
                atom_info = (chain_idx, residue_name, residue_idx, atom_name)
            else:
                atom_element = atom.GetSymbol()
                atom_name = atom_element + str(internal_atom_idx + 1)
                atom.SetProp("atom_name", atom_name)
                atom.SetProp("residue_name", "MOL")
                atom.SetIntProp("residue_idx", 1)
                atom.SetProp("chain_idx", "A")
                atom_info = ("A", "MOL", 1, atom_name)
                internal_atom_idx += 1

            atom_coords_point_3D = deepcopy(conformer.GetAtomPosition(atom_idx))
            atom_coords_dict[atom_info] = atom_coords_point_3D

    covalent_mol = get_mol_without_indices(
        mol,
        remove_indices=removed_atom_idx_list,
        keep_properties=["atom_name", "residue_name", "residue_idx", "chain_idx"],
        keep_mol_properties=["_Name", "source_sdf_file_name", "source_mol_idx", "ud2_molecule_name"],
    )

    num_covalent_atoms = covalent_mol.GetNumAtoms()
    covalent_conformer = Chem.Conformer(num_covalent_atoms)

    for atom_idx in range(num_covalent_atoms):
        atom = covalent_mol.GetAtomWithIdx(atom_idx)
        atom_name = atom.GetProp("atom_name")
        residue_name = atom.GetProp("residue_name")
        residue_idx = atom.GetIntProp("residue_idx")
        chain_idx = atom.GetProp("chain_idx")
        atom_info = (chain_idx, residue_name, residue_idx, atom_name)
        atom_coords_point_3D = atom_coords_dict[atom_info]
        covalent_conformer.SetAtomPosition(atom_idx, atom_coords_point_3D)

    _ = covalent_mol.AddConformer(covalent_conformer)
    _ = Chem.SanitizeMol(covalent_mol)

    return covalent_mol, covalent_anchor_atom_info, covalent_atom_info_list


class CovalentMolHook(LigandMolHook):
    """The per-molecule covalent branches: the ligand rebuilt without its dummy atoms (the molecule the builder
    types) and the root fragment that holds the covalent anchor atom."""

    def __init__(self):
        self.covalent_anchor_atom_info = None

    def place_core(self, mol, working_dir_name):
        mol, covalent_anchor_atom_info, _ = (
            prepare_covalent_ligand_mol(mol)
        )
        self.covalent_anchor_atom_info = covalent_anchor_atom_info
        return mol

    def root_fragment_idx(self, splitted_mol_list):
        covalent_anchor_atom_info = self.covalent_anchor_atom_info
        num_fragments = len(splitted_mol_list)
        root_fragment_idx = None
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            for atom in fragment.GetAtoms():
                atom_info = (
                    atom.GetProp('chain_idx'),
                    atom.GetProp('residue_name'),
                    atom.GetIntProp('residue_idx'),
                    atom.GetProp('atom_name'),
                )
                if atom_info == covalent_anchor_atom_info:
                    root_fragment_idx = fragment_idx
                    break

            if root_fragment_idx is not None:
                break

        if root_fragment_idx is None:
            raise ValueError('Bugs in root finding code for covalent docking!')
        return root_fragment_idx


class CovalentLigandHook(LigandHook):
    """The run-level covalent hook: no inputs (the ligand SDF carries the residue annotation), a static root, the
    same object for every batch (picklable), one ``CovalentMolHook`` per ligand."""

    def wants_static_root(self):
        return True

    def for_ligand(self, ligand_idx):
        return CovalentMolHook()
