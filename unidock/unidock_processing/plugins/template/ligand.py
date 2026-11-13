"""The template-docking branches of the ligand topology builder as a ``LigandHook`` (the bodies are the ones
``unidocktools/unidock_ligand_topology_builder.py`` carried; the builder now calls them through its hooks).
"""
import os
import re
import logging

from rdkit import Chem

from unidock_processing.plugins.base import LigandHook, LigandMolHook
from unidock_processing.plugins.template import mapping

logger = logging.getLogger(__name__)

# atom_mapper_align: use AlignRotatableBond SMARTS (excludes only amide)
_ALIGN_ROT_SMARTS = (
    "[!$(*#*)&!D1&!$([CH3])&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$"
    "(C([CH3])([CH3])[CH3])]-!@[!$(*#*)&!D1&!$([CH3])&!$(C(F)(F)F)&!$"
    "(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
)
_ALIGN_AMIDE_SMARTS = "[C&$(C=O)]-[N&$(NC=O);v3;H1,H2;0]"


def atom_mapper_rotatable_bonds(mol):
    """The rotatable bonds of ``atom_mapper_align`` mode: the AlignRotatableBond SMARTS (excludes only amide)."""
    _align_rot_pattern = Chem.MolFromSmarts(_ALIGN_ROT_SMARTS)
    _align_amide_pattern = Chem.MolFromSmarts(_ALIGN_AMIDE_SMARTS)

    rotatable_bond_info_list = list(mol.GetSubstructMatches(_align_rot_pattern))
    for excl in list(mol.GetSubstructMatches(_align_amide_pattern)):
        excl_rev = tuple(reversed(excl))
        if excl in rotatable_bond_info_list:
            rotatable_bond_info_list.remove(excl)
        elif excl_rev in rotatable_bond_info_list:
            rotatable_bond_info_list.remove(excl_rev)
    return rotatable_bond_info_list


class TemplateMolHook(LigandMolHook):
    """The per-molecule template branches: core placement, the atom_mapper rotatable-bond finder, the frozen
    core bonds and the core root fragment."""

    def __init__(self, reference_mol, core_atom_mapping_dict=None, atom_mapper_align=False):
        self.reference_mol = reference_mol
        self.core_atom_mapping_dict = core_atom_mapping_dict
        self.atom_mapper_align = atom_mapper_align
        self.core_atom_idx_list = None

    def place_core(self, mol, working_dir_name):
        reference_mol = self.reference_mol
        core_atom_mapping_dict = self.core_atom_mapping_dict
        if core_atom_mapping_dict is None:
            core_atom_mapping_dict = mapping.get_template_docking_atom_mapping(
                reference_mol, mol
            )
        else:
            if not mapping.check_manual_atom_mapping_connection(
                reference_mol, mol, core_atom_mapping_dict
            ):
                raise ValueError(
                    'Specified core atom mapping makes unconnected fragments!!'
                )

        if self.atom_mapper_align:
            # atom_mapper_align mode: skip re-alignment, use smart ring expansion
            core_atom_idx_list = mapping.get_full_ring_core_atoms(
                mol, list(core_atom_mapping_dict.values())
            )
        else:
            core_atom_idx_list = mapping.get_core_alignment_for_template_docking(
                reference_mol, mol, core_atom_mapping_dict
            )

            temp_template_aligned_ligand_sdf_file_name = os.path.join(
                working_dir_name, 'ligand_template_aligned.sdf'
            )
            writer = Chem.SDWriter(temp_template_aligned_ligand_sdf_file_name)
            writer.write(mol)
            writer.close()

        self.core_atom_mapping_dict = core_atom_mapping_dict
        self.core_atom_idx_list = core_atom_idx_list
        return mol

    def rotatable_bonds(self, mol):
        if not self.atom_mapper_align:
            return None
        return atom_mapper_rotatable_bonds(mol)

    def filter_rotatable_bonds(self, mol, rotatable_bond_info_list):
        core_atom_idx_list = self.core_atom_idx_list
        atom_mapper_align = self.atom_mapper_align
        filtered_rotatable_bond_info_list = []
        for rotatable_bond_info in rotatable_bond_info_list:
            rotatable_begin_atom_idx = rotatable_bond_info[0]
            rotatable_end_atom_idx = rotatable_bond_info[1]
            if (
                rotatable_begin_atom_idx in core_atom_idx_list
                and rotatable_end_atom_idx in core_atom_idx_list
            ):
                if atom_mapper_align:
                    # Only freeze if ALL 4 torsion atoms are mapped
                    begin_neighbors = [
                        nbr.GetIdx() for nbr in mol.GetAtomWithIdx(rotatable_begin_atom_idx).GetNeighbors()
                        if nbr.GetIdx() != rotatable_end_atom_idx and nbr.GetAtomicNum() > 1
                    ]
                    end_neighbors = [
                        nbr.GetIdx() for nbr in mol.GetAtomWithIdx(rotatable_end_atom_idx).GetNeighbors()
                        if nbr.GetIdx() != rotatable_begin_atom_idx and nbr.GetAtomicNum() > 1
                    ]
                    if not begin_neighbors or not end_neighbors:
                        continue
                    has_fully_mapped_torsion = False
                    for begin_neighbor in begin_neighbors:
                        for end_neighbor in end_neighbors:
                            torsion_atoms = {
                                begin_neighbor, rotatable_begin_atom_idx,
                                rotatable_end_atom_idx, end_neighbor
                            }
                            if len(torsion_atoms.intersection(core_atom_idx_list)) == 4:
                                has_fully_mapped_torsion = True
                                break
                        if has_fully_mapped_torsion:
                            break
                    if not has_fully_mapped_torsion:
                        filtered_rotatable_bond_info_list.append(rotatable_bond_info)
                else:
                    continue
            else:
                filtered_rotatable_bond_info_list.append(rotatable_bond_info)

        return filtered_rotatable_bond_info_list

    def root_fragment_idx(self, splitted_mol_list):
        core_atom_idx_list = self.core_atom_idx_list
        num_fragments = len(splitted_mol_list)
        root_fragment_idx = None
        for fragment_idx in range(num_fragments):
            fragment = splitted_mol_list[fragment_idx]
            for atom in fragment.GetAtoms():
                internal_atom_idx = (
                    int(re.split(r"(\d+)", atom.GetProp('atom_name'))[1]) - 1
                )
                if internal_atom_idx in core_atom_idx_list:
                    if self.atom_mapper_align:
                        # Root atom must have a core neighbor
                        nbr_heavy = [
                            nbr for nbr in atom.GetNeighbors()
                            if nbr.GetAtomicNum() > 1
                        ]
                        if nbr_heavy:
                            nbr_heavy_idx_list = [
                                int(re.split(r"(\d+)", nbr.GetProp("atom_name"))[1]) - 1
                                for nbr in nbr_heavy
                            ]
                            if any(idx in core_atom_idx_list for idx in nbr_heavy_idx_list):
                                root_fragment_idx = fragment_idx
                                break
                    else:
                        root_fragment_idx = fragment_idx
                        break

            if root_fragment_idx is not None:
                break

        if root_fragment_idx is None:
            raise ValueError('Bugs in root finding code for template docking!')
        return root_fragment_idx


class AtomMapperAlignMolHook(LigandMolHook):
    """``atom_mapper_align`` without template docking: only the rotatable-bond finder changes (no core placement,
    no frozen core bonds, the builder's own root), as the builder's ``if atom_mapper_align:`` branch did on its own
    before the plugin."""

    def rotatable_bonds(self, mol):
        return atom_mapper_rotatable_bonds(mol)


class AtomMapperAlignHook(LigandHook):
    """The run-level hook of ``atom_mapper_align`` without template docking: no inputs, no static root, one
    ``AtomMapperAlignMolHook`` per ligand."""

    def for_ligand(self, ligand_idx):
        return AtomMapperAlignMolHook()


class TemplateLigandHook(LigandHook):
    """The run-level template hook: holds the reference SDF path, one core atom map per ligand (None = automatic
    MCS) and the atom_mapper_align mode; ``for_batch`` slices the maps for a batch worker, which loads the
    reference once (``open``) and binds one ``TemplateMolHook`` per ligand."""

    def __init__(self, reference_sdf_file_name, core_atom_mapping_dict_list=None, atom_mapper_align=False):
        if reference_sdf_file_name is not None:
            self.reference_sdf_file_name = os.path.abspath(reference_sdf_file_name)
        else:
            self.reference_sdf_file_name = None
        self.core_atom_mapping_dict_list = core_atom_mapping_dict_list
        self.atom_mapper_align = atom_mapper_align
        self.reference_mol = None

    def bind(self, num_ligands):
        if self.core_atom_mapping_dict_list is None:
            self.core_atom_mapping_dict_list = [None] * num_ligands
        if len(self.core_atom_mapping_dict_list) != num_ligands:
            raise ValueError(
                'Number of user specified core atom mapping dicts does not match \
                    the number of input molecules!!'
            )
        if self.reference_sdf_file_name is None:
            raise ValueError(
                'template docking mode specified without reference SDF file!!'
            )

    def wants_static_root(self):
        return True

    def for_batch(self, begin, end):
        return TemplateLigandHook(
            self.reference_sdf_file_name,
            self.core_atom_mapping_dict_list[begin:end],
            self.atom_mapper_align,
        )

    def open(self):
        self.reference_mol = Chem.SDMolSupplier(self.reference_sdf_file_name, removeHs=False)[0]

    def for_ligand(self, ligand_idx):
        return TemplateMolHook(
            self.reference_mol,
            self.core_atom_mapping_dict_list[ligand_idx],
            self.atom_mapper_align,
        )
