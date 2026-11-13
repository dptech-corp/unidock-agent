"""The covalent residue preparation of the receptor topology builder (``find_covalent_hydrogen_atoms`` and
``prepare_covalent_bond_on_residue``, moved verbatim from ``unidocktools/unidock_receptor_topology_builder.py``): a
``CovalentReceptorHook`` operates on the builder's own attributes (``covalent_residue_atom_info_list``,
``covalent_residue_idx``, ``covalent_residue_atom_idx_list``, ``protein_residue_property_mol_list``). The builder
calls it from ``analyze_receptor_topology`` when its covalent residue list is set, the one place the branch sat
before the move (the runner never calls that method, so in the runner path this code is unreachable, as before).
"""
from unidock_processing.utils.molecule_processing import get_mol_without_indices


class CovalentReceptorHook:
    def __init__(self, builder):
        self.builder = builder

    def find_covalent_hydrogen_atoms(self, atom):
        builder = self.builder
        for neighbor_atom in atom.GetNeighbors():
            neighbor_atom_idx = neighbor_atom.GetIdx()
            neighbor_atom_name = neighbor_atom.GetProp("atom_name")

            if neighbor_atom_name.startswith("H"):
                builder.covalent_residue_atom_idx_list.append(neighbor_atom_idx)

    def prepare_covalent_bond_on_residue(self):
        builder = self.builder
        covalent_anchor_atom_info = tuple(builder.covalent_residue_atom_info_list[0])
        covalent_bond_start_atom_info = tuple(builder.covalent_residue_atom_info_list[1])
        covalent_bond_end_atom_info = tuple(builder.covalent_residue_atom_info_list[2])

        num_protein_residues = len(builder.protein_residue_property_mol_list)
        for residue_idx in range(num_protein_residues):
            residue_mol = builder.protein_residue_property_mol_list[residue_idx]
            atom = residue_mol.GetAtomWithIdx(0)
            chain_idx = atom.GetProp("chain_idx")
            resname = atom.GetProp("residue_name")
            resid = atom.GetIntProp("residue_idx")
            atom_info = (chain_idx, resname, resid)

            if atom_info == covalent_anchor_atom_info[:3]:
                builder.covalent_residue_idx = residue_idx
                break

        if builder.covalent_residue_idx is None:
            raise ValueError("Cannot find covalent residues from user inputs!!")

        covalent_residue_mol = builder.protein_residue_property_mol_list[
            builder.covalent_residue_idx
        ]
        num_residue_atoms = covalent_residue_mol.GetNumAtoms()

        for atom_idx in range(num_residue_atoms):
            atom = covalent_residue_mol.GetAtomWithIdx(atom_idx)
            chain_idx = atom.GetProp("chain_idx")
            resname = atom.GetProp("residue_name")
            resid = atom.GetIntProp("residue_idx")
            atom_name = atom.GetProp("atom_name")
            atom_info = (chain_idx, resname, resid, atom_name)

            if atom_info == covalent_anchor_atom_info:
                builder.covalent_residue_atom_idx_list.append(atom_idx)
                self.find_covalent_hydrogen_atoms(atom)

            elif atom_info == covalent_bond_start_atom_info:
                builder.covalent_residue_atom_idx_list.append(atom_idx)
                self.find_covalent_hydrogen_atoms(atom)

            elif atom_info == covalent_bond_end_atom_info:
                builder.covalent_residue_atom_idx_list.append(atom_idx)
                self.find_covalent_hydrogen_atoms(atom)

        processed_covalent_residue_mol = get_mol_without_indices(
            covalent_residue_mol,
            remove_indices=builder.covalent_residue_atom_idx_list,
            keep_properties=[
                "atom_idx",
                "atom_name",
                "atom_charge",
                "ff_atom_type",
                "vina_atom_type",
                "residue_idx",
                "residue_name",
                "chain_idx",
                "internal_atom_idx",
                "internal_residue_idx",
                "x",
                "y",
                "z",
            ],
        )

        builder.protein_residue_property_mol_list[builder.covalent_residue_idx] = (
            processed_covalent_residue_mol
        )
