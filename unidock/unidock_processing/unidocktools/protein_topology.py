import msys

from rdkit import Chem
from rdkit.Chem import SplitMolByPDBResidues, GetMolFrags, FragmentOnBonds
from rdkit.Chem.PropertyMol import PropertyMol

from unidock_processing.utils.molecule_processing import (
    get_mol_with_indices,
    get_mol_without_indices,
)
from unidock_processing.unidocktools.vina_atom_type import AtomType
from unidock_processing.unidocktools.supported_protein_residue_name import (
    PROTEIN_RESIUDE_NAME_LIST,
)


def is_peptide_bond(bond):
    """Checks if a bond is a peptide bond based on the residue_id and chain_id
    of the atoms on each part of the bond. Also works for disulfide bridges or any bond
    that links two residues in biopolymers.

    Parameters
    ----------
    bond: rdkit.Chem.rdchem.Bond
        The bond to check
    """

    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    begin_residue_idx = begin_atom.GetIntProp("internal_residue_idx")
    end_residue_idx = end_atom.GetIntProp("internal_residue_idx")

    begin_chain_idx = begin_atom.GetProp("chain_idx")
    end_chain_idx = end_atom.GetProp("chain_idx")

    if begin_residue_idx == end_residue_idx and begin_chain_idx == end_chain_idx:
        return False
    else:
        return True


def get_linked_residue_idx_set(mol):
    """The ``internal_residue_idx`` of the residues of *mol* bonded to another residue of *mol*
    (nucleotides of a nucleic-acid chain, sugars of a glycan)."""
    linked_residue_idx_set = set()
    for bond in mol.GetBonds():
        begin_residue_idx = bond.GetBeginAtom().GetIntProp("internal_residue_idx")
        end_residue_idx = bond.GetEndAtom().GetIntProp("internal_residue_idx")
        if begin_residue_idx != end_residue_idx:
            linked_residue_idx_set.update((begin_residue_idx, end_residue_idx))
    return linked_residue_idx_set


def split_mol_by_residues(protein_mol):
    """Splits a protein_mol in multiple fragments based on residues.

    Uses the ``internal_residue_idx`` atom property to group atoms and
    builds each residue mol directly.  This avoids
    ``SplitMolByPDBResidues`` (which segfaults when ``MonomerInfo`` is
    absent from molecules produced by ``msys.ConvertToRdkit``) and avoids
    the O(N*R) cost of calling ``get_mol_with_indices`` per residue.

    Parameters
    ----------
    protein_mol: rdkit.Chem.Mol
        The protein molecule to fragment

    Returns
    -------
    residue_mol_list : list
        A list of :class:`rdkit.Chem.Mol` containing sorted residues
    """
    if protein_mol.GetNumAtoms() == 0:
        return []

    # Group atom indices by internal_residue_idx
    residue_groups = {}
    for atom_idx in range(protein_mol.GetNumAtoms()):
        atom = protein_mol.GetAtomWithIdx(atom_idx)
        res_idx = atom.GetIntProp("internal_residue_idx")
        if res_idx not in residue_groups:
            residue_groups[res_idx] = []
        residue_groups[res_idx].append(atom_idx)

    # Collect property names to copy
    keep_properties = [
        "atom_idx", "atom_name", "atom_charge", "ff_atom_type",
        "residue_idx", "residue_name", "chain_idx",
        "internal_atom_idx", "internal_residue_idx", "x", "y", "z",
    ]
    test_props = protein_mol.GetAtomWithIdx(0).GetPropsAsDict()
    if "vina_atom_type" in test_props:
        keep_properties.append("vina_atom_type")

    # Build a set for each residue's atom indices for O(1) bond lookup
    residue_idx_to_set = {
        res_idx: set(indices) for res_idx, indices in residue_groups.items()
    }

    # Build all residue mols in one pass
    protein_residue_mol_list = []
    for res_idx in sorted(residue_groups.keys()):
        atom_indices = residue_groups[res_idx]
        idx_set = residue_idx_to_set[res_idx]

        # Map old atom index -> new atom index within this residue
        old_to_new = {}
        rwmol = Chem.RWMol(Chem.Mol())

        has_heavy = False
        for new_idx, old_idx in enumerate(atom_indices):
            src_atom = protein_mol.GetAtomWithIdx(old_idx)
            new_atom = Chem.Atom(src_atom.GetAtomicNum())
            new_atom.SetFormalCharge(src_atom.GetFormalCharge())
            new_atom.SetChiralTag(src_atom.GetChiralTag())
            new_atom.SetNumExplicitHs(src_atom.GetNumExplicitHs())

            # Copy properties
            src_props = src_atom.GetPropsAsDict()
            for prop_name in keep_properties:
                if prop_name in src_props:
                    val = src_props[prop_name]
                    if isinstance(val, str):
                        new_atom.SetProp(prop_name, val)
                    elif isinstance(val, int):
                        new_atom.SetIntProp(prop_name, val)
                    elif isinstance(val, float):
                        new_atom.SetDoubleProp(prop_name, val)

            rwmol.AddAtom(new_atom)
            old_to_new[old_idx] = new_idx

            if src_atom.GetAtomicNum() > 1:
                has_heavy = True

        if not has_heavy:
            continue

        # Add intra-residue bonds only
        for old_idx in atom_indices:
            src_atom = protein_mol.GetAtomWithIdx(old_idx)
            for bond in src_atom.GetBonds():
                other_idx = bond.GetOtherAtomIdx(old_idx)
                if other_idx in idx_set and other_idx > old_idx:
                    rwmol.AddBond(
                        old_to_new[old_idx],
                        old_to_new[other_idx],
                        bond.GetBondType(),
                    )

        mol = Chem.Mol(rwmol)
        Chem.GetSymmSSSR(mol)
        mol.UpdatePropertyCache(strict=False)
        protein_residue_mol_list.append(mol)

    return protein_residue_mol_list


def prepare_receptor_residue_mol_list(receptor_msys_system):
    num_receptor_atoms = receptor_msys_system.natoms
    receptor_nb_table = receptor_msys_system.getTable("nonbonded")

    if num_receptor_atoms != receptor_nb_table.nterms:
        raise ValueError("Problematic receptor preparation!!")

    receptor_atom_idx_list = [None] * num_receptor_atoms
    receptor_atom_name_list = [None] * num_receptor_atoms
    receptor_atom_charge_list = [None] * num_receptor_atoms
    receptor_resid_list = [None] * num_receptor_atoms
    receptor_resname_list = [None] * num_receptor_atoms
    receptor_chain_idx_list = [None] * num_receptor_atoms
    receptor_ff_atom_type_list = [None] * num_receptor_atoms
    receptor_internal_atom_idx_list = [None] * num_receptor_atoms
    receptor_internal_residue_idx_list = [None] * num_receptor_atoms

    for atom_idx in range(num_receptor_atoms):
        atom = receptor_msys_system.atom(atom_idx)
        receptor_atom_idx_list[atom_idx] = atom_idx + 1
        receptor_atom_name_list[atom_idx] = atom.name
        receptor_atom_charge_list[atom_idx] = atom.charge
        receptor_resid_list[atom_idx] = atom.residue.resid
        receptor_resname_list[atom_idx] = atom.residue.name
        receptor_chain_idx_list[atom_idx] = atom.residue.chain.name
        receptor_internal_atom_idx_list[atom_idx] = atom_idx
        receptor_internal_residue_idx_list[atom_idx] = atom.residue.id

        nb_term = receptor_nb_table.term(atom_idx)
        atom_type = nb_term["type"]
        receptor_ff_atom_type_list[atom_idx] = atom_type

    receptor_mol = msys.ConvertToRdkit(receptor_msys_system)

    receptor_positions = receptor_mol.GetConformer().GetPositions()
    num_receptor_mol_atoms = receptor_mol.GetNumAtoms()

    if num_receptor_atoms != num_receptor_mol_atoms:
        raise ValueError("Problematic msys receptor system conversion to rdkit mol!!")

    for atom_idx in range(num_receptor_atoms):
        atom = receptor_mol.GetAtomWithIdx(atom_idx)
        atom_positions = receptor_positions[atom_idx, :]

        atom.SetIntProp("atom_idx", int(receptor_atom_idx_list[atom_idx]))
        atom.SetProp("atom_name", receptor_atom_name_list[atom_idx])
        atom.SetDoubleProp("atom_charge", receptor_atom_charge_list[atom_idx])
        atom.SetProp("ff_atom_type", receptor_ff_atom_type_list[atom_idx])
        atom.SetIntProp("residue_idx", int(receptor_resid_list[atom_idx]))
        atom.SetProp("residue_name", receptor_resname_list[atom_idx])
        atom.SetProp("chain_idx", receptor_chain_idx_list[atom_idx])
        atom.SetIntProp("internal_atom_idx", receptor_internal_atom_idx_list[atom_idx])
        atom.SetIntProp(
            "internal_residue_idx", receptor_internal_residue_idx_list[atom_idx]
        )
        atom.SetDoubleProp("x", float(atom_positions[0]))
        atom.SetDoubleProp("y", float(atom_positions[1]))
        atom.SetDoubleProp("z", float(atom_positions[2]))

    non_protein_atom_idx_list = []
    for atom_idx in range(num_receptor_atoms):
        atom = receptor_mol.GetAtomWithIdx(atom_idx)
        receptor_resname = atom.GetProp("residue_name")
        if receptor_resname not in PROTEIN_RESIUDE_NAME_LIST:
            non_protein_atom_idx_list.append(atom_idx)

    protein_mol = get_mol_without_indices(
        receptor_mol,
        remove_indices=non_protein_atom_idx_list,
        keep_properties=[
            "atom_idx",
            "atom_name",
            "atom_charge",
            "ff_atom_type",
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

    cofactor_mol = get_mol_with_indices(
        receptor_mol,
        selected_indices=non_protein_atom_idx_list,
        keep_properties=[
            "atom_idx",
            "atom_name",
            "atom_charge",
            "ff_atom_type",
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

    atom_typer = AtomType()
    atom_typer.assign_atom_types(protein_mol)

    protein_residue_mol_list = split_mol_by_residues(protein_mol)

    protein_property_mol = PropertyMol(protein_mol)
    protein_residue_property_mol_list = [
        PropertyMol(protein_residue_mol)
        for protein_residue_mol in protein_residue_mol_list
    ]

    cofactor_residue_group_dict = {}
    num_cofactor_atoms = cofactor_mol.GetNumAtoms()
    for atom_idx in range(num_cofactor_atoms):
        atom = cofactor_mol.GetAtomWithIdx(atom_idx)
        internal_residue_idx = atom.GetIntProp("internal_residue_idx")
        if internal_residue_idx not in cofactor_residue_group_dict:
            cofactor_residue_group_dict[internal_residue_idx] = [atom_idx]
        else:
            cofactor_residue_group_dict[internal_residue_idx].append(atom_idx)

    cofactor_internal_residue_idx_list = list(cofactor_residue_group_dict.keys())
    num_cofactor_residues = len(cofactor_internal_residue_idx_list)
    cofactor_keep_properties = [
        "atom_idx",
        "atom_name",
        "atom_charge",
        "ff_atom_type",
        "residue_idx",
        "residue_name",
        "chain_idx",
        "internal_atom_idx",
        "internal_residue_idx",
        "x",
        "y",
        "z",
    ]

    # Residues bonded to each other (nucleic-acid chains, glycans) are typed in place and split in one pass: cut out
    # one at a time, the atom at a cut (the O3' of a phosphodiester) would be typed as an OH, and each cut walks the
    # whole mol. The other residues are typed one at a time as before, cut from a mol of their atoms only.
    linked_residue_idx_set = get_linked_residue_idx_set(cofactor_mol)
    linked_residue_mol_dict = {}
    single_residue_mol = cofactor_mol
    single_atom_idx_map = None
    if linked_residue_idx_set:
        atom_typer.assign_atom_types(cofactor_mol)
        for residue_mol in split_mol_by_residues(cofactor_mol):
            residue_idx = residue_mol.GetAtomWithIdx(0).GetIntProp("internal_residue_idx")
            if residue_idx in linked_residue_idx_set:
                linked_residue_mol_dict[residue_idx] = residue_mol

        single_atom_idx_list = sorted(
            atom_idx
            for residue_idx, atom_idx_list in cofactor_residue_group_dict.items()
            if residue_idx not in linked_residue_mol_dict
            for atom_idx in atom_idx_list
        )
        single_atom_idx_map = {atom_idx: new_idx for new_idx, atom_idx in enumerate(single_atom_idx_list)}
        single_residue_mol = get_mol_with_indices(
            cofactor_mol,
            selected_indices=set(single_atom_idx_list),
            keep_properties=cofactor_keep_properties,
        )

    cofactor_residue_property_mol_list = [None] * num_cofactor_residues
    for cofactor_idx in range(num_cofactor_residues):
        cofactor_internal_residue_idx = cofactor_internal_residue_idx_list[cofactor_idx]
        if cofactor_internal_residue_idx in linked_residue_mol_dict:
            cofactor_residue_property_mol_list[cofactor_idx] = PropertyMol(
                linked_residue_mol_dict[cofactor_internal_residue_idx]
            )
            continue

        cofactor_atom_idx_list = cofactor_residue_group_dict[
            cofactor_internal_residue_idx
        ]
        if single_atom_idx_map is not None:
            cofactor_atom_idx_list = [single_atom_idx_map[atom_idx] for atom_idx in cofactor_atom_idx_list]
        cofactor_residue_mol = get_mol_with_indices(
            single_residue_mol,
            selected_indices=cofactor_atom_idx_list,
            keep_properties=cofactor_keep_properties,
        )

        atom_typer.assign_atom_types(cofactor_residue_mol)
        cofactor_residue_property_mol_list[cofactor_idx] = PropertyMol(
            cofactor_residue_mol
        )

    return (
        protein_property_mol,
        protein_residue_property_mol_list,
        cofactor_residue_property_mol_list,
    )
