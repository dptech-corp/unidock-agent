"""The core atom mapping and template alignment of template docking (moved verbatim from
``ligand_topology/utils.py``; that module keeps re-exports of these names for one release).
"""
from copy import deepcopy
import re

from rdkit import Chem
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem import rdFMCS

from unidock_processing.ligand_topology.utils import recover_full_atom_mapping_from_heavy_atoms
from unidock_processing.utils.molecule_processing import get_mol_with_indices


def get_template_docking_atom_mapping(reference_mol, query_mol):
    reference_mol = deepcopy(reference_mol)
    query_mol = deepcopy(query_mol)

    num_reference_atoms = reference_mol.GetNumAtoms()
    for atom_idx in range(num_reference_atoms):
        atom = reference_mol.GetAtomWithIdx(atom_idx)
        atom.SetIntProp("internal_atom_idx", atom_idx)

    num_query_atoms = query_mol.GetNumAtoms()
    for atom_idx in range(num_query_atoms):
        atom = query_mol.GetAtomWithIdx(atom_idx)
        atom.SetIntProp("internal_atom_idx", atom_idx)

    reference_mol_no_H = Chem.RemoveHs(reference_mol)
    query_mol_no_H = Chem.RemoveHs(query_mol)

    mcs = rdFMCS.FindMCS(
        [query_mol_no_H, reference_mol_no_H],
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchChiralTag=False,
        timeout=100,
    )

    mcs_string = mcs.smartsString.replace("#0", "*")
    generic_core_smarts_string = re.sub(r"\[\*.*?\]", "[*]", mcs_string)

    generic_core_mol = Chem.MolFromSmarts(generic_core_smarts_string)

    reference_atom_mapping = reference_mol_no_H.GetSubstructMatches(generic_core_mol)[0]
    query_atom_mapping = query_mol_no_H.GetSubstructMatches(generic_core_mol)[0]

    core_atom_mapping_dict = {
        reference_atom_idx: query_atom_idx
        for reference_atom_idx, query_atom_idx in zip(
            reference_atom_mapping, query_atom_mapping
        )
    }

    full_core_atom_mapping_dict = recover_full_atom_mapping_from_heavy_atoms(
        reference_mol,
        reference_mol_no_H,
        query_mol,
        query_mol_no_H,
        core_atom_mapping_dict,
    )

    return full_core_atom_mapping_dict


def get_core_alignment_for_template_docking(
    reference_mol, query_mol, core_atom_mapping_dict
):
    core_atom_mapping_dict = {
        query_atom_idx: reference_atom_idx
        for reference_atom_idx, query_atom_idx in core_atom_mapping_dict.items()
    }
    # the initial position of query_mol is random, so align to the reference_mol firstly
    _ = AlignMol(query_mol, reference_mol, atomMap=list(core_atom_mapping_dict.items()))

    # assign template positions from reference mol to query mol
    core_fixed_query_conformer = Chem.Conformer(query_mol.GetNumAtoms())
    reference_conformer = reference_mol.GetConformer()
    query_conformer = query_mol.GetConformer()

    for query_atom_idx in range(query_mol.GetNumAtoms()):
        if query_atom_idx in core_atom_mapping_dict:
            reference_atom_idx = core_atom_mapping_dict[query_atom_idx]
            atom_position = reference_conformer.GetAtomPosition(reference_atom_idx)
            core_fixed_query_conformer.SetAtomPosition(query_atom_idx, atom_position)
        else:
            atom_position = query_conformer.GetAtomPosition(query_atom_idx)
            core_fixed_query_conformer.SetAtomPosition(query_atom_idx, atom_position)

    query_mol.RemoveAllConformers()
    query_mol.AddConformer(core_fixed_query_conformer)

    # optimize conformer using chemical forcefield
    ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(query_mol, "MMFF94s")
    ff = ChemicalForceFields.MMFFGetMoleculeForceField(query_mol, ff_property, confId=0)

    for query_atom_idx in core_atom_mapping_dict.keys():
        reference_atom_idx = core_atom_mapping_dict[query_atom_idx]
        core_atom_position = reference_conformer.GetAtomPosition(reference_atom_idx)
        virtual_site_atom_idx = (
            ff.AddExtraPoint(
                core_atom_position.x,
                core_atom_position.y,
                core_atom_position.z,
                fixed=True,
            )
            - 1
        )
        ff.AddDistanceConstraint(virtual_site_atom_idx, query_atom_idx, 0, 0, 100.0)

    ff.Initialize()

    max_minimize_iteration = 5
    for _ in range(max_minimize_iteration):
        minimize_seed = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        if minimize_seed == 0:
            break

    query_mol.SetProp("aligned_conformer_energy", str(ff.CalcEnergy()))

    core_atom_idx_list = list(core_atom_mapping_dict.keys())

    return core_atom_idx_list


def check_manual_atom_mapping_connection(
    reference_mol, query_mol, core_atom_mapping_dict
):
    reference_atom_idx_list = list(core_atom_mapping_dict.keys())
    query_atom_idx_list = list(core_atom_mapping_dict.values())

    reference_core_mol = get_mol_with_indices(
        reference_mol, selected_indices=reference_atom_idx_list
    )
    query_core_mol = get_mol_with_indices(
        query_mol, selected_indices=query_atom_idx_list
    )

    # Only verify the core is a single connected fragment. Do NOT sanitize/kekulize the
    # extracted core here: a loose MCS can legitimately include a PARTIAL aromatic ring
    # (e.g. 5/6 atoms of a benzene), which is a valid connected core that
    # get_full_ring_core_atoms() completes downstream. Sanitizing such a core raises
    # AtomKekulizeException ("non-ring atom marked aromatic") -- which is NOT a subclass
    # of KekulizeException, so the old try/except missed it -- and crashes docking.
    # GetMolFrags checks connectivity without requiring kekulization.
    if len(Chem.GetMolFrags(reference_core_mol)) != 1:
        return False
    if len(Chem.GetMolFrags(query_core_mol)) != 1:
        return False
    return True


def get_full_ring_core_atoms(mol, core_atom_idx_list):
    """Expand core atom set to handle partial ring mappings.

    - If only 1 ring atom is in core -> remove it (too weak a constraint)
    - If >1 but not all ring atoms in core -> add ALL ring atoms (complete the ring)
    """
    core_atom_idx_set = set(core_atom_idx_list)
    full_core_atom_idx_set = set(core_atom_idx_list)

    rings = Chem.GetSymmSSSR(mol)

    for ring in rings:
        ring_atoms = set(ring)
        intersection = ring_atoms.intersection(core_atom_idx_set)

        if len(intersection) == 1:
            full_core_atom_idx_set.difference_update(intersection)
        elif len(intersection) > 1 and len(intersection) < len(ring_atoms):
            full_core_atom_idx_set.update(ring_atoms)

    return list(full_core_atom_idx_set)

