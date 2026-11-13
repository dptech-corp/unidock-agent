"""Force-field parameters for a fepfixer receptor from tleap, for hosts without utop.

utop adds the force-field tables to fepfixer's DMS in place, so the chains, residue numbers and names that the
het recovery looks residues up by survive. tleap builds a new topology with its own atom order and without
chain or residue numbers. So tleap runs on a PDB written from the fepfixer DMS, and the tables it builds are
kept on a copy that takes names, coordinates, formal charges, bond orders and residue identity from the DMS.
"""
import logging
import os
import subprocess

import msys

logger = logging.getLogger(__name__)

TLEAP_TEMPLATE = os.path.join(os.path.dirname(__file__), "data", "tleap_receptor_template.in")
LOAD_LINE = "receptor = loadPdb receptor_final.pdb\n"
MAX_RESIDUES = 9999  # the PDB residue number is four digits


def _coordinate_key(xyz):
    return tuple(int(round(float(f"{value:.3f}") * 1000)) for value in xyz)


def write_tleap_pdb(system, pdb_file_name):
    """Number the residues 1..N in DMS order and put a TER between consecutive residues that share no bond, so
    LEaP sees every chain break as a terminus. Returns {residue id: LEaP residue index}."""
    linked = set()
    for bond in system.bonds:
        first, second = bond.first.residue.id, bond.second.residue.id
        if first != second:
            linked.update(((first, second), (second, first)))

    index, lines, serial, previous = {}, [], 0, None
    for residue in system.residues:
        if previous is not None and (previous.id, residue.id) not in linked:
            lines.append("TER")
        index[residue.id] = len(index) + 1
        for atom in residue.atoms:
            serial += 1
            x, y, z = atom.pos
            name = atom.name if len(atom.name) == 4 else f" {atom.name:<3}"
            lines.append(f"ATOM  {serial % 100000:5d} {name} {residue.name:>3}  {index[residue.id]:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00")
        previous = residue
    lines += ["TER", "END"]
    with open(pdb_file_name, "w") as pdb_file:
        pdb_file.write("\n".join(lines) + "\n")
    return index


def parameterize_receptor_with_tleap(structure_dms_file_name, working_dir_name, parameterized_dms_file_name):
    """Write parameterized_dms_file_name: the fepfixer DMS with the tleap force-field tables. Raises if tleap
    changed the atoms or bonds, since the parameters would then not belong to the receptor."""
    structure = msys.LoadDMS(structure_dms_file_name)
    if structure.nresidues > MAX_RESIDUES:
        raise RuntimeError(f"{structure.nresidues} residues do not fit the PDB residue number that tleap reads "
                           f"(at most {MAX_RESIDUES})")

    residue_index = write_tleap_pdb(structure, os.path.join(working_dir_name, "receptor_final.pdb"))
    disulfides = [f"bond receptor.{residue_index[bond.first.residue.id]}.SG "
                  f"receptor.{residue_index[bond.second.residue.id]}.SG"
                  for bond in structure.bonds if bond.first.atomic_number == 16 and bond.second.atomic_number == 16]
    with open(TLEAP_TEMPLATE) as template_file:
        template = template_file.read()
    with open(os.path.join(working_dir_name, "tleap.in"), "w") as tleap_input:
        tleap_input.write(template.replace(LOAD_LINE, LOAD_LINE + "".join(f"{line}\n" for line in disulfides)))
    with open(os.path.join(working_dir_name, "tleap.log"), "w") as tleap_log:
        subprocess.run(["tleap", "-f", "tleap.in"], cwd=working_dir_name, stdout=tleap_log,
                       stderr=subprocess.STDOUT, check=False)

    prmtop = os.path.join(working_dir_name, "receptor.prmtop")
    inpcrd = os.path.join(working_dir_name, "receptor.inpcrd")
    if not (os.path.exists(prmtop) and os.path.exists(inpcrd)):
        raise RuntimeError(f"tleap wrote no topology, see {os.path.join(working_dir_name, 'tleap.log')}")
    parameterized = msys.LoadPrmTop(prmtop)
    msys.ReadCrdCoordinates(parameterized, inpcrd)

    structure_atoms = list(structure.atoms)
    parameterized_atoms = list(parameterized.atoms)
    if len(structure_atoms) != len(parameterized_atoms):
        raise RuntimeError(f"tleap changed the atom count from {len(structure_atoms)} to {len(parameterized_atoms)}, "
                           f"see {os.path.join(working_dir_name, 'tleap.log')}")

    structure_xyz = structure.getPositions()
    atom_by_key = {_coordinate_key(xyz): atom for atom, xyz in zip(structure_atoms, structure_xyz)}
    if len(atom_by_key) != len(structure_atoms):
        raise RuntimeError("two receptor atoms share the same coordinates to 0.001 A")
    source_of, parameterized_atom_of = {}, {}
    for atom, xyz in zip(parameterized_atoms, parameterized.getPositions()):
        source = atom_by_key.get(tuple(int(round(float(value) * 1000)) for value in xyz))
        if source is None or source.atomic_number != atom.atomic_number:
            raise RuntimeError(f"tleap atom {atom.id} matches no fepfixer atom")
        source_of[atom.id] = source
        parameterized_atom_of[source.id] = atom
    if len(parameterized_atom_of) != len(parameterized_atoms):
        raise RuntimeError("two tleap atoms match the same fepfixer atom")

    for atom in parameterized_atoms:
        source = source_of[atom.id]
        atom.pos = source.pos
        atom.name = source.name
        atom.formal_charge = source.formal_charge

    for bond in structure.bonds:
        match = parameterized.findBond(parameterized_atom_of[bond.first.id], parameterized_atom_of[bond.second.id])
        if match is None:
            raise RuntimeError("a fepfixer bond is missing from the tleap topology, see "
                               f"{os.path.join(working_dir_name, 'tleap.log')}")
        match.order = bond.order
    if parameterized.nbonds != structure.nbonds:
        raise RuntimeError(f"tleap bonded atoms that fepfixer left apart: {parameterized.nbonds} bonds against "
                           f"{structure.nbonds}")

    chains = {}
    for residue in parameterized.residues:
        origins = {source_of[atom.id].residue.id for atom in residue.atoms}
        if len(origins) != 1:
            raise RuntimeError(f"tleap residue {residue.name} {residue.resid} mixes atoms of several fepfixer residues")
        origin = structure.residue(origins.pop())
        residue.name, residue.resid, residue.insertion = origin.name, origin.resid, origin.insertion
        if origin.chain.name not in chains:
            chains[origin.chain.name] = parameterized.addChain()
            chains[origin.chain.name].name = origin.chain.name
        residue.chain = chains[origin.chain.name]

    msys.SaveDMS(parameterized, parameterized_dms_file_name)
    logger.info(f"Receptor force-field parameters from tleap: {len(parameterized_atoms)} atoms, "
                f"{len(disulfides)} disulfide bonds")
