"""Ligand and receptor DMS preparation: the ud2_* enrichment tables, the minimal and GAFF2 ligand DMS
builders, the multi-ligand merge and the lightweight (no GAFF2, no docking) preparation entry point.

Moved verbatim from unidocktools/unidock_protocol_runner.py; that module re-exports the old underscore names.
"""
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
import os
import logging
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

from rdkit import Chem

from unidock_processing.unidocktools.unidock_receptor_topology_builder import (
    UnidockReceptorTopologyBuilder,
)
from unidock_processing.ligand_topology import utils

if TYPE_CHECKING:
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        UnidockLigandTopologyBuilder,
    )


def dms_has_ff_params(dms_path: str) -> bool:
    """Check if a DMS file contains force field parameters.

    Returns True if the tables nonbonded_param, pair_12_6_es_term,
    dihedral_trig_term, and exclusion all exist and have at least 1 row.
    """
    required_tables = [
        'nonbonded_param',
        'pair_12_6_es_term',
        'dihedral_trig_term',
        'exclusion',
    ]
    try:
        conn = sqlite3.connect(dms_path)
        cur = conn.cursor()
        for table in required_tables:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if cur.fetchone() is None:
                conn.close()
                return False
            cur.execute(f"SELECT COUNT(*) FROM [{table}]")
            count = cur.fetchone()[0]
            if count == 0:
                conn.close()
                return False
        conn.close()
        return True
    except Exception:
        return False


def dms_has_ud2_atom_props(dms_path: str) -> bool:
    """Check if every particle of a DMS file already has a Vina atom type (pre-enriched).

    The engine joins particle with ud2_atom_props on atom id, so a partial table would
    silently drop the untyped atoms; it counts as not enriched.
    """
    try:
        conn = sqlite3.connect(dms_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ud2_atom_props'"
        )
        if cur.fetchone() is None:
            conn.close()
            return False
        num_particles = cur.execute("SELECT COUNT(*) FROM particle").fetchone()[0]
        num_typed = cur.execute(
            "SELECT COUNT(*) FROM particle p JOIN ud2_atom_props ua ON p.id = ua.atom_id"
        ).fetchone()[0]
        conn.close()
        return num_particles > 0 and num_typed == num_particles
    except Exception:
        return False


def dms_has_nonbonded_params(dms_path: str) -> bool:
    """Check if a DMS file has the nonbonded_param table that the full receptor analysis reads."""
    try:
        conn = sqlite3.connect(dms_path)
        found = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='nonbonded_param'"
        ).fetchone() is not None
        conn.close()
        return found
    except Exception:
        return False


def read_gaff2_types_from_dms(dms_path: str) -> Tuple[List[str], List[float]]:
    """Read GAFF2 atom type names and partial charges from a DMS file.

    Returns (atom_type_list, partial_charge_list) ordered by particle.id.
    Uses msys to read the nonbonded TermTable (sigma, epsilon, type).
    """
    import msys as _msys
    mol = _msys.Load(dms_path)
    nb = mol.table('nonbonded')
    # Build atom_id -> GAFF2 type mapping from nonbonded terms
    atom_type_map: Dict[int, str] = {}
    for term in nb.terms:
        atom_type_map[term.atoms[0].id] = term['type']
    atom_type_list = []
    partial_charge_list = []
    for atom in mol.atoms:
        atom_type_list.append(atom_type_map.get(atom.id, ''))
        partial_charge_list.append(float(atom.charge))
    return atom_type_list, partial_charge_list


def write_sdf_tables_from_mol(cur, dms_path: str, rdkit_mol) -> None:
    """Write ud2_sdf_properties, ud2_formal_charges, ud2_bond_stereo tables.

    These tables store the information needed for the C++ engine to write
    output SDF files directly, without going through the Python PoseWriter.

    The atom/bond index mapping between RDKit mol and DMS uses coordinate
    matching (robust against atom reordering by antechamber).
    """
    from copy import deepcopy
    from rdkit import Chem

    num_atoms = rdkit_mol.GetNumAtoms()

    # -- DMS-to-RDKit atom index mapping: identity when the DMS was built from this mol
    # (same order, same elements), else nearest coordinate of the same element (antechamber
    # reorders atoms). Exact rounded-coordinate matching is not enough: an SDF written with 4
    # decimals from a 6-decimal mol lands on the other side of a rounding boundary for some atoms,
    # the mapping fails silently, and charged ligands then come out of the engine unparsable.
    conf = rdkit_mol.GetConformer()
    rdkit_xyz = conf.GetPositions()
    rdkit_z = [a.GetAtomicNum() for a in rdkit_mol.GetAtoms()]
    dms_rows = list(cur.execute("SELECT id, anum, x, y, z FROM particle ORDER BY id"))
    dms_to_rdkit = {}
    if len(dms_rows) == num_atoms and all(r[1] == rdkit_z[i] for i, r in enumerate(dms_rows)):
        dms_to_rdkit = {r[0]: i for i, r in enumerate(dms_rows)}
    else:
        for r in dms_rows:
            d = np.linalg.norm(rdkit_xyz - np.array(r[2:5]), axis=1)
            d[[z != r[1] for z in rdkit_z]] = np.inf
            j = int(np.argmin(d))
            if d[j] < 0.05:
                dms_to_rdkit[r[0]] = j

    mapping_ok = len(dms_to_rdkit) == num_atoms and len(set(dms_to_rdkit.values())) == num_atoms
    if not mapping_ok:
        logger.warning(f"{dms_path}: could not map {num_atoms - len(dms_to_rdkit)} DMS atoms to the molecule; "
                       "formal charges and stereo flags are not written")

    # -- ud2_sdf_properties: user-defined SDF tags --
    cur.execute(
        "CREATE TABLE ud2_sdf_properties (key TEXT, value TEXT)"
    )
    SKIP_PREFIXES = ('_', 'ud2_', 'source_sdf_file_name', 'source_mol_idx')
    for key in rdkit_mol.GetPropsAsDict():
        if any(key.startswith(p) for p in SKIP_PREFIXES):
            continue
        # Use GetProp (returns raw string) to avoid float precision loss
        val = rdkit_mol.GetProp(key)
        cur.execute(
            "INSERT INTO ud2_sdf_properties VALUES (?, ?)",
            (key, val),
        )

    # -- ud2_formal_charges: only non-zero entries --
    cur.execute(
        "CREATE TABLE ud2_formal_charges "
        "(atom_id INTEGER, formal_charge INTEGER)"
    )
    if mapping_ok:
        rdkit_to_dms = {v: k for k, v in dms_to_rdkit.items()}
        for rdkit_idx in range(num_atoms):
            fc = rdkit_mol.GetAtomWithIdx(rdkit_idx).GetFormalCharge()
            if fc != 0:
                dms_idx = rdkit_to_dms.get(rdkit_idx)
                if dms_idx is not None:
                    cur.execute(
                        "INSERT INTO ud2_formal_charges VALUES (?, ?)",
                        (dms_idx, fc),
                    )

    # -- ud2_bond_stereo: SDF V2000 stereo flags from RDKit bond directions --
    # RDKit BondDir → SDF V2000 stereo: BEGINWEDGE=1 (Up), BEGINDASH=6 (Down)
    BONDDIR_TO_SDF = {1: 1, 6: 6}  # Chem.BondDir.BEGINWEDGE=1, BEGINDASH=6
    cur.execute(
        "CREATE TABLE ud2_bond_stereo "
        "(bond_p0 INTEGER, bond_p1 INTEGER, stereo INTEGER)"
    )
    if mapping_ok:
        # Kekulize a copy for bond order; use original for stereo directions
        for bond in rdkit_mol.GetBonds():
            bdir = int(bond.GetBondDir())
            sdf_stereo = BONDDIR_TO_SDF.get(bdir, 0)
            if sdf_stereo != 0:
                ri = bond.GetBeginAtomIdx()
                rj = bond.GetEndAtomIdx()
                di = rdkit_to_dms.get(ri)
                dj = rdkit_to_dms.get(rj)
                if di is not None and dj is not None:
                    cur.execute(
                        "INSERT INTO ud2_bond_stereo VALUES (?, ?, ?)",
                        (di, dj, sdf_stereo),
                    )

    # -- Store original SDF _Name for C++ SDF output header --
    if rdkit_mol.HasProp('_Name'):
        cur.execute(
            "INSERT OR REPLACE INTO ud2_metadata VALUES ('original_name', ?)",
            (rdkit_mol.GetProp('_Name'),),
        )


def enrich_ligand_dms(
    dms_path: str,
    atom_info_nested_list: list,
    torsion_info_nested_list: list,
    root_atom_idx_list: list,
    molecule_name: str = '',
    rdkit_mol=None,
) -> None:
    """Write ud2 topology tables into a ligand DMS file (modified in place).

    Tables written:
      - ud2_atom_props (atom_id, vina_type, ff_type)
      - ud2_torsions (torsion_id, atom_i, atom_j, atom_k, atom_l,
                       dihedral_value, mobile_atoms)
      - ud2_torsion_ranges (torsion_id, range_lo, range_hi)
      - ud2_root_atoms (atom_id)
      - ud2_metadata (key, value) -- molecule_name for C++ output keying
      - ud2_sdf_properties (key, value) -- original SDF tags for C++ SDF output
      - ud2_formal_charges (atom_id, formal_charge) -- for M CHG in SDF
      - ud2_bond_stereo (bond_p0, bond_p1, stereo) -- for SDF bond stereo column
    """
    conn = sqlite3.connect(dms_path)
    cur = conn.cursor()

    # Drop existing ud2 tables if present (idempotent enrichment)
    for tbl in ['ud2_atom_props', 'ud2_torsions', 'ud2_torsion_ranges',
                'ud2_root_atoms', 'ud2_metadata',
                'ud2_sdf_properties', 'ud2_formal_charges', 'ud2_bond_stereo']:
        cur.execute(f"DROP TABLE IF EXISTS [{tbl}]")

    # ud2_atom_props
    cur.execute(
        "CREATE TABLE ud2_atom_props ("
        "atom_id INTEGER PRIMARY KEY, vina_type INTEGER, ff_type INTEGER)"
    )
    for atom_id, atom_info in enumerate(atom_info_nested_list):
        vina_type = atom_info[3]
        ff_type = atom_info[4]
        cur.execute(
            "INSERT INTO ud2_atom_props VALUES (?, ?, ?)",
            (atom_id, vina_type, ff_type),
        )

    # ud2_torsions
    cur.execute(
        "CREATE TABLE ud2_torsions ("
        "torsion_id INTEGER PRIMARY KEY, "
        "atom_i INTEGER, atom_j INTEGER, atom_k INTEGER, atom_l INTEGER, "
        "dihedral_value REAL, mobile_atoms TEXT)"
    )
    for tor_id, tor_info in enumerate(torsion_info_nested_list):
        atoms = tor_info[0]  # [i, j, k, l]
        dihedral_value = tor_info[1]  # degrees
        mobile_atoms_list = tor_info[3]
        mobile_atoms_str = ','.join(str(a) for a in mobile_atoms_list)
        cur.execute(
            "INSERT INTO ud2_torsions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tor_id, atoms[0], atoms[1], atoms[2], atoms[3],
             dihedral_value, mobile_atoms_str),
        )

    # ud2_torsion_ranges
    cur.execute(
        "CREATE TABLE ud2_torsion_ranges ("
        "torsion_id INTEGER, range_lo REAL, range_hi REAL)"
    )
    for tor_id, tor_info in enumerate(torsion_info_nested_list):
        range_list = tor_info[2]  # list of [lo, hi] pairs
        for lo, hi in range_list:
            cur.execute(
                "INSERT INTO ud2_torsion_ranges VALUES (?, ?, ?)",
                (tor_id, lo, hi),
            )

    # ud2_root_atoms
    cur.execute(
        "CREATE TABLE ud2_root_atoms (atom_id INTEGER)"
    )
    for atom_id in root_atom_idx_list:
        cur.execute("INSERT INTO ud2_root_atoms VALUES (?)", (atom_id,))

    # ud2_metadata -- stores molecule_name so C++ output JSON uses the right key
    cur.execute(
        "CREATE TABLE ud2_metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    if molecule_name:
        cur.execute(
            "INSERT INTO ud2_metadata VALUES ('molecule_name', ?)",
            (molecule_name,),
        )

    # --- SDF-output tables (populated from RDKit mol when available) ---
    if rdkit_mol is not None:
        write_sdf_tables_from_mol(cur, dms_path, rdkit_mol)

    conn.commit()
    conn.close()


def enrich_receptor_dms(dms_path: str, receptor_atom_info: list) -> None:
    """Write ud2_atom_props (Vina types) into a receptor DMS file (in place).

    receptor_atom_info: list of [x, y, z, vina_type, charge] per atom.
    """
    conn = sqlite3.connect(dms_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ud2_atom_props")
    cur.execute(
        "CREATE TABLE ud2_atom_props ("
        "atom_id INTEGER PRIMARY KEY, vina_type INTEGER, ff_type INTEGER)"
    )
    for atom_id, atom_info in enumerate(receptor_atom_info):
        vina_type = atom_info[3]
        ff_type = atom_info[4]
        cur.execute(
            "INSERT INTO ud2_atom_props VALUES (?, ?, ?)",
            (atom_id, vina_type, ff_type),
        )
    conn.commit()
    conn.close()


def prepare_dms_lightweight(
    sdf_path: str,
    receptor_path: str,
    working_dir: str,
    keep_h_rotors: bool = False,
) -> Tuple[str, str]:
    """Build enriched ligand + receptor DMS without GAFF2 or throwaway docking.

    Ligand:   SDF -> build_ligand_dms_minimal (msys+RDKit)
              -> build_molecular_graph (RDKit, skip_gaff2=True)
              -> enrich_ligand_dms (sqlite3)
    Receptor: PDB/DMS -> UnidockReceptorTopologyBuilder
              -> analyze_receptor_topology_vina_only
              -> enrich_receptor_dms (sqlite3)

    Returns (ligand_dms_path, receptor_dms_path).
    """
    os.makedirs(working_dir, exist_ok=True)

    # --- Ligand side ---
    dms_path = build_ligand_dms_minimal(sdf_path, working_dir)
    if dms_path is None:
        raise RuntimeError(f"Failed to build minimal DMS from {sdf_path}")

    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        raise RuntimeError(f"Cannot read molecule from {sdf_path}")

    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        build_molecular_graph,
        load_torsion_library,
    )
    torsion_dict = load_torsion_library()
    atom_info, torsion_info, root_atoms, _frag = build_molecular_graph(
        mol, torsion_dict, skip_gaff2=True, keep_h_rotors=keep_h_rotors,
    )

    mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else 'MOL_0'
    enrich_ligand_dms(dms_path, atom_info, torsion_info, root_atoms,
                       molecule_name=mol_name, rdkit_mol=mol)
    logger.info("Lightweight ligand DMS ready: %s", dms_path)

    # --- Receptor side ---
    builder = UnidockReceptorTopologyBuilder(
        receptor_path, working_dir_name=working_dir,
    )
    builder.generate_receptor_topology()
    rec_dms = builder.receptor_parameterized_dms_file_name
    if dms_has_ud2_atom_props(rec_dms):
        logger.info("Receptor DMS already has Vina atom types, skipping typing: %s", rec_dms)
        return dms_path, rec_dms

    builder.analyze_receptor_topology_vina_only()
    builder.get_summary_receptor_info_dict()
    enrich_receptor_dms(rec_dms, builder.atom_info_nested_list)
    logger.info("Lightweight receptor DMS ready: %s", rec_dms)

    return dms_path, rec_dms


def merge_ligand_dms_files(
    dms_dict: Dict[str, str], output_path: str
) -> str:
    """Merge N per-ligand DMS files into one using msys.

    Appends all ligand systems into a single msys System and writes a
    ``ud2_molecules`` tracking table so the C++ reader can split atoms
    back into per-ligand groups.

    Parameters
    ----------
    dms_dict : dict
        Mapping of mol_name -> per-ligand DMS file path.
    output_path : str
        Where to write the merged DMS.

    Returns
    -------
    str
        *output_path* (for chaining).
    """
    import msys

    mol_info: list = []  # (mol_name, atom_offset, natom)
    bias_rows: list = []  # ud2align position-bias rows, atom ids shifted to the merged system
    merged = None

    for mol_name in sorted(dms_dict, key=lambda x: int(x.split('_')[1])):
        mol = msys.Load(dms_dict[mol_name])
        atom_offset = merged.natoms if merged else 0
        natom = mol.natoms
        if 'ud2_bias' in mol.auxtable_names:
            bias_rows.extend(
                (p['atom_id'] + atom_offset, p['ref_x'], p['ref_y'], p['ref_z'],
                 p['v_set'], p['r'])
                for p in mol.auxtable('ud2_bias').params
            )
        if merged is None:
            merged = mol
        else:
            merged.append(mol)
        mol_info.append((mol_name, atom_offset, natom))

    msys.SaveDMS(merged, output_path)

    # Add ud2_molecules tracking table
    conn = sqlite3.connect(output_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE ud2_molecules ("
        "molecule_id INTEGER PRIMARY KEY, mol_name TEXT, "
        "atom_offset INTEGER, natom INTEGER)"
    )
    for i, (name, off, na) in enumerate(mol_info):
        cur.execute(
            "INSERT INTO ud2_molecules VALUES (?,?,?,?)", (i, name, off, na)
        )
    # msys keeps only the first ligand's ud2_bias table through append, with per-ligand atom
    # ids; rebuild it from every ligand with merged ids (per-ligand row order kept: the engine
    # expects one atom's rows to be contiguous) and index it, since the engine queries it once
    # per ligand (an unindexed 45k-row table cost 9 s of reads on a 7316-ligand screen).
    cur.execute("DROP TABLE IF EXISTS ud2_bias")
    if bias_rows:
        cur.execute(
            "CREATE TABLE ud2_bias ("
            "atom_id INTEGER, ref_x REAL, ref_y REAL, ref_z REAL, "
            "v_set REAL, r REAL)"
        )
        cur.executemany("INSERT INTO ud2_bias VALUES (?,?,?,?,?,?)", bias_rows)
        cur.execute("CREATE INDEX ix_bias_atom ON ud2_bias(atom_id)")
    conn.commit()
    conn.close()
    logger.info(
        "Merged %d ligand DMS files into %s (%d bias rows)",
        len(mol_info), output_path, len(bias_rows),
    )
    return output_path


def create_merged_enrichment_tables(cur) -> None:
    """Drop and recreate the ud2_* enrichment tables with molecule_id columns.

    ``ud2_atom_aromatic`` and ``ud2_torsion_penalties`` are keyed by global atom id
    only (the engine filters them by atom range), so they keep their single-ligand
    schema. They must still be rebuilt here: msys copies the per-ligand rows into
    the merged system verbatim, without shifting the atom ids.
    """
    for tbl in [
        'ud2_atom_props', 'ud2_torsions', 'ud2_torsion_ranges',
        'ud2_root_atoms', 'ud2_metadata',
        'ud2_sdf_properties', 'ud2_formal_charges', 'ud2_bond_stereo',
        'ud2_atom_aromatic', 'ud2_torsion_penalties',
    ]:
        cur.execute(f"DROP TABLE IF EXISTS [{tbl}]")

    cur.execute(
        "CREATE TABLE ud2_atom_props ("
        "atom_id INTEGER, molecule_id INTEGER, "
        "vina_type INTEGER, ff_type INTEGER)"
    )
    cur.execute(
        "CREATE TABLE ud2_torsions ("
        "torsion_id INTEGER, molecule_id INTEGER, "
        "atom_i INTEGER, atom_j INTEGER, atom_k INTEGER, atom_l INTEGER, "
        "dihedral_value REAL, mobile_atoms TEXT)"
    )
    cur.execute(
        "CREATE TABLE ud2_torsion_ranges ("
        "torsion_id INTEGER, molecule_id INTEGER, "
        "range_lo REAL, range_hi REAL)"
    )
    cur.execute(
        "CREATE TABLE ud2_root_atoms (atom_id INTEGER, molecule_id INTEGER)"
    )
    cur.execute(
        "CREATE TABLE ud2_metadata (molecule_id INTEGER, key TEXT, value TEXT)"
    )
    cur.execute(
        "CREATE TABLE ud2_sdf_properties ("
        "molecule_id INTEGER, key TEXT, value TEXT)"
    )
    cur.execute(
        "CREATE TABLE ud2_formal_charges ("
        "atom_id INTEGER, molecule_id INTEGER, formal_charge INTEGER)"
    )
    cur.execute(
        "CREATE TABLE ud2_bond_stereo ("
        "molecule_id INTEGER, bond_p0 INTEGER, bond_p1 INTEGER, "
        "stereo INTEGER)"
    )
    cur.execute(
        "CREATE TABLE ud2_atom_aromatic (atom_id INTEGER)"
    )
    cur.execute(
        "CREATE TABLE ud2_torsion_penalties ("
        "atom_i INTEGER, atom_j INTEGER, atom_k INTEGER, atom_l INTEGER, "
        "kind INTEGER, tolerance REAL, weight_share REAL, class_name TEXT)"
    )


def insert_merged_ligand_data(
    cur, dms_path: str, mol_id: int, atom_offset: int,
    lig_name: str, lig_data: dict, rdkit_mol, global_torsion_id: int,
) -> int:
    """Insert one ligand's topology data into the merged enrichment tables.

    Returns the updated ``global_torsion_id``.
    """
    for local_id, atom_info in enumerate(lig_data['atoms']):
        vina_type = atom_info[3]
        ff_type = atom_info[4]
        cur.execute(
            "INSERT INTO ud2_atom_props VALUES (?,?,?,?)",
            (local_id + atom_offset, mol_id, vina_type, ff_type),
        )

    for tor_info in lig_data['torsions']:
        atoms = tor_info[0]  # [i, j, k, l]
        dihedral_value = tor_info[1]
        mobile_atoms_list = tor_info[3]
        mobile_atoms_str = ','.join(
            str(a + atom_offset) for a in mobile_atoms_list
        )
        cur.execute(
            "INSERT INTO ud2_torsions VALUES (?,?,?,?,?,?,?,?)",
            (
                global_torsion_id, mol_id,
                atoms[0] + atom_offset,
                atoms[1] + atom_offset,
                atoms[2] + atom_offset,
                atoms[3] + atom_offset,
                dihedral_value,
                mobile_atoms_str,
            ),
        )

        range_list = tor_info[2]  # list of [lo, hi] pairs
        for lo, hi in range_list:
            cur.execute(
                "INSERT INTO ud2_torsion_ranges VALUES (?,?,?,?)",
                (global_torsion_id, mol_id, lo, hi),
            )

        global_torsion_id += 1

    for atom_id in lig_data['root_atoms']:
        cur.execute(
            "INSERT INTO ud2_root_atoms VALUES (?,?)",
            (atom_id + atom_offset, mol_id),
        )

    cur.execute(
        "INSERT INTO ud2_metadata VALUES (?,?,?)",
        (mol_id, 'molecule_name', lig_name),
    )

    if rdkit_mol is not None:
        write_sdf_tables_merged(
            cur, dms_path, rdkit_mol, mol_id, atom_offset
        )
        write_scoring_annotations_merged(cur, rdkit_mol, atom_offset)

    return global_torsion_id


def write_scoring_annotations_merged(cur, rdkit_mol, atom_offset: int) -> None:
    """Write aromatic flags and torsion-strain quadruples for one merged ligand.

    Atom ids are the merged (global) ones, mapped from RDKit order by coordinate
    just as ``write_sdf_tables_merged`` does.
    """
    try:
        from unidock_processing.ligand_topology.scoring_annotations import (
            aromatic_atom_ids, torsion_penalty_rows,
        )
    except ImportError as e:
        logger.warning("Cannot write scoring annotations: %s", e)
        return

    num_atoms = rdkit_mol.GetNumAtoms()
    conf = rdkit_mol.GetConformer()
    rdkit_pos_to_idx = {
        (round(conf.GetAtomPosition(i).x, 3),
         round(conf.GetAtomPosition(i).y, 3),
         round(conf.GetAtomPosition(i).z, 3)): i
        for i in range(num_atoms)
    }
    rdkit_to_dms = {}
    for row in cur.execute(
        "SELECT id, x, y, z FROM particle WHERE id >= ? AND id < ?",
        (atom_offset, atom_offset + num_atoms),
    ):
        key = (round(row[1], 3), round(row[2], 3), round(row[3], 3))
        if key in rdkit_pos_to_idx:
            rdkit_to_dms[rdkit_pos_to_idx[key]] = row[0]

    if len(rdkit_to_dms) != num_atoms:
        logger.warning(
            "Coordinate mapping incomplete at atom offset %d (%d/%d); "
            "skipping scoring annotations for this ligand",
            atom_offset, len(rdkit_to_dms), num_atoms,
        )
        return

    cur.executemany(
        "INSERT INTO ud2_atom_aromatic VALUES (?)",
        [(rdkit_to_dms[i],) for i in aromatic_atom_ids(rdkit_mol)],
    )
    cur.executemany(
        "INSERT INTO ud2_torsion_penalties VALUES (?,?,?,?,?,?,?,?)",
        [
            (rdkit_to_dms[i], rdkit_to_dms[j], rdkit_to_dms[k], rdkit_to_dms[l],
             kind, tol, share, name)
            for i, j, k, l, kind, tol, share, name in torsion_penalty_rows(rdkit_mol)
        ],
    )


def enrich_merged_ligand_dms(
    dms_path: str,
    ligand_builder: 'UnidockLigandTopologyBuilder',
) -> None:
    """Write ud2_* topology tables for ALL ligands into a merged DMS.

    Each table includes a ``molecule_id`` column and atom IDs are offset
    to global (merged) indices using the ``ud2_molecules`` table.

    Parameters
    ----------
    dms_path : str
        Path to the merged DMS (already contains ``ud2_molecules``).
    ligand_builder : UnidockLigandTopologyBuilder
        Provides topology data via ``total_ligand_info_summary_dict``
        and ``ligand_mol_list``.
    """
    conn = sqlite3.connect(dms_path)
    cur = conn.cursor()

    mol_boundaries: list = []
    for row in cur.execute(
        "SELECT molecule_id, mol_name, atom_offset, natom "
        "FROM ud2_molecules ORDER BY molecule_id"
    ):
        mol_boundaries.append(row)
    name_to_boundary = {row[1]: row for row in mol_boundaries}

    create_merged_enrichment_tables(cur)

    summary = ligand_builder.total_ligand_info_summary_dict
    mol_list = ligand_builder.ligand_mol_list
    global_torsion_id = 0

    for mol_idx, lig_name in enumerate(
        sorted(summary.keys(), key=lambda x: int(x.split('_')[1]))
    ):
        if lig_name not in name_to_boundary:
            logger.warning(
                "Ligand %s not found in ud2_molecules; skipping", lig_name
            )
            continue

        mol_id, _, atom_offset, _ = name_to_boundary[lig_name]
        rdkit_mol = mol_list[mol_idx] if mol_idx < len(mol_list) else None

        global_torsion_id = insert_merged_ligand_data(
            cur, dms_path, mol_id, atom_offset,
            lig_name, summary[lig_name], rdkit_mol, global_torsion_id,
        )

    # The engine reads a merged DMS one ligand at a time, restricted to that ligand's atom range.
    # Without these the per-ligand queries scan the whole table, which is quadratic in the ligand
    # count: on a 7316-ligand file the loader took 324 s of a 1740 s run.
    for stmt in (
        "CREATE INDEX IF NOT EXISTS ix_bond_p0 ON bond(p0)",
        "CREATE INDEX IF NOT EXISTS ix_props_atom ON ud2_atom_props(atom_id)",
        "CREATE INDEX IF NOT EXISTS ix_arom_atom ON ud2_atom_aromatic(atom_id)",
        "CREATE INDEX IF NOT EXISTS ix_pen_atom ON ud2_torsion_penalties(atom_j)",
        "CREATE INDEX IF NOT EXISTS ix_fcharge_atom ON ud2_formal_charges(atom_id)",
        "CREATE INDEX IF NOT EXISTS ix_root_atom ON ud2_root_atoms(atom_id)",
        "CREATE INDEX IF NOT EXISTS ix_meta_mol ON ud2_metadata(molecule_id, key)",
    ):
        try:
            cur.execute(stmt)
        except sqlite3.Error as e:      # a table may be absent in a minimal build
            logger.debug("index skipped: %s (%s)", stmt, e)

    conn.commit()
    conn.close()
    logger.info("Enriched merged ligand DMS with %d ligands", len(summary))


def write_sdf_tables_merged(
    cur, dms_path: str, rdkit_mol, molecule_id: int, atom_offset: int
) -> None:
    """Write SDF output tables for one ligand into a merged DMS.

    Like ``write_sdf_tables_from_mol`` but adds ``molecule_id`` and
    offsets atom indices by *atom_offset*.
    """
    from rdkit import Chem

    num_atoms = rdkit_mol.GetNumAtoms()

    # Build DMS-to-RDKit atom index mapping via coordinate matching
    conf = rdkit_mol.GetConformer()
    rdkit_pos_to_idx = {}
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        rdkit_pos_to_idx[
            (round(pos.x, 3), round(pos.y, 3), round(pos.z, 3))
        ] = i

    # Match against the merged particle table in the atom_offset range
    dms_to_rdkit = {}
    for row in cur.execute(
        "SELECT id, x, y, z FROM particle WHERE id >= ? AND id < ?",
        (atom_offset, atom_offset + num_atoms),
    ):
        key = (round(row[1], 3), round(row[2], 3), round(row[3], 3))
        if key in rdkit_pos_to_idx:
            dms_to_rdkit[row[0]] = rdkit_pos_to_idx[key]

    mapping_ok = len(dms_to_rdkit) == num_atoms

    # ud2_sdf_properties
    SKIP_PREFIXES = ('_', 'ud2_', 'source_sdf_file_name', 'source_mol_idx')
    for key in rdkit_mol.GetPropsAsDict():
        if any(key.startswith(p) for p in SKIP_PREFIXES):
            continue
        val = rdkit_mol.GetProp(key)
        cur.execute(
            "INSERT INTO ud2_sdf_properties VALUES (?,?,?)",
            (molecule_id, key, val),
        )

    # ud2_formal_charges (offset, only non-zero)
    if mapping_ok:
        rdkit_to_dms = {v: k for k, v in dms_to_rdkit.items()}
        for rdkit_idx in range(num_atoms):
            fc = rdkit_mol.GetAtomWithIdx(rdkit_idx).GetFormalCharge()
            if fc != 0:
                dms_idx = rdkit_to_dms.get(rdkit_idx)
                if dms_idx is not None:
                    cur.execute(
                        "INSERT INTO ud2_formal_charges VALUES (?,?,?)",
                        (dms_idx, molecule_id, fc),
                    )

    # ud2_bond_stereo (offset)
    BONDDIR_TO_SDF = {1: 1, 6: 6}
    if mapping_ok:
        rdkit_to_dms = {v: k for k, v in dms_to_rdkit.items()}
        for bond in rdkit_mol.GetBonds():
            bdir = int(bond.GetBondDir())
            sdf_stereo = BONDDIR_TO_SDF.get(bdir, 0)
            if sdf_stereo != 0:
                ri = bond.GetBeginAtomIdx()
                rj = bond.GetEndAtomIdx()
                di = rdkit_to_dms.get(ri)
                dj = rdkit_to_dms.get(rj)
                if di is not None and dj is not None:
                    cur.execute(
                        "INSERT INTO ud2_bond_stereo VALUES (?,?,?,?)",
                        (molecule_id, di, dj, sdf_stereo),
                    )

    # Store original SDF _Name
    if rdkit_mol.HasProp('_Name'):
        cur.execute(
            "INSERT INTO ud2_metadata VALUES (?,?,?)",
            (molecule_id, 'original_name', rdkit_mol.GetProp('_Name')),
        )


def write_scoring_annotations_safe(dms_path: str, mol) -> None:
    """Aromatic flags and torsion-strain quadruples for the engine's rescoring terms."""
    try:
        from unidock_processing.ligand_topology.scoring_annotations import write_scoring_annotations
        write_scoring_annotations(dms_path, mol)
    except Exception as e:  # annotations are optional; the engine scores those terms as zero
        logger.warning("Could not write scoring annotations for %s: %s", dms_path, e)


def build_ligand_dms_minimal(
    sdf_path: str, working_dir: str, mol=None, write_annotations: bool = True
) -> Optional[str]:
    """Build a minimal DMS from an SDF file without antechamber/GAFF2.

    Creates a DMS with coordinates, bonds, and Gasteiger charges only.
    No FF TermTables (nonbonded, exclusion, pair_12_6_es, dihedral_trig).
    Used as fallback when antechamber fails (e.g. covalent ligands).

    ``mol`` is the already-parsed ligand; when given, ``sdf_path`` is not read and
    only names the output files. ``write_annotations=False`` leaves the scoring
    annotation tables to the caller (the merged-DMS path rebuilds them anyway).

    Returns the DMS path on success, None on failure.
    """
    import math
    from copy import deepcopy

    try:
        import msys
    except ImportError as e:
        logger.warning("Cannot build minimal DMS (missing msys): %s", e)
        return None

    if mol is None:
        mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        logger.warning("Cannot read SDF: %s", sdf_path)
        return None

    num_atoms = mol.GetNumAtoms()
    basename = os.path.splitext(os.path.basename(sdf_path))[0]
    dms_dir = os.path.join(working_dir, f'_dms_build_{basename}')
    os.makedirs(dms_dir, exist_ok=True)

    # Compute Gasteiger charges
    from rdkit.Chem import AllChem
    AllChem.ComputeGasteigerCharges(mol)

    # Kekulize for bond orders
    kek_mol = deepcopy(mol)
    try:
        Chem.Kekulize(kek_mol, clearAromaticFlags=False)
    except Exception:
        kek_mol = mol

    # Build msys system
    sys_mol = msys.CreateSystem()
    ct = sys_mol.addCt()
    chain = ct.addChain()
    res = chain.addResidue()
    res.name = 'LIG'

    conf = mol.GetConformer()
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        a = res.addAtom()
        a.atomic_number = atom.GetAtomicNum()
        a.name = atom.GetSymbol() + str(i)
        pos = conf.GetAtomPosition(i)
        a.x = pos.x
        a.y = pos.y
        a.z = pos.z
        try:
            q = atom.GetDoubleProp('_GasteigerCharge')
            if not math.isfinite(q):
                q = 0.0
        except (KeyError, RuntimeError):
            q = 0.0
        a.charge = q
        a.formal_charge = atom.GetFormalCharge()

    # Add bonds with orders from Kekulized mol
    for bond in kek_mol.GetBonds():
        b = sys_mol.atom(bond.GetBeginAtomIdx()).addBond(
            sys_mol.atom(bond.GetEndAtomIdx()))
        b.order = int(bond.GetBondTypeAsDouble())

    dms_path = os.path.join(dms_dir, f'{basename}.dms')
    msys.SaveDMS(sys_mol, dms_path)

    # Create empty FF tables so the C++ engine reader doesn't crash.
    # These tables have zero rows, so no FF scoring occurs.
    import sqlite3
    conn = sqlite3.connect(dms_path)
    # The DMS is a scratch file rebuilt on every run: skip the per-statement fsync.
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=MEMORY")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS nonbonded_param "
        "(id INTEGER PRIMARY KEY, sigma REAL, epsilon REAL, type TEXT)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS pair_12_6_es_param "
        "(id INTEGER PRIMARY KEY, aij REAL, bij REAL, qij REAL)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS pair_12_6_es_term "
        "(p0 INTEGER, p1 INTEGER, param INTEGER)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS dihedral_trig_param "
        "(id INTEGER PRIMARY KEY, phi0 REAL, "
        "fc0 REAL, fc1 REAL, fc2 REAL, fc3 REAL, fc4 REAL, fc5 REAL, fc6 REAL)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS dihedral_trig_term "
        "(p0 INTEGER, p1 INTEGER, p2 INTEGER, p3 INTEGER, param INTEGER)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS exclusion (p0 INTEGER, p1 INTEGER)"
    )
    # Add nbtype column to particle if missing (engine JOINs on it)
    try:
        cur.execute("ALTER TABLE particle ADD COLUMN nbtype INTEGER DEFAULT 0")
    except Exception:
        pass  # column already exists
    conn.commit()
    conn.close()

    if write_annotations:
        write_scoring_annotations_safe(dms_path, mol)

    logger.info(
        "Built minimal ligand DMS (no FF) from SDF: %s (%d atoms)",
        dms_path, num_atoms,
    )
    return dms_path


def build_ligand_dms_from_sdf(sdf_path: str, working_dir: str) -> Optional[str]:
    """Build a DMS file with GAFF2 FF parameters from an SDF file.

    Pipeline: SDF -> antechamber -> mol2 -> parmchk2 -> frcmod
              -> tleap -> prmtop/inpcrd -> msys + parmed -> SaveDMS.

    Coordinates, bond orders, charges, and all FF TermTables (nonbonded,
    exclusion, pair_12_6_es, dihedral_trig) are built on the msys system
    object and saved in a single msys.SaveDMS call.

    Returns the DMS path on success, None on failure.
    """
    import math

    try:
        import parmed
        import msys
    except ImportError as e:
        logger.warning("Cannot build DMS from SDF (missing dep): %s", e)
        return None

    basename = os.path.splitext(os.path.basename(sdf_path))[0]
    dms_dir = os.path.join(working_dir, f'_dms_build_{basename}')
    os.makedirs(dms_dir, exist_ok=True)

    # --- 1. Read SDF and compute formal charge ---
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        logger.warning("Cannot read SDF: %s", sdf_path)
        return None
    num_atoms = mol.GetNumAtoms()
    formal_charge = Chem.GetFormalCharge(mol)

    temp_sdf = os.path.join(dms_dir, 'lig.sdf')
    utils.convert_v3000_mol_to_v2000_sdf(mol, temp_sdf)

    # --- 2. Run antechamber (-dr no handles sulfonamide/tetrazole) + parmchk2 + tleap ---
    mol2 = os.path.join(dms_dir, 'lig.mol2')
    frcmod = os.path.join(dms_dir, 'lig.frcmod')
    prmtop = os.path.join(dms_dir, 'lig.prmtop')
    inpcrd = os.path.join(dms_dir, 'lig.inpcrd')

    ante_cmd = (
        f'cd {dms_dir} && '
        f'antechamber -i {temp_sdf} -fi sdf -o {mol2} -fo mol2 '
        f'-at gaff2 -c gas -nc {formal_charge} -eq 2 -pf y -dr no '
        f'> ante.log 2>&1'
    )
    os.system(ante_cmd)
    if not os.path.isfile(mol2):
        logger.warning("antechamber failed for %s", sdf_path)
        return None

    parmchk_cmd = (
        f'cd {dms_dir} && '
        f'parmchk2 -i {mol2} -f mol2 -a Y -s 2 -o {frcmod}'
    )
    os.system(parmchk_cmd)
    if not os.path.isfile(frcmod):
        logger.warning("parmchk2 failed for %s", sdf_path)
        return None

    tleap_in = os.path.join(dms_dir, 'tleap.in')
    with open(tleap_in, 'w') as f:
        f.write(
            f'source leaprc.gaff2\n'
            f'MOL = loadmol2 {mol2}\n'
            f'loadamberparams {frcmod}\n'
            f'saveamberparm MOL {prmtop} {inpcrd}\n'
            f'quit\n'
        )
    os.system(f'cd {dms_dir} && tleap -f {tleap_in} > tleap.log 2>&1')
    if not os.path.isfile(prmtop) or not os.path.isfile(inpcrd):
        logger.warning("tleap failed for %s", sdf_path)
        return None

    # --- 3. Create msys system and load parmed for FF parameters ---
    try:
        sys_mol = msys.LoadPrmTop(prmtop, inpcrd)
    except Exception as e:
        logger.warning("msys.LoadPrmTop failed: %s", e)
        return None

    try:
        parm = parmed.load_file(prmtop, inpcrd)
    except Exception as e:
        logger.warning("parmed.load_file failed: %s", e)
        return None

    # --- 4. Build topology + FF on msys system (single SaveDMS at end) ---
    import numpy as np
    from copy import deepcopy
    from collections import defaultdict

    # 4a. Fix coordinates (msys.LoadPrmTop doesn't read inpcrd coords)
    positions = np.array([[a.xx, a.xy, a.xz] for a in parm.atoms])
    sys_mol.setPositions(positions)

    # 4b. Copy Kekulized bond orders from original SDF via coordinate matching
    kek_mol = deepcopy(mol)
    try:
        Chem.Kekulize(kek_mol, clearAromaticFlags=False)
    except Exception:
        kek_mol = mol

    conf = kek_mol.GetConformer()
    rdkit_pos_to_idx = {}
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        rdkit_pos_to_idx[
            (round(pos.x, 3), round(pos.y, 3), round(pos.z, 3))
        ] = i

    parm_to_rdkit = {}
    for a in parm.atoms:
        key = (round(a.xx, 3), round(a.xy, 3), round(a.xz, 3))
        if key in rdkit_pos_to_idx:
            parm_to_rdkit[a.idx] = rdkit_pos_to_idx[key]

    if len(parm_to_rdkit) == num_atoms:
        for bond in sys_mol.bonds:
            p0, p1 = bond.first.id, bond.second.id
            ri = parm_to_rdkit.get(p0)
            rj = parm_to_rdkit.get(p1)
            if ri is not None and rj is not None:
                rb = kek_mol.GetBondBetweenAtoms(ri, rj)
                if rb is not None:
                    bond.order = int(rb.GetBondTypeAsDouble())
        logger.info("Bond orders copied from SDF into DMS for %s", basename)
    else:
        logger.warning(
            "Atom coordinate matching incomplete (%d/%d); "
            "bond orders not updated for %s",
            len(parm_to_rdkit), num_atoms, basename,
        )

    # 4c. Set charges from parmed
    for i, a in enumerate(parm.atoms):
        sys_mol.atom(i).charge = a.charge

    # 4d. nonbonded TermTable (1-atom: sigma, epsilon, type)
    nb = sys_mol.addTable('nonbonded', 1)
    nb.category = 'nonbonded'
    nb.params.addProp('sigma', float)
    nb.params.addProp('epsilon', float)
    nb.params.addProp('type', str)
    type_to_param = {}
    for i, a in enumerate(parm.atoms):
        if a.type not in type_to_param:
            type_to_param[a.type] = nb.params.addParam(
                sigma=a.sigma, epsilon=a.epsilon, type=a.type)
        nb.addTerm([sys_mol.atom(i)], type_to_param[a.type])

    # 4e. exclusion TermTable (2-atom, 1-2 + 1-3 + 1-4 pairs)
    excl = sys_mol.addTableFromSchema('exclusion')
    all_excl = set()
    for a in parm.atoms:
        for p in a.bond_partners:
            all_excl.add((min(a.idx, p.idx), max(a.idx, p.idx)))
        for p in a.angle_partners:
            all_excl.add((min(a.idx, p.idx), max(a.idx, p.idx)))
        for p in a.dihedral_partners:
            all_excl.add((min(a.idx, p.idx), max(a.idx, p.idx)))
    for i, j in sorted(all_excl):
        excl.addTerm([sys_mol.atom(i), sys_mol.atom(j)])

    # 4f. pair_12_6_es TermTable (2-atom, 1-4 pairs)
    pair = sys_mol.addTableFromSchema('pair_12_6_es')
    pairs_14 = set()
    for a in parm.atoms:
        for p in a.dihedral_partners:
            pairs_14.add((min(a.idx, p.idx), max(a.idx, p.idx)))
    SCNB = 2.0  # AMBER/GAFF2 standard 1-4 LJ scaling
    SCEE = 1.2  # AMBER/GAFF2 standard 1-4 EEL scaling
    pp_map = {}
    for i, j in sorted(pairs_14):
        ai, aj = parm.atoms[i], parm.atoms[j]
        sig_ij = (ai.sigma + aj.sigma) / 2.0
        eps_ij = math.sqrt(ai.epsilon * aj.epsilon)
        sig6 = sig_ij ** 6
        sig12 = sig6 * sig6
        a_ij = 4.0 * eps_ij * sig12 / SCNB
        b_ij = 4.0 * eps_ij * sig6 / SCNB
        q_ij = ai.charge * aj.charge / SCEE
        key = (round(a_ij, 6), round(b_ij, 6), round(q_ij, 6))
        if key not in pp_map:
            pp_map[key] = pair.params.addParam(aij=a_ij, bij=b_ij, qij=q_ij)
        pair.addTerm([sys_mol.atom(i), sys_mol.atom(j)], pp_map[key])

    # 4g. dihedral_trig TermTable (4-atom, merged periodicities)
    dihe = sys_mol.addTableFromSchema('dihedral_trig')
    dihedral_groups = defaultdict(list)
    for d in parm.dihedrals:
        dihedral_groups[
            (d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx)
        ].append(d)
    dp_map = {}
    for quad, terms in dihedral_groups.items():
        fc = [0.0] * 7  # fc0..fc6
        for d in terms:
            n = int(d.type.per)
            phi_k = d.type.phi_k
            phase_rad = d.type.phase * math.pi / 180.0
            fc[0] += phi_k
            if 0 <= n <= 6:
                fc[n] += phi_k * math.cos(phase_rad)
        key = tuple(round(v, 6) for v in fc)
        if key not in dp_map:
            dp_map[key] = dihe.params.addParam(
                phi0=0.0, fc0=fc[0], fc1=fc[1], fc2=fc[2],
                fc3=fc[3], fc4=fc[4], fc5=fc[5], fc6=fc[6])
        dihe.addTerm([sys_mol.atom(q) for q in quad], dp_map[key])

    # --- 5. Save DMS with all topology and FF data ---
    dms_path = os.path.join(dms_dir, f'{basename}.dms')
    msys.SaveDMS(sys_mol, dms_path)

    # --- 6. Verify the DMS has all required FF tables ---
    if not dms_has_ff_params(dms_path):
        logger.warning("Built DMS missing FF tables: %s", dms_path)
        return None

    logger.info("Built ligand DMS from SDF: %s (%d atoms)", dms_path, num_atoms)
    return dms_path


def ligand_atom_offsets(dms_path: str) -> Dict[str, int]:
    """Atom offset of every ligand in an enriched DMS, keyed by molecule name (0 for a single-ligand file)."""
    conn = sqlite3.connect(dms_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ud2_molecules'")
        if cur.fetchone() is not None:
            return {
                name: offset
                for name, offset in cur.execute(
                    "SELECT mol_name, atom_offset FROM ud2_molecules ORDER BY molecule_id"
                )
            }
        row = cur.execute("SELECT value FROM ud2_metadata WHERE key='molecule_name'").fetchone()
        return {row[0] if row else 'MOL_0': 0}
    finally:
        conn.close()
