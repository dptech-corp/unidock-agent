"""Ligand annotations used by the engine's rescoring terms.

Two tables are written into the ligand DMS:

``ud2_atom_aromatic``
    One row per aromatic ligand atom. The engine gives an extra hydrophobic weight to
    aromatic-carbon / aromatic-ring-carbon contacts, and it has no aromaticity model of its own.

``ud2_torsion_penalties``
    Flat list of dihedral quadruples with a penalty kind, a tolerance and a weight share. It
    encodes a ligand torsional-strain term: conjugated single bonds (amide, ester, anilide,
    aniline, aryl carbonyl, aryl ether, biaryl, sulfonamide) are penalised when twisted out of
    plane, sp3-sp3 bonds when eclipsed, and amides/esters additionally when cis/E. Every quadruple
    of one (class, bond) group carries 1 / n_quads so the engine only sums.

Both tables are optional: an engine that does not find them scores the terms as zero.
"""

import logging
import sqlite3
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

KIND_PLANAR = 0      # flat-bottom penalty on the deviation from planarity
KIND_ECLIPSED = 1    # flat-bottom penalty on the distance to the nearest eclipse
KIND_CIS = 2         # (1 - cos phi) / 2, penalises the cis amide / E ester

TOLERANCE_DEG = 20.0

# name -> (SMARTS with the rotatable axis mapped as :1 and :2, penalty kind)
TORSION_CLASSES = (
    ('amide_plan',  '[CX3:1](=O)-!@[NX3:2]', KIND_PLANAR),
    ('ester_plan',  '[CX3:1](=O)-!@[OX2:2]-[#6]', KIND_PLANAR),
    ('anilide',     '[a:1]-!@[NX3:2]-[CX3]=O', KIND_PLANAR),
    ('aniline',     '[a:1]-!@[NX3;!$(N[CX3]=[O,S,N]);!$(N-S(=O)=O):2]', KIND_PLANAR),
    ('arylco',      '[a:1]-!@[CX3:2]=[O,N,S,C]', KIND_PLANAR),
    ('arylether',   '[a:1]-!@[OX2:2]-[#6]', KIND_PLANAR),
    ('biaryl',      '[a:1]-!@[a:2]', KIND_PLANAR),
    ('sulfonamide', '[#6:1]-!@[NX3:2]-S(=O)=O', KIND_PLANAR),
    ('amide_cis',   '[OX1]=[CX3:1]-!@[NX3;H1:2]-[#6]', KIND_CIS),
    ('ester_e',     '[OX1]=[CX3:1]-!@[OX2:2]-[#6]', KIND_CIS),
    ('sp3_ecl',     '[CX4,NX3,OX2;^3:1]-!@[CX4,NX3,OX2;^3:2]', KIND_ECLIPSED),
)


def aromatic_atom_ids(mol: Chem.Mol) -> List[int]:
    """Indices of aromatic atoms, in the molecule's own atom order."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]


def _heavy_neighbors(mol: Chem.Mol, atom_idx: int, exclude: int) -> List[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() != exclude]


def torsion_penalty_rows(mol: Chem.Mol) -> List[Tuple[int, int, int, int, int, float, float, str]]:
    """(atom_i, atom_j, atom_k, atom_l, kind, tolerance_deg, weight_share, class_name) per quadruple.

    The axis is (atom_j, atom_k). A class claims each bond at most once; a bond claimed by two
    classes contributes to both, which is intended (a twisted cis amide is penalised twice).
    """
    rows = []
    for name, smarts, kind in TORSION_CLASSES:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning('Cannot parse torsion SMARTS for class %s', name)
            continue
        mapping = {pattern.GetAtomWithIdx(i).GetAtomMapNum(): i
                   for i in range(pattern.GetNumAtoms())
                   if pattern.GetAtomWithIdx(i).GetAtomMapNum()}
        if 1 not in mapping or 2 not in mapping:
            continue
        claimed = set()
        for match in mol.GetSubstructMatches(pattern, uniquify=False):
            axis_j, axis_k = match[mapping[1]], match[mapping[2]]
            key = (min(axis_j, axis_k), max(axis_j, axis_k))
            if key in claimed:
                continue
            if kind == KIND_CIS:
                # the carbonyl oxygen of the pattern, double bonded to the axis carbon
                oxygens = [match[i] for i in range(len(match))
                           if pattern.GetAtomWithIdx(i).GetAtomicNum() == 8 and i not in mapping.values()]
                oxygens = [o for o in oxygens
                           if mol.GetBondBetweenAtoms(o, axis_j) is not None
                           and mol.GetBondBetweenAtoms(o, axis_j).GetBondTypeAsDouble() == 2.0]
                tails = _heavy_neighbors(mol, axis_k, axis_j)
                if not oxygens or not tails:
                    continue
                claimed.add(key)
                share = 1.0 / len(tails)
                for tail in tails:
                    rows.append((oxygens[0], axis_j, axis_k, tail, kind, 0.0, share, name))
                continue
            heads = _heavy_neighbors(mol, axis_j, axis_k)
            tails = _heavy_neighbors(mol, axis_k, axis_j)
            if not heads or not tails:
                continue
            claimed.add(key)
            share = 1.0 / (len(heads) * len(tails))
            for head in heads:
                for tail in tails:
                    rows.append((head, axis_j, axis_k, tail, kind, TOLERANCE_DEG, share, name))
    return rows


def write_scoring_annotations(dms_path: str, mol: Chem.Mol, atom_offset: int = 0,
                              molecule_id: Optional[int] = None) -> None:
    """Write both tables into an existing DMS. Atom indices are shifted by ``atom_offset``.

    ``mol`` must have the same atom order as the DMS particle table (hydrogens included).
    Safe to call repeatedly: rows of this molecule are replaced.
    """
    conn = sqlite3.connect(dms_path)
    # Skip the per-statement fsync; callers hand in a scratch DMS that is rebuilt on every run.
    conn.execute('PRAGMA synchronous=OFF')
    conn.execute('PRAGMA journal_mode=MEMORY')
    try:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS ud2_atom_aromatic (atom_id INTEGER)')
        cur.execute(
            'CREATE TABLE IF NOT EXISTS ud2_torsion_penalties ('
            'atom_i INTEGER, atom_j INTEGER, atom_k INTEGER, atom_l INTEGER, '
            'kind INTEGER, tolerance REAL, weight_share REAL, class_name TEXT)'
        )
        n_atom = mol.GetNumAtoms()
        cur.execute('DELETE FROM ud2_atom_aromatic WHERE atom_id >= ? AND atom_id < ?',
                    (atom_offset, atom_offset + n_atom))
        cur.execute('DELETE FROM ud2_torsion_penalties WHERE atom_j >= ? AND atom_j < ?',
                    (atom_offset, atom_offset + n_atom))
        cur.executemany('INSERT INTO ud2_atom_aromatic VALUES (?)',
                        [(atom_offset + i,) for i in aromatic_atom_ids(mol)])
        cur.executemany(
            'INSERT INTO ud2_torsion_penalties VALUES (?,?,?,?,?,?,?,?)',
            [(atom_offset + i, atom_offset + j, atom_offset + k, atom_offset + l, kind, tol, share, name)
             for i, j, k, l, kind, tol, share, name in torsion_penalty_rows(mol)]
        )
        conn.commit()
    finally:
        conn.close()


def torsion_penalty_value(mol_conf_positions: Sequence[Sequence[float]],
                          rows: Sequence[Tuple[int, int, int, int, int, float, float]]) -> float:
    """Reference implementation of the term, used by tests and by the offline validation."""
    import numpy as np

    coords = np.asarray(mol_conf_positions, dtype=float)
    total = 0.0
    for row in rows:
        i, j, k, l, kind, tol, share = row[:7]
        b0 = coords[i] - coords[j]
        b1 = coords[k] - coords[j]
        b2 = coords[l] - coords[k]
        n1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, n1) * n1
        w = b2 - np.dot(b2, n1) * n1
        phi = np.arctan2(np.dot(np.cross(n1, v), w), np.dot(v, w))
        if kind == KIND_PLANAR:
            dev = np.degrees(np.arcsin(min(abs(np.sin(phi)), 1.0)))
            pen = np.clip((dev - tol) / (90.0 - tol), 0.0, 1.0)
        elif kind == KIND_ECLIPSED:
            dev = np.degrees(np.arccos(np.clip(np.cos(3 * phi), -1.0, 1.0))) / 3.0
            pen = np.clip((60.0 - tol - dev) / (60.0 - tol), 0.0, 1.0)
        else:
            pen = (1.0 - np.cos(phi)) / 2.0
        total += share * pen
    return float(total)
