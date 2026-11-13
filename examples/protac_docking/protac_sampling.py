#!/usr/bin/env python3
"""
protac_sampling.py – Build Uni-Dock2 enriched DMS for PROTAC ternary complex sampling.

Workflow
--------
1. Constrained-embed PROTAC to EP (MCS atoms pinned to EP.sdf coords)
2. Constrained-embed aligned PROTAC to TP (MCS atoms pinned to TP.sdf coords)
3. Compute rigid-body R, t from EP-part movement (Kabsch SVD)
4. Build PROTAC torsion tree (root = TP, EP frozen)
5. Transform VHL with R, t and merge as rigid ligand extension
6. Build enriched DMS files: ligand (PROTAC+VHL) + receptor
"""
import os
import math
import json
import sqlite3
import logging
import numpy as np
from copy import deepcopy

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS, rdDistGeom
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Geometry import Point3D

from unidock_processing.unidocktools.vina_atom_type import AtomType
from unidock_processing.unidocktools.unidock_vina_atom_types import (
    VINA_ATOM_TYPE_DICT,
    VINA_VDW_RADII,
    build_type_property_arrays,
)
from unidock_processing.ligand_topology.generic_rotatable_bond import (
    GenericRotatableBond,
)
from unidock_processing.ligand_topology.utils import (
    assign_atom_properties,
    calculate_nonbonded_atom_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

# ── Input / output paths ─────────────────────────────────────────────────────
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
EP_SDF = os.path.join(INPUT_DIR, "EP.sdf")
TP_SDF = os.path.join(INPUT_DIR, "TP.sdf")
PROTAC_SDF = os.path.join(INPUT_DIR, "protac.sdf")
VHL_PDB = os.path.join(INPUT_DIR, "VHL.pdb")
RECEPTOR_PDB = os.path.join(INPUT_DIR, "receptor.pdb")
OUTPUT_JSON = os.path.join(INPUT_DIR, "protac_system.json")  # legacy
OUTPUT_LIGAND_DMS = os.path.join(INPUT_DIR, "protac_ligand.dms")
OUTPUT_RECEPTOR_DMS = os.path.join(INPUT_DIR, "protac_receptor.dms")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MCS matching + constrained embedding
# ═══════════════════════════════════════════════════════════════════════════════

def find_mcs_mapping(mol, ref):
    """Return (mol_match, ref_match) tuples of atom indices in the full
    (with-H) molecules that correspond via the heavy-atom MCS."""
    mol_noH = Chem.RemoveHs(mol)
    ref_noH = Chem.RemoveHs(ref)

    mcs = rdFMCS.FindMCS(
        [mol_noH, ref_noH],
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        matchChiralTag=False,
        timeout=100,
    )
    logger.info("MCS: %d atoms  SMARTS: %s", mcs.numAtoms, mcs.smartsString)
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

    # Match the SMARTS (heavy-atom only) against the full molecules
    mol_match = mol.GetSubstructMatch(mcs_mol)
    ref_match = ref.GetSubstructMatch(mcs_mol)
    if not mol_match or not ref_match:
        raise ValueError(
            f"MCS substructure match failed "
            f"(mol={len(mol_match)}, ref={len(ref_match)})"
        )
    logger.info("  mol_match: %d atoms   ref_match: %d atoms",
                len(mol_match), len(ref_match))
    return tuple(mol_match), tuple(ref_match)


def _build_coord_map(mol_match, ref, ref_match):
    """Build {mol_atom_idx: Point3D} from the reference conformer."""
    ref_conf = ref.GetConformer()
    coord_map = {}
    for i, mol_idx in enumerate(mol_match):
        ref_idx = ref_match[i]
        p = ref_conf.GetAtomPosition(ref_idx)
        coord_map[mol_idx] = Point3D(p.x, p.y, p.z)
    return coord_map


def _ff_optimize_with_fixed(mol, fixed_indices, max_iters=500):
    """Run FF minimization with *fixed_indices* atoms held in place."""
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    if mp is not None:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    else:
        ff = AllChem.UFFGetMoleculeForceField(mol)
    if ff is None:
        return
    for idx in fixed_indices:
        ff.AddFixedPoint(idx)
    ff.Minimize(maxIts=max_iters)


def constrained_embed(mol, coord_map, orig_conf, seed=42):
    """Generate a new 3-D conformer for *mol* with atoms in *coord_map*
    pinned to the given positions (distance-geometry + force-field).

    *orig_conf* supplies reference distances for saturated-ring puckering
    preservation.
    """
    mol = Chem.RWMol(deepcopy(mol))
    mol.RemoveAllConformers()

    # Lock saturated ring puckering via bounds matrix
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    sat_rings = _find_saturated_rings(mol, set(coord_map.keys()))
    ring_tol = 0.05
    for ring in sat_rings:
        logger.info("  Locking saturated ring puckering: %s", ring)
        for i in ring:
            for j in ring:
                if i < j:
                    d = orig_conf.GetAtomPosition(i).Distance(
                        orig_conf.GetAtomPosition(j)
                    )
                    bm[j][i] = d - ring_tol
                    bm[i][j] = d + ring_tol

    params = rdDistGeom.EmbedParameters()
    params.SetBoundsMat(bm)
    params.SetCoordMap(coord_map)
    params.randomSeed = seed
    params.useRandomCoords = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    params.ignoreSmoothingFailures = True

    ci = rdDistGeom.EmbedMolecule(mol, params)
    if ci < 0:
        raise RuntimeError("EmbedMolecule failed – try adjusting seed or constraints")

    # Transplant ring atoms via per-ring Kabsch to preserve exact puckering
    conf = mol.GetConformer()
    ring_pos = {}
    for ring in sat_rings:
        dg_r = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in ring])
        ref_r = np.array([[orig_conf.GetAtomPosition(i).x,
                            orig_conf.GetAtomPosition(i).y,
                            orig_conf.GetAtomPosition(i).z] for i in ring])
        cP = ref_r.mean(axis=0); cQ = dg_r.mean(axis=0)
        H = (ref_r - cP).T @ (dg_r - cQ)
        U, _, Vt = np.linalg.svd(H)
        det = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1.0, 1.0, np.sign(det)])
        R = Vt.T @ S @ U.T
        t = cQ - R @ cP
        transplanted = (R @ ref_r.T).T + t
        for k, idx in enumerate(ring):
            ring_pos[idx] = Point3D(*transplanted[k].tolist())
            conf.SetAtomPosition(idx, ring_pos[idx])

    # Set constrained atoms to exact target positions
    for idx, pt in coord_map.items():
        conf.SetAtomPosition(idx, pt)

    # FF-optimize with constrained + ring atoms fixed
    fixed = set(coord_map.keys()) | set(ring_pos.keys())
    _ff_optimize_with_fixed(mol, fixed)

    return mol.GetMol()


def _find_saturated_rings(mol, exclude_atoms):
    """Return list of ring-atom-index lists for saturated rings not fully
    contained in *exclude_atoms* (EP or TP)."""
    ri = mol.GetRingInfo()
    rings = []
    for ring in ri.AtomRings():
        ring_set = set(ring)
        if ring_set <= exclude_atoms:
            continue  # fully inside EP or TP – already constrained
        # Check if all ring bonds are single (saturated)
        is_sat = True
        for k in range(len(ring)):
            bond = mol.GetBondBetweenAtoms(ring[k], ring[(k + 1) % len(ring)])
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                is_sat = False
                break
        if is_sat:
            rings.append(list(ring))
    return rings


def constrained_embed_with_rigid_group(mol, coord_map, rigid_atoms, ref_conf,
                                       seed=42):
    """Embed *mol* with *coord_map* position constraints **and** rigid-body
    distance constraints on *rigid_atoms*.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule to embed (with H).
    coord_map : dict
        ``{atom_idx: Point3D}`` for atoms that must be pinned (TP).
    rigid_atoms : list[int]
        Atom indices whose pairwise distances are locked so they move
        as a rigid body (EP).
    ref_conf : RDKit Conformer
        Conformer that supplies the reference distances between
        *rigid_atoms* (the step-1 result).
    seed : int
        Random seed for distance geometry.

    Returns
    -------
    RDKit Mol with a single conformer.
    """
    mol = Chem.RWMol(deepcopy(mol))
    mol.RemoveAllConformers()

    # ── bounds matrix with tight EP pairwise distance constraints ─────
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    tol = 0.01  # Å tolerance on each pairwise distance
    for i in rigid_atoms:
        for j in rigid_atoms:
            if i < j:
                d = ref_conf.GetAtomPosition(i).Distance(
                    ref_conf.GetAtomPosition(j)
                )
                bm[j][i] = d - tol   # lower bound  (row > col)
                bm[i][j] = d + tol   # upper bound  (row < col)

    # ── preserve puckering of saturated rings in the linker ───────────
    exclude = set(rigid_atoms) | set(coord_map.keys())
    sat_rings = _find_saturated_rings(mol, exclude)
    ring_tol = 0.05  # slightly looser than EP rigid-body
    for ring in sat_rings:
        logger.info("  Locking saturated ring puckering: %s", ring)
        for i in ring:
            for j in ring:
                if i < j:
                    d = ref_conf.GetAtomPosition(i).Distance(
                        ref_conf.GetAtomPosition(j)
                    )
                    bm[j][i] = d - ring_tol
                    bm[i][j] = d + ring_tol

    # ── embed with bounds + coordMap ──────────────────────────────────
    params = rdDistGeom.EmbedParameters()
    params.SetBoundsMat(bm)
    params.SetCoordMap(coord_map)
    params.randomSeed = seed
    params.useRandomCoords = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    params.ignoreSmoothingFailures = True

    ci = rdDistGeom.EmbedMolecule(mol, params)
    if ci < 0:
        raise RuntimeError(
            "EmbedMolecule failed with rigid-group constraints – "
            "try adjusting seed or tolerance"
        )

    # Transplant exact step-1 EP positions via rigid-body alignment.
    # DG satisfies pairwise distances only approximately; instead of using
    # the DG-produced positions, we find where DG *placed* the rigid group
    # and move the exact ref_conf coordinates there.
    conf = mol.GetConformer()
    dg_pts = np.array([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in rigid_atoms])
    ref_pts = np.array([[ref_conf.GetAtomPosition(i).x,
                          ref_conf.GetAtomPosition(i).y,
                          ref_conf.GetAtomPosition(i).z]
                         for i in rigid_atoms])
    # Kabsch: ref_pts → dg_pts
    cP = ref_pts.mean(axis=0)
    cQ = dg_pts.mean(axis=0)
    H = (ref_pts - cP).T @ (dg_pts - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])
    R_snap = Vt.T @ S @ U.T
    t_snap = cQ - R_snap @ cP

    transplanted = (R_snap @ ref_pts.T).T + t_snap
    rigid_pos = {}
    for k, idx in enumerate(rigid_atoms):
        rigid_pos[idx] = Point3D(*transplanted[k].tolist())
        conf.SetAtomPosition(idx, rigid_pos[idx])

    # Transplant saturated ring atoms via per-ring Kabsch (same idea as EP).
    ring_pos = {}
    for ring in sat_rings:
        dg_r = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in ring])
        ref_r = np.array([[ref_conf.GetAtomPosition(i).x,
                            ref_conf.GetAtomPosition(i).y,
                            ref_conf.GetAtomPosition(i).z] for i in ring])
        cP_r = ref_r.mean(axis=0); cQ_r = dg_r.mean(axis=0)
        H_r = (ref_r - cP_r).T @ (dg_r - cQ_r)
        U_r, _, Vt_r = np.linalg.svd(H_r)
        d_r = np.linalg.det(Vt_r.T @ U_r.T)
        S_r = np.diag([1.0, 1.0, np.sign(d_r)])
        R_r = Vt_r.T @ S_r @ U_r.T
        t_r = cQ_r - R_r @ cP_r
        transplanted_r = (R_r @ ref_r.T).T + t_r
        for k, idx in enumerate(ring):
            ring_pos[idx] = Point3D(*transplanted_r[k].tolist())
            conf.SetAtomPosition(idx, ring_pos[idx])

    # Set TP atoms to exact target positions as well
    for idx, pt in coord_map.items():
        conf.SetAtomPosition(idx, pt)

    # FF-optimize with EP + TP + ring atoms fixed so neighbors relax
    fixed = set(rigid_atoms) | set(coord_map.keys()) | set(ring_pos.keys())
    _ff_optimize_with_fixed(mol, fixed)

    return mol.GetMol()


def _get_coords(mol, atom_indices):
    """Return (N, 3) array of atom coordinates for the given indices."""
    conf = mol.GetConformer()
    return np.array([
        [conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in atom_indices
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Kabsch rigid-body transform
# ═══════════════════════════════════════════════════════════════════════════════

def kabsch_transform(P, Q):
    """Return (R, t) such that  Q ≈ R @ P + t  (least-squares).

    P, Q : (N, 3)  corresponding point sets.
    """
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ S @ U.T
    t = cQ - R @ cP
    return R, t


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Torsion-tree builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_torsion_tree(mol, ep_atom_set, tp_atom_set):
    """Build the PROTAC torsion tree.

    * Root fragment = fragment containing TP MCS atoms
    * EP internal rotatable bonds are frozen
    * Returns (torsion_info_list, root_atom_idx_list, fragment_atom_idx_list)
    """
    # ── rotatable bonds ──────────────────────────────────────────────
    rot_finder = GenericRotatableBond()
    all_rot_bonds = rot_finder.identify_rotatable_bonds(mol)
    logger.info("Rotatable bonds (total): %d", len(all_rot_bonds))

    # Freeze EP-internal bonds
    active_bonds = []
    for a, b in all_rot_bonds:
        if a in ep_atom_set and b in ep_atom_set:
            logger.info("  Freezing EP-internal bond %d – %d", a, b)
        else:
            active_bonds.append((a, b))
    logger.info("Active bonds (after EP freeze): %d", len(active_bonds))

    # Also freeze TP-internal bonds (TP is root, no torsions inside root)
    linker_bonds = []
    for a, b in active_bonds:
        if a in tp_atom_set and b in tp_atom_set:
            logger.info("  Freezing TP-internal bond %d – %d", a, b)
        else:
            linker_bonds.append((a, b))
    logger.info("Linker bonds (after TP freeze): %d", len(linker_bonds))

    # ── fragment at active rotatable bonds ───────────────────────────
    bond_idx_list = []
    for bond in mol.GetBonds():
        bi = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bi_r = (bi[1], bi[0])
        if bi in linker_bonds or bi_r in linker_bonds:
            bond_idx_list.append(bond.GetIdx())

    if bond_idx_list:
        split_mol = FragmentOnBonds(mol, bond_idx_list, addDummies=False)
        frag_mols = list(GetMolFrags(split_mol, asMols=True, sanitizeFrags=False))
    else:
        frag_mols = [mol]
    logger.info("Fragments: %d", len(frag_mols))

    # ── root fragment (contains TP atoms) ────────────────────────────
    root_frag_idx = None
    for fi, frag in enumerate(frag_mols):
        for atom in frag.GetAtoms():
            if atom.GetIntProp("internal_atom_idx") in tp_atom_set:
                root_frag_idx = fi
                break
        if root_frag_idx is not None:
            break
    if root_frag_idx is None:
        raise ValueError("No fragment contains TP atoms")

    root_atom_idx_list = [
        atom.GetIntProp("internal_atom_idx")
        for atom in frag_mols[root_frag_idx].GetAtoms()
    ]
    logger.info("Root (TP) fragment: %d atoms", len(root_atom_idx_list))

    # ── fragment atom idx ────────────────────────────────────────────
    fragment_atom_idx_list = []
    for frag in frag_mols:
        fragment_atom_idx_list.append([
            atom.GetIntProp("internal_atom_idx") for atom in frag.GetAtoms()
        ])

    # ── torsion info per active bond ─────────────────────────────────
    root_atom = root_atom_idx_list[0]
    conf = mol.GetConformer()
    torsion_info_list = []

    for a, b in linker_bonds:
        # pick dihedral atoms i-a-b-l
        atom_a = mol.GetAtomWithIdx(a)
        atom_b = mol.GetAtomWithIdx(b)
        i_atom = next((n.GetIdx() for n in atom_a.GetNeighbors() if n.GetIdx() != b), None)
        l_atom = next((n.GetIdx() for n in atom_b.GetNeighbors() if n.GetIdx() != a), None)
        if i_atom is None or l_atom is None:
            continue

        # mobile atoms (side away from root)
        bond_obj = mol.GetBondBetweenAtoms(a, b)
        sp = FragmentOnBonds(mol, [bond_obj.GetIdx()], addDummies=False)
        sp_frags = list(GetMolFrags(sp, asMols=True, sanitizeFrags=False))
        mobile_frag = None
        for sf in sp_frags:
            if not any(
                at.GetIntProp("internal_atom_idx") == root_atom
                for at in sf.GetAtoms()
            ):
                mobile_frag = sf
                break
        if mobile_frag is None:
            mobile_frag = sp_frags[-1]

        mobile_atoms = sorted(
            at.GetIntProp("internal_atom_idx") for at in mobile_frag.GetAtoms()
        )

        # orient dihedral: first atom on static (root) side
        dihedral = [i_atom, a, b, l_atom]
        if i_atom in mobile_atoms:
            dihedral = list(reversed(dihedral))

        dihedral_val = float(AllChem.GetDihedralDeg(conf, *dihedral))

        torsion_info_list.append([
            dihedral,              # [i, a, b, l]
            dihedral_val,          # current angle
            [[-180.0, 180.0]],     # allowed ranges (full)
            mobile_atoms,          # atom indices that rotate
            [],                    # FF parameters (none)
        ])

    logger.info("Torsions built: %d", len(torsion_info_list))
    return torsion_info_list, root_atom_idx_list, fragment_atom_idx_list


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Atom typing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_gasteiger(mol):
    """Compute Gasteiger charges, replacing NaN with 0."""
    try:
        ComputeGasteigerCharges(mol)
    except Exception:
        pass
    for atom in mol.GetAtoms():
        try:
            q = atom.GetDoubleProp("_GasteigerCharge")
            if np.isnan(q) or np.isinf(q):
                atom.SetDoubleProp("_GasteigerCharge", 0.0)
        except KeyError:
            atom.SetDoubleProp("_GasteigerCharge", 0.0)


def type_molecule(mol):
    """Assign Vina atom types + Gasteiger charges + coordinate properties."""
    typer = AtomType()
    typer.assign_atom_types(mol)
    _safe_gasteiger(mol)
    assign_atom_properties(mol)


def build_ligand_atom_list(mol, with_pairs=True, strip_nonpolar_h=False):
    """Return list of 8-element atom-info arrays for the ligand JSON.

    If *with_pairs* is True, compute proper 1-2/1-3 and 1-4 bonded-pair
    lists from the molecular graph.  Otherwise use empty lists (suitable
    for protein atoms that rely on fragment-based exclusion).

    If *strip_nonpolar_h* is True, atoms with Vina type 0 (non-polar H)
    are omitted.  Safe only when the atom list has no index-dependent
    cross-references (i.e. VHL protein atoms).
    """
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    if with_pairs:
        pairs_12_13, pairs_14 = calculate_nonbonded_atom_pairs(mol)
    else:
        pairs_12_13 = [[] for _ in range(n)]
        pairs_14 = [[] for _ in range(n)]

    atoms = []
    for idx in range(n):
        atom = mol.GetAtomWithIdx(idx)
        p = conf.GetAtomPosition(idx)
        vt = atom.GetProp("vina_atom_type") if atom.HasProp("vina_atom_type") else "H"
        vina_int = VINA_ATOM_TYPE_DICT.get(vt, 0)
        if strip_nonpolar_h and vina_int == 0:
            continue
        q = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if np.isnan(q) or np.isinf(q):
            q = 0.0
        atoms.append([p.x, p.y, p.z, vina_int, 0, q,
                       pairs_12_13[idx], pairs_14[idx]])
    return atoms


def build_receptor_atom_list(mol):
    """Return list of 6-element atom-info arrays for the receptor JSON.

    Non-polar hydrogens (Vina type 0) are skipped — they contribute
    nothing to the Vina scoring function.
    """
    conf = mol.GetConformer()
    atoms = []
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        p = conf.GetAtomPosition(idx)
        vt = atom.GetProp("vina_atom_type") if atom.HasProp("vina_atom_type") else "H"
        vina_int = VINA_ATOM_TYPE_DICT.get(vt, 0)
        if vina_int == 0:
            continue
        q = atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else 0.0
        if np.isnan(q) or np.isinf(q):
            q = 0.0
        atoms.append([p.x, p.y, p.z, vina_int, 0, q])
    return atoms


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VHL merge into ligand
# ═══════════════════════════════════════════════════════════════════════════════

def merge_vhl_into_ligand(protac_atoms, torsions, fragments,
                          vhl_atoms, ep_atom_set):
    """Append VHL atoms to the ligand.

    * VHL indices are added to mobile_atoms of every torsion whose
      current mobile set overlaps with *ep_atom_set*.
    * The EP-containing fragment in *fragments* is extended with VHL indices.
    """
    n_protac = len(protac_atoms)
    vhl_indices = list(range(n_protac, n_protac + len(vhl_atoms)))

    combined_atoms = protac_atoms + vhl_atoms

    # Update torsion mobile atoms
    updated_torsions = []
    for tor in torsions:
        dih, val, rng, mobile, ff = tor
        if set(mobile) & ep_atom_set:
            updated_torsions.append([dih, val, rng, mobile + vhl_indices, ff])
        else:
            updated_torsions.append(tor)

    # Update fragment list – add VHL to the EP fragment
    updated_frags = []
    for frag in fragments:
        if set(frag) & ep_atom_set:
            updated_frags.append(frag + vhl_indices)
        else:
            updated_frags.append(frag)

    return combined_atoms, updated_torsions, updated_frags


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DMS assembly (replaces JSON)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_msys_system(mol, kept_indices=None):
    """Build an msys System from an RDKit mol.

    If kept_indices is provided, only include those atom indices.
    Returns (system, old_to_new_index_map).
    """
    import msys

    sys_mol = msys.CreateSystem()
    ct = sys_mol.addCt()
    chain = ct.addChain()
    res = chain.addResidue()
    res.name = 'LIG'

    conf = mol.GetConformer()
    old_to_new = {}

    if kept_indices is None:
        kept_indices = list(range(mol.GetNumAtoms()))

    for new_idx, old_idx in enumerate(kept_indices):
        atom = mol.GetAtomWithIdx(old_idx)
        a = res.addAtom()
        a.atomic_number = atom.GetAtomicNum()
        a.name = atom.GetSymbol() + str(new_idx)
        pos = conf.GetAtomPosition(old_idx)
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
        old_to_new[old_idx] = new_idx

    # Add bonds between kept atoms
    kek_mol = deepcopy(mol)
    try:
        Chem.Kekulize(kek_mol, clearAromaticFlags=False)
    except Exception:
        kek_mol = mol

    for bond in kek_mol.GetBonds():
        bi = bond.GetBeginAtomIdx()
        ei = bond.GetEndAtomIdx()
        if bi in old_to_new and ei in old_to_new:
            b = sys_mol.atom(old_to_new[bi]).addBond(
                sys_mol.atom(old_to_new[ei]))
            b.order = int(bond.GetBondTypeAsDouble())

    return sys_mol, old_to_new


def _add_empty_ff_tables(conn):
    """Add empty FF tables so the C++ engine reader doesn't crash."""
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
        "CREATE TABLE IF NOT EXISTS dihedral_trig_param "
        "(id INTEGER PRIMARY KEY, phi0 REAL, "
        "fc0 REAL, fc1 REAL, fc2 REAL, fc3 REAL, fc4 REAL, fc5 REAL, fc6 REAL)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS dihedral_trig_term "
        "(p0 INTEGER, p1 INTEGER, p2 INTEGER, p3 INTEGER, param INTEGER)"
    )
    # nbtype column needed for FF JOINs
    try:
        cur.execute(
            "ALTER TABLE particle ADD COLUMN nbtype INTEGER DEFAULT 0")
    except Exception:
        pass
    conn.commit()


def build_ligand_dms(
    protac_mol, vhl_mol, combined_atoms, combined_torsions,
    root_atoms, output_path, ep_atom_indices=None, e3_resname="VHL",
):
    """Build an enriched ligand DMS (PROTAC + VHL merged).

    Parameters
    ----------
    protac_mol : RDKit Mol
        Typed PROTAC molecule (all atoms including H).
    vhl_mol : RDKit Mol
        Typed VHL molecule (all atoms including H).
    combined_atoms : list
        Merged atom info list [x,y,z,vina_type,ff_type,charge,pairs12_13,pairs14].
        First n_protac entries are PROTAC (all atoms), rest are VHL (stripped).
    combined_torsions : list
        Torsion info list for the merged ligand.
    root_atoms : list
        Root fragment atom indices.
    output_path : str
        Where to write the DMS file.
    ep_atom_indices : list[int] or None
        PROTAC atom indices that correspond to EP pharmacophore atoms.
        Stored in DMS for post-processing VHL reconstruction.
    """
    import msys

    n_protac = protac_mol.GetNumAtoms()

    # VHL kept indices (non-polar H stripped)
    vhl_kept = []
    for i in range(vhl_mol.GetNumAtoms()):
        atom = vhl_mol.GetAtomWithIdx(i)
        vt = atom.GetProp("vina_atom_type") if atom.HasProp("vina_atom_type") else "H"
        vina_int = VINA_ATOM_TYPE_DICT.get(vt, 0)
        if vina_int != 0:
            vhl_kept.append(i)

    # Build PROTAC msys system (all atoms)
    protac_sys, protac_map = _build_msys_system(protac_mol)

    # Build VHL msys system (kept atoms only)
    vhl_sys, vhl_map = _build_msys_system(vhl_mol, kept_indices=vhl_kept)

    # Merge: append VHL atoms/bonds to PROTAC system
    merged = msys.CreateSystem()
    ct = merged.addCt()
    chain_p = ct.addChain()
    res_p = chain_p.addResidue()
    res_p.name = 'PRO'

    # Copy PROTAC atoms
    for i in range(protac_sys.natoms):
        a_src = protac_sys.atom(i)
        a = res_p.addAtom()
        a.atomic_number = a_src.atomic_number
        a.name = a_src.name
        a.x = a_src.x
        a.y = a_src.y
        a.z = a_src.z
        a.charge = a_src.charge

    # Copy PROTAC bonds
    for b_src in protac_sys.bonds:
        a0 = b_src.first.id
        a1 = b_src.second.id
        b = merged.atom(a0).addBond(merged.atom(a1))
        b.order = b_src.order

    # Copy E3 atoms (offset by n_protac)
    chain_v = ct.addChain()
    res_v = chain_v.addResidue()
    res_v.name = e3_resname
    for i in range(vhl_sys.natoms):
        a_src = vhl_sys.atom(i)
        a = res_v.addAtom()
        a.atomic_number = a_src.atomic_number
        a.name = a_src.name
        a.x = a_src.x
        a.y = a_src.y
        a.z = a_src.z
        a.charge = a_src.charge

    # Copy VHL bonds (offset by n_protac)
    for b_src in vhl_sys.bonds:
        a0 = b_src.first.id + n_protac
        a1 = b_src.second.id + n_protac
        b = merged.atom(a0).addBond(merged.atom(a1))
        b.order = b_src.order

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    msys.SaveDMS(merged, output_path)

    # --- Enrich with ud2 tables via sqlite3 ---
    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    # ud2_atom_props
    cur.execute(
        "CREATE TABLE ud2_atom_props "
        "(atom_id INTEGER PRIMARY KEY, vina_type INTEGER, ff_type INTEGER)"
    )
    for atom_id, atom_info in enumerate(combined_atoms):
        cur.execute(
            "INSERT INTO ud2_atom_props VALUES (?, ?, ?)",
            (atom_id, atom_info[3], atom_info[4]),
        )

    # ud2_torsions
    cur.execute(
        "CREATE TABLE ud2_torsions "
        "(torsion_id INTEGER PRIMARY KEY, "
        "atom_i INTEGER, atom_j INTEGER, atom_k INTEGER, atom_l INTEGER, "
        "dihedral_value REAL, mobile_atoms TEXT)"
    )
    for tor_id, tor_info in enumerate(combined_torsions):
        atoms = tor_info[0]
        dihedral_value = tor_info[1]
        mobile_atoms_str = ','.join(str(a) for a in tor_info[3])
        cur.execute(
            "INSERT INTO ud2_torsions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tor_id, atoms[0], atoms[1], atoms[2], atoms[3],
             dihedral_value, mobile_atoms_str),
        )

    # ud2_torsion_ranges
    cur.execute(
        "CREATE TABLE ud2_torsion_ranges "
        "(torsion_id INTEGER, range_lo REAL, range_hi REAL)"
    )
    for tor_id, tor_info in enumerate(combined_torsions):
        for lo, hi in tor_info[2]:
            cur.execute(
                "INSERT INTO ud2_torsion_ranges VALUES (?, ?, ?)",
                (tor_id, lo, hi),
            )

    # ud2_root_atoms
    cur.execute("CREATE TABLE ud2_root_atoms (atom_id INTEGER)")
    for atom_id in root_atoms:
        cur.execute("INSERT INTO ud2_root_atoms VALUES (?)", (atom_id,))

    # ud2_metadata
    cur.execute(
        "CREATE TABLE ud2_metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    cur.execute(
        "INSERT INTO ud2_metadata VALUES ('molecule_name', 'MOL_0')"
    )
    cur.execute(
        "INSERT INTO ud2_metadata VALUES ('sdf_natom', ?)",
        (str(n_protac),),
    )

    # ud2_ep_mapping – EP atom indices in the PROTAC for VHL reconstruction
    cur.execute(
        "CREATE TABLE ud2_ep_mapping (protac_atom_id INTEGER)"
    )
    if ep_atom_indices is not None:
        for idx in ep_atom_indices:
            cur.execute(
                "INSERT INTO ud2_ep_mapping VALUES (?)", (idx,))


    # exclusion table (1-2/1-3 bonded pairs from PROTAC part)
    cur.execute("CREATE TABLE IF NOT EXISTS exclusion (p0 INTEGER, p1 INTEGER)")
    seen_excl = set()
    for atom_id, atom_info in enumerate(combined_atoms):
        if len(atom_info) > 6:
            for partner in atom_info[6]:  # pairs_12_13
                pair = (min(atom_id, partner), max(atom_id, partner))
                if pair not in seen_excl:
                    seen_excl.add(pair)
                    cur.execute(
                        "INSERT INTO exclusion VALUES (?, ?)", pair)

    # pair_12_6_es_term table (1-4 bonded pairs)
    cur.execute(
        "CREATE TABLE IF NOT EXISTS pair_12_6_es_term "
        "(p0 INTEGER, p1 INTEGER, param INTEGER)"
    )
    seen_14 = set()
    for atom_id, atom_info in enumerate(combined_atoms):
        if len(atom_info) > 7:
            for partner in atom_info[7]:  # pairs_14
                pair = (min(atom_id, partner), max(atom_id, partner))
                if pair not in seen_14:
                    seen_14.add(pair)
                    cur.execute(
                        "INSERT INTO pair_12_6_es_term VALUES (?, ?, 0)",
                        pair)

    # ud2_sdf_properties (empty for PROTAC)
    cur.execute(
        "CREATE TABLE ud2_sdf_properties (key TEXT, value TEXT)"
    )

    # ud2_formal_charges
    cur.execute(
        "CREATE TABLE ud2_formal_charges "
        "(atom_id INTEGER, formal_charge INTEGER)"
    )
    # PROTAC formal charges
    for i in range(n_protac):
        fc = protac_mol.GetAtomWithIdx(i).GetFormalCharge()
        if fc != 0:
            cur.execute(
                "INSERT INTO ud2_formal_charges VALUES (?, ?)", (i, fc))
    # VHL formal charges (offset by n_protac)
    for new_idx, old_idx in enumerate(vhl_kept):
        fc = vhl_mol.GetAtomWithIdx(old_idx).GetFormalCharge()
        if fc != 0:
            cur.execute(
                "INSERT INTO ud2_formal_charges VALUES (?, ?)",
                (n_protac + new_idx, fc))

    # ud2_bond_stereo
    cur.execute(
        "CREATE TABLE ud2_bond_stereo "
        "(bond_p0 INTEGER, bond_p1 INTEGER, stereo INTEGER)"
    )
    BONDDIR_TO_SDF = {1: 1, 6: 6}
    for bond in protac_mol.GetBonds():
        bdir = int(bond.GetBondDir())
        sdf_stereo = BONDDIR_TO_SDF.get(bdir, 0)
        if sdf_stereo != 0:
            cur.execute(
                "INSERT INTO ud2_bond_stereo VALUES (?, ?, ?)",
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), sdf_stereo))

    conn.commit()
    conn.close()

    # Add empty FF tables
    conn = sqlite3.connect(output_path)
    _add_empty_ff_tables(conn)
    conn.close()

    n_total = len(combined_atoms)
    logger.info(
        "Built ligand DMS: %s (%d atoms: %d PROTAC + %d VHL, "
        "%d torsions, %d exclusions, %d 1-4 pairs)",
        output_path, n_total, n_protac, n_total - n_protac,
        len(combined_torsions), len(seen_excl), len(seen_14),
    )
    return output_path


def build_receptor_dms(receptor_mol, output_path):
    """Build an enriched receptor DMS from an RDKit mol.

    Includes ALL atoms (including H) so the C++ engine can build
    RecDonorHTopo for angular HBond scoring.
    """
    import msys

    sys_mol, idx_map = _build_msys_system(receptor_mol)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    msys.SaveDMS(sys_mol, output_path)

    # Enrich with ud2_atom_props
    conn = sqlite3.connect(output_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE ud2_atom_props "
        "(atom_id INTEGER PRIMARY KEY, vina_type INTEGER, ff_type INTEGER)"
    )
    for i in range(receptor_mol.GetNumAtoms()):
        atom = receptor_mol.GetAtomWithIdx(i)
        vt = atom.GetProp("vina_atom_type") if atom.HasProp("vina_atom_type") else "H"
        vina_int = VINA_ATOM_TYPE_DICT.get(vt, 0)
        cur.execute(
            "INSERT INTO ud2_atom_props VALUES (?, ?, ?)",
            (i, vina_int, 0),
        )
    conn.commit()
    conn.close()

    # Add empty FF tables
    conn = sqlite3.connect(output_path)
    _add_empty_ff_tables(conn)
    conn.close()

    logger.info(
        "Built receptor DMS: %s (%d atoms)",
        output_path, receptor_mol.GetNumAtoms(),
    )
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── load input molecules ──────────────────────────────────────────
    protac = Chem.SDMolSupplier(PROTAC_SDF, removeHs=False)[0]
    ep = Chem.SDMolSupplier(EP_SDF, removeHs=False)[0]
    tp = Chem.SDMolSupplier(TP_SDF, removeHs=False)[0]
    vhl = Chem.MolFromPDBFile(VHL_PDB, removeHs=False)
    receptor = Chem.MolFromPDBFile(RECEPTOR_PDB, removeHs=False)

    for name, mol in [("protac", protac), ("EP", ep), ("TP", tp),
                       ("VHL", vhl), ("receptor", receptor)]:
        assert mol is not None, f"Failed to load {name}"
        logger.info("Loaded %-8s  %d atoms", name, mol.GetNumAtoms())

    # ── step 1 : constrained embed PROTAC → EP ───────────────────────
    logger.info("\n=== Step 1: Constrained embed PROTAC → EP ===")
    ep_mol_match, ep_ref_match = find_mcs_mapping(protac, ep)
    ep_coord_map = _build_coord_map(ep_mol_match, ep, ep_ref_match)
    protac_ep = constrained_embed(protac, ep_coord_map,
                                   orig_conf=protac.GetConformer(), seed=42)

    # "before" coords = step 1 EP positions in the PROTAC (VHL frame)
    ep_coords_before = _get_coords(protac_ep, ep_mol_match)

    # ── step 2 : constrained embed aligned PROTAC → TP (EP rigid) ────
    logger.info("\n=== Step 2: Constrained embed PROTAC → TP (EP rigid) ===")
    tp_mol_match, tp_ref_match = find_mcs_mapping(protac_ep, tp)
    tp_coord_map = _build_coord_map(tp_mol_match, tp, tp_ref_match)
    ep_ref_conf = protac_ep.GetConformer()  # step 1 conformer
    protac_final = constrained_embed_with_rigid_group(
        protac_ep, tp_coord_map,
        rigid_atoms=list(ep_mol_match),
        ref_conf=ep_ref_conf,
        seed=42,
    )

    # Dump intermediate PROTAC for inspection
    _w = Chem.SDWriter(os.path.join(INPUT_DIR, "protac_temp_cursor.sdf"))
    _w.write(protac_final)
    _w.close()
    logger.info("Saved protac_temp_cursor.sdf")

    # "after" coords = EP-part positions in PROTAC after TP embedding
    ep_coords_after = _get_coords(protac_final, ep_mol_match)

    # ── step 3 : Kabsch transform (EP.sdf → EP-in-protac_final) ──────
    logger.info("\n=== Step 3: Compute EP rigid-body transform ===")
    R, t = kabsch_transform(ep_coords_before, ep_coords_after)
    fitted = (R @ ep_coords_before.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((ep_coords_after - fitted) ** 2, axis=1)))
    logger.info("Kabsch RMSD = %.4f Å", rmsd)

    # ── step 4 : prepare PROTAC torsion tree ─────────────────────────
    logger.info("\n=== Step 4: Build PROTAC torsion tree ===")
    type_molecule(protac_final)

    ep_atom_set = set(ep_mol_match)
    tp_atom_set = set(tp_mol_match)

    torsions, root_atoms, fragments = build_torsion_tree(
        protac_final, ep_atom_set, tp_atom_set,
    )
    protac_atoms = build_ligand_atom_list(protac_final, with_pairs=True)

    # ── step 5 : transform VHL and merge ─────────────────────────────
    logger.info("\n=== Step 5: Transform VHL & merge into ligand ===")
    vhl_pos = vhl.GetConformer().GetPositions()
    vhl_new = (R @ vhl_pos.T).T + t
    vhl_conf = vhl.GetConformer()
    for i in range(vhl.GetNumAtoms()):
        vhl_conf.SetAtomPosition(i, Point3D(*vhl_new[i].tolist()))

    type_molecule(vhl)
    vhl_atoms = build_ligand_atom_list(vhl, with_pairs=False, strip_nonpolar_h=True)

    combined_atoms, combined_torsions, combined_frags = merge_vhl_into_ligand(
        protac_atoms, torsions, fragments, vhl_atoms, ep_atom_set,
    )

    # ── step 6 : target receptor ─────────────────────────────────────
    logger.info("\n=== Step 6: Type target receptor ===")
    type_molecule(receptor)

    # ── step 7 : build enriched DMS files ────────────────────────────
    logger.info("\n=== Step 7: Build enriched DMS ===")
    # Label the E3 residue after whatever PDB it came from, so a CRBN system
    # does not end up with its E3 recorded as "VHL". Residue names are 4
    # characters at most, and the whole point is that this is not always VHL.
    e3_resname = os.path.splitext(os.path.basename(VHL_PDB))[0].upper()[:4] or "E3"
    build_ligand_dms(
        protac_final, vhl, combined_atoms, combined_torsions,
        root_atoms, OUTPUT_LIGAND_DMS,
        ep_atom_indices=list(ep_mol_match),
        e3_resname=e3_resname,
    )
    build_receptor_dms(receptor, OUTPUT_RECEPTOR_DMS)

    # ── summary ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Receptor:             %d atoms", receptor.GetNumAtoms())
    logger.info("Ligand  (PROTAC+VHL): %d atoms  (%d PROTAC + %d VHL)",
                len(combined_atoms), len(protac_atoms), len(vhl_atoms))
    logger.info("Torsions:             %d", len(combined_torsions))
    logger.info("Root atoms (TP):      %d", len(root_atoms))
    logger.info("Fragments:            %d", len(combined_frags))
    logger.info("Ligand DMS:           %s", OUTPUT_LIGAND_DMS)
    logger.info("Receptor DMS:         %s", OUTPUT_RECEPTOR_DMS)

    return OUTPUT_LIGAND_DMS, OUTPUT_RECEPTOR_DMS


if __name__ == "__main__":
    main()
