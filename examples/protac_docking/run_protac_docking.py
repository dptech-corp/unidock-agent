#!/usr/bin/env python3
"""
run_protac_docking.py – End-to-end PROTAC ternary complex docking with Uni-Dock2.

1. Build enriched DMS  (protac_sampling.py)
2. Run multiple Uni-Dock2 docking jobs (different seeds) and merge poses
3. Post-process: reconstruct PROTAC SDF, E3 PDB, receptor PDB, full complex PDB
4. Cluster PROTAC poses by RMSD, save representative structures
"""
import os
import sys
import json
import time
import shutil
import logging
import argparse

import numpy as np
from copy import deepcopy
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
from scipy.cluster.hierarchy import linkage, fcluster

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ENGINE_API = os.path.join(REPO_ROOT, "unidock", "unidock_engine", "api", "python")

sys.path.insert(0, ENGINE_API)
try:
    from pipeline import run_docking_pipeline
except ImportError:  # no in-tree build: the module `pip install ./unidock/unidock_engine` installs
    from unidock_engine.api.python.pipeline import run_docking_pipeline


def compute_protac_cluster_indices(ligand_dms: str, input_dir: str) -> list[int]:
    """Return PROTAC heavy-atom indices (0-based) for C++ clustering RMSD.

    In the ligand DMS, ligand = PROTAC (all atoms) + VHL (non-polar H stripped).
    We want clustering to use only PROTAC heavy atoms (vina_type != 0).
    """
    import sqlite3

    # Get n_protac from the template SDF
    protac_temp = os.path.join(input_dir, "protac_temp_cursor.sdf")
    if not os.path.isfile(protac_temp):
        protac_temp = os.path.join(input_dir, "protac_temp.sdf")
    mol = Chem.SDMolSupplier(protac_temp, removeHs=False)[0]
    n_protac = mol.GetNumAtoms()

    # Read vina_types from DMS
    conn = sqlite3.connect(ligand_dms)
    rows = conn.execute(
        "SELECT atom_id, vina_type FROM ud2_atom_props "
        "WHERE atom_id < ? ORDER BY atom_id", (n_protac,)
    ).fetchall()
    conn.close()

    heavy_idx = [r[0] for r in rows if r[1] != 0]
    n_total = sqlite3.connect(ligand_dms).execute(
        "SELECT COUNT(*) FROM ud2_atom_props").fetchone()[0]
    logger.info("Cluster indices: %d PROTAC heavy atoms (of %d PROTAC, %d total ligand)",
                len(heavy_idx), n_protac, n_total)
    return heavy_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Input preparation  (calls protac_sampling.main)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_input(
    input_dir: str,
    ep_sdf: str = "EP.sdf",
    tp_sdf: str = "TP.sdf",
    protac_sdf: str = "protac.sdf",
    receptor_pdb: str = "receptor.pdb",
    e3_pdb: str = "VHL.pdb",
) -> tuple[str, str]:
    """Run protac_sampling.py and return (ligand_dms, receptor_dms) paths."""
    sys.path.insert(0, input_dir)
    import protac_sampling

    protac_sampling.INPUT_DIR = input_dir
    protac_sampling.EP_SDF = os.path.join(input_dir, ep_sdf)
    protac_sampling.TP_SDF = os.path.join(input_dir, tp_sdf)
    protac_sampling.PROTAC_SDF = os.path.join(input_dir, protac_sdf)
    protac_sampling.VHL_PDB = os.path.join(input_dir, e3_pdb)
    protac_sampling.RECEPTOR_PDB = os.path.join(input_dir, receptor_pdb)
    protac_sampling.OUTPUT_LIGAND_DMS = os.path.join(input_dir, "protac_ligand.dms")
    protac_sampling.OUTPUT_RECEPTOR_DMS = os.path.join(input_dir, "protac_receptor.dms")

    ligand_dms, receptor_dms = protac_sampling.main()
    return ligand_dms, receptor_dms


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Run docking (single job)
# ═══════════════════════════════════════════════════════════════════════════════

def run_docking(
    ligand_dms: str,
    receptor_dms: str,
    output_dir: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float] = (100.0, 100.0, 100.0),
    exhaustiveness: int = 128,
    mc_steps: int = 20,
    opt_steps: int = 15,
    refine_steps: int = 5,
    num_pose: int = 1000,
    energy_range: float = 100.0,
    seed: int = 42,
) -> str:
    """Run Uni-Dock2 and return the output SDF path."""
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    run_docking_pipeline(
        ligand_dms_path=os.path.abspath(ligand_dms),
        receptor_dms_path=os.path.abspath(receptor_dms),
        output_dir=os.path.abspath(output_dir),
        center_x=center[0],
        center_y=center[1],
        center_z=center[2],
        size_x=box_size[0],
        size_y=box_size[1],
        size_z=box_size[2],
        task="screen",
        search_mode="free",
        exhaustiveness=exhaustiveness,
        randomize=True,
        mc_steps=mc_steps,
        opt_steps=opt_steps,
        refine_steps=refine_steps,
        num_pose=num_pose,
        rmsd_limit=0.5,
        energy_range=energy_range,
        seed=seed,
        use_tor_lib=True,
        constraint_docking=True,
        gpu_device_id=0,
    )
    elapsed = time.time() - t0
    logger.info("Docking (seed=%d) completed in %.1fs", seed, elapsed)

    # Engine now writes SDF output
    sdf_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir) if f.endswith('.sdf')
    ]
    if not sdf_files:
        raise FileNotFoundError(f"No SDF output found in {output_dir}")
    return sdf_files[0]


def _run_docking_subprocess(
    ligand_dms: str,
    receptor_dms: str,
    output_dir: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int,
    mc_steps: int,
    opt_steps: int,
    refine_steps: int,
    num_pose: int,
    energy_range: float,
    seed: int,
    cluster_atom_indices: list[list[int]] | None = None,
) -> str:
    """Run a single docking job in a subprocess to avoid C++ library re-init crashes.

    Returns the path to the output SDF file.
    """
    import subprocess

    cluster_arg = repr(cluster_atom_indices) if cluster_atom_indices else "[]"
    script = f"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "{ENGINE_API}")
try:
    from pipeline import run_docking_pipeline
except ImportError:
    from unidock_engine.api.python.pipeline import run_docking_pipeline
os.makedirs("{output_dir}", exist_ok=True)
run_docking_pipeline(
    ligand_dms_path="{os.path.abspath(ligand_dms)}",
    receptor_dms_path="{os.path.abspath(receptor_dms)}",
    output_dir="{os.path.abspath(output_dir)}",
    center_x={center[0]}, center_y={center[1]}, center_z={center[2]},
    size_x={box_size[0]}, size_y={box_size[1]}, size_z={box_size[2]},
    task="screen", search_mode="free",
    exhaustiveness={exhaustiveness}, randomize=True,
    mc_steps={mc_steps}, opt_steps={opt_steps}, refine_steps={refine_steps},
    num_pose={num_pose}, rmsd_limit=0.5, energy_range={energy_range},
    seed={seed}, use_tor_lib=True, constraint_docking=True, gpu_device_id=0,
    cluster_atom_indices={cluster_arg},
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.error("Docking subprocess failed (seed=%d):\n%s", seed, result.stderr)
        raise RuntimeError(f"Docking failed for seed={seed}")

    # Find output SDF
    sdf_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir) if f.endswith('.sdf')
    ]
    if not sdf_files:
        raise FileNotFoundError(f"No SDF output found in {output_dir}")
    return sdf_files[0]


def rank_score(pose: dict) -> float:
    """The score merged poses are ranked and clustered by: vina_pose_score, the score the engine's pose pass orders
    each run with. vina_binding_free_energy is not comparable across poses: it subtracts the intra energy of the
    first pose of the pose's own run, so poses with a more negative intra than their run's top pose gain a bonus
    (on MZ1 it moved the 0.88 A pose from rank 1 to rank 6). Poses read from outputs without the tag fall back to
    vina_binding_free_energy."""
    score = pose.get("pose_score")
    return pose["energy"][0] if score is None else score


def _sdf_to_pose_dicts(sdf_path: str, seed: int = 0, run_idx: int = 0) -> list[dict]:
    """Read an SDF file and convert to pose dicts compatible with postprocessing.

    Uses a custom parser that reads coordinates from fixed-width V2000 atom lines.
    Works for both standard and PROTAC SDF files.

    Each pose dict has:
      - "coords": flat list of [x,y,z, x,y,z, ...] for PROTAC atoms
      - "energy": [binding_free, intra_inter, intra, inter, box_penalty, torsion]
      - "pose_score": vina_pose_score, the engine's pose-pass score (None when the SDF has no such tag)
      - "_seed", "_run": metadata
    """
    with open(sdf_path) as fh:
        content = fh.read()

    poses = []
    for record in content.split("$$$$"):
        record = record.strip()
        if not record:
            continue

        lines = record.split("\n")
        if len(lines) < 5:
            continue

        # Lines 0-2: header (name, program, comment)
        # Line 3: counts line (may have overflowed %3d fields)
        # Lines 4+: atom block — each atom line is ≥34 chars with %10.4f coords

        coords = []
        i = 4
        while i < len(lines):
            line = lines[i]
            if len(line) < 34:
                break
            try:
                x = float(line[0:10])
                y = float(line[10:20])
                z = float(line[20:30])
                coords.append(x)
                coords.append(y)
                coords.append(z)
                i += 1
            except ValueError:
                break

        if not coords:
            continue

        # Parse SDF properties (>  <key>\nvalue\n)
        props = {}
        while i < len(lines):
            line = lines[i]
            if line.startswith(">") and "<" in line:
                key = line.split("<")[1].split(">")[0]
                if i + 1 < len(lines):
                    props[key] = lines[i + 1].strip()
                    i += 2
                    continue
            i += 1

        def _prop(name, default=0.0):
            try:
                return float(props[name])
            except (KeyError, ValueError):
                return default

        energy = [
            _prop("vina_binding_free_energy"),
            _prop("vina_intra_inter"),
            _prop("vina_intra"),
            _prop("vina_inter"),
            _prop("vina_box_penalty"),
            _prop("vina_torsion_number_energy"),
        ]
        poses.append({
            "coords": coords,
            "energy": energy,
            "pose_score": _prop("vina_pose_score", default=None),
            "_seed": seed,
            "_run": run_idx,
        })
    return poses


def run_multi_seed_docking(
    ligand_dms: str,
    receptor_dms: str,
    base_output_dir: str,
    n_runs: int,
    seeds: list[int] | None = None,
    **dock_kwargs,
) -> list[dict]:
    """Run multiple docking jobs with different seeds (each as subprocess), merge all poses.

    Returns a list of pose dicts sorted by rank_score (best first).
    """
    if seeds is None:
        seeds = [42 + i for i in range(n_runs)]

    all_poses = []
    for run_idx, seed in enumerate(seeds):
        run_dir = os.path.join(base_output_dir, f"run_{run_idx:03d}")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)

        logger.info("── Run %d/%d  seed=%d ──", run_idx + 1, n_runs, seed)
        t0 = time.time()
        out_sdf = _run_docking_subprocess(
            ligand_dms, receptor_dms, run_dir, seed=seed, **dock_kwargs,
        )
        elapsed = time.time() - t0

        poses = _sdf_to_pose_dicts(out_sdf, seed=seed, run_idx=run_idx)
        logger.info("  Got %d poses in %.1fs", len(poses), elapsed)
        all_poses.extend(poses)

    # Sort by the pose score of each pose (see rank_score)
    all_poses.sort(key=rank_score)
    logger.info("Total merged poses: %d", len(all_poses))
    return all_poses


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Post-processing  (raw PDB lines — preserves chain/residue/CONECT for PyMOL)
# ═══════════════════════════════════════════════════════════════════════════════

def _read_pdb_lines(pdb_path: str) -> list[str]:
    with open(pdb_path) as fh:
        return [line.rstrip("\n") for line in fh]


def _parse_pdb_coords(pdb_lines: list[str]) -> tuple[list[int], np.ndarray]:
    line_indices = []
    coords = []
    for i, line in enumerate(pdb_lines):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
            line_indices.append(i)
    return line_indices, np.array(coords, dtype=np.float64)


def _update_pdb_coord(line: str, x: float, y: float, z: float) -> str:
    return f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"


def _pdb_atom_lines(pdb_lines: list[str]) -> list[str]:
    out = []
    for line in pdb_lines:
        if line.startswith("END") and not line.startswith("ENDMDL"):
            continue
        out.append(line)
    return out


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ S @ U.T
    t = cQ - R @ cP
    return R, t


def _build_vhl_kept_indices(vhl_mol):
    from unidock_processing.unidocktools.vina_atom_type import AtomType
    from unidock_processing.unidocktools.unidock_vina_atom_types import (
        VINA_ATOM_TYPE_DICT,
    )
    typer = AtomType()
    typer.assign_atom_types(vhl_mol)
    kept = []
    for i in range(vhl_mol.GetNumAtoms()):
        atom = vhl_mol.GetAtomWithIdx(i)
        vt = atom.GetProp("vina_atom_type") if atom.HasProp("vina_atom_type") else "H"
        vina_int = VINA_ATOM_TYPE_DICT.get(vt, 0)
        if vina_int != 0:
            kept.append(i)
    return kept


def _rdkit_idx_to_pdb_serial(vhl_mol) -> dict[int, int]:
    mapping = {}
    for i in range(vhl_mol.GetNumAtoms()):
        info = vhl_mol.GetAtomWithIdx(i).GetPDBResidueInfo()
        if info:
            mapping[i] = info.GetSerialNumber()
    return mapping


class PoseReconstructor:
    """Reconstructs molecular files from docking output coordinates.

    Precomputes all E3-ligase mapping data once, then can be called per-pose.
    """

    def __init__(self, input_dir: str, ligand_dms_or_json: str,
                 e3_pdb: str = "VHL.pdb", receptor_pdb: str = "receptor.pdb"):
        # PROTAC template
        protac_temp = os.path.join(input_dir, "protac_temp_cursor.sdf")
        if not os.path.isfile(protac_temp):
            protac_temp = os.path.join(input_dir, "protac_temp.sdf")
        self.protac_template = Chem.SDMolSupplier(protac_temp, removeHs=False)[0]
        self.n_protac = self.protac_template.GetNumAtoms()

        # E3 ligase kept-atom mapping
        e3_mol = Chem.MolFromPDBFile(
            os.path.join(input_dir, e3_pdb), removeHs=False
        )
        kept_rdkit_indices = _build_vhl_kept_indices(deepcopy(e3_mol))
        self.n_vhl_kept = len(kept_rdkit_indices)

        # Get total ligand atom count from DMS or JSON
        if ligand_dms_or_json.endswith('.dms'):
            import sqlite3
            conn = sqlite3.connect(ligand_dms_or_json)
            n_lig_atoms = conn.execute(
                "SELECT COUNT(*) FROM ud2_atom_props"
            ).fetchone()[0]
            conn.close()
        else:
            with open(ligand_dms_or_json) as fh:
                inp = json.load(fh)
            n_lig_atoms = len(inp["MOL_0"]["atoms"])
        assert self.n_protac + self.n_vhl_kept == n_lig_atoms

        rdkit_to_serial = _rdkit_idx_to_pdb_serial(e3_mol)

        # Raw PDB lines
        self.vhl_pdb_lines = _read_pdb_lines(os.path.join(input_dir, e3_pdb))
        self.receptor_pdb_lines = _read_pdb_lines(os.path.join(input_dir, receptor_pdb))
        self.vhl_atom_line_idx, self.vhl_all_coords = _parse_pdb_coords(
            self.vhl_pdb_lines
        )

        serial_to_pdb_line_pos = {}
        for pos, li in enumerate(self.vhl_atom_line_idx):
            line = self.vhl_pdb_lines[li]
            serial = int(line[6:11].strip())
            serial_to_pdb_line_pos[serial] = pos

        kept_pdb_positions = []
        for rdkit_idx in kept_rdkit_indices:
            serial = rdkit_to_serial[rdkit_idx]
            kept_pdb_positions.append(serial_to_pdb_line_pos[serial])

        self.vhl_kept_pdb_coords = self.vhl_all_coords[kept_pdb_positions]
        self.receptor_body = _pdb_atom_lines(self.receptor_pdb_lines)

        # EP-based VHL reconstruction: load EP atom indices + pre-docking coords
        self.ep_indices = None
        if ligand_dms_or_json.endswith('.dms'):
            import sqlite3
            conn = sqlite3.connect(ligand_dms_or_json)
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if 'ud2_ep_mapping' in tables:
                rows = conn.execute(
                    "SELECT protac_atom_id FROM ud2_ep_mapping ORDER BY protac_atom_id"
                ).fetchall()
                if rows:
                    self.ep_indices = [r[0] for r in rows]
                    # VHL pre-docking kept coords from particle table
                    sdf_natom_row = conn.execute(
                        "SELECT value FROM ud2_metadata WHERE key = 'sdf_natom'"
                    ).fetchone()
                    if sdf_natom_row:
                        sdf_natom = int(sdf_natom_row[0])
                        vhl_rows = conn.execute(
                            "SELECT x, y, z FROM particle WHERE id >= ? ORDER BY id",
                            (sdf_natom,)
                        ).fetchall()
                        self.vhl_pre_kept_coords = np.array(
                            [[r[0], r[1], r[2]] for r in vhl_rows], dtype=np.float64
                        )
                    # EP pre-docking coords from PROTAC template conformer
                    conf = self.protac_template.GetConformer()
                    self.ep_pre_coords = np.array([
                        [conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in self.ep_indices
                    ], dtype=np.float64)
            conn.close()

    def build_protac_mol(self, coords: np.ndarray, energy: list,
                         pose_idx: int, pose_score: float | None = None) -> Chem.Mol:
        mol = deepcopy(self.protac_template)
        conf = mol.GetConformer()
        for i in range(self.n_protac):
            conf.SetAtomPosition(i, Point3D(*coords[i].tolist()))
        mol.SetProp("_Name", f"PROTAC_pose_{pose_idx + 1}")
        mol.SetProp("pose_rank", str(pose_idx + 1))
        mol.SetProp("vina_binding_free_energy", f"{energy[0]:.3f}")
        mol.SetProp("vina_intra_inter", f"{energy[1]:.3f}")
        mol.SetProp("vina_intra", f"{energy[2]:.3f}")
        mol.SetProp("vina_inter", f"{energy[3]:.3f}")
        mol.SetProp("vina_box_penalty", f"{energy[4]:.3f}")
        mol.SetProp("vina_torsion_energy", f"{energy[5]:.3f}")
        if pose_score is not None:
            mol.SetProp("vina_pose_score", f"{pose_score:.3f}")
        return mol

    def build_vhl_pdb_lines(self, coords: np.ndarray) -> list[str]:
        if self.ep_indices is not None:
            # EP-based reconstruction: SDF has only PROTAC atoms.
            # Compute rigid-body transform from EP pre→post docking,
            # then apply to VHL pre-docking coords.
            ep_docked = coords[self.ep_indices]
            R_dock, t_dock = _kabsch(self.ep_pre_coords, ep_docked)
            vhl_docked_kept = (R_dock @ self.vhl_pre_kept_coords.T).T + t_dock
        else:
            # Legacy path: SDF has PROTAC+VHL atoms, extract VHL directly
            vhl_docked_kept = coords[self.n_protac:]

        R, t = _kabsch(self.vhl_kept_pdb_coords, vhl_docked_kept)
        vhl_transformed = (R @ self.vhl_all_coords.T).T + t

        vhl_pose_lines = list(self.vhl_pdb_lines)
        for pos, li in enumerate(self.vhl_atom_line_idx):
            x, y, z = vhl_transformed[pos]
            vhl_pose_lines[li] = _update_pdb_coord(vhl_pose_lines[li], x, y, z)
        return _pdb_atom_lines(vhl_pose_lines)


def postprocess(
    poses: list[dict],
    input_dir: str,
    poses_dir: str,
    input_json: str,
    e3_pdb: str = "VHL.pdb",
    receptor_pdb: str = "receptor.pdb",
    dump_all: bool = False,
) -> list[Chem.Mol]:
    """Reconstruct molecular files from docking poses.

    Returns list of PROTAC RDKit Mol objects (for clustering).

    Reconstruction always happens — the returned mols are what clustering runs
    on. `dump_all` only controls whether every pose is *also* written to disk.
    Nothing downstream reads those files: the representatives are built from the
    in-memory list, so leaving them off changes no result.
    """
    os.makedirs(poses_dir, exist_ok=True)
    n_poses = len(poses)
    logger.info("Post-processing %d poses", n_poses)

    recon = PoseReconstructor(input_dir, input_json,
                              e3_pdb=e3_pdb, receptor_pdb=receptor_pdb)

    # Copy receptor
    receptor_name = os.path.splitext(receptor_pdb)[0].lower()
    receptor_out_path = os.path.join(poses_dir, f"receptor_{receptor_name}.pdb")
    shutil.copy2(os.path.join(input_dir, receptor_pdb), receptor_out_path)

    e3_name = os.path.splitext(e3_pdb)[0].lower()
    protac_sdf_path = os.path.join(poses_dir, "docked_protac_all.sdf")
    vhl_pdb_path = os.path.join(poses_dir, f"docked_{e3_name}_all.pdb")
    complex_pdb_path = os.path.join(poses_dir, "docked_complex_all.pdb")

    sdf_writer = Chem.SDWriter(protac_sdf_path) if dump_all else None
    vhl_pdb_fh = open(vhl_pdb_path, "w") if dump_all else None
    complex_pdb_fh = open(complex_pdb_path, "w") if dump_all else None

    protac_mols = []

    for pose_idx, pose in enumerate(poses):
        coords = np.array(pose["coords"]).reshape(-1, 3)
        energy = pose["energy"]

        # Always needed: this is what gets clustered.
        protac_mol = recon.build_protac_mol(coords, energy, pose_idx, pose.get("pose_score"))
        protac_mols.append(protac_mol)

        if not dump_all:
            continue

        # PROTAC SDF
        sdf_writer.write(protac_mol)

        # VHL PDB
        vhl_body = recon.build_vhl_pdb_lines(coords)
        vhl_pdb_fh.write(f"MODEL     {pose_idx + 1:4d}\n")
        vhl_pdb_fh.write(
            f"REMARK   POSE {pose_idx + 1}  "
            f"ENERGY {energy[0]:.3f} kcal/mol\n"
        )
        for line in vhl_body:
            vhl_pdb_fh.write(line + "\n")
        vhl_pdb_fh.write("ENDMDL\n")

        # PROTAC PDB block
        protac_pdb_block = Chem.MolToPDBBlock(protac_mol)
        protac_pdb_lines_pose = [
            l for l in protac_pdb_block.splitlines()
            if not l.startswith("END")
        ]

        # Full complex PDB. The receptor is re-emitted for every pose, which is
        # what makes this file enormous (~half its bytes are duplicate receptor).
        complex_pdb_fh.write(f"MODEL     {pose_idx + 1:4d}\n")
        complex_pdb_fh.write(
            f"REMARK   POSE {pose_idx + 1}  "
            f"ENERGY {energy[0]:.3f} kcal/mol\n"
        )
        for line in recon.receptor_body:
            complex_pdb_fh.write(line + "\n")
        complex_pdb_fh.write("TER\n")
        for line in protac_pdb_lines_pose:
            complex_pdb_fh.write(line + "\n")
        complex_pdb_fh.write("TER\n")
        for line in vhl_body:
            complex_pdb_fh.write(line + "\n")
        complex_pdb_fh.write("ENDMDL\n")

    if dump_all:
        sdf_writer.close()
        vhl_pdb_fh.write("END\n")
        vhl_pdb_fh.close()
        complex_pdb_fh.write("END\n")
        complex_pdb_fh.close()
        logger.info("Wrote %s  (%d poses)", protac_sdf_path, n_poses)
        logger.info("Wrote %s  (%d models)", vhl_pdb_path, n_poses)
        logger.info("Wrote %s  (%d models)", complex_pdb_path, n_poses)
    else:
        logger.info("Reconstructed %d poses in memory "
                    "(per-pose dumps skipped; pass --dump-all-poses to write "
                    "docked_protac_all.sdf / docked_*_all.pdb)", n_poses)

    return protac_mols


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Clustering  (based on pose_explorer approach)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_rmsd_matrix(mols: list[Chem.Mol]) -> np.ndarray:
    """Calculate pairwise heavy-atom RMSD matrix (condensed form).

    All poses share the same atom ordering (same molecule, different conformers),
    so no symmetry matching is needed.  Uses vectorized numpy for speed.
    """
    n = len(mols)
    logger.info("Computing %d x %d heavy-atom RMSD matrix...", n, n)

    heavy_idx = [
        i for i in range(mols[0].GetNumAtoms())
        if mols[0].GetAtomWithIdx(i).GetAtomicNum() > 1
    ]
    logger.info("  Heavy atoms: %d (of %d total)", len(heavy_idx), mols[0].GetNumAtoms())

    # (n_poses, n_heavy, 3)
    coordinates = np.array([
        mol.GetConformer().GetPositions()[heavy_idx]
        for mol in mols
    ], dtype=np.float64)

    n_heavy = len(heavy_idx)
    rmsd_condensed = np.zeros(n * (n - 1) // 2, dtype=np.float64)

    idx = 0
    for i in range(n):
        # Vectorized: diff between pose i and all j > i at once
        diffs = coordinates[i + 1:] - coordinates[i]  # (n-i-1, n_heavy, 3)
        rmsds = np.sqrt(np.mean(np.sum(diffs * diffs, axis=2), axis=1))
        count = len(rmsds)
        rmsd_condensed[idx:idx + count] = rmsds
        idx += count

    return rmsd_condensed


def cluster_poses(
    mols: list[Chem.Mol],
    poses: list[dict],
    rmsd_threshold: float = 2.0,
    method: str = "average",
    max_clusters: int | None = None,
) -> tuple[list[int], dict[int, list[int]], dict[int, int]]:
    """Cluster PROTAC poses by heavy-atom RMSD.

    Returns:
        clusters: cluster assignment per pose (1-indexed)
        cluster_members: {cluster_id: [pose_indices]}
        representatives: {cluster_id: pose_index}
    """
    rmsd_condensed = calculate_rmsd_matrix(mols)

    linked = linkage(rmsd_condensed, method=method)

    if max_clusters is not None:
        clusters_raw = fcluster(linked, max_clusters, criterion="maxclust")
    else:
        clusters_raw = fcluster(linked, rmsd_threshold, criterion="distance")

    energies = [rank_score(p) for p in poses]

    # Order clusters by their best-scoring member, not by population.  Cluster
    # size is a sampling artifact — a wide shallow basin collects more hits than
    # a narrow deep one — so ranking by size buries the best pose.  On the MZ1
    # benchmark the cluster holding the 0.67 A near-native pose ranked 64/359 by
    # size and 1/359 by energy.
    groups = defaultdict(list)
    for i, c in enumerate(clusters_raw):
        groups[c].append(i)
    sorted_clusters = sorted(groups.items(),
                             key=lambda kv: min(energies[i] for i in kv[1]))
    rename = {old_id: new_id + 1
              for new_id, (old_id, _) in enumerate(sorted_clusters)}

    clusters = [rename[c] for c in clusters_raw]

    # Build member dict
    cluster_members = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_members[c].append(i)

    # Representative = best-scoring member, not the medoid.  The medoid is the
    # geometric centre of the basin and throws away the pose the search actually
    # ranked best (on MZ1 it replaced a 0.67 A pose with a 1.45 A one).
    representatives = {cid: min(members, key=lambda i: energies[i])
                       for cid, members in cluster_members.items()}

    n_clusters = len(cluster_members)
    logger.info("Clusters: %d  (threshold=%.1f A)", n_clusters, rmsd_threshold)
    for cid in sorted(cluster_members.keys()):
        rep = representatives[cid]
        logger.info("  Cluster %2d: %3d members, rep=pose_%d (pose score %.3f, binding free energy %.3f kcal/mol)",
                     cid, len(cluster_members[cid]), rep + 1, rank_score(poses[rep]), poses[rep]["energy"][0])

    return clusters, dict(cluster_members), representatives


def save_cluster_results(
    clusters: list[int],
    cluster_members: dict[int, list[int]],
    representatives: dict[int, int],
    protac_mols: list[Chem.Mol],
    poses: list[dict],
    input_dir: str,
    input_json: str,
    poses_dir: str,
    e3_pdb: str = "VHL.pdb",
    receptor_pdb: str = "receptor.pdb",
) -> None:
    """Save clustering results: representative structures + CSV summary."""
    recon = PoseReconstructor(input_dir, input_json,
                              e3_pdb=e3_pdb, receptor_pdb=receptor_pdb)

    # CSV summary
    csv_path = os.path.join(poses_dir, "cluster_info.csv")
    with open(csv_path, "w") as fh:
        fh.write("cluster_id,cluster_size,representative_pose,energy_kcal,pose_score\n")
        for cid in sorted(cluster_members.keys()):
            rep = representatives[cid]
            e = poses[rep]["energy"][0]
            fh.write(f"{cid},{len(cluster_members[cid])},{rep + 1},{e:.3f},{rank_score(poses[rep]):.3f}\n")
    logger.info("Wrote %s", csv_path)

    # Representative PROTAC SDF
    rep_sdf_path = os.path.join(poses_dir, "representatives_protac.sdf")
    sdf_writer = Chem.SDWriter(rep_sdf_path)
    for cid in sorted(representatives.keys()):
        rep = representatives[cid]
        mol = deepcopy(protac_mols[rep])
        mol.SetProp("cluster_id", str(cid))
        mol.SetProp("cluster_size", str(len(cluster_members[cid])))
        sdf_writer.write(mol)
    sdf_writer.close()
    logger.info("Wrote %s  (%d clusters)", rep_sdf_path, len(representatives))

    # Representative E3 ligase PDB (multi-MODEL)
    e3_name = os.path.splitext(e3_pdb)[0].lower()
    rep_vhl_path = os.path.join(poses_dir, f"representatives_{e3_name}.pdb")
    with open(rep_vhl_path, "w") as fh:
        for cid in sorted(representatives.keys()):
            rep = representatives[cid]
            coords = np.array(poses[rep]["coords"]).reshape(-1, 3)
            energy = poses[rep]["energy"]
            vhl_body = recon.build_vhl_pdb_lines(coords)

            fh.write(f"MODEL     {cid:4d}\n")
            fh.write(
                f"REMARK   CLUSTER {cid}  REP_POSE {rep + 1}  "
                f"ENERGY {energy[0]:.3f} kcal/mol  POSE_SCORE {rank_score(poses[rep]):.3f}  "
                f"SIZE {len(cluster_members[cid])}\n"
            )
            for line in vhl_body:
                fh.write(line + "\n")
            fh.write("ENDMDL\n")
        fh.write("END\n")
    logger.info("Wrote %s  (%d models)", rep_vhl_path, len(representatives))

    # Representative complex PDB (multi-MODEL)
    rep_complex_path = os.path.join(poses_dir, "representatives_complex.pdb")
    with open(rep_complex_path, "w") as fh:
        for cid in sorted(representatives.keys()):
            rep = representatives[cid]
            coords = np.array(poses[rep]["coords"]).reshape(-1, 3)
            energy = poses[rep]["energy"]
            vhl_body = recon.build_vhl_pdb_lines(coords)

            protac_pdb_block = Chem.MolToPDBBlock(protac_mols[rep])
            protac_lines = [
                l for l in protac_pdb_block.splitlines()
                if not l.startswith("END")
            ]

            fh.write(f"MODEL     {cid:4d}\n")
            fh.write(
                f"REMARK   CLUSTER {cid}  REP_POSE {rep + 1}  "
                f"ENERGY {energy[0]:.3f} kcal/mol  POSE_SCORE {rank_score(poses[rep]):.3f}  "
                f"SIZE {len(cluster_members[cid])}\n"
            )
            for line in recon.receptor_body:
                fh.write(line + "\n")
            fh.write("TER\n")
            for line in protac_lines:
                fh.write(line + "\n")
            fh.write("TER\n")
            for line in vhl_body:
                fh.write(line + "\n")
            fh.write("ENDMDL\n")
        fh.write("END\n")
    logger.info("Wrote %s  (%d models)", rep_complex_path, len(representatives))

    # Per-pose cluster assignment added to the all-pose SDF
    assigned_sdf_path = os.path.join(poses_dir, "docked_protac_clustered.sdf")
    sdf_writer = Chem.SDWriter(assigned_sdf_path)
    for i, mol in enumerate(protac_mols):
        mol_copy = deepcopy(mol)
        mol_copy.SetProp("cluster_id", str(clusters[i]))
        is_rep = "1" if representatives.get(clusters[i]) == i else "0"
        mol_copy.SetProp("is_representative", is_rep)
        sdf_writer.write(mol_copy)
    sdf_writer.close()
    logger.info("Wrote %s  (%d poses)", assigned_sdf_path, len(protac_mols))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_center_from_sdf(sdf_path: str) -> tuple[float, float, float]:
    """Compute centroid of heavy atoms from an SDF file."""
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        raise ValueError(f"Failed to load SDF: {sdf_path}")
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() > 1:
            p = conf.GetAtomPosition(i)
            coords.append([p.x, p.y, p.z])
    center = np.mean(coords, axis=0)
    return tuple(center.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="PROTAC ternary complex docking with Uni-Dock2"
    )
    parser.add_argument(
        "--input-dir", default=".",
        help="Directory with EP, TP, PROTAC SDFs and receptor/E3 PDBs "
             "(default: current directory)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Uni-Dock2 raw output directory (default: <input-dir>/../ud2_output)",
    )
    parser.add_argument(
        "--poses-dir", default=None,
        help="Post-processed poses directory (default: <output-dir>/poses)",
    )
    # File name configuration. Defaults match the shipped MZ1-BRD4-VHL example
    # so it runs with just --input-dir; the previous defaults (EP.sdf, TP.sdf,
    # protac.sdf, receptor.pdb) named files that exist nowhere in the repo, so
    # every invocation had to spell out all five.
    parser.add_argument("--ep-sdf", default="vhl_lig.sdf",
                        help="E3-ligase pharmacophore SDF filename")
    parser.add_argument("--tp-sdf", default="brd_lig.sdf",
                        help="Target pharmacophore SDF filename")
    parser.add_argument("--protac-sdf", default="protac_random.sdf",
                        help="PROTAC SDF filename")
    parser.add_argument("--receptor-pdb", default="BRD4.pdb",
                        help="Target receptor PDB filename")
    parser.add_argument("--e3-pdb", default="VHL.pdb",
                        help="E3 ligase PDB filename")
    # Docking parameters
    parser.add_argument(
        "--center", nargs=3, type=float, default=None,
        metavar=("X", "Y", "Z"),
        help="Box center (Angstrom). Auto-computed from TP SDF if not given.",
    )
    parser.add_argument(
        "--box-size", nargs=3, type=float, default=[100.0, 100.0, 100.0],
        metavar=("X", "Y", "Z"), help="Box size (Angstrom)",
    )
    parser.add_argument("--exhaustiveness", type=int, default=128)
    parser.add_argument("--mc-steps", type=int, default=20)
    parser.add_argument("--opt-steps", type=int, default=15)
    parser.add_argument("--refine-steps", type=int, default=5)
    parser.add_argument("--num-pose", type=int, default=1000,
                        help="Max poses per run (set high to keep all)")
    parser.add_argument("--energy-range", type=float, default=100.0)
    parser.add_argument("--n-runs", type=int, default=8,
                        help="Number of independent docking runs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (run i uses seed + i)")
    parser.add_argument("--rmsd-threshold", type=float, default=2.0,
                        help="RMSD threshold for clustering (Angstrom)")
    parser.add_argument(
        "--cluster-top-n", type=int, default=500,
        help="Cluster only the N best-scoring poses (0 = all). The RMSD matrix "
             "is O(N^2); the tail of the pose list is redundant anyway (the top "
             "100 of 7760 MZ1 poses held only 13 distinct basins).",
    )
    parser.add_argument("--max-clusters", type=int, default=None,
                        help="Force this many clusters (overrides --rmsd-threshold)")
    parser.add_argument(
        "--skip-prep", action="store_true",
        help="Skip input preparation (use existing DMS files)",
    )
    parser.add_argument(
        "--skip-dock", action="store_true",
        help="Skip docking (post-process existing merged_poses.json only)",
    )
    parser.add_argument(
        "--skip-cluster", action="store_true",
        help="Skip clustering step",
    )
    parser.add_argument(
        "--dump-all-poses", action="store_true",
        help="Also write every pose to docked_protac_all.sdf and "
             "docked_*_all.pdb. Off by default: for 7760 poses these come to "
             "~4.5 GB (the complex PDB repeats the whole receptor per pose) and "
             "nothing downstream reads them — the representatives are built "
             "from the in-memory poses either way.",
    )
    args = parser.parse_args()

    # --input-dir defaults to the current directory, so running in the wrong
    # place is the common mistake. Report every missing file up front rather
    # than letting RDKit raise "Bad input file" from somewhere deeper in.
    if not args.skip_prep:
        required = [args.ep_sdf, args.tp_sdf, args.protac_sdf,
                    args.receptor_pdb, args.e3_pdb]
        missing = [f for f in required
                   if not os.path.isfile(os.path.join(args.input_dir, f))]
        if missing:
            logger.error("Input directory: %s",
                         os.path.abspath(args.input_dir))
            logger.error("Missing %d of %d required input files:",
                         len(missing), len(required))
            for f in missing:
                logger.error("    %s", f)
            logger.error("Pass --input-dir, or override the individual "
                         "filenames (--ep-sdf/--tp-sdf/--protac-sdf/"
                         "--receptor-pdb/--e3-pdb).")
            sys.exit(1)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_dir.rstrip("/")),
                                        "ud2_output")
    if args.poses_dir is None:
        args.poses_dir = os.path.join(args.output_dir, "poses")

    # Auto-compute box center from TP SDF if not given
    if args.center is None:
        tp_path = os.path.join(args.input_dir, args.tp_sdf)
        args.center = list(_compute_center_from_sdf(tp_path))
        logger.info("Auto-computed box center from %s: [%.3f, %.3f, %.3f]",
                     args.tp_sdf, *args.center)

    ligand_dms = os.path.join(args.input_dir, "protac_ligand.dms")
    receptor_dms = os.path.join(args.input_dir, "protac_receptor.dms")
    merged_json = os.path.join(args.output_dir, "merged_poses.json")

    # Step 1: prepare input
    if not args.skip_prep:
        logger.info("=" * 60)
        logger.info("STEP 1: Preparing enriched DMS files")
        logger.info("=" * 60)
        ligand_dms, receptor_dms = prepare_input(
            args.input_dir,
            ep_sdf=args.ep_sdf,
            tp_sdf=args.tp_sdf,
            protac_sdf=args.protac_sdf,
            receptor_pdb=args.receptor_pdb,
            e3_pdb=args.e3_pdb,
        )
    else:
        logger.info("Skipping input preparation (--skip-prep)")
        if not os.path.isfile(ligand_dms):
            logger.error("Ligand DMS not found: %s", ligand_dms)
            sys.exit(1)
        if not os.path.isfile(receptor_dms):
            logger.error("Receptor DMS not found: %s", receptor_dms)
            sys.exit(1)

    # Step 2: run multi-seed docking
    if not args.skip_dock:
        logger.info("=" * 60)
        logger.info("STEP 2: Running %d docking jobs (exhaustiveness=%d)",
                     args.n_runs, args.exhaustiveness)
        logger.info("=" * 60)

        # Stride must be 1, not 1000: the engine seeds pose j of a run with
        # `seed + j * 1000` (mc_cpu.cpp), so a 1000-stride between runs makes
        # run i and run i+1 share exhaustiveness-1 identical starting points.
        seeds = [args.seed + i for i in range(args.n_runs)]
        protac_heavy_idx = compute_protac_cluster_indices(ligand_dms, args.input_dir)
        all_poses = run_multi_seed_docking(
            ligand_dms,
            receptor_dms,
            args.output_dir,
            n_runs=args.n_runs,
            seeds=seeds,
            center=tuple(args.center),
            box_size=tuple(args.box_size),
            exhaustiveness=args.exhaustiveness,
            mc_steps=args.mc_steps,
            opt_steps=args.opt_steps,
            refine_steps=args.refine_steps,
            num_pose=args.num_pose,
            energy_range=args.energy_range,
            cluster_atom_indices=[protac_heavy_idx],
        )

        # Save merged poses
        with open(merged_json, "w") as fh:
            json.dump({"MOL_0": all_poses}, fh)
        logger.info("Saved merged poses: %s", merged_json)
    else:
        logger.info("Skipping docking (--skip-dock)")
        if not os.path.isfile(merged_json):
            logger.error("Merged poses not found: %s", merged_json)
            sys.exit(1)
        with open(merged_json) as fh:
            all_poses = json.load(fh)["MOL_0"]
        all_poses.sort(key=rank_score)
        logger.info("Loaded %d merged poses", len(all_poses))

    # Step 3: post-process all poses
    logger.info("=" * 60)
    logger.info("STEP 3: Post-processing %d poses", len(all_poses))
    logger.info("=" * 60)
    protac_mols = postprocess(all_poses, args.input_dir, args.poses_dir,
                              ligand_dms,
                              e3_pdb=args.e3_pdb,
                              receptor_pdb=args.receptor_pdb,
                              dump_all=args.dump_all_poses)

    # Step 4: cluster
    if not args.skip_cluster:
        logger.info("=" * 60)
        logger.info("STEP 4: Clustering poses")
        logger.info("=" * 60)
        # all_poses is sorted best-first, so the head is the top-N by score
        n_cl = args.cluster_top_n
        if n_cl and n_cl < len(all_poses):
            logger.info("Clustering the %d best-scoring poses (of %d)",
                         n_cl, len(all_poses))
            cl_poses, cl_mols = all_poses[:n_cl], protac_mols[:n_cl]
        else:
            cl_poses, cl_mols = all_poses, protac_mols

        clusters, cluster_members, representatives = cluster_poses(
            cl_mols, cl_poses,
            rmsd_threshold=args.rmsd_threshold,
            max_clusters=args.max_clusters,
        )
        save_cluster_results(
            clusters, cluster_members, representatives,
            cl_mols, cl_poses,
            args.input_dir, ligand_dms, args.poses_dir,
            e3_pdb=args.e3_pdb,
            receptor_pdb=args.receptor_pdb,
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("Total poses: %d", len(all_poses))
    logger.info("Best:  pose score %.3f", rank_score(all_poses[0]))
    logger.info("Worst: pose score %.3f", rank_score(all_poses[-1]))
    if not args.skip_cluster:
        logger.info("Clusters: %d", len(cluster_members))
    logger.info("Output: %s", args.poses_dir)


if __name__ == "__main__":
    main()
