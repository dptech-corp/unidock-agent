"""Reusable helper functions for integration tests.

Functions
---------
load_dms_ligand        Load DMS → RDKit mol + center.
run_docking_protocol   Run UnidockProtocolRunner.
read_sdf_poses         Parse SDF into list[Mol].
get_heavy_positions    Heavy-atom coordinate array.
compute_rmsd           RMSD between two coordinate arrays.
find_ud2_binary        Locate the ud2 binary.
prepare_enriched_dms   Build enriched receptor/ligand DMS from SDF inputs.
create_query_from_cat4a  Create -OH modified CAT-4a for H-rotor tests.
extract_oh_coords      Find -OH group geometry in an SDF.
"""

import math
import os
import shutil
import tempfile

import numpy as np
from rdkit import Chem


# ---------------------------------------------------------------------------
# DMS utilities
# ---------------------------------------------------------------------------
def load_dms_ligand(dms_path: str):
    """Load a DMS ligand file.

    Returns
    -------
    mol : rdkit.Chem.Mol
    center : list[float]  (length 3)
    """
    import msys

    sys = msys.LoadDMS(dms_path)
    msys.AssignBondOrderAndFormalCharge(sys)
    mol = msys.ConvertToRdkit(sys)
    center = mol.GetConformer().GetPositions().mean(axis=0).tolist()
    return mol, center


# ---------------------------------------------------------------------------
# Docking
# ---------------------------------------------------------------------------
def run_docking_protocol(
    receptor: str,
    ligand_list: list,
    center,
    output_sdf: str,
    work_dir: str,
    **kwargs,
) -> str:
    """Run :class:`UnidockProtocolRunner` and return *output_sdf*."""
    from unidock_processing.unidocktools.unidock_protocol_runner import (
        UnidockProtocolRunner,
    )

    runner = UnidockProtocolRunner(
        receptor_file_name=receptor,
        ligand_sdf_file_name_list=ligand_list,
        target_center=tuple(center),
        working_dir_name=work_dir,
        docking_pose_sdf_file_name=output_sdf,
        **kwargs,
    )
    runner.run_unidock_protocol()
    return output_sdf


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
def read_sdf_poses(sdf_path: str, remove_hs: bool = False) -> list:
    """Return list of valid RDKit Mol objects from an SDF file."""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=remove_hs)
    return [m for m in suppl if m is not None]


def get_heavy_positions(mol) -> np.ndarray:
    """Heavy-atom positions (N×3) from an RDKit molecule with a conformer."""
    conf = mol.GetConformer()
    pts = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            p = conf.GetAtomPosition(atom.GetIdx())
            pts.append([p.x, p.y, p.z])
    return np.array(pts)


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------
def compute_rmsd(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """RMSD between two (N, 3) coordinate arrays."""
    return float(np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------
def find_ud2_binary() -> str | None:
    """Search for the *ud2* binary in PATH and common build dirs."""
    ud2 = shutil.which("ud2")
    if ud2:
        return ud2
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for build_dir in ("build_cpu", "build"):
        candidate = os.path.join(
            repo, "unidock", "unidock_engine", build_dir, "bin", "ud2"
        )
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Enriched DMS preparation (for ud2align bias tests)
# ---------------------------------------------------------------------------
def prepare_enriched_dms(receptor_pdb, query_sdf, center, work_dir):
    """Build enriched receptor & ligand DMS from SDF-based inputs.

    Returns
    -------
    lig_dms, rec_dms : str
        Paths to enriched DMS files.
    """
    from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
        UnidockLigandTopologyBuilder,
    )
    from unidock_processing.unidocktools.unidock_protocol_runner import (
        UnidockProtocolRunner,
        _build_ligand_dms_from_sdf,
    )
    from unidock_processing.unidocktools.unidock_receptor_topology_builder import (
        UnidockReceptorTopologyBuilder,
    )

    runner = UnidockProtocolRunner(
        receptor_file_name=receptor_pdb,
        ligand_sdf_file_name_list=[query_sdf],
        target_center=tuple(center),
        working_dir_name=work_dir,
        ff_minimize_steps=0,
        exhaustiveness=1,
        num_pose=1,
    )

    # Receptor
    rec_builder = UnidockReceptorTopologyBuilder(
        runner.receptor_file_name,
        prepared_hydrogen=runner.preserve_receptor_hydrogen,
        covalent_residue_atom_info_list=runner.covalent_residue_atom_info_list,
        working_dir_name=runner.working_dir_name,
    )
    rec_builder.generate_receptor_topology()
    if not runner._try_full_receptor_analysis_subprocess(rec_builder):
        rec_builder.analyze_receptor_topology_vina_only()
    rec_builder.get_summary_receptor_info_dict()

    # Ligand DMS from SDF
    if runner.ligand_dms is None and runner.ligand_sdf_file_name_list:
        dms_path = _build_ligand_dms_from_sdf(
            runner.ligand_sdf_file_name_list[0], runner.working_dir_name
        )
        if dms_path:
            runner.ligand_dms = dms_path
            runner.ligand_dms_dict = {"MOL_0": dms_path}
            runner.ligand_has_ff = True

    # Ligand topology
    lig_builder = UnidockLigandTopologyBuilder(
        runner.ligand_sdf_file_name_list,
        covalent_ligand=runner.covalent_ligand,
        template_docking=runner.template_docking,
        reference_sdf_file_name=runner.reference_sdf_file_name,
        core_atom_mapping_dict_list=runner.core_atom_mapping_dict_list,
        n_cpu=runner.n_cpu,
        working_dir_name=runner.working_dir_name,
        has_ff=runner.ligand_has_ff,
        ligand_dms_path_dict={"MOL_0": runner.ligand_dms} if runner.ligand_dms else None,
    )
    lig_builder.generate_batch_ligand_topology()
    lig_builder.get_summary_ligand_info_dict()

    # Enrich
    runner._produce_enriched_dms(lig_builder, rec_builder)
    assert runner.enriched_ligand_dms, "Failed to produce enriched ligand DMS"
    assert runner.enriched_receptor_dms, "Failed to produce enriched receptor DMS"
    return runner.enriched_ligand_dms, runner.enriched_receptor_dms


# ---------------------------------------------------------------------------
# Template-docking helpers (H-rotor tests)
# ---------------------------------------------------------------------------
def create_query_from_cat4a(cat4a_sdf: str, output_sdf: str):
    """Add an -OH group to the para carbon of terminal phenyl in CAT-4a.

    Returns
    -------
    output_sdf : str
    o_idx, h_idx : int  Atom indices of added O and H.
    """
    from rdkit.Chem import rdFMCS
    from rdkit.Geometry import Point3D

    ref_mol = Chem.SDMolSupplier(cat4a_sdf, removeHs=False)[0]
    ref_noh = Chem.RemoveHs(ref_mol)

    para_pat = Chem.MolFromSmarts("[cH]1[cH][cH]c(-c)[cH][cH]1")
    matches = ref_noh.GetSubstructMatches(para_pat)
    assert matches, "Could not find para carbon pattern in CAT-4a"
    para_c_idx_noh = matches[0][0]

    conf_noh = ref_noh.GetConformer()
    pos_target = conf_noh.GetAtomPosition(para_c_idx_noh)

    conf_full = ref_mol.GetConformer()
    full_para_idx = None
    for i in range(ref_mol.GetNumAtoms()):
        if ref_mol.GetAtomWithIdx(i).GetAtomicNum() != 6:
            continue
        if conf_full.GetAtomPosition(i).Distance(pos_target) < 0.01:
            full_para_idx = i
            break
    assert full_para_idx is not None

    rw = Chem.RWMol(ref_mol)
    h_to_remove = None
    for nb in rw.GetAtomWithIdx(full_para_idx).GetNeighbors():
        if nb.GetAtomicNum() == 1:
            h_to_remove = nb.GetIdx()
            break
    assert h_to_remove is not None

    h_pos = conf_full.GetAtomPosition(h_to_remove)
    c_pos = conf_full.GetAtomPosition(full_para_idx)

    rw.RemoveAtom(h_to_remove)
    if h_to_remove < full_para_idx:
        full_para_idx -= 1

    o_idx = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(full_para_idx, o_idx, Chem.BondType.SINGLE)
    h_idx = rw.AddAtom(Chem.Atom(1))
    rw.AddBond(o_idx, h_idx, Chem.BondType.SINGLE)

    mol = rw.GetMol()
    conf = mol.GetConformer()

    o_xyz = np.array([h_pos.x, h_pos.y, h_pos.z])
    c_xyz = np.array([c_pos.x, c_pos.y, c_pos.z])
    conf.SetAtomPosition(o_idx, h_pos)

    co_dir = o_xyz - c_xyz
    co_dir /= np.linalg.norm(co_dir)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(co_dir, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    perp = np.cross(co_dir, arbitrary)
    perp /= np.linalg.norm(perp)
    angle = math.radians(109.5)
    oh_dir = co_dir * math.cos(math.pi - angle) + perp * math.sin(math.pi - angle)
    oh_dir /= np.linalg.norm(oh_dir)
    h_xyz = o_xyz + oh_dir * 0.96

    conf.SetAtomPosition(h_idx, Point3D(float(h_xyz[0]), float(h_xyz[1]), float(h_xyz[2])))
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    writer = Chem.SDWriter(output_sdf)
    writer.write(mol)
    writer.close()
    return output_sdf, o_idx, h_idx


def extract_oh_coords(sdf_path: str):
    """Find the hydroxyl -OH group in the first molecule of *sdf_path*.

    Returns
    -------
    dict with keys: o_pos, h_pos, c_pos, angle, props
    or None if no hydroxyl found.
    """
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        return None
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8:
            continue
        h_nb = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        c_nb = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
        if len(h_nb) == 1 and len(c_nb) == 1:
            o_pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
            h_pos = np.array(conf.GetAtomPosition(h_nb[0].GetIdx()))
            c_pos = np.array(conf.GetAtomPosition(c_nb[0].GetIdx()))
            co = c_pos - o_pos
            oh = h_pos - o_pos
            cos_a = np.dot(co, oh) / (np.linalg.norm(co) * np.linalg.norm(oh))
            angle = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
            return {
                "o_pos": o_pos,
                "h_pos": h_pos,
                "c_pos": c_pos,
                "angle": angle,
                "props": mol.GetPropsAsDict(),
            }
    return None
