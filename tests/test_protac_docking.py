"""Integration tests for the PROTAC ternary complex docking pipeline.

Uses the MZ1-BRD4-VHL system (PDB 5T35) as the test case.
Input files are in examples/protac_docking/MZ1-BRD4-VHL/.

Run:
    pytest tests/test_protac_docking.py -v
"""
import os
import sys
import json
import sqlite3
import tempfile

import numpy as np
import pytest

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Engine API
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_API = os.path.join(
    REPO_ROOT, "unidock", "unidock_engine", "api", "python"
)
sys.path.insert(0, ENGINE_API)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_center(sdf_path: str) -> tuple[float, float, float]:
    """Compute heavy-atom centroid from an SDF."""
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() > 1:
            p = conf.GetAtomPosition(i)
            coords.append([p.x, p.y, p.z])
    return tuple(np.mean(coords, axis=0).tolist())


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dms_paths(protac_mz1_dir, protac_scripts_dir):
    """Build enriched DMS files for MZ1-BRD4-VHL (once per module)."""
    sys.path.insert(0, protac_scripts_dir)
    import protac_sampling

    with tempfile.TemporaryDirectory(prefix="/tmp/protac_test_dms_") as tmpdir:
        protac_sampling.INPUT_DIR = protac_mz1_dir
        protac_sampling.EP_SDF = os.path.join(protac_mz1_dir, "vhl_lig.sdf")
        protac_sampling.TP_SDF = os.path.join(protac_mz1_dir, "brd_lig.sdf")
        protac_sampling.PROTAC_SDF = os.path.join(protac_mz1_dir, "protac_random.sdf")
        protac_sampling.VHL_PDB = os.path.join(protac_mz1_dir, "VHL.pdb")
        protac_sampling.RECEPTOR_PDB = os.path.join(protac_mz1_dir, "BRD4.pdb")
        protac_sampling.OUTPUT_LIGAND_DMS = os.path.join(tmpdir, "ligand.dms")
        protac_sampling.OUTPUT_RECEPTOR_DMS = os.path.join(tmpdir, "receptor.dms")

        lig_dms, rec_dms = protac_sampling.main()
        yield lig_dms, rec_dms


@pytest.fixture(scope="module")
def docked_output(dms_paths, protac_mz1_dir):
    """Run a minimal docking job via subprocess and return (sdf_path, output_dir).

    Uses subprocess to avoid SIGSEGV from C++ engine re-initialization
    when other tests already loaded the pipeline library.
    """
    import subprocess

    lig_dms, rec_dms = dms_paths
    center = _compute_center(os.path.join(protac_mz1_dir, "brd_lig.sdf"))

    with tempfile.TemporaryDirectory(prefix="/tmp/protac_test_dock_") as tmpdir:
        script = f"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "{ENGINE_API}")
from pipeline import run_docking_pipeline
run_docking_pipeline(
    ligand_dms_path="{os.path.abspath(lig_dms)}",
    receptor_dms_path="{os.path.abspath(rec_dms)}",
    output_dir="{tmpdir}",
    center_x={center[0]}, center_y={center[1]}, center_z={center[2]},
    size_x=100.0, size_y=100.0, size_z=100.0,
    task="screen", search_mode="free",
    exhaustiveness=8, randomize=True,
    mc_steps=2, opt_steps=2, refine_steps=1,
    num_pose=5, rmsd_limit=0.5, energy_range=100.0,
    seed=42, use_tor_lib=True, constraint_docking=True,
    gpu_device_id=0,
)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"Docking subprocess failed:\n{result.stderr}"
        )

        sdf_files = [
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir) if f.endswith(".sdf")
        ]
        assert sdf_files, "Engine produced no SDF output"
        yield sdf_files[0], tmpdir


# ═════════════════════════════════════════════════════════════════════════════
# Tests: DMS building
# ═════════════════════════════════════════════════════════════════════════════

class TestDMSBuild:
    """Verify DMS files are structurally correct."""

    def test_ligand_dms_exists(self, dms_paths):
        lig_dms, _ = dms_paths
        assert os.path.isfile(lig_dms)
        assert os.path.getsize(lig_dms) > 0

    def test_receptor_dms_exists(self, dms_paths):
        _, rec_dms = dms_paths
        assert os.path.isfile(rec_dms)
        assert os.path.getsize(rec_dms) > 0

    def test_ligand_atom_count(self, dms_paths):
        """Ligand = 131 PROTAC + 769 VHL kept = 900 atoms."""
        lig_dms, _ = dms_paths
        conn = sqlite3.connect(lig_dms)
        n = conn.execute("SELECT COUNT(*) FROM ud2_atom_props").fetchone()[0]
        conn.close()
        assert n == 900

    def test_receptor_atom_count(self, dms_paths):
        """Receptor includes all atoms (heavy + H) for RecDonorHTopo."""
        _, rec_dms = dms_paths
        conn = sqlite3.connect(rec_dms)
        n = conn.execute("SELECT COUNT(*) FROM ud2_atom_props").fetchone()[0]
        conn.close()
        assert n == 1744

    def test_torsion_count(self, dms_paths):
        """MZ1 has 11 active linker torsions."""
        lig_dms, _ = dms_paths
        conn = sqlite3.connect(lig_dms)
        n = conn.execute("SELECT COUNT(*) FROM ud2_torsions").fetchone()[0]
        conn.close()
        assert n == 11

    def test_root_atoms(self, dms_paths):
        """Root fragment = TP pharmacophore, 44 atoms."""
        lig_dms, _ = dms_paths
        conn = sqlite3.connect(lig_dms)
        n = conn.execute("SELECT COUNT(*) FROM ud2_root_atoms").fetchone()[0]
        conn.close()
        assert n == 44

    def test_required_tables_exist(self, dms_paths):
        """Check all ud2_* tables the engine needs are present."""
        lig_dms, _ = dms_paths
        conn = sqlite3.connect(lig_dms)
        tables = {
            r[0] for r in
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        required = {
            "ud2_atom_props", "ud2_torsions", "ud2_torsion_ranges",
            "ud2_root_atoms", "ud2_metadata",
            "ud2_sdf_properties", "ud2_formal_charges", "ud2_bond_stereo",
            "exclusion", "pair_12_6_es_term",
            "nonbonded_param", "pair_12_6_es_param",
            "dihedral_trig_param", "dihedral_trig_term",
            "ud2_ep_mapping",
        }
        missing = required - tables
        assert not missing, f"Missing tables: {missing}"

    def test_sdf_natom_metadata(self, dms_paths):
        """sdf_natom in ud2_metadata must equal PROTAC atom count (131)."""
        lig_dms, _ = dms_paths
        conn = sqlite3.connect(lig_dms)
        row = conn.execute(
            "SELECT value FROM ud2_metadata WHERE key = 'sdf_natom'"
        ).fetchone()
        conn.close()
        assert row is not None, "sdf_natom not found in ud2_metadata"
        assert int(row[0]) == 131

    def test_ep_mapping_table(self, dms_paths):
        """ud2_ep_mapping must have EP atom indices."""
        lig_dms, _ = dms_paths
        conn = sqlite3.connect(lig_dms)
        rows = conn.execute(
            "SELECT protac_atom_id FROM ud2_ep_mapping"
        ).fetchall()
        conn.close()
        assert len(rows) > 0, "ud2_ep_mapping is empty"
        # All EP indices must be within PROTAC range
        for r in rows:
            assert 0 <= r[0] < 131, f"EP index {r[0]} out of PROTAC range"

    def test_protac_temp_generated(self, dms_paths, protac_mz1_dir):
        """protac_temp_cursor.sdf should be written during DMS build."""
        path = os.path.join(protac_mz1_dir, "protac_temp_cursor.sdf")
        assert os.path.isfile(path)
        mol = Chem.SDMolSupplier(path, removeHs=False)[0]
        assert mol is not None
        assert mol.GetNumAtoms() == 131


# ═════════════════════════════════════════════════════════════════════════════
# Tests: Engine docking
# ═════════════════════════════════════════════════════════════════════════════

class TestDocking:
    """Verify engine produces valid output."""

    def test_sdf_output_exists(self, docked_output):
        sdf_path, _ = docked_output
        assert os.path.isfile(sdf_path)
        assert os.path.getsize(sdf_path) > 0

    def test_poses_parsed(self, docked_output, protac_scripts_dir):
        """Custom SDF parser must read at least 1 pose."""
        sys.path.insert(0, protac_scripts_dir)
        from run_protac_docking import _sdf_to_pose_dicts

        sdf_path, _ = docked_output
        poses = _sdf_to_pose_dicts(sdf_path, seed=42)
        assert len(poses) >= 1

    def test_atom_count_is_protac_only(self, dms_paths, docked_output,
                                       protac_scripts_dir):
        """Each pose must have exactly 131 PROTAC atoms (not 900 total)."""
        sys.path.insert(0, protac_scripts_dir)
        from run_protac_docking import _sdf_to_pose_dicts

        lig_dms, _ = dms_paths
        conn = sqlite3.connect(lig_dms)
        row = conn.execute(
            "SELECT value FROM ud2_metadata WHERE key = 'sdf_natom'"
        ).fetchone()
        conn.close()
        n_protac = int(row[0])

        sdf_path, _ = docked_output
        poses = _sdf_to_pose_dicts(sdf_path, seed=42)
        for i, p in enumerate(poses):
            n_atoms = len(p["coords"]) // 3
            assert n_atoms == n_protac, (
                f"Pose {i}: {n_atoms} atoms, expected {n_protac}"
            )

    def test_energy_negative(self, docked_output, protac_scripts_dir):
        """Best pose should have negative binding energy."""
        sys.path.insert(0, protac_scripts_dir)
        from run_protac_docking import _sdf_to_pose_dicts

        sdf_path, _ = docked_output
        poses = _sdf_to_pose_dicts(sdf_path, seed=42)
        best_energy = min(p["energy"][0] for p in poses)
        assert best_energy < 0, f"Best energy is non-negative: {best_energy}"

    def test_energy_components(self, docked_output, protac_scripts_dir):
        """Each pose must have 6 energy components."""
        sys.path.insert(0, protac_scripts_dir)
        from run_protac_docking import _sdf_to_pose_dicts

        sdf_path, _ = docked_output
        poses = _sdf_to_pose_dicts(sdf_path, seed=42)
        for i, p in enumerate(poses):
            assert len(p["energy"]) == 6, (
                f"Pose {i}: {len(p['energy'])} energy components, expected 6"
            )


# ═════════════════════════════════════════════════════════════════════════════
# Tests: Post-processing
# ═════════════════════════════════════════════════════════════════════════════

class TestPostProcessing:
    """Verify PoseReconstructor and output file generation."""

    def test_pose_reconstructor_init(self, dms_paths, protac_mz1_dir,
                                      protac_scripts_dir):
        """PoseReconstructor must resolve n_protac + n_vhl_kept == n_lig."""
        sys.path.insert(0, protac_scripts_dir)
        from run_protac_docking import PoseReconstructor

        lig_dms, _ = dms_paths
        recon = PoseReconstructor(
            protac_mz1_dir, lig_dms,
            e3_pdb="VHL.pdb", receptor_pdb="BRD4.pdb",
        )
        assert recon.n_protac == 131
        assert recon.n_vhl_kept == 769
        assert recon.n_protac + recon.n_vhl_kept == 900

    def test_build_protac_mol(self, dms_paths, docked_output,
                               protac_mz1_dir, protac_scripts_dir):
        """Reconstructed PROTAC mol should have 131 atoms (coords from PROTAC-only SDF)."""
        sys.path.insert(0, protac_scripts_dir)
        from run_protac_docking import PoseReconstructor, _sdf_to_pose_dicts

        lig_dms, _ = dms_paths
        sdf_path, _ = docked_output

        recon = PoseReconstructor(
            protac_mz1_dir, lig_dms,
            e3_pdb="VHL.pdb", receptor_pdb="BRD4.pdb",
        )
        poses = _sdf_to_pose_dicts(sdf_path, seed=42)
        coords = np.array(poses[0]["coords"]).reshape(-1, 3)
        assert coords.shape == (131, 3), f"Expected 131x3 coords, got {coords.shape}"
        energy = poses[0]["energy"]

        mol = recon.build_protac_mol(coords, energy, pose_idx=0)
        assert mol is not None
        assert mol.GetNumAtoms() == 131
        assert mol.HasProp("vina_binding_free_energy")

    def test_postprocess_full(self, dms_paths, docked_output,
                               protac_mz1_dir, protac_scripts_dir):
        """Full postprocess with EP-based VHL reconstruction."""
        sys.path.insert(0, protac_scripts_dir)
        from run_protac_docking import _sdf_to_pose_dicts, postprocess

        lig_dms, _ = dms_paths
        sdf_path, _ = docked_output
        poses = _sdf_to_pose_dicts(sdf_path, seed=42)

        # Verify poses have PROTAC-only coords (131 atoms)
        for p in poses:
            assert len(p["coords"]) == 131 * 3

        with tempfile.TemporaryDirectory(prefix="/tmp/protac_test_pp_") as tmpdir:
            protac_mols = postprocess(
                poses, protac_mz1_dir, tmpdir, lig_dms,
                e3_pdb="VHL.pdb", receptor_pdb="BRD4.pdb",
            )
            assert len(protac_mols) == len(poses)

            # PROTAC SDF
            protac_sdf = os.path.join(tmpdir, "docked_protac_all.sdf")
            assert os.path.isfile(protac_sdf)
            suppl = Chem.SDMolSupplier(protac_sdf, removeHs=False)
            sdf_mols = [m for m in suppl if m is not None]
            assert len(sdf_mols) == len(poses)
            for m in sdf_mols:
                assert m.GetNumAtoms() == 131

            # VHL PDB
            vhl_pdb = os.path.join(tmpdir, "docked_vhl_all.pdb")
            assert os.path.isfile(vhl_pdb)
            with open(vhl_pdb) as f:
                n_models = sum(1 for line in f if line.startswith("MODEL"))
            assert n_models == len(poses)

            # Complex PDB
            complex_pdb = os.path.join(tmpdir, "docked_complex_all.pdb")
            assert os.path.isfile(complex_pdb)
            with open(complex_pdb) as f:
                n_models = sum(1 for line in f if line.startswith("MODEL"))
            assert n_models == len(poses)


# ═════════════════════════════════════════════════════════════════════════════
# Tests: SDF parser edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestSDFParser:
    """Test the custom SDF parser handles V2000 overflow and edge cases."""

    @pytest.fixture(autouse=True)
    def _import_parser(self, protac_scripts_dir):
        sys.path.insert(0, protac_scripts_dir)
        from run_protac_docking import _sdf_to_pose_dicts
        self._parse = _sdf_to_pose_dicts

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w",
                                         delete=False) as f:
            f.write("")
            path = f.name
        try:
            assert self._parse(path) == []
        finally:
            os.unlink(path)

    def test_standard_v2000(self):
        """Parser must work on normal V2000 SDF (< 999 atoms)."""
        with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w",
                                         delete=False) as f:
            f.write("test\n")
            f.write("     RDKit          3D\n")
            f.write("\n")
            f.write("  3  2  0  0  0  0  0  0  0  0999 V2000\n")
            f.write("    1.0000    2.0000    3.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n")
            f.write("    4.0000    5.0000    6.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n")
            f.write("    7.0000    8.0000    9.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n")
            f.write("  1  2  1  0\n")
            f.write("  2  3  1  0\n")
            f.write("M  END\n")
            f.write(">  <vina_binding_free_energy>\n")
            f.write("-5.123\n")
            f.write("\n")
            f.write("$$$$\n")
            path = f.name
        try:
            poses = self._parse(path)
            assert len(poses) == 1
            assert len(poses[0]["coords"]) == 9
            assert poses[0]["coords"][:3] == [1.0, 2.0, 3.0]
            assert abs(poses[0]["energy"][0] - (-5.123)) < 1e-6
        finally:
            os.unlink(path)

    def test_overflow_v2000(self):
        """Parser must handle >999 atoms (counts line overflow)."""
        with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w",
                                         delete=False) as f:
            f.write("bigmol\n")
            f.write("     UniDock2       3D\n")
            f.write("\n")
            f.write("12001300  0  0  0  0  0  0  0  0999 V2000\n")
            for i in range(1200):
                x = float(i) * 0.1
                f.write(f"{x:10.4f}{0.0:10.4f}{0.0:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0\n")
            for i in range(min(10, 1199)):
                f.write(f"{i+1:3d}{i+2:3d}  1  0\n")
            f.write("M  END\n")
            f.write(">  <vina_binding_free_energy>\n")
            f.write("-12.5\n")
            f.write("\n")
            f.write("$$$$\n")
            path = f.name
        try:
            poses = self._parse(path)
            assert len(poses) == 1
            assert len(poses[0]["coords"]) == 1200 * 3
            assert abs(poses[0]["energy"][0] - (-12.5)) < 1e-6
        finally:
            os.unlink(path)

    def test_multi_record(self):
        """Parser must handle multiple $$$$ records."""
        with tempfile.NamedTemporaryFile(suffix=".sdf", mode="w",
                                         delete=False) as f:
            for idx in range(3):
                energy = -5.0 - idx
                f.write(f"mol_{idx}\n")
                f.write("     UniDock2       3D\n")
                f.write("\n")
                f.write("  2  1  0  0  0  0  0  0  0  0999 V2000\n")
                f.write(f"{float(idx):10.4f}{0.0:10.4f}{0.0:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0\n")
                f.write(f"{0.0:10.4f}{float(idx):10.4f}{0.0:10.4f} N   0  0  0  0  0  0  0  0  0  0  0  0\n")
                f.write("  1  2  1  0\n")
                f.write("M  END\n")
                f.write(">  <vina_binding_free_energy>\n")
                f.write(f"{energy:.3f}\n")
                f.write("\n")
                f.write("$$$$\n")
            path = f.name
        try:
            poses = self._parse(path)
            assert len(poses) == 3
            assert abs(poses[0]["energy"][0] - (-5.0)) < 1e-6
            assert abs(poses[1]["energy"][0] - (-6.0)) < 1e-6
            assert abs(poses[2]["energy"][0] - (-7.0)) < 1e-6
        finally:
            os.unlink(path)
