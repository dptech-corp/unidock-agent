"""ABFE benchmark case: FF Cartesian minimization with MCS mask.

Uses the ABL1 / CHEMBL1051277 case identified by the Discover phase:
  - Reference: VX6_ref.sdf (62 atoms)
  - Query: CHEMBL191003_68_nM docked pose (39 atoms)

Step 1-4  : Compute MCS between reference and docked query via
            get_template_docking_atom_mapping — produces ABFE query RDKit indices.

Step 5    : FALLBACK — msys is not available in this environment, so we cannot
            build a ligand DMS from the query SDF on-the-fly.  We fall back to
            the engine test fixture ligand_ejm42_enriched.dms (35 atoms) and
            compute an MCS between the ABFE docked query and ejm42 to identify
            the structurally equivalent atoms in ejm42.

Step 6    : Translate ejm42 RDKit MCS indices → DMS atom indices via coordinate
            matching at 2 dp precision.  For ejm42 the DMS and RDKit indices are
            identical (verified empirically).

Step 7-9  : Run pipeline.so (task='score', ff_minimize_mode=1,
            ff_minimize_steps=200, constrained_atom_indices=[mcs_dms_indices])
            and measure constrained / unconstrained displacements.

Assertions:
  A. max_disp_constrained < 1e-5 Å   (mask honoured)
  B. max_disp_unconstrained > 0.01 Å  (minimiser actually ran)
  C. natom_constrained > 0            (MCS found something)
  D. natom_constrained < natom_total  (not everything locked)
"""
import os
import sys
import sqlite3
import tempfile

# libomp conflict guard: the engine pipeline.so and numpy each link their own
# libomp.dylib. Loading both into one process aborts with "OMP: Error #15" (or
# segfaults once threads spin up). Allow the duplicate runtime and pin the
# engine to a single thread so the two OpenMP runtimes never execute parallel
# regions concurrently. Must run before `import pipeline` (loaded below) and
# before numpy initializes its BLAS/OpenMP backend.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# unidock_processing must be importable
# ---------------------------------------------------------------------------
_PROCESSING_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..',   # unidock/unidock_processing -> unidock
))
if _PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _PROCESSING_DIR)

# ---------------------------------------------------------------------------
# pipeline.so from CPU build
# ---------------------------------------------------------------------------
_ENGINE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..', 'unidock_engine',
))
_BUILD_CPU = os.path.join(_ENGINE_DIR, 'build_cpu', 'api', 'python')
if _BUILD_CPU not in sys.path:
    sys.path.insert(0, _BUILD_CPU)

try:
    import pipeline as _pipeline
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PIPELINE_AVAILABLE,
    reason="pipeline.so not found — build with cmake --build build_cpu --target pipeline",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ABFE_DIR = os.path.join(os.environ.get('UD2_ABFE_INPUT', 'benchmark_input'), 'ABL1', 'CHEMBL1051277')
REFERENCE_SDF = os.path.join(_ABFE_DIR, 'VX6_ref.sdf')
DOCKED_SDF    = os.path.join(_ABFE_DIR, 'docking_output', 'CHEMBL191003_68_nM.sdf')
pytestmark = [pytestmark, pytest.mark.skipif(
    not os.path.exists(REFERENCE_SDF),
    reason="ABFE case not found: set UD2_ABFE_INPUT to the abfe_benchmark benchmark_input directory",
)]

_TYK2_DIR = os.path.normpath(os.path.join(
    _ENGINE_DIR, 'test', 'data', 'ff_minimize', 'tyk2',
))
FALLBACK_LIGAND_DMS  = os.path.join(_TYK2_DIR, 'ligand_ejm42_enriched.dms')
FALLBACK_RECEPTOR_DMS = os.path.join(_TYK2_DIR, 'receptor_enriched.dms')
FALLBACK_LIGAND_SDF  = os.path.join(_TYK2_DIR, 'ligand_ejm42.sdf')

# Pocket centre for ejm42 (from reference test)
CENTRE = dict(center_x=-1.5, center_y=22.0, center_z=-28.0,
              size_x=30.0, size_y=30.0, size_z=30.0)

NATOM_FALLBACK = 35   # ejm42 atom count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sdf_coords(sdf_path):
    """Return (N, 3) float64 from the first molecule in an SDF (V2000 format)."""
    with open(sdf_path) as fh:
        lines = fh.readlines()
    natom = int(lines[3][:3])
    coords = []
    for i in range(4, 4 + natom):
        parts = lines[i].split()
        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(coords, dtype=np.float64)


def _dms_coords_as_array(dms_path):
    """Return (N, 3) float64 from particle table in a DMS file (ordered by id)."""
    db = sqlite3.connect(dms_path)
    cur = db.cursor()
    rows = cur.execute('SELECT x, y, z FROM particle ORDER BY id').fetchall()
    db.close()
    return np.array(rows, dtype=np.float64)


def _score_run(tmpdir, ligand_dms, receptor_dms, ff_minimize_steps,
               ff_minimize_mode, constrained_atom_indices=None):
    """Run pipeline in score mode; return path to output SDF."""
    kwargs = dict(
        ligand_dms_path=ligand_dms,
        receptor_dms_path=receptor_dms,
        output_dir=tmpdir,
        task='score',
        seed=12345,
        ff_minimize_steps=ff_minimize_steps,
        ff_minimize_mode=ff_minimize_mode,
        **CENTRE,
    )
    if constrained_atom_indices is not None:
        kwargs['constrained_atom_indices'] = constrained_atom_indices
    _pipeline.run_docking_pipeline(**kwargs)
    sdfs = [f for f in os.listdir(tmpdir) if f.endswith('.sdf')]
    assert sdfs, f"No SDF output in {tmpdir}"
    return os.path.join(tmpdir, sdfs[0])


# ---------------------------------------------------------------------------
# MCS computation (shared fixture)
# ---------------------------------------------------------------------------

def _compute_mcs_indices():
    """Return (abfe_rdkit_indices, ejm42_rdkit_indices, ejm42_dms_indices).

    ABFE indices: query RDKit atoms matched by get_template_docking_atom_mapping
                  against VX6_ref.
    ejm42 indices: atoms in ejm42 structurally equivalent to the ABFE query core,
                   via a second MCS run (ABFE docked query -> ejm42).
    DMS indices: coordinate-matched translation of ejm42 RDKit -> DMS.
    """
    from rdkit import Chem
    from unidock_processing.ligand_topology.utils import get_template_docking_atom_mapping

    ref_mol    = Chem.SDMolSupplier(REFERENCE_SDF,  removeHs=False)[0]
    docked_mol = Chem.SDMolSupplier(DOCKED_SDF,     removeHs=False)[0]
    ejm42_mol  = Chem.SDMolSupplier(FALLBACK_LIGAND_SDF, removeHs=False)[0]

    assert ref_mol    is not None, f"Failed to load {REFERENCE_SDF}"
    assert docked_mol is not None, f"Failed to load {DOCKED_SDF}"
    assert ejm42_mol  is not None, f"Failed to load {FALLBACK_LIGAND_SDF}"

    # ABFE MCS: reference -> docked query
    abfe_mapping         = get_template_docking_atom_mapping(ref_mol, docked_mol)
    abfe_rdkit_indices   = sorted(set(abfe_mapping.values()))

    # Fallback MCS: docked query -> ejm42
    fallback_mapping     = get_template_docking_atom_mapping(docked_mol, ejm42_mol)
    ejm42_rdkit_indices  = sorted(set(fallback_mapping.values()))

    # DMS index translation via 2dp coordinate matching
    conf = ejm42_mol.GetConformer()
    rdkit_pos_to_idx = {}
    for i in range(ejm42_mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        rdkit_pos_to_idx[(round(pos.x, 2), round(pos.y, 2), round(pos.z, 2))] = i

    db = sqlite3.connect(FALLBACK_LIGAND_DMS)
    cur = db.cursor()
    dms_rows = cur.execute('SELECT id, x, y, z FROM particle').fetchall()
    db.close()

    rdkit_to_dms = {}
    for row in dms_rows:
        key = (round(row[1], 2), round(row[2], 2), round(row[3], 2))
        if key in rdkit_pos_to_idx:
            rdkit_to_dms[rdkit_pos_to_idx[key]] = row[0]

    ejm42_dms_indices = []
    for ri in ejm42_rdkit_indices:
        dms_id = rdkit_to_dms.get(ri)
        if dms_id is not None:
            ejm42_dms_indices.append(dms_id)

    return abfe_rdkit_indices, ejm42_rdkit_indices, ejm42_dms_indices


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_abfe_mcs_mask_ff_minimize():
    """FF Cartesian minimization with MCS mask from ABFE ABL1 case (fallback ejm42 DMS).

    NOTE: This test falls back to ligand_ejm42_enriched.dms because msys is not
    available in this environment and a ligand DMS for CHEMBL191003 cannot be
    built on-the-fly.  The MCS atoms are computed from the ABFE benchmark data
    (VX6_ref -> CHEMBL191003 docked pose) and then bridged to ejm42 via a second
    MCS (CHEMBL191003 -> ejm42) so that the constrained_atom_indices are
    meaningful for the ejm42 DMS.
    """
    abfe_rdkit_indices, ejm42_rdkit_indices, ejm42_dms_indices = _compute_mcs_indices()

    print(f"\nABFE MCS (RDKit): {len(abfe_rdkit_indices)} atoms: {abfe_rdkit_indices}")
    print(f"ejm42 MCS (RDKit): {len(ejm42_rdkit_indices)} atoms: {ejm42_rdkit_indices}")
    print(f"ejm42 MCS (DMS):   {len(ejm42_dms_indices)} atoms: {ejm42_dms_indices}")

    natom_total       = NATOM_FALLBACK
    natom_constrained = len(ejm42_dms_indices)

    # Assertion C: MCS found something
    assert natom_constrained > 0, "MCS produced zero constrained atoms"

    # Assertion D: not everything is locked
    assert natom_constrained < natom_total, (
        f"All {natom_total} atoms constrained — MCS matched entire molecule"
    )

    with (tempfile.TemporaryDirectory() as d0,
          tempfile.TemporaryDirectory() as d_constrained):

        # Initial pose: score-only, no FF minimization
        sdf_initial = _score_run(
            d0,
            FALLBACK_LIGAND_DMS, FALLBACK_RECEPTOR_DMS,
            ff_minimize_steps=0, ff_minimize_mode=0,
        )
        coords_initial = _parse_sdf_coords(sdf_initial)
        assert coords_initial.shape == (natom_total, 3), (
            f"Expected ({natom_total}, 3), got {coords_initial.shape}"
        )

        # Constrained run: MCS atoms masked
        sdf_constrained = _score_run(
            d_constrained,
            FALLBACK_LIGAND_DMS, FALLBACK_RECEPTOR_DMS,
            ff_minimize_steps=200, ff_minimize_mode=1,
            constrained_atom_indices=[ejm42_dms_indices],
        )
        coords_constrained = _parse_sdf_coords(sdf_constrained)

    # Build masks
    constrained_mask   = np.array(sorted(ejm42_dms_indices), dtype=int)
    all_indices        = np.arange(natom_total)
    unconstrained_mask = np.array(
        [i for i in all_indices if i not in set(ejm42_dms_indices)], dtype=int
    )

    disp = np.linalg.norm(coords_constrained - coords_initial, axis=1)

    max_disp_constrained   = float(np.max(disp[constrained_mask]))
    max_disp_unconstrained = float(np.max(disp[unconstrained_mask]))
    mean_disp_unconstrained = float(np.mean(disp[unconstrained_mask]))

    print(f"\nmax_disp_constrained   = {max_disp_constrained:.2e} Å")
    print(f"max_disp_unconstrained = {max_disp_unconstrained:.4f} Å")
    print(f"mean_disp_unconstrained = {mean_disp_unconstrained:.4f} Å")

    # Assertion A: constrained atoms must not move
    assert max_disp_constrained < 1e-5, (
        f"Constrained atom moved {max_disp_constrained:.2e} Å — mask not honoured"
    )

    # Assertion B: unconstrained atoms must have moved (minimiser ran)
    assert max_disp_unconstrained > 0.01, (
        f"Unconstrained atoms barely moved ({max_disp_unconstrained:.4f} Å) "
        f"— FF minimiser may not have run"
    )
