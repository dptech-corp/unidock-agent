# Uni-Dock2 Testing Guide

A practical guide for testing the Uni-Dock2 molecular docking engine, grounded in
the actual codebase, its architecture, and its existing test infrastructure.

---

## If You Only Do 5 Things

1. **Lock re-docking golden cases** (TYK2, 1G9V, 7SJ3) with RMSD thresholds and
   score snapshots — these catch regressions in scoring, sampling, and geometry
   simultaneously
2. **Test scoring function invariants** — translation/rotation invariance, symmetry,
   boundedness, angular HBond monotonicity
3. **Control all randomness with `seed=42`** — MC search and pose randomization make
   tests flaky without deterministic replay
4. **Validate DMS enrichment contracts** — a silently misparsed `ud2_atom_props` or
   missing `ud2_torsions` table produces wrong results with no error
5. **Finite-difference gradient checks** — every new scoring term must have FD
   validation against its analytic gradient

---

## Risk Assessment

| Component | Probability | Impact | Detectability | Risk | Current Coverage |
|---|---|---|---|---|---|
| **Vina scoring terms** | High | Critical | Low | **Critical** | Good (test_vina.cpp) |
| **Angular HBond** | High | High | Low | **Critical** | Excellent (28 cases, FD checks) |
| **MC search** | High | Critical | Low | **Critical** | Good (test_mc_cpu.cpp) |
| **BFGS optimizer** | High | High | Low | **High** | Good (convergence + idempotency in test_cpu_bfgs.cpp) |
| **DMS reader/writer** | Medium | High | Medium | **High** | Good (5 cases) |
| **Spatial grid** | Medium | High | Low | **High** | Good (test_spatial_grid.cpp) |
| **Quaternion/rotation** | High | High | Low | **High** | Good (14 cases) |
| **SDF output** | Low | Medium | High | **Low** | Good (7 cases) |
| **Clustering** | Medium | Medium | Medium | **Medium** | Good (10 cases) |

---

## Table of Contents

1. [Existing Test Infrastructure](#1-existing-test-infrastructure)
2. [Golden File Tests — Re-docking References](#2-golden-file-tests--re-docking-references)
3. [C++ Unit Tests — Scoring Terms](#3-c-unit-tests--scoring-terms)
4. [Property-Based Tests — Invariants](#4-property-based-tests--invariants)
5. [Gradient Validation — Finite Difference](#5-gradient-validation--finite-difference)
6. [Contract Tests — DMS Enrichment](#6-contract-tests--dms-enrichment)
7. [Nondeterminism Control](#7-nondeterminism-control)
8. [Performance Budget Tests](#8-performance-budget-tests)
9. [CPU vs GPU Consistency](#9-cpu-vs-gpu-consistency)
10. [Test Data and Organization](#10-test-data-and-organization)
11. [Test Tiers and Execution](#11-test-tiers-and-execution)
12. [Coverage Gaps and Priorities](#12-coverage-gaps-and-priorities)
13. [Anti-Patterns](#13-anti-patterns)

---

## 1. Existing Test Infrastructure

### Layout

```
Uni-Dock2/
├── tests/                         # Python integration tests (18 files)
│   ├── conftest.py                # Session fixtures: TYK2, BACE, CDK4, 1EWL
│   └── helpers.py                 # compute_rmsd(), run_docking()
├── unidock/
│   ├── unidock_engine/test/       # C++ engine tests
│   │   ├── unit/                  # Catch2 unit tests (31 .cpp files)
│   │   │   ├── cpu/               # CPU backend: score, bfgs, cluster, angular, mc, spatial_grid, custom types
│   │   │   ├── cuda/              # CUDA backend tests
│   │   │   ├── score/             # Scoring terms: vina, gaff2, bias, rescore
│   │   │   ├── search/            # MC search tests
│   │   │   ├── format/            # DMS reader, SDF writer
│   │   │   ├── geometry/          # Quaternion, rotation
│   │   │   └── myutils/           # Math, matrix
│   │   ├── component/             # Component integration (1 file)
│   │   └── data/                  # Test DMS/SDF/PDB files
│   └── unidock_processing/test/   # Processing pipeline tests (22+ files)
│       └── scripts/
│           └── snapshots/         # Golden file regression suite
```

### Frameworks

| Layer | Framework | Assertions |
|---|---|---|
| C++ unit | Catch2 | `REQUIRE`, `REQUIRE_THAT(val, WithinAbs(expected, tol))` |
| Python integration | pytest | `pytest.approx(expected, abs=tol)`, `np.allclose` |
| Python parametrized | pytest + JSON fixtures | `compare_sdf_rmsd(actual, expected, threshold)` |
| Golden files | Custom snapshot suite | `compare_values()` in snapshot_helpers.py |

### Running Tests

```bash
# C++ unit tests (requires build)
cd unidock/unidock_engine/build_cpu && cmake --build . --target unit_test
./test/unit/unit_test                          # all C++ tests
./test/unit/unit_test "[angular_hbond]"        # by tag
./test/unit/unit_test "[vina]"                 # scoring terms only

# Python — processing pipeline
python -m pytest unidock/unidock_processing/test/ -v

# Python — integration (slow tests need AmberTools + engine binary)
python -m pytest tests/ -v
python -m pytest tests/ -m "not slow"          # skip expensive tests

# Snapshot regression
python -m pytest unidock/unidock_processing/test/scripts/snapshots/ -v
```

---

## 2. Golden File Tests — Re-docking References

The most valuable pattern. Dock a known system, lock the result.

### Existing golden test systems

| System | Use case | Data location |
|---|---|---|
| TYK2 (ejm42) | FF minimization, scoring, DMS workflow | `unidock_engine/test/data/ff_minimize/tyk2/` |
| 1G9V | Free docking, search modes | `unidock_processing/test/data/free_docking/` |
| 7SJ3 (CDK4) | external ud2align, HBond alignment | external ud2align test data |
| 1E9 (ERα) | external ud2align, ZAligner | external ud2align test data |
| 1EWL | Covalent docking | `unidock_processing/test/data/covalent_docking/` |
| BACE | Constraint docking | `examples/constraint_docking/` |

### Score reproducibility pattern

```python
# tests/test_reproducibility.py — existing pattern
@pytest.mark.slow
def test_same_seed_same_coordinates(tyk2_result_seed42_a, tyk2_result_seed42_b):
    """Identical seeds must produce identical coordinates."""
    rmsd = compute_rmsd(coords_a, coords_b)
    assert rmsd < 1e-6

def test_same_seed_same_score(tyk2_result_seed42_a, tyk2_result_seed42_b):
    """Identical seeds must produce identical scores."""
    assert score_a == pytest.approx(score_b, abs=1e-8)
```

### RMSD threshold conventions

| Context | Threshold | Rationale |
|---|---|---|
| Reproducibility (same seed) | `< 1e-6 Å` | Bit-level determinism |
| Score-only (no search) | `< 0.01 Å` | Coordinate preservation |
| ud2align facade | `< 0.1 Å` | Alignment accuracy |
| Re-docking vs crystal | `< 2.0 Å` | Standard success criterion |

### Adding a new golden system

1. Choose a well-resolved crystal (< 2.0 Å) with clear ligand density
2. Prepare enriched DMS: receptor + ligand
3. Run docking with `seed=42`, `search_mode=balance`
4. Record best RMSD to crystal and best score
5. Commit reference SDF + expected values to `tests/data/`
6. Write test asserting RMSD and score within tolerance

```python
# Template for new golden test
@pytest.mark.slow
def test_redocking_XXXX_reproduces_crystal(dock_result_XXXX):
    """Re-dock XXXX; best pose RMSD to crystal < 2.0 Å."""
    best = dock_result_XXXX.poses[0]
    rmsd = compute_rmsd(best.coords, crystal_coords)
    assert rmsd < 2.0, f"RMSD={rmsd:.2f} Å"
    # Also lock score to detect scoring drift
    assert best.score == pytest.approx(EXPECTED_SCORE, abs=0.1)
```

**Golden value updates** should be treated like schema migrations — intentional,
reviewed, with a clear reason in the commit message.

---

## 3. C++ Unit Tests — Scoring Terms

### Existing coverage (test_vina.cpp)

All five Vina terms have energy + force assertions at 3–4 distances:

```cpp
// test_vina.cpp — existing pattern
TEST_CASE("vina_gaussian1 energy and force", "[vina][gauss1]") {
    Real f = 0;
    Real e = vina_gaussian1(0.0, &f);
    REQUIRE_THAT(e, WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(f, WithinAbs(0.0, 1e-6));
}
```

### What to add: multi-type parametrized tests

The existing tests cover one atom type combination. Each term should be tested
across diverse atom type pairs:

```cpp
// Proposed: parametrize across HBond-capable type pairs
TEST_CASE("eval_ef hbond active for donor-acceptor pairs", "[vina][hbond]") {
    // N_D donor + O_A acceptor → should produce hbond energy
    auto [e, f] = eval_ef_at_distance(-0.5, XS_TYPE_N_D, XS_TYPE_O_A);
    REQUIRE(e < 0);  // favorable

    // C_H + C_H → no hbond
    auto [e2, f2] = eval_ef_at_distance(-0.5, XS_TYPE_C_H, XS_TYPE_C_H);
    // hbond component should be zero
}

TEST_CASE("eval_ef hydrophobic only for hydrophobic atoms", "[vina][hydrophobic]") {
    // C_H + C_H → hydrophobic active
    auto [e1, f1] = eval_ef_at_distance(0.2, XS_TYPE_C_H, XS_TYPE_C_H);
    REQUIRE(e1 < 0);

    // N_D + O_A → hydrophobic inactive
    auto [e2, f2] = eval_ef_at_distance(0.2, XS_TYPE_N_D, XS_TYPE_O_A);
    // hydrophobic component should be zero
}
```

### Angular HBond — existing excellent coverage

The angular HBond tests (`test_angular_hbond.cpp`, `test_hbond_angle.cpp`,
`test_rescore_hbond.cpp`) are the gold standard in this codebase:

- Weight function at known angles with derivative checks
- Finite-difference gradient validation at 3 geometries per direction
- Cross-validation: angle correction == eval_ef_decomp hbond energy
- BondedHTopo construction with edge cases (no H, too far, multiple H)
- RecDonorHTopo manual construction
- rescore_hbond boundary cases (angle just above/below 100°)

**These 28 test cases should be the model for any new scoring term.**

---

## 4. Property-Based Tests — Invariants

### Translation invariance

Not currently tested. Catching absolute-coordinate bugs.

```cpp
TEST_CASE("scoring invariant to system translation", "[invariant]") {
    // Score original system
    Real e1 = cal_e_f_cpu(pose, flex_param, fix_param, fix_coords, ...);

    // Translate everything by (100, 100, 100)
    translate_system(pose, fix_coords, 100.0, 100.0, 100.0);
    shift_box(100.0, 100.0, 100.0);
    rebuild_spatial_grid();
    Real e2 = cal_e_f_cpu(pose, flex_param, fix_param, fix_coords, ...);

    REQUIRE_THAT(e1, WithinAbs(e2, 1e-4));
}
```

### Rotation invariance

```cpp
TEST_CASE("scoring invariant to system rotation", "[invariant]") {
    Real e1 = cal_e_f_cpu(original_system...);

    // Rotate entire system by 90° around Z
    rotate_system(original_system, axis_z, 90.0);
    rebuild_spatial_grid();
    Real e2 = cal_e_f_cpu(rotated_system...);

    REQUIRE_THAT(e1, WithinAbs(e2, 1e-4));
}
```

### Pairwise symmetry (already implicit in eval_ef)

```cpp
TEST_CASE("eval_ef symmetric in atom types", "[invariant][vina]") {
    Real f1 = 0, f2 = 0;
    Real e1 = eval_ef(1.0, XS_TYPE_N_D, XS_TYPE_O_A, &f1, g_atp);
    Real e2 = eval_ef(1.0, XS_TYPE_O_A, XS_TYPE_N_D, &f2, g_atp);
    REQUIRE_THAT(e1, WithinAbs(e2, 1e-10));
}
```

### Boundedness

```cpp
TEST_CASE("no NaN or Inf for valid distances", "[invariant]") {
    for (Real d = 0.1; d < 20.0; d += 0.1) {
        Real f = 0;
        Real e = eval_ef(d - R1_PLUS_R2, XS_TYPE_C_H, XS_TYPE_O_A, &f, g_atp);
        REQUIRE(std::isfinite(e));
        REQUIRE(std::isfinite(f));
    }
}
```

### Monotonicity: angular HBond weight

```cpp
TEST_CASE("angular weight monotonically increases toward 180°", "[invariant][angular]") {
    Real prev_w = 0;
    for (Real angle_deg = 100; angle_deg <= 180; angle_deg += 5) {
        Real cos_dha = std::cos(angle_deg * M_PI / 180.0);
        Real dw;
        Real w = angular_hbond_weight(cos_dha, &dw);
        REQUIRE(w >= prev_w);
        prev_w = w;
    }
}
```

### Idempotency: re-minimizing a minimized pose

```python
# Python integration
def test_reminimize_is_idempotent(minimized_pose):
    """BFGS on an already-converged pose should change nothing."""
    result = optimize(minimized_pose, opt_steps=15)
    rmsd = compute_rmsd(result.coords, minimized_pose.coords)
    assert rmsd < 0.01, f"Re-minimization moved pose by {rmsd:.3f} Å"
    assert abs(result.score - minimized_pose.score) < 0.01
```

---

## 5. Gradient Validation — Finite Difference

The standard pattern in this codebase. Every scoring term must pass FD checks.

### Existing pattern (from test_angular_hbond.cpp)

```cpp
TEST_CASE("angular hbond gradient matches finite difference", "[angular][gradient]") {
    // Evaluate analytic gradient
    Real e0 = cal_e_f_cpu(pose, ..., forces);

    // For each movable coordinate:
    const Real h = 1e-4;
    for (int i = 0; i < natom * 3; ++i) {
        pose_coords[i] += h;
        Real e_plus = cal_e_f_cpu(pose, ..., nullptr);
        pose_coords[i] -= 2 * h;
        Real e_minus = cal_e_f_cpu(pose, ..., nullptr);
        pose_coords[i] += h;  // restore

        Real fd = (e_plus - e_minus) / (2 * h);
        REQUIRE_THAT(forces[i], WithinAbs(-fd, 1e-3));
    }
}
```

### Where FD checks exist

| Component | File | Status |
|---|---|---|
| Angular HBond (direction 1) | test_angular_hbond.cpp | 3 geometries |
| Angular HBond (direction 2) | test_angular_hbond.cpp | 3 geometries |
| Angular weight function | test_angular_hbond.cpp | Analytic vs FD |
| FF LJ | test_gaff2.cpp | Single case |
| FF Coulomb | test_gaff2.cpp | Single case |
| FF Pair14 | test_gaff2.cpp | Single case |
| FF Torsion | test_gaff2.cpp | Single case |
| Dihedral forces | test_score_ff_cpu_detailed.cpp | Real DMS data |
| cal_e_f_cpu (full) | **Missing** | **Gap** |
| Bias (pos/align) | test_bias.cpp | Per-coordinate FD |

### What to add: full cal_e_f_cpu gradient check

```cpp
TEST_CASE("cal_e_f_cpu full gradient FD check on ejm42", "[cpu][gradient]") {
    // Load TYK2 + ejm42 from DMS
    auto [fix, flex] = load_test_system("tyk2", "ejm42");
    // Place ligand at a reasonable pose
    set_pose_at_crystal(flex);

    // Analytic
    std::vector<Real> forces(flex.natom * 3, 0);
    Real e0 = cal_e_f_cpu(flex, fix, forces.data());

    // FD per coordinate
    const Real h = 1e-4;
    for (int i = 0; i < flex.natom * 3; ++i) {
        flex.coords[i] += h;
        Real ep = cal_e_f_cpu(flex, fix, nullptr);
        flex.coords[i] -= 2 * h;
        Real em = cal_e_f_cpu(flex, fix, nullptr);
        flex.coords[i] += h;

        Real fd = (ep - em) / (2 * h);
        REQUIRE_THAT(forces[i], WithinAbs(-fd, 5e-3));
    }
}
```

---

## 6. Contract Tests — DMS Enrichment

The DMS file is the single data interface between Python and C++. Silent
mis-enrichment is the most dangerous bug class.

### Existing coverage

- `test_dms_reader.cpp`: reads ligand + receptor DMS, validates atom counts, LJ
  params, charges, 1-4 pairs, torsions, exclusions
- `test_verify_snapshots.py`: golden file regression for `ud2_*` table contents
- `test_dms_schema.py`: schema validation for enrichment tables

### Schema contract: required tables and columns

```python
REQUIRED_LIGAND_TABLES = {
    "ud2_atom_props":     ["atom_id", "vina_type"],
    "ud2_torsions":       ["torsion_id", "atom_i", "atom_j", "atom_k", "atom_l",
                           "dihedral_value", "mobile_atoms"],
    "ud2_root_atoms":     ["atom_id"],
    "ud2_metadata":       ["key", "value"],
    "ud2_sdf_properties": ["key", "value"],
}

REQUIRED_RECEPTOR_TABLES = {
    "ud2_atom_props": ["atom_id", "vina_type"],
}

def test_enriched_dms_has_required_tables(enriched_lig_dms, enriched_rec_dms):
    """All ud2_* enrichment tables must exist with correct schema."""
    for table, columns in REQUIRED_LIGAND_TABLES.items():
        assert table_exists(enriched_lig_dms, table)
        assert table_columns(enriched_lig_dms, table) == columns
```

### Vina type consistency: Python == C++

```python
# unidock_processing/test/scripts/test_atom_type_props.py — existing
def test_python_vina_types_match_cpp():
    """VINA_ATOM_TYPE_PROPERTIES in Python must match C++ constants.h."""
    # Compares VDW radii, boolean properties across all 21 standard types
```

### prepare_dms_lightweight contract

```python
def test_lightweight_dms_has_all_enrichment_tables():
    """prepare_dms_lightweight must produce fully enriched DMS."""
    from unidock_processing.unidocktools.unidock_protocol_runner import (
        prepare_dms_lightweight,
    )
    lig_dms, rec_dms = prepare_dms_lightweight(sdf_path, receptor_path, tmp_dir)

    # Ligand DMS
    assert table_exists(lig_dms, "ud2_atom_props")
    assert table_exists(lig_dms, "ud2_torsions")
    assert table_exists(lig_dms, "ud2_root_atoms")
    assert row_count(lig_dms, "ud2_atom_props") > 0

    # Receptor DMS (enriched in-place on receptor_parameterized.dms)
    assert table_exists(rec_dms, "ud2_atom_props")
    assert row_count(rec_dms, "ud2_atom_props") > 0
```

### SDF output roundtrip

```python
# tests/test_sdf_output.py — existing pattern
def test_output_sdf_parseable_by_rdkit(docking_result_sdf):
    """Output SDF must be valid — parseable by RDKit."""
    suppl = Chem.SDMolSupplier(str(docking_result_sdf), removeHs=False)
    mols = [m for m in suppl if m is not None]
    assert len(mols) > 0
    for mol in mols:
        assert mol.GetNumAtoms() > 0
        assert mol.HasProp("vina_binding_free_energy")
        val = float(mol.GetProp("vina_binding_free_energy"))
        assert np.isfinite(val)
```

### Bond order and stereo preservation

```python
def test_sdf_output_preserves_bond_orders(dock_aromatic_ligand):
    """Bond orders from input SDF must survive DMS roundtrip."""
    input_mol = Chem.SDMolSupplier(INPUT_SDF, removeHs=False)[0]
    output_mol = Chem.SDMolSupplier(OUTPUT_SDF, removeHs=False)[0]
    for bond_in, bond_out in zip(input_mol.GetBonds(), output_mol.GetBonds()):
        assert bond_in.GetBondTypeAsDouble() == bond_out.GetBondTypeAsDouble()
```

---

## 7. Nondeterminism Control

### Seed convention

All stochastic tests use `seed=42` as the default. This is enforced throughout:

```python
FAST_PARAMS = dict(
    search_mode="fast",
    exhaustiveness=1,
    mc_steps=0,
    opt_steps=0,
    seed=42,
    num_pose=1,
)
```

### Deterministic replay (existing)

```python
# tests/test_reproducibility.py
def test_same_seed_produces_same_results():
    r1 = run_docking(**params, seed=42)
    r2 = run_docking(**params, seed=42)
    assert r1.score == pytest.approx(r2.score, abs=1e-8)
    assert np.allclose(r1.coords, r2.coords)
```

### Cross-seed convergence

```python
def test_different_seeds_converge():
    """Different seeds should find similar best scores for an easy system."""
    scores = []
    for seed in [1, 2, 3, 4, 5]:
        result = run_docking(**params, seed=seed, exhaustiveness=32)
        scores.append(result.poses[0].score)
    spread = max(scores) - min(scores)
    assert spread < 2.0, f"Score spread={spread:.2f} kcal/mol"
```

### RDKit molecule fixtures

The ud2align unit fixtures (`conftest.py`) create molecules via
`AllChem.EmbedMolecule()` without explicit seeds. For reproducibility:

```python
@pytest.fixture
def simple_mol():
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)  # fix seed
    return mol
```

---

## 8. Performance Budget Tests

No explicit time-based assertions currently exist. The `@pytest.mark.slow` marker
separates expensive tests but does not enforce time limits.

### Proposed: add time budgets to key operations

```python
import time

@pytest.mark.slow
def test_single_ligand_docking_under_budget():
    """TYK2 ejm42 (35 atoms) must dock in < 30s on CPU."""
    start = time.perf_counter()
    run_docking(
        receptor=TYK2_RECEPTOR, ligand=TYK2_LIGAND,
        search_mode="balance", seed=42,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"Took {elapsed:.1f}s, budget is 30s"

def test_score_only_under_budget():
    """Score-only mode must complete in < 2s."""
    start = time.perf_counter()
    run_docking(
        receptor=TYK2_RECEPTOR, ligand=TYK2_LIGAND,
        task="score", seed=42,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"Took {elapsed:.1f}s, budget is 2s"
```

### Scaling tests

```python
def test_scoring_scales_linearly_with_ligand_atoms():
    """Scoring time should not grow worse than O(n) with ligand size."""
    times = {}
    for ligand, natom in [("small1.dms", 17), ("ejm42.dms", 35)]:
        start = time.perf_counter()
        run_docking(receptor=TYK2_RECEPTOR, ligand=ligand, task="score")
        times[natom] = time.perf_counter() - start

    ratio = times[35] / times[17]
    # Should scale roughly linearly (allow 3x for overhead)
    assert ratio < 3.0 * (35 / 17), f"Time ratio {ratio:.1f} suggests superlinear scaling"
```

---

## 9. CPU vs GPU Consistency

Angular HBond scoring is CPU-only. The GPU path uses distance-only Vina HBond.
This asymmetry requires explicit testing.

### Final score consistency

```python
def test_cpu_gpu_scores_agree_for_non_hbond_system():
    """For a system with no H-bonds, CPU and GPU scores must match."""
    cpu_result = run_docking(receptor=R, ligand=HYDROPHOBIC_LIG, backend="cpu", seed=42)
    gpu_result = run_docking(receptor=R, ligand=HYDROPHOBIC_LIG, backend="gpu", seed=42)
    # run_score always on CPU, so final scores should match
    assert cpu_result.score == pytest.approx(gpu_result.score, abs=0.01)

def test_cpu_gpu_run_score_uses_angular_hbond():
    """run_score() should apply angular HBond regardless of sampling backend."""
    result = run_docking(receptor=R, ligand=HBOND_LIG, backend="gpu", seed=42)
    # The atom_hbond decomposition should show angular modulation
    hbond_energies = parse_atom_hbond(result.poses[0])
    # At least some atoms should have modified (not full) hbond credit
    # if angular scoring is active
```

### Documenting the known asymmetry

```python
def test_gpu_sampling_uses_distance_only_hbond():
    """
    GPU MC/BFGS uses distance-only HBond (by design).
    This test documents the expected behavior, not a bug.
    If GPU angular HBond is implemented, update this test.
    """
    # Run with exhaustiveness=1 to get a single MC pose
    # Compare MC-stage energy (pre-run_score) vs final energy
    # The difference reveals angular correction magnitude
```

---

## 10. Test Data and Organization

### Existing test data

```
unidock_engine/test/data/
├── ff_minimize/tyk2/              # TYK2: protein + 3 ligands (DMS, SDF, PDB)
│   ├── protein.dms, protein.pdb
│   ├── ligand_ejm42.dms, ligand_ejm42.sdf
│   ├── ligand_small1.dms, ligand_small2.dms
│   └── *_enriched.dms             # Pre-enriched variants
├── dev1/                          # Development test case
└── ff/                            # Force field parameters

unidock_processing/test/data/
├── free_docking/molecular_docking/  # 1G9V: receptor + ligand
├── free_docking/virtual_screening/  # 5WIU: virtual screening set
├── constraint_docking/              # Template docking data
├── covalent_docking/1EWL/           # Covalent bond docking
├── receptor_topology/               # Receptor prep test data
└── ligand_topology/                 # Ligand prep test data

```

### Naming conventions

- **Name files for intent**: `ligand_ejm42.sdf` not `lig_003.sdf`
- **Golden outputs use `_old` suffix**: `res_zalign_old.sdf` (locked reference)
- **New outputs use `_new` suffix**: `res_zalign_new.sdf` (generated at test time)

### Test molecule fixtures (external ud2align)

```python
# In the external ud2align package
@pytest.fixture
def simple_mol():      # Ethanol — minimal HBond donor/acceptor
def benzene_mol():     # Aromatic, no HBond
def charged_mol():     # Carboxylate — formal charge
def amine_mol():       # Methylamine — HBond donor
def ketone_mol():      # Acetone — HBond acceptor
```

### Adding new test data

1. Keep test systems **small** — 50 residue binding site, not full 500 residue protein
2. Include both DMS (for C++ tests) and PDB/SDF (for Python tests)
3. Commit expected outputs to version control
4. Document the source (PDB ID, resolution, preparation steps)

---

## 11. Test Tiers and Execution

### Current markers

| Marker | Meaning | Usage |
|---|---|---|
| `@pytest.mark.slow` | Requires full docking run (> 60s) | Integration tests |
| (no marker) | Fast tests (< 5s each) | Unit, schema, parsing |

### Proposed tier system

| Tier | Budget | What | When |
|---|---|---|---|
| **Unit** | < 10s total | C++ Catch2 + Python unit tests | Every build / save |
| **Component** | < 60s | DMS roundtrip, score-only, seed determinism | Pre-commit |
| **Integration** | < 10min | Re-docking golden cases, RMSD checks, output validation | CI / PR |
| **Benchmark** | < 1hr | Multi-system enrichment, cross-seed convergence, performance budgets | Nightly |

### Execution commands

```bash
# Tier 1: Unit (fast)
./build_cpu/test/unit/unit_test                    # C++ unit

# Tier 2: Component
python -m pytest tests/ -m "not slow" -v           # Fast integration
python -m pytest unidock/unidock_processing/test/scripts/snapshots/ -v  # Snapshots

# Tier 3: Integration
python -m pytest tests/ -v                         # All integration

# Tier 4: Benchmark
python -m pytest tests/ -m "slow" -v               # Slow integration only
```

### Fixture scoping strategy

Expensive docking runs are session- or module-scoped to amortize cost:

```python
@pytest.fixture(scope="module")
def tyk2_docking_result():
    """Run TYK2 docking once, reuse across all tests in module."""
    return run_docking(receptor=TYK2_REC, ligand=TYK2_LIG, seed=42)
```

Multiple tests then assert different properties of the same result:

```python
def test_score_is_finite(tyk2_docking_result):
    assert np.isfinite(tyk2_docking_result.score)

def test_atom_count_preserved(tyk2_docking_result):
    assert tyk2_docking_result.natom == EXPECTED_NATOM

def test_coordinates_inside_box(tyk2_docking_result):
    assert all_coords_inside_box(tyk2_docking_result.coords, BOX)
```

---

## 12. Coverage Gaps and Priorities

### Critical gaps (should be addressed first)

| Gap | Current state | What to add |
|---|---|---|
| **cal_e_f_cpu full FD** | Individual terms only | Full-system FD gradient check on real DMS data (TYK2 ejm42) |

**Recently addressed:**
- **MC search** — `test_mc_cpu.cpp`: energy decrease, timeout, invariants
- **BFGS convergence** — `test_cpu_bfgs.cpp`: never-increase, idempotency, restore-on-failure
- **Spatial grid** — `test_spatial_grid.cpp`: construction, neighbor query, boundary atoms
- **Custom atom types** — `test_custom_atom_type.cpp`: VDW radii, boolean props, scoring
- **Custom FF params** — `test_custom_ff_param.cpp`: sigma/epsilon/charge/torsion/1-4 pairs
- **FD gradients** — `test_cal_e_f_gradient.cpp`: translation/rotation invariance, box penalty FD

### High priority gaps

| Gap | What to add |
|---|---|
| ~~Translation invariance~~ | ~~Addressed in test_cal_e_f_gradient.cpp~~ |
| ~~Rotation invariance~~ | ~~Addressed in test_cal_e_f_gradient.cpp~~ |
| ~~Box penalty gradient~~ | ~~Addressed in test_cal_e_f_gradient.cpp~~ |
| **DockTask run() sequencing** | Test that stages run in correct order |
| **Multi-ligand batching** | Verify batch boundaries don't affect scores |

### Medium priority gaps

| Gap | What to add |
|---|---|
| **Torsion library ranges** | Random angle falls within allowed ranges |
| **Ligand size classes** | Small/medium/large/extra classification is correct |
| **PROTAC clustering** | Custom cluster_atom_indices produces expected clusters |
| **DMS merge** | ud2_molecules table correct after merging N ligands |
| **Error handling** | Corrupt DMS, empty SDF, missing receptor — graceful failures |

---

## 13. Anti-Patterns

### Testing only the happy path

```python
# Incomplete — only tests ideal input
def test_dock():
    result = run_docking(receptor=NICE_REC, ligand=NICE_LIG, seed=42)
    assert result.poses[0].score < 0

# Missing: What about empty SDF? Ligand outside box? Clashing initial pose?
```

### Asserting "not None" instead of correctness

```python
# Bad — passes even if score is completely wrong
def test_scoring():
    result = run_docking(...)
    assert result is not None

# Good — asserts against locked reference
def test_scoring():
    result = run_docking(receptor=TYK2_REC, ligand=TYK2_LIG, seed=42)
    assert result.poses[0].score == pytest.approx(-8.34, abs=0.1)
```

### Tolerance too loose

```python
# Bad — accepts 50% error
assert energy == pytest.approx(expected, rel=0.5)

# Good — appropriate for the quantity
assert energy == pytest.approx(expected, abs=0.01)    # kcal/mol
assert rmsd == pytest.approx(expected, abs=0.001)      # sub-mÅ for reproducibility
```

### No seed → flaky test

```python
# Bad — fails randomly
def test_dock():
    result = run_docking(receptor=R, ligand=L)  # no seed!
    assert result.poses[0].score < -5.0

# Good — deterministic
def test_dock():
    result = run_docking(receptor=R, ligand=L, seed=42)
    assert result.poses[0].score < -5.0
```

### AI-generated expected values without ground truth

```python
# Dangerous — where did -7.23 come from?
assert score == pytest.approx(-7.23)

# Safe — value from prior validated run, documented
# Expected: -8.34 from v2.1 validated against crystal RMSD < 1.5 Å
assert score == pytest.approx(-8.34, abs=0.1)
```

### Duplicating work in fixtures

```python
# Bad — runs docking 10 times for 10 tests
def test_score(run_docking_fresh):  ...
def test_rmsd(run_docking_fresh):   ...
def test_atoms(run_docking_fresh):  ...

# Good — run once, assert many properties
@pytest.fixture(scope="module")
def docking_result():
    return run_docking(...)

def test_score(docking_result):  ...
def test_rmsd(docking_result):   ...
def test_atoms(docking_result):  ...
```

### Ignoring the CPU/GPU angular HBond gap

```python
# Misleading — claims "docking accuracy" but only tests final CPU rescore
def test_hbond_accuracy():
    result = run_docking(backend="gpu", ...)
    assert hbond_energy < 0  # True, but GPU sampling didn't optimize for it

# Honest — documents what is and isn't tested
def test_hbond_in_final_rescore():
    """Angular HBond is applied at run_score() (CPU). GPU sampling is distance-only."""
    ...
```

---

## Summary

| Principle | Action | Priority |
|---|---|---|
| Lock golden results | RMSD + score snapshots for TYK2, 1G9V, 7SJ3, 1E9 | **Critical** |
| FD gradient checks | Every scoring term: Vina, angular HBond, FF, bias, box | **Critical** |
| Verify invariants | Translation/rotation invariance, symmetry, boundedness | **Critical** |
| Control randomness | `seed=42` everywhere; test deterministic replay | **Critical** |
| Validate DMS contracts | Schema checks, table existence, column types | **High** |
| Fill MC/BFGS gap | Test search quality and optimizer convergence | **High** |
| Guard coordinate math | Quaternion, rotation, RMSD unit tests (already good) | **Maintained** |
| CPU/GPU consistency | Document angular HBond gap; test final-score equivalence | **High** |
| Set time budgets | Score-only < 2s, single docking < 30s | **Medium** |
| Keep data small and named | 50-residue pockets, intent-named files | **Ongoing** |

> **Ground truth ownership**: Crystal structures, experimental binding data, and
> physics invariants define correctness. Golden values come from validated runs
> against those references — not from the code itself. When values change,
> the commit message must explain why.
