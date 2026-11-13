# Uni-Dock2 Docking Flow

A comprehensive reference for the end-to-end molecular docking pipeline,
from raw input files through the C++ engine to final output poses.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Input Files](#2-input-files)
3. [Python Preprocessing Layer](#3-python-preprocessing-layer)
   - 3.1 [Classes and Responsibilities](#31-classes-and-responsibilities)
   - 3.2 [Receptor Preparation](#32-receptor-preparation)
   - 3.3 [Ligand Preparation](#33-ligand-preparation)
   - 3.4 [DMS Enrichment Tables](#34-dms-enrichment-tables)
   - 3.5 [Engine Invocation](#35-engine-invocation)
   - 3.6 [Working Directory and Intermediate Files](#36-working-directory-and-intermediate-files)
4. [Intermediate File Formats](#4-intermediate-file-formats)
5. [C++ Engine Entry Points](#5-c-engine-entry-points)
6. [Core Docking Algorithm](#6-core-docking-algorithm)
7. [Scoring Function](#7-scoring-function)
8. [Pose Clustering and Filtering](#8-pose-clustering-and-filtering)
9. [Output Generation](#9-output-generation)
   - 9.1 [SDF Output](#91-sdf-output)
   - 9.2 [Molecule Naming](#92-molecule-naming)
10. [Special Docking Modes](#10-special-docking-modes)
11. [Build System](#11-build-system)
12. [Key Files Reference](#12-key-files-reference)

---

## 1. High-Level Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  USER INPUT                                                      │
│  Receptor: PDB / DMS         Ligands: SDF / DMS                  │
│  Config:   YAML              (box center, size, search_mode …)   │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│  PYTHON PREPROCESSING  (unidock_processing)                      │
│                                                                  │
│  1. Receptor topology  →  enriched receptor DMS                  │
│  2. Ligand topology    →  enriched ligand DMS                    │
│  3. (Optional) SDF → DMS with force-field via antechamber/tleap  │
│  4. DMS enrichment: write ud2_* tables                           │
│  5. Generate YAML config for C++ engine                          │
└───────────────────────────────┬──────────────────────────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
   ┌──────────────────┐               ┌───────────────────┐
   │  ud2 binary       │               │  pipeline.so       │
   │  (subprocess)     │               │  (pybind11)        │
   │  reads YAML →     │               │  direct call →     │
   │  reads DMS        │               │  reads DMS         │
   └────────┬─────────┘               └────────┬──────────┘
            │                                   │
            └────────────────┬──────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  C++ DOCKING ENGINE                                              │
│                                                                  │
│  run_screening()                                                 │
│    ├── Batch ligands by size / memory                            │
│    └── Per batch: DockTask::run()                                │
│          ├── prepare_vina()      pre-compute vdW radius sums     │
│          ├── run_search()        MC + BFGS  (exhaustiveness N)   │
│          ├── run_cluster()       RMSD-based greedy clustering    │
│          ├── run_refine()        (optional BFGS on clusters)     │
│          ├── run_filter()        top-N by energy_range           │
│          ├── run_place_h()       cone-donor H toward acceptors   │
│          ├── run_ff_minimize()   (optional force-field)          │
│          ├── run_score()         final Vina + decomposition      │
│          └── dump_poses()        write SDF                       │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
                   ┌──────────────────────┐
                   │  DMS path output      │
                   │  <name>_N.sdf         │
                   │  (per batch)          │
                   └────────┬─────────────┘
                            │
                            ▼
                   ┌──────────────────────┐
                   │  Python: concatenate  │
                   │  batch SDFs           │
                   │  → unidock2_pose.sdf  │
                   └──────────────────────┘
```

---

## 2. Input Files

### 2.1 Receptor

| Format | Description |
|--------|-------------|
| **PDB** | Standard protein structure. Python layer converts to DMS via PDBFixer → tleap → msys. |
| **DMS** | Pre-prepared receptor (SQLite). Skips conversion. Must contain `particle`, `bond` tables. |

### 2.2 Ligands

| Format | Description |
|--------|-------------|
| **SDF** | One or more molecules. RDKit reads them; each gets a unique internal name (`MOL_0`, `MOL_1`, …). Original `_Name` property is preserved for output. |
| **DMS** | Pre-prepared ligand(s) with optional force-field parameters. Can be single-ligand or merged multi-ligand (tracked via `ud2_molecules` table). |

### 2.3 Configuration (YAML)

There are two YAML layers. Users write the **Python-layer YAML**, which
accepts raw PDB/SDF. The Python preprocessing layer then generates the
**C++ engine YAML** internally to invoke the `ud2` binary.

**User-facing YAML** (read by Python `io/yaml.py`):

```yaml
Required:
  receptor: receptor.pdb          # PDB or DMS
  ligand: ligands.sdf             # SDF or DMS
  center: [15.0, -3.0, 0.5]

Settings:
  box_size: [22.5, 22.5, 22.5]
  task: screen                    # screen | score | mc
  search_mode: balance            # fast | balance | detail | free

Advanced:
  exhaustiveness: 512
  randomize: true
  mc_steps: 40
  opt_steps: -1                   # -1 = heuristic (resolves to 15 at runtime)
  refine_steps: 5
  num_pose: 10
  rescore_pool: 10                # cluster representatives re-ranked by the pose pass; num_pose of them are written
  rmsd_limit: 1.0
  energy_range: 5.0
  seed: 1234567
  use_tor_lib: false

Hardware:
  gpu_device_id: 0
```

**C++ engine YAML** (read by `main.cpp`, generated by Python layer):

```yaml
Inputs:
  ligand_dms_full: ./ligand.dms       # Enriched DMS input
  receptor_dms_full: ./receptor.dms   # Enriched receptor DMS

Settings:
  center_x: 15.0
  center_y: -3.0
  center_z: 0.5
  size_x: 22.5
  size_y: 22.5
  size_z: 22.5
  task: screen                    # screen | score | mc | randomize | optimize
  search_mode: balance            # fast | balance | detail | free
  constraint_docking: false

Advanced:
  exhaustiveness: 512
  randomize: true
  mc_steps: 40
  opt_steps: -1                   # -1 = heuristic (resolves to 15 at runtime)
  refine_steps: 5
  num_pose: 10
  rescore_pool: 10                # cluster representatives re-ranked by the pose pass; num_pose of them are written
  rmsd_limit: 1.0
  energy_range: 10.0
  seed: 1234567
  tor_lib: false
  tor_prec: 0.3
  box_prec: 2.0
  cluster_atom_indices: []        # per-ligand custom atom indices
  bias: no                        # no | pos | align
  bias_k: 0.1

Hardware:
  ncpu: 10                        # OpenMP threads (default: all detected CPUs)
  gpu_device_id: 0

Outputs:
  dir: ./output
```

**Key differences between the two YAMLs:**

| | Python-layer YAML | C++ engine YAML |
|---|---|---|
| Inputs | `Required.receptor` (PDB/DMS), `Required.ligand` (SDF/DMS) | `Inputs.ligand_dms_full` + `Inputs.receptor_dms_full` |
| Center | `Required.center: [x, y, z]` | `Settings.center_x/y/z` (separate keys) |
| Box size | `Settings.box_size: [x, y, z]` | `Settings.size_x/y/z` (separate keys) |
| Torsion lib | `Advanced.use_tor_lib` | `Advanced.tor_lib` |
| Energy range default | 5.0 | 10.0 |
| Extra C++ keys | — | `tor_prec`, `box_prec`, `bias`, `bias_k`, `cluster_atom_indices` |

**Search mode presets:**

Binary (`ud2`, preferred path) — `opt_steps` defaults to -1 (heuristic),
resolved to 15 by `screening.cpp` at runtime:

| Mode | exhaustiveness (CPU) | mc_steps |
|------|---------------------|----------|
| fast | 64 | 20 |
| balance | 128 | 30 |
| detail | 256 | 40 |
| free | (user values) | (user values) |

Pybind11 (`pipeline.so`, fallback path):

| Mode | exhaustiveness | mc_steps | opt_steps |
|------|---------------|----------|-----------|
| fast | 64 | 30 | 3 |
| balance | 64 | 200 | 5 |
| detail | 512 | 300 | 5 |
| free | (user values) | (user values) | (user values) |

**Note:** The two entry points use different preset values.

---

## 3. Python Preprocessing Layer

### 3.1 Classes and Responsibilities

| Class | File | Role |
|-------|------|------|
| `UnidockProtocolRunner` | `unidock_protocol_runner.py` | Top-level orchestrator |
| `UnidockReceptorTopologyBuilder` | `unidock_receptor_topology_builder.py` | Receptor atom-typing and DMS enrichment |
| `UnidockLigandTopologyBuilder` | `unidock_ligand_topology_builder.py` | Ligand torsion detection, atom-typing, parallel batch processing |
| `UnidockLigandPoseWriter` | `unidock_ligand_pose_writer.py` | Mol-list-based SDF writing: updates coordinates on original RDKit mols for output |

### 3.2 Receptor Preparation

```
PDB
 ├── PDBFixer (missing atoms/residues, CYS→CYX for disulfides)
 ├── tleap (ff14SB → PRMTOP/INPCRD)
 ├── msys.LoadPrmTop → msys System
 ├── msys.AssignBondOrderAndFormalCharge
 └── msys.SaveDMS → receptor_parameterized.dms
       │
       └── _enrich_receptor_dms()  (in-place on receptor_parameterized.dms)
             ├── Assign Vina atom types  (C, N, O, S, H, HD, …)
             └── Write ud2_atom_props table (vina_type per atom)
```

The receptor DMS is enriched in-place on `receptor_parameterized.dms`.
There is no separate `receptor_enriched.dms` copy.

### 3.3 Ligand Preparation

```
SDF (one or more molecules)
 │
 ├── RDKit: read molecules, assign _Name
 │
 ├── generate_batch_ligand_topology()   [parallel over ligands]
 │     ├── Assign Vina atom types  (internal RDKit-based typer)
 │     ├── Assign FF atom types (GAFF2):
 │     │     • DMS input (has_ff): reads GAFF2 types + LJ params from DMS
 │     │     • SDF input, ff_minimize_steps > 0: runs `antechamber -at gaff2`
 │     │     • SDF input, ff_minimize_steps = 0 (skip_gaff2): uses placeholder
 │     │       GAFF2 types ('c3') and RDKit Gasteiger charges, skipping
 │     │       antechamber entirely. AmberTools is NOT required for Vina-only
 │     │       docking.
 │     ├── Identify rotatable bonds → torsion tree
 │     ├── Build root atom set
 │     ├── Match torsion library ranges (if enabled)
 │     └── Aggregate topology summary dict
 │
 ├── _build_ligand_dms_from_sdf()
 │     ├── antechamber:  SDF → MOL2 (GAFF2 atom types)
 │     ├── parmchk2:     MOL2 → FRCMOD (missing parameters)
 │     ├── tleap:        MOL2 + FRCMOD → PRMTOP/INPCRD
 │     ├── msys + ParmEd: load PRMTOP → msys System
 │     └── msys.SaveDMS → ligand.dms
 │           (includes FF tables: nonbonded, pair_12_6_es,
 │            dihedral_trig, exclusion)
 │
 └── _produce_enriched_dms()
       ├── _enrich_ligand_dms():  write ud2_* tables
       └── _enrich_receptor_dms(): write ud2_atom_props
```

### 3.4 DMS Enrichment Tables

These `ud2_*` tables are written into the DMS SQLite file by Python
and read by the C++ engine:

| Table | Contents |
|-------|----------|
| `ud2_atom_props` | `(atom_id, vina_type)` per atom |
| `ud2_torsions` | `(torsion_id, atom_i/j/k/l, dihedral_value, mobile_atoms)` |
| `ud2_torsion_ranges` | `(torsion_id, range_lo, range_hi)` per allowed range |
| `ud2_root_atoms` | `(atom_id)` — root fragment atoms |
| `ud2_metadata` | `(key, value)` — `molecule_name`, `original_name` |
| `ud2_molecules` | `(molecule_id, mol_name, atom_offset, natom)` — merged DMS |
| `ud2_sdf_properties` | `(key, value)` — original SDF tags for round-trip |
| `ud2_formal_charges` | `(atom_id, formal_charge)` — non-zero only |
| `ud2_bond_stereo` | `(bond_p0, bond_p1, stereo)` — V2000 stereo flags; empty in practice (RDKit clears wedge directions on read), and the engine does not write them |

### 3.5 Engine Invocation

`UnidockProtocolRunner._run_engine()` invokes the C++ engine:

1. Write `_ud2_engine_config.yaml` with all parameters + DMS paths
2. `subprocess.run([ud2_bin, _ud2_engine_config.yaml])`
3. `ud2` reads YAML → reads enriched DMS → runs docking → writes SDF per batch
4. Python concatenates `unidock2_output/*.sdf` → `unidock2_pose.sdf`

The pybind11 entry point (`pipeline.so`) also accepts DMS paths directly.

#### Lightweight DMS Preparation (`prepare_dms_lightweight`)

A public helper in `unidock_protocol_runner.py` builds enriched ligand +
receptor DMS directly, without GAFF2 parameterization or throwaway docking.
Used by `ud2align`.

```python
def prepare_dms_lightweight(sdf_path, receptor_path, working_dir) -> (lig_dms, rec_dms)
```

Pipeline:

```
Ligand:
  _build_ligand_dms_minimal()
    → build_molecular_graph(skip_gaff2=True)
    → _enrich_ligand_dms()

Receptor:
  UnidockReceptorTopologyBuilder
    → generate_receptor_topology()
    → analyze_receptor_topology_vina_only()
    → _enrich_receptor_dms()  (in-place on receptor_parameterized.dms)
```

### 3.6 Working Directory and Intermediate Files

All intermediate files are created under the `working_dir_name` directory
(default: current directory). The exact set of files depends on the input
format and whether force-field minimization is enabled.

#### 3.6.1 DMS Input Path

```
working_dir/
 ├── receptor_parameterized.dms   # Receptor DMS enriched in-place with ud2_atom_props
 ├── ligand_enriched.dms          # (single ligand) Copy + ud2_* topology tables
 │   OR ligand_merged.dms         # (multi-ligand) Merged DMS + ud2_* tables
 ├── _ud2_engine_config.yaml      # Generated YAML for ud2 binary
 │
 ├── _receptor_atom_info.json     # Temporary: receptor atom types (subprocess)
 │                                  (deleted after use)
 │
 │   (If SDF→DMS conversion for FF, per ligand:)
 ├── _dms_input_MOL_0.sdf         # Temp single-mol SDF for antechamber
 ├── _dms_build__dms_input_MOL_0/  # Temp AmberTools working dir:
 │     ├── lig.sdf                  #   input for antechamber
 │     ├── lig.mol2                 #   antechamber output (GAFF2 types)
 │     ├── lig.frcmod               #   parmchk2 output (missing params)
 │     ├── lig.prmtop               #   tleap output (topology)
 │     ├── lig.inpcrd               #   tleap output (coordinates)
 │     └── lig.dms                  #   final per-ligand DMS
 │
 └── unidock2_output/             # C++ engine output directory
       ├── <name>_0.sdf             # Batch 0 SDF (direct from C++ engine)
       ├── <name>_1.sdf             # Batch 1 SDF (if multiple batches)
       └── ...

→ Final output: unidock2_pose.sdf   # Python concatenates all batch SDFs
```

**Key points:**
- The C++ engine writes SDF directly via `sdf_writer.cpp`.
  Python just concatenates the per-batch SDF files.
- `<name>` is derived from the ligand DMS filename stem
  (e.g. `ligand_enriched` or `ligand_merged`).

#### 3.6.2 DMS Input via Ligand DMS Only (no receptor DMS)

When the user provides ligand as DMS but receptor as PDB:

```
working_dir/
 ├── receptor_pdb_fixed.pdb       # PDBFixer output
 ├── receptor.prmtop              # tleap output
 ├── receptor.inpcrd              # tleap output
 ├── receptor_parameterized.dms   # msys conversion from prmtop, enriched in-place with ud2_atom_props
 ├── ligand_from_dms_0.sdf        # DMS→SDF via msys for RDKit processing
 ├── ligand_enriched.dms          # enriched ligand DMS
 ├── _ud2_engine_config.yaml
 │
 └── unidock2_output/
       └── <name>_0.sdf           # Direct SDF output
```

#### 3.6.3 Batch Output File Naming

The C++ engine processes ligands in batches (grouped by atom count to
fit in memory). Each batch produces one output file:

```
<name>_<batch_id>.sdf

  <name>     = ligand input filename stem
               e.g. "ligand_enriched" or "ligand_merged"
  <batch_id> = 0, 1, 2, ...  (sequential batch index)
```

For example, docking 100 ligands from `ligand_merged.dms` might produce:
```
unidock2_output/
  ├── ligand_merged_0.sdf     # batch 0 (ligands 0-31)
  ├── ligand_merged_1.sdf     # batch 1 (ligands 32-63)
  └── ligand_merged_2.sdf     # batch 2 (ligands 64-99)
```

Python concatenates them in sorted order to produce the final
`unidock2_pose.sdf`.

---

## 4. Intermediate File Formats

### 4.1 DMS (Enriched Input)

SQLite database read by `dms_reader.cpp`. Contains:

- **Core msys tables**: `particle`, `bond`, `nonbonded_param`, `nonbonded`,
  `exclusion`, `pair_12_6_es_term/param`, `dihedral_trig_term/param`
- **Uni-Dock2 enrichment tables**: `ud2_atom_props`, `ud2_torsions`,
  `ud2_torsion_ranges`, `ud2_root_atoms`, `ud2_metadata`, `ud2_molecules`,
  `ud2_sdf_properties`, `ud2_formal_charges`, `ud2_bond_stereo`

---

## 5. C++ Engine Entry Points

### 5.1 Binary Entry (`main.cpp`)

```
main(argc, argv)
 ├── Parse YAML config (yaml-cpp)
 ├── Read DMS files:
 │     read_ligand_dms_full()   → UDFlexMolList
 │     read_receptor_dms_full() → UDFixMol
 ├── Populate DockParam from YAML
 └── run_screening(fix_mol, flex_mol_list, …)
```

### 5.2 Pybind11 Entry (`pipeline.cpp`)

```python
pipeline.run_docking_pipeline(
    ligand_dms_path, receptor_dms_path, output_dir,
    center_x, center_y, center_z,
    size_x, size_y, size_z,
    task="screen", search_mode="balance",
    exhaustiveness=512, randomize=True,
    mc_steps=40, opt_steps=-1,
    refine_steps=5, num_pose=10,
    rmsd_limit=1.0, energy_range=5.0,
    seed=1234567,
    constraint_docking=False,
    use_tor_lib=False,
    cluster_atom_indices=[],     # list[list[int]], per-ligand
    decomp=False,
    ff_minimize_steps=0,
    bias="no",
    bias_k=0.0,
    gpu_device_id=0
)
```

Reads DMS via `read_ligand_dms_full()` and `read_receptor_dms_full()`,
then calls `run_screening()`.

---

## 6. Core Docking Algorithm

### 6.1 Screening Loop (`screening.cpp`)

```
run_screening()
 ├── Group ligands by size (small / medium / large / extra)
 ├── Batch ligands within memory limits
 └── Per batch:
       DockTask task(fix_mol, flex_mol_batch, dock_param, …)
       task.run()
```

### 6.2 DockTask Pipeline (`DockTask.cpp`)

```
DockTask::run()
 ├── prepare_vina()      Pre-compute r1+r2 radius sums for all atom pairs
 ├── alloc_gpu()         Allocate memory (GPU buffers or CPU arrays)
 │
 ├── run_search()        Monte Carlo + BFGS  ← main compute
 ├── run_cluster()       RMSD-based greedy clustering
 ├── run_refine()        Optional: BFGS on cluster representatives
 ├── run_filter()        Keep the re-ranking pool: max(rescore_pool, num_pose) within energy_range
 ├── run_place_h()       Turn cone-donor H (-OH, -NH3+, pyramidal -NH2) toward the credited acceptors (H only)
 ├── run_ff_minimize()   Optional: force-field minimization (GAFF2/ff14SB)
 ├── run_score()         Final Vina scoring with per-atom decomposition
 │
 └── dump_poses()        Write SDF output
```

### 6.3 Monte Carlo Search (`mc_cpu.cpp`)

Parallelized over `nflex × exhaustiveness` poses via OpenMP.

```
mc_cpu()
 └── #pragma omp parallel for
     └── mc_single_pose_cpu(pose_idx)

mc_single_pose_cpu():
 ├── randomize_pose_cpu()
 │     ├── Center:      random point in box (adjusted for gyration radius)
 │     ├── Orientation: Super-Fibonacci spiral (quasi-uniform on sphere)
 │     └── Torsions:    random from allowed ranges
 │
 └── MC loop (mc_steps iterations):
       ├── mutate_pose_cpu()
       │     Randomly pick one DOF:
       │       0 → translate (random vector, clamp to box)
       │       1 → rotate (random rotation scaled by amplitude/gyration_radius)
       │       2+ → rotate one torsion
       │
       ├── Quick energy check (large systems only):
       │     if natom ≥ 600 && E > 1000 → skip BFGS
       │
       ├── bfgs_with_memory_cpu() → bfgs_cpu()
       │     BFGS optimization (opt_steps iterations)
       │     └── cal_e_grad_cpu() → cal_e_f_cpu()
       │
       ├── H-bond angle correction:
       │     Remove favorable hbond E if D-H…A angle < 100°
       │
       ├── Metropolis acceptance (T = 1.2):
       │     Accept if ΔE < 0, or with probability exp(-ΔE/T)
       │     Always accept first step.
       │
       └── Track best pose (lowest energy seen)
```

### 6.4 BFGS Optimization (`bfgs_cpu.cpp`)

```
bfgs_cpu():
 ├── cal_e_grad_cpu() → initial energy + gradient
 ├── H = Identity matrix
 │
 └── Loop (max_steps):
       ├── Search direction:  p = -H × g
       ├── Line search (Armijo backtracking):
       │     α₀ = 1.0, shrink by 0.5 until sufficient decrease
       ├── Update pose:  x_new = x + α × p
       ├── New gradient:  cal_e_grad_cpu()
       ├── BFGS Hessian update:
       │     s = x_new - x,  y = g_new - g
       │     H = (I - s·yᵀ/(yᵀ·s)) H (I - y·sᵀ/(yᵀ·s)) + s·sᵀ/(yᵀ·s)
       ├── Convergence check: ||g|| < 1e-5
       └── Timeout check: abort if > 2s wall-time
```

### 6.5 Safety Mechanisms

| Mechanism | Value | Purpose |
|-----------|-------|---------|
| `ENERGY_CEILING` | 1000.0 kcal/mol | Skip BFGS for badly clashing poses (only when `natom ≥ 600`) |
| `POSE_TIMEOUT_SEC` | 2.0 s | Per-pose wall-time limit; aborts MC loop |
| `curl()` | v = 10.0 | Caps repulsive energies and squashes gradients to smooth landscape |
| `normalize_angle_cpu` | uses `fmod` + `isfinite` | Prevents infinite loops from Inf/NaN angles |
| `axis_norm` epsilon guard | `std::numeric_limits<Real>::epsilon()` | Prevents division-by-zero in gradient computation |

---

## 7. Scoring Function

### 7.1 Vina Scoring Terms (`score_cpu.cpp`)

All pairwise terms are distance-dependent, evaluated between ligand and
receptor atoms (inter) and within the ligand (intra).

| Term | Description | Distance dependence | Activation condition |
|------|-------------|-------------------|---------------------|
| **gauss1** | Attractive | exp(−(d/0.5)²) | Always |
| **gauss2** | Attractive | exp(−((d−3)/2)²) | Always |
| **repulsion** | Repulsive | d² if d < 0 | Always |
| **hydrophobic** | Hydrophobic contact | 1 if d < 0.5, linear ramp to 0 at 1.5 | Both atoms `is_hydrophobic` |
| **hbond** | H-bond (donor-acceptor) | 1 if d < −0.7, linear ramp to 0 at 0 | One `is_donor` AND other `is_acceptor` |

Where `d = |r₁−r₂| − (R_vdw1 + R_vdw2)` (surface distance).

**Data-driven design:** The Vina scoring function is fully data-driven.
VDW radii and boolean properties (is_hydrogen, is_hydrophobic, is_donor,
is_acceptor) are read from DMS at runtime and stored in `AtomTypeProps`
+ `vdw_radii[]`.  No atom type parameters are hardcoded in the scoring
hot path.  Standard Vina types use IDs 0–20 (defined in `constants.h`);
IDs 21–31 are free for user-defined types.
See Section 10.6 for details on injecting custom types.

**H-bond angle correction:** After scoring, if the D-H...A angle < 100 degrees,
the favorable hbond energy is removed. This prevents rewarding
geometrically poor hydrogen bonds.

**Angular HBond scoring** (modulating HBond energy by the D-H...A angle)
is available on the **CPU backend only**. The GPU backend uses
distance-only HBond scoring. See `docs/hbond_angular_scoring.md` for
details.

### 7.2 Energy Capping — `curl()`

```cpp
// Applied to every pairwise evaluation (both inter- and intra-molecular)
if (e > 0) {           // Only positive (repulsive) energies
    tmp = v / (v + e);  // v = 10.0
    e *= tmp;           // Caps energy at v
    f *= tmp * tmp;     // Squashes gradient quadratically
}
```

This prevents extreme gradients from stalling BFGS optimization
when atoms clash heavily.

### 7.3 Additional Energy Components

| Component | When active | Description |
|-----------|-------------|-------------|
| **Box penalty** | Always | Linear penalty for atoms outside the docking box: E = PENALTY_SLOPE × distance (slope = 10.0). Atom coordinates are clamped to box boundary during scoring. |
| **Torsion library penalty** | `tor_lib: true` (C++ YAML) / `use_tor_lib: true` (Python YAML) | Soft harmonic penalty: E = k × dist², k = 5.0 |
| **Position bias** | `bias=pos` | Attractive spring to reference atom positions |
| **Alignment bias** | `bias=align` | Similarity-matrix-weighted alignment potential |
| **FF scoring** | When DMS with FF loaded | LJ 12-6 + Coulomb + GAFF2 torsions |

### 7.4 Spatial Grid for Receptor

The receptor is pre-binned into a 3D grid (cell size = 8.0 Å) for
O(N_lig × ~27) neighbor lookups instead of O(N_lig × N_rec).

```
Build:
  1. Count atoms per cell
  2. Prefix-sum → cell_start array
  3. Place atoms into flat array sorted by cell

Query (per ligand atom):
  1. Compute cell index from ligand atom position
  2. Loop over 27 neighboring cells (3×3×3)
  3. Evaluate pairwise terms for atoms in those cells
```

---

## 8. Pose Clustering and Filtering

### 8.1 Clustering (`cluster_cpu.cpp`)

Per-ligand greedy clustering:

```
1. Sort all exhaustiveness poses by energy (ascending)
2. For the best-energy pose:
     a. Mark it as a cluster center
     b. Compute RMSD to all remaining poses
     c. Mark poses with RMSD < rmsd_limit as belonging to this cluster
3. Repeat step 2 for the next-best unclustered pose
4. Continue until all poses are assigned
```

**RMSD calculation:**
- Default: over all heavy atoms (non-hydrogen)
- Custom: over user-specified atom indices (`cluster_atom_indices`)
  - Per-ligand `vector<int>`, 0-indexed
  - Useful for PROTAC / multi-component systems where only the
    pharmacophore atoms should be compared

### 8.2 Filtering

After clustering:
1. Keep the best pose from each cluster
2. Sort clusters by energy
3. Keep at most `max(rescore_pool, num_pose)` poses (the re-ranking pool)
4. Keep only poses within `energy_range` kcal/mol of the best

### 8.3 Optional Refinement

If `refine_steps > 0`, run BFGS on each surviving cluster
representative to polish the geometry.

---

## 9. Output Generation

The C++ engine always writes SDF output. Python concatenates per-batch
SDF files into the final output.

```
DockTask::dump_poses()
 └── write_poses_to_sdf()  →  <name>_<batch>.sdf

Python post-processing (run_unidock_protocol):
 └── concatenate batch SDFs → unidock2_pose.sdf
```

### 9.1 SDF Output

**Writer:** `sdf_writer.cpp` via `write_poses_to_sdf()`.

**Process:**
1. C++ engine writes one `.sdf` file per batch into `unidock2_output/`
2. Each SDF record = one pose with docked coordinates from the engine
   + bond/charge data from the original DMS (bond stereo column always 0)
3. Python detects `.sdf` files in the output directory
4. Python concatenates all batch SDF files (sorted by name) into the
   final `unidock2_pose.sdf`

```
C++ engine                           Python post-processing
───────────                          ──────────────────────
SDFMolData (from DMS)
  ├── anum[]           ─┐
  ├── formal_charge[]   │
  ├── bonds[] (order)   ├── write_poses_to_sdf()
  │                     │     → unidock2_output/<name>_0.sdf
  ├── properties[]      │     → unidock2_output/<name>_1.sdf
  └── mol_name         ─┘     → ...
                                      │
Docked coords from engine ────────────┘  cat *.sdf → unidock2_pose.sdf
```

**SDF data sources per field:**

| SDF Field | Source |
|-----------|--------|
| Atom coordinates | Docking engine pose results |
| Atom element symbols | `SDFMolData.anum[]` (from DMS `particle.anum`) |
| Bond orders | `SDFMolData.bonds[].order` (from DMS `bond.order`) |
| Bond stereo flags | Always 0: DMS `ud2_bond_stereo` wedges describe the input orientation, not the docked pose; readers take chirality from the 3D coordinates |
| Formal charges (M  CHG) | `SDFMolData.formal_charge[]` (from DMS `ud2_formal_charges`) |
| User properties | `SDFMolData.properties[]` (from DMS `ud2_sdf_properties`) |
| Energy properties | Computed by engine (Vina + FF scoring) |

Each SDF record contains these properties (as `> <tag>` lines):

| Property | Description |
|----------|-------------|
| `ud2_molecule_name` | Original molecule name from input |
| `ud2_pose_index` | Pose rank (0-based) |
| `vina_binding_free_energy` | Final Vina affinity (kcal/mol) |
| `vina_intra_inter` | Combined inter + intra |
| `vina_inter` | Ligand-receptor interaction |
| `vina_intra` | Ligand internal strain |
| `vina_box_penalty` | Box boundary penalty |
| `vina_torsion_number_energy` | Entropic torsion count term |
| `ff_total_energy` | Total force-field energy (if FF loaded) |
| `ff_interaction_energy` | FF intermolecular |
| `ff_intra_ligand_energy` | FF intramolecular |
| `atom_inter_energy` | Per-atom Vina decomposition array (if `decomp=true`) |
| `atom_gauss1` | Per-atom gauss1 term (if `decomp=true`) |
| `atom_gauss2` | Per-atom gauss2 term (if `decomp=true`) |
| `atom_repulsion` | Per-atom repulsion term (if `decomp=true`) |
| `atom_hydrophobic` | Per-atom hydrophobic term (if `decomp=true`) |
| `atom_hbond` | Per-atom hbond term (if `decomp=true`) |
| `atom_ff_total` | Per-atom FF decomposition (if FF + `decomp=true`) |
| `atom_ff_inter_lj` | Per-atom FF inter LJ (if FF + `decomp=true`) |
| `atom_ff_inter_coulomb` | Per-atom FF inter Coulomb (if FF + `decomp=true`) |
| `atom_ff_intra_lj` | Per-atom FF intra LJ (if FF + `decomp=true`) |
| `atom_ff_intra_coulomb` | Per-atom FF intra Coulomb (if FF + `decomp=true`) |
| `atom_ff_torsion` | Per-atom FF torsion (if FF + `decomp=true`) |

### 9.2 Molecule Naming

| Context | Name used |
|---------|-----------|
| Internal C++ processing | `MOL_0`, `MOL_1`, … (unique, from `ud2_molecules`) |
| SDF `_Name` property | Original input `_Name` from SDF (via `SDFMolData`) |
| SDF `ud2_molecule_name` tag | Original input `_Name` |

Multiple ligands with the same input name are supported; the C++ engine
uses unique internal IDs but always writes the original name to output.

---

## 10. Special Docking Modes

### 10.1 Constraint Docking

- **Activation:** `constraint_docking: true`
- **Effect:** `randomize` forced to `false`; MC mutations only
  change torsion angles (center and orientation are fixed)
- **Use case:** Template docking with core atom constraints

### 10.2 ud2align — Position / Alignment Bias

Separate tool (`ud2align/`) that injects bias potentials into the
ligand DMS before running `ud2`.

| Bias type | Parameter | Description |
|-----------|-----------|-------------|
| `BT_POS` | `bias=pos` | Spring potential pulling atoms to reference coordinates |
| `BT_ALIGN` | `bias=align` | Similarity-matrix-weighted alignment (ZAligner, HBondAligner) |

Workflow:
1. Prepare enriched DMS (normal path)
2. Compute similarity matrix between query and template
3. Inject bias data into ligand DMS via `bias_injector.py`
4. Run `ud2` with bias-enabled config

**Note:** `ENERGY_CEILING` only activates for ligands with ≥ 600 atoms,
so it does not interfere with ud2align's position bias (which can
have legitimately high initial energies that need optimizing down).

### 10.3 Torsion Library

- **Activation:** `tor_lib: true` (binary YAML) / `use_tor_lib=True` (pybind)
- **Source:** XML library (`tor_lib_2020.xml`) or ranges stored in DMS
- **Effect:** Restricts dihedral angles to preferred ranges with a
  soft harmonic penalty: `E = k × dist²`, where `k = 5.0` and `dist`
  is the shortest distance to any allowed range boundary
- **Applied in:** randomization (sample from ranges) and scoring
  (penalty gradient flows through BFGS)

### 10.4 Force-Field Minimization

- **Activation:** `ff_minimize_steps > 0` and DMS files with FF parameters
- **Runs after** Vina-based search and clustering as a post-processing
  refinement step
- **Hydrogen-only torsions** (-OH, -NH2, -NH3+, -SH rotors) are left out of the Vina search at DMS
  read (`ff_only_torsions`; a one-H fixed donor such as S-N(-)H stays) and are rotated here only

**Data source:** All FF parameters are read from the DMS file.
The C++ engine populates an `FFMinData` struct:

```
FFMinData {
    FFAtomParams lig_atoms;         // per-atom: sigma, epsilon, charge, anum, x/y/z
    FFAtomParams rec_atoms;         // per-atom: sigma, epsilon, charge, anum, x/y/z
    vector<FFPair14> lig_pairs_14;  // pre-scaled 1-4 LJ (aij, bij) + Coulomb (qij)
    vector<FFTorsion> lig_torsions; // dihedral params: phi0, fc0..fc6
    vector<FFExclusion> lig_exclusions;  // 1-2/1-3 bonded exclusions
}
```

**DMS tables used:**

| FF Component | DMS Table(s) | Key Columns |
|-------------|-------------|-------------|
| LJ sigma/epsilon | `particle` JOIN `nonbonded_param` | `charge`, `sigma`, `epsilon` |
| 1-4 pairs | `pair_12_6_es_term` + `pair_12_6_es_param` | `p0`, `p1`, `aij`, `bij`, `qij` |
| Torsions | `dihedral_trig_term` + `dihedral_trig_param` | `p0..p3`, `phi0`, `fc0..fc6` |
| Exclusions | `exclusion` | `p0`, `p1` |

**Energy terms** (computed by `cal_e_f_ff_cpu` in `score_ff_cpu.cpp`):

1. **Intra-molecular LJ:** Lorentz-Berthelot combining rules:
   `σ_ij = (σ_i + σ_j)/2`, `ε_ij = √(ε_i × ε_j)`,
   `E = 4ε_ij[(σ_ij/r)¹² − (σ_ij/r)⁶]`
2. **Intra-molecular Coulomb** (distance-dependent dielectric):
   `E = 332.0637 × q_i × q_j / (4 × r²)`
3. **1-4 pair interactions** (pre-scaled by SCNB=2.0 and SCEE=1.2):
   `E = a_ij/r¹² − b_ij/r⁶ + 332.0637 × q_ij/(4r²)`
4. **Torsion energy** (AMBER/GAFF2 convention):
   `E = fc₀ + Σ_{n=1..6} fc_n × (1 + cos(nφ − φ₀))`
5. **Inter-molecular LJ + Coulomb** (ligand-receptor, 12 Å cutoff)
6. **Box penalty** (same as Vina)

**Data-driven design:** The FF scoring engine reads all parameters
(sigma, epsilon, charges, torsion force constants, 1-4 pair coefficients)
directly from the DMS at runtime.  No GAFF2 atom type IDs are stored or
used by the C++ engine — only the raw numerical parameters matter.
See Section 10.6 for how to inject custom FF parameters.

### 10.5 Task Types

| Task | Entry point | Behavior |
|------|-------------|----------|
| `screen` | binary + pybind | Full docking: randomize → MC → BFGS → cluster → refine → output |
| `score` | binary + pybind | Single-point scoring: no randomization, no search, no clustering. exhaustiveness=1, num_pose=1 |
| `mc` | binary + pybind | MC random walk: randomize → MC (no BFGS, no refine). Clustering still runs. |
| `randomize` | binary only | Randomize only: generate `exhaustiveness` random poses, no MC, no BFGS, no refine, no clustering (rmsd_limit=0) |
| `optimize` | binary only | Refine input pose: no randomization, no MC, no BFGS. Uses `refine_steps` BFGS on the single input pose. exhaustiveness=1, num_pose=1 |

### 10.6 Custom Atom Type / FF Parameter Injection

Both the Vina scoring function and the FF minimization engine are fully
data-driven: the C++ engine reads all type definitions and parameters
from the input DMS at runtime.  This means users can define new atom
types or override FF parameters **without modifying any C++ code**.

#### Vina Custom Types (via DMS)

Standard Vina types use IDs 0–20.  IDs 21–31 are available for custom
types (`MAX_VN_TYPES = 32`).

**To add a custom Vina type:**

1. Pick an unused type ID (e.g., 21)
2. Set the custom type properties in the DMS `ud2_atom_props` table
   for the target atoms, and extend the `AtomTypeProps` arrays via
   the DMS metadata:
   - `vdw_radii[21] = <radius>`
   - `type_is_hydrogen[21] = false`
   - `type_is_hydrophobic[21] = true/false`
   - `type_is_donor[21] = true/false`
   - `type_is_acceptor[21] = true/false`
3. Set the `vina_type = 21` for the target atoms in `ud2_atom_props`

The VDW radius controls the surface distance `d` in all 5 scoring terms.
The boolean properties control which conditional terms (hydrophobic,
hbond) are activated.

No example script ships for this. The mechanism is exercised in
`unidock_engine/test/unit/cpu/test_custom_atom_type.cpp`: custom radii and boolean
props through the spatial grid, `eval_ef`, `cal_e_f_cpu` and the hbond correction.

#### GAFF2/FF Custom Parameters (via DMS)

FF minimization parameters are stored in DMS SQLite tables.  To inject
custom parameters, modify the DMS file directly using `sqlite3`:

| What to change | DMS table | Column(s) |
|---------------|-----------|-----------|
| LJ sigma/epsilon | `nonbonded_param` | `sigma`, `epsilon` |
| Partial charges | `particle` | `charge` |
| 1-4 pair coefficients | `pair_12_6_es_param` | `aij`, `bij`, `qij` |
| Torsion force constants | `dihedral_trig_param` | `fc0`..`fc6` |

**Three integration approaches:**

1. **Direct DMS edit** (simplest): Build DMS via standard pipeline, then
   modify specific values with `sqlite3` before running the engine
2. **Custom frcmod**: Edit the GAFF2 frcmod file before running `tleap`,
   so the resulting prmtop contains the custom parameters
3. **Programmatic msys**: Build the DMS programmatically using `msys` +
   `parmed`, setting all parameter values directly

No example script ships for this. Approach 1 is exercised by the `[dms]` cases of
`unidock_engine/test/unit/cpu/test_custom_ff_param.cpp`, which perturb sigma, epsilon,
charge and the torsion force constants on a real DMS and check that the energy moves.

#### Test Coverage

| Layer | Test file | What it verifies |
|-------|-----------|-----------------|
| C++ Vina types | `test/unit/cpu/test_custom_atom_type.cpp` | Custom VDW radii, boolean props, spatial grid, eval_ef, cal_e_f_cpu, hbond angle correction |
| C++ FF params | `test/unit/cpu/test_custom_ff_param.cpp` | Custom sigma/epsilon/charge/torsion/1-4 pairs change FF scoring energy |
| Python ↔ C++ | `test/scripts/test_atom_type_props.py` | Python VINA_ATOM_TYPE_PROPERTIES matches C++ defaults |

---

## 11. Build System

### 11.1 Build Targets

| Target | CMake option | Output | Description |
|--------|-------------|--------|-------------|
| `ud2` | `BUILD_BIN=ON` | Binary executable | CLI docking, reads YAML |
| `pipeline` | `BUILD_API=ON` | `pipeline.cpython-*.so` | Pybind11 Python module |
| `unit_test` | `BUILD_TEST=ON` | Test binary | C++ unit tests |

### 11.2 Backend Selection

| Option | Default | Description |
|--------|---------|-------------|
| `USE_CUDA` | ON | Enable CUDA GPU backend |
| `USE_OPENMP` | ON | Enable OpenMP CPU parallelization |
| `USE_DOUBLE` | OFF | Float (OFF) vs double (ON) precision |

### 11.3 Build Directories

```
unidock/unidock_engine/
  ├── build_cpu/    → ud2 binary + unit_test + pipeline.so
  │                   (BUILD_BIN=ON, BUILD_TEST=ON, BUILD_API=ON)
  └── rebuild.sh    → rebuilds build_cpu in one shot
```

Configure once with all targets enabled:
```bash
cmake -B build_cpu -S . -DUSE_CUDA=OFF -DUSE_OPENMP=ON \
  -DBUILD_TEST=ON -DBUILD_API=ON -DBUILD_BIN=ON \
  -DCMAKE_BUILD_TYPE=Release
```

---

## 12. Key Files Reference

### Python Layer

| File | Purpose |
|------|---------|
| `unidock_processing/main.py` | CLI entry point |
| `unidock_processing/cli/docking.py` | Docking CLI command |
| `unidock_processing/io/yaml.py` | YAML config parser (Pydantic) |
| `unidock_processing/unidocktools/unidock_protocol_runner.py` | Orchestrator |
| `unidock_protocol_runner.prepare_dms_lightweight()` | Lightweight DMS prep (no GAFF2/docking); used by ud2align |
| `unidock_processing/unidocktools/unidock_receptor_topology_builder.py` | Receptor prep |
| `unidock_processing/unidocktools/unidock_ligand_topology_builder.py` | Ligand prep |
| `unidock_processing/unidocktools/unidock_ligand_pose_writer.py` | Mol-list-based SDF writing |

### C++ Engine

| File | Purpose |
|------|---------|
| `unidock_engine/bin/unidock/main.cpp` | Binary entry, YAML parsing |
| `unidock_engine/api/python/pipeline.cpp` | Pybind11 entry |
| `unidock_engine/src/screening/screening.cpp` | Batch orchestration |
| `unidock_engine/src/task/DockTask.cpp` | Per-batch pipeline |
| `unidock_engine/src/cpu/mc_cpu.cpp` | Monte Carlo search |
| `unidock_engine/src/cpu/bfgs_cpu.cpp` | BFGS optimization |
| `unidock_engine/src/cpu/score_cpu.cpp` | Vina scoring + forces |
| `unidock_engine/src/cpu/score_ff_cpu.cpp` | FF scoring + forces (GAFF2/ff14SB) |
| `unidock_engine/src/cpu/cluster_cpu.cpp` | RMSD clustering |
| `unidock_engine/src/cpu/spatial_grid.h` | Receptor neighbor grid |
| `unidock_engine/src/score/vina.h` | Vina term evaluation (eval_ef) |
| `unidock_engine/src/score/gaff2.h` | FF energy functions (ff_lj, ff_coulomb, ff_torsion, ff_pair14) |
| `unidock_engine/src/constants/constants.h` | Atom type defaults, VDW radii, MAX_VN_TYPES |
| `unidock_engine/src/format/dms_reader.cpp` | DMS reader (topology + FF params) |
| `unidock_engine/src/format/dms_reader.h` | FFMinData, FFAtomParams, FFTorsion structs |
| `unidock_engine/src/format/sdf_writer.cpp` | SDF writer |
| `unidock_engine/src/model/model.h` | Core data structures (UDFlexMol, UDFixMol) |

### Key Tests

| File | Purpose |
|------|---------|
| `unidock_engine/test/unit/cpu/test_custom_atom_type.cpp` | C++ tests: custom Vina types in scoring pipeline |
| `unidock_engine/test/unit/cpu/test_custom_ff_param.cpp` | C++ tests: custom FF params in FF scoring pipeline |
| `unidock_engine/test/unit/cpu/test_score_ff_cpu.cpp` | C++ tests: FF scoring on real DMS data |
| `unidock_processing/test/scripts/test_atom_type_props.py` | Python↔C++ atom type consistency |
