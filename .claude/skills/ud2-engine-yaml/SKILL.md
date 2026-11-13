---
name: ud2-engine-yaml
description: Run the Uni-Dock2 engine (`ud2`) directly from a YAML file on prepared DMS inputs, for everything the unidock2 CLI does not expose — task modes (score, optimize, mc, randomize), the ligand-centroid inner box, scoring-weight overrides, per-atom energy decomposition, receptor grid settings, hand-written position-bias or ring-bias rows, clustering atoms, and reading the engine log. Use whenever the user asks for an engine option by its YAML key, wants to rescore or minimise given poses, override weights, add an inner box, run ud2 directly, or debug what the engine actually resolved.
---

# The `ud2` engine YAML

This is the engine's own YAML; the unidock2 CLI's configuration YAML is ud2-config-yaml.

`ud2 config.yaml` docks prepared DMS files. Every key is listed in `docs/features_and_plugins.md` (section 2); the
parser is `unidock/unidock_engine/src/config/parse.cpp`. The fastest way to get valid inputs is to run
`unidock2 docking` once and take `_ud2_engine_config.yaml` and the DMS files from its working directory (set
`Preprocessing: temp_dir_name` to a directory other than `/tmp`, which is deleted after the run).

```yaml
Advanced:
  seed: 9996
  exhaustiveness: 512
  mc_steps: 40
  opt_steps: -1              # -1 = per-ligand heuristic
  refine_steps: 5
  num_pose: 10
  rescore_pool: 10
  rmsd_limit: 1.0
  energy_range: 10.0
  tor_lib: false             # the engine default is true; unidock2 writes false unless use_tor_lib is set
  # use_grid: true           # grid_spacing: 0.375, grid_margin: 8.0
  # ff_minimize_steps: 0     # ff_minimize_mode: 0 internal | 1 Cartesian; needs FF-parameterised DMS inputs
  # rng_period: 0            # CUDA only
  # cluster_atom_indices: [[0, 1, 2]]   # one list per ligand
Hardware: {ncpu: 32, gpu_device_id: 0}
Settings:
  task: screen
  search_mode: free          # anything else overwrites exhaustiveness and mc_steps
  center_x: 0.0
  center_y: 0.0
  center_z: 0.0
  size_x: 25
  size_y: 25
  size_z: 25
  inner_box: 10              # optional: heavy-atom centroid kept within 5 A of the centre
  decomp: false
Outputs: {dir: out}
Inputs: {ligand_dms_full: ligand.dms, receptor_dms_full: receptor.dms}
```

## Tasks (`Settings.task`)

| task | use |
|---|---|
| `screen` | normal search |
| `score` | score the input pose as given (one pose) |
| `optimize` | local optimisation of the input pose only |
| `mc` | Monte Carlo without local optimisation |
| `randomize` | random starting poses, no clustering |

## Options worth knowing

- **Inner box** (`Settings.inner_box`, edge in Å): the benchmark protocol uses 10 with a 25–30 Å box. It is not
  reachable from the CLI or the runner.
- **Weights:** `Settings.score_weights` (search), `rescore_weights` (pose order, `vina_pose_score`) and
  `affinity_weights` (`vina_affinity_score`). Each overrides only the keys it names. Keys: gauss1, gauss2,
  repulsion, hydrophobic, hbond, intra, elec, elec_charged_scale, polar_env, arom_contact, torsion_strain. All three
  resolved sets are printed at the start of the log; check them there.
- **Decomposition** (`Settings.decomp: true`): per-atom and per-residue terms that add up to `vina_pose_score`.
- **Receptor grid** (`Advanced.use_grid`, on by default on CUDA; `grid_spacing` 0.375 Å, `grid_margin` 8 Å beyond the
  box on every side): the search runs on maps and the final scores are analytic. Turning it off is slower and
  changes poses.
- **Clustering atoms** (`Advanced.cluster_atom_indices`): per-ligand atom lists for the clustering RMSD.
- **Torsion library** (`Advanced.tor_lib`): the engine's own default is `true`. `unidock2` writes `false` unless the
  CLI YAML sets `use_tor_lib: true`, so a hand-written engine YAML without the key samples differently from a CLI run.
- **FF minimisation** (`Advanced.ff_minimize_steps: N`, `ff_minimize_mode: 0` internal coordinates or `1`
  Cartesian): runs after the search when both DMS inputs carry force-field parameters. Otherwise the engine logs
  "FF minimization requires both ligand_dms and protein_dms paths" and turns it off. The ligand DMS gets GAFF2
  parameters only from a CLI run with `--ff_minimize_steps` > 0. `Advanced.ligand_dms` / `protein_dms` name other
  DMS files to take the parameters from. Serial on CUDA, so slow.
- **Shared random streams** (`Advanced.rng_period: P`, CUDA only; the CPU backend rejects it): input ligand k draws
  the Monte Carlo streams of ligand k mod P. One launch of many conformers then reproduces runs made P conformers
  at a time.
- **Exact repeats on CPU** (`Hardware.ncpu: 1`): the CPU engine repeats its output only with one thread.

## Bias rows by hand

The engine applies a bias when the ligand DMS has rows and the YAML has `bias: {type: align, k: 0.05}`
(`Advanced.bias` / `bias_k` are read as deprecated aliases). ud2align writes these tables; to write them yourself:

```sql
CREATE TABLE ud2_bias (atom_id INTEGER, ref_x REAL, ref_y REAL, ref_z REAL, v_set REAL, r REAL);
-- one row per (ligand atom, target position); v_set = weight; rows of one atom contiguous
CREATE TABLE ud2_ring_bias (ring_row INTEGER, cx REAL, cy REAL, cz REAL, nx REAL, ny REAL, nz REAL, v_set REAL);
CREATE TABLE ud2_ring_bias_atoms (ring_row INTEGER, pos INTEGER, atom_id INTEGER);   -- ring atoms in ring order
```

`type: pos` is a Gaussian well, `type: align` the Z-align well (ring rows apply under `align` only). Optional ring
parameters: `bias: {ring: {k: 30, s: 3, d0: 1}}`. Engine 0.4.5 reads the block, 0.4.6 the ring rows; older binaries
ignore both and dock unbiased.

## Read the log

The engine log (`ud.log` next to where it ran) prints the resolved `DockParam` (seed, exhaustiveness, steps,
num_pose, rescore_pool, bias type and k), the three weight sets, the inner box and the bias block. Check it before
trusting a run, especially after a `search_mode` preset or an alias warning.
