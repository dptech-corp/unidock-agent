# Uni-Dock2 features: the engine, the core package and the plugins

This page lists what each layer of Uni-Dock2 supports and which features need something installed besides the
`unidock2` conda package. It describes engine version 0.4.6 and the plugin layout of the reorganised tree.

Uni-Dock2 has three layers:

| layer | what it is | how you reach it |
|---|---|---|
| engine | the C++/CUDA docking engine, `unidock/unidock_engine` | the `ud2` binary with a YAML file, or the pybind11 module `unidock_engine.api.python.pipeline` |
| core package | `unidock_processing`: receptor and ligand preparation, the runner, the CNN stage, the CLI | `unidock2 docking`, `unidock2 protein_prep`, `UnidockProtocolRunner` |
| plugins | features that need a second molecule or a chemical annotation: a template pose, a core atom map, a covalent residue | a YAML block named after the plugin, `--<plugin>_*` CLI options, or `UnidockProtocolRunner(plugins=[...])` |

## 1. What needs installing

| feature | where it lives | install |
|---|---|---|
| free docking, re-docking, screening, score-only | engine + core | `unidock2` |
| CNN rescoring (`--cnn_rescore`) | core | included: `pytorch` in every build, with the torch gridder (same scores, slower). libmolgrid's CUDA gridder is optional (`scripts/build_molgrid.sh`, README step 4); the conda package pulls conda-forge's `molgrid` and `openbabel 3.1.*` in the CUDA builds. |
| energy decomposition (`--decomp`) | engine | included |
| torsion-library sampling ranges | engine + core | included |
| receptor preparation from a PDB | core | included fallback: MDAnalysis + PDBFixer + AmberTools `tleap`. `fepfixer` is used instead when it is on PATH; it is not a declared dependency. |
| FF minimisation (`--ff_minimize_steps` > 0) | engine + core | needs `utop` (unitop) or `tleap` on PATH when the receptor goes through fepfixer; not a declared dependency |
| template (constraint) docking | **template plugin**, built into the core | included |
| covalent docking | **covalent plugin**, built into the core | included |
| position bias and aromatic-ring bias (alignment to a template pose) | **external bias plugin = the `ud2align` package** | install `ud2align` separately inside the unidock2 environment |
| MCS-tree workflow (`tree_align`, `scheduled_docking`) | external `atom_mapper` package; it drives the template plugin | install atom_mapper separately; it also needs `numba` and `rustworkx`, which no Uni-Dock2 environment provides |

Only the bias feature needs a plugin package. Template and covalent docking are plugins too, but they ship in the core
wheel and are always registered.

The engine packages ship `ud2` inside site-packages (`unidock_engine/api/python/bin/ud2`), not on PATH. Put a `ud2`
on PATH (the build's `bin/` directory) before running `unidock2 docking`.

## 2. The engine

### Backends and entry points

- **CUDA** (default build) and **CPU** (`-DUSE_CUDA=OFF`, OpenMP). A given pose scores the same on both; the
  searches differ (random streams, receptor maps on CUDA), so the written poses can differ. The CPU engine gives the same output on a re-run only at `ncpu: 1`.
- **`ud2 <config.yaml>`**: the binary reads one YAML file with the blocks below. `ud2 --version` prints the engine
  version.
- **pybind11 pipeline** (`run_docking_pipeline`): the same parameters as keyword arguments. Its `search_mode` presets
  differ from the binary's, and it honours the caller's `randomize` under constraint docking. Since 0.4.6 the module
  reports `__version__`.

### Tasks (`Settings.task`)

| task | what it runs |
|---|---|
| `screen` (default) | the full search: Monte Carlo + local optimisation, clustering, the pose pass, output |
| `score` | scores the input pose only (exhaustiveness 1, no search, 1 pose) |
| `optimize` | local optimisation of the input pose (`refine_steps` only) |
| `mc` | Monte Carlo without local optimisation, with clustering |
| `randomize` | random starting poses only, no clustering |

`benchmark_one` is accepted but not implemented.

### Search (`Settings.search_mode` and the `Advanced` block)

| `search_mode` | CUDA exhaustiveness / mc_steps | CPU exhaustiveness / mc_steps |
|---|---|---|
| `fast` | 128 / 20 | 64 / 20 |
| `balance` (default) | 256 / 30 | 128 / 30 |
| `detail` | 512 / 40 | 256 / 40 |
| `free` | the YAML values | the YAML values |

Every mode except `free` overwrites `exhaustiveness` and `mc_steps` from the YAML.

| key (`Advanced`) | meaning |
|---|---|
| `seed`, `exhaustiveness`, `randomize`, `mc_steps` | Monte Carlo search |
| `opt_steps` | local-optimisation steps per Monte Carlo step; -1 = the per-ligand heuristic (default) |
| `refine_steps` | local refinement after the search (default 5) |
| `rmsd_limit`, `energy_range`, `num_pose` | clustering and the output window |
| `rescore_pool` | how many clustered poses the pose pass re-ranks before writing `num_pose` (default 10) |
| `tor_lib` | restrict torsion sampling to the torsion-library ranges (`ud2_torsion_ranges`) |
| `use_grid`, `grid_spacing`, `grid_margin` | receptor grid maps for the CUDA search (on by default, 0.375 A, 8 A); the final scores are analytic |
| `ff_minimize_steps`, `ff_minimize_mode`, `ligand_dms`, `protein_dms` | force-field minimisation after the search (0 = off; mode 0 internal coordinates, 1 Cartesian) |
| `cluster_atom_indices` | per-ligand atom subsets for the clustering RMSD |
| `rng_period` | ensemble runs: conformers of one ligand share random subsequences |

`Hardware`: `ncpu`, `gpu_device_id`. `Inputs`: `ligand_dms_full`, `receptor_dms_full`. `Outputs`: `dir`.

### Box (`Settings`)

- `center_x/y/z`, `size_x/y/z`: the docking box.
- `inner_box`: edge length of an inner box around the centre that the ligand's heavy-atom centroid must stay in (0 =
  off, the default; the benchmark protocol uses 10 A).
- `constraint_docking`: no translation or rotation of the ligand root (set by the template and covalent plugins);
  the binary then also turns `randomize` off.

### Scoring

A Vina-form function with these terms: `gauss1`, `gauss2`, `repulsion`, `hydrophobic`, `hbond`, `intra`, `elec`
(with `elec_charged_scale`), `polar_env`, `arom_contact` and `torsion_strain`.

- **Three weight sets**, each with compiled defaults and each overridable key by key in `Settings`:
  - `score_weights`: the search (fitted set)
  - `rescore_weights`: the pose pass, which orders the output poses (`vina_pose_score`)
  - `affinity_weights`: the reported affinity (`vina_affinity_score`, plain Vina weights)

  All three sets are logged at start.
- **Angular hydrogen bonds** (`Settings.angular_hbond`, on by default): the hbond term is weighted by the D-H...A
  angle.
- **Torsion strain** from the `ud2_torsion_penalties` rows that ligand preparation writes (amides, esters, anilides,
  aryl carbonyls, biaryls, sulfonamides and others).
- **Energy decomposition** (`Settings.decomp`): per-atom and per-residue terms that reconcile exactly with
  `vina_pose_score`.

### Bias terms (used only through the bias plugin)

- **Block:** `bias: {type: no | pos | align, k: <weight>, ring: {k, s, d0}}`. `Advanced.bias` and `Advanced.bias_k`
  are still read as aliases, with a deprecation warning.
- **Engine requirements:** engine >= 0.4.5 for the block, >= 0.4.6 for ring rows. An older engine ignores the bias
  silently, so the Python side refuses to run a biased job on it.
- **Atom position bias:** Gaussian (`pos`) or alignment (`align`) wells at template positions, read from the
  `ud2_bias` table.
- **Aromatic-ring bias:** parallel stacking of query rings on template rings, from the `ud2_ring_bias` and
  `ud2_ring_bias_atoms` tables. It applies under `align` only. Defaults are k 30, s 3, d0 1.
- **Search and ranking:** the bias acts in the search. In a biased run the pose pass orders the poses by
  `vina_pose_score` + the bias energy. `vina_pose_score` and every affinity value stay unbiased, and the bias energy
  is written as `ud2_bias_energy`.
- **Hydrogens:** after a search the engine places polar hydrogens (with angular hbonds on); biased and constrained
  runs skip that and keep the input hydrogens.

### Inputs and outputs

- **Inputs:** DMS files only.
  - Core tables: `particle`, `bond`, `nonbonded_param`, `ud2_atom_props`, `ud2_molecules`, `ud2_atom_aromatic`,
    `ud2_torsion_ranges`, `ud2_torsion_penalties`, `ud2_formal_charges`, `ud2_bond_stereo`, `ud2_sdf_properties`,
    `ud2_metadata`, and the FF tables.
  - Extension tables come through a table-reader registry: `ud2_bias`, `ud2_ring_bias`. An extension table that is
    missing switches its feature off.
- **Outputs:** one SDF per batch. The main tags are:
  - scores: `vina_pose_score`, `vina_affinity_score`, `vina_binding_free_energy`, `vina_inter`, `vina_intra`
  - diagnostics: `ud2_pose_index`, `ud2_torsion_strain_raw`, `ud2_polar_env_*`, `ud2_arom_contact_raw`
  - with `decomp`: the `residue_*` decomposition tags
  - with FF parameters: the `ff_*` energies
  - biased runs: `ud2_bias_energy` and `ud2_ring_bias_raw`

## 3. The core package (`unidock2`)

### Commands

- `unidock2 docking` runs preparation, the engine and the optional CNN stage. It takes options or a YAML file
  (`-cf`).
- `unidock2 protein_prep` prepares a receptor DMS once, for reuse.

### What the runner does

1. **Receptor preparation.**
   - A DMS that already carries `ud2_atom_props` is used as is.
   - A PDB goes through fepfixer (and utop, or tleap, when FF parameters are needed), or through the PDBFixer + tleap fallback.
     The receptor is then typed for Vina from connectivity.
   - fepfixer's hydrogen placement is not deterministic. To compare runs, prepare the receptor once and reuse the
     DMS.
2. **Ligand preparation.**
   - Each SDF becomes a DMS, with atom typing, the torsion tree and the torsion-library ranges.
   - Scoring annotations are added: torsion penalties and aromatic atoms.
   - GAFF2 parameters are added only when FF minimisation is on.
3. **The engine run** (the `ud2` binary on PATH, or the pybind11 module).
4. **The CNN rescoring stage** (optional).
   - It rescores the pool with the gnina pose-CNN ensemble and blends the score with the engine score.
   - It keeps `num_pose` poses per ligand; for the CNN the engine writes `rescore_pool` poses.
   - It is skipped for constraint, template, covalent and biased runs, which keep the engine order.
   - It runs in the current interpreter; `UD2_CNN_PYTHON` selects another.

### `unidock2 docking` options

| option | YAML | meaning |
|---|---|---|
| `-r`, `-l`, `-lb` | `Required: receptor, ligand, ligand_batch` | receptor (PDB or DMS), one ligand (SDF or DMS), a text file of ligand SDFs |
| `-c x y z` | `Required: center` | box centre; without `-c` the YAML centre is used |
| `-o` | `Preprocessing: output_docking_pose_sdf_file_name` | output SDF |
| `-cf` | | the YAML file |
| `--decomp` | | energy decomposition |
| `--cnn_rescore` | `Settings: cnn_rescore` | CNN rescoring stage |
| `--ff_minimize_steps N` | | FF minimisation steps (default 0 = off) |

YAML-only settings:
- `Advanced`: exhaustiveness, randomize, mc_steps, opt_steps, refine_steps, num_pose, rescore_pool, rmsd_limit,
  energy_range, seed, use_tor_lib
- `Hardware`: n_cpu, gpu_device_id
- `Settings`: box_size, task, search_mode, cnn_rescore
- `Preprocessing`: preserve_receptor_hydrogen, temp_dir_name, output file names

The CLI and the runner do not expose `inner_box`, `angular_hbond` or the weight blocks. Those exist only in the
engine YAML: runner runs use the compiled weights, angular hbonds on, and no inner box.

## 4. The plugins

Plugins are found through the `unidock2.plugins` entry-point group, a built-in list (template, covalent) and a list of
known external plugins (bias -> `ud2align.plugin:BiasPlugin`).

- **A missing plugin:** if a configuration uses a plugin that is not installed (a `bias:` block or a `--bias_*`
  option without ud2align), the run stops with an error that names the package.
- **Plugin hooks:** each plugin can add a YAML block and CLI options. It can edit the receptor builder before
  preparation and add per-ligand hooks to the ligand builder. It can write extra DMS tables, set engine settings
  (`constraint_docking`) and add an engine YAML block. It can also post-process the output.

### Template plugin (built in): constraint docking on a mapped core

- **What:** the ligand's core, mapped to a reference pose, is placed on the reference coordinates and held there.
  - The ligand root stays where it is placed.
  - The core's rotatable bonds are frozen, and the engine runs with `constraint_docking` on.
  - The core map is either computed (RDKit MCS against the reference) or given per ligand.
  - The box centre defaults to the reference centre when the run gives (0, 0, 0).
- **Enable:**
  - YAML: `template: {template_docking: true, reference_sdf_file_name: ref.sdf, core_atom_mapping_dict_list: [...]}`
  - CLI: `--template_docking --reference_sdf_file_name ref.sdf [--core_atom_mapping_dict_list JSON]`
  - API: `UnidockProtocolRunner(template_docking=True, reference_sdf_file_name=..., core_atom_mapping_dict_list=...)`
  - The old `Preprocessing:` keys still work, with a deprecation warning.
- **atom_mapper** builds an MCS tree over a series and docks it parent by parent through this plugin, with
  `atom_mapper_align=True` for its own rotatable-bond rule. atom_mapper itself is a separate package (section 1).
- **No CNN stage:** template runs keep the engine order.

### Covalent plugin (built in): covalent docking

- **What:** the ligand SDF carries the receptor residue atoms it is bonded to, with the properties
  `covalent_atom_indices`, `covalent_atom_names`, `covalent_residue_names`, `covalent_residue_indices` and
  `covalent_chain_indices`.
  - The plugin removes those dummy atoms and roots the ligand at the anchor.
  - It hands the anchor, bond-start and bond-end receptor atoms to the receptor builder. The receptor-side
    residue edit is reached only by the builder's full receptor analysis, which the runner does not call, so in a
    runner run the receptor is not modified.
  - The engine runs with `constraint_docking` on.
- **Enable:**
  - YAML: `covalent: {covalent_ligand: true, covalent_residue_atom_info_list: [[chain, resname, resid, atom], x3]}`
  - CLI: `--covalent_ligand --covalent_residue_atom_info_list JSON`
  - API: `UnidockProtocolRunner(covalent_ligand=True, covalent_residue_atom_info_list=...)`
  - The old `Preprocessing:` keys still work, with a deprecation warning.
- **No CNN stage:** covalent runs keep the engine order.

### Bias plugin (the `ud2align` package): alignment to a template pose

`ud2align` is external and is not included in this repository. Install it separately inside the unidock2 environment.

- **What:** a soft bias that pulls query atoms toward the positions of a template ligand. Unlike template docking,
  the search stays free.
  - **Aligners** pair query and template atoms:
    - `zalign` (default): element, charge, and bond and ring environment
    - `hbond`: hydrogen-bond donors and acceptors
    - `hbond_charge`: H-bond + formal charge
  - **Ring bias** (`ring_bias`, default off): adds the aromatic-ring term, pairing every query aromatic ring with
    every similar template ring.
  - By default each ligand is first moved to the template centroid (`translate`).
  - Each ligand's pose order uses the bias-aware pose pass (section 2), and the CNN stage is skipped.
- **Enable, inside a docking run:**
  - YAML: `bias: {template_sdf: tmpl.sdf, aligner: zalign, bias_k: 0.05, bias_type: align, ref_atom_indices: [...], ring_bias: false, ref_ring_indices: [...], translate: true}`
  - CLI: `--bias_template_sdf`, `--bias_aligner`, `--bias_k`, `--bias_type`, `--bias_ref_atom_indices`,
    `--bias_ring`, `--bias_ref_ring_indices`, `--no_bias_translate`
  - API: `UnidockProtocolRunner(..., plugins=[BiasPlugin(template_sdf=..., aligner_name=..., ...)])`
- **Enable, standalone (one template; one or more queries, all aligned in one engine run):**
  - `ud2align-cli -q query.sdf -t template.sdf [-r receptor] -o out.sdf`, with the options `--aligner`, `--topn`,
    `--exhaustiveness`, `--bias-weight`, `--tor-lib`, `--ring-bias` and `--score-only`
  - Python: `perform_mol_align` (no receptor), `perform_dock_align` (with a receptor) and their `_score` variants
- **Engine:** >= 0.4.5 for the bias block, >= 0.4.6 for ring rows.

## 5. Compatibility kept for one release

These older interfaces still work during the transition and print a deprecation warning where noted:

- the runner keyword arguments `template_docking`, `reference_sdf_file_name`, `core_atom_mapping_dict_list`,
  `atom_mapper_align`, `covalent_ligand` and `covalent_residue_atom_info_list`;
- the same keys under `Preprocessing:` in the YAML (warning);
- the engine keys `Advanced.bias` and `Advanced.bias_k` (warning);
- the module `unidock_processing.unidocktools.unidock_protocol_runner`.
