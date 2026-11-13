---
name: ud2-config-yaml
description: Write, edit or check the configuration YAML of the unidock2 CLI (`unidock2 docking -cf config.yaml`, unidock_configurations.yaml) — the Required / Advanced / Hardware / Settings / Preprocessing blocks, the template / covalent / bias plugin blocks, recipes for re-docking, screening, scoring, CNN rescoring, template, covalent and biased runs, and a checker that prints the values the run will actually use. Use whenever the user asks to set up, write, change or debug a Uni-Dock2 config or YAML, asks why a setting had no effect, or before starting a unidock2 docking run with a new or edited YAML. For the `ud2` engine YAML (inner box, weights, decomposition, grid, hand-written bias rows) use ud2-engine-yaml.
---

# The unidock2 configuration YAML

Two different YAML files exist:

| file | read by | for |
|---|---|---|
| configuration YAML (this skill) | `unidock2 docking -cf`, `UnidockProtocolRunner` | everyday runs from PDB / SDF / DMS inputs |
| engine YAML (ud2-engine-yaml) | `ud2 config.yaml` | prepared DMS inputs and the options the CLI does not expose |

## Workflow

1. Copy `templates/unidock_configurations.yaml` (every key, CLI defaults, one-line meaning each).
2. Change only what the job needs; the recipes below list those keys.
3. Check it from the directory the run will start in, with the same command-line options:
   ```bash
   python <skill>/scripts/check_config.py config.yaml [-r receptor.dms] [-l ligands.sdf] [-c X Y Z] [--cnn_rescore]
   ```
   It reads the file with the CLI's own parser and prints the resolved receptor, ligand count, centre and its source,
   box, search size, poses, plugins, CNN stage, work directory, output and engine. Exit status 1 on errors. Fix every
   ERROR; read every WARN.
4. Run `unidock2 docking ... -cf config.yaml`. The CLI first prints the YAML values as read; the engine's
   `exhaustiveness=... mc_steps=...` line further down the log is what ran.

## Rules the parser does not enforce

The CLI reads the YAML with pydantic models that accept unknown keys, so these mistakes run without a word:

- **Unknown or misspelt keys and blocks are ignored.** `exhaustivness: 1024` and `settings:` (lower case) change
  nothing.
- **Engine-YAML keys are ignored here.** `inner_box`, `angular_hbond`, `score_weights`, `rescore_weights`,
  `affinity_weights`, `use_grid`, `ncpu`, `center_x`, `Inputs:`, `Outputs:` and the engine bias form
  `bias: {type, k}` belong to the engine YAML. A runner run uses the compiled fitted weights, angular hydrogen bonds
  and no inner box.
- **Command-line only:** `--decomp` and `--ff_minimize_steps N`. The FF minimisation default is 0. Above 0 it needs
  force-field parameters on the receptor, and it runs serially on CUDA, so it is slow:
  - the PDBFixer + tleap route, and a DMS made by it, have them;
  - the fepfixer route needs utop or tleap on PATH;
  - with any other DMS the engine logs "FF scoring disabled" and the poses get no `ff_*` tags.

  `--cnn_rescore` and `Settings: cnn_rescore: true` are equivalent. The checker takes all four flags.
- **`search_mode` overwrites the search size.** `fast` / `balance` / `detail` set exhaustiveness and mc_steps
  (CUDA 128/20, 256/30, 512/40; CPU half the exhaustiveness). `balance` is the default, so the Advanced values of
  512/40 apply only with `search_mode: free`.
- **Command-line options win over the YAML:** `-r`, `-l`, `-lb`, `-c`, `-o`. A centre of (0, 0, 0) counts as not
  given; template runs then centre on the reference ligand, every other run docks at the origin.
- **Relative paths resolve against the current directory**, not the YAML's directory. That includes the paths
  inside a `ligand_batch` file and `template.reference_sdf_file_name`.
- **`temp_dir_name` must exist.** Work directories under `/tmp` are deleted after the run; anywhere else they are
  kept, which is where `_ud2_engine_config.yaml` and the enriched DMS files are found.
- **`Hardware.n_cpu` sets the ligand-preparation workers only.** The CPU engine always uses every core.
- **Plugin keys under `Preprocessing:`** (template_docking, reference_sdf_file_name, core_atom_mapping_dict_list,
  covalent_ligand, covalent_residue_atom_info_list) still work, with a deprecation warning. New files put them in
  the `template:` / `covalent:` blocks.
- **A plugin block needs its plugin.** A `bias:` block that sets anything is an error when ud2align is not installed.
- **The CNN stage needs CUDA torch** in the scorer's interpreter (`UD2_CNN_PYTHON`, default the current one), on
  device `gpu_device_id`. It is skipped for template, covalent and biased runs, which keep the engine order.

## Recipes

Only the keys that differ from the template are shown.

Re-docking a crystal ligand, reproducible, intermediate files kept:
```yaml
Required: {receptor: receptor.dms, ligand: ligand.sdf, center: [X, Y, Z]}
Advanced: {seed: 1234567}
Settings: {search_mode: free}            # exhaustiveness 512, mc_steps 40 from Advanced
Preprocessing: {temp_dir_name: ./work}   # create ./work first
```

Screening a library: rank ligands on `vina_affinity_score`, not on the pose score.
```yaml
Required: {receptor: receptor.dms, ligand: null, ligand_batch: ligands.txt, center: [X, Y, Z]}
Advanced: {num_pose: 1}                  # the pose pass still re-ranks rescore_pool (10) poses
```

Scoring given poses (no search):
```yaml
Settings: {task: score}                  # optimize = local optimisation of the input pose
```

CNN pose re-ranking (pose prediction; it does not improve the ranking of ligands in a screen):
```yaml
Settings: {cnn_rescore: true}            # the engine writes rescore_pool poses, the stage keeps num_pose
```

Template, covalent and biased runs: see ud2-template-docking, ud2-covalent-docking and ud2-bias-align for the inputs.
```yaml
template: {template_docking: true, reference_sdf_file_name: reference.sdf, core_atom_mapping_dict_list: null}
covalent: {covalent_ligand: true, covalent_residue_atom_info_list: [["", "CYX", 25, "CA"], ["", "CYX", 25, "CB"], ["", "CYX", 25, "SG"]]}
bias: {template_sdf: template.sdf, aligner: hbond, bias_k: 0.05}
```

## Every feature: where it is set

"CLI YAML" keys go in this file; "flag" means a `unidock2 docking` option only; "engine YAML" keys exist only in the
`ud2` YAML (ud2-engine-yaml) and the checker reports them as ignored here. The features are those of
`docs/features_and_plugins.md`.

| feature | set in | snippet |
|---|---|---|
| task mode | CLI YAML | `Settings: {task: score}` (screen, score, optimize, mc, randomize) |
| search size | CLI YAML | `Settings: {search_mode: free}` with `Advanced: {exhaustiveness: 512, mc_steps: 40}` |
| random start | CLI YAML | `Advanced: {randomize: true}` (template and covalent runs force it off) |
| local optimisation, refinement | CLI YAML | `Advanced: {opt_steps: -1, refine_steps: 5}` |
| written poses, clustering | CLI YAML | `Advanced: {num_pose: 10, rmsd_limit: 1.0, energy_range: 5.0}` |
| pose-pass pool | CLI YAML | `Advanced: {rescore_pool: 10}` |
| seed | CLI YAML | `Advanced: {seed: 1234567}` |
| torsion library | CLI YAML | `Advanced: {use_tor_lib: true}` (samples torsions only within the `ud2_torsion_ranges` the ligand preparation writes) |
| box | CLI YAML | `Required: {center: [x, y, z]}` with `Settings: {box_size: [30, 30, 30]}` |
| hardware | CLI YAML | `Hardware: {n_cpu: 8, gpu_device_id: 0}` |
| receptor preparation | CLI YAML | `Preprocessing: {preserve_receptor_hydrogen: true}`; or `unidock2 protein_prep -r x.pdb -o x.dms` once |
| work directory | CLI YAML | `Preprocessing: {temp_dir_name: ./work}` |
| CNN rescoring | CLI YAML or flag | `Settings: {cnn_rescore: true}` or `--cnn_rescore` |
| energy decomposition | flag | `--decomp`; engine YAML `Settings: {decomp: true}` |
| FF minimisation | flag | `--ff_minimize_steps 1000`; engine YAML `Advanced: {ff_minimize_steps: 1000, ff_minimize_mode: 0}` |
| template docking | plugin block | `template: {template_docking: true, reference_sdf_file_name: ref.sdf}` |
| covalent docking | plugin block | `covalent: {covalent_ligand: true, covalent_residue_atom_info_list: [...]}` |
| atom position bias | plugin block (ud2align) | `bias: {template_sdf: t.sdf, aligner: zalign, bias_k: 0.05, bias_type: align}` |
| bias on chosen template atoms | plugin block (ud2align) | `bias: {template_sdf: t.sdf, ref_atom_indices: [0, 1, 2, 5]}` |
| bias without moving to the template | plugin block (ud2align) | `bias: {template_sdf: t.sdf, translate: false}` |
| aromatic-ring bias | plugin block (ud2align) | `bias: {template_sdf: t.sdf, ring_bias: true, ref_ring_indices: null, ring: {k: 30, s: 3, d0: 1}}` |
| inner box | engine YAML | `Settings: {inner_box: 10}` |
| weight sets | engine YAML | `Settings: {affinity_weights: {hbond: -0.587439}}` (keys not named keep their default) |
| torsion strain | automatic | from the `ud2_torsion_penalties` rows of ligand preparation; its weight is `torsion_strain` in the weight sets |
| receptor grid maps | engine YAML | `Advanced: {use_grid: true, grid_spacing: 0.375, grid_margin: 8.0}` |
| clustering atoms | engine YAML | `Advanced: {cluster_atom_indices: [[0, 1, 2, 5]]}` (runner API: `cluster_atom_indices_list`) |
| shared random streams | engine YAML | `Advanced: {rng_period: 16}` (CUDA only) |
| constraint docking | set by the plugins | engine YAML `Settings: {constraint_docking: true}` |
| CPU run that repeats exactly | engine YAML | `Hardware: {ncpu: 1}` (the CLI uses every core) |
| PROTAC ternary docking | no YAML | `examples/protac_docking/run_protac_docking.py` options (protac-docking) |

The ring bias adds rows only for template rings the ring aligner pairs with a query ring. When none pair, the run is
an atom-only biased run with no `ud2_ring_bias_raw` tag in the output, and nothing warns. Check the tag.

CNN stage environment variables:
- `UD2_CNN_PYTHON`: interpreter of the per-run scorer process.
- `UD2_CNN_PRECISION`: `fp16` (default) or `fp32` for that process.
- `UD2_CNN_SOCKET`: socket of a resident scorer, default `/tmp/ud2_cnn_<user>.sock`. Start one with
  `python unidock_processing/cnn_rescore/rescore.py --serve` to load the networks once for many runs.

## What the checker covers

It reports:
- unknown, misspelt and engine-only keys, with the closest valid name;
- the search size each `search_mode` resolves to;
- the centre and where it comes from;
- missing input, batch and reference files, with a hint when a file sits next to the YAML instead;
- a `temp_dir_name` or output directory that does not exist;
- `task` and `search_mode` values the engine rejects;
- template and covalent keys that are set but switched off;
- covalent ligands whose SDF records lack the five `covalent_*` properties;
- a CNN stage that will be skipped or has no CUDA device;
- `ud2` missing from PATH.

It does not check the chemistry: ligand preparation, receptor protonation and whether the box covers the pocket
stay with the user.
