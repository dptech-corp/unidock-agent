---
name: ud2-docking
description: Dock, re-dock or virtually screen small molecules with Uni-Dock2 (the `unidock2 docking` CLI, `UnidockProtocolRunner`, the `ud2` engine) — free docking into a pocket, re-docking a crystal ligand, screening a ligand library, scoring a given pose, CNN rescoring of poses, and reading the output SDF tags. Use whenever the user asks to dock, re-dock, screen, score or rank ligands against a receptor with Uni-Dock2, or asks what vina_pose_score / vina_affinity_score / vina_binding_free_energy mean. For core-constrained docking on a reference pose use ud2-template-docking; covalent ligands use ud2-covalent-docking; aligning to a template pose with a bias use ud2-bias-align; engine-only options (inner box, weight overrides, decomposition, task modes) use ud2-engine-yaml.
---

# Docking and screening with Uni-Dock2

The full option list is in `docs/features_and_plugins.md`. This skill is the decisions and the traps.

## Before the first run

1. **`ud2` must be on PATH.** `which ud2 && ud2 --version` (0.4.6 for this tree). A pip install of `unidock_engine`
   puts the binary inside site-packages (`unidock_engine/api/python/bin/ud2`), not on PATH, and the runner then
   fails. Put that directory, or a build's `bin/`, first on PATH.
2. **CUDA or CPU build.** Both score a pose the same; their searches differ, so poses differ. The CPU engine gives the
   same output on a re-run only at `ncpu: 1`.

## 1. Prepare the receptor once, then reuse the DMS

```bash
unidock2 protein_prep -r receptor.pdb -o receptor.dms
```

Dock against `receptor.dms` from then on. A PDB is re-prepared on every run (fepfixer when on PATH, otherwise
PDBFixer + tleap), which costs seconds per run, and fepfixer's hydrogen placement is not deterministic: two runs from
the same PDB are not comparable. A DMS that already carries `ud2_atom_props` is used as is.

## 2. Run

```bash
unidock2 docking -r receptor.dms -l ligand.sdf -c X Y Z -cf config.yaml -o poses.sdf
unidock2 docking -r receptor.dms -lb ligands.txt -cf config.yaml        # one SDF path per line
```

```yaml
Required:  {center: [X, Y, Z]}           # -c on the command line overrides it
Advanced:  {exhaustiveness: 512, mc_steps: 40, opt_steps: -1, refine_steps: 5,
            num_pose: 10, rescore_pool: 10, rmsd_limit: 1.0, energy_range: 10.0, seed: 9996}
Settings:  {box_size: [25, 25, 25], task: screen, search_mode: free}
Preprocessing: {temp_dir_name: ./tmp, preserve_receptor_hydrogen: true}
```

```python
from unidock_processing.core.runner import UnidockProtocolRunner
UnidockProtocolRunner("receptor.dms", ["lig.sdf"], target_center=(x, y, z), box_size=(25, 25, 25),
                      search_mode="free", exhaustiveness=512, mc_steps=40, num_pose=10, seed=9996,
                      working_dir_name="work", docking_pose_sdf_file_name="poses.sdf").run_unidock_protocol()
```

The full YAML layout, recipes and a checker that prints the values the run will use are in ud2-config-yaml.

Rules that change results:

- **`search_mode` overwrites your search size.** `fast` / `balance` (the default and what the shipped examples use)
  / `detail` set exhaustiveness and mc_steps (CUDA 128/20, 256/30, 512/40; CPU half the exhaustiveness). Only
  `free` uses the values you wrote. Check `ud2.log` / `ud.log` for the resolved values.
- **The pose pass re-ranks `rescore_pool` clustered poses** and writes the best `num_pose`. With `num_pose: 1`, raise
  `rescore_pool` (the CNN stage uses 100) or the re-ranking has nothing to choose from.
- **`ff_minimize_steps` defaults to 0.** Above 0 it needs force-field parameters on the receptor: the PDBFixer + tleap
  route has them, and the fepfixer route needs `utop` or `tleap` on PATH. It runs serially on CUDA (very slow).
- **Not settable here:** inner box, angular hbonds (always on), weight overrides. Runner runs use the compiled
  fitted weights and no inner box. Use the engine YAML for those (ud2-engine-yaml).

## 3. CNN rescoring

`--cnn_rescore` (YAML `Settings: cnn_rescore: true`) rescores each ligand's pool with the gnina pose-CNN ensemble
blended with the engine score, and keeps `num_pose` poses. Use it for **pose prediction**: on the re-docking
benchmarks it lifted top-1 from 76.4 to 81.8 % (udbench) and 58.8 to 66.2 % (PoseX). It gave **no screening gain**,
so do not use it to rank a library. torch ships with unidock2 and libmolgrid's GPU gridder is optional (`scripts/build_molgrid.sh`); `UD2_CNN_PYTHON`
runs the stage in another interpreter. It is skipped for template, covalent and biased runs.

## 4. Read the output

Each ligand's poses come in the order the engine (or the CNN blend) ranked them; `ud2_pose_index` is the engine rank.

| tag | use it for |
|---|---|
| `vina_pose_score` | ranking the poses of one ligand; the written order |
| `vina_affinity_score` | comparing ligands (plain Vina weights) |
| `vina_binding_free_energy` | reporting only: it is relative to the intra energy of the run's first pose, so never sort poses from several runs by it |
| `ud2_cnn_pose_score`, `ud2_cnn_blend_score` | CNN stage results, when it ran |
| `residue_*`, `ud2_*_raw` | decomposition and diagnostics (`--decomp`) |

## Traps

- Docking far from the pocket: check the centre in the engine log; a missing or wrong centre docks elsewhere.
- Comparing two runs: same binary, same receptor DMS, same seed; kernel or build changes move GPU results.
- A score is not an affinity prediction: `vina_affinity_score` ties heavy-atom count on assay data.
