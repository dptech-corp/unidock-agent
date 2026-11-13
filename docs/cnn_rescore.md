# CNN pose rescoring

An optional last step of a docking run that reorders each ligand's output poses by the engine's pose score blended
with the gnina pose-CNN ensemble (the five published gnina 1.0 models, Apache 2.0, bundled under
`unidock_processing/cnn_rescore/weights`). It does not help rank ligands in a screen, so leave it off for screening.

## Enable

    unidock2 docking ... --cnn_rescore            # or  Settings: cnn_rescore: true  in the YAML

Requirements: `torch` (with CUDA; CPU works) and, for the GPU gridder, `molgrid` in the
interpreter that runs the stage. `torch` comes with unidock2 (the conda package depends on `pytorch` in every build).
`molgrid` is optional: `scripts/build_molgrid.sh` (README, step 4) builds libmolgrid 0.5.5 with GPU machine code for
sm_70/75/80/86/89/90, which runs on any CUDA 12 driver. The conda package of unidock2 depends on conda-forge's molgrid
in the CUDA builds, and that build holds machine code for sm_52 only plus PTX (ISA 8.5): on a driver older than CUDA
12.5/12.6 its kernel fails with `the provided PTX was compiled with an unsupported toolchain`, and because the default
`auto` gridder uses molgrid whenever it imports, the stage stops; set `UD2_CNN_GRIDDER=torch` there. molgrid 0.5.5 links
openbabel 3.1's library without bounding the version, so with openbabel 3.2 `import molgrid` fails and the stage falls
back to the torch gridder with a logged warning. The atom typing comes from our DMS
tables; the density grids come from libmolgrid's CUDA GridMaker when `molgrid` imports in that interpreter, otherwise
(CPU builds, macOS) from the torch gridder in `cnn_rescore/grid.py`, which paints the same grids (agreement 1e-5).
`UD2_CNN_GRIDDER=torch` (or `--gridder`) forces the torch path. The
stage does no OpenBabel typing (molgrid only links its library). It runs outside the main process so that torch and the networks are loaded once per
server rather than once per docking run. Default off.

Two ways to run it:
- **Resident scorer (recommended when docking many ligands or many runs):** start once,
  `python unidock_processing/cnn_rescore/rescore.py --serve` in an interpreter that has torch. It loads the five networks once and listens on
  `/tmp/ud2_cnn_<user>.sock` (`UD2_CNN_SOCKET` to change). Every docking run then rescores through it. Stop it with Ctrl-C or SIGTERM.
- **Per-run process (fallback):** when no server answers, the runner starts `rescore.py` itself in the current
  interpreter, which has torch (and molgrid, if built); set `UD2_CNN_PYTHON` only to run the stage in
  another interpreter. Every run then pays for interpreter start, CUDA context and
  model loading on top of the scoring.

Precision: fp16 autocast by default. `--precision fp32` on the server, or `UD2_CNN_PRECISION=fp32` for the per-run process, restores the
reference numbers.

## What it does

1. Types every atom from the enriched DMS files (engine Vina type + aromatic flag -> gnina atom class; receptor
   aromatic carbons from the standard residue templates).
2. Paints each pose into a 28-channel density grid (23.5 A cube, 0.5 A, exactly libmolgrid's Gaussian profile, verified to
   2e-5 against it) and scores it with the five networks; the pose score is the mean probability that the pose is within 2 A.
3. Inside each ligand: blend = -z(vina_pose_score) + 1.5 z(cnn); poses are rewritten in decreasing blend order with the
   tags `ud2_cnn_pose_score`, `ud2_cnn_blend_score`, `ud2_cnn_rank`. All other tags, including `ud2_pose_index` (the
   engine's order), are kept, so the engine ranking can always be recovered.

## Pose count

The engine keeps a re-ranking pool of `max(rescore_pool, num_pose)` cluster representatives (Advanced.rescore_pool,
default 10), sorts it with its pose pass and writes the first `num_pose`. The CNN stage re-ranks what is written, so
with the stage on the runner has the engine write the whole pool and the stage writes back only the first `num_pose`
of each ligand after the reorder (`rescore.py --keep K`, or `keep` in a server request). Asking for one pose yields the
CNN-blended choice with every tag the engine wrote for that pose; the batch files in `unidock2_output/` still hold the
whole shortlist. The affinity label of a promoted pose keeps the engine's definition: it is referenced to the intra
energy of the engine's rank-1 pose, as Vina does for every non-top pose. With the stage off, `num_pose` reaches the
engine unchanged.

## When it does nothing

- Template / covalent (constraint) docking runs: skipped, engine order kept.
- Biased runs (an enabled bias plugin: `bias: template_sdf` or `--bias_template_sdf`): skipped, the engine order of the
  bias-aware pose pass is kept and `num_pose` reaches the engine unchanged (`Plugin.keeps_engine_order`).
- A ligand with a pose extending beyond the 11.75 A grid half-edge (about 40+ heavy atoms, stretched): that ligand keeps
  the engine order and gets no CNN tags; a warning names it.
- Failure of the stage (missing torch / molgrid, no enriched DMS): the engine SDF is left as written and the error is raised.
