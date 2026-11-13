# unidock-agent

GPU-accelerated molecular docking, built from Uni-Dock2 and reorganised to be easy for people and coding agents to use
and to extend.

unidock-agent starts from the Uni-Dock2 docking engine (dp.tech) and differs from it in three ways:

- **A basic engine plus plugins.** Free docking, screening, scoring, CNN rescoring, torsion-library sampling, force
  fields and energy decomposition are the core. Features that need a second molecule or a chemical annotation are
  plugins: template (core-constrained) docking and covalent docking ship with the core, and the template-alignment
  bias is provided by the external `ud2align` package, which is not included in this repository. Plugins are found
  through the `unidock2.plugins` entry-point group.
- **Agent skills and feature docs.** `.claude/skills/` holds one skill per task an agent is asked to do (docking,
  template docking, covalent docking, bias alignment, the engine YAML, PROTAC docking, verifying a change), and
  [docs/features_and_plugins.md](docs/features_and_plugins.md) lists every feature and option.
- **Measured behaviour.** Fitted scoring weights are the compiled defaults, the CNN pose rescoring stage is built
  in, and unbiased runs were checked byte-identical to the original code on the re-docking, screening, template and
  PROTAC benchmarks.

The command, package and binary names are still the Uni-Dock2 ones: the `unidock2` CLI, the `ud2` engine binary, and
the `unidock_processing` and `unidock_engine` packages.

---

## Installation

Install from this repository. The `unidock2` conda package on the Uni-Dock2 conda channel is a Uni-Dock2 build (0.6.x),
not unidock-agent: it has none of the bundled plugins or the changes above.

`scripts/install.sh` runs steps 1–3 below in one go and checks the result:
`UD2_CHANNEL=<channel> bash scripts/install.sh [--profile gpu] [--name NAME]`.
For environment choices, full verification, shared-disk/GPU setup and troubleshooting, see
[docs/installation.md](docs/installation.md).

### 1. Clone and create the environment

```sh
git clone https://github.com/dptech-corp/unidock-agent.git
cd unidock-agent

# UD2_CHANNEL: the Uni-Dock2 conda channel, which serves msys_viparr_lpsolve55 and ambertools_stable
# GPU (Linux, an NVIDIA GPU of compute capability 7.0, 8.0, 8.6, 8.9 or 9.0; tested with driver 535, CUDA 12.2)
conda create -n unidock-agent -c "$UD2_CHANNEL" -c conda-forge --no-repodata-use-zst \
    python=3.10 cuda-version=12.9 pyyaml pydantic pathos numpy scipy pandas networkx rdkit mdanalysis pdbfixer openmm \
    msys_viparr_lpsolve55 ambertools_stable pytorch \
    cuda-nvcc cuda-cudart-dev libcurand-dev cmake scikit-build-core pybind11 setuptools-scm pip
conda activate unidock-agent
```

For a CPU-only machine (macOS, CPU servers) leave out `cuda-version`, `cuda-nvcc`, `cuda-cudart-dev` and
`libcurand-dev`. Docking works there, but `--cnn_rescore` does not: the runner starts the CNN
scorer on `cuda:<gpu_device_id>`.

`cuda-nvcc`, `cuda-cudart-dev` and `libcurand-dev` are the CUDA compiler and headers the engine build needs. pytorch
brings only part of the toolkit, and with that partial toolkit CMake's CUDA compiler check fails.

`pytorch` is what the CNN rescoring stage needs: its density grids are painted in torch, so `--cnn_rescore` works with
nothing more. Step 4 adds libmolgrid's CUDA gridder, which makes the whole stage about four times faster.

### 2. Build and install the engine

```sh
# GPU build (USE_CUDA is OFF by default, so pass it)
pip install --no-deps --no-build-isolation ./unidock/unidock_engine -C cmake.define.USE_CUDA=ON
# CPU build
pip install --no-deps --no-build-isolation ./unidock/unidock_engine
```

This installs the pybind11 module and the `ud2` binary, but the binary lands inside site-packages, not on PATH. Put it
on PATH, because the CLI and the runner look for `ud2` there first:

```sh
export PATH="$(python -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "unidock_engine/api/python/bin"))'):$PATH"
ud2 --version
```

Alternatively build the binary directly with CMake and put its `bin/` on PATH:

```sh
cmake -S unidock/unidock_engine -B build_cuda -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DBUILD_TEST=ON
cmake --build build_cuda -j          # binary: build_cuda/bin/ud2, unit tests: build_cuda/test/unit/unit_test
```

Use `-DUSE_CUDA=OFF` for the CPU backend (OpenMP). The CPU engine repeats its output exactly only with one thread
(`Hardware: ncpu: 1` in the engine YAML); `unidock2` runs it on every core. The CUDA engine repeats its output for the
same binary.

### 3. Install the core package

```sh
pip install --no-deps --no-build-isolation .  # unidock_processing and the `unidock2` command
unidock2 --version && unidock2 docking --help
```

The core version comes from git (setuptools-scm), so install from a clone, not from a copied directory.
The optional `ud2align` bias plugin is a separate package and is not included in this repository.

### 4. Optional: the GPU gridder for `--cnn_rescore`

Install what the build needs, then build libmolgrid into the env:

```sh
conda install -n unidock-agent -c "$UD2_CHANNEL" -c conda-forge --no-repodata-use-zst \
    "openbabel=3.1" cudnn nccl libcublas-dev "libboost-devel=1.86" "libboost-python-devel=1.86" eigen zlib pyquaternion \
    gxx_linux-64 gcc_linux-64 ninja patch pytest cuda-cuobjdump
conda deactivate && conda activate unidock-agent     # the compilers' activation scripts set CXX and CMAKE_ARGS
bash scripts/build_molgrid.sh                        # about 3 minutes; downloads the 36 MB source archive from GitHub
```

Do not install the conda package `molgrid` in the same env. conda-forge's build holds GPU machine code for sm_52 only,
plus PTX that the driver must compile at run time; a driver older than CUDA 12.5/12.6 cannot, and `--cnn_rescore` then
stops with `the provided PTX was compiled with an unsupported toolchain`. `scripts/build_molgrid.sh` builds the same
upstream source (with conda-forge's two patches, `scripts/molgrid/`) for sm_70, 75, 80, 86, 89 and 90
(`MOLGRID_ARCHS`), so no particular driver version is needed. `MOLGRID_ZIP=<file>` uses a local copy of
`v0.5.5.zip`. molgrid links the OpenBabel of the env (3.1: it does not load with 3.2); nothing of libmolgrid or
OpenBabel is shipped in this repository. Skip the step and the stage keeps working with the torch gridder.

### Optional tools

- `fepfixer` on PATH is used to prepare receptor PDBs; without it the preparation falls back to PDBFixer + tleap.
- `utop` (unitop) or `tleap` (AmberTools) is needed only for force-field minimisation (`--ff_minimize_steps` > 0) of a
  fepfixer-prepared receptor; utop is used when both are on PATH.
- The atom_mapper package (MCS trees for congeneric series) is separate; it also needs `numba` and `rustworkx`.

---

## Usage

### Prepare the receptor once

```sh
unidock2 protein_prep -r receptor.pdb -o receptor.dms
```

Dock against the DMS from then on. Preparing a PDB costs seconds on every run, and fepfixer's hydrogen placement is
not deterministic, so runs started from the same PDB are not comparable.

### Dock

```sh
unidock2 docking -r receptor.dms -l ligand.sdf -c X Y Z -cf config.yaml -o poses.sdf
unidock2 docking -r receptor.dms -lb ligand_list.txt -cf config.yaml      # one SDF path per line
```

| option | meaning |
|---|---|
| `-r`, `--receptor` | receptor, PDB or DMS |
| `-l`, `--ligand` / `-lb`, `--ligand_batch` | one ligand (SDF or DMS) / a text file of ligand SDF paths; give at least one |
| `-c`, `--center X Y Z` | box centre; without it the YAML `Required: center` is used |
| `-o` | output pose SDF (default `unidock2_pose.sdf`) |
| `-cf`, `--configurations` | the YAML file with every other option |
| `--cnn_rescore` | rescore each ligand's poses with the gnina pose-CNN ensemble |
| `--decomp` | per-atom and per-residue energy decomposition |
| `--ff_minimize_steps N` | force-field minimisation steps after the search (default 0, off; serial and slow on CUDA) |
| `--template_docking`, `--reference_sdf_file_name`, `--core_atom_mapping_dict_list` | template plugin |
| `--covalent_ligand`, `--covalent_residue_atom_info_list` | covalent plugin |
| `--bias_template_sdf`, `--bias_aligner`, `--bias_k`, `--bias_type`, `--bias_ring`, ... | external bias plugin (`ud2align` installed separately) |

`unidock2 --version` and `unidock2 docking --help` show the installed version and the full option list.

### Configuration YAML

```yaml
Required:
  receptor: receptor.dms
  ligand: ligand.sdf
  center: [5.122, 18.327, 37.332]
Advanced:
  exhaustiveness: 512        # MC chains per ligand
  randomize: true
  mc_steps: 40
  opt_steps: -1              # -1 = per-ligand heuristic
  refine_steps: 5
  num_pose: 10               # poses written per ligand
  rescore_pool: 10           # clustered poses the pose pass re-ranks before writing num_pose
  rmsd_limit: 1.0            # clustering RMSD between output poses
  energy_range: 5.0          # kcal/mol window above the best pose
  seed: 1234567
  use_tor_lib: false         # restrict torsion sampling to the torsion-library ranges
Hardware:
  n_cpu: null                # CPU workers for ligand preparation
  gpu_device_id: 0
Settings:
  box_size: [30.0, 30.0, 30.0]
  task: screen               # screen | score (input pose) | optimize (local optimisation of the input pose) | mc
  search_mode: balance       # fast | balance | detail overwrite exhaustiveness and mc_steps; only free keeps yours
  cnn_rescore: false
Preprocessing:
  preserve_receptor_hydrogen: false
  temp_dir_name: /tmp        # intermediate files; must exist; /tmp is deleted after the run, any other directory is kept
  output_docking_pose_sdf_file_name: unidock2_pose.sdf
  output_receptor_dms_file_name: receptor_parameterized.dms    # for protein_prep
```

The values shown are the defaults. `search_mode` presets are CUDA 128/20, 256/30 and 512/40 (exhaustiveness /
mc_steps) and half the exhaustiveness on CPU; write `search_mode: free` to use your own values.

Plugin blocks go next to these:

```yaml
template:                    # template (core-constrained) docking
  template_docking: true
  reference_sdf_file_name: reference.sdf
  core_atom_mapping_dict_list: null    # null = automatic MCS; or one {reference_atom_idx: query_atom_idx} per ligand
covalent:                    # covalent docking
  covalent_ligand: true
  covalent_residue_atom_info_list: [["", "CYX", 25, "CA"], ["", "CYX", 25, "CB"], ["", "CYX", 25, "SG"]]
bias:                        # alignment to a template pose (needs external ud2align)
  template_sdf: template.sdf
  aligner: zalign            # zalign | hbond | hbond_charge
  bias_k: 0.05
```

The template and covalent keys are still read under `Preprocessing:`, with a deprecation warning.

### Output

Each ligand's poses are written in rank order.

| tag | use it for |
|---|---|
| `vina_pose_score` | ranking the poses of one ligand (the written order) |
| `vina_affinity_score` | comparing ligands (plain Vina weights); rank a virtual screen on this |
| `vina_binding_free_energy` | reporting; it is relative to the intra energy of the run's first pose, so do not sort poses from several runs by it |
| `ud2_cnn_pose_score`, `ud2_cnn_blend_score` | the CNN stage, when it ran |
| `ud2_bias_energy` | the bias energy of a biased run (the bias never enters the scores above) |

### Engine-only options

The `ud2` binary reads its own YAML (`ud2 config.yaml`). It accepts options the CLI does not expose:

- `Settings.inner_box`: a ligand-centroid inner box (10 Å is the benchmark protocol; 0 = off).
- `Settings.angular_hbond`: on by default.
- `Settings.score_weights`, `rescore_weights`, `affinity_weights`: the three weight sets; the compiled defaults are
  the fitted sets.
- `Settings.decomp`, `Advanced.use_grid`.
- The `bias:` block (`type`, `k`, `ring`).

The keys are listed in [docs/features_and_plugins.md](docs/features_and_plugins.md) and the `ud2-engine-yaml` skill.
The fastest way to get a valid engine YAML and its DMS inputs is to run `unidock2 docking` once with
`temp_dir_name` set to an existing directory other than `/tmp` and take `_ud2_engine_config.yaml` from the
`docking_*` directory it leaves there.

### Python API

```python
from unidock_processing.core.runner import UnidockProtocolRunner

runner = UnidockProtocolRunner("receptor.dms", ["ligand.sdf"], target_center=(5.122, 18.327, 37.332),
                               box_size=(30, 30, 30), search_mode="free", exhaustiveness=512, mc_steps=40,
                               num_pose=10, working_dir_name="work", docking_pose_sdf_file_name="poses.sdf")
runner.run_unidock_protocol()
```

A configured plugin goes in `plugins=[...]`, for example `BiasPlugin` from `ud2align.plugin`.

### Receptor PDB input and the non-protein part of the pocket

When `fepfixer` is on PATH a receptor PDB goes through it, and then through `utop` if that is on PATH too (utop adds
the force-field parameters). Without utop, `tleap` adds them when `--ff_minimize_steps` > 0; otherwise the DMS is
Vina-only. Without fepfixer the PDBFixer + tleap route is used instead, whether or not utop is present.

fepfixer deletes every residue it has no Amber template for (hemes, FAD/NAD/SAM, metal ions, glycans,
phospho-residues, free amino acids, ...) and utop aborts on the ones it keeps but cannot parametrize. On the fepfixer
route the PDB is therefore split first:

- **Polymer and water** go through fepfixer, and through utop when it is present.
- **Every other HETATM residue** is rebuilt from the wwPDB Chemical Component Dictionary: bond orders, formal
  charges, hydrogens placed by RDKit, and phosphate, sulfate and carboxylic acid groups deprotonated. It is appended
  to the DMS with bonds but without force-field parameters, which is all the Vina typing needs.
- **A modified polymer residue** (TPO, PTR, SEP, MSE, KCX, ...) stays in the chain as its parent residue, and the
  modification comes back as a bonded fragment, so the backbone is never broken.
- **Residues with an incomplete backbone** are dropped with a warning.

Build the CCD index once:

    curl -O https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
    python -m unidock_processing.unidocktools.ccd components.cif.gz     # ~/.cache/unidock/ccd.sqlite, or UD2_CCD_DB

Without the index, het residues get bonds perceived from geometry, no hydrogens and no charges.

---

## Examples

The `examples/` directory has one case per mode. The example YAMLs use `search_mode: balance`.

| case | command (run inside the example directory) |
|---|---|
| free docking, `examples/free_docking/molecular_docking` | `unidock2 docking -r 1G9V_protein_water_cleaned.pdb -l ligand_prepared.sdf -cf unidock_configurations.yaml` |
| virtual screening, `examples/free_docking/virtual_screening` | `unidock2 docking -r 5WIU_protein_cleaned.pdb -l actives_cleaned.sdf -cf unidock_configurations.yaml` |
| template docking, automatic map, `examples/constraint_docking/automatic_atom_mapping` | `unidock2 docking -r Bace.pdb -lb ligand_batch.dat -cf unidock_configurations.yaml` |
| template docking, manual map, `examples/constraint_docking/manual_atom_mapping` | `unidock2 docking -r protein.pdb -l ligand.sdf -cf unidock_configurations.yaml` |
| covalent docking, `examples/covalent_docking/1EWL` | `unidock2 docking -r 1EWL_prepared.pdb -l covalent_mol.sdf -cf unidock_configurations.yaml` |
| template docking into RNA, `examples/constraint_docking/RNA_case` | `unidock2 docking -r test_receptor_topology_RNA.pdb -l ligand.sdf -cf unidock_configurations.yaml` |

PROTAC ternary complex, from the repository root:

```sh
# MZ1 / BRD4 / VHL; the script writes into its input directory, so run it on a copy
mkdir -p run_mz1 && cp -r examples/protac_docking/MZ1-BRD4-VHL run_mz1/input
python examples/protac_docking/run_protac_docking.py --input-dir run_mz1/input    # results in run_mz1/ud2_output
```

The `protac-docking` skill explains the options and how to check the result.

Notes on these examples:

- The box centre comes from each example YAML; add `-c X Y Z` to override it.
- Every example YAML still lists the template and covalent keys under `Preprocessing:`, so each run prints a
  deprecation warning; the results are not affected.
- Covalent ligands carry their residue atoms and the `covalent_*` properties; see the `ud2-covalent-docking` skill
  for the input contract.

---

## For agents and developers

**Skills** (`.claude/skills/`):

| skill | for |
|---|---|
| `ud2-docking` | docking, re-docking, screening, scoring, the output tags |
| `ud2-template-docking` | core-constrained docking and the atom_mapper workflow |
| `ud2-covalent-docking` | covalent ligands |
| `ud2-bias-align` | alignment to a template pose with external ud2align |
| `ud2-config-yaml` | writing and checking the `unidock2` configuration YAML |
| `ud2-engine-yaml` | the `ud2` engine YAML |
| `protac-docking` | PROTAC ternary complexes |
| `ud2-verify-change` | checking a code change |

**Docs** (`docs/`):

| doc | covers |
|---|---|
| `features_and_plugins.md` | every feature and option, and what needs installing |
| `cnn_rescore.md` | the CNN stage |
| `energy_decomposition.md` | the decomposition |
| `hbond_angular_scoring.md` | angular hydrogen bonds |
| `docking_flow.md` | the pipeline |
| `testing_guide.md` | the tests |

**Tests:**

```sh
pip install pytest pytest-cov                     # once; pytest.ini passes --cov
pytest unidock/unidock_processing/test/core      # core, no plugin needed
pytest                                            # core + template + covalent suites (pytest.ini)
<build>/test/unit/unit_test                       # engine unit tests
```

The benchmark scripts, run records and frozen md5 baselines are kept outside this repository.
