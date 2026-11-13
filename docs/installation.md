# Installation guide

This guide installs unidock-agent into a conda environment, builds the engine, and checks the result. The short
version is README "Installation" (steps 1–4) or `scripts/install.sh`; this page adds the choices behind each step, the
checks that prove the install works, and the known failure modes.

Contents: 1. Requirements · 2. Quick install · 3. Step by step · 4. Optional: libmolgrid · 5. Verify the install ·
6. Receptor preparation without fepfixer · 7. Machines with little disk or a shared GPU · 8. Troubleshooting

## 1. Requirements

| | CPU install | GPU install |
|---|---|---|
| OS | Linux x86_64 or macOS | Linux x86_64 |
| GPU | none | NVIDIA, compute capability 7.0, 8.0, 8.6, 8.9 or 9.0 |
| driver | – | any CUDA 12 driver (tested: 535 / CUDA 12.2) |
| conda | conda or mamba | conda or mamba |
| disk | a few GB | 7–10 GB for the conda packages of the CUDA stack |

Two conda channels are used:

- **`$UD2_CHANNEL`**, the Uni-Dock2 conda channel. It serves `msys_viparr_lpsolve55` and `ambertools_stable`. Set it
  before any command below: `export UD2_CHANNEL=<channel URL>`.
- **conda-forge**, for everything else. A conda-forge mirror can replace `-c conda-forge` if the default is slow.

The engine carries GPU machine code for sm_70/80/86/89/90 and no PTX, so a GPU of another architecture (for example
sm_75: T4, RTX 20; sm_120: RTX 50) is expected to fail with "no kernel image available". This has not been tested.

## 2. Quick install

```sh
git clone https://github.com/dptech-corp/unidock-agent.git
cd unidock-agent
UD2_CHANNEL=<channel URL> bash scripts/install.sh                     # CPU, env "unidock-agent"
UD2_CHANNEL=<channel URL> bash scripts/install.sh --profile gpu --name unidock-agent-gpu
```

The script runs steps 3.1–3.3 below, never overwrites an existing env, and ends with a short start-up check. Run the
checks of section 5 afterwards for a full test.

## 3. Step by step

### 3.1 Create the conda environment

```sh
# common to both
PKGS="python=3.10 pyyaml pydantic pathos numpy scipy pandas networkx rdkit mdanalysis pdbfixer openmm \
      msys_viparr_lpsolve55 ambertools_stable cmake scikit-build-core pybind11 setuptools-scm pip pytest"

# GPU
conda create -n unidock-agent -c "$UD2_CHANNEL" -c conda-forge --no-repodata-use-zst \
    $PKGS cuda-version=12.9 pytorch cuda-nvcc cuda-cudart-dev libcurand-dev

# CPU
conda create -n unidock-agent -c "$UD2_CHANNEL" -c conda-forge --no-repodata-use-zst \
    $PKGS "pytorch=*=cpu*"

conda activate unidock-agent
```

- `cuda-nvcc`, `cuda-cudart-dev` and `libcurand-dev` are the CUDA compiler and headers the engine build needs. pytorch
  brings only part of the toolkit, and with that partial toolkit CMake's CUDA compiler check fails.
- On Linux, `"pytorch=*=cpu*"` keeps the CPU env free of the CUDA build of torch. On macOS use `pytorch` without the
  build-string pin; `scripts/install.sh` selects this automatically.
- `pytest` is only needed for the tests of section 5 and for the libmolgrid build.
- Use `msys_viparr_lpsolve55`, not `msys_viparr`; section 5 checks that the right one is installed.
- To see whether the packages resolve without creating anything, add `--dry-run` to `conda create`.

### 3.2 Build and install the engine

```sh
pip install --no-deps --no-build-isolation ./unidock/unidock_engine                              # CPU
pip install --no-deps --no-build-isolation ./unidock/unidock_engine -C cmake.define.USE_CUDA=ON  # GPU
```

- `USE_CUDA` is OFF by default, so a GPU build must pass it.
- `--no-deps --no-build-isolation` everywhere: pip must not change what conda solved, and the build uses the env's
  cmake, pybind11 and compilers.
- **glibc older than 2.34** (check with `ldd --version`): add `-C cmake.define.CMAKE_CXX_STANDARD_LIBRARIES=-ldl`. The
  bundled SQLite calls `dlclose`, and `libdl` is a separate library before glibc 2.34; without it the engine does not
  link. `scripts/install.sh` adds the flag by itself.
- `CMAKE_BUILD_PARALLEL_LEVEL=<N>` sets the build parallelism. A CUDA build takes several minutes.

The `ud2` binary lands inside site-packages, not on PATH. The CLI and the runner look for `ud2` on PATH first:

```sh
export PATH="$(python -c 'import sysconfig, os; print(os.path.join(sysconfig.get_paths()["purelib"], "unidock_engine/api/python/bin"))'):$PATH"
ud2 --version
```

### 3.3 Install the Python packages

```sh
pip install --no-deps --no-build-isolation .  # unidock_processing and the `unidock2` command
unidock2 docking --help
```

The version of `unidock_processing` comes from git (setuptools-scm). A tree without git metadata installs as `0.0.0`.
The optional `ud2align` bias plugin is a separate package and is not included in this repository.

## 4. Optional: libmolgrid for `--cnn_rescore` (GPU only)

`--cnn_rescore` works without it: the density grids are then painted by the torch gridder. libmolgrid adds a CUDA
gridder for the same stage.

```sh
conda install -n unidock-agent -c "$UD2_CHANNEL" -c conda-forge --no-repodata-use-zst \
    "openbabel=3.1" cudnn nccl libcublas-dev "libboost-devel=1.86" "libboost-python-devel=1.86" eigen zlib pyquaternion \
    gxx_linux-64 gcc_linux-64 ninja patch pytest cuda-cuobjdump
conda deactivate && conda activate unidock-agent    # the compilers' activation scripts set CXX and CMAKE_ARGS
bash scripts/build_molgrid.sh                       # downloads the 36 MB source archive from GitHub
```

- **Do not install the conda package `molgrid`.** conda-forge's build holds GPU machine code for sm_52 only plus PTX
  (ISA 8.5). On a driver older than CUDA 12.5/12.6 the PTX cannot be compiled, and `--cnn_rescore` stops with
  `the provided PTX was compiled with an unsupported toolchain`. `scripts/build_molgrid.sh` builds the same upstream
  source (v0.5.5, with conda-forge's two patches in `scripts/molgrid/`) with machine code for every card in
  `MOLGRID_ARCHS`, so the driver version does not matter.
- `MOLGRID_ARCHS` (default `70-real;75-real;80-real;86-real;89-real;90`), `MOLGRID_ZIP=<file>` (a local copy of
  `v0.5.5.zip`, for slow or no access to GitHub; the sha256 is checked), `MOLGRID_WORKDIR`, `JOBS`.
- molgrid 0.5.5 needs `libboost-python` 1.86 and OpenBabel 3.1; with OpenBabel 3.2 `import molgrid` fails and the stage
  falls back to the torch gridder with a warning.
- `UD2_CNN_GRIDDER=torch` forces the torch gridder; `UD2_CNN_GRIDDER=molgrid` forces libmolgrid (an error instead of a
  fallback, useful to prove the GPU gridder works).

## 5. Verify the install

Run in the activated env, from the repository root, with `ud2` on PATH.

### 5.1 Packages and start-up

```sh
conda list | grep -E '^msys_viparr(_lpsolve55)? '    # expect msys_viparr_lpsolve55 only
python -I -c "import numpy, scipy, pandas, networkx, rdkit, MDAnalysis, yaml, pydantic, pathos, msys, viparr
import unidock_processing, unidock_engine, torch"
python -I -c "import msys
s = msys.FromSmilesString('c1ccccc1O')
msys.AssignBondOrderAndFormalCharge(s)
assert sorted(b.order for b in s.bonds if b.first.atomic_number == 6 and b.second.atomic_number == 6) == [1, 1, 1, 2, 2, 2]"
ud2 --version
unidock2 docking --help >/dev/null
```

For a GPU engine, `strings "$(command -v ud2)" | grep -q libcuda` should succeed. For a CPU engine it should find
nothing. If libmolgrid was built in section 4:

```sh
python -I -c "import molgrid"
cuobjdump -lelf "$CONDA_PREFIX/lib/libmolgrid.so" | grep -o 'sm_[0-9]*' | sort -u
```

The second command should list the architectures requested through `MOLGRID_ARCHS`.

### 5.2 Python tests

```sh
pytest unidock/unidock_processing/test/core -q --tb=short -p no:cacheprovider -o addopts=
```

This suite checks configuration, plugin discovery and the core runner without starting a full screen. The broader
test suites are listed in `.claude/skills/ud2-verify-change/SKILL.md`.

### 5.3 Smoke docking

The following copies the bundled 1G9V example to a temporary directory, gives the config absolute paths, lowers the
search effort for a smoke test, and checks that poses were written:

```sh
WORK=$(mktemp -d "${TMPDIR:-/tmp}/unidock-agent-smoke.XXXXXX")
mkdir -p "$WORK/tmp"
cp examples/free_docking/molecular_docking/1G9V_protein_water_cleaned.pdb "$WORK/"
cp examples/free_docking/molecular_docking/ligand_prepared.sdf "$WORK/"
sed \
  -e "s|^  receptor:.*|  receptor: $WORK/1G9V_protein_water_cleaned.pdb|" \
  -e "s|^  ligand:.*|  ligand: $WORK/ligand_prepared.sdf|" \
  -e "s|  temp_dir_name:.*|  temp_dir_name: $WORK/tmp|" \
  -e "s|exhaustiveness: .*|exhaustiveness: 64|" \
  examples/free_docking/molecular_docking/unidock_configurations.yaml > "$WORK/config.yaml"

unidock2 docking -cf "$WORK/config.yaml" -o "$WORK/poses.sdf"
grep -cF '$$$$' "$WORK/poses.sdf"       # expect at least one pose
```

Optional checks for the force-field and CNN paths:

```sh
unidock2 docking -cf "$WORK/config.yaml" --ff_minimize_steps 20 -o "$WORK/poses_ff.sdf"
grep -c '<ff_total_energy>' "$WORK/poses_ff.sdf"

# GPU only; force the source-built libmolgrid instead of silently falling back
UD2_CNN_GRIDDER=molgrid unidock2 docking -cf "$WORK/config.yaml" --cnn_rescore -o "$WORK/poses_cnn.sdf"
grep -c '<ud2_cnn_pose_score>' "$WORK/poses_cnn.sdf"
```

Keep `$WORK` when diagnosing a failure; it contains the generated DMS files, engine logs and outputs.

## 6. Receptor preparation and optional tools

```sh
unidock2 protein_prep -r receptor.pdb -o receptor.dms
```

- If `fepfixer` is on PATH, the preparation uses it. Otherwise it falls back to PDBFixer plus AmberTools `tleap`,
  which are installed by the commands above.
- The fallback removes cofactors, metals and ligands from the PDB. Modified residues can be converted to their parent
  residue, and tleap assigns its own protonation. Use fepfixer or a pre-prepared receptor DMS when those details matter.
- `utop` (unitop) or `tleap` is needed only for force-field minimisation (`--ff_minimize_steps` greater than zero) of
  a fepfixer-prepared receptor; `utop` is preferred when both are on PATH.
- `atom_mapper` is a separate package used by the MCS-tree workflow. It also requires `numba` and `rustworkx`.

## 7. Machines with little root disk or a shared GPU

Large conda environments, caches and build trees should be redirected before creating the env:

```sh
export UD2_BASE=/path/on/a/large/filesystem
export CONDA_PKGS_DIRS="$UD2_BASE/conda-pkgs"
export PIP_CACHE_DIR="$UD2_BASE/pip-cache"
export TMPDIR="$UD2_BASE/tmp"
export XDG_CACHE_HOME="$UD2_BASE/cache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
```

Use `conda create -p "$UD2_BASE/envs/unidock-agent" ...` instead of `-n` when the environment itself must also live
there. `scripts/install.sh` creates a named env, so use the manual commands in section 3 for a prefix install.

On a machine whose GPUs are managed by NVIDIA MPS:

- `conda` may load `libcuda` while computing its `__cuda` virtual package. Set `CONDA_OVERRIDE_CUDA` to the driver
  compatibility level (for example `12.2`) before `conda create`, `install`, `remove` or `info` if probing the GPU is
  undesirable.
- OpenMM can load its CUDA/OpenCL plugins during import. To force its CPU plugins:

```sh
CPU_PLUGINS="$UD2_BASE/openmm-cpu-plugins"
mkdir -p "$CPU_PLUGINS"
for f in "$CONDA_PREFIX"/lib/plugins/libOpenMMCPU.so \
         "$CONDA_PREFIX"/lib/plugins/libOpenMMPME.so \
         "$CONDA_PREFIX"/lib/plugins/*Reference.so; do
    [ -e "$f" ] && ln -sf "$f" "$CPU_PLUGINS/"
done
export OPENMM_PLUGIN_DIR="$CPU_PLUGINS"
```

- Before a GPU smoke test, inspect `nvidia-smi`, check whether `nvidia-cuda-mps` is running, and use a timeout suitable
  for the host. Do not assume that a successful import proves a CUDA kernel can run.

## 8. Troubleshooting

| symptom | cause and fix |
|---|---|
| conda cannot solve | Check that `$UD2_CHANNEL` serves `msys_viparr_lpsolve55` and `ambertools_stable`; retry the `conda create` command with `--dry-run`. |
| CMake cannot find a CUDA compiler | Install `cuda-nvcc`, `cuda-cudart-dev` and `libcurand-dev` in the active env; do not rely on pytorch's partial toolkit. |
| engine link fails at `dlclose` | On glibc older than 2.34, rebuild with `-C cmake.define.CMAKE_CXX_STANDARD_LIBRARIES=-ldl`. |
| `ud2` is not found | Add the engine's `site-packages/unidock_engine/api/python/bin` directory to PATH as shown in section 3.2. |
| `unidock_processing` has version `0.0.0` | The source tree has no git metadata. This is the configured fallback version and does not change runtime behaviour. |
| `mkdtemp` or temporary-directory error | `Preprocessing.temp_dir_name` or `$TMPDIR` does not exist; create it first. |
| `--cnn_rescore` reports unsupported PTX | A conda-forge molgrid build is being used with an older driver. Remove that package and build with `scripts/build_molgrid.sh`, or set `UD2_CNN_GRIDDER=torch`. |
| `import molgrid` fails after installing OpenBabel 3.2 | Use OpenBabel 3.1 for molgrid 0.5.5. |
| `no kernel image is available` | The GPU architecture is not in the engine or libmolgrid build. Rebuild for that architecture where supported. |
| libmolgrid download is slow or unavailable | Download `v0.5.5.zip` separately and pass its path through `MOLGRID_ZIP`; the build script verifies its sha256. |
| smoke docking writes no poses | Read the run directory and engine log first; verify the receptor and ligand paths in the generated config and that `ud2 --version` works on the same PATH. |