#!/usr/bin/env bash
# Create a conda env with unidock-agent installed (README installation steps 1-3), then check it starts.
#
#   UD2_CHANNEL=<Uni-Dock2 conda channel> bash scripts/install.sh [--name NAME] [--profile cpu|gpu]
#
#   --name NAME      conda env to create (default unidock-agent); an existing env is never overwritten
#   --profile cpu    CPU engine (default)
#   --profile gpu    CUDA engine (Linux, NVIDIA GPU)
#
# UD2_CHANNEL serves msys_viparr_lpsolve55 and ambertools_stable. The optional libmolgrid build is README step 4.
set -euo pipefail

NAME=unidock-agent
PROFILE=cpu
while [ $# -gt 0 ]; do
    case $1 in
        --name) NAME=$2; shift 2 ;;
        --profile) PROFILE=$2; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$PROFILE" = cpu ] || [ "$PROFILE" = gpu ] || die "--profile must be cpu or gpu"
[ -n "${UD2_CHANNEL:-}" ] || die "set UD2_CHANNEL to the Uni-Dock2 conda channel"
command -v conda >/dev/null || die "conda not found on PATH"
if conda env list | awk '{print $1}' | grep -qx "$NAME"; then die "conda env $NAME already exists; pick another --name"; fi
SRC=$(cd "$(dirname "$0")/.." && pwd)

PKGS=(python=3.10 pyyaml pydantic pathos numpy scipy pandas networkx rdkit mdanalysis pdbfixer openmm
      msys_viparr_lpsolve55 ambertools_stable cmake scikit-build-core pybind11 setuptools-scm pip)
ENGINE_FLAGS=()
if [ "$PROFILE" = gpu ]; then
    PKGS+=(pytorch cuda-version=12.9 cuda-nvcc cuda-cudart-dev libcurand-dev)
    ENGINE_FLAGS+=(-C cmake.define.USE_CUDA=ON)
elif [ "$(uname -s)" = Linux ]; then
    PKGS+=("pytorch=*=cpu*")
else
    PKGS+=(pytorch)
fi
# the bundled SQLite calls dlclose, and libdl is a separate library before glibc 2.34
GLIBC=$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)
if [ -n "$GLIBC" ] && [ "$(printf '%s\n' 2.34 "$GLIBC" | sort -V | head -1)" != 2.34 ]; then
    ENGINE_FLAGS+=(-C cmake.define.CMAKE_CXX_STANDARD_LIBRARIES=-ldl)
fi

in_env() { conda run -n "$NAME" --no-capture-output "$@"; }

echo "[1/4] conda create -n $NAME (profile $PROFILE)"
conda create -y -n "$NAME" --override-channels -c "$UD2_CHANNEL" -c conda-forge --no-repodata-use-zst "${PKGS[@]}"

echo "[2/4] build and install the engine"
in_env pip install --no-deps --no-build-isolation "$SRC/unidock/unidock_engine" "${ENGINE_FLAGS[@]+"${ENGINE_FLAGS[@]}"}"

echo "[3/4] install unidock_processing"
in_env pip install --no-deps --no-build-isolation "$SRC"

echo "[4/4] check"
UD2BIN=$(in_env python -c "import sysconfig, os; print(os.path.join(sysconfig.get_paths()['purelib'], 'unidock_engine/api/python/bin'))")
in_env "$UD2BIN/ud2" --version
in_env unidock2 docking --help >/dev/null

echo
echo "Installed. To use it:"
echo "  conda activate $NAME"
echo "  export PATH=\"$UD2BIN:\$PATH\""
