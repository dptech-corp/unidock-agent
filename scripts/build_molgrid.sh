#!/usr/bin/env bash
# Build libmolgrid 0.5.5 into the active conda env, with GPU machine code for the architectures in MOLGRID_ARCHS.
#
# Why: conda-forge's molgrid holds machine code for sm_52 only, plus PTX (ISA 8.5). On any newer GPU the driver has to
# compile that PTX, which fails on a driver older than CUDA 12.5/12.6 ("the provided PTX was compiled with an unsupported
# toolchain") and then `unidock2 docking --cnn_rescore` stops. Built here, the library holds real machine code for the
# cards in use, so the driver version does not matter. Same recipe as conda-forge (upstream v0.5.5 plus the two patches in
# scripts/molgrid/), with the architectures passed on the cmake command line.
#
# Run it in the activated env of the README, after the build dependencies of README step 4 are installed:
#   bash scripts/build_molgrid.sh
#
# Environment, all optional:
#   MOLGRID_ARCHS    CMake CUDA architectures (default "70-real;75-real;80-real;86-real;89-real;90": machine code for each,
#                    and PTX for the last one for newer cards)
#   MOLGRID_ZIP      a local copy of v0.5.5.zip (36 MB); otherwise it is downloaded from GitHub, resuming a partial file
#   MOLGRID_WORKDIR  where the source is unpacked and built (default ./build_molgrid)
#   JOBS             build parallelism (default: number of cores, at most 32)
set -euo pipefail

VERSION=0.5.5
SHA256=9d1810145f21777a66f45827f539d5e829cd1c23e251268996741dd6551a5220
ARCHS=${MOLGRID_ARCHS:-70-real;75-real;80-real;86-real;89-real;90}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${MOLGRID_WORKDIR:-$PWD/build_molgrid}
CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
JOBS=${JOBS:-$(( CORES < 32 ? CORES : 32 ))}

die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || die "libmolgrid is built here for linux-64 only"
[ -n "${CONDA_PREFIX:-}" ] || die "no conda env is active: run 'conda activate <env>' first"
PREFIX=$CONDA_PREFIX
[ "$(command -v python)" = "$PREFIX/bin/python" ] || die "python is not the one of $PREFIX; activate the env first"
[ -x "$PREFIX/bin/nvcc" ] && [ -d "$PREFIX/targets/x86_64-linux/include" ] \
    || die "the CUDA toolkit is not in $PREFIX: install cuda-nvcc, cuda-cudart-dev and libcurand-dev (README step 1)"
[ -n "${CXX:-}" ] && command -v "$CXX" >/dev/null \
    || die "the conda C++ compiler is not active: install gxx_linux-64 and gcc_linux-64 (README step 4), then 'conda deactivate && conda activate $(basename "$PREFIX")' so that its activation script sets CXX"
for tool in cmake ninja patch curl; do command -v "$tool" >/dev/null || die "$tool not found; install the packages of README step 4"; done
[ -d "$PREFIX/include/openbabel3" ] || die "openbabel headers not found in $PREFIX; install openbabel=3.1 (README step 4)"
python -c "import pytest" 2>/dev/null || die "pytest is not installed in $PREFIX: libmolgrid's CMake refuses to configure without it (README step 4)"
compgen -G "$PREFIX/conda-meta/molgrid-*.json" >/dev/null && die "the conda package molgrid is installed in this env; run 'conda remove molgrid' first, this build replaces its files"

mkdir -p "$WORK"
ZIP=${MOLGRID_ZIP:-$WORK/v$VERSION.zip}
if [ ! -f "$ZIP" ]; then
    echo "downloading libmolgrid $VERSION (36 MB) from GitHub; set MOLGRID_ZIP to use a local copy"
    curl -fL -C - --retry 5 --retry-delay 10 --speed-limit 1000 --speed-time 120 -o "$ZIP.part" \
        "https://github.com/gnina/libmolgrid/archive/refs/tags/v$VERSION.zip" \
        || die "download failed (a partial file stays at $ZIP.part and the next run resumes it); or fetch v$VERSION.zip elsewhere and set MOLGRID_ZIP"
    mv "$ZIP.part" "$ZIP"
fi
echo "$SHA256  $ZIP" | sha256sum -c --status - || die "$ZIP does not match the sha256 of libmolgrid $VERSION"

SRC=$WORK/libmolgrid-$VERSION
rm -rf "$SRC"
python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$WORK"
for p in cmake-fix.patch python-setup.patch; do
    patch -d "$SRC" -p1 --forward -s <"$HERE/molgrid/$p" || die "patch $p failed"
done

# The CUDA 12 conda packages keep headers and libraries under targets/x86_64-linux, which is also the root that CMake's
# FindCUDA must be given: with the env root it finds nvcc but not CUDA_INCLUDE_DIRS.
export CPATH=$PREFIX/targets/x86_64-linux/include:$PREFIX/include
export LIBRARY_PATH=$PREFIX/targets/x86_64-linux/lib:$PREFIX/lib
export NCCL_ROOT_DIR=$PREFIX NCCL_INCLUDE_DIR=$PREFIX/include USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=0
export USE_STATIC_CUDNN=0 CUDNN_INCLUDE_DIR=$PREFIX/include

mkdir -p "$SRC/build"
cd "$SRC/build"
echo "configuring libmolgrid $VERSION: machine code for $ARCHS"
# CMAKE_ARGS is set by the conda compiler packages and is meant to be split into words
# shellcheck disable=SC2086
cmake -GNinja ${CMAKE_ARGS:-} .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCUDA_TOOLKIT_ROOT_DIR="$PREFIX/targets/x86_64-linux" -DCUDA_NVCC_EXECUTABLE="$PREFIX/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
    -DOPENBABEL3_INCLUDE_DIR="$PREFIX/include/openbabel3/" -DOPENBABEL3_LIBRARIES="$PREFIX/lib/libopenbabel.so" \
    -DPYTHON_NUMPY_INCLUDE_DIR="$(python -c 'import numpy; print(numpy.get_include())')" -DBUILD_STATIC=0 -DBUILD_SHARED=1 \
    >"$WORK/cmake.log" 2>&1 || { tail -15 "$WORK/cmake.log" >&2; die "cmake failed, see $WORK/cmake.log"; }
echo "building with $JOBS jobs (about 3 minutes)"
ninja -j "$JOBS" >"$WORK/build.log" 2>&1 || { tail -15 "$WORK/build.log" >&2; die "build failed, see $WORK/build.log"; }
# setup.py reads the version from PKG_VERSION; without it the Python package is not installed
PKG_VERSION=$VERSION ninja install >"$WORK/install.log" 2>&1 || { tail -15 "$WORK/install.log" >&2; die "install failed, see $WORK/install.log"; }

python -c "import molgrid" || die "installed, but 'import molgrid' fails"
echo "installed $PREFIX/lib/libmolgrid.so"
if [ -x "$PREFIX/bin/cuobjdump" ]; then
    echo "GPU machine code in it: $("$PREFIX/bin/cuobjdump" -lelf "$PREFIX/lib/libmolgrid.so" | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
fi
