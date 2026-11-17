#!/usr/bin/env bash
set -euo pipefail

export CMAKE_GENERATOR=Ninja
# CMake resolves a relative CMAKE_STRIP against the source directory when linking the Python module.
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_STRIP=$(command -v "${STRIP:?}")"
case "$BACKEND" in
    cpu) backend_args=(-Ccmake.define.USE_CUDA=OFF) ;;
    cuda128)
        backend_args=(-Ccmake.define.USE_CUDA=ON '-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=70;75;80;86;89;90;120')
        ;;
    *) echo "Unknown backend: $BACKEND" >&2; exit 1 ;;
esac
python -m pip install --no-deps --no-build-isolation ./unidock/unidock_engine \
    "${backend_args[@]}" -Ccmake.define.USE_OPENMP=ON \
    -Ccmake.define.BUILD_BIN=ON -Ccmake.define.BUILD_TEST=OFF
ln -s "../lib/python3.10/site-packages/unidock_engine/api/python/bin/ud2" "$PREFIX/bin/ud2"

SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_VERSION" python -m pip install --no-deps --no-build-isolation .
python -c "import unidock_processing"
