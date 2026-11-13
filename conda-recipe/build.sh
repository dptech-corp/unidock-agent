#!/usr/bin/env bash
set -euo pipefail

export CMAKE_GENERATOR=Ninja
python -m pip install --no-deps --no-build-isolation ./unidock/unidock_engine \
    -Ccmake.define.USE_CUDA=OFF -Ccmake.define.USE_OPENMP=ON \
    -Ccmake.define.BUILD_BIN=ON -Ccmake.define.BUILD_TEST=OFF
ln -s "../lib/python3.10/site-packages/unidock_engine/api/python/bin/ud2" "$PREFIX/bin/ud2"

SETUPTOOLS_SCM_PRETEND_VERSION="$PKG_VERSION" python -m pip install --no-deps --no-build-isolation .
python -c "import unidock_processing"
