#!/bin/bash
# Rebuild ud2 binary, unit_test, and Python .so in one shot.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOBS="${1:-8}"

echo "=== Building ud2 + unit_test + Python .so (build_cpu) ==="
cmake --build "$SCRIPT_DIR/build_cpu" -j "$JOBS"
