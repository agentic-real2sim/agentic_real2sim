#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install mycpp
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-${CONDA_PREFIX:-}}" \
  -DBoost_ROOT="${CONDA_PREFIX:-}" \
  -DBOOST_ROOT="${CONDA_PREFIX:-}" \
  -DBoost_NO_SYSTEM_PATHS=ON \
  .. && \
make -j$(nproc)

# Install mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python -m pip install --no-build-isolation -e .

cd ${PROJ_ROOT}
