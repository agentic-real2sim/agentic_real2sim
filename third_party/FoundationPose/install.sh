# create conda environment
conda create -n foundationpose python=3.9

# activate conda environment
conda activate foundationpose


conda install --override-channels --strict-channel-priority \
  -c nvidia/label/cuda-11.8.0 \
  cuda

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Install C++ dependencies under the active conda environment
conda install conda-forge::eigen=3.4.0 conda-forge::libboost
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CONDA_PREFIX}/lib/cmake:${CONDA_PREFIX}/lib/python3.9/site-packages/pybind11/share/cmake/pybind11${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# install dependencies
python -m pip install -r requirements.txt

# Prefer the CUDA toolkit from the active conda environment when present.
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
  elif command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  else
    echo "CUDA toolkit not found. Install CUDA with nvcc available before building nvdiffrast."
    exit 1
  fi
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
for cuda_lib_dir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib"; do
  if [[ -d "${cuda_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${cuda_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done

# The conda nvcc package expects a conda-prefixed host compiler unless ccbin is forced.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin ${CUDAHOSTCXX}"

CC="${CC}" \
CXX="${CXX}" \
CUDAHOSTCXX="${CUDAHOSTCXX}" \
CUDA_HOME="${CUDA_HOME}" \
NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS}" \
python -m pip install --quiet --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
bash build_all_conda.sh
