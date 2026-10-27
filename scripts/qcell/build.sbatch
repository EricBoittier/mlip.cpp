#!/bin/bash -l
#SBATCH --job-name=qcell-build
#SBATCH --account=lp161
#SBATCH --partition=debug
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/capstor/scratch/cscs/boittier/qcell/logs/%x-%j.out
# CUDA build of src/mlip.cpp (EricBoittier/mlip.cpp feat/plato-qcell = feat/plato-d2 dc44038) for GH200 sm_90,
# inside the qm9esp-cuda container (gcc 13, CUDA 12.8); cmake from mptrj-py put on PATH explicitly.
set -euo pipefail
Q=/capstor/scratch/cscs/boittier/qcell
B=$Q/build-cuda
srun --environment=qm9esp-cuda bash -c "
  set -euo pipefail
  export PATH=/users/boittier/venvs/mptrj-py/bin:/users/boittier/.local/bin:\$PATH
  which cmake gcc nvcc; gcc --version | head -1
  cmake -S $Q/src/mlip.cpp -B $B -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 > $Q/logs/configure.log 2>&1 || { tail -30 $Q/logs/configure.log; exit 3; }
  cmake --build $B -j 72 --target mlipcpp-train-plato mlipcpp-plato > $Q/logs/make.log 2>&1 && echo BUILD OK || { tail -40 $Q/logs/make.log; exit 4; }
  ls -la $B/bin/
  nvidia-smi --query-gpu=name,memory.total --format=csv
"
