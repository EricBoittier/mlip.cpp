#!/bin/bash -l
#SBATCH --job-name=petlr-cuda-test
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: rebuild the CUDA tree (build_lr, every target) and run the PET-LR tests on the GPU. Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda
cmake --build $W/build_lr -j 16 > $W/build_test.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_test.log
$W/build_lr/tests/test_pet_lr_interp 2>&1 | tail -1
$W/build_lr/tests/test_pet_lr_port -s 2>&1 | grep -E -B2 -A10 "FAILED" | head -80
$W/build_lr/tests/test_pet_lr_port 2>&1 | tail -2
