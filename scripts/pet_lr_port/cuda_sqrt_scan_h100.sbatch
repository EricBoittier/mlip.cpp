#!/bin/bash -l
#SBATCH --job-name=petlr-sqrtscan
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: the slab virial FD step scan ([.slab_fd_scan], conserving residual preset) on a full H100 with
# MLIPCPP_PRECISE_SQRT off / on (build_kv): does W shift with the flag, or only the small-h FD (roundoff)?
# Output to out/sqrt_scan_h100/scan_sqrt<0|1>.txt. Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_PRECISE_SQRT MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_MATMUL_KERNEL
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
C=/scratch/boittier/pet_lr_port/cuda; O=$C/out/sqrt_scan_h100; mkdir -p $O
nvidia-smi --query-gpu=name --format=csv
for sq in 0 1; do
  MLIPCPP_PRECISE_SQRT=$sq $C/build_kv/tests/test_pet_lr_port "[.slab_fd_scan]" > $O/scan_sqrt$sq.txt 2>&1; echo "scan $sq rc=$?"
done
MLIPCPP_PRECISE_SQRT=1 $C/build_kv/tests/test_pet_lr_port "pet_lr_port: slab virial chain rule (normal / basis): W = dE/d eps at frozen Ewald parameters (FD)" -s > $O/slabtest_sqrt1.txt 2>&1; echo "slab test sqrt1 rc=$?"
