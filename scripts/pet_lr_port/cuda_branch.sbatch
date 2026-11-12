#!/bin/bash -l
#SBATCH --job-name=petlr-branch-cuda
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: rebuild the CUDA tree at the head (species_weighted / species_fill / open long-range transfer), the GPU
# tests, then his branch presets on CUDA (out/branch_cuda/). Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; F=/scratch/boittier/pet_lr_port/frames; B=$W/build_lr/bin
cmake --build $W/build_lr -j 16 > $W/build_branch.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_branch.log
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A12 "FAILED" | head -40
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
O=$W/out/branch_cuda; mkdir -p $O
for p in lrt_weighted lrt_fill lrt_fill_cold lrt_species_fill lrt_species_weighted; do for s in cluster mol_z18; do
  fr=$F/set_${s}.extxyz; [ $s = cluster ] && fr=$F/set_cluster_40.extxyz
  $B/mlipcpp-eval $W/gguf_branch/$p.gguf $fr $O/${p}_${s}_cuda.json --backend cuda > $O/${p}_${s}.log 2>&1; echo "$p $s rc=$?"
done; done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
