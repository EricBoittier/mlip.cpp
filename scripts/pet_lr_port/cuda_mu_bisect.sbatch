#!/bin/bash -l
#SBATCH --job-name=petlr-mu-bisect
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# seeded lr_both ion on CUDA: mu with the precise-math flags per op kind skipped (MLIPCPP_PRECISE_MATH_SKIP). Eval only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; F=/scratch/boittier/pet_lr_port/frames; O=$W/out/mu_bisect; mkdir -p $O
cmake --build $W/build_lr -j 16 --target mlipcpp-eval > $W/build_mu.log 2>&1; echo "build rc=$?"
for k in none sigmoid div sigmoid,div; do
  MLIPCPP_PRECISE_MATH_SKIP=$k $W/build_lr/bin/mlipcpp-eval $W/gguf/seeded_lr_both.gguf $F/set_ion_40.extxyz $O/skip_${k/,/_}.json --backend cuda > /dev/null 2>&1; echo "skip $k rc=$?"
done
