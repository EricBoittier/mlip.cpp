#!/bin/bash -l
#SBATCH --job-name=petlr-les-flags
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: the les ladder presets' CUDA E excess (P 0.70-0.72 vs his gpu32 on mol_z18): the same build with the
# opt-in precise flags one at a time (double-accumulating SUM / SUM_ROWS; double-accumulating MUL_MAT everywhere; the
# per-atom energy sum in double), to name the op or rule it out. Diagnostic; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_ENERGY_DOUBLE
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
W=/scratch/boittier/pet_lr_port/cuda; F=/scratch/boittier/pet_lr_port/frames; B=$W/build_lr/bin
O=$W/out/les_cuda_flags; mkdir -p $O
for p in les_ladder les_ladder_c0; do
  for fl in sum matmul edouble; do
    case $fl in sum) v=MLIPCPP_PRECISE_SUM;; matmul) v=MLIPCPP_PRECISE_MATMUL;; edouble) v=MLIPCPP_ENERGY_DOUBLE;; esac
    env $v=1 $B/mlipcpp-eval $W/gguf_les/$p.gguf $F/set_mol_z18.extxyz $O/${p}_mol_z18_cuda_$fl.json --backend cuda \
      > /dev/null 2>&1; echo "$p $fl rc=$?"
  done
done
