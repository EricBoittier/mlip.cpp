#!/bin/bash -l
#SBATCH --job-name=petlr-kvrehearse
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: rehearse the h100 follow-up on a MIG slice: build_kv at the synced worktree (d5c78d5+), GPU tests
# (default, MLIPCPP_PRECISE_SQRT=1), the slab FD scan flag off / on, induced dipoles (cc_ind_*) and the periodic field
# force on CUDA. Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_SQRT MLIPCPP_ENERGY_DOUBLE MLIPCPP_PRECISE_MATMUL_KERNEL
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
C=/scratch/boittier/pet_lr_port/cuda; W=$C; F=/scratch/boittier/pet_lr_port/frames; BD=$C/build_kv; B=$BD/bin
cmake --build $BD -j 5 > $C/build_kv_rehearse.log 2>&1; echo "build_kv rc=$?"; grep -m10 -E " error|error:" $C/build_kv_rehearse.log
echo "== test_pet_lr_interp"; $BD/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
echo "== test_pet_lr_port default"; $BD/tests/test_pet_lr_port 2>&1 | grep -E -A10 "FAILED" | head -30
$BD/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
echo "== test_pet_lr_port PRECISE_SQRT"; MLIPCPP_PRECISE_SQRT=1 $BD/tests/test_pet_lr_port 2>&1 | grep -E -A10 "FAILED" | head -30
MLIPCPP_PRECISE_SQRT=1 $BD/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
for sq in 0 1; do
  echo "== slab FD scan, PRECISE_SQRT=$sq"
  MLIPCPP_PRECISE_SQRT=$sq $BD/tests/test_pet_lr_port "[.slab_fd_scan]" 2>&1 | grep -E "slab FD|\(i |h [0-9]"
done
O=$C/out/ind_cuda_mig; mkdir -p $O
for p in cc_ind_field cc_ind_full; do
  for s in mol_z18_field mol_z18_field0; do
    $B/mlipcpp-eval $W/gguf_extra/$p.gguf $F/set_$s.extxyz $O/${p}_${s}_cuda.json --backend cuda > $O/${p}_${s}.log 2>&1; echo "$p $s rc=$?"
  done
  $B/mlipcpp-eval $W/gguf_extra/$p.gguf $F/set_bulk_f32_field.extxyz $O/${p}_bulk_f32_field_cuda.json --backend cuda > $O/${p}_bulk.log 2>&1; echo "$p bulk rc=$?"
done
for p in cc_fermi_field cc_residual_field; do
  $B/mlipcpp-eval $W/gguf_extra/${p}_mc.gguf $F/set_bulk_f32_field.extxyz $O/${p}_bulk_f32_field_full_cuda.json --backend cuda > $O/${p}_bulk_full.log 2>&1; echo "$p bulk full rc=$?"
done
