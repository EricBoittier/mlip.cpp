#!/bin/bash -l
#SBATCH --job-name=petlr-kvh100
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, full H100. Replaces cuda_followup_h100.sbatch (4894719 / 4894720: an unclosed `if` -> bash syntax error
# at EOF, nothing ran). Trees: build_lr = main tree (old default: precise MUL_MAT naive), build_kv = kernel-variant tree
# (MLIPCPP_PRECISE_MATMUL_KERNEL unset = naive there), build_td = the committed head 6e2ead3 (unset = tiled).
# (a) explicit MLIPCPP_PRECISE_MATMUL_K=1036 control on build_lr -> out/pbc/mm_explicit_h100, and on build_td
#     -> out/pbc/td_h100_explicit (same-binary control for the new tiled default; that default is in cuda_sqrtdec_h100);
# (b) kernel variants on build_kv: op test, speed, charge tables -> out/speed_kv_h100, out/pbc/kv_h100_<mode>.
# Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_SQRT MLIPCPP_ENERGY_DOUBLE MLIPCPP_PRECISE_MATMUL_KERNEL
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
C=/scratch/boittier/pet_lr_port/cuda; W=$C; F=/scratch/boittier/pet_lr_port/frames
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
# (a)
O=$C/out/pbc/mm_explicit_h100; mkdir -p $O
for g in readout-pol control readout; do for s in bulk slab; do
  MLIPCPP_PRECISE_MATMUL_K=1036 $C/build_lr/bin/mlipcpp-eval $W/gguf_pbc/${g}_lr_nobase.gguf $F/set_${s}_60.extxyz \
    $O/${g}_lr_nobase_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1; echo "explicit $g $s rc=$?"
done; done
# (a') same-binary control for the new default: build_td with the explicit K -> out/pbc/td_h100_explicit
BT=$C/build_td; cmake --build $BT -j 16 > $C/build_td_h100a.log 2>&1; echo "build_td rc=$?"
O=$C/out/pbc/td_h100_explicit; mkdir -p $O
for g in readout-pol control readout; do for s in bulk slab; do
  MLIPCPP_PRECISE_MATMUL_K=1036 $BT/bin/mlipcpp-eval $W/gguf_pbc/${g}_lr_nobase.gguf $F/set_${s}_60.extxyz \
    $O/${g}_lr_nobase_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1; echo "td explicit $g $s rc=$?"
done; done
# (b)
BD=$C/build_kv; B=$BD/bin
cmake --build $BD -j 16 > $C/build_kv_h100.log 2>&1; echo "build_kv rc=$?"; grep -m10 -E " error|error:" $C/build_kv_h100.log
for kv in naive tiled chunk dot2; do
  if [ $kv = naive ]; then unset MLIPCPP_PRECISE_MATMUL_KERNEL; else export MLIPCPP_PRECISE_MATMUL_KERNEL=$kv; fi
  echo "== op test $kv"; $BD/tests/test_pet_lr_interp "*MUL_MAT*" -s 2>&1 | grep -A2 "flagged =" | head -3
done
S=$C/out/speed_kv_h100; mkdir -p $S
for mode in off naive tiled chunk dot2; do
  unset MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_MATMUL_KERNEL
  [ $mode = off ] && export MLIPCPP_PRECISE_MATMUL_K=0
  case $mode in tiled|chunk|dot2) export MLIPCPP_PRECISE_MATMUL_KERNEL=$mode;; esac
  for fam in bulk_60 mol_chunked; do
    fr=$F/set_$fam.extxyz; g=$W/gguf/readout-pol_lr_nobase.gguf; [ $fam = bulk_60 ] && g=$W/gguf_pbc/readout-pol_lr_nobase.gguf
    $B/mlipcpp-eval $g $fr $S/${fam}_$mode.json --backend cuda --repeat 5 > /dev/null 2>&1; echo "speed $mode $fam rc=$?"
  done
  O=$C/out/pbc/kv_h100_$mode; mkdir -p $O
  for g in readout-pol control readout; do for s in bulk slab; do
    $B/mlipcpp-eval $W/gguf_pbc/${g}_lr_nobase.gguf $F/set_${s}_60.extxyz $O/${g}_lr_nobase_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
  done; done
  echo "tables $mode done"
done
unset MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_MATMUL_KERNEL
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
echo "ALL DONE"
