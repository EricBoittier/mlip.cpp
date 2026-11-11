#!/bin/bash -l
#SBATCH --job-name=petlr-followup
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, full H100, after the default-config job (4894059):
# (a) SAME-BINARY control: the main tree's build_lr (as 4894059 built it; cuda/src untouched since) with the explicit
#     MLIPCPP_PRECISE_MATMUL_K=1036 on the charge families (readout-pol / control / readout x bulk / slab 60) ->
#     out/pbc/mm_explicit_h100: on = explicit K bit for bit (E / q), F within the run-to-run spread;
# (b) the precise MUL_MAT kernel variants (separate tree src_kv / deps/ggml_kv / build_kv;
#     MLIPCPP_PRECISE_MATMUL_KERNEL = naive (default) | tiled | chunk | dot2, and the flag off): op test, speed
#     (readout-pol bulk_60 / mol_chunked, --repeat 5), charge families -> out/pbc/kv_h100_<mode>;
# (c) test_pet_lr_port (build_kv: the slab virial FD with two Richardson step pairs) default and MLIPCPP_PRECISE_SQRT=1,
#     and the [.slab_fd_scan] step scan of the conserving residual preset, flag off / on;
# (d) induced dipoles on CUDA if exported (gguf_extra/cc_ind_*.gguf) -> out/ind_cuda; (e) periodic E . Z0 on CUDA.
# Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_SQRT MLIPCPP_ENERGY_DOUBLE MLIPCPP_PRECISE_MATMUL_KERNEL
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
C=/scratch/boittier/pet_lr_port/cuda; W=$C; F=/scratch/boittier/pet_lr_port/frames; BD=$C/build_kv; B=$BD/bin
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
# (a)
O=$C/out/pbc/mm_explicit_h100; mkdir -p $O
for g in readout-pol control readout; do for s in bulk slab; do
  MLIPCPP_PRECISE_MATMUL_K=1036 $C/build_lr/bin/mlipcpp-eval $W/gguf_pbc/${g}_lr_nobase.gguf $F/set_${s}_60.extxyz \
    $O/${g}_lr_nobase_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1; echo "explicit $g $s rc=$?"
done; done
# (b)
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
# (c)
echo "== test_pet_lr_port default"; $BD/tests/test_pet_lr_port 2>&1 | grep -E -A8 "FAILED" | head -24
$BD/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
echo "== test_pet_lr_port PRECISE_SQRT"; MLIPCPP_PRECISE_SQRT=1 $BD/tests/test_pet_lr_port 2>&1 | grep -E -A8 "FAILED" | head -24
MLIPCPP_PRECISE_SQRT=1 $BD/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
for sq in 0 1; do
  echo "== slab FD scan, PRECISE_SQRT=$sq"
  MLIPCPP_PRECISE_SQRT=$sq $BD/tests/test_pet_lr_port "[.slab_fd_scan]" 2>&1 | grep -E "slab FD|\(i |h [0-9]"
done
# (d)
O=$C/out/ind_cuda; mkdir -p $O
for p in cc_ind_field cc_ind_full; do
  g=$W/gguf_extra/$p.gguf; [ -f $g ] || continue
  for s in mol_z18_field mol_z18_field0; do
    $B/mlipcpp-eval $g $F/set_$s.extxyz $O/${p}_${s}_cuda.json --backend cuda > $O/${p}_${s}.log 2>&1; echo "$p $s rc=$?"
  done
  MLIPCPP_FIELD_CONSERVATIVE_ONLY=1 $B/mlipcpp-eval $g $F/set_bulk_f32_field.extxyz $O/${p}_bulk_f32_field_cuda.json --backend cuda > $O/${p}_bulk.log 2>&1; echo "$p bulk rc=$?"
done
# (e) the periodic field force E . Z0 (his unfolded levers) on CUDA: full forces on bulk_f32_field
for p in cc_fermi_field cc_residual_field; do
  $B/mlipcpp-eval $W/gguf_extra/${p}_mc.gguf $F/set_bulk_f32_field.extxyz $O/${p}_bulk_f32_field_full_cuda.json --backend cuda > $O/${p}_bulk_full.log 2>&1; echo "$p bulk full rc=$?"
done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
