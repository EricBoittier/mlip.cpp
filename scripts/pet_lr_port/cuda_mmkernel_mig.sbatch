#!/bin/bash -l
#SBATCH --job-name=petlr-mmkernel
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: kernel variants of the precise MUL_MAT (MLIPCPP_PRECISE_MATMUL_KERNEL = tiled | chunk | dot2; unset =
# the default naive kernel), built in a SEPARATE tree (src_kv + deps/ggml_kv + build_kv: the main cuda/ tree is left
# alone for the queued full-H100 job). Op test per kernel, speed (readout-pol bulk_60 / mol_chunked, --repeat 5) per
# kernel and with the flag off, the charge families (readout-pol / control / readout x bulk / slab 60) per kernel;
# and test_pet_lr_port with MLIPCPP_PRECISE_SQRT=1 (does the slab virial FD failure reproduce on a MIG slice).
# KV_PART=build|all. Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_SQRT MLIPCPP_ENERGY_DOUBLE MLIPCPP_PRECISE_MATMUL_KERNEL
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
C=/scratch/boittier/pet_lr_port/cuda; W=$C; F=/scratch/boittier/pet_lr_port/frames; D=$C/deps; BD=$C/build_kv; B=$BD/bin
if [ ! -f $BD/CMakeCache.txt ]; then
  cmake -S $C/src_kv -B $BD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DMLIPCPP_BUILD_TESTS=ON -DMLIPCPP_BUILD_TRAINING=ON -DCPM_ggml_SOURCE=$D/ggml_kv -DCPM_fmt_SOURCE=$D/fmt \
    -DCPM_gpulite_SOURCE=$D/gpulite -DCPM_nlohmann_json_SOURCE=$D/nlohmann_json -DCPM_vesin_SOURCE=$D/vesin \
    -DCPM_Catch2_SOURCE=$D/Catch2 > $C/cmake_kv.log 2>&1; echo "cmake rc=$?"
fi
cmake --build $BD -j 5 > $C/build_kv.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $C/build_kv.log
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
[ "${KV_PART:-all}" = build ] && exit 0
for kv in naive tiled chunk dot2; do
  if [ $kv = naive ]; then unset MLIPCPP_PRECISE_MATMUL_KERNEL; else export MLIPCPP_PRECISE_MATMUL_KERNEL=$kv; fi
  echo "== op test $kv"; $BD/tests/test_pet_lr_interp "*MUL_MAT*" -s 2>&1 | grep -A4 "flagged =" | head -6
done
unset MLIPCPP_PRECISE_MATMUL_KERNEL
S=$C/out/speed_kv_mig; mkdir -p $S
for mode in off naive tiled chunk dot2; do
  unset MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_MATMUL_KERNEL
  [ $mode = off ] && export MLIPCPP_PRECISE_MATMUL_K=0
  case $mode in tiled|chunk|dot2) export MLIPCPP_PRECISE_MATMUL_KERNEL=$mode;; esac
  for fam in bulk_60 mol_chunked; do
    fr=$F/set_$fam.extxyz; g=$W/gguf/readout-pol_lr_nobase.gguf; [ $fam = bulk_60 ] && g=$W/gguf_pbc/readout-pol_lr_nobase.gguf
    $B/mlipcpp-eval $g $fr $S/${fam}_$mode.json --backend cuda --repeat 5 > /dev/null 2>&1; echo "speed $mode $fam rc=$?"
  done
  O=$C/out/pbc/kv_mig_$mode; mkdir -p $O
  for g in readout-pol control readout; do for s in bulk slab; do
    $B/mlipcpp-eval $W/gguf_pbc/${g}_lr_nobase.gguf $F/set_${s}_60.extxyz $O/${g}_lr_nobase_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
  done; done
  echo "tables $mode done"
done
unset MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_MATMUL_KERNEL
echo "== test_pet_lr_port default"; $BD/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
echo "== test_pet_lr_port PRECISE_SQRT"; MLIPCPP_PRECISE_SQRT=1 $BD/tests/test_pet_lr_port 2>&1 | grep -E -B4 -A12 "FAILED" | head -60
MLIPCPP_PRECISE_SQRT=1 $BD/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
