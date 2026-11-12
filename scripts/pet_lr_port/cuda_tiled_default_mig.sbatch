#!/bin/bash -l
#SBATCH --job-name=petlr-tileddef
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: the precise MUL_MAT's DEFAULT kernel is now tiled (user decision 2026-11-11; MLIPCPP_PRECISE_MATMUL_KERNEL
# unset = tiled, =naive the earlier one-thread-per-output kernel). Built in a THIRD tree (src_td + deps/ggml_td +
# build_td): src_kv / build_kv (unset = naive) and the main tree stay as the queued full-H100 jobs expect them.
# Op test per kernel (unset / naive / tiled / chunk / dot2); speed (readout-pol bulk_60 / mol_chunked, --repeat 5)
# unset / naive / tiled / off (timing is the discriminator: tiled and naive give identical E / q by construction);
# the charge families (readout-pol / control / readout x bulk / slab 60) unset / tiled / naive -> out/pbc/td_mig_<mode>
# (compare with the old explicit-tiled out/pbc/kv_mig_tiled), and the opt-out (MLIPCPP_PRECISE_MATMUL_K=0) on one
# table vs kv_mig_off; test_pet_lr_port and test_pet_lr_interp at the default. Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_SQRT MLIPCPP_ENERGY_DOUBLE MLIPCPP_PRECISE_MATMUL_KERNEL
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
C=/scratch/boittier/pet_lr_port/cuda; W=$C; F=/scratch/boittier/pet_lr_port/frames; D=$C/deps; BD=$C/build_td; B=$BD/bin
if [ ! -f $BD/CMakeCache.txt ]; then
  cmake -S $C/src_td -B $BD -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DMLIPCPP_BUILD_TESTS=ON -DMLIPCPP_BUILD_TRAINING=ON -DCPM_ggml_SOURCE=$D/ggml_td -DCPM_fmt_SOURCE=$D/fmt \
    -DCPM_gpulite_SOURCE=$D/gpulite -DCPM_nlohmann_json_SOURCE=$D/nlohmann_json -DCPM_vesin_SOURCE=$D/vesin \
    -DCPM_Catch2_SOURCE=$D/Catch2 > $C/cmake_td.log 2>&1; echo "cmake rc=$?"
fi
cmake --build $BD -j 5 > $C/build_td.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $C/build_td.log
md5sum $D/ggml_td/src/ggml-cuda/ggml-cuda.cu
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
for kv in default naive tiled chunk dot2; do
  if [ $kv = default ]; then unset MLIPCPP_PRECISE_MATMUL_KERNEL; else export MLIPCPP_PRECISE_MATMUL_KERNEL=$kv; fi
  echo "== op test $kv"; $BD/tests/test_pet_lr_interp "*MUL_MAT*" -s 2>&1 | grep -A2 "flagged =" | head -3
done
unset MLIPCPP_PRECISE_MATMUL_KERNEL
S=$C/out/speed_td_mig; mkdir -p $S
for mode in off default naive tiled; do
  unset MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_MATMUL_KERNEL
  [ $mode = off ] && export MLIPCPP_PRECISE_MATMUL_K=0
  case $mode in naive|tiled) export MLIPCPP_PRECISE_MATMUL_KERNEL=$mode;; esac
  for fam in bulk_60 mol_chunked; do
    fr=$F/set_$fam.extxyz; g=$W/gguf/readout-pol_lr_nobase.gguf; [ $fam = bulk_60 ] && g=$W/gguf_pbc/readout-pol_lr_nobase.gguf
    $B/mlipcpp-eval $g $fr $S/${fam}_$mode.json --backend cuda --repeat 5 > /dev/null 2>&1; echo "speed $mode $fam rc=$?"
  done
  O=$C/out/pbc/td_mig_$mode; mkdir -p $O
  if [ $mode = off ]; then
    $B/mlipcpp-eval $W/gguf_pbc/readout-pol_lr_nobase.gguf $F/set_bulk_60.extxyz $O/readout-pol_lr_nobase_bulk_cuda.json \
      --backend cuda > $O/readout-pol_bulk.log 2>&1
  else
    for g in readout-pol control readout; do for s in bulk slab; do
      $B/mlipcpp-eval $W/gguf_pbc/${g}_lr_nobase.gguf $F/set_${s}_60.extxyz $O/${g}_lr_nobase_${s}_cuda.json --backend cuda > $O/${g}_${s}.log 2>&1
    done; done
  fi
  echo "tables $mode done"
done
unset MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_MATMUL_KERNEL
echo "== test_pet_lr_port default"; $BD/tests/test_pet_lr_port 2>&1 | grep -E -A8 "FAILED" | head -24
$BD/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
echo "== test_pet_lr_interp default"; $BD/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
