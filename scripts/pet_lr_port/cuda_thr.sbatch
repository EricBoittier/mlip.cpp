#!/bin/bash -l
#SBATCH --job-name=petlr-thr
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: rebuild the CUDA tree at the head (threshold reference, extra output columns, field input, head
# charges), the GPU tests (port test twice at the defaults, once with MLIPCPP_PRECISE_SQRT=1: the cc_residual slab
# virial FD check failed once in job 4893238 with SQRT flagged), then the threshold presets on CUDA
# (out/thr_cuda/: mol_z18, cluster, bulk_f32) and, if present, the GGUFs in gguf_extra/ on their frame lists.
# Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_PRECISE_SQRT MLIPCPP_ENERGY_DOUBLE
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; F=/scratch/boittier/pet_lr_port/frames; B=$W/build_lr/bin
cmake --build $W/build_lr -j 16 > $W/build_thr.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_thr.log
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
for k in 1 2; do $W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A8 "FAILED" | head -24; $W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"; done
MLIPCPP_PRECISE_SQRT=1 $W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A8 "FAILED" | head -24
MLIPCPP_PRECISE_SQRT=1 $W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
O=$W/out/thr_cuda; mkdir -p $O
for p in thr thr_cold_lrt; do for s in mol_z18 cluster bulk_f32; do
  $B/mlipcpp-eval $W/gguf_thr/$p.gguf $F/set_$s.extxyz $O/${p}_${s}_cuda.json --backend cuda > $O/${p}_$s.log 2>&1
  echo "$p $s rc=$?"
done; done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
