#!/bin/bash -l
#SBATCH --job-name=petlr-qeq-cuda
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: rebuild the CUDA tree at the head (QEq three-stage evaluation; the precise MUL_MAT op test), the GPU
# tests, then tiny_qeq on CUDA (out/qeq_cuda/: mol_z18, bulk_f32; and maxiter 3, a binding truncation). Evaluation
# only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; F=/scratch/boittier/pet_lr_port/frames; B=$W/build_lr/bin
cmake --build $W/build_lr -j 16 > $W/build_qeq.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_qeq.log
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E -A8 "FAILED" | head -30
$W/build_lr/tests/test_pet_lr_interp "*MUL_MAT*" -s 2>&1 | grep -E "flagged =|passed|failed" | head -4
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A12 "FAILED" | head -40
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
O=$W/out/qeq_cuda; mkdir -p $O
for s in mol_z18 bulk_f32; do
  MLIPCPP_QEQ_LOG=1 $B/mlipcpp-eval $W/gguf_qeq/tiny_qeq.gguf $F/set_$s.extxyz $O/tiny_qeq_${s}_cuda.json --backend cuda \
    > $O/tiny_qeq_$s.log 2>&1; echo "tiny_qeq $s rc=$?"
done
$B/mlipcpp-eval $W/gguf_qeq/tiny_qeq.gguf $F/set_mol_z18.extxyz $O/tiny_qeq_mol_z18_cuda_it3.json --backend cuda \
  --iris-qeq-maxiter 3 > $O/tiny_qeq_mol_z18_it3.log 2>&1; echo "tiny_qeq mol_z18 maxiter 3 rc=$?"
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
