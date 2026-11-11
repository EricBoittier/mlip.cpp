#!/bin/bash -l
#SBATCH --job-name=petlr-sqrt
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: the flagged SQRT (__fsqrt_rn, ggml-cuda-precise-sqrt.patch; flagged by default in iris graphs) vs the
# fast-math sqrt (MLIPCPP_PRECISE_MATH_SKIP=sqrt): rebuild at the head, GPU tests (the SQRT op test printed), then the
# les presets (mol_z18, bulk_f32), seeded LR presets (ion, cluster) and readout-pol / control (ion, cluster) both ways,
# plus the ladder node dump with the flag. Evaluation only; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_ENERGY_DOUBLE
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; F=/scratch/boittier/pet_lr_port/frames; B=$W/build_lr/bin
cmake --build $W/build_lr -j 16 > $W/build_sqrt.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_sqrt.log
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
$W/build_lr/tests/test_pet_lr_interp "*SQRT*" -s 2>&1 | grep -E "flagged =|passed|failed" | head -4
$W/build_lr/tests/test_pet_lr_interp 2>&1 | grep -E "passed|failed"
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E -A12 "FAILED" | head -40
$W/build_lr/tests/test_pet_lr_port 2>&1 | grep -E "passed|failed"
O=$W/out/sqrt; mkdir -p $O
ev() { # gguf frames tag
  $B/mlipcpp-eval $1 $2 $O/${3}_on.json --backend cuda > /dev/null 2>&1; a=$?
  MLIPCPP_PRECISE_MATH_SKIP=sqrt $B/mlipcpp-eval $1 $2 $O/${3}_off.json --backend cuda > /dev/null 2>&1; echo "$3 rc=$a/$?"
}
for p in les_local2 les_local3 les_ladder les_ladder_c0; do for s in mol_z18 bulk_f32; do
  ev $W/gguf_les/$p.gguf $F/set_$s.extxyz ${p}_$s; done; done
for g in seeded_lr_species seeded_lr_both seeded_lr_fermi_cold; do for s in ion cluster; do
  ev $W/gguf/$g.gguf $F/set_$s.extxyz ${g}_$s; done; done
for g in readout-pol control; do for s in ion cluster; do
  ev $W/gguf/${g}_lr_nobase.gguf $F/set_$s.extxyz ${g}_$s; done; done
N=$W/nodegrad/les; rm -f $N/les_ladder_cuda_sqrt.bin
MLIPCPP_NODE_GRAD_DUMP=$N/les_ladder_cuda_sqrt.bin $B/mlipcpp-eval $W/gguf_les/les_ladder.gguf $N/mol_w.extxyz $N/les_ladder_cuda_sqrt.json \
  --backend cuda > /dev/null 2>&1; echo "dump rc=$?"
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
