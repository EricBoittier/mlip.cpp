#!/bin/bash -l
#SBATCH --job-name=petlr-les-nodegrad
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: the les ladder presets' CUDA E excess (P 0.70-0.72 vs his gpu32 on mol_z18). (1) per-node forward
# values + dE/d(node) (MLIPCPP_NODE_GRAD_DUMP) on 16 mol_z18 frames (the 6 worst + 10 others, nodegrad/les/mol_w.extxyz)
# for compare_node_grads_iris.py / les_heads.py on cosmopc7; (2) the full mol_z18 / bulk_f32 E tables with the opt-in
# flags COMBINED (PRECISE_SUM + PRECISE_MATMUL; singles were job 4893146). Diagnostic; FP32, no TF32.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32 MLIPCPP_PRECISE_SIGMOID MLIPCPP_PRECISE_MATH_SKIP MLIPCPP_PRECISE_SUM \
  MLIPCPP_PRECISE_MATMUL MLIPCPP_PRECISE_MATMUL_K MLIPCPP_ENERGY_DOUBLE
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
W=/scratch/boittier/pet_lr_port/cuda; F=/scratch/boittier/pet_lr_port/frames; B=$W/build_lr/bin; N=$W/nodegrad/les
O=$W/out/les_cuda_flags; mkdir -p $O $N
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
for p in les_ladder les_ladder_c0; do
  rm -f $N/${p}_cuda.bin
  MLIPCPP_NODE_GRAD_DUMP=$N/${p}_cuda.bin $B/mlipcpp-eval $W/gguf_les/$p.gguf $N/mol_w.extxyz $N/${p}_cuda.json \
    --backend cuda > $N/${p}_cuda.log 2>&1; echo "$p dump rc=$? $(stat -c %s $N/${p}_cuda.bin)"
  for s in mol_z18 bulk_f32; do
    MLIPCPP_PRECISE_SUM=1 MLIPCPP_PRECISE_MATMUL=1 $B/mlipcpp-eval $W/gguf_les/$p.gguf $F/set_$s.extxyz \
      $O/${p}_${s}_cuda_summatmul.json --backend cuda > /dev/null 2>&1; echo "$p $s sum+matmul rc=$?"
  done
done
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
