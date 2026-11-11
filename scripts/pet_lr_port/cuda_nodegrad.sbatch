#!/bin/bash -l
#SBATCH --job-name=petlr-nodegrad
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: per-node forward values + dE/d(node) on CUDA (MLIPCPP_NODE_GRAD_DUMP, H100, FP32, no TF32) for the
# per-node fp64 oracle (compare_node_grads_iris.py, run on cosmopc7): readout-pol trunk ion, readout-1m LR
# cluster / bulk / slab, readout-pol LR bulk / slab, seeded lr_fermi_cold bulk / slab. Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE MLIPCPP_TF32
export GGML_CUDA_NO_TF32=1
module load gcc/13.2.0 cmake/3.27.9 && module load nvhpc/24.7-mpi && module load cuda/12.5.1 || exit 1
export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++)
W=/scratch/boittier/pet_lr_port/cuda; N=$W/nodegrad
cmake --build $W/build_lr -j 16 --target mlipcpp-eval > $W/build_nodegrad.log 2>&1; echo "build rc=$?"; grep -m10 -E " error|error:" $W/build_nodegrad.log
nvidia-smi --query-gpu=name,power.limit,clocks.max.sm --format=csv
run() { rm -f $N/$1_cuda.bin; MLIPCPP_NODE_GRAD_DUMP=$N/$1_cuda.bin $W/build_lr/bin/mlipcpp-eval $N/$2 $N/$3 $N/$1_cuda.json --backend cuda > $N/$1.log 2>&1; echo "$1 rc=$? $(ls -la $N/$1_cuda.bin | awk '{print $5}')"; }
run pol_trunk_ion readout-pol_trunk_v5.gguf ion_8.extxyz
run r1m_cluster readout-1m_lr_v5.gguf cluster_6.extxyz
run r1m_bulk readout-1m_lr_v5.gguf bulk_60_3.extxyz
run r1m_slab readout-1m_lr_v5.gguf slab_60_3.extxyz
run pol_bulk readout-pol_lr_v5.gguf bulk_60_3.extxyz
run pol_slab readout-pol_lr_v5.gguf slab_60_3.extxyz
run cold_bulk seeded_lr_fermi_cold_v5.gguf bulk_60_3.extxyz
run cold_slab seeded_lr_fermi_cold_v5.gguf slab_60_3.extxyz
nvidia-smi --query-gpu=name,power.draw,power.limit,memory.used --format=csv
