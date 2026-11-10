#!/bin/bash
#SBATCH --job-name=petlr-ref-trunk
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, phase 1 oracle: the trunk of CKPT (env CK, a path under his checkpoints) on the five frame
# families, JAX CPU fp32 + CPU fp64 (in parallel) and GPU fp32 (his noise floor). M. Langer's env, read-only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
P="/scratch/boittier/ipsum/env/bin/python -I scripts/ref_trunk.py"
C=/work/cosmo/langer/kuma/argos/checkpoints/omol-4m/${CK:-finetune/readout-pol}/RMSE_F
T=$(echo ${CK:-finetune/readout-pol} | tr / _)
mkdir -p ref/$T
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
nvidia-smi -L
for s in mol ion cluster slab bulk; do
  ( JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=true $P $C frames/set_$s.extxyz ref/$T/${s}_cpu32.npz > ref/$T/${s}_cpu32.log 2>&1;
    JAX_PLATFORMS=cpu $P $C frames/set_$s.extxyz ref/$T/${s}_cpu64.npz --x64 > ref/$T/${s}_cpu64.log 2>&1 ) &
done
for s in mol ion cluster slab bulk; do
  $P $C frames/set_$s.extxyz ref/$T/${s}_gpu32.npz > ref/$T/${s}_gpu32.log 2>&1; echo "gpu $s rc=$?"
done
wait
for f in ref/$T/*.log; do echo "$f: $(tail -1 $f | cut -c1-200)"; done
