#!/bin/bash
#SBATCH --job-name=petlr-ref-full
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, phase 2 oracle: the full PETLR (ref_petlr.py) of CKPT (env CK, a path under his checkpoints) on the five frame
# families, JAX CPU fp32 + CPU fp64 (in parallel) and GPU fp32 (his noise floor). M. Langer's env, read-only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
P="/scratch/boittier/ipsum/env/bin/python -I scripts/ref_petlr.py"
C=/work/cosmo/langer/kuma/argos/checkpoints/omol-4m/${CK:-finetune/readout-pol}/RMSE_F
T=$(echo ${CK:-finetune/readout-pol} | tr / _)
mkdir -p ref/$T
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
nvidia-smi -L
for s in ${FAMS:-mol ion cluster slab bulk}; do
  JAX_PLATFORMS=cpu $P $C frames/set_$s.extxyz ref/$T/full_${s}_cpu64.npz --x64 > ref/$T/full_${s}_cpu64.log 2>&1
  JAX_PLATFORMS=cpu $P $C frames/set_$s.extxyz ref/$T/full_${s}_cpu32.npz > ref/$T/full_${s}_cpu32.log 2>&1
  echo "$s done"
done
for f in ref/$T/full_*.log; do echo "$f: $(tail -n 1 $f | cut -c1-120)"; done
