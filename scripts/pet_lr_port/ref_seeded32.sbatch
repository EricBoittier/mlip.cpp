#!/bin/bash
#SBATCH --job-name=petlr-ref-seed32
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: his fp32 runs (CPU default, CPU single-thread, GPU) of the seeded LR presets (made by ref_seeded_lr.sbatch).
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
P="/scratch/boittier/ipsum/env/bin/python -I scripts/ref_petlr.py"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
for p in ${PRESETS:-lr_species lr_both lr_fermi_cold}; do
  for s in ${FAMS:-ion cluster}; do
    JAX_PLATFORMS=cpu $P seeded/$p frames/set_$s.extxyz ref/seeded/${p}_${s}_cpu32.npz --max 40 > ref/seeded/${p}_${s}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $P seeded/$p frames/set_$s.extxyz ref/seeded/${p}_${s}_cpu32_v2.npz --max 40 > ref/seeded/${p}_${s}_cpu32_v2.log 2>&1
    $P seeded/$p frames/set_$s.extxyz ref/seeded/${p}_${s}_gpu32.npz --max 40 > ref/seeded/${p}_${s}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ref/seeded/${p}_${s}_cpu32.log | cut -c1-80)"
  done
done
