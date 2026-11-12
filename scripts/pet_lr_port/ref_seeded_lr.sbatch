#!/bin/bash
#SBATCH --job-name=petlr-ref-seedlr
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, phase 2: seeded PETLRs with the LR branch on (make_seeded.py lr_* presets: species / both references,
# softplus conductance, unshared passes, Gaussian charges, T = 0.02 fermi), his full PETLR (ref_petlr.py) in fp64 and fp32
# on CPU: the oracle for the IrisLR rows no omol-4m checkpoint exercises, values and forces. M. Langer's env, read-only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
PY="/scratch/boittier/ipsum/env/bin/python -I"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
mkdir -p ref/seeded params
for p in ${PRESETS:-lr_species lr_both lr_fermi_cold}; do
  $PY scripts/make_seeded.py seeded/$p --preset $p --seed 3 \
    --config /work/cosmo/langer/kuma/argos/checkpoints/omol-4m/finetune/readout-pol/RMSE_F/config.yaml || continue
  $PY scripts/dump_params.py seeded/$p params/seeded_$p.npz
  for s in ${FAMS:-ion cluster}; do
    $PY scripts/ref_petlr.py seeded/$p frames/set_$s.extxyz ref/seeded/${p}_${s}_cpu64.npz --x64 --max ${MAXF:-40} > ref/seeded/${p}_${s}_cpu64.log 2>&1
    $PY scripts/ref_petlr.py seeded/$p frames/set_$s.extxyz ref/seeded/${p}_${s}_cpu32.npz --max ${MAXF:-40} > ref/seeded/${p}_${s}_cpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ref/seeded/${p}_${s}_cpu64.log | cut -c1-160)"
  done
done
