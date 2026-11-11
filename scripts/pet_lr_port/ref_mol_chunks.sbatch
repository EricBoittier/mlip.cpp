#!/bin/bash
#SBATCH --job-name=petlr-ref-mol
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, phase 2: his full PETLR (ref_petlr.py) on the mol family in chunks of 40 frames, one process each (one
# process over all 240 frames exhausted the slice's memory in XLA compilations), then merged. M. Langer's env, read-only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
PY="/scratch/boittier/ipsum/env/bin/python -I"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
mkdir -p frames/mol_chunks
$PY -c "
from ase.io import read, write
a = read('frames/set_mol.extxyz', ':')
for i in range(0, len(a), 40): write(f'frames/mol_chunks/c{i // 40}.extxyz', a[i:i + 40])
"
for CK in ${CKS:-finetune/readout-pol finetune/control finetune/readout scaling/readout-1m scaling/readout-2m scaling/readout-4m}; do
  C=/work/cosmo/langer/kuma/argos/checkpoints/omol-4m/$CK/RMSE_F
  T=$(echo $CK | tr / _)
  mkdir -p ref/$T/mol_chunks
  for prec in 64 32; do
    X=""; [ $prec = 64 ] && X="--x64"
    parts=""
    for f in frames/mol_chunks/c*.extxyz; do
      c=$(basename $f .extxyz)
      $PY scripts/ref_petlr.py $C $f ref/$T/mol_chunks/${c}_cpu$prec.npz $X > ref/$T/mol_chunks/${c}_cpu$prec.log 2>&1 || echo "$CK $c $prec FAILED"
      parts="$parts ref/$T/mol_chunks/${c}_cpu$prec.npz"
    done
    $PY scripts/merge_npz.py ref/$T/full_mol_cpu$prec.npz $(for i in 0 1 2 3 4 5; do echo ref/$T/mol_chunks/c${i}_cpu$prec.npz; done) | cut -c1-200
  done
done
