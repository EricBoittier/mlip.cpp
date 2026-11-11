#!/bin/bash
#SBATCH --job-name=petlr-ref-branch-pbc
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: his branch oracle (ref_branch.py) on PERIODIC 3-D cells (float32-exact frames), with the stress.
# 1. validation of ref_branch.py's periodic path against the calculator oracle (ref_petlr.py), main env, seeded lr_both,
#    3 frames, both in fp64 on the same float32-exact frames;
# 2. the long-range-transfer presets on bulk water and NaCl in water: cpu64 x64 + cpu32 + cpu32_v2 + gpu32.
# Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
PYM=/scratch/boittier/ipsum/env/bin/python
PYB=/scratch/boittier/ipsum/envs/iris-long-range-transfer/bin/python
mkdir -p ref/branchval ref/seeded_branch
c="--cutoffs differentiable"
if [ -z "${SKIP_VAL:-}" ]; then
  o=ref/branchval/lr_both_bulkf32
  JAX_PLATFORMS=cpu $PYM -I scripts/ref_petlr.py seeded/lr_both frames/set_bulk_f32.extxyz ${o}_calc64.npz --x64 \
    --precision highest --max 3 > ${o}_calc64.log 2>&1
  JAX_PLATFORMS=cpu $PYM -I scripts/ref_branch.py seeded/lr_both frames/set_bulk_f32.extxyz ${o}_branch64.npz --x64 \
    --precision highest --max 3 --cutoffs hard > ${o}_branch64.log 2>&1
  echo "validation: $(tail -n 1 ${o}_calc64.log | cut -c1-80) | $(tail -n 1 ${o}_branch64.log | cut -c1-80)"
fi
for p in ${PRESETS:-lrt_species_fill lrt_species_weighted}; do
  for s in ${FAMS:-bulk_f32 bulknacl_f32}; do
    o=ref/seeded_branch/${p}_${s}; m="--max ${MAX:-12}"
    [ -f ${o}_cpu64.npz ] && continue
    JAX_PLATFORMS=cpu $PYB -I scripts/ref_branch.py seeded_branch/$p frames/set_$s.extxyz ${o}_cpu64.npz --x64 --precision highest $c $m > ${o}_cpu64.log 2>&1
    [ -n "${ONLY64:-}" ] && { echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-100)"; continue; }
    JAX_PLATFORMS=cpu $PYB -I scripts/ref_branch.py seeded_branch/$p frames/set_$s.extxyz ${o}_cpu32.npz $c $m > ${o}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $PYB -I scripts/ref_branch.py seeded_branch/$p frames/set_$s.extxyz ${o}_cpu32_v2.npz $c $m > ${o}_cpu32_v2.log 2>&1
    $PYB -I scripts/ref_branch.py seeded_branch/$p frames/set_$s.extxyz ${o}_gpu32.npz $c $m > ${o}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-100) | $(tail -n 1 ${o}_gpu32.log | cut -c1-60)"
  done
done
