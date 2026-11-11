#!/bin/bash
#SBATCH --job-name=petlr-ref-ccfield
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, his branch feat/conserving-charges (a98c2f6) with his `field` input: the presets cc_fermi_field /
# cc_residual_field (make_seeded.py, his JAX init, that branch's env) through his batch builder (ref_branch_cc.py, which
# also records his field force E . Z0 on its own as field_forces, and dipole_flow [S, 3]): fp64 (cpu64) and three fp32
# runs on mol_z18_field (60 molecules, a uniform field of 0.1-0.3 V/A each) and bulk_f32_field (12 water cells).
# Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
PYB=/scratch/boittier/ipsum/envs/iris-conserving-charges/bin/python
P="$PYB -I scripts/ref_branch_cc.py"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
mkdir -p seeded_cc ref/seeded_cc params
for p in ${PRESETS:-cc_fermi_field cc_residual_field}; do
  if [ ! -d seeded_cc/$p ]; then
    JAX_PLATFORMS=cpu $PYB -I scripts/make_seeded.py seeded_cc/$p --preset $p | tail -n 1 | cut -c1-120
  fi
  [ -f params/seeded_$p.npz ] || JAX_PLATFORMS=cpu $PYB -I scripts/dump_params.py seeded_cc/$p params/seeded_$p.npz 2>&1 | tail -n 1
  for s in ${FAMS:-mol_z18_field bulk_f32_field}; do
    o=ref/seeded_cc/${p}_${s}; m="--max ${MAX:-60}"; [ $s = bulk_f32_field ] && m="--max 12"
    [ -f ${o}_gpu32.npz ] && continue
    JAX_PLATFORMS=cpu $P seeded_cc/$p frames/set_$s.extxyz ${o}_cpu64.npz --x64 --precision highest $m > ${o}_cpu64.log 2>&1
    JAX_PLATFORMS=cpu $P seeded_cc/$p frames/set_$s.extxyz ${o}_cpu32.npz $m > ${o}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $P seeded_cc/$p frames/set_$s.extxyz ${o}_cpu32_v2.npz $m > ${o}_cpu32_v2.log 2>&1
    $P seeded_cc/$p frames/set_$s.extxyz ${o}_gpu32.npz $m > ${o}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-100) | $(tail -n 1 ${o}_gpu32.log | cut -c1-60)"
  done
done
