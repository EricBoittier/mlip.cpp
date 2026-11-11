#!/bin/bash
#SBATCH --job-name=petlr-ref-qeq
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, QEq (iris.pet.charges.qeq_reference, a2658e3): the seeded tiny_qeq preset through his calculator
# (ref_petlr.py, main env): fp64 (cpu64, if missing) and three fp32 runs (CPU default, CPU single-thread, GPU) on the
# open mol_z18 family and the periodic water bulk_f32 family. Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
P="/scratch/boittier/ipsum/env/bin/python -I scripts/ref_petlr.py"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
p=tiny_qeq
for s in ${FAMS:-mol_z18 bulk_f32}; do
  o=ref/seeded/${p}_${s}; m="--max 60"; [ $s = bulk_f32 ] && m="--max 12"
  [ -f ${o}_cpu64.npz ] || JAX_PLATFORMS=cpu $P seeded/$p frames/set_$s.extxyz ${o}_cpu64.npz --x64 --precision highest $m > ${o}_cpu64.log 2>&1
  JAX_PLATFORMS=cpu $P seeded/$p frames/set_$s.extxyz ${o}_cpu32.npz $m > ${o}_cpu32.log 2>&1
  JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    $P seeded/$p frames/set_$s.extxyz ${o}_cpu32_v2.npz $m > ${o}_cpu32_v2.log 2>&1
  $P seeded/$p frames/set_$s.extxyz ${o}_gpu32.npz $m > ${o}_gpu32.log 2>&1
  echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-100) | $(tail -n 1 ${o}_gpu32.log | cut -c1-60)"
done
