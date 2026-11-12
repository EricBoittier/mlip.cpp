#!/bin/bash
#SBATCH --job-name=petlr-ref-thr
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, his branch feat/threshold-reference (101575d): the threshold presets (make_seeded.py thr / thr_cold_lrt,
# his JAX init) and his oracle through his batch builder (ref_branch.py): fp64 (cpu64) and three fp32 runs (CPU default,
# CPU single-thread, GPU) on mol_z18 (60, Z <= 18; charged molecules) / cluster (40, water) and bulk_f32 (periodic, 12 frames). Run in the long-range-transfer env
# with PYTHONPATH = a git worktree of his iris at 101575d (not -I: the path must win over the env's editable iris).
# Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
export PYTHONPATH=/scratch/boittier/ipsum/src/iris-threshold PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
PYB=/scratch/boittier/ipsum/envs/iris-long-range-transfer/bin/python
$PYB -c "import iris, iris.pet.charges as c; print(iris.__file__, 'threshold' in c.REFERENCES)"
mkdir -p seeded_thr ref/seeded_thr params
for p in ${PRESETS:-thr thr_cold_lrt}; do
  [ -d seeded_thr/$p ] || JAX_PLATFORMS=cpu $PYB scripts/make_seeded.py seeded_thr/$p --preset $p | tail -n 1 | cut -c1-120
  [ -f params/seeded_$p.npz ] || JAX_PLATFORMS=cpu $PYB scripts/dump_params.py seeded_thr/$p params/seeded_$p.npz 2>&1 | tail -n 1
  for s in ${FAMS:-mol_z18 cluster bulk_f32}; do
    o=ref/seeded_thr/${p}_${s}; c="--cutoffs differentiable --max ${MAX:-40}"; [ $s = mol_z18 ] && c="--cutoffs differentiable --max 60"; [ $s = bulk_f32 ] && c="--cutoffs differentiable --max 12"
    [ -f ${o}_gpu32.npz ] && continue
    JAX_PLATFORMS=cpu $PYB scripts/ref_branch.py seeded_thr/$p frames/set_$s.extxyz ${o}_cpu64.npz --x64 --precision highest $c > ${o}_cpu64.log 2>&1
    JAX_PLATFORMS=cpu $PYB scripts/ref_branch.py seeded_thr/$p frames/set_$s.extxyz ${o}_cpu32.npz $c > ${o}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $PYB scripts/ref_branch.py seeded_thr/$p frames/set_$s.extxyz ${o}_cpu32_v2.npz $c > ${o}_cpu32_v2.log 2>&1
    $PYB scripts/ref_branch.py seeded_thr/$p frames/set_$s.extxyz ${o}_gpu32.npz $c > ${o}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-100) | $(tail -n 1 ${o}_gpu32.log | cut -c1-60)"
  done
done
