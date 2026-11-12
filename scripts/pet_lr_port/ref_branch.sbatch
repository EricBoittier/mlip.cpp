#!/bin/bash
#SBATCH --job-name=petlr-ref-branch
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, his branch feat/long-range-transfer (3ed96b0): make the branch presets (make_seeded.py, his JAX init,
# in the branch env), then his oracle through his batch builder (ref_branch.py): fp64 (cpu64) and three fp32 runs
# (CPU default, CPU single-thread, GPU) on the open families. Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
PYB=/scratch/boittier/ipsum/envs/iris-long-range-transfer/bin/python
P="$PYB -I scripts/ref_branch.py"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
mkdir -p seeded_branch ref/seeded_branch
for p in ${PRESETS:-lrt_weighted lrt_fill lrt_fill_cold}; do
  [ -d seeded_branch/$p ] || JAX_PLATFORMS=cpu $PYB -I scripts/make_seeded.py seeded_branch/$p --preset $p | tail -n 1 | cut -c1-120
  for s in ${FAMS:-ion cluster}; do
    o=ref/seeded_branch/${p}_${s}; c="--cutoffs differentiable --max ${MAX:-40}"
    [ -f ${o}_gpu32.npz ] && continue
    JAX_PLATFORMS=cpu $P seeded_branch/$p frames/set_$s.extxyz ${o}_cpu64.npz --x64 --precision highest $c > ${o}_cpu64.log 2>&1
    JAX_PLATFORMS=cpu $P seeded_branch/$p frames/set_$s.extxyz ${o}_cpu32.npz $c > ${o}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $P seeded_branch/$p frames/set_$s.extxyz ${o}_cpu32_v2.npz $c > ${o}_cpu32_v2.log 2>&1
    $P seeded_branch/$p frames/set_$s.extxyz ${o}_gpu32.npz $c > ${o}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-100) | $(tail -n 1 ${o}_gpu32.log | cut -c1-60)"
  done
done
