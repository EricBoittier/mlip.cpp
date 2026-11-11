#!/bin/bash -l
#SBATCH --job-name=petlr-ref-leslr
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: the charge_layer oracle. His les-water ladder WITH its LR side (iris explore/alessandro-strict-local
# 1d412cf: LayeredBackbone + per_layer_readout + charge_layer + the pre-flows physical charge head), presets
# les_ladder_lr / les_ladder_lr_c0 (make_seeded.py, his JAX init), through that branch's batch builder (ref_les_lr.py:
# AdaptiveToSample + pet_batcher + predict, per-atom charges from _forward): fp64 and three fp32 runs on mol_z18 (60)
# and bulk_f32 (12). MAIN env with PYTHONPATH = that worktree + his backbones.py. Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
export PYTHONPATH=/scratch/boittier/ipsum/src/iris-strict-local:$W/les_backbones PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
PY=/scratch/boittier/ipsum/env/bin/python
mkdir -p seeded_les ref/seeded_les params
for p in ${PRESETS:-les_ladder_lr les_ladder_lr_c0}; do
  [ -d seeded_les/$p ] || JAX_PLATFORMS=cpu $PY scripts/make_seeded.py seeded_les/$p --preset $p | tail -n 1 | cut -c1-160
  [ -f params/seeded_$p.npz ] || JAX_PLATFORMS=cpu $PY scripts/dump_params.py seeded_les/$p params/seeded_$p.npz 2>&1 | tail -n 1
  for s in ${FAMS:-mol_z18 bulk_f32}; do
    o=ref/seeded_les/${p}_${s}; m="--max 60"; [ $s = bulk_f32 ] && m="--max 12"
    [ -f ${o}_gpu32.npz ] && continue
    JAX_PLATFORMS=cpu $PY scripts/ref_les_lr.py seeded_les/$p frames/set_$s.extxyz ${o}_cpu64.npz --x64 --precision highest $m > ${o}_cpu64.log 2>&1
    JAX_PLATFORMS=cpu $PY scripts/ref_les_lr.py seeded_les/$p frames/set_$s.extxyz ${o}_cpu32.npz $m > ${o}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $PY scripts/ref_les_lr.py seeded_les/$p frames/set_$s.extxyz ${o}_cpu32_v2.npz $m > ${o}_cpu32_v2.log 2>&1
    $PY scripts/ref_les_lr.py seeded_les/$p frames/set_$s.extxyz ${o}_gpu32.npz $m > ${o}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-100) | $(tail -n 1 ${o}_gpu32.log | cut -c1-60)"
  done
done
