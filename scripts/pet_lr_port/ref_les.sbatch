#!/bin/bash
#SBATCH --job-name=petlr-ref-les
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, his les-water backbones (ipsum-work play/les-water backbones.py + iris explore/alessandro-strict-local-20261012-v1, 1d412cf; worktree
# kuma /scratch/boittier/ipsum/src/iris-strict-local, imported under the MAIN env via PYTHONPATH, no -I): the seeded
# les_* presets (make_seeded.py, his JAX init), then his trunk oracle (ref_trunk.py: pet-jax structure + in-forward
# selection + that branch's iris.pet.model.PET): fp64 and three fp32 runs on mol_z18 (60) and bulk_f32 (12).
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
# his backbones.py (ipsum-work play/les-water, read-only copy of /work/cosmo/langer/kuma/argos/ipsum/ml/play/les-water/
# backbones.py) alone in its own directory, so nothing else of that folder lands on the path
mkdir -p $W/les_backbones && cp -n /work/cosmo/langer/kuma/argos/ipsum/ml/play/les-water/backbones.py $W/les_backbones/
export PYTHONPATH=/scratch/boittier/ipsum/src/iris-strict-local:$W/les_backbones PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
PY=/scratch/boittier/ipsum/env/bin/python
mkdir -p seeded_les ref/seeded_les params
for p in ${PRESETS:-les_local2 les_local3 les_ladder les_ladder_c0}; do
  [ -d seeded_les/$p ] || JAX_PLATFORMS=cpu $PY scripts/make_seeded.py seeded_les/$p --preset $p | tail -n 1 | cut -c1-160
  [ -f params/seeded_$p.npz ] || JAX_PLATFORMS=cpu $PY scripts/dump_params.py seeded_les/$p params/seeded_$p.npz 2>&1 | tail -n 1
  for s in ${FAMS:-mol_z18 bulk_f32}; do
    o=ref/seeded_les/${p}_${s}; m="--max 60"; [ $s = bulk_f32 ] && m="--max 12"
    [ -f ${o}_gpu32.npz ] && continue
    JAX_PLATFORMS=cpu $PY scripts/ref_trunk.py seeded_les/$p frames/set_$s.extxyz ${o}_cpu64.npz --x64 --precision highest $m > ${o}_cpu64.log 2>&1
    JAX_PLATFORMS=cpu $PY scripts/ref_trunk.py seeded_les/$p frames/set_$s.extxyz ${o}_cpu32.npz $m > ${o}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $PY scripts/ref_trunk.py seeded_les/$p frames/set_$s.extxyz ${o}_cpu32_v2.npz $m > ${o}_cpu32_v2.log 2>&1
    $PY scripts/ref_trunk.py seeded_les/$p frames/set_$s.extxyz ${o}_gpu32.npz $m > ${o}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-120) | $(tail -n 1 ${o}_gpu32.log | cut -c1-60)"
  done
done
