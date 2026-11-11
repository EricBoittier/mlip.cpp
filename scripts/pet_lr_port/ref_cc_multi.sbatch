#!/bin/bash
#SBATCH --job-name=petlr-ref-ccmulti
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port: multi-component conserving charges (his a98c2f6) WITHOUT fragment inputs on mlip.cpp's side; his oracle
# (ref_branch_cc.py) run WITH fragment labels = the geometric components (pairs within the cutoff) and mlip.cpp's R-only
# Q_c (multi_comp.py label, torch fp64): presets cc_fermi / cc_residual on frames/multi/<p>_multi.extxyz (40: water
# pairs, Na...Cl, water + Na + Cl, split water clusters) and <p>_scan_nacl.extxyz (NaCl + water, 4.0-6.0 A across the
# 5 A cutoff). fp64 + three fp32 runs. Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
PYB=/scratch/boittier/ipsum/envs/iris-conserving-charges/bin/python
P="$PYB -I scripts/ref_branch_cc.py"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
for p in ${PRESETS:-cc_fermi cc_residual}; do
  for s in ${FAMS:-multi scan_nacl}; do
    o=ref/seeded_cc/${p}_${s}; fr=frames/multi/${p}_$s.extxyz
    [ -f ${o}_gpu32.npz ] && continue
    JAX_PLATFORMS=cpu $P seeded_cc/$p $fr ${o}_cpu64.npz --x64 --precision highest > ${o}_cpu64.log 2>&1
    JAX_PLATFORMS=cpu $P seeded_cc/$p $fr ${o}_cpu32.npz > ${o}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" $P seeded_cc/$p $fr ${o}_cpu32_v2.npz > ${o}_cpu32_v2.log 2>&1
    $P seeded_cc/$p $fr ${o}_gpu32.npz > ${o}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ${o}_cpu64.log | cut -c1-100) | $(tail -n 1 ${o}_gpu32.log | cut -c1-60)"
  done
done
