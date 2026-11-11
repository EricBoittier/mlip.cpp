#!/bin/bash
#SBATCH --job-name=petlr-ref-ccind
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, his feat/conserving-charges (a98c2f6) induced_dipoles. Oracle jax-pme: his pin 1c26495 + the EwaldDipole
# commits of lab-cosmo/jax-pme#38 (3ad12e4, bd74953, 11a3e7a, 2a8d052) cherry-picked LOCALLY (local branch
# local/ewalddipole-1c26495-4tuple = 7443866 in our clone; 785337c, "dipoles as first-class batch inputs", left out:
# its 5-tuple get_batch is what broke his 4-tuple _flatten_jaxpme), a git archive at /scratch/boittier/ipsum/src/
# jax-pme-dip4 on PYTHONPATH over the env's 1c26495 (not -I). His own tests pass on it (test_dipoles 19, test_conserving
# -k induced 1, jax-pme dipole tests 34; job 4894205).
# (0) the existing cc_fermi mol_z18 fp64 ref reproduced with that jax-pme (5 frames); (1) presets cc_ind_field /
# cc_ind_full: his oracle (ref_branch_cc.py) fp64 + three fp32 runs on mol_z18_field (open), mol_z18_field0 (the first
# 20 with a zero field), bulk_f32_field (12 water cells, periodic). Evaluation only.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
export PYTHONPATH=/scratch/boittier/ipsum/src/jax-pme-dip4 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
PYB=/scratch/boittier/ipsum/envs/iris-conserving-charges/bin/python
P="$PYB scripts/ref_branch_cc.py"
$PYB -c "import jaxpme, jaxpme.batched_tiled as b; print(jaxpme.__file__, hasattr(b, 'EwaldDipole'))"
mkdir -p seeded_cc ref/seeded_cc params
if [ ! -f ref/seeded_cc/cc_fermi_dip4_check.npz ]; then
  JAX_PLATFORMS=cpu $P seeded_cc/cc_fermi frames/set_mol_z18.extxyz ref/seeded_cc/cc_fermi_dip4_check.npz --x64 --precision highest --max 5 > ref/seeded_cc/cc_fermi_dip4_check.log 2>&1
  $PYB -c "
import numpy as np
a = np.load('ref/seeded_cc/cc_fermi_dip4_check.npz'); b = np.load('ref/seeded_cc/cc_fermi_mol_z18_cpu64.npz'); n = a['atom_ptr'][-1]
print('dip4 vs 1c26495 cc_fermi fp64 (5 frames): E', np.abs(a['energy'] - b['energy'][:5]).max(), 'F', np.abs(a['forces'] - b['forces'][:n]).max(), 'q', np.abs(a['charges'] - b['charges'][:n]).max())"
fi
for p in ${PRESETS:-cc_ind_field cc_ind_full}; do
  [ -d seeded_cc/$p ] || JAX_PLATFORMS=cpu $PYB scripts/make_seeded.py seeded_cc/$p --preset $p | tail -n 3 | cut -c1-160
  [ -f params/seeded_$p.npz ] || JAX_PLATFORMS=cpu $PYB scripts/dump_params.py seeded_cc/$p params/seeded_$p.npz 2>&1 | tail -n 1
  for s in ${FAMS:-mol_z18_field mol_z18_field0 bulk_f32_field}; do
    o=ref/seeded_cc/${p}_${s}; m="--max 60"; [ $s = bulk_f32_field ] && m="--max 12"
    [ -z "${FORCE:-}" ] && [ -f ${o}_gpu32.npz ] && continue
    JAX_PLATFORMS=cpu $P seeded_cc/$p frames/set_$s.extxyz ${o}_cpu64.npz --x64 --precision highest $m > ${o}_cpu64.log 2>&1
    echo "$p $s cpu64: $(tail -n 1 ${o}_cpu64.log | cut -c1-140)"
    [ -n "${ONLY64:-}" ] && continue
    JAX_PLATFORMS=cpu $P seeded_cc/$p frames/set_$s.extxyz ${o}_cpu32.npz $m > ${o}_cpu32.log 2>&1
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $P seeded_cc/$p frames/set_$s.extxyz ${o}_cpu32_v2.npz $m > ${o}_cpu32_v2.log 2>&1
    $P seeded_cc/$p frames/set_$s.extxyz ${o}_gpu32.npz $m > ${o}_gpu32.log 2>&1
    echo "$p $s: $(tail -n 1 ${o}_gpu32.log | cut -c1-100)"
  done
done
