#!/bin/bash
#SBATCH --job-name=petlr-ref-fullspread
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=29000M
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# PET-LR port, phase 2: independent fp32 runs of his FULL PETLR (ref_petlr.py, M. Langer's env, read-only) for the
# floor of the LR outputs (E, F, charges, mu, dipole, metallicity): CPU single-threaded (XLA flags) and GPU (MIG
# H100, default), next to the CPU default run of ref_petlr.sbatch.
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
P="/scratch/boittier/ipsum/env/bin/python -I scripts/ref_petlr.py"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
for CK in ${CKS:-finetune/readout-pol finetune/control}; do
  C=/work/cosmo/langer/kuma/argos/checkpoints/omol-4m/$CK/RMSE_F
  T=$(echo $CK | tr / _)
  for s in ${FAMS:-ion cluster}; do
    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
      $P $C frames/set_$s.extxyz ref/$T/full_${s}_cpu32_v2.npz > ref/$T/full_${s}_cpu32_v2.log 2>&1
    $P $C frames/set_$s.extxyz ref/$T/full_${s}_gpu32.npz > ref/$T/full_${s}_gpu32.log 2>&1
    echo "$CK $s rc=$?"
  done
done
