#!/bin/bash
#SBATCH --job-name=petlr-ref-spread
#SBATCH --account=cosmo
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/boittier/pet_lr_port/logs/%x-%j.out
# His fp32 floor spread: independent JAX fp32 runs of the trunk oracle that differ in reduction shapes / threads /
# device / autotuning (MODE=cpu: three CPU variants in parallel; MODE=gpu: one GPU variant per job, VAR names it).
set -uo pipefail
unset NVIDIA_TF32_OVERRIDE
W=/scratch/boittier/pet_lr_port; cd $W
P="/scratch/boittier/ipsum/env/bin/python -I scripts/ref_trunk.py"
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
for CK in finetune/readout-pol scaling/readout-4m; do
  C=/work/cosmo/langer/kuma/argos/checkpoints/omol-4m/$CK/RMSE_F; T=$(echo $CK | tr / _); mkdir -p ref/$T
  for s in mol cluster bulk; do
    if [ "$MODE" = cpu ]; then
      # sequential: three JAX CPU processes at once exhaust a MIG slice's memory
      JAX_PLATFORMS=cpu $P $C frames/set_$s.extxyz ref/$T/${s}_cpu32_v1.npz --kpad 16 --bucket powers_of_2 > ref/$T/${s}_cpu32_v1.log 2>&1
      JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        $P $C frames/set_$s.extxyz ref/$T/${s}_cpu32_v2.npz > ref/$T/${s}_cpu32_v2.log 2>&1
      JAX_PLATFORMS=cpu $P $C frames/set_$s.extxyz ref/$T/${s}_cpu32_v3.npz --kpad 24 --bucket multiples_of_32 > ref/$T/${s}_cpu32_v3.log 2>&1
    else
      $P $C frames/set_$s.extxyz ref/$T/${s}_gpu32_$VAR.npz $GPUARGS > ref/$T/${s}_gpu32_$VAR.log 2>&1; echo "$CK $s rc=$?"
    fi
  done
done
