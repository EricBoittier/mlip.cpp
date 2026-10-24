#!/usr/bin/env bash
# The long plain-PET run in mlipcpp-train with the protocol of the long Plato-PET-LR runs
# (scripts/plato/longrun.sbatch, LOREM's bio_dimers protocol): 4000 epochs, batch 32, Adam 1e-4
# constant for 10 epochs then linear to 1e-6, no gradient clipping, O(3) augmentation,
# conservative forces only (energy weight 1, force weight 1 on -dE/dx, direct head weight 0),
# FP32 GEMMs, validation on the full val split every epoch; OUT.best.gguf by R^2(E) + R^2(F),
# a GGUF every 100 epochs. A chain of jobs that resume from the trainer state; each job
# resubmits the next until the run is done.
#   NAME=pet sbatch pet_long.sbatch        (GPUS=2 for two GPUs of one node, data parallel; OPT=muon [MUON_SCALE=10])
#SBATCH --job-name=pet-long
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-pet/logs/%x-%j.out
set -uo pipefail
P=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-pet
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
echo "node $(hostname) gpu $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) job $SLURM_JOB_ID"
: "${NAME:?}"
RUN=$P/long/$NAME; mkdir -p $RUN
[ -x $RUN/mlipcpp-train ] || cp $P/build-cuda/bin/mlipcpp-train $RUN/   # the binary of the run's first job
export GGML_CUDA_NO_TF32=1
if [ -f $RUN/model.gguf.state ]; then
  FROM="--init $RUN/model.gguf --resume $RUN/model.gguf.state"
else
  FROM="--init $P/models/${INIT:-init.gguf}"
fi
$RUN/mlipcpp-train $FROM --data $P/data --gpus ${GPUS:-1} --batch-size $((32 / ${GPUS:-1})) \
  --conservative --energy-weight 1 --force-weight 1 --nc-force-weight 0 \
  --optimizer ${OPT:-adam} --muon-lr-scale ${MUON_SCALE:-10} --lr 1e-4 --schedule linear --decay-after 10 --min-lr 1e-6 --grad-clip 1e30 --seed ${SEED:-0} \
  --epochs ${EPOCHS:-4000} --save-every 100 --state-every 10 --max-hours ${HOURS:-23.3} \
  --threads 12 --cache-dir $P/nl-cache --log-every 400 \
  -o $RUN/model.gguf --log $RUN/log.jsonl 2>&1 | grep -v "^ggml_cuda\|Device \|^Backend\|packing cost" > $RUN/job-$SLURM_JOB_ID.out
tail -3 $RUN/job-$SLURM_JOB_ID.out
if grep -q "^paused at epoch" $RUN/job-$SLURM_JOB_ID.out; then
  cd $P && NAME=$NAME GPUS=${GPUS:-1} sbatch --job-name=pet-long-$NAME --gpus-per-node=${GPUS:-1} $P/mlip.cpp/scripts/train/pet_long.sbatch
fi
