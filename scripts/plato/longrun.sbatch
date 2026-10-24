#!/usr/bin/env bash
# The long Plato-PET-LR run in mlip.cpp (LOREM's bio_dimers protocol: 4000 epochs, batch 32,
# Adam 1e-4, constant for 10 epochs then linear to 1e-6), as a chain of jobs that resume
# from the trainer state; each job resubmits the next until the run is done.
#   NAME=chi INIT=init-chi.gguf sbatch longrun.sbatch
# Optional: BASE (run directory under dev-plato, default long), DATA (directory with train.bin and
# val.bin, default data), CLIP (gradient clip, default 1; 1e30 = none), VAL_EVERY (default 1), SEED.
#SBATCH --job-name=plato-long
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -uo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
echo "node $(hostname) gpu $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) job $SLURM_JOB_ID"
: "${NAME:?}" "${INIT:?}"
RUN=$D/${BASE:-long}/$NAME; mkdir -p $RUN
DDIR=$D/${DATA:-data}
T=$D/build-cuda/bin/mlipcpp-train-plato
[ -x $RUN/mlipcpp-train-plato ] || cp $T $RUN/   # the binary of the run's first job
export GGML_CUDA_NO_TF32=1
if [ -f $RUN/model.gguf.state ]; then
  ARGS="--init $RUN/model.gguf --resume $RUN/model.gguf.state"
else
  ARGS="--init $D/models/$INIT"
fi
$RUN/mlipcpp-train-plato $ARGS --train $DDIR/train.bin --val $DDIR/val.bin --backend cuda \
  --epochs ${EPOCHS:-4000} --batch-size 32 --lr 1e-4 --schedule linear --decay-after 10 --min-lr 1e-6 \
  --grad-clip ${CLIP:-1} --val-every ${VAL_EVERY:-1} --fd-eps 5e-3 --seed ${SEED:-0} --save-every 100 --state-every 10 --max-hours ${HOURS:-23.3} \
  -o $RUN/model.gguf --log $RUN/log.jsonl 2>&1 | grep -v "^ggml_cuda\|Device \|warmup\|^Backend" > $RUN/job-$SLURM_JOB_ID.out
tail -2 $RUN/job-$SLURM_JOB_ID.out
if grep -q "^paused at epoch" $RUN/job-$SLURM_JOB_ID.out; then
  cd $D && NAME=$NAME INIT=$INIT sbatch --job-name=plato-long-$NAME $D/longrun.sbatch
fi
