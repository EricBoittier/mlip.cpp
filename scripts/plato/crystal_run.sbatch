#!/usr/bin/env bash
# A crystal (ai4sci-atomistic-materials) Plato-PET-LR run on the B200 as a chain of jobs that resume
# from the trainer state: E + F + stress (weight SW), linear schedule after DECAY epochs, validation
# every epoch on the 5,000 labeled-dev frames, a checkpoint every 5 epochs, the best by R2(E) + R2(F).
#   NAME=D2 INIT=crystal/init128h8-D2-0.gguf EPOCHS=60 sbatch crystal_run.sbatch   [SW=1000 LR=1e-4 DECAY=2 SEED=0 MAXLEN=100 OPT=adam|muon MUON_SCALE=3]
#SBATCH --job-name=plato-crystal
#SBATCH --account=cosmo
#SBATCH --partition=b200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato/logs/%x-%j.out
set -uo pipefail
D=/work/cosmo/boittier/lyra/mlipcpp/pd-bench/dev-plato
X=$D/crystal
module purge; module load gcc/14.3.0 cuda/12.9.1 cmake/3.31.9 nccl
echo "node $(hostname) gpu $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) job $SLURM_JOB_ID"
: "${NAME:?}" "${INIT:?}" "${EPOCHS:?}"
RUN=$X/runs/$NAME; mkdir -p $RUN
[ -x $RUN/mlipcpp-train-plato ] || cp $D/build-cuda/bin/mlipcpp-train-plato $RUN/   # the binary of the run's first job
export GGML_CUDA_NO_TF32=1
if [ -f $RUN/model.gguf.state ]; then
  ARGS="--init $RUN/model.gguf --resume $RUN/model.gguf.state"
else
  ARGS="--init $D/$INIT"
fi
$RUN/mlipcpp-train-plato $ARGS --train $X/train.bin --val $X/val.bin --backend cuda \
  --epochs $EPOCHS --batch-size 32 --lr ${LR:-1e-4} --schedule linear --decay-after ${DECAY:-2} --min-lr 1e-6 \
  --grad-clip 1e30 --stress-weight ${SW:-1000} --optimizer ${OPT:-adam} --muon-lr-scale ${MUON_SCALE:-3} --max-len ${MAXLEN:-100} --seed ${SEED:-0} --save-every 5 --state-every 1 --max-hours ${HOURS:-23.3} \
  -o $RUN/model.gguf --log $RUN/log.jsonl 2>&1 | grep --line-buffered -v "^ggml_cuda\|Device \|warmup\|^Backend" > $RUN/job-$SLURM_JOB_ID.out
rc=${PIPESTATUS[0]}
tail -2 $RUN/job-$SLURM_JOB_ID.out
if grep -q "^paused at epoch" $RUN/job-$SLURM_JOB_ID.out; then
  cd $D && sbatch --job-name=plato-crystal-$NAME $D/mlip.cpp/scripts/plato/crystal_run.sbatch
elif [ $rc -ne 0 ] && [ -f $RUN/model.gguf.state ] && [ ${RETRIES:-0} -lt 3 ]; then
  # a crash (e.g. a GPU hardware error): resume from the last state on another node, at most 3 times
  echo "trainer exited with $rc on $(hostname): resubmitting (retry $(( ${RETRIES:-0} + 1 )))"
  cd $D && RETRIES=$(( ${RETRIES:-0} + 1 )) sbatch --exclude=$(hostname) --job-name=plato-crystal-$NAME $D/mlip.cpp/scripts/plato/crystal_run.sbatch
fi
