#!/bin/bash -l
#SBATCH --job-name=qcell-train
#SBATCH --account=lp161
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --output=/capstor/scratch/cscs/boittier/qcell/logs/%x-%j.out
# QCell Plato-PET-LR (D2-att, no E_chi) in mlip.cpp: four runs, one per GH200 of one node, each
# resumable from its trainer state (runs/NAME/model.gguf.state). A job trains for HOURS, then, if a
# run paused (not done), resubmits itself; a run that is done gets its test/val dumps and metrics
# (scripts/finish.sh) in the same job.
#   EPOCHS=... sbatch scripts/train.sbatch
set -uo pipefail
Q=/capstor/scratch/cscs/boittier/qcell
EPOCHS=${EPOCHS:?}; HOURS=${HOURS:-22.5}; BS=${BS:-32}; DECAY=${DECAY:-1}
echo "job $SLURM_JOB_ID node $(hostname) start $(date -Is) epochs $EPOCHS hours $HOURS"
# gpu name init data optimizer lr seed
RUNS=("0 ft-muon ft-init.gguf res-ft muon 1e-4 0"
      "1 ft-adam ft-init.gguf res-ft adam 3e-5 0"
      "2 scratch-muon scratch-init-s0.gguf res-scratch muon 1e-4 0"
      "3 ft-muon-s1 ft-init.gguf res-ft muon 1e-4 1")
srun --environment=qm9esp-cuda bash -c '
  Q='$Q' EPOCHS='$EPOCHS' HOURS='$HOURS' BS='$BS' DECAY='$DECAY'
  export GGML_CUDA_NO_TF32=1
  run() {
    G=$1 NAME=$2 INIT=$3 DATA=$4 OPT=$5 LR=$6 SEED=$7
    R=$Q/runs/$NAME; mkdir -p $R
    if grep -q "^done at epoch" $R/train.out 2>/dev/null; then
      [ -f $R/test.metrics.json ] || CUDA_VISIBLE_DEVICES=$G bash $Q/scripts/finish.sh $NAME $DATA
      return
    fi
    [ -x $R/mlipcpp-train-plato ] || cp $Q/build-cuda/bin/mlipcpp-train-plato $R/   # one binary per run
    if [ -f $R/model.gguf.state ]; then A="--init $R/model.gguf --resume $R/model.gguf.state"; else A="--init $Q/models/$INIT"; fi
    echo "$(date -Is) $NAME gpu $G start ($A)" >> $R/train.out
    CUDA_VISIBLE_DEVICES=$G $R/mlipcpp-train-plato $A --train $Q/data/$DATA/train.bin --val $Q/data/$DATA/val.bin --backend cuda \
      --epochs $EPOCHS --batch-size $BS --lr $LR --schedule linear --decay-after $DECAY --min-lr 1e-6 --grad-clip 1 --val-every 1 \
      --optimizer $OPT --muon-lr-scale 3 --fd-eps 5e-3 --seed $SEED --save-every 5 --state-every 1 --max-hours $HOURS --bucket-growth 1.3 --share-graph-buffer --graph-cache 256 --prep-threads 12 \
      -o $R/model.gguf --log $R/log.jsonl 2>&1 | grep --line-buffered -v "^ggml_cuda\|Device \|warmup\|^Backend\|  step " >> $R/train.out
    echo "$(date -Is) $NAME rc ${PIPESTATUS[0]}" >> $R/train.out
    grep -q "^done at epoch" $R/train.out && CUDA_VISIBLE_DEVICES=$G bash $Q/scripts/finish.sh $NAME $DATA
  }
  '"$(for r in "${RUNS[@]}"; do echo "run $r &"; done)"'
  wait
'
left=0
for r in "${RUNS[@]}"; do set -- $r; grep -q "^done at epoch" $Q/runs/$2/train.out 2>/dev/null || left=1; done
echo "end $(date -Is) left=$left"
if [ $left = 1 ] && [ ! -f $Q/STOP ]; then cd $Q && EPOCHS=$EPOCHS HOURS=$HOURS BS=$BS DECAY=$DECAY sbatch $Q/scripts/train.sbatch; fi
