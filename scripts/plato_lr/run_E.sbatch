#!/bin/bash
#SBATCH --job-name=petlr-E
#SBATCH --account=lp161
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/boittier/agents/petlr/E/job-%j.out
# PET-OMol-LR fine-tune E (LES tail sigma_lr 4 A, les_gate 2e-3) and its control (A continued, gates frozen),
# two seeds each, one GPU per run; same data, schedule, steps; all checkpoints per epoch
D=/capstor/scratch/cscs/boittier/agents/petlr; W=$D/abc; O=$D/E; mkdir -p $O
EPOCHS=${EPOCHS:-8}
srun --environment=qm9esp-cuda bash -c '
D=/capstor/scratch/cscs/boittier/agents/petlr; W=$D/abc; O=$D/E; F=/capstor/scratch/cscs/boittier/omol25/ft
# the trainer of feat/pet-plato-lr (LES tail) built into build/ (incremental), frozen for the runs in E/bin
export PATH=$D/stubbin:/users/boittier/venvs/mptrj-py/bin:$PATH
cmake --build $D/build -j 64 --target mlipcpp-train mlipcpp-eval > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
mkdir -p $O/bin && cp $D/build/bin/mlipcpp-train $D/build/bin/mlipcpp-eval $O/bin/
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1 MLIPCPP_LOG_PARAMS=plato_les_gate,plato_mesh_gate
run() { # name gpu init spec seed extra
  mkdir -p $O/$1
  CUDA_VISIBLE_DEVICES=$2 $O/bin/mlipcpp-train --data $W/data --exclude $W/exclude --init $3 -o $O/$1/model.gguf \
    --gpus 1 --batch-size 16 --epochs '$EPOCHS' --conservative --model-baselines \
    --energy-weight 1 --force-weight 1 --nc-force-weight 0 --loss huber --huber-delta-e 0.01 --huber-delta-f 0.02 \
    --lr-schedule $4 --seed $5 --threads 32 --cache-dir $W/nlcache --save-every 1 --log $O/$1/log.jsonl --log-every 200 $6 \
    > $O/$1/train.log 2>&1
}
QC="--charge-corr-weight 0.0004 --charge-corr-target mulliken"
run E-s7    0 $F/E-init/model.gguf $D/sched_E.json    7  "$QC" &
run Ectl-s7 1 $F/A/model.gguf      $D/sched_Ectl.json 7  ""    &
run E-s13   2 $F/E-init/model.gguf $D/sched_E.json    13 "$QC" &
run Ectl-s13 3 $F/A/model.gguf     $D/sched_Ectl.json 13 ""    &
wait'
