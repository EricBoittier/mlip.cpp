#!/bin/bash
#SBATCH --job-name=petlr-Eseeds
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/boittier/agents/petlr/E/job-seeds-%j.out
# More seeds of E (E-init, gates trainable, charge-corr 4e-4) and its control Ectl (A, every gate frozen at 0),
# exactly as run_E.sbatch (same frozen E/bin binaries, schedules, data, 8 epochs), for the seed spread of the
# E - Ectl comparison: SEEDS="a b" (two seeds, one E + one Ectl run each, one GPU per run), then val_sel and
# QCell curves of each best checkpoint into omol25/ft-eval-G (post_G.sh pairs E-sN with Ectl-sN).
srun --environment=qm9esp-cuda bash -c '
D=/capstor/scratch/cscs/boittier/agents/petlr; W=$D/abc; O=$D/E; F=/capstor/scratch/cscs/boittier/omol25/ft
OM=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell; R=$OM/ft-eval-G; BIN=$O/bin
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1 MLIPCPP_LOG_PARAMS=plato_les_gate,plato_mesh_gate
run() { # name gpu init spec seed extra
  mkdir -p $O/$1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-train --data $W/data --exclude $W/exclude --init $3 -o $O/$1/model.gguf \
    --gpus 1 --batch-size 16 --epochs 8 --conservative --model-baselines \
    --energy-weight 1 --force-weight 1 --nc-force-weight 0 --loss huber --huber-delta-e 0.01 --huber-delta-f 0.02 \
    --lr-schedule $4 --seed $5 --threads 32 --cache-dir $W/nlcache --save-every 1 --log $O/$1/log.jsonl --log-every 200 $6 \
    > $O/$1/train.log 2>&1
  mkdir -p $R/$1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-eval $O/$1/model.gguf.best.gguf $OM/xyz/val_sel.xyz $R/$1/val_sel.json --backend cuda > $R/$1/val_sel.log 2>&1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-eval $O/$1/model.gguf.best.gguf $Q/data-omol/xyz/curves.xyz $R/$1/qcell_curves.json --backend cuda > $R/$1/qcell_curves.log 2>&1
}
QC="--charge-corr-weight 0.0004 --charge-corr-target mulliken"
set -- '"$SEEDS"'
run E-s$1    0 $F/E-init/model.gguf $D/sched_E.json    $1 "$QC" &
run Ectl-s$1 1 $F/A/model.gguf      $D/sched_Ectl.json $1 ""    &
run E-s$2    2 $F/E-init/model.gguf $D/sched_E.json    $2 "$QC" &
run Ectl-s$2 3 $F/A/model.gguf      $D/sched_Ectl.json $2 ""    &
wait'
