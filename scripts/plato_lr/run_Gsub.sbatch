#!/bin/bash
#SBATCH --job-name=petlr-Gsub
#SBATCH --account=lp161
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/boittier/agents/petlr/Gsub/job-%j.out
# PET-OMol-LR fine-tune Gsub: E-init re-exported with the LES tail's within-cutoff subtraction
# (G/in/E-init-sub.gguf: LES energy at gate 1 about 151 meV/atom mean |.| on 12 val_sel frames, against
# 1022 without), so LES adds only what lies beyond the PET cutoff. Same data, schedule, seed 7 and steps as
# E-s7 / Ectl-s7 (the control) and G; every gate trainable, the LES gate started at
#   Gs1 0.001 (E's start)   Gs2 0.05   Gs3 0.3   Gs4 1
# GPU 0 also evaluates the LES decomposition of E-init-sub (gate 0 vs 1); afterwards every Gs run is
# evaluated (val_sel, QCell curves). Host side: post_G.sh (ft-eval-G).
D=/capstor/scratch/cscs/boittier/agents/petlr; O=$D/Gsub; mkdir -p $O
EPOCHS=${EPOCHS:-8}
srun --environment=qm9esp-cuda bash -c '
D=/capstor/scratch/cscs/boittier/agents/petlr; W=$D/abc; O=$D/Gsub; F=/capstor/scratch/cscs/boittier/omol25/ft
OM=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell; R=$OM/ft-eval-G
BIN=$D/E/bin; PY=/users/boittier/venvs/mptrj-py/bin/python; S=$D/scripts
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1 MLIPCPP_LOG_PARAMS=plato_les_gate,plato_mesh_gate
mkdir -p $O/in $R
P() { $PY -I $S/patch_gguf.py "$@" > /dev/null; }
I=$D/G/in/E-init-sub.gguf
P $I $O/in/Gs1.gguf --tensor plato_les_gate=0.001
P $I $O/in/Gs2.gguf --tensor plato_les_gate=0.05
P $I $O/in/Gs3.gguf --tensor plato_les_gate=0.3
P $I $O/in/Gs4.gguf --tensor plato_les_gate=1
run() { # name gpu spec
  mkdir -p $O/$1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-train --data $W/data --exclude $W/exclude --init $O/in/$1.gguf -o $O/$1/model.gguf \
    --gpus 1 --batch-size 16 --epochs '$EPOCHS' --conservative --model-baselines \
    --energy-weight 1 --force-weight 1 --nc-force-weight 0 --loss huber --huber-delta-e 0.01 --huber-delta-f 0.02 \
    --lr-schedule $S/$3 --seed 7 --threads 32 --cache-dir $W/nlcache --save-every 1 --log $O/$1/log.jsonl --log-every 200 \
    --charge-corr-weight 0.0004 --charge-corr-target mulliken > $O/$1/train.log 2>&1
}
ev() { # name model gpu [qcell]
  mkdir -p $R/$1
  [ -s $R/$1/val_sel.json ] || CUDA_VISIBLE_DEVICES=$3 $BIN/mlipcpp-eval $2 $OM/xyz/val_sel.xyz $R/$1/val_sel.json --backend cuda > $R/$1/val_sel.log 2>&1
  [ -n "$4" ] && { [ -s $R/$1/qcell_curves.json ] || CUDA_VISIBLE_DEVICES=$3 $BIN/mlipcpp-eval $2 $Q/data-omol/xyz/curves.xyz $R/$1/qcell_curves.json --backend cuda > $R/$1/qcell_curves.log 2>&1; }
  echo "eval $1 $(date -Is)" >> $O/eval.log
}
run Gs1 0 finetune_E_lr_schedule.json &
run Gs2 1 finetune_E_lr_schedule.json &
run Gs3 2 finetune_E_lr_schedule.json &
run Gs4 3 finetune_E_lr_schedule.json &
( for g in 0 1; do P $I $O/in/E-init-sub-g$g.gguf --tensor plato_les_gate=$g; ev les-E-init-sub-g$g $O/in/E-init-sub-g$g.gguf 0; rm -f $O/in/E-init-sub-g$g.gguf; done ) &
wait
for k in 1 2 3 4; do ev Gs$k $O/Gs$k/model.gguf.best.gguf $((k-1)) q & done
wait'
