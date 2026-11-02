#!/bin/bash
#SBATCH --job-name=petlr-G
#SBATCH --account=lp161
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/boittier/agents/petlr/G/job-%j.out
# PET-OMol-LR fine-tune G: the LES gate started open (E's 1e-3 start closes within 1000 steps; D2 from 0.05 stayed
# at 0.03). Same data, schedule, seed 7 and steps as E-s7 / Ectl-s7 (the control), one GPU per run:
#   G1 E-init (LES tail sigma 4 A), les_gate 0.05, trainable
#   G2 C (LES sigma 1 A, no tail), les_gate 0.05, trainable (D2 for 8 epochs)
#   G3 E-init, les_gate 0.05, fixed
#   G4 E-init, les_gate 0.2, fixed for 2 epochs, then trainable
# GPU 0 also evaluates E / Ectl (val_sel, QCell curves) and the LES decomposition (gate 0 vs 1) meanwhile;
# after training every G run is evaluated the same way. Host side: post_G.sh.
D=/capstor/scratch/cscs/boittier/agents/petlr; O=$D/G; mkdir -p $O
EPOCHS=${EPOCHS:-8}
srun --environment=qm9esp-cuda bash -c '
D=/capstor/scratch/cscs/boittier/agents/petlr; W=$D/abc; O=$D/G; F=/capstor/scratch/cscs/boittier/omol25/ft
OM=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell; R=$OM/ft-eval-G
BIN=$D/E/bin; PY=/users/boittier/venvs/mptrj-py/bin/python; S=$D/scripts
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1 MLIPCPP_LOG_PARAMS=plato_les_gate,plato_mesh_gate
mkdir -p $O/in $R
P() { $PY -I $S/patch_gguf.py "$@" > /dev/null; }
P $F/E-init/model.gguf $O/in/G1.gguf --tensor plato_les_gate=0.05
P $F/C/model.gguf      $O/in/G2.gguf --tensor plato_les_gate=0.05
cp $O/in/G1.gguf $O/in/G3.gguf
P $F/E-init/model.gguf $O/in/G4.gguf --tensor plato_les_gate=0.2
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
run G1 0 finetune_E_lr_schedule.json &
run G2 1 finetune_E_lr_schedule.json &
run G3 2 finetune_G_les_fixed_lr_schedule.json &
run G4 3 finetune_G_les_hold2_lr_schedule.json &
( for X in E-s7 E-s13 Ectl-s7 Ectl-s13; do ev $X $D/E/$X/model.gguf.best.gguf 0 q; done
  # the LES energy alone: the same model with the gate at 0 and at 1
  for X in E-init:$F/E-init/model.gguf E-s7:$D/E/E-s7/model.gguf.best.gguf C:$F/C/model.gguf; do
    n=${X%%:*}; m=${X#*:}
    for g in 0 1; do P $m $O/in/$n-g$g.gguf --tensor plato_les_gate=$g; ev les-$n-g$g $O/in/$n-g$g.gguf 0; rm -f $O/in/$n-g$g.gguf; done
  done ) &
wait
for k in 1 2 3 4; do ev G$k $O/G$k/model.gguf.best.gguf $((k-1)) q & done
wait'
