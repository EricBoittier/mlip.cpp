#!/bin/bash
#SBATCH --job-name=petlr-G2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/boittier/agents/petlr/G/job2-%j.out
# G3b / G4b: G3 and G4 again with the les_gate schedule group matching the trainer's state-dict name
# (backend.les.les_gate; the GGUF name matched no tensor, so G3 / G4 ran with a trainable gate), and seed-13
# repeats of G1 and Gs2. Same data, schedule and steps as run_G.sbatch / run_Gsub.sbatch.
srun --environment=qm9esp-cuda bash -c '
D=/capstor/scratch/cscs/boittier/agents/petlr; W=$D/abc; O=$D/G; OM=/capstor/scratch/cscs/boittier/omol25
Q=/capstor/scratch/cscs/boittier/qcell; R=$OM/ft-eval-G; BIN=$D/E/bin; S=$D/scripts
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1 MLIPCPP_LOG_PARAMS=plato_les_gate,plato_mesh_gate
run() { # name gpu init spec seed
  mkdir -p $O/$1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-train --data $W/data --exclude $W/exclude --init $3 -o $O/$1/model.gguf \
    --gpus 1 --batch-size 16 --epochs 8 --conservative --model-baselines \
    --energy-weight 1 --force-weight 1 --nc-force-weight 0 --loss huber --huber-delta-e 0.01 --huber-delta-f 0.02 \
    --lr-schedule $S/$4 --seed $5 --threads 32 --cache-dir $W/nlcache --save-every 1 --log $O/$1/log.jsonl --log-every 200 \
    --charge-corr-weight 0.0004 --charge-corr-target mulliken > $O/$1/train.log 2>&1
}
ev() { # name model gpu
  mkdir -p $R/$1
  [ -s $R/$1/val_sel.json ] || CUDA_VISIBLE_DEVICES=$3 $BIN/mlipcpp-eval $2 $OM/xyz/val_sel.xyz $R/$1/val_sel.json --backend cuda > $R/$1/val_sel.log 2>&1
  [ -s $R/$1/qcell_curves.json ] || CUDA_VISIBLE_DEVICES=$3 $BIN/mlipcpp-eval $2 $Q/data-omol/xyz/curves.xyz $R/$1/qcell_curves.json --backend cuda > $R/$1/qcell_curves.log 2>&1
  echo "eval $1 $(date -Is)" >> $O/eval.log
}
run G3b    0 $O/in/G3.gguf        finetune_G_les_fixed_lr_schedule.json 7  &
run G4b    1 $O/in/G4.gguf        finetune_G_les_hold2_lr_schedule.json 7  &
run G1-s13 2 $O/in/G1.gguf        finetune_E_lr_schedule.json           13 &
run Gs2-s13 3 $D/Gsub/in/Gs2.gguf finetune_E_lr_schedule.json           13 &
wait
k=0; for X in G3b G4b G1-s13 Gs2-s13; do ev $X $O/$X/model.gguf.best.gguf $k & k=$((k+1)); done
wait'
