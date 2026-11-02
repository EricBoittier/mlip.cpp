#!/bin/bash
#SBATCH --job-name=petlr-split
#SBATCH --account=ab010
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/boittier/agents/petlr/split/job-%j.out
# PET-OMol-LR fine-tune S: E-init plus the gated E_split term (split/in/E-init-split.gguf: plato_split_gate 0,
# plato_split_radius 8 A, delta 1 A, radius_max 10 A), exactly as run_E_seeds.sbatch's E runs otherwise
# (sched_E.json: the gate in "gates", backend.les.split_radius in "new"; charge-corr 4e-4), with the
# split/bin binaries (feat/pet-plato-lr-split); SEEDS="a b c d": one run per GPU, then val_sel and QCell
# curves of each best checkpoint into omol25/ft-eval-G.
srun --environment=qm9esp-cuda bash -c '
D=/capstor/scratch/cscs/boittier/agents/petlr; W=$D/abc; O=$D/split
OM=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell; R=$OM/ft-eval-G; BIN=$O/bin
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1 MLIPCPP_LOG_PARAMS=plato_les_gate,plato_mesh_gate,plato_split_gate,plato_split_radius
run() { # name gpu seed
  mkdir -p $O/$1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-train --data $W/data --exclude $W/exclude --init $O/in/E-init-split.gguf -o $O/$1/model.gguf \
    --gpus 1 --batch-size 16 --epochs 8 --conservative --model-baselines \
    --energy-weight 1 --force-weight 1 --nc-force-weight 0 --loss huber --huber-delta-e 0.01 --huber-delta-f 0.02 \
    --lr-schedule $D/sched_E.json --seed $3 --threads 32 --cache-dir $W/nlcache --save-every 1 --log $O/$1/log.jsonl --log-every 200 \
    --charge-corr-weight 0.0004 --charge-corr-target mulliken > $O/$1/train.log 2>&1
  mkdir -p $R/$1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-eval $O/$1/model.gguf.best.gguf $OM/xyz/val_sel.xyz $R/$1/val_sel.json --backend cuda > $R/$1/val_sel.log 2>&1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-eval $O/$1/model.gguf.best.gguf $Q/data-omol/xyz/curves.xyz $R/$1/qcell_curves.json --backend cuda > $R/$1/qcell_curves.log 2>&1
}
gpu=0
for s in '"$SEEDS"'; do run S-s$s $gpu $s & gpu=$((gpu + 1)); done
wait'
