#!/bin/bash
#SBATCH --job-name=petlr-F
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=07:00:00
#SBATCH --output=/capstor/scratch/cscs/boittier/agents/petlr/F/job-%j.out
# F: the LES gate held at a fixed value for all 8 epochs (lr x0 on backend.les.les_gate), four seeds per arm,
# against the E / Ectl seeds of the same number. G4b (0.2 held 2 epochs) and Gs4 (cutoff-subtracted, 1 -> 0.69)
# were the runs that resolved the separated-charged-fragment stratum (one seed each).
#   ARM=Fa: E-init (LES tail 4 A), les_gate fixed 0.2
#   ARM=Fb: E-init-sub (tail 4 A, within-cutoff part subtracted), les_gate fixed 0.5
# SEEDS="a b c d". Same data, schedule shape, loss and binaries (E/bin, a0f9aea) as E.
srun --environment=qm9esp-cuda bash -c '
D=/capstor/scratch/cscs/boittier/agents/petlr; W=$D/abc; O=$D/F; F=/capstor/scratch/cscs/boittier/omol25/ft
OM=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell; R=$OM/ft-eval-G; BIN=$D/E/bin; S=$D/scripts
PY=/users/boittier/venvs/mptrj-py/bin/python
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1 MLIPCPP_LOG_PARAMS=plato_les_gate,plato_mesh_gate
mkdir -p $O/in
case '"$ARM"' in
  Fa) $PY -I $S/patch_gguf.py $F/E-init/model.gguf $O/in/Fa.gguf --tensor plato_les_gate=0.2 > /dev/null ;;
  Fb) $PY -I $S/patch_gguf.py $D/G/in/E-init-sub.gguf $O/in/Fb.gguf --tensor plato_les_gate=0.5 > /dev/null ;;
  *) echo "ARM must be Fa or Fb"; exit 1 ;;
esac
run() { # name gpu seed
  mkdir -p $O/$1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-train --data $W/data --exclude $W/exclude --init $O/in/'"$ARM"'.gguf -o $O/$1/model.gguf \
    --gpus 1 --batch-size 16 --epochs 8 --conservative --model-baselines \
    --energy-weight 1 --force-weight 1 --nc-force-weight 0 --loss huber --huber-delta-e 0.01 --huber-delta-f 0.02 \
    --lr-schedule $S/finetune_G_les_fixed_lr_schedule.json --seed $3 --threads 32 --cache-dir $W/nlcache --save-every 1 \
    --log $O/$1/log.jsonl --log-every 200 --charge-corr-weight 0.0004 --charge-corr-target mulliken > $O/$1/train.log 2>&1
  mkdir -p $R/$1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-eval $O/$1/model.gguf.best.gguf $OM/xyz/val_sel.xyz $R/$1/val_sel.json --backend cuda > $R/$1/val_sel.log 2>&1
  CUDA_VISIBLE_DEVICES=$2 $BIN/mlipcpp-eval $O/$1/model.gguf.best.gguf $Q/data-omol/xyz/curves.xyz $R/$1/qcell_curves.json --backend cuda > $R/$1/qcell_curves.log 2>&1
}
k=0; for s in '"$SEEDS"'; do run '"$ARM"'-s$s $k $s & k=$((k+1)); done
wait'
