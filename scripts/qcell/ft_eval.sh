#!/usr/bin/env bash
# Evaluation of the OMol25 fine-tunes A (gates frozen at 0), B (PET-OMol-LR), C (B + Mulliken charge-corr
# loss) against each other and the zero-shot PET-OMol-S (Z), on GPU 3 of the holder job:
#   srun --jobid=5161597 --overlap -n1 --environment=qm9esp-cuda bash ft_eval.sh [MODEL_NAME=model.gguf]
# For each X with $O/ft/X/$MODEL: mlipcpp-eval on the OMol25 val selection and on QCell's charged curves
# holdout; eval_omol.py strata (+ per-structure errors); then (on the host, see ft_compare.sh) paired B-A,
# C-A (and X-Z) with bootstrap CIs, and latent-charge correlations for B and C.
set -uo pipefail
O=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell
EV=${EVAL_BIN:-$Q/omol-eval/bin/mlipcpp-eval}; MODEL=${1:-model.gguf}
export GGML_CUDA_NO_TF32=1 CUDA_VISIBLE_DEVICES=${GPU:-3}
for X in Z A B C; do
  M=$O/ft/$X/$MODEL; [ $X = Z ] && M=$Q/omol-eval/pet-omol-s.gguf
  [ -f $M ] || { echo "$X: no $M yet"; continue; }
  R=$O/ft-eval/$X; mkdir -p $R
  [ -s $R/val_sel.json ] || $EV $M $O/xyz/val_sel.xyz $R/val_sel.json --backend cuda > $R/val_sel.log 2>&1
  [ -s $R/qcell_curves.json ] || $EV $M $Q/data-omol/xyz/curves.xyz $R/qcell_curves.json --backend cuda > $R/qcell_curves.log 2>&1
  echo "$X evaluated $(date -Is)"
done
