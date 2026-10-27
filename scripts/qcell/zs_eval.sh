#!/usr/bin/env bash
# Zero-shot PET-OMol-S (mlipcpp-eval, feat/conditioning 903c04a build) on the QCell PET-OMol splits:
# xyz parts on GPUs 0-3 in parallel (inside qm9esp-cuda), via the holder job:
#   srun --jobid=5161597 --overlap --environment=qm9esp-cuda bash zs_eval.sh XYZ... 
set -uo pipefail
Q=/capstor/scratch/cscs/boittier/qcell
E=$Q/omol-eval; B=$E/bin/mlipcpp-eval; M=$E/pet-omol-s.gguf
export GGML_CUDA_NO_TF32=1
g=0
for x in "$@"; do
  o=$E/zs/$(basename $x .xyz).json; mkdir -p $E/zs
  (CUDA_VISIBLE_DEVICES=$g $B $M $x $o --backend cuda > $o.log 2>&1; echo "$(date -Is) $x rc $?" >> $E/zs/done.log) &
  g=$(( (g + 1) % 4 ))
done
wait
