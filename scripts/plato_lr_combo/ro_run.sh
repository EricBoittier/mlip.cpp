#!/bin/bash
# Readout-init inputs (gpupool task, one GPU): node-347 features h_i (input of the last node-energy head), model
# energies / forces and the reference charges, for the w45 (cluster split 4.5 / 7.5) and neutral C8q0 variants.
#   ro_run.sh VARIANT   (VARIANT = CBw45 | CBneu; the GGUFs in models/ladder, q.gguf beside them)
set -eu
B=/capstor/scratch/cscs/boittier/agents/plato-v2; M=$B/models/ladder; v=$1; o=$B/combo/ro/$v; mkdir -p $o
export GGML_CUDA_NO_TF32=1 MLIPCPP_PAD_EXACT=1; unset NVIDIA_TF32_OVERRIDE
for s in train heldout; do
  x=$B/combo/ro/data/$s.xyz
  [ -s $o/$s.feats.bin ] || $B/bin7/mlipcpp-feats $M/$v.gguf $x $o/$s.feats.bin 347 > $o/$s.feats.log 2>&1
  [ -s $o/$s.eval.json ] || $B/bin7/mlipcpp-eval $M/$v.gguf $x $o/$s.eval.json --backend cuda > $o/$s.eval.log 2>&1
  [ -s $o/$s.q.json ] || $B/bin7/mlipcpp-qpass - - $x $o/$s.q.json --combo $M/$v.gguf --backend cuda > $o/$s.q.log 2>&1
done
echo "$v done" > $o/DONE
