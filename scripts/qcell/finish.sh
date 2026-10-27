#!/usr/bin/env bash
# Test and val predictions of a finished QCell run's final (model.gguf) and best-val
# (model.gguf.best.gguf) weights, then metrics (eval_qcell.py) with the run's baseline.
#   CUDA_VISIBLE_DEVICES=g bash finish.sh NAME DATA     (inside the qm9esp-cuda container)
set -uo pipefail
Q=/capstor/scratch/cscs/boittier/qcell
NAME=$1 DATA=$2; R=$Q/runs/$NAME
PY=/users/boittier/venvs/agents-torch/bin/python
for w in final best; do
  M=$R/model.gguf; [ $w = best ] && M=$R/model.gguf.best.gguf
  for s in test val; do
    [ -s $R/$s.$w.jsonl ] || $Q/build-cuda/bin/mlipcpp-plato dump $M $Q/data/xyz/$s.xyz --backend cuda > $R/$s.$w.jsonl 2> $R/$s.$w.err
  done
done
for w in final best; do
  for s in test val; do
    $PY -I $Q/scripts/eval_qcell.py $Q/data/split/$s.npz $R/$s.$w.jsonl $Q/data/$DATA/baseline.json --label $NAME-$w-$s --json $R/$s.$w.metrics.json >> $R/eval.log 2>&1
  done
done
cp $R/test.final.metrics.json $R/test.metrics.json
echo "$(date -Is) finished $NAME" >> $R/train.out
