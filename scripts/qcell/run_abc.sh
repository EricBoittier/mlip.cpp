#!/usr/bin/env bash
# Inside the holder (GPU 3, qm9esp-cuda): wait for ft/{A,B,C}/selection.json (abc_post.sh), then evaluate
# Z (zero-shot) and model.gguf of A, B, C with the code lane's mlipcpp-eval (MLIPCPP_PAD_EXACT=1); then, if
# before LAST_DEADLINE, last.gguf into ft-eval/{A,B,C}-last. Hard stop at STOP (holder ends 18:39).
set -uo pipefail
O=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell
export EVAL_BIN=/capstor/scratch/cscs/boittier/agents/petlr/abc/bin/mlipcpp-eval MLIPCPP_PAD_EXACT=1 GPU=3
STOP=$(date -d "18:25" +%s); LAST_DEADLINE=$(date -d "17:55" +%s)
until [ -f $O/ft/A/selection.json ] && [ -f $O/ft/B/selection.json ] && [ -f $O/ft/C/selection.json ]; do
  [ $(date +%s) -ge $STOP ] && { echo "no selection.json by stop time"; exit 2; }; sleep 30; done
echo "selection.json present $(date -Is)"
rm -rf $O/ft-eval/{Z,A,B,C}
timeout $(( STOP - $(date +%s) )) bash $Q/scripts/ft_eval.sh model.gguf
echo "model.gguf evaluated $(date -Is)"
if [ $(date +%s) -lt $LAST_DEADLINE ]; then
  for X in A B C; do
    [ -f $O/ft/$X/last.gguf ] || continue
    R=$O/ft-eval/$X-last; mkdir -p $R
    timeout $(( STOP - $(date +%s) )) $EVAL_BIN $O/ft/$X/last.gguf $O/xyz/val_sel.xyz $R/val_sel.json --backend cuda > $R/val_sel.log 2>&1
    timeout $(( STOP - $(date +%s) )) $EVAL_BIN $O/ft/$X/last.gguf $Q/data-omol/xyz/curves.xyz $R/qcell_curves.json --backend cuda > $R/qcell_curves.log 2>&1
  done
  echo "last.gguf evaluated $(date -Is)"
fi
echo ALL_DONE
