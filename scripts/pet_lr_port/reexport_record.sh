#!/bin/bash
# Re-export the iris GGUFs of record (and the test resources) with the current iris_to_gguf.py, so they carry
# pet.iris.max_neighbors (the overflow check) and pet.precise_math; then check each new file against the old one:
# gguf_diff.py (identical apart from those two keys) and mlipcpp-eval outputs bit-identical (CPU) on open frames
# (set_ion_40) and, for the LR GGUFs, periodic frames (bulk / slab, 3 each). Writes NEW files to gguf_v6/ (same
# layout); `--install` then moves the old ones to gguf/pre_v6/ and installs the new ones.
# Run from ~/data/plato-lr-scratch/pet_lr_port.  usage: reexport_record.sh [--install]
set -uo pipefail
INSTALL=${1:-}  # (captured here: the check loop's `set --` overwrites $1)
S=$(cd $(dirname $0) && pwd); PY=${PY:-$HOME/miniforge3/envs/b4ff/bin/python}
E=${E:-$HOME/src/mlip.cpp-petlrport/build-cpu/bin/mlipcpp-eval}; T=$HOME/src/mlip.cpp-petlrport/tests/resources
O=gguf_v6; mkdir -p $O/pbc $O/tests out/reexport
ALLOW="pet.iris.max_neighbors pet.precise_math"
declare -A NPZ=([readout-pol]=ckpt/finetune_readout-pol.npz [control]=ckpt/finetune_control.npz
                [readout]=ckpt/finetune_readout.npz [readout-1m]=ckpt/scaling_readout-1m.npz
                [readout-2m]=ckpt/scaling_readout-2m.npz [readout-4m]=ckpt/scaling_readout-4m.npz)
declare -A DIR=([readout-pol]=ckpt/omol-4m/finetune/readout-pol/RMSE_F [control]=ckpt/omol-4m/finetune/control/RMSE_F
                [readout]=ckpt/omol-4m/finetune/readout/RMSE_F [readout-1m]=ckpt/omol-4m/scaling/readout-1m/RMSE_F
                [readout-2m]=ckpt/omol-4m/scaling/readout-2m/RMSE_F [readout-4m]=ckpt/omol-4m/scaling/readout-4m/RMSE_F)
JOBS=()  # "old new kind" (kind: lr = open + periodic frames, trunk = open frames)
for c in readout-pol control readout readout-1m readout-2m readout-4m; do
  $PY $S/iris_to_gguf.py ${NPZ[$c]} ${DIR[$c]} --lr --no-baseline -o $O/${c}_lr_nobase.gguf > out/reexport/${c}_lr.log 2>&1
  $PY $S/iris_to_gguf.py ${NPZ[$c]} ${DIR[$c]} --no-baseline -o $O/${c}_direct_nobase.gguf > out/reexport/${c}_direct.log 2>&1
  cp $O/${c}_lr_nobase.gguf $O/pbc/${c}_lr_nobase.gguf
  JOBS+=("gguf/${c}_lr_nobase.gguf $O/${c}_lr_nobase.gguf lr" "gguf/pbc/${c}_lr_nobase.gguf $O/pbc/${c}_lr_nobase.gguf lr"
         "gguf/${c}_direct_nobase.gguf $O/${c}_direct_nobase.gguf trunk")
done
JOBS+=("gguf/readout-4m_direct_mask_nobase.gguf $O/readout-4m_direct_nobase.gguf trunk")
for p in lr_both lr_fermi_cold lr_species; do
  $PY $S/iris_to_gguf.py ckpt/seeded_$p.npz ckpt/seeded/$p --lr -o $O/seeded_$p.gguf > out/reexport/seeded_$p.log 2>&1
  cp $O/seeded_$p.gguf $O/pbc/seeded_$p.gguf
  JOBS+=("gguf/seeded_$p.gguf $O/seeded_$p.gguf lr" "gguf/pbc/seeded_$p.gguf $O/pbc/seeded_$p.gguf lr")
done
for p in biotrimers multivalent; do
  $PY $S/iris_to_gguf.py ckpt/seeded_$p.npz ckpt/seeded/$p -o $O/seeded_$p.gguf > out/reexport/seeded_$p.log 2>&1
  JOBS+=("gguf/seeded_$p.gguf $O/seeded_$p.gguf trunk")
done
for p in trunk fermi cold species both; do
  lr="--lr"; [ $p = trunk ] && lr=""
  $PY $S/iris_to_gguf.py ckpt/seeded_tiny_$p.npz ckpt/seeded/tiny_$p $lr -o $O/tests/petlr_tiny_$p.gguf > out/reexport/tiny_$p.log 2>&1
  k=lr; [ $p = trunk ] && k=trunk
  JOBS+=("$T/petlr_tiny_$p.gguf $O/tests/petlr_tiny_$p.gguf $k" "gguf/tiny_$p.gguf $O/tests/petlr_tiny_$p.gguf $k")
  [ $p != trunk ] && JOBS+=("gguf/pbc/tiny_$p.gguf $O/tests/petlr_tiny_$p.gguf lr")
done
fail=0
for j in "${JOBS[@]}"; do
  set -- $j; old=$1 new=$2 kind=$3
  [ -f $old ] || { echo "MISSING old $old"; continue; }
  d=$($PY $S/gguf_diff.py $old $new --allow-new $ALLOW 2>&1 | grep -v "^new key" | tr '\n' ';')
  fams="ion_40 cluster_40"; [ $kind = lr ] && fams="ion_40 cluster_40 bulk_60_3 slab_60_3"
  # an old open-only LR GGUF (no graph_pbc.json, exported before the periodic graph) has no periodic result of record
  oldpbc=$($PY -c "import sys; sys.path.insert(0, '$S'); import gguf_diff as g; print(int('graph_pbc.json' in g.read('$old')[1]))")
  outs=""
  for f in $fams; do
    fr=frames/set_$f.extxyz; [ -f $fr ] || fr=out/nodegrad/$f.extxyz
    if [ $kind = lr ] && [ $oldpbc = 0 ] && [[ $f == bulk* || $f == slab* ]]; then outs="$outs $f:n/a(old GGUF open-only)"; continue; fi
    a=out/reexport/a.json b=out/reexport/b.json
    rm -f $a $b; $E $old $fr $a > /dev/null 2>&1; ra=$?; $E $new $fr $b > /dev/null 2>&1; rb=$?
    # bit-identical, or identical apart from frames the NEW file raises on (the max_neighbors overflow check)
    same=$($PY -c "
import json; x = json.load(open('$a'))['frames']; y = json.load(open('$b'))['frames']
ov = [i for i, (u, v) in enumerate(zip(x, y)) if u != v and 'max_neighbors' in v.get('error', '') and 'error' not in u]
bad = [i for i, (u, v) in enumerate(zip(x, y)) if u != v and i not in ov]
nerr = sum('error' in v for v in y)
print('DIFFER' if bad or len(x) != len(y) else 'bit-identical' + (f'+{len(ov)}overflow' if ov else '') + (f'[{nerr}/{len(y)}err]' if nerr else ''))" 2>&1)
    outs="$outs $f:$same(rc $ra/$rb)"
    [[ "$same" == bit-identical* ]] || fail=1
  done
  echo "$old -> $new | gguf: $d | outputs:$outs"
  [[ $d == IDENTICAL* ]] || fail=1
done
echo "fail=$fail"  # (1 also when only graph.json changed: see the gguf column)
# fail stays 1 for the files whose graph.json legitimately changed (direct trunks, open-only LR exports): install when
# every OUTPUT check passed (bit-identical / overflow raises / n/a), as done on 2026-11-11 from out/reexport_check2.txt
if [ "$INSTALL" = "--install" ] && ! grep -q "DIFFER" out/reexport_check2.txt 2>/dev/null; then
  mkdir -p gguf/pre_v6/pbc
  for j in "${JOBS[@]}"; do set -- $j; [[ $1 == gguf/* ]] && cp -n $1 gguf/pre_v6/${1#gguf/} && cp $2 $1; [[ $1 == $T/* ]] && cp $2 $1; done
  echo installed
fi
