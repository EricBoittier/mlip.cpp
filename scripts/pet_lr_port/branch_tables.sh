#!/bin/bash
# His branch presets (ref_branch.py oracle, kuma ref/seeded_branch/) vs mlip.cpp CPU (out/branch*/<preset>_<fam>.json)
# and CUDA (out/branch_cuda/<preset>_<fam>_cuda.json): sync, convert (branch_to_ref.py), lr_parity per preset x family.
# Run from ~/data/plato-lr-scratch/pet_lr_port.  usage: [REF=seeded_cc CUDA_DIR=out/cc_cuda] branch_tables.sh OUT.txt
# "PRESETS" "FAMS" [CPU_DIR]   (REF: the ref/ subfolder, default seeded_branch)
set -uo pipefail
S=$(cd $(dirname $0) && pwd); PY=${PY:-$HOME/miniforge3/envs/b4ff/bin/python}
OUT=$1; PRE=$2; FAMS=$3; CPU=${4:-out/branch}; R=${REF:-seeded_branch}; CU=${CUDA_DIR:-out/branch_cuda}
mkdir -p ref/$R
rsync -a --include='*.npz' --exclude='*' boittier@kuma.hpc.epfl.ch:/scratch/boittier/pet_lr_port/ref/$R/ ref/$R/ 2>/dev/null
: > $OUT
for p in $PRE; do for s in $FAMS; do
  o=ref/$R/${p}_${s}
  [ -f ${o}_cpu64.npz ] || { echo "== $p $s: no fp64 reference" >> $OUT; continue; }
  runs=()
  for r in cpu64 cpu32 cpu32_v2 gpu32; do
    [ -f ${o}_$r.npz ] && $PY $S/branch_to_ref.py ${o}_$r.npz ${o}_$r.ref.npz && [ $r != cpu64 ] && runs+=(${o}_$r.ref.npz)
  done
  for side in cpu cuda; do
    j=$CPU/${p}_${s}.json; [ $side = cuda ] && j=$CU/${p}_${s}_cuda.json
    [ -f $j ] || continue
    echo "== $p $s $side (his runs: ${#runs[@]})" >> $OUT
    $PY $S/lr_parity.py $j ${o}_cpu64.ref.npz "${runs[@]}" --label "$p $s $side" >> $OUT 2>&1
  done
done; done
echo "wrote $OUT"
