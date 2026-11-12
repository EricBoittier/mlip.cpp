#!/bin/bash
# His les-water backbone presets (ref_les.sbatch: ref_trunk.py on his branch, kuma ref/seeded_les/) vs mlip.cpp CPU
# (out/les/<preset>_<fam>.json) and CUDA (out/les_cuda/<preset>_<fam>_cuda.json): sync, export (trunk-only GGUFs
# gguf/les/), evaluate on CPU, convert (trunk_to_ref.py), lr_parity per preset x family.
# Run from ~/data/plato-lr-scratch/pet_lr_port.  usage: les_tables.sh OUT.txt "PRESETS" "FAMS"
set -uo pipefail
S=$(cd $(dirname $0) && pwd); PY=${PY:-$HOME/miniforge3/envs/b4ff/bin/python}
E=${E:-$HOME/src/mlip.cpp-petlrport/build-cpu/bin/mlipcpp-eval}
OUT=$1; PRE=$2; FAMS=$3
K=boittier@kuma.hpc.epfl.ch:/scratch/boittier/pet_lr_port
mkdir -p ref/seeded_les ckpt/seeded_les gguf/les out/les
rsync -a $K/ref/seeded_les/ ref/seeded_les/ 2>/dev/null; rsync -a $K/seeded_les/ ckpt/seeded_les/ 2>/dev/null
rsync -a "$K/params/seeded_les_*.npz" ckpt/ 2>/dev/null
: > $OUT
for p in $PRE; do
  [ -f gguf/les/$p.gguf ] || $PY $S/iris_to_gguf.py ckpt/seeded_$p.npz ckpt/seeded_les/$p --no-baseline -o gguf/les/$p.gguf > out/les/$p.export.log 2>&1
  for s in $FAMS; do
    o=ref/seeded_les/${p}_${s}
    [ -f ${o}_cpu64.npz ] || { echo "== $p $s: no fp64 reference" >> $OUT; continue; }
    runs=()
    for r in cpu64 cpu32 cpu32_v2 gpu32; do
      [ -f ${o}_$r.npz ] && $PY $S/trunk_to_ref.py ${o}_$r.npz ${o}_$r.ref.npz && [ $r != cpu64 ] && runs+=(${o}_$r.ref.npz)
    done
    [ -f out/les/${p}_${s}.json ] || $E gguf/les/$p.gguf frames/set_$s.extxyz out/les/${p}_${s}.json > /dev/null 2>&1
    for side in cpu cuda; do
      j=out/les/${p}_${s}.json; [ $side = cuda ] && j=out/les_cuda/${p}_${s}_cuda.json
      [ -f $j ] || continue
      echo "== $p $s $side (his runs: ${#runs[@]})" >> $OUT
      $PY $S/lr_parity.py $j ${o}_cpu64.ref.npz "${runs[@]}" --label "$p $s $side" 2>/dev/null >> $OUT
    done
  done
done
echo "wrote $OUT"
