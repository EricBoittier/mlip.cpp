#!/bin/bash
# His threshold reference (feat/threshold-reference 101575d): presets thr / thr_cold_lrt, his oracle refs
# (ref_threshold.sbatch -> kuma ref/seeded_thr/) vs mlip.cpp CPU (out/thr/<p>_<fam>.json) and CUDA
# (out/thr_cuda/<p>_<fam>_cuda.json): branch_to_ref + lr_parity per preset x family.
# Run from ~/data/plato-lr-scratch/pet_lr_port.  usage: thr_tables.sh OUT.txt
set -uo pipefail
S=$(cd $(dirname $0) && pwd); PY=${PY:-$HOME/miniforge3/envs/b4ff/bin/python}
OUT=$1; : > $OUT
for p in thr thr_cold_lrt; do for s in mol_z18 cluster bulk_f32; do
  o=ref/seeded_thr/${p}_${s}; runs=()
  for r in cpu64 cpu32 cpu32_v2 gpu32; do
    [ -f ${o}_$r.npz ] && $PY $S/branch_to_ref.py ${o}_$r.npz ${o}_$r.ref.npz && [ $r != cpu64 ] && runs+=(${o}_$r.ref.npz)
  done
  for side in cpu cuda; do
    j=out/thr/${p}_${s}.json; [ $side = cuda ] && j=out/thr_cuda/${p}_${s}_cuda.json
    [ -f $j ] || continue
    echo "== $p $s $side (his runs: ${#runs[@]})" >> $OUT
    $PY $S/lr_parity.py $j ${o}_cpu64.ref.npz "${runs[@]}" --label "$p $s $side" 2>/dev/null | grep -v "^  JAX\|^  mlip.cpp \|^   " >> $OUT
  done
done; done
echo "wrote $OUT"
