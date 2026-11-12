#!/bin/bash
# CUDA parity at the precise-MUL_MAT K=1036 DEFAULT (cuda_default.sbatch -> kuma cuda/out/{pbc,lr}/cuda_mmdef): every
# omol-4m checkpoint on bulk / slab (60) and ion / cluster / mol vs his fp32 runs. Run from
# ~/data/plato-lr-scratch/pet_lr_port.  usage: mm_tables.sh OUT.txt
set -uo pipefail
S=$(cd $(dirname $0) && pwd); PY=${PY:-$HOME/miniforge3/envs/b4ff/bin/python}
OUT=$1; : > $OUT
for g in readout-pol control readout readout-1m readout-2m readout-4m; do
  for s in bulk slab ion cluster mol; do
    R=ref/$g/full_${s}; [ -f ${R}_cpu64.npz ] || continue
    if [ $s = bulk ] || [ $s = slab ]; then j=out/pbc/cuda_mmdef/${g}_lr_nobase_${s}_cuda.json; else j=out/lr/cuda_mmdef/${g}_${s}_cuda.json; fi
    [ -f $j ] || continue
    echo "== $g $s cuda (K=1036 precise by default)" >> $OUT
    $PY $S/lr_parity.py $j ${R}_cpu64.npz $(ls ${R}_*32*.npz) --label "$g $s cuda_mmdef" 2>/dev/null | grep -E "^  (E|F|W|q|mu|metallicity) " >> $OUT
  done
done
echo "wrote $OUT"
