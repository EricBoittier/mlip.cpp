#!/bin/bash
# Periodic parity tables (after M. Langer, ipsum/PET-LR): mlip.cpp (out/pbc/{cpu,cuda}/<gguf>_<fam>[_cuda].json, first
# 60 frames) against every fp32 run of his in ref/<ckpt>/full_<fam>_*32*.npz (seeded: ref/seeded/<preset>_<fam>_*32*),
# vs his fp64 -> out/pbc/parity_<gguf>_<fam>_<backend>.txt. Run from ~/data/plato-lr-scratch/pet_lr_port.
S=$(dirname $0); PY=${PY:-$HOME/miniforge3/envs/b4ff/bin/python}
for g in readout-pol control readout readout-1m readout-2m readout-4m seeded_lr_species seeded_lr_both seeded_lr_fermi_cold; do
  for s in bulk slab; do
    ov=""
    if [[ $g == seeded_* ]]; then p=${g#seeded_}; R=ref/seeded/${p}_${s}; gg=$g; ov="--overflow frames/set_${s}_60.extxyz 5.0 59"
    else R=ref/$g/full_${s}; gg=${g}_lr_nobase; fi
    [ -f ${R}_cpu64.npz ] || continue
    runs=$(ls ${R}_*32*.npz 2>/dev/null)
    for b in cpu cuda; do
      j=out/pbc/$b/${gg}_${s}.json; [ $b = cuda ] && j=out/pbc/cuda/${gg}_${s}_cuda.json
      [ -f $j ] && $PY $S/lr_parity.py $j ${R}_cpu64.npz $runs $ov --label "$g $s $b" 2>/dev/null | grep -v "dipole" > out/pbc/parity_${g}_${s}_${b}.txt
    done
  done
done
grep -h "ABOVE\|^==" out/pbc/parity_*.txt | cut -c1-170
