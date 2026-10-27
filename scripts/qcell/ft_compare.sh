#!/usr/bin/env bash
# Host side of ft_eval.sh: strata, paired differences (B-A, C-A, A-Z, B-Z), latent-charge correlations.
set -uo pipefail
O=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell; S=$Q/scripts
PY=/users/boittier/venvs/agents-torch/bin/python
for X in Z A B C; do
  R=$O/ft-eval/$X; [ -s $R/val_sel.json ] || continue
  $PY -I $S/eval_omol.py eval $O/sel/omol25_val_sel.npz $R/val_sel.json $R/val_sel.metrics.json --refit $O/sel/omol_atom_refs.json \
    --label "$X OMol25 val selection" --per-structure $R/val_sel.per.npz > /dev/null
  # QCell curves: PBE0+MBD references, a different functional from OMol25's; only within-scan Delta E is meaningful
  $PY -I $S/eval_omol.py eval $Q/data-omol/split/curves.npz $R/qcell_curves.json $R/qcell_curves.metrics.json --refit $O/sel/omol_atom_refs.json \
    --label "$X QCell curves (different functional: PBE0+MBD)" --per-structure $R/qcell_curves.per.npz > /dev/null
  if [ $X = B ] || [ $X = C ]; then
    L=$R/val_sel.json; [ -f $O/ft/$X/latent_charges.npz ] && L=$O/ft/$X/latent_charges.npz
    $PY -I $S/latent_charge_corr.py $O/sel/omol25_val_sel.npz $O/sel/omol25_val_sel_partial_charges.npz $L $R/latent_charge_corr.json \
      > $R/latent_charge_corr.txt 2>&1 || echo "$X: no latent charges in $L (see $R/latent_charge_corr.txt)"
  fi
done
for p in "B A" "C A" "A Z" "B Z" "C Z"; do
  set -- $p
  for s in val_sel qcell_curves; do
    [ -f $O/ft-eval/$1/$s.per.npz ] && [ -f $O/ft-eval/$2/$s.per.npz ] && \
      $PY -I $S/compare_paired.py $O/ft-eval/$2/$s.per.npz $O/ft-eval/$1/$s.per.npz $O/ft-eval/$1-minus-$2.$s.json --label "$1-$2 $s"
  done
done
