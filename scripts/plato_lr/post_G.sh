#!/usr/bin/env bash
# Host side of run_G.sbatch (daint login node): strata of every evaluated model in omol25/ft-eval-G, paired
# differences against the frozen control Ectl-s7 (and E-s13 against Ectl-s13, everything against Z), the LES
# decomposition (les_decomp.py) and the training curves (collect_curves.py) for the dashboard.
set -uo pipefail
O=/capstor/scratch/cscs/boittier/omol25; Q=/capstor/scratch/cscs/boittier/qcell; QS=$Q/scripts
D=/capstor/scratch/cscs/boittier/agents/petlr; S=$D/scripts; R=$O/ft-eval-G
PY=/users/boittier/venvs/agents-torch/bin/python
for d in $R/*/; do
  X=$(basename $d)
  for s in val_sel qcell_curves; do
    [ -s $d/$s.json ] || continue
    [ -s $d/$s.per.npz ] && [ $d/$s.per.npz -nt $d/$s.json ] && continue
    split=$O/sel/omol25_val_sel.npz; [ $s = qcell_curves ] && split=$Q/data-omol/split/curves.npz
    $PY -I $QS/eval_omol.py eval $split $d/$s.json $d/$s.metrics.json --refit $O/sel/omol_atom_refs.json \
      --label "$X $s" --per-structure $d/$s.per.npz > /dev/null
  done
done
mkdir -p $R/Z && for f in val_sel.per.npz qcell_curves.per.npz val_sel.metrics.json qcell_curves.metrics.json; do
  [ -f $R/Z/$f ] || cp $O/ft-eval/Z/$f $R/Z/; done
pair() { for s in val_sel qcell_curves; do
  [ -f $R/$1/$s.per.npz ] && [ -f $R/$2/$s.per.npz ] && \
    $PY -I $QS/compare_paired.py $R/$2/$s.per.npz $R/$1/$s.per.npz $R/$1-minus-$2.$s.json --label "$1-$2 $s" > /dev/null
done; }
for X in E-s13 G1-s13 Gs2-s13; do pair $X Ectl-s13; done
for s in 21 42 63 84; do pair E-s$s Ectl-s$s; done
for X in E-s7 G1 G2 G3b G4b Gs1 Gs2 Gs3 Gs4; do pair $X Ectl-s7; done
for X in E-s7 E-s13 Ectl-s7 Ectl-s13 G1 G2 G3b G4b G1-s13 Gs1 Gs2 Gs3 Gs4 Gs2-s13; do pair $X Z; done
for X in E-init E-init-sub E-s7 C; do
  [ -f $R/les-$X-g0/val_sel.per.npz ] && [ -f $R/les-$X-g1/val_sel.per.npz ] && \
    $PY -I $S/les_decomp.py $R/les-$X-g0/val_sel.per.npz $R/les-$X-g1/val_sel.per.npz $R/les-$X.decomp.json --label $X
done
/users/boittier/venvs/mptrj-py/bin/python -I $S/collect_curves.py -o $D/curves_G.json \
  A=$D/abc/A B=$D/abc/B C=$D/abc/C D1=$D/abc/D1 D2=$D/abc/D2 D3=$D/abc/D3 D4=$D/abc/D4 \
  E-s7=$D/E/E-s7 E-s13=$D/E/E-s13 Ectl-s7=$D/E/Ectl-s7 Ectl-s13=$D/E/Ectl-s13 \
  $(for s in 21 42 63 84; do echo -n "E-s$s=$D/E/E-s$s Ectl-s$s=$D/E/Ectl-s$s "; done) \
  G1=$D/G/G1 G2=$D/G/G2 G3b=$D/G/G3b G4b=$D/G/G4b G1-s13=$D/G/G1-s13 Gs2-s13=$D/G/Gs2-s13 Gs1=$D/Gsub/Gs1 Gs2=$D/Gsub/Gs2 Gs3=$D/Gsub/Gs3 Gs4=$D/Gsub/Gs4
