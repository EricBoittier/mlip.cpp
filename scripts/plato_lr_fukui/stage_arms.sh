#!/bin/bash
# Stage the Fukui-arm screens (run on the clariden login node; no GPU): charge / Fukui passes (rsynced to
# $B/models/fukui/<pass>/) -> arm GGUFs = FTq13.gguf (the C8q0 start) with set_fukui_keys (absolute pass paths), .data = FTq13.data.
set -e
B=/capstor/scratch/cscs/boittier/agents/plato-v2; M=$B/models/ladder; F=$B/models/fukui; PY=/users/boittier/venvs/agents-torch/bin/python
S=$B/src6/scripts/plato_lr_fukui/set_fukui_keys.py
arm() { name=$1; pass=$2; mode=$3; scope=$4; d=$F/$pass
  ex=""; [ -e $d/f.gguf ] && ex="--fmodel $d/f.gguf"; [ -e $d/f_minus.gguf ] && ex="$ex --fmodel-minus $d/f_minus.gguf"
  $PY $S $M/FTq13.gguf $M/$name.gguf --fukui $d/fukui.json --mode $mode --scope $scope --qmodel $d/q.gguf $ex
  cp $M/FTq13.data $M/$name.data; }
arm FKE13 element element global      # Ectl (refit element Fukui, same pipeline as the other arms)
arm FKE14 element element global      # Ectl' (data-order seed 14: noise floor)
arm FKU13 uniform uniform global      # U
arm FKA13 atom atom global            # A
arm FKA14 atom atom global            # A, data-order seed 14 (paired with FKE14)
arm FKApm13 atom_pm atom_pm global    # A+-
# AC not screened (stage-1 per-atom charges 2x worse than A; SUMMARY.md)
# EC (element pass, cluster scope) dropped: offline 0.74 e fragment error (q_raw cluster sums of a globally trained pass are not near-integer)
[ -d $F/atom_fl ] && arm FKAfl13 atom_fl atom global   # A with Fukui-label supervision (only if the signal check passed it)
ls -la $M/FK*.gguf
