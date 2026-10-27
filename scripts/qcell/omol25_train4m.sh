#!/usr/bin/env bash
# train_4M: wait for the download, check members, extract to train_4M/, scan, select ~100k
# charged / open-shell (seed 0), write arrays + partial charges, memmap (atom-ref energies), verify.
set -euo pipefail
O=/capstor/scratch/cscs/boittier/omol25; S=/capstor/scratch/cscs/boittier/qcell/scripts
PY=/users/boittier/venvs/agents-torch/bin/python; export OMOL_PYLIB=$O/pylib
cd $O
until [ "$(stat -c %s raw/train_4M.tar.gz)" -ge 19983081456 ]; do sleep 20; done; sleep 10
tar tzvf raw/train_4M.tar.gz > meta/train_4M_members.txt
if grep -qE " /|\.\./" meta/train_4M_members.txt || grep -qvE "^[-d]" meta/train_4M_members.txt; then echo unsafe; exit 3; fi
mkdir train_4M && tar xzf raw/train_4M.tar.gz -C train_4M --no-same-owner --no-same-permissions
echo "extracted $(find train_4M -name "*.aselmdb" | wc -l) shards $(date -Is)"
$PY -I $S/prep_omol25_val.py scan $O/train_4M sel/meta_train4m.npz --procs 64
$PY -I $S/prep_omol25_val.py select sel/meta_train4m.npz sel/selection_train4m.json --seed 0 --quota 30000 --sep-quota 5000
$PY -I $S/prep_omol25_val.py write $O/train_4M sel/meta_train4m.npz sel/selection_train4m.json $O/sel/omol25_train4m_sel.npz
$PY -I $S/make_memmap.py sel/omol25_train4m_sel.npz memmap/train4m_sel --baseline sel/omol_atom_refs.json --partial-charges sel/omol25_train4m_sel_partial_charges.npz
$PY -I -c "
import numpy as np, sys
m=memmap/train4m_sel; d=np.load(sel/omol25_train4m_sel.npz)
ns=int(np.load(m+/ns.npy)); na=np.load(m+/na.npy); N=int(na[-1])
assert ns==len(d[n]) and N==d[n].sum()
for f,w in ((x,3),(a,1),(non_conservative_force,3),(mulliken_charge,1),(lowdin_charge,1),(nbo_charge,1)):
    sz=np.fromfile(f{m}/{f}.bin, np.float32 if f!=a else np.int32).size; assert sz==w*N,(f,sz,w*N)
for f in (energy,charge,spin_multiplicity): assert np.fromfile(f{m}/{f}.bin,np.float32).size==ns
x=np.fromfile(m+/x.bin,np.float32).reshape(-1,3); assert np.abs(x-d[R]).max()<1e-3
print(memmap verified, ns, N)
"
echo "done $(date -Is)"
