#!/usr/bin/env python3
"""Readout-init data: N_TR training structures (abc_clean_keepunbound train minus exclude/train.npy) and N_HO held-out
structures (its val split), as extxyz with energy, forces, charge, spin_multiplicity (seed 2611).
    python ro_data.py OUTDIR [N_TR 3000] [N_HO 1000]"""
import sys, os
import numpy as np
from ase import Atoms
from ase.io import write
D = "/capstor/scratch/cscs/boittier/data-audit/abc_clean_keepunbound"
out = sys.argv[1]; ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; nho = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(2611)


def split(name, n, excl):
    d = f"{D}/data/{name}"
    na = np.load(f"{d}/na.npy"); ns = len(na) - 1
    x = np.memmap(f"{d}/x.bin", np.float32, "r").reshape(-1, 3); a = np.memmap(f"{d}/a.bin", np.int32, "r")
    e = np.memmap(f"{d}/energy.bin", np.float32, "r"); f = np.memmap(f"{d}/non_conservative_force.bin", np.float32, "r").reshape(-1, 3)
    q = np.memmap(f"{d}/charge.bin", np.float32, "r"); s = np.memmap(f"{d}/spin_multiplicity.bin", np.float32, "r")
    ok = np.setdiff1d(np.arange(ns), excl)
    idx = np.sort(rng.choice(ok, n, replace=False))
    frames = []
    for k in idx:
        sl = slice(na[k], na[k + 1])
        at = Atoms(numbers=a[sl], positions=x[sl])
        at.info.update(energy=float(e[k]), charge=int(round(q[k])), spin_multiplicity=int(round(s[k])), idx=int(k))
        at.arrays["forces"] = np.array(f[sl], float)
        frames.append(at)
    return frames, idx


tr, itr = split("train", ntr, np.load(f"{D}/exclude/train.npy"))
ho, iho = split("val", nho, np.load(f"{D}/exclude/val.npy"))
write(f"{out}/train.xyz", tr, format="extxyz"); write(f"{out}/heldout.xyz", ho, format="extxyz")
np.save(f"{out}/train_idx.npy", itr); np.save(f"{out}/heldout_idx.npy", iho)
print(f"{out}: {len(tr)} train ({sum(len(t) for t in tr)} atoms), {len(ho)} held-out")
