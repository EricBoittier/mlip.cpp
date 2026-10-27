#!/usr/bin/env python3
"""metrics.py XYZ REF.json NAME=X.json ...: E MAE (meV/atom, vs e_res), F MAE (meV/A, pooled),
their change vs REF (the same model with LES off), and mean |E - E_REF| per atom (meV)."""
import json, sys
import numpy as np
xyz = sys.argv[1]
eref, fref, nat = [], [], []
with open(xyz) as f:
    while True:
        line = f.readline()
        if not line: break
        n = int(line); info = f.readline()
        kv = dict(t.split("=", 1) for t in info.split() if "=" in t and not t.startswith(("Properties", "pbc")))
        eref.append(float(kv["e_res"])); nat.append(n)
        fref.append(np.array([list(map(float, f.readline().split()[4:7])) for _ in range(n)]))
nat = np.array(nat); eref = np.array(eref)
def load(p):
    fr = json.load(open(p))["frames"]
    return np.array([x["energy"] for x in fr]), [np.array(x["forces"]).reshape(-1, 3) for x in fr]
def mae(E, F):
    return 1e3 * np.mean(np.abs(E - eref) / nat), 1e3 * sum(np.abs(a - b).sum() for a, b in zip(F, fref)) / (3 * nat.sum())
E0, F0 = load(sys.argv[2]); m0 = mae(E0, F0)
print(f"{'ref (LES off)':16s} E {m0[0]:8.2f}  F {m0[1]:7.2f}")
for arg in sys.argv[3:]:
    name, p = arg.split("=", 1)
    E, F = load(p); m = mae(E, F)
    print(f"{name:16s} E {m[0]:8.2f} ({m[0]-m0[0]:+8.2f})  F {m[1]:7.2f} ({m[1]-m0[1]:+7.2f})  |E_LES|/atom {1e3*np.mean(np.abs(E-E0)/nat):9.2f}")
