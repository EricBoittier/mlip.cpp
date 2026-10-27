"""Does composition determine the total charge in QCell (train split)? And the b0 residual by subset / charge.

    python -I charge_check.py TRAIN.npz BASELINE_B0.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

d = np.load(sys.argv[1])
b = {int(k): v for k, v in json.load(open(sys.argv[2]))["b0"].items()}
n, Z, E, q, sub = d["n"], d["Z"], d["E"], d["charge"], d["subset"]
off = np.concatenate([[0], np.cumsum(n)])
bz = np.zeros(119)
for z, v in b.items():
    bz[z] = v
res = np.array([E[k] - bz[Z[off[k]:off[k + 1]]].sum() for k in range(len(n))]) / n
names = ["nucleic_acids", "lipids", "sugars", "ions_water", "dimers"]
for s in range(5):
    m = sub == s
    print(f"{names[s]}: per-atom residual after b0: RMS {np.sqrt(np.mean(res[m]**2)):.3f} eV, mean {res[m].mean():.3f}")
for c in np.unique(q):
    m = q == c
    print(f"charge {c:+.0f}: {m.sum()} structures, per-atom residual mean {res[m].mean():.3f} RMS {np.sqrt(np.mean(res[m]**2)):.3f}")
by = defaultdict(set)
cnt = defaultdict(int)
for k in range(len(n)):
    key = tuple(np.bincount(Z[off[k]:off[k + 1]], minlength=55))
    by[key].add(float(q[k]))
    cnt[key] += 1
amb = [k for k, v in by.items() if len(v) > 1]
namb = sum(cnt[k] for k in amb)
print(f"{len(by)} distinct compositions; {len(amb)} occur with more than one total charge, covering {namb} of {len(n)} training structures")
for k in amb[:10]:
    print("  ", {int(z): int(c) for z, c in enumerate(k) if c}, sorted(by[k]))
# electron parity: closed shell needs an even electron count
ne = np.array([Z[off[k]:off[k + 1]].sum() - q[k] for k in range(len(n))])
print("odd electron counts:", int((ne % 2 == 1).sum()))
