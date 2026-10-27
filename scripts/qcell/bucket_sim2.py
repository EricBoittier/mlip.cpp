"""bucket_sim.py with --bucket-growth G: distinct (n, E, L, B) keys of random batches.

    python -I bucket_sim2.py SPLIT.npz B G [N]
"""
import math
import sys
from collections import Counter

import numpy as np

d = np.load(sys.argv[1]); B = int(sys.argv[2]); G = float(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
n, R, L = d["n"], d["R"], d["L"]
off = np.concatenate([[0], np.cumsum(n)])
rng = np.random.default_rng(0)
idx = rng.choice(len(n), min(N, len(n)), replace=False)
E = {}; M = {}
for k in idx:
    P = R[off[k]:off[k + 1]]
    c = ((np.linalg.norm(P[:, None] - P[None], axis=-1) < 5.0).sum(1) - 1)
    E[k] = int(c.sum()); M[k] = int(c.max())
ru = lambda x, m: (x + m - 1) // m * m
def rg(x, m):
    v = float(m)
    while ru(int(math.ceil(v)), m) < x: v *= G
    return ru(int(math.ceil(v)), m)
tot = Counter(); waste = []
for box in sorted(set(L[idx])):
    ks = [k for k in idx if L[k] == box]
    keys = Counter()
    for _ in range(3):
        rng.shuffle(ks)
        for lo in range(0, len(ks), B):
            b = ks[lo:lo + B]
            Er = sum(E[k] for k in b); nr = sum(n[k] for k in b); Lm = max(8, ru(1 + max(M[k] for k in b), 8))
            Ep = rg(max(Er, 1), 1024); need = 1 + (Ep - Er + Lm - 2) // (Lm - 1)
            keys[(rg(nr + need, 128), Ep, Lm, len(b), box)] += 1; waste.append(Ep / max(Er, 1))
    tot.update(keys)
    print(f"box {box}: {sum(keys.values())} batches, {len(keys)} distinct keys")
print(f"total distinct keys {len(tot)}, mean edge padding factor {np.mean(waste):.3f}")
