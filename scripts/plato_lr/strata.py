#!/usr/bin/env python3
"""E / F MAE of mlipcpp-eval outputs on val_sel per stratum (masks from the data lane's per.npz):
strata.py VAL_SEL.xyz PER.npz NAME=eval.json [NAME=eval.json ...]"""
import json, sys
import numpy as np
xyz, per = sys.argv[1], np.load(sys.argv[2])
names = [str(s) for s in per["strata_names"]]
want = ["all", "separated (>=8 A, charged fragment)", "|charge|>=2"]
# references: E= / e_res and forces from the xyz (minimal parser)
eref, fref, nat = [], [], []
with open(xyz) as f:
    while True:
        line = f.readline()
        if not line: break
        n = int(line); info = f.readline()
        kv = dict(t.split("=", 1) for t in info.replace('pbc="F F F"', "").split() if "=" in t and not t.startswith("Properties"))
        eref.append(float(kv["e_res"]))
        F = np.array([list(map(float, f.readline().split()[4:7])) for _ in range(n)])
        fref.append(F); nat.append(n)
nat = np.array(nat)
print(f"{'model':6s} " + "  ".join(f"{w[:22]:>22s} E / F" for w in want))
res = {}
args = [a for a in sys.argv[3:] if not a.startswith("--paired=")]
paired = [a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--paired=")]
for arg in args:
    name, path = arg.split("=", 1)
    fr = json.load(open(path))["frames"]
    dE = np.array([abs(fr[k]["energy"] - eref[k]) / nat[k] for k in range(len(fr))]) * 1e3
    fab = np.array([np.abs(np.array(fr[k]["forces"]).reshape(-1, 3) - fref[k]).sum() for k in range(len(fr))])
    res[name] = (dE, fab)
    cells = []
    for w in want:
        m = per["strata"][names.index(w)].astype(bool)
        cells.append(f"{dE[m].mean():>22.2f} {1e3 * fab[m].sum() / (3 * nat[m].sum()):.2f}")
    print(f"{name:6s} " + "  ".join(cells))

# paired differences X - REF with a bootstrap 95% CI over frames (E: mean |dE|/atom, F: pooled MAE)
rng = np.random.default_rng(0)
for pr in paired:
    x, ref = pr.split("-")
    for w in want[1:]:
        m = per["strata"][names.index(w)].astype(bool)
        idx = np.nonzero(m)[0]
        eX, eR = res[x][0][idx], res[ref][0][idx]
        fX, fR, n3 = res[x][1][idx], res[ref][1][idx], 3 * nat[idx]
        dE = eX.mean() - eR.mean(); dF = 1e3 * (fX.sum() - fR.sum()) / n3.sum()
        bE, bF = [], []
        for _ in range(2000):
            b = rng.integers(0, len(idx), len(idx))
            bE.append(eX[b].mean() - eR[b].mean()); bF.append(1e3 * (fX[b].sum() - fR[b].sum()) / n3[b].sum())
        qE, qF = np.percentile(bE, [2.5, 97.5]), np.percentile(bF, [2.5, 97.5])
        print(f"{x}-{ref} {w}: E {dE:+.2f} [{qE[0]:+.2f},{qE[1]:+.2f}]  F {dF:+.2f} [{qF[0]:+.2f},{qF[1]:+.2f}]")
