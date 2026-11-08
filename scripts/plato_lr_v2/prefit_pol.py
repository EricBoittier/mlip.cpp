#!/usr/bin/env python3
"""Prefit of the lrc induced-dipole polarisabilities (R3 of the ladder): additive isotropic atomic polarisabilities per
element (A^3) with sum_i alpha_Z(i) = tr(alpha_DFT) / 3 of each monomer (DFT CPHF, merge-aug set A monomers, train side
only), non-negative least squares on relative errors. Reports the held-out (set A val) relative error.
    python prefit_pol.py MONOMERS.jsonl SPLIT.json OUT.json"""
import json, sys
import numpy as np
from scipy.optimize import nnls
from ase import Atoms
BOHR3 = 0.529177210903 ** 3
mons, split, out = sys.argv[1:4]
side = {}
for p in json.load(open(split))["pairs"]:
    for s in p.get("scan_ids") or [p["pair_id"]]:
        side[str(s)] = p["side"]
rows = {"train": [], "val": []}
for l in open(mons):
    m = json.loads(l)
    if not m.get("conv", True) or m.get("alpha_au") is None:
        continue
    s = side.get(m["id"], "train_aug" if m["id"].startswith("A-train") else "heldout")
    rows["train" if s == "train_aug" else "val"].append((Atoms(m["sym"]).numbers, np.trace(np.array(m["alpha_au"])) / 3 * BOHR3))
Zs = sorted({int(z) for Z, _ in rows["train"] for z in Z})
zi = {z: k for k, z in enumerate(Zs)}
def X(r):
    A = np.zeros((len(r), len(Zs))); y = np.array([a for _, a in r])
    for k, (Z, _) in enumerate(r):
        for z in Z:
            if int(z) in zi: A[k, zi[int(z)]] += 1
    return A, y
A, y = X(rows["train"])
w = 1 / np.maximum(y, 0.5)
a, _ = nnls(A * w[:, None], y * w)
Av, yv = X(rows["val"])
rel = lambda A, y: float(np.median(np.abs(A @ a - y) / np.maximum(y, 0.5)))
rep = {"n_train": len(y), "n_val": len(yv), "median_rel_err_train": rel(A, y), "median_rel_err_val": rel(Av, yv),
       "alpha_A3": {int(z): round(float(a[zi[z]]), 4) for z in Zs}}
json.dump({"pol": {"Z": Zs, "alpha": a.tolist()}, "report": rep}, open(out, "w"), indent=1)
print(json.dumps(rep, indent=1))
