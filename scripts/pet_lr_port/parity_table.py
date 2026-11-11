#!/usr/bin/env python3
"""Phase-1 parity table from the dumps: per checkpoint x family, max over frames of
  E   |dE| / sum_i |E_i|   (oracle per-atom trunk energies; well conditioned on baseline-free totals)
  F   max|dF| / max|F|     per frame
  h   max|dh| / max|h|     per frame (node features)
  V   symmetric virial, max|dV| / max|V| per periodic frame
for mlip.cpp fp32 (CPU backend) vs JAX CPU fp64, mlip.cpp vs JAX CPU fp32, and his floors: JAX CPU fp32 vs fp64,
JAX GPU fp32 vs fp64, JAX GPU fp32 vs CPU fp32.  Verdicts: cpp vs fp64 <= max(1e-5, his cpu32 and gpu32 vs fp64); cpp vs cpu32 <= max(1e-5, his gpu32 vs cpu32)
("no floor" where no GPU dump exists). Floors are single runs.

    python parity_table.py ROOT  (ROOT = ~/data/plato-lr-scratch/pet_lr_port; prints markdown)
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from compare_trunk import volumes
from compare_feats import read_bin

R = Path(sys.argv[1])


def ref(p):
    z = np.load(p); a = z["atom_ptr"]
    return [dict(e=z["energy"][a[k]:a[k+1]].astype(float), F=z["forces"][a[k]:a[k+1]].astype(float),
                 h=z["node"][a[k]:a[k+1]].astype(float), V=z["virial"][k].astype(float)) for k in range(len(a)-1)]


def cpp(js, feats, frames):
    vols = volumes(frames); hs = read_bin(feats); out = []
    for k, f in enumerate(json.load(open(js))["frames"]):
        V = np.full((3, 3), np.nan)
        if "stress" in f and np.isfinite(vols[k]):
            s = np.asarray(f["stress"], float); V = np.array([[s[0],s[5],s[4]],[s[5],s[1],s[3]],[s[4],s[3],s[2]]]) * vols[k]
        out.append(dict(E=float(f["energy"]), F=np.asarray(f["forces"], float).reshape(-1, 3), h=hs[k], V=V))
    return out


def m(x, y):
    E, F, H, V = [], [], [], []
    for a, b in zip(x, y):
        Ea = a["E"] if "E" in a else a["e"].sum()
        E.append(abs(Ea - b["e"].sum()) / np.abs(b["e"]).sum())
        F.append(np.abs(a["F"] - b["F"]).max() / np.abs(b["F"]).max())
        H.append(np.abs(a["h"] - b["h"]).max() / np.abs(b["h"]).max())
        if np.isfinite(a["V"]).all() and np.isfinite(b["V"]).all():
            sa, sb = (a["V"] + a["V"].T) / 2, (b["V"] + b["V"].T) / 2
            V.append(np.abs(sa - sb).max() / np.abs(sb).max())
    return [max(E), max(F), max(H), max(V) if V else None]


fmt = lambda v: " / ".join("–" if x is None else f"{x:.1e}" for x in v)
print("| ckpt | family | cpp vs fp64 | cpu32 vs fp64 | gpu32 vs fp64 | cpp vs cpu32 | gpu32 vs cpu32 | vs fp64 | vs fp32 |")
print("|---|---|---|---|---|---|---|---|---|")
for ck, tag in (("readout-pol", ""), ("readout-4m", "_r4m")):
    for s in ("mol", "ion", "cluster", "slab", "bulk"):
        d = R / "ref" / ck
        r64, r32 = ref(d / f"{s}_cpu64.npz"), ref(d / f"{s}_cpu32.npz")
        c = cpp(R / f"out/{s}_trunk{tag}.json", R / f"out/{s}_feats{tag}.bin", R / f"frames/set_{s}_q0.extxyz")
        g = ref(d / f"{s}_gpu32.npz") if (d / f"{s}_gpu32.npz").exists() else None
        c64, f64, c32 = m(c, r64), m(r32, r64), m(c, r32)
        g64, g32 = (m(g, r64), m(g, r32)) if g else ([None] * 4, [None] * 4)
        floor64 = [max(1e-5, *(v for v in (a, b) if v is not None)) if a is not None else None for a, b in zip(f64, g64)]
        def verdict(c, fl):
            if any(f is None for x, f in zip(c, fl) if x is not None):
                return "no floor"
            worst = max(x / f for x, f in zip(c, fl) if x is not None)
            return "PASS" if worst <= 1 else f"EXCEEDS {worst:.2f}x"
        floor32 = [max(1e-5, b) if b is not None else None for b in g32]
        print(f"| {ck} | {s} | {fmt(c64)} | {fmt(f64)} | {fmt(g64)} | {fmt(c32)} | {fmt(g32)} | "
              f"{verdict(c64, floor64)} | {verdict(c32, floor32)} |")
