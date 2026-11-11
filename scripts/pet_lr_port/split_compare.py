#!/usr/bin/env python3
"""Compare mlip.cpp force-split dumps (MLIPCPP_FORCE_SPLIT) against cached torch fp64 pieces (force_split.py
--save-torch --torch32) and his fp32 runs: per dump, the total error and the arithmetic-only error (against torch fp64
maths on the same fp32 edge vectors), P(worse) against each of his runs and against each other dump.

    python split_compare.py TORCH.npz REF_DIR_GLOB_PREFIX --dumps NAME=SPLIT.bin ... --refs RUN.npz ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from force_split import read_split, scatter_edges  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("torch_npz")
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--refs", nargs="*", default=[])
    a = ap.parse_args()
    T = np.load(a.torch_npz); nf = int(T["nf"])
    tot, ari = {}, {}
    for spec in a.dumps:
        name, path = spec.split("=", 1)
        d = read_split(path)[:nf]
        tot[name], ari[name] = [], []
        for f, r in enumerate(d):
            F = (scatter_edges(r["nb"], r["pm"] < 0.5, r["gev"].astype(float)) + r["fcut"] - r["gpos"]) * r["scale"]
            Fmax = np.abs(T[f"F_t_{f}"]).max()
            tot[name].append(np.abs(F - T[f"F_t_{f}"]).max() / Fmax)
            ari[name].append(np.abs(F - T[f"F_ein_{f}"] - T[f"F_cut_t_{f}"] - T[f"F_pos_t_{f}"]).max() / Fmax)
    for spec in a.refs:
        z = np.load(spec); ap_ = z["atom_ptr"]; nm = Path(spec).stem
        tot[nm], ari[nm] = [], []
        for f in range(nf):
            Fj = z["forces"][ap_[f]:ap_[f + 1]].astype(float); Fmax = np.abs(T[f"F_t_{f}"]).max()
            tot[nm].append(np.abs(Fj - T[f"F_t_{f}"]).max() / Fmax)
            ari[nm].append(np.abs(Fj - T[f"F_ein_{f}"] - T[f"F_cut_t_{f}"] - T[f"F_pos_t_{f}"]).max() / Fmax)
    t32 = [np.abs(T[f"F_e32p_{f}"] + T[f"F_cut_t_{f}"] + T[f"F_pos_t_{f}"] - T[f"F_ein_{f}"] - T[f"F_cut_t_{f}"]
                  - T[f"F_pos_t_{f}"]).max() / np.abs(T[f"F_t_{f}"]).max() for f in range(nf) if f"F_e32p_{f}" in T]
    if t32:
        ari["torch_fp32"] = t32
    names = list(tot)
    print(f"{nf} frames; error = max|dF| / max|F| vs torch fp64; arithmetic = vs torch fp64 maths on the fp32 edge vectors")
    for n in names:
        print(f"  {n:22s} total med {np.median(tot[n]):.3e} max {np.max(tot[n]):.3e} | arithmetic med "
              f"{np.median(ari[n]):.3e} max {np.max(ari[n]):.3e}")
    if t32:
        print(f"  {'torch_fp32':22s} arithmetic med {np.median(t32):.3e} max {np.max(t32):.3e}")
    print("P(row worse than column), total / arithmetic:")
    cols = list(ari)
    print(" " * 24 + " ".join(f"{c[:13]:>13s}" for c in cols))
    for r in cols:
        cells = []
        for c in cols:
            pt = np.mean(np.array(tot[r]) > np.array(tot[c])) if r in tot and c in tot else np.nan
            pa = np.mean(np.array(ari[r]) > np.array(ari[c]))
            cells.append(f"{pt:5.2f}/{pa:4.2f}" if r != c else f"{'-':>13s}")
        print(f"  {r:22s}" + " ".join(f"{x:>13s}" for x in cells))


if __name__ == "__main__":
    main()
