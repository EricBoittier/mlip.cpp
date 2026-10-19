#!/usr/bin/env python3
"""Compare prediction files written by mlipcpp-predict / predict_metatomic.py (and, for the
energies and per-atom force errors, mlipcpp-train's --dump-errors) on the same structures.

    python compare_predictions.py <split dir> ids.npy ref=PREFIX name=PREFIX ... [--errors DUMP.test]
"""
import argparse

import numpy as np


def load(prefix, n_atoms):
    e = np.fromfile(f"{prefix}.energy.bin", np.float64)
    raw = open(f"{prefix}.nc_forces.bin", "rb").read()
    f = np.frombuffer(raw, np.float64 if len(raw) == 24 * n_atoms else np.float32).reshape(-1, 3).astype(np.float64)
    return e, f


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("split"); ap.add_argument("ids"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--errors", help="mlipcpp-train --dump-errors prefix.split (e.g. errors.test)")
    a = ap.parse_args()
    d = a.split
    na = np.load(f"{d}/na.npy"); ids = np.load(a.ids)
    n = np.array([na[s + 1] - na[s] for s in ids]); N = int(n.sum())
    E = np.fromfile(f"{d}/energy.bin", np.float32)[ids].astype(np.float64)
    F = np.concatenate([np.memmap(f"{d}/non_conservative_force.bin", np.float32, "r").reshape(-1, 3)[na[s]:na[s + 1]]
                        for s in ids]).astype(np.float64)
    runs = dict(r.split("=", 1) for r in a.runs)
    preds = {k: load(v, N) for k, v in runs.items()}
    ref = next(iter(preds))
    e0, f0 = preds[ref]
    print(f"{len(ids)} structures, {N} atoms; reference: {ref}")
    print(f"{'run':>16} | {'max |dE|/atom':>14} {'rms dE/atom':>12} | {'max |dF|':>10} {'rms dF':>10} {'max rel dF':>10}"
          f" | vs DFT: E MAE meV/atom, F RMSE meV/A")
    for k, (e, f) in preds.items():
        de = (e - e0) / n; df = f - f0
        rel = np.abs(df).max() / np.abs(f0).max()
        print(f"{k:>16} | {np.abs(de).max():14.3e} {np.sqrt((de ** 2).mean()):12.3e} | {np.abs(df).max():10.3e}"
              f" {np.sqrt((df ** 2).mean()):10.3e} {rel:10.3e} | {1e3 * np.abs((e - E) / n).mean():8.3f}"
              f" {1e3 * np.sqrt(((f - F) ** 2).mean()):8.3f}")
    if a.errors:
        dt = [("id", "<i8"), ("step", "<i4"), ("n", "<i4"), ("de", "<f4"), ("fse", "<f4")]
        r = np.fromfile(f"{a.errors}.bin", dtype=dt)
        at = np.fromfile(f"{a.errors}.atoms.bin", np.float32)
        start = np.concatenate([[0], np.cumsum(r["n"])])
        where = {int(i): k for k, i in enumerate(r["id"])}
        de_t = np.array([r["de"][where[s]] for s in ids], np.float64)  # (E_pred - E) / n, eV/atom
        fse_t = np.concatenate([at[start[where[s]]:start[where[s] + 1]] for s in ids]).astype(np.float64)
        print("mlipcpp-train's own errors (CUDA, the training run's evaluation) vs each run:")
        for k, (e, f) in preds.items():
            dde = de_t - (e - E) / n
            fse = ((f - F) ** 2).sum(1)
            print(f"{k:>16} | energy error: max diff {np.abs(dde).max():.3e} eV/atom | per-atom |dF|^2: "
                  f"max diff {np.abs(fse_t - fse).max():.3e}, rms of sqrt diff "
                  f"{np.sqrt(((np.sqrt(fse_t) - np.sqrt(fse)) ** 2).mean()):.3e} eV/A")


if __name__ == "__main__":
    main()
