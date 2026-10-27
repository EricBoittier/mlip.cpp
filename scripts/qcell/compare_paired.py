"""Paired differences between two models on the same structures, per stratum, with bootstrap 95% CIs.

    python -I compare_paired.py BASE.npz MODEL.npz OUT.json [--label "B-A"] [--boot 2000] [--seed 0]

Inputs: eval_omol.py --per-structure files of the same split (same order, same strata). Per stratum:
E MAE (meV/atom, mean over structures of |dE|/n), F MAE (meV/A, sum |dF| / 3 atoms), and the within-scan
Delta E MAE (meV, where present) of each model, the difference MODEL - BASE, and a percentile bootstrap
CI of the difference resampling structures (paired: the same resample for both models).
"""
import argparse
import json

import numpy as np


def metrics(d, idx):
    n = d["n"][idx]
    e = 1e3 * np.mean(np.abs(d["dE"][idx]) / n)
    f = 1e3 * d["f_abs"][idx].sum() / (3 * n.sum())
    se = d["scan_err"][idx]
    s = 1e3 * np.nanmean(np.abs(se)) if np.isfinite(se).any() else np.nan
    return np.array([e, f, s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--label", default="model-base"); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    A, B = np.load(a.base), np.load(a.model)
    assert (A["n"] == B["n"]).all() and list(A["strata_names"]) == list(B["strata_names"])
    rng = np.random.default_rng(a.seed)
    res = {"label": a.label, "base": a.base, "model": a.model, "boot": a.boot, "strata": {}}
    names = ["E_mae_mev_atom", "F_mae_mev_a", "scan_dE_mae_mev"]
    for name, m in zip(A["strata_names"], A["strata"]):
        idx = np.nonzero(m)[0]
        if len(idx) == 0:
            continue
        ma, mb = metrics(A, idx), metrics(B, idx)
        bs = np.array([metrics(B, j) - metrics(A, j) for j in (rng.choice(idx, len(idx)) for _ in range(a.boot))])
        lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
        res["strata"][str(name)] = {"structures": int(len(idx)), **{
            k: {"base": float(ma[i]), "model": float(mb[i]), "diff": float(mb[i] - ma[i]), "ci95": [float(lo[i]), float(hi[i])]}
            for i, k in enumerate(names) if np.isfinite(ma[i])}}
    json.dump(res, open(a.out, "w"), indent=1)
    print("==", a.label)
    for name, v in res["strata"].items():
        parts = [f"{k.split('_')[0]} {x['base']:.2f}->{x['model']:.2f} d {x['diff']:+.2f} [{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]"
                 for k, x in v.items() if isinstance(x, dict)]
        print(f"  {name:36s} {v['structures']:6d}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
