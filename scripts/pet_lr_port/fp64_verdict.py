#!/usr/bin/env python3
"""Settle an fp32 order-statistic verdict with the fp64 executor: split each fp32 run's error against his JAX fp64 into
an INPUT-ROUNDING part and an ARITHMETIC part, using X_f = mlip.cpp's fp64 executor on the float32-rounded inputs both
fp32 pipelines see (mlipcpp-eval --fp64-f32-inputs: positions / cell rounded to float, every operation after that and
the outputs in double; MLIPCPP_EXEC=fp64 also rounds the OUTPUTS to float, an upper bound only).

    python fp64_verdict.py REF64.npz XF.json --cpp NAME=CPP.json [...] --his NAME=RUN32.npz [...] [--keys E F q]
                           [--xd XD.json]

Per output key and frame: total = err(run, REF64), arithmetic = err(run, X_f), input = err(X_f, REF64) (and, with
--xd, the executor on double inputs vs REF64: the executor's own floor). Reported: medians / maxima, the paired
P(mlip.cpp worse than his run) on the TOTAL errors (the earlier verdict statistic) and on the ARITHMETIC errors, and
his own pairwise P range on each. Error metrics as lr_parity.py: E |dE| / sum|E_i|, F max|dF| / max|F|, W max|dW| /
max|W|, q max|dq|, mu |dmu|.
"""
import argparse
import itertools
import json

import numpy as np


def load_json(path):
    fr = json.load(open(path))["frames"]
    return fr


def get_cpp(fr, f, ap):
    c = fr[f] if f < len(fr) else {"error": "missing"}
    if "error" in c:
        return None
    d = {"E": c["energy"], "F": np.reshape(c.get("forces", []), (-1, 3)),
         "W": np.reshape(c["virial"], (3, 3)) if "virial" in c else None,
         "q": np.asarray(c["charges"]) if "charges" in c else None,
         "mu": c.get("fermi_level")}
    return d


def get_npz(z, f, ap):
    A = slice(ap[f], ap[f + 1])
    e = z["atom/energy"][A] if "atom/energy" in z.files else z["energy"][A]
    if not np.all(np.isfinite(e)):
        return None
    d = {"E": float(e.sum()), "F": z["forces"][A], "W": z["virial"][f] if "virial" in z.files else None,
         "q": z["atom/charges"][A] if "atom/charges" in z.files else None,
         "mu": float(z["int/fermi_level"][A][0]) if "int/fermi_level" in z.files else None, "Eabs": np.abs(e).sum()}
    return d


def err(a, b, key, ref):
    if a is None or b is None:
        return np.nan
    if key == "E":
        return abs(a["E"] - b["E"]) / ref["Eabs"]
    x, y = a.get(key), b.get(key)
    if x is None or y is None or (hasattr(x, "size") and np.size(x) == 0):
        return np.nan
    if key == "F":
        return np.abs(np.asarray(x) - np.asarray(y)).max() / np.abs(ref["F"]).max()
    if key == "W":
        if ref["W"] is None or not np.all(np.isfinite(ref["W"])):
            return np.nan
        return np.abs(np.asarray(x) - np.asarray(y)).max() / np.abs(ref["W"]).max()
    if key == "q":
        return np.abs(np.asarray(x) - np.asarray(y)).max()
    if key == "mu":
        return abs(x - y)
    return np.nan


def paired(a, b):
    """P(a worse than b), ties counted as 1/2 (bitwise-equal runs give 0.5, not 0)"""
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return np.nan, 0
    return float(np.mean(a[m] > b[m]) + 0.5 * np.mean(a[m] == b[m])), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("xf")
    ap.add_argument("--xd")
    ap.add_argument("--cpp", nargs="*", default=[])
    ap.add_argument("--his", nargs="*", default=[])
    ap.add_argument("--keys", nargs="*", default=["E", "F", "W", "q", "mu"])
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    z = np.load(a.ref)
    apr = z["atom_ptr"]
    nf = len(apr) - 1
    xf = load_json(a.xf)
    xd = load_json(a.xd) if a.xd else None
    nf = min(nf, len(xf))
    cpps = {k: load_json(v) for k, v in (s.split("=", 1) for s in a.cpp)}
    his = {k: np.load(v) for k, v in (s.split("=", 1) for s in a.his)}
    print(f"== {a.label} n={nf} ref {a.ref}")
    for key in a.keys:
        rows = {}
        tot, ari = {}, {}
        inp = np.array([err(get_cpp(xf, f, apr), get_npz(z, f, apr), key, get_npz(z, f, apr)) for f in range(nf)])
        if np.all(np.isnan(inp)):
            continue
        for name, fr in cpps.items():
            tot[name] = np.array([err(get_cpp(fr, f, apr), get_npz(z, f, apr), key, get_npz(z, f, apr)) for f in range(nf)])
            ari[name] = np.array([err(get_cpp(fr, f, apr), get_cpp(xf, f, apr), key, get_npz(z, f, apr)) for f in range(nf)])
        hn = []
        for name, zz in his.items():
            if len(zz["atom_ptr"]) != len(apr) or not np.array_equal(zz["atom_ptr"][:nf + 1], apr[:nf + 1]):
                continue
            hn.append(name)
            tot[name] = np.array([err(get_npz(zz, f, apr), get_npz(z, f, apr), key, get_npz(z, f, apr)) for f in range(nf)])
            ari[name] = np.array([err(get_npz(zz, f, apr), get_cpp(xf, f, apr), key, get_npz(z, f, apr)) for f in range(nf)])
        fmt = lambda v: f"med {np.nanmedian(v):.2e} max {np.nanmax(v):.2e}" if np.any(np.isfinite(v)) else "-"
        print(f"  [{key}] input rounding (X_f vs fp64): {fmt(inp)}" +
              (f"; executor floor (X_d vs fp64): {fmt(np.array([err(get_cpp(xd, f, apr), get_npz(z, f, apr), key, get_npz(z, f, apr)) for f in range(nf)]))}" if xd else ""))
        for name in list(cpps) + hn:
            print(f"    {name:>14}: total {fmt(tot[name])} | arithmetic {fmt(ari[name])}")
        for c in cpps:
            pt = [f"{h} {paired(tot[c], tot[h])[0]:.2f}" for h in hn]
            pa = [f"{h} {paired(ari[c], ari[h])[0]:.2f}" for h in hn]
            print(f"    P({c} worse)  total: {', '.join(pt)}  |  arithmetic: {', '.join(pa)}")
        # his pairwise range over INDEPENDENT pairs (bitwise-identical runs, e.g. his cpu32 / cpu32_v2 on E, dropped)
        pairs = [(x, y) for x, y in itertools.permutations(hn, 2) if not np.array_equal(tot[x], tot[y], equal_nan=True)]
        if pairs:
            pt = [paired(tot[x], tot[y])[0] for x, y in pairs]
            pa = [paired(ari[x], ari[y])[0] for x, y in pairs]
            print(f"    his pairwise P range ({len(pairs)} independent ordered pairs)  total: {min(pt):.2f}-{max(pt):.2f}  |  "
                  f"arithmetic: {min(pa):.2f}-{max(pa):.2f}")


if __name__ == "__main__":
    main()
