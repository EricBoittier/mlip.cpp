#!/usr/bin/env python3
"""Leak and size consistency of the Fukui arms through the mlip.cpp GGUF pipeline (mlipcpp-qpass outputs on the
held-out Q != 0 frames at 8-25 A and their 100 A copies), and the torch-to-mlip.cpp parity of the fragment sums
against the fit_fukui.py predictions of the same arm (same weights, same geometries).
    python leak_check.py OUT.json leak/points.json ARM=leak/q_ARM.json[:fits/PREFIX] ..."""
import json, pickle, sys
import numpy as np

rng = np.random.default_rng(7)
meta = json.load(open(sys.argv[2]))
out = {}
for spec in sys.argv[3:]:
    name, rest = spec.split("=")
    qf, pre = (rest.split(":") + [None])[:2]
    fr = json.load(open(qf))["frames"]
    assert len(fr) == len(meta)
    rows = []
    for m, f in zip(meta, fr):
        g = np.array(m["frag"])
        for key in ("q", "q_proj"):
            if key not in f: continue
        q = np.array(f["q"]); qp = np.array(f.get("q_proj", f["q"]))
        s = [q[g == k].sum() for k in (0, 1)]; sp = [qp[g == k].sum() for k in (0, 1)]
        dft = np.array(m["dft"]); neu = [k for k in (0, 1) if abs(dft[k]) < 0.5]
        rows.append(dict(i=m["i"], d=m["d"], far=m["far"], cls=m["cls"], scan=m["scan"], s=s, err=float(np.abs(np.array(s) - dft).max()),
                         err_p=float(np.abs(np.array(sp) - dft).max()),
                         qN=float(abs(s[neu[0]])) if len(neu) == 1 else np.nan, qN_p=float(abs(sp[neu[0]])) if len(neu) == 1 else np.nan))
    r = {}
    near = [x for x in rows if not x["far"]]
    for col in ("err", "err_p", "qN", "qN_p"):
        v = np.array([x[col] for x in near]); r[f"{col}_8_25_mean"] = float(np.nanmean(v))
        r[f"{col}_8_25_max"] = float(np.nanmax(v)); r[f"{col}_8_25_frac_gt_0p05"] = float(np.nanmean(v[np.isfinite(v)] > 0.05))
    # size consistency: fragment sums at 100 A vs the same frame at 25 A
    at25 = {x["i"]: x for x in near if abs(x["d"] - 25) < 1e-3}
    dd = [max(abs(x["s"][k] - at25[x["i"]]["s"][k]) for k in (0, 1)) for x in rows if x["far"]]
    r["size_consistency_25_vs_100_max_e"] = float(np.max(dd)); r["n_100"] = len(dd)
    v100 = np.array([x["qN"] for x in rows if x["far"]]); r["qN_100_mean"] = float(np.nanmean(v100))
    if pre:
        pts = {p["i"]: p for p in pickle.load(open(pre + "_pred.pkl", "rb"))["points"]}
        dif = [max(abs(x["s"][k] - pts[x["i"]]["q"][k]) for k in (0, 1)) for x in near if x["i"] in pts]
        r["parity_torch_vs_gguf_frag_sum_max_e"] = float(np.max(dif)); r["parity_median_e"] = float(np.median(dif)); r["parity_n"] = len(dif)
    out[name] = r
    print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(out, open(sys.argv[1], "w"), indent=1)
