#!/usr/bin/env python3
"""Offline signal check of the Fukui arms (fit_fukui.py fits): leak / fragment sums on the held-out compare set
(pre-registered primary metric), DFT Fukui labels on ipea (secondary), guard rails; paired differences vs the E
control with bootstrap 95 % CIs (scans for the compare set, parents for ipea).
    python analyze.py OUT.json NAME=PREFIX [NAME=PREFIX ...]   (PREFIX_pred.pkl, PREFIX.json; first E arm = control
    'E'; NAME=P1,P2,... pools the held-out folds of CV fits)"""
import json, pickle, sys
import numpy as np

rng = np.random.default_rng(2611)
B = 2000
out = sys.argv[1]
arms = {}
for a in sys.argv[2:]:
    name, pre = a.split("=")
    parts = pre.split(",")
    preds = [pickle.load(open(p + "_pred.pkl", "rb")) for p in parts]
    js = [json.load(open(p + ".json")) for p in parts]
    fk = [x for p in preds for x in (p["fukui_test"] if len(parts) > 1 else p["fukui_all"])]
    arms[name] = dict(points=preds[0]["points"], fukui=fk, json=js[0])


def sstep(t):
    t = np.clip(t, 0, 1)
    return t ** 3 * (t * (t * 6 - 15) + 10)


def project(Qf, Ff, w=0.6):  # hard separated clusters, lrc_qproj (sum preserved)
    k = np.floor(Qf); u = (Qf - k - 0.5) / w + 0.5
    P = k + sstep(u); dP = np.where((u > 0) & (u < 1), 30 * u * u * (u - 1) ** 2, 0) / w
    s = P - Qf; g = Ff * (1e-3 + dP)
    return Qf + s - s.sum() * g / g.sum()


def table(pts, post):
    rows = []
    for p in pts:
        q = np.array(p["q"]); dft = np.array(p["dft"])
        if post and p["ncl"] == 2:
            q = project(q, np.array(p["F"]))
        neu = [k for k in (0, 1) if abs(dft[k]) < 0.5]
        rows.append(dict(scan=p["scan"], err=float(np.abs(q - dft).max()),
                         qN=float(abs(q[neu[0]])) if len(neu) == 1 else np.nan, **{k: p[k] for k in ("set", "split", "cls", "d", "Q")}))
    return rows


def select(rows, kind):
    held = lambda r: (r["set"] == "dimers") or (r["set"] == "setA" and r["split"] == "val")
    far = lambda r: 8 - 1e-6 <= r["d"] <= 25 + 1e-6
    if kind == "leak":  # primary: Q != 0, ion + polar / ion + nonpolar / like charge, 8-25 A, held out
        return [i for i, r in enumerate(rows) if held(r) and far(r) and r["Q"] != 0 and r["cls"] in ("ion + polar", "ion + nonpolar", "like charge")]
    if kind == "neutral":
        return [i for i, r in enumerate(rows) if held(r) and far(r) and r["Q"] != 0 and r["cls"] in ("ion + polar", "ion + nonpolar")]
    if kind == "ionpair0":
        return [i for i, r in enumerate(rows) if r["set"] == "dimers" and far(r) and r["Q"] == 0 and r["cls"] == "ion pair"]
    if kind == "ct":
        return [i for i, r in enumerate(rows) if "ct" in r["split"] or "ct ambiguous" in r["cls"]]
    if kind == "near":  # 4-8 A held out, every class
        return [i for i, r in enumerate(rows) if held(r) and r["d"] < 8 - 1e-6]
    raise ValueError(kind)


def boot_mean(vals, groups):
    """mean and 95 % CI, resampling groups"""
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    m = float(np.nanmean(vals))
    bs = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug))
        bs.append(np.nanmean(np.concatenate([vals[idx[g]] for g in pick])))
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


res = {"n_boot": B, "metrics": {}}
ctl = "E"
for post in (False, True):
    tabs = {n: table(a["points"], post) for n, a in arms.items()}
    for kind, col in (("leak", "err"), ("neutral", "qN"), ("ionpair0", "err"), ("ct", "err"), ("near", "err")):
        sel = select(tabs[ctl], kind)
        key = f"{kind}{'_post' if post else ''}"
        res["metrics"][key] = {}
        for n in arms:
            v = np.array([tabs[n][i][col] for i in sel]); vc = np.array([tabs[ctl][i][col] for i in sel])
            g = [tabs[ctl][i]["scan"] for i in sel]
            m = boot_mean(v, g)
            dlt = boot_mean(v - vc, g) if n != ctl else (0.0, 0.0, 0.0)
            res["metrics"][key][n] = dict(n=len(sel), mean=m[0], ci=m[1:], d_vs_E=dlt[0], d_ci=dlt[1:])
# Fukui labels (per parent, per kind)
res["metrics"]["fukui"] = {}
for kind in ("all", "f-", "f+"):
    per = {}
    for n, a in arms.items():
        per[n] = {(f, k): float(np.mean(np.abs(m - d))) for f, k, m, d in a["fukui"] if kind == "all" or k == kind}
    keys = sorted(set.intersection(*[set(v) for v in per.values()]))
    g = [k[0] for k in keys]
    res["metrics"]["fukui"][kind] = {}
    for n in arms:
        v = np.array([per[n][k] for k in keys]); vc = np.array([per[ctl][k] for k in keys])
        m = boot_mean(v, g); dlt = boot_mean(v - vc, g) if n != ctl else (0.0, 0.0, 0.0)
        res["metrics"]["fukui"][kind][n] = dict(n=len(keys), mean=m[0], ci=m[1:], d_vs_E=dlt[0], d_ci=dlt[1:])
res["guard"] = {n: {"omol_val_atom_mae": a["json"]["omol_val"]["atom_mae"], "mono_val_dipole_mae_au": a["json"]["mono_val"]["dipole_mae_au"],
                    "mono_val_atom_mae": a["json"]["mono_val"]["atom_mae"]} for n, a in arms.items()}
json.dump(res, open(out, "w"), indent=1)
for key, d in res["metrics"].items():
    if key == "fukui":
        for kind, dd in d.items():
            print(f"fukui {kind}:", " | ".join(f"{n} {v['mean']:.4f} (d {v['d_vs_E']:+.4f} [{v['d_ci'][0]:+.4f},{v['d_ci'][1]:+.4f}])" for n, v in dd.items()), f"n={next(iter(dd.values()))['n']}")
        continue
    print(f"{key}:", " | ".join(f"{n} {v['mean']:.4f} (d {v['d_vs_E']:+.4f} [{v['d_ci'][0]:+.4f},{v['d_ci'][1]:+.4f}])" for n, v in d.items()), f"n={next(iter(d.values()))['n']}")
for n, v in res["guard"].items():
    print("guard", n, {k: round(x, 4) for k, x in v.items()})
