"""Errors of mlipcpp-eval predictions on a prep_qcell_omol.py / prep_omol25_val.py split, by stratum.

    python -I eval_omol.py fit   SPLIT.npz PRED.json SAMPLE_IDX.txt REFIT.json   (PRED on the SAMPLE_IDX structures)
    python -I eval_omol.py eval  SPLIT.npz PRED.json OUT.json [--refit REFIT.json] [--label L] [--sel IDX.txt]

fit: per-element energy correction c = least squares of (E_ref - E_model) on the element counts over
the sampled training structures (a different functional or reference: zero-shot context only).
eval: E_pred = E_model + c.n. Metrics per stratum: E MAE/RMSE (meV/atom, meV/structure), the same
after removing the stratum's mean error (offset-free), F MAE/RMSE per component (meV/A).
Strata: all, subset/domain, total charge (0, +-1, |q|>=2), spin multiplicity, size bins, and, when
the split has them, min intermolecular distance bins (<4, 4-8, >=8 A), dimer class x separation, and
the no-pair-within-5/8 A tags. Curves (scans): Delta E(r) = E(r) - E(r_max) per scan, so per-element
offsets cancel; reported as MAE over all non-reference points and per scan.
"""
import argparse
import json

import numpy as np

KCAL = 43.36410


def counts(Z, n, types):
    sid = np.repeat(np.arange(len(n)), n)
    X = np.zeros((len(n), len(types)))
    np.add.at(X, (sid, np.searchsorted(types, Z)), 1)
    return X


def load_pred(path, n):
    P = json.load(open(path))["frames"]
    assert len(P) == len(n), (len(P), len(n))
    err = [k for k, p in enumerate(P) if "error" in p]
    if err:
        raise SystemExit(f"{len(err)} frames with errors, first {err[0]}: {P[err[0]]['error']}")
    return np.array([p["energy"] for p in P]), [np.asarray(p["forces"], float).reshape(-1, 3) for p in P]


def stats(dE, n, dF):
    if len(dE) == 0:
        return None
    c = dE - dE.mean()
    return {"structures": int(len(dE)), "atoms": int(n.sum()),
            "e_mae_mev_atom": float(1e3 * np.mean(np.abs(dE / n))), "e_rmse_mev_atom": float(1e3 * np.sqrt(np.mean((dE / n) ** 2))),
            "e_mae_mev": float(1e3 * np.mean(np.abs(dE))), "e_rmse_mev": float(1e3 * np.sqrt(np.mean(dE ** 2))),
            "e_mean_err_mev": float(1e3 * dE.mean()), "e_mae_offset_free_mev": float(1e3 * np.mean(np.abs(c))),
            "f_mae_mev_a": float(1e3 * np.mean(np.abs(dF))), "f_rmse_mev_a": float(1e3 * np.sqrt(np.mean(dF ** 2))),
            "f_mae_kcal_mol_a": float(1e3 * np.mean(np.abs(dF)) / KCAL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "eval"])
    ap.add_argument("npz")
    ap.add_argument("pred")
    ap.add_argument("out")
    ap.add_argument("idx", nargs="?")
    ap.add_argument("--refit", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    d = np.load(a.npz)
    n, Z, E, F = d["n"], d["Z"], d["E"], d["F"]
    off = np.concatenate([[0], np.cumsum(n)])
    types = np.unique(Z)
    if a.mode == "fit":
        sel = np.loadtxt(a.out, dtype=int).reshape(-1)  # positional: fit NPZ PRED IDX REFIT
        Em, _ = load_pred(a.pred, n[sel])
        X = counts(np.concatenate([Z[off[k]:off[k + 1]] for k in sel]), n[sel], types)
        y = E[sel] - Em
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ c
        j = {"composition": {str(int(t)): float(v) for t, v in zip(types, c)}, "pred": a.pred, "npz": a.npz,
             "sample": len(sel), "types_missing": [int(t) for t, k in zip(types, (X > 0).sum(0)) if k == 0],
             "residual_per_atom_rms_ev": float(np.sqrt(np.mean((res / n[sel]) ** 2))),
             "note": "E_ref - E_model refit on train-split structures; zero-shot context only"}
        json.dump(j, open(a.idx, "w"), indent=1)
        print(json.dumps({k: v for k, v in j.items() if k != "composition"}))
        return
    Em, Fm = load_pred(a.pred, n)
    b = np.zeros(119)
    if a.refit:
        for z, v in json.load(open(a.refit))["composition"].items():
            b[int(z)] = v
    dE = np.array([Em[k] + b[Z[off[k]:off[k + 1]]].sum() - E[k] for k in range(len(n))])
    dF = [Fm[k] - F[off[k]:off[k + 1]] for k in range(len(n))]
    strata = {"all": np.ones(len(n), bool)}
    keys = d.files
    if "subset" in keys:
        names = ["nucleic_acids", "lipids", "sugars", "ions_water", "dimers"]
        for s in np.unique(d["subset"]):
            strata[f"subset={names[s]}"] = d["subset"] == s
    if "domain" in keys:
        for s in np.unique(d["domain"]):
            strata[f"domain={s}"] = d["domain"] == s
    q = d["charge"]
    strata["charge=0"] = q == 0; strata["|charge|=1"] = np.abs(q) == 1; strata["|charge|>=2"] = np.abs(q) >= 2
    sm = d["spin_multiplicity"]
    for s in np.unique(sm):
        strata[f"spin_multiplicity={int(s)}"] = sm == s
    for lo, hi in ((0, 30), (30, 100), (100, 1000)):
        strata[f"atoms {lo}-{hi}"] = (n >= lo) & (n < hi)
    if "min_inter" in keys:
        mi = d["min_inter"]
        bins = {"sep<4": mi < 4, "sep 4-8": (mi >= 4) & (mi < 8), "sep>=8": (mi >= 8) & np.isfinite(mi)}
        for k, m in bins.items():
            strata[f"multi-component {k}"] = m
        if "dimer_class" in keys:
            dc = d["dimer_class"]
            for c in ("nn", "cn", "cc"):
                for k, m in bins.items():
                    strata[f"dimer {c} {k}"] = (dc == c) & m
        strata["no pair within 5 A"] = d["min_pair"] >= 5
        strata["no pair within 8 A"] = d["min_pair"] >= 8
    r = {"label": a.label, "pred": a.pred, "npz": a.npz, "refit": a.refit, "strata": {}}
    for name, m in strata.items():
        ks = np.nonzero(m)[0]
        st = stats(dE[ks], n[ks], np.concatenate([dF[k].ravel() for k in ks]) if len(ks) else np.zeros(0))
        if st:
            r["strata"][name] = st
    if "scan" in keys and (d["scan"] >= 0).any():
        sc, sr = d["scan"], d["scan_r"]
        per, allerr = {}, []
        for s in np.unique(sc[sc >= 0]):
            ks = np.nonzero(sc == s)[0]
            if len(ks) < 3:
                continue
            ks = ks[np.argsort(sr[ks])]
            ref = ks[-1]
            dd = [(E[k] - E[ref]) - (Em[k] - Em[ref]) for k in ks[:-1]]
            allerr += dd
            per[int(s)] = {"points": int(len(ks)), "r_min": float(sr[ks[0]]), "r_max": float(sr[ref]),
                           "min_inter_max": float(d["min_inter"][ref]), "class": str(d["dimer_class"][ref]),
                           "qA": int(d["qA"][ref]), "qB": int(d["qB"][ref]),
                           "dE_mae_mev": float(1e3 * np.mean(np.abs(dd))),
                           "ref_curve_mev": [float(1e3 * (E[k] - E[ref])) for k in ks],
                           "pred_curve_mev": [float(1e3 * (Em[k] - Em[ref])) for k in ks],
                           "scan_r": [float(sr[k]) for k in ks]}
        if per:
            r["curves"] = {"scans": len(per), "dE_mae_mev": float(1e3 * np.mean(np.abs(allerr))), "per_scan": per}
    json.dump(r, open(a.out, "w"), indent=1)
    show = ("all", "charge=0", "|charge|=1", "|charge|>=2", "dimer cc sep<4", "dimer cc sep 4-8", "dimer cn sep 4-8",
            "dimer cn sep>=8", "no pair within 5 A")
    print(a.label, json.dumps({k: {kk: round(v[kk], 2) for kk in ("structures", "e_mae_mev_atom", "e_mae_offset_free_mev", "f_mae_mev_a")}
                               for k, v in r["strata"].items() if k in show}))
    if "curves" in r:
        print("curves", r["curves"]["scans"], "scans, dE MAE meV", round(r["curves"]["dE_mae_mev"], 2))


if __name__ == "__main__":
    main()
