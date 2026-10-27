"""Agreement of OMol25 partial-charge schemes on a selection (prep_omol25_val.py write output).

    python -I charge_corr.py SEL.npz SEL_partial_charges.npz OUT.json

Pearson r of Mulliken vs Lowdin (and each vs NBO where present): pooled over atoms, per structure
(median, 5th / 95th percentile, fraction r < 0; structures with < 3 atoms or zero variance skipped),
per element (pooled); least-squares slope of Lowdin on Mulliken (with intercept, pooled). All also by
domain and by charge state (0, +-1, |q|>=2).
"""
import json
import sys

import numpy as np

SYM = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
       "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
       "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()


def r(x, y):
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def block(sel, d, pc, a, b):
    n = d["n"]; off = np.concatenate([[0], np.cumsum(n)])
    hasb = pc["has_" + b]; hasa = pc["has_" + a]
    ks = [k for k in sel if hasa[k] and hasb[k]]
    if not ks:
        return None
    idx = np.concatenate([np.arange(off[k], off[k + 1]) for k in ks])
    x, y, Z = pc[a][idx], pc[b][idx], d["Z"][idx]
    per = np.array([r(pc[a][off[k]:off[k + 1]], pc[b][off[k]:off[k + 1]]) for k in ks])
    per = per[np.isfinite(per)]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]
    el = {}
    for z in np.unique(Z):
        m = Z == z
        if m.sum() >= 20:
            el[SYM[z]] = {"atoms": int(m.sum()), "r": r(x[m], y[m]), "mean_" + a: float(x[m].mean()), "mean_" + b: float(y[m].mean())}
    return {"structures": len(ks), "atoms": int(len(idx)), "pooled_r": r(x, y), "slope_" + b + "_on_" + a: float(slope),
            "intercept": float(icpt),
            "per_structure_r": {"n": int(len(per)), "median": float(np.median(per)) if len(per) else None,
                                "p05": float(np.percentile(per, 5)) if len(per) else None,
                                "p95": float(np.percentile(per, 95)) if len(per) else None,
                                "frac_negative": float((per < 0).mean()) if len(per) else None},
            "per_element": el}


def main():
    d = dict(np.load(sys.argv[1])); pc = dict(np.load(sys.argv[2])); out = sys.argv[3]
    q, dom = d["charge"], d["domain"]
    groups = {"all": np.arange(len(q))}
    for g in np.unique(dom):
        groups[f"domain={g}"] = np.nonzero(dom == g)[0]
    groups["charge=0"] = np.nonzero(q == 0)[0]; groups["|charge|=1"] = np.nonzero(np.abs(q) == 1)[0]
    groups["|charge|>=2"] = np.nonzero(np.abs(q) >= 2)[0]
    res = {}
    for pair in (("mulliken_charges", "lowdin_charges"), ("mulliken_charges", "nbo_charges"), ("lowdin_charges", "nbo_charges")):
        key = f"{pair[0].split('_')[0]}-{pair[1].split('_')[0]}"
        res[key] = {g: block(ks, d, pc, *pair) for g, ks in groups.items()}
    json.dump(res, open(out, "w"), indent=1)
    for key, v in res.items():
        print("==", key)
        for g, b in v.items():
            if b:
                ps = b["per_structure_r"]
                sl = [kk for kk in b if kk.startswith("slope")][0]
                print("  %-26s %5d str %7d at  pooled r %.3f  slope %.3f  per-str r median %.3f [p05 %.3f, p95 %.3f] frac<0 %.3f" % (
                    g, b["structures"], b["atoms"], b["pooled_r"], b[sl], ps["median"], ps["p05"], ps["p95"], ps["frac_negative"]))
        b = v["all"]
        if b:
            print("  per element (all):", {e: round(x["r"], 3) for e, x in b["per_element"].items()})


if __name__ == "__main__":
    main()
