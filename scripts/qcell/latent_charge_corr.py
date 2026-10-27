"""Latent (LES) charges of a model against OMol25 partial charges on the val selection.

    python -I latent_charge_corr.py SEL.npz SEL_partial_charges.npz LATENT OUT.json

LATENT: mlipcpp-eval-style JSON whose frames carry per-atom "charges" (or "latent_charges"), or an
npz with a flat "charges" array in SEL.npz atom order. Per reference scheme (mulliken, lowdin, nbo):
pooled Pearson r, slope of reference on latent, per-structure r (median, p05, p95, fraction < 0), sign
agreement on atoms with |reference| > 0.1 e, per element r and sign agreement, per-molecule (covalent
component) charge-sum r. By domain and charge state. A global sign flip of the latent gauge shows as
r < 0 everywhere.
"""
import json
import sys

import numpy as np

SYM = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
       "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
       "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 3 and x.std() > 0 and y.std() > 0 else np.nan


def load_latent(p, N):
    if p.endswith(".npz"):
        q = np.load(p)["charges"]
    else:
        fr = json.load(open(p))["frames"]
        q = np.concatenate([np.asarray(f.get("charges", f.get("latent_charges")), float).reshape(len(f["forces"]) // 3, -1)[:, 0]
                            for f in fr])
    assert q.size == N, (q.size, N)
    return q


def block(ks, d, ref, has, lat):
    n = d["n"]; off = np.concatenate([[0], np.cumsum(n)])
    ks = [k for k in ks if has[k]]
    if not ks:
        return None
    idx = np.concatenate([np.arange(off[k], off[k + 1]) for k in ks])
    x, y, Z = lat[idx], ref[idx], d["Z"][idx]
    per = np.array([r(lat[off[k]:off[k + 1]], ref[off[k]:off[k + 1]]) for k in ks]); per = per[np.isfinite(per)]
    big = np.abs(y) > 0.1
    slope = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else np.nan
    el = {SYM[z]: {"atoms": int((Z == z).sum()), "r": r(x[Z == z], y[Z == z]),
                   "sign_agree": float((np.sign(x[(Z == z) & big]) == np.sign(y[(Z == z) & big])).mean()) if ((Z == z) & big).any() else None}
          for z in np.unique(Z) if (Z == z).sum() >= 20}
    mol_l, mol_r = [], []
    comp = d["comp"]
    for k in ks:
        c = comp[off[k]:off[k + 1]]
        for j in np.unique(c):
            m = c == j
            mol_l.append(lat[off[k]:off[k + 1]][m].sum()); mol_r.append(ref[off[k]:off[k + 1]][m].sum())
    return {"structures": len(ks), "atoms": int(len(idx)), "pooled_r": r(x, y), "slope_ref_on_latent": slope,
            "sign_agree_|ref|>0.1": float((np.sign(x[big]) == np.sign(y[big])).mean()) if big.any() else None,
            "per_structure_r": {"median": float(np.median(per)) if len(per) else None,
                                "p05": float(np.percentile(per, 5)) if len(per) else None,
                                "p95": float(np.percentile(per, 95)) if len(per) else None,
                                "frac_negative": float((per < 0).mean()) if len(per) else None},
            "molecule_sum_r": r(np.array(mol_l), np.array(mol_r)), "per_element": el}


def main():
    d = dict(np.load(sys.argv[1])); pc = dict(np.load(sys.argv[2])); lat = load_latent(sys.argv[3], int(d["n"].sum()))
    q, dom = d["charge"], d["domain"]
    groups = {"all": np.arange(len(q))}
    for g in np.unique(dom):
        groups[f"domain={g}"] = np.nonzero(dom == g)[0]
    groups["charge=0"] = np.nonzero(q == 0)[0]; groups["|charge|=1"] = np.nonzero(np.abs(q) == 1)[0]
    groups["|charge|>=2"] = np.nonzero(np.abs(q) >= 2)[0]
    res = {}
    for key in ("mulliken_charges", "lowdin_charges", "nbo_charges"):
        res[key] = {g: block(ks, d, pc[key], pc["has_" + key], lat) for g, ks in groups.items()}
    json.dump(res, open(sys.argv[4], "w"), indent=1)
    for key, v in res.items():
        print("==", key)
        for g, b in v.items():
            if b:
                ps = b["per_structure_r"]
                print("  %-26s pooled r %.3f slope %.3f sign %.3f per-str r median %.3f frac<0 %.3f molecule-sum r %.3f" % (
                    g, b["pooled_r"], b["slope_ref_on_latent"], b["sign_agree_|ref|>0.1"] or np.nan, ps["median"] or np.nan,
                    ps["frac_negative"] or np.nan, b["molecule_sum_r"]))


if __name__ == "__main__":
    main()
