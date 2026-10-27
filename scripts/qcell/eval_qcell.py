"""Errors of `mlipcpp-plato dump` predictions on a QCell split (prep_qcell.py order).

    python -I eval_qcell.py SPLIT.npz PRED.jsonl BASELINE.json [--label NAME] [--json OUT]

E_pred = E_model + b.n (b = BASELINE.json composition, the one the run's targets used) against
the total energy E; forces against F. Energy MAE/RMSE in meV/atom and meV per structure, force
MAE/RMSE per Cartesian component in meV/A (and kcal/mol/A), overall and per QCell subset.
"""
import argparse
import json

import numpy as np

NAMES = ["nucleic_acids", "lipids", "sugars", "ions_water", "dimers"]
KCAL = 43.36410  # meV per kcal/mol


def stats(dE, n, dF):
    return {"structures": int(len(dE)), "atoms": int(n.sum()),
            "e_mae_mev_atom": float(1e3 * np.mean(np.abs(dE / n))), "e_rmse_mev_atom": float(1e3 * np.sqrt(np.mean((dE / n) ** 2))),
            "e_mae_mev": float(1e3 * np.mean(np.abs(dE))), "e_rmse_mev": float(1e3 * np.sqrt(np.mean(dE ** 2))),
            "f_mae_mev_a": float(1e3 * np.mean(np.abs(dF))), "f_rmse_mev_a": float(1e3 * np.sqrt(np.mean(dF ** 2))),
            "f_mae_kcal_mol_a": float(1e3 * np.mean(np.abs(dF)) / KCAL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("pred")
    ap.add_argument("baseline")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = np.load(a.npz)
    n, Z, E, F, sub = d["n"], d["Z"], d["E"], d["F"], d["subset"]
    off = np.concatenate([[0], np.cumsum(n)])
    b = np.zeros(119)
    for z, v in json.load(open(a.baseline))["composition"].items():
        b[int(z)] = v
    P = [json.loads(l) for l in open(a.pred) if l.strip()]
    assert len(P) == len(n), (len(P), len(n))
    dE = np.empty(len(n))
    dF = np.empty((off[-1], 3))
    for k, p in enumerate(P):
        s = slice(off[k], off[k + 1])
        assert p["n"] == n[k]
        dE[k] = p["energy"] + b[Z[s]].sum() - E[k]
        dF[s] = np.asarray(p["forces"], float).reshape(-1, 3) - F[s]
    asub = np.repeat(sub, n)
    r = {"label": a.label, "pred": a.pred, "baseline": a.baseline, "all": stats(dE, n, dF)}
    for s, name in enumerate(NAMES):
        m = sub == s
        r[name] = stats(dE[m], n[m], dF[asub == s])
    print(json.dumps({"label": a.label, **{k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r.items() if isinstance(v, dict)}}))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
