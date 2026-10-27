"""Per-element energy baseline on the QCell training set.

    python -I fit_baseline.py TRAIN.npz OUT.json [--pred DUMP.jsonl --idx SAMPLE.idx.txt]

Without --pred: b0 = least squares of E (eV) on the element counts over all training
structures. With --pred (mlipcpp-plato dump of an init model on the structures listed in
--idx): b = b0 + c, c = least squares over that sample of (E - b0.n - E_model) (the
make_res.py rule of the malonaldehyde runs: the model keeps its own composition and scale,
the data carry E - b.n). Writes {composition: {Z: eV}, energy_scale (RMS per-atom
residual after b0), ...}.
"""
import argparse
import json

import numpy as np


def counts(d):
    n = d["n"]
    sid = np.repeat(np.arange(len(n)), n)
    types = np.unique(d["Z"])
    X = np.zeros((len(n), len(types)))
    np.add.at(X, (sid, np.searchsorted(types, d["Z"])), 1)
    return X, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--idx", default=None)
    a = ap.parse_args()
    d = np.load(a.npz)
    X, types = counts(d)
    E = d["E"]
    b0 = np.linalg.lstsq(X, E, rcond=None)[0]
    r0 = (E - X @ b0) / d["n"]
    j = {"types": types.tolist(), "b0": dict(zip(map(str, types.tolist()), b0.tolist())),
         "energy_scale": float(np.sqrt(np.mean(r0 ** 2))), "residual_per_atom_mae_ev_b0": float(np.mean(np.abs(r0))),
         "train_structures_per_type": dict(zip(map(str, types.tolist()), (X > 0).sum(0).tolist()))}
    b = b0.copy()
    if a.pred:
        idx = np.loadtxt(a.idx, dtype=int).reshape(-1)
        P = [json.loads(l)["energy"] for l in open(a.pred) if l.strip()]
        assert len(P) == len(idx), (len(P), len(idx))
        Xs = X[idx]
        y = E[idx] - Xs @ b0 - np.array(P)
        c = np.linalg.lstsq(Xs, y, rcond=None)[0]
        b = b0 + c
        res = (y - Xs @ c)
        j.update(pred=a.pred, sample=len(idx), correction=dict(zip(map(str, types.tolist()), c.tolist())),
                 sample_types_missing=[int(t) for t, k in zip(types, (Xs > 0).sum(0)) if k == 0],
                 sample_residual_rms_ev=float(np.sqrt(np.mean(res ** 2))),
                 sample_residual_per_atom_rms_ev=float(np.sqrt(np.mean((res / d["n"][idx]) ** 2))))
    j["composition"] = dict(zip(map(str, types.tolist()), b.tolist()))
    json.dump(j, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in j.items() if k not in ("b0", "correction")}))


if __name__ == "__main__":
    main()
