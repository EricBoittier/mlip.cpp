#!/usr/bin/env python3
"""How much the LES energy could help a model, per val_sel stratum: from eval_omol.py --per-structure files of
the same model with the LES gate at 0 (dE0 = E - E_ref) and at 1 (E_les = dE1 - dE0, linear in the gate), the
least-squares gate g* = -sum(dE0 E_les / n^2) / sum(E_les^2 / n^2) (per-atom errors), the energy MAE at g* and
at 0, and corr(-dE0, E_les). g* far from 0 with a lower MAE: the LES term carries signal the model lacks.

    python -I les_decomp.py G0.per.npz G1.per.npz OUT.json [--label E-init]
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("g0"); ap.add_argument("g1"); ap.add_argument("out"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    d0, d1 = np.load(a.g0), np.load(a.g1)
    n = d0["n"].astype(float)
    e0 = d0["dE"] / n                    # per-atom error, eV
    les = (d1["dE"] - d0["dE"]) / n      # per-atom LES energy at gate 1, eV
    rows = {}
    for name, m in zip(d0["strata_names"], d0["strata"]):
        if m.sum() < 5:
            continue
        x, y = les[m], e0[m]
        g = -float((x * y).sum() / max((x * x).sum(), 1e-30))
        r = float(np.corrcoef(-y, x)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        rows[str(name)] = dict(structures=int(m.sum()), g_star=g, corr=r,
                               e_mae_g0=1e3 * float(np.abs(y).mean()), e_mae_gstar=1e3 * float(np.abs(y + g * x).mean()),
                               les_mean_abs=1e3 * float(np.abs(x).mean()))
    json.dump({"label": a.label, "strata": rows}, open(a.out, "w"), indent=1)
    for k in ("all", "charge=0", "|charge|=1", "|charge|>=2", "separated (>=8 A, charged fragment)"):
        if k in rows:
            v = rows[k]
            print(f"{a.label:8s} {k:38s} g* {v['g_star']:+.2e}  r {v['corr']:+.2f}  E MAE {v['e_mae_g0']:.2f} -> {v['e_mae_gstar']:.2f}"
                  f"  |E_les| {v['les_mean_abs']:.0f} meV/atom")


if __name__ == "__main__":
    main()
