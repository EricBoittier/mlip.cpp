#!/usr/bin/env python3
"""Step-0 errors on the held-out split: energy MAE per atom (after a per-element baseline refit on the same split for
both models, so composition offsets cancel) and force MAE / max, readout-init model vs its no-readout variant.
    python ro_step0.py DATA_DIR VARIANT=DIR ..."""
import json, sys
import numpy as np
from ase.io import read
fr = read(f"{sys.argv[1]}/heldout.xyz", ":")
Ed = np.array([a.get_potential_energy() for a in fr]); Fd = [a.get_forces() for a in fr]
C = np.array([np.bincount(a.numbers, minlength=119) for a in fr], float); C = C[:, C.sum(0) > 0]
nat = np.array([len(a) for a in fr])
out = {}
for arg in sys.argv[2:]:
    name, d = arg.split("=")
    ev = json.load(open(f"{d}/heldout.eval.json"))["frames"]
    E = np.array([e["energy"] for e in ev]); r = Ed - E
    c = np.linalg.lstsq(C, r, rcond=None)[0]; re = r - C @ c
    df = np.concatenate([np.abs(np.array(e["forces"]).reshape(-1, 3) - f).ravel() for e, f in zip(ev, Fd)])
    out[name] = {"E_mae_meV_atom_refit": float(np.mean(np.abs(re) / nat) * 1e3), "F_mae_meV_A": float(df.mean() * 1e3),
                 "F_max_err_eV_A": float(df.max())}
print(json.dumps(out, indent=1))
