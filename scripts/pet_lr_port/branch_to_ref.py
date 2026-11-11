#!/usr/bin/env python3
"""ref_branch.py output (his branches: predict's outputs only) -> the ref_petlr.py layout lr_parity.py reads:
atom/energy (the structure energy on its first atom: the per-structure sums are exact), forces, atom/charges,
struct/metallicity, struct/dipole (NaN: not emitted by predict), atom_ptr, and virial [S, 3, 3] on periodic frames
(his predict's stress, dU/d eps).

    python branch_to_ref.py IN.npz OUT.npz
"""
import sys

import numpy as np

r = np.load(sys.argv[1])
ap = r["atom_ptr"]
S = len(ap) - 1
e = np.zeros(ap[-1])
e[ap[:-1]] = np.asarray(r["energy"]).reshape(-1)[:S]
out = {"atom_ptr": ap, "atom/energy": e, "forces": r["forces"], "struct/dipole": np.full((S, 3), np.nan)}
if "charges" in r.files:
    out["atom/charges"] = r["charges"]
if "metallicity" in r.files:
    out["struct/metallicity"] = np.asarray(r["metallicity"]).reshape(-1)[:S]
if "virial" in r.files:
    out["virial"] = np.asarray(r["virial"]).reshape(S, 3, 3)
np.savez(sys.argv[2], **out)
