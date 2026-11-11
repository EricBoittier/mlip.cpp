#!/usr/bin/env python3
"""Concatenate ref_petlr.py / ref_trunk.py dumps of consecutive frame chunks into one (atom arrays with their atom_ptr
offsets, frame arrays stacked; meta of the first chunk).    python merge_npz.py OUT.npz A.npz B.npz ..."""
import sys

import numpy as np

out, parts = sys.argv[1], [np.load(p) for p in sys.argv[2:]]
ptr = [0]
for z in parts:
    ptr += list(ptr[-1] + z["atom_ptr"][1:])
merged = {"atom_ptr": np.asarray(ptr), "meta": parts[0]["meta"]}
na = [int(z["atom_ptr"][-1]) for z in parts]
nf = [len(z["atom_ptr"]) - 1 for z in parts]
for k in parts[0].files:
    if k in ("atom_ptr", "meta"):
        continue
    vs = [z[k] for z in parts]
    if all(v.shape[0] == n for v, n in zip(vs, na)) and not all(v.shape[0] == n for v, n in zip(vs, nf)):
        merged[k] = np.concatenate(vs)
    elif all(v.shape[0] == n for v, n in zip(vs, nf)):
        merged[k] = np.concatenate(vs)
    else:
        raise ValueError(f"{k}: cannot tell atom from frame axis")
np.savez(out, **merged)
print(out, {k: v.shape for k, v in merged.items() if k != "meta"})
