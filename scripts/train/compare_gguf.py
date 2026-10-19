#!/usr/bin/env python3
"""Byte-level comparison of two PET GGUFs: every tensor and every metadata field.
    python compare_gguf.py a.gguf b.gguf [--ignore general.name ...]"""
import argparse
import gguf
import numpy as np


def fields(r):
    out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        out[k] = b"".join(bytes(np.asarray(f.parts[i]).tobytes()) for i in f.data)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--ignore", nargs="*", default=["general.name"])
x = ap.parse_args()
ra, rb = gguf.GGUFReader(x.a), gguf.GGUFReader(x.b)
fa, fb = fields(ra), fields(rb)
bad = 0
for k in sorted(set(fa) | set(fb)):
    if k in x.ignore:
        continue
    if fa.get(k) != fb.get(k):
        bad += 1
        print(f"field {k}: {'missing in b' if k not in fb else 'missing in a' if k not in fa else 'differs'}")
ta, tb = {t.name: t for t in ra.tensors}, {t.name: t for t in rb.tensors}
for k in sorted(set(ta) | set(tb)):
    if k not in ta or k not in tb:
        bad += 1; print(f"tensor {k}: only in {'a' if k in ta else 'b'}"); continue
    a, b = ta[k], tb[k]
    da, db = np.asarray(a.data).tobytes(), np.asarray(b.data).tobytes()
    shape = lambda t: [int(n) for n in t.shape] or [1]  # a 0-d constant: [] from Python, [1] from mlipcpp-train
    if a.tensor_type != b.tensor_type or shape(a) != shape(b) or da != db:
        bad += 1
        diff = np.abs(np.asarray(a.data, np.float64).ravel() - np.asarray(b.data, np.float64).ravel()).max() \
            if len(da) == len(db) else float("nan")
        print(f"tensor {k}: differs (max |diff| {diff:.3g})")
print(f"{len(fa)} fields, {len(ta)} tensors compared ({', '.join(x.ignore)} ignored): "
      + ("IDENTICAL" if not bad else f"{bad} differences"))
