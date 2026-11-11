#!/usr/bin/env python3
"""Bitwise comparison of two mlipcpp-eval JSON runs (or directories of them, matched by file name): per file, whether
energies / charges / forces are bit-identical, and the max abs differences (CUDA forces are run-to-run
nondeterministic: compare F against a same-config rerun's spread). Speed: --speed DIR prints the median seconds per
structure of every *.json in DIR (mlipcpp-eval --repeat).

    python bitwise_runs.py A[.json|DIR] B[.json|DIR]
    python bitwise_runs.py --speed DIR
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(p):
    d = json.load(open(p))
    return d["frames"] if isinstance(d, dict) else d


def cmp(a, b):
    fa, fb = load(a), load(b)
    n = min(len(fa), len(fb))
    E = [np.array([f.get("energy", np.nan) for f in x[:n]]) for x in (fa, fb)]
    q = [np.concatenate([np.ravel(f.get("charges", [])) for f in x[:n]]) for x in (fa, fb)]
    F = [np.concatenate([np.ravel(f.get("forces", [])) for f in x[:n]]) for x in (fa, fb)]
    s = lambda u, v: (np.array_equal(u, v, equal_nan=True), float(np.nanmax(np.abs(u - v))) if u.size else 0.0)
    (eE, dE), (eq, dq), (eF, dF) = s(*E), s(*q), s(*F)
    return f"E {'==' if eE else '!='} ({dE:.1e})  q {'==' if eq else '!='} ({dq:.1e})  F {'==' if eF else '!='} ({dF:.1e})  n {n}"


def main():
    if sys.argv[1] == "--speed":
        for p in sorted(Path(sys.argv[2]).glob("*.json")):
            t = [f["seconds"] for f in load(p) if "seconds" in f]
            print(f"  {p.stem:28s} median {1e3 * np.median(t):7.2f} ms / structure  ({1 / np.median(t):6.1f} struct/s, n {len(t)})")
        return
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
    if A.is_dir():
        for p in sorted(A.glob("*.json")):
            q = B / p.name
            print(f"  {p.name:42s} " + (cmp(p, q) if q.exists() else "(missing in B)"))
    else:
        print(cmp(A, B))


if __name__ == "__main__":
    main()
