#!/usr/bin/env python3
"""Benchmark frames for PET-OMol-S timings: H/C/N/O molecular clusters of about N atoms
(g2 molecules placed without overlap at roughly liquid density), charge 0, singlet or
doublet, extxyz with OMol's charge / spin keys.

    python make_bench.py OUT.xyz [--sizes 20 100 300]
"""
import argparse

import ase
import ase.collections
import ase.io
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--sizes", type=int, nargs="+", default=[20, 100, 300])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
pool = [ase.collections.g2[n] for n in ase.collections.g2.names
        if set(ase.collections.g2[n].numbers) <= {1, 6, 7, 8} and len(ase.collections.g2[n]) >= 4]
frames = []
for n in args.sizes:
    box = (n / 0.1) ** (1 / 3)  # ~0.1 atoms / A^3
    a = ase.Atoms()
    while len(a) < n:
        b = pool[rng.integers(len(pool))].copy()
        b.positions += rng.uniform(0, box, 3) - b.positions.mean(0)
        if len(a) and np.min(np.linalg.norm(a.positions[:, None] - b.positions[None], axis=-1)) < 2.2:
            continue
        a += b
    electrons = int(a.numbers.sum())
    a.info = {"charge": 0, "spin": 1 if electrons % 2 == 0 else 2}
    frames.append(a)
    print(n, "->", len(a), "atoms")
ase.io.write(args.out, frames, format="extxyz")
