#!/usr/bin/env python3
"""His own fp32 floor (after M. Langer, ipsum/PET-LR): for every pair of his fp32 runs on a family, the paired per-frame
P(run a worse than run b) against his fp64, per output (ties count 1/2; pairs that tie on most frames, i.e.
bitwise-identical runs, are skipped); the range over ordered pairs is the floor criterion that
mlip.cpp's P against his runs is judged by. Same metrics and frame masking as lr_parity.py.

    python his_pairwise.py REF64.npz RUN32.npz RUN32.npz [...] [--overflow FRAMES CUTOFF MAXNB] [--label NAME] [--first N]
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_parity import from_ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref64"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--label", default="")
    ap.add_argument("--first", type=int, default=60, help="frames compared (mlip.cpp's periodic runs: first 60)")
    ap.add_argument("--overflow", nargs=3)
    a = ap.parse_args()
    z = np.load(a.ref64)
    runs = {os.path.basename(p)[:-4]: from_ref(p, z) for p in a.runs}
    n = len(next(iter(runs.values()))["E"])
    mask = np.zeros(n, bool); mask[:a.first] = True
    if a.overflow:
        from ase.io import read
        from ase.neighborlist import neighbor_list
        fr = read(a.overflow[0], ":")
        for f in range(min(n, len(fr))):
            if np.bincount(neighbor_list("i", fr[f], float(a.overflow[1])), minlength=len(fr[f])).max() > int(a.overflow[2]):
                mask[f] = False
    cells = []
    for k in ("E", "F", "W", "q", "mu", "metallicity"):
        ps = []
        for (na, ra), (nb, rb) in itertools.permutations(runs.items(), 2):
            if k not in ra or k not in rb:
                continue
            ok = mask[:len(ra[k])] & ~np.isnan(ra[k]) & ~np.isnan(rb[k])
            tie = ra[k][ok] == rb[k][ok]
            if tie.mean() > 0.5:  # bitwise-identical runs (e.g. two CPU runs of his): not a spread sample
                continue
            ps.append(float((ra[k][ok] > rb[k][ok]).mean() + 0.5 * tie.mean()))
        if ps:
            cells.append(f"{k} P {min(ps):.2f}-{max(ps):.2f}")
        else:
            cells.append(f"{k} identical")
    print(f"{a.label:34s} ({len(runs)} runs, {int(mask.sum())} frames) " + "; ".join(cells))


if __name__ == "__main__":
    main()
