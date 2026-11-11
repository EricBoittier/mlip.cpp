#!/usr/bin/env python3
"""One line per (model, family, backend): for each output, mlip.cpp max / max of his fp32 runs and the range of the
paired P(mlip.cpp worse than his run) over his runs, plus the range of P of his runs against each other (his own
spread). Uses lr_parity.per_frame. Files: CPP json pattern, REF dir with full_<fam>_cpu64.npz and his runs
full_<fam>_{cpu32,cpu32_v2,gpu32}.npz (whatever exists).

    python lr_summary.py --root PARITY_DIR --models readout-pol control ... --fams ion cluster mol
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_parity import from_cpp, from_ref  # noqa: E402

KEYS = ("E", "F", "q", "mu", "dipole_abs", "metallicity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--fams", nargs="+", default=["ion", "cluster", "mol"])
    ap.add_argument("--refdir", default="ref/{model}", help="ref dir pattern")
    ap.add_argument("--refpat", default="full_{fam}_", help="ref file prefix pattern")
    ap.add_argument("--cpu", default="out/lr/{model}_{fam}.json")
    ap.add_argument("--cuda", default="out/cuda/lr/{model}_{fam}_cuda.json")
    a = ap.parse_args()
    os.chdir(a.root)
    for m, fam in itertools.product(a.models, a.fams):
        rd = a.refdir.format(model=m); pre = a.refpat.format(fam=fam, model=m)
        r64 = f"{rd}/{pre}cpu64.npz"
        if not os.path.exists(r64):
            continue
        z = np.load(r64)
        runs = {os.path.basename(p)[len(pre):-4]: from_ref(p, z)
                for p in sorted(glob.glob(f"{rd}/{pre}*32*.npz"))}
        if not runs:
            print(f"{m:22s} {fam:8s} no fp32 run of his"); continue
        his_pp = {}
        for k in KEYS:
            ps = [float(np.nanmean(runs[x][k] > runs[y][k])) for x in runs for y in runs if x != y and k in runs[x]]
            his_pp[k] = (min(ps), max(ps)) if ps else None
        for be, pat in (("CPU", a.cpu), ("CUDA", a.cuda)):
            cp = pat.format(model=m, fam=fam)
            if not os.path.exists(cp):
                continue
            c = from_cpp(cp, z)
            cells = []
            for k in KEYS:
                if k not in c or not any(k in r for r in runs.values()):
                    continue
                hm = max(np.nanmax(r[k]) for r in runs.values() if k in r)
                n = int(np.sum(~np.isnan(c[k])))
                ps = [float(np.nanmean(c[k][:len(r[k])] > r[k])) for r in runs.values() if k in r]
                hp = his_pp[k]
                hs = f" his[{hp[0]:.2f},{hp[1]:.2f}]" if hp else ""
                cells.append(f"{k} {np.nanmax(c[k]) / hm:.2f}x P[{min(ps):.2f},{max(ps):.2f}]{hs}")
            print(f"{m:22s} {fam:8s} {be:4s} n{n:4d} runs {len(runs)}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
