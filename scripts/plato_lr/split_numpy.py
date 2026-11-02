#!/usr/bin/env python3
"""E_split of a Plato-PET-LR GGUF (pet.plato_lr.les_split) against a direct numpy evaluation:
mlipcpp-eval's energies at gate g minus those at gate 0, and g (sum_ij q_i q_j w(r_ij) - Q^2)
(periodic: every image within --radius-max, minus Q^2 V_w / V) of its latent charges.

    mlipcpp-eval m-g0.gguf F.xyz g0.json; mlipcpp-eval m-g.gguf F.xyz g.json
    python split_numpy.py F.xyz g0.json g.json --gate 0.1 [--radius 8 --delta 1 --radius-max 10]
"""
import argparse
import json

import ase.io
import numpy as np


def switch(r, R, D):
    x = np.clip((r - R + D) / D, 0, 1)
    return 1 - x**3 * (10 - 15 * x + 6 * x * x)


def volume(R, D):
    a = R - D
    return 4 * np.pi * (a**3 / 3 + D * (a * a / 2 + 2 * a * D / 7 + D * D * 5 / 84))


def e_split(a, q, g, R, D, rmax):
    R = min(max(R, D), rmax)
    if not a.pbc.any():
        r = np.linalg.norm(a.positions[:, None] - a.positions[None], axis=-1)
        return g * (q @ switch(r, R, D) @ q - q.sum() ** 2)
    C = a.cell.array
    V = abs(np.linalg.det(C))
    M = [int(np.floor(rmax * np.linalg.norm(np.cross(C[(k + 1) % 3], C[(k + 2) % 3])) / V + 0.5)) for k in range(3)]
    d = a.positions[None] - a.positions[:, None]
    n0 = -np.round(d @ np.linalg.inv(C))
    tot = 0.0
    for m in np.array(np.meshgrid(*[np.arange(-x, x + 1) for x in M], indexing="ij")).reshape(3, -1).T:
        W = switch(np.linalg.norm(d + (n0 + m) @ C, axis=-1), R, D)
        if not m.any():
            np.fill_diagonal(W, 1.0)  # n0 = 0 on the diagonal: the atom itself, w(0) = 1
        tot += q @ W @ q
    return g * (tot - q.sum() ** 2 * volume(R, D) / V)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xyz")
    ap.add_argument("g0")
    ap.add_argument("g")
    ap.add_argument("--gate", type=float, required=True)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--radius-max", type=float, default=10.0)
    args = ap.parse_args()
    frames = ase.io.read(args.xyz, ":")
    f0, f1 = json.load(open(args.g0))["frames"], json.load(open(args.g))["frames"]
    worst = 0.0
    for k, (a, x0, x1) in enumerate(zip(frames, f0, f1)):
        q = np.array(x1["latent_charges"], float)[: len(a)]
        got = x1["energy"] - x0["energy"]
        ref = e_split(a, q, args.gate, args.radius, args.delta, args.radius_max)
        worst = max(worst, abs(got - ref))
        print(f"{k:3d} n {len(a):4d} pbc {int(a.pbc.any())} Q {q.sum():+.4f}  E_split mlip.cpp {got:+.6f}  numpy {ref:+.6f}")
    print(f"max |dE_split| {worst:.2e} eV")


if __name__ == "__main__":
    main()
