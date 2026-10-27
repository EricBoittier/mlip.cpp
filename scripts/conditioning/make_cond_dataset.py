#!/usr/bin/env python3
"""A synthetic dataset with per-structure charge and spin multiplicity, for the
conditioning tests (gradient checks, short training runs): small H/C/N/O molecules
(isolated) and periodic boxes, rattled, with a pair-potential energy plus a charge /
spin dependent offset (so that the conditioning has something to learn), forces its
exact negative gradient.

    python make_cond_dataset.py OUT [--n 64] [--seed 0] [--max-charge 2] [--max-spin 3]

Writes OUT/{train,val,test}.xyz (extxyz, info charge / spin_multiplicity / e_ref,
arrays f_ref) and the memmap splits OUT/{train,val,test}/ through
scripts/train/xyz_to_memmap.py (charge.bin, spin_multiplicity.bin included), and
OUT/{train,val,test}.bin (Plato format v2, plato_dataset.py) when --plato is given.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import ase
import ase.build
import ase.io
import numpy as np

MOLS = ["H2O", "CH4", "NH3", "CH3OH", "C2H6", "HCN", "CH3NO2", "C2H4", "H2CO", "N2H4"]


def energy_forces(a, q, s):
    """Morse-like pair energy (minimum image for periodic) + 0.3 q^2 + 0.2 (s - 1) + 0.05 q n."""
    pos = a.positions
    n = len(a)
    e, f = 0.0, np.zeros_like(pos)
    for i in range(n):
        d = pos - pos[i]
        if a.pbc.all():
            frac = np.linalg.solve(a.cell.array.T, d.T).T
            d = (frac - np.round(frac)) @ a.cell.array
        r = np.linalg.norm(d, axis=1)
        m = (r > 1e-8) & (r < 4.0)
        x = np.exp(-1.2 * (r[m] - 1.3))
        sw = 0.5 * (1 + np.cos(np.pi * r[m] / 4.0))
        dsw = -0.5 * np.pi / 4.0 * np.sin(np.pi * r[m] / 4.0)
        u = 0.5 * (x * x - 2 * x)  # half: each pair twice
        du = 0.5 * (-2.4 * x * x + 2.4 * x)
        e += float((u * sw).sum())
        g = (du * sw + u * dsw)[:, None] * d[m] / r[m][:, None]  # dE/d(x_j - x_i), half
        f[i] += 2 * g.sum(0)  # -dE/dx_i = + sum (dE/dd) (both halves)
    e += 0.3 * q * q + 0.2 * (s - 1) + 0.05 * q * n
    return e, f


def make(n, rng, max_q, max_s):
    out = []
    for k in range(n):
        if k % 4 == 3:  # periodic box of 2-3 molecules
            a = ase.Atoms()
            for j in range(2 + k % 2):
                m = ase.build.molecule(MOLS[rng.integers(len(MOLS))])
                m.rotate(rng.uniform(0, 360), rng.normal(size=3))
                m.positions += rng.uniform(0, 5.0, 3)
                a += m
            L = 6.0 + rng.uniform(0, 1.0)
            a.cell = np.diag([L, L * 1.05, L * 0.95]) + rng.normal(0, 0.2, (3, 3))
            a.pbc = True
        else:
            a = ase.build.molecule(MOLS[rng.integers(len(MOLS))])
        a.positions += rng.normal(0, 0.08, a.positions.shape)
        q = int(rng.integers(-max_q, max_q + 1))
        s = int(rng.integers(1, max_s + 1))
        a.positions = a.positions.astype(np.float32).astype(np.float64)
        if a.pbc.any():
            a.cell = a.cell.array.astype(np.float32).astype(np.float64)
        e, f = energy_forces(a, q, s)
        a.info.update(charge=q, spin_multiplicity=s, e_ref=e)
        a.arrays["f_ref"] = f
        out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-charge", type=int, default=2)
    ap.add_argument("--max-spin", type=int, default=3)
    ap.add_argument("--plato", action="store_true", help="also the Plato .bin splits (plato_dataset.py)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    here = Path(__file__).resolve().parents[1]
    for split, n in (("train", args.n), ("val", max(8, args.n // 4)), ("test", max(8, args.n // 4))):
        frames = make(n, rng, args.max_charge, args.max_spin)
        ase.io.write(out / f"{split}.xyz", frames, format="extxyz")
        subprocess.run([sys.executable, str(here / "train/xyz_to_memmap.py"), str(out / f"{split}.xyz"),
                        str(out / split), "--energy-key", "e_ref", "--forces-key", "f_ref"], check=True)
        if args.plato:
            subprocess.run([sys.executable, str(here / "plato/plato_dataset.py"), str(out / f"{split}.xyz"),
                            str(out / f"{split}.bin")], check=True)


if __name__ == "__main__":
    main()
