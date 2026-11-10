#!/usr/bin/env python3
"""Read-only export of frames from a marathon ragged-mmap split (any of M. Langer's ipsum/PET-LR
datasets) to extxyz + a labels npz, for the PET-LR port parity tests.

The record layout is marathon's `flatten_atoms` (43e5c5b): [N, pbc(3), positions(3N), cell(9), Z(N)],
then every property of the split's properties.yaml in sorted-name order, each of its declared shape
("atom" = N). Missing values are NaN. Needs numpy and pyyaml only; nothing is written next to the source.

    python export_frames.py SPLIT_DIR OUT_PREFIX [--n 240] [--seed 0]

Frames are a seeded random sample (without replacement) of the split. The extxyz carries pbc, Lattice
(periodic frames), and the per-structure info properties (charge, spin_multiplicity, field, dipole_origin, ...)
as plain key=value; per-atom labels go to the npz only.
"""
import argparse
from math import prod

import numpy as np
import yaml

SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
           "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
           "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()


def records(split):
    s = np.memmap(f"{split}/mmap/starts/data.ninja", dtype=np.int64, mode="r")
    e = np.memmap(f"{split}/mmap/ends/data.ninja", dtype=np.int64, mode="r")
    d = np.memmap(f"{split}/mmap/data.ninja", dtype=np.float64, mode="r")
    return s, e, d


def unflatten(r, props):
    n = int(r[0])
    pbc = r[1:4].astype(bool)
    o = 4
    pos = r[o:o + 3 * n].reshape(n, 3); o += 3 * n
    cell = r[o:o + 9].reshape(3, 3); o += 9
    Z = r[o:o + n].astype(int); o += n
    out = {}
    for name in sorted(props):
        shape = tuple(n if d == "atom" else int(d) for d in props[name]["shape"])
        size = prod(shape)
        out[name] = np.asarray(r[o:o + size]).reshape(shape); o += size
    assert o == len(r), (o, len(r))
    return n, pbc, pos, cell, Z, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=240); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    props = yaml.safe_load(open(f"{a.split}/properties.yaml"))
    s, e, d = records(a.split)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(s), size=min(a.n, len(s)), replace=False))
    labels = {"rec": idx, "natoms": []}
    per_atom = {k for k, v in props.items() if v["shape"] and v["shape"][0] == "atom"}
    with open(a.out + ".extxyz", "w") as fx:
        for i in idx:
            n, pbc, pos, cell, Z, p = unflatten(np.asarray(d[s[i]:e[i]]), props)
            info = [f'pbc="{" ".join("T" if b else "F" for b in pbc)}"', f"rec={int(i)}"]
            if pbc.any():
                info.append('Lattice="' + " ".join(f"{x:.17g}" for x in cell.flatten()) + '"')
            for k in sorted(props):
                if k in per_atom or props[k]["storage"] == "atoms.calc":
                    continue
                v = p[k].flatten()
                if np.isnan(v).all():
                    continue
                if k in ("charge", "spin_multiplicity"):
                    info.append(f"{k}={int(round(v[0]))}")
                elif v.size == 1:
                    info.append(f"{k}={v[0]:.17g}")
                else:
                    info.append(f'{k}="' + " ".join(f"{x:.17g}" for x in v) + '"')
            fx.write(f"{n}\nProperties=species:S:1:pos:R:3 " + " ".join(info) + "\n")
            for z, x in zip(Z, pos):
                fx.write(f"{SYMBOLS[z]} {x[0]:.17g} {x[1]:.17g} {x[2]:.17g}\n")
            labels["natoms"].append(n)
            for k, v in p.items():
                labels.setdefault(k, []).append(v)
    out = {}
    for k, v in labels.items():
        if k in per_atom:
            out[k] = np.concatenate(v)
        else:
            out[k] = np.asarray(v)
    np.savez(a.out + ".labels.npz", **out)
    print(a.split, "->", a.out, len(idx), "frames of", len(s))


if __name__ == "__main__":
    main()
