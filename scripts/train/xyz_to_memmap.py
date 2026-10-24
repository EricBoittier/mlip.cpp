#!/usr/bin/env python3
"""Write an extended-XYZ file as a metatrain MemmapDataset split for mlipcpp-train.

The layout of metatrain's MemmapWriter (utils/data/writers/memmap.py): ns.npy,
na.npy (int64 offsets), x.bin, a.bin, c.bin (zero cell for non-periodic
structures), energy.bin, and the forces twice: non_conservative_force.bin
(what mlipcpp-train reads, and the direct-force target the init needs) and
energy_forces.bin (the energy target's position gradients, for metatrain).

    python xyz_to_memmap.py data/train.xyz train --energy-key e_res --forces-key f
"""
import argparse
from pathlib import Path

import numpy as np
from ase.io import iread

ap = argparse.ArgumentParser()
ap.add_argument("xyz")
ap.add_argument("out", help="directory (created, must not hold a split already)")
ap.add_argument("--energy-key", default="energy")
ap.add_argument("--forces-key", default="forces")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if (out / "ns.npy").exists():
    raise SystemExit(f"{out} already holds a split")
files = {k: open(out / f"{k}.bin", "wb") for k in ("x", "a", "c", "energy", "non_conservative_force", "energy_forces")}
counts = []
for atoms in iread(args.xyz, ":"):
    counts.append(len(atoms))
    atoms.positions.astype(np.float32).tofile(files["x"])
    atoms.numbers.astype(np.int32).tofile(files["a"])
    cell = atoms.cell.array if atoms.pbc.any() else np.zeros((3, 3))
    if atoms.pbc.any() and not atoms.pbc.all():
        cell = cell * atoms.pbc[:, None]  # non-periodic directions: zero rows, as metatrain
    cell.astype(np.float32).tofile(files["c"])
    np.float32(atoms.info[args.energy_key]).tofile(files["energy"])
    f = atoms.arrays[args.forces_key].astype(np.float32)
    f.tofile(files["non_conservative_force"])
    f.tofile(files["energy_forces"])
for fh in files.values():
    fh.close()
np.save(out / "ns.npy", np.int64(len(counts)))
np.save(out / "na.npy", np.cumsum([0] + counts).astype(np.int64))
print(f"{args.xyz}: {len(counts)} structures, {sum(counts)} atoms -> {out}")
