#!/usr/bin/env python3
"""Predictions of a metatrain PET (a checkpoint, e.g. from gguf_to_checkpoint.py, or a
make_init.py dump) through its exported metatomic model, on structures of a training-format
split: the reference side of the mlip.cpp round trip (mlipcpp-predict writes the same files,
compare_predictions.py compares them).

    python predict_metatomic.py model.ckpt <split dir> ids.npy <out prefix> [--dtype float64]

Writes <out>.energy.bin (f64 [n], eV) and <out>.nc_forces.bin (f32, or f64 with
--dtype float64, [sum n_atoms, 3], eV/A) in the order of ids.npy.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import ase
import numpy as np
import torch
from metatomic_ase import MetatomicCalculator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_pytorch.export_pet_full import load_local_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model"); ap.add_argument("split"); ap.add_argument("ids"); ap.add_argument("out")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    a = ap.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = load_local_model(a.model).to(getattr(torch, a.dtype)).eval()
        calc = MetatomicCalculator(model.export(), non_conservative=True, do_gradients_with_energy=False,
                                   device="cpu")
    d = a.split
    na = np.load(f"{d}/na.npy")
    x = np.memmap(f"{d}/x.bin", np.float32, "r").reshape(-1, 3)
    z = np.memmap(f"{d}/a.bin", np.int32, "r")
    c = np.memmap(f"{d}/c.bin", np.float32, "r").reshape(-1, 3, 3)
    ids = np.load(a.ids)
    energy, forces = [], []
    t0 = time.time()
    for s in ids:
        lo, hi = na[s], na[s + 1]
        cell = c[s].astype(np.float64)
        atoms = ase.Atoms(numbers=z[lo:hi], positions=x[lo:hi].astype(np.float64), cell=cell,
                          pbc=np.any(cell != 0, axis=1))  # as MemmapSplit / metatrain
        atoms.calc = calc
        energy.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
    np.asarray(energy, np.float64).tofile(f"{a.out}.energy.bin")
    np.concatenate(forces).astype(np.float64 if a.dtype == "float64" else np.float32).tofile(f"{a.out}.nc_forces.bin")
    print(f"{len(ids)} structures in {time.time() - t0:.1f} s ({a.dtype}) -> {a.out}.*")


if __name__ == "__main__":
    main()
