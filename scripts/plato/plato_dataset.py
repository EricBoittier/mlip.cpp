"""extxyz -> the flat binary dataset of mlipcpp-train-plato.

    python scripts/plato/plato_dataset.py train.xyz train.bin --energy e_res --forces f [--stress stress]
        [--molecules bio_dimers_train.xyz]

Layout (little endian): b"PLATOD1\\0", int64 count, then per structure: int32 n,
int32 Z[n], f64 positions[n][3], f64 cell[3][3] (rows), uint8 pbc, uint8 has_forces,
uint8 has_stress, uint8 has_molecules, f64 energy, f64 forces[n][3] if has_forces, f64 stress[3][3]
if has_stress (ASE sign, eV/A^3), then if has_molecules: int32 split (first atom of the second
molecule), f64 formal charge of the first and of the second molecule.

--molecules reads split / charges (indexB, chargeA, chargeB) from a file with the same
structures in the same order (the LOREM archive's bio_dimers files); atoms must match.
"""
import argparse
import struct

import ase.io
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xyz")
    ap.add_argument("out")
    ap.add_argument("--energy", default="energy")
    ap.add_argument("--forces", default="forces")
    ap.add_argument("--stress", default="stress")
    ap.add_argument("--molecules", default=None)
    a = ap.parse_args()
    frames = ase.io.read(a.xyz, ":")
    mols = [None] * len(frames)
    if a.molecules:
        ref = ase.io.read(a.molecules, ":")
        if len(ref) != len(frames):
            raise SystemExit(f"{a.molecules}: {len(ref)} structures, {a.xyz}: {len(frames)}")
        for k, (x, y) in enumerate(zip(frames, ref)):
            if (x.numbers != y.numbers).any() or np.abs(x.positions - y.positions).max() > 1e-6:
                raise SystemExit(f"structure {k} differs between {a.xyz} and {a.molecules}")
            mols[k] = (int(y.info["indexB"]), float(y.info["chargeA"]), float(y.info["chargeB"]))
    with open(a.out, "wb") as f:
        f.write(b"PLATOD1\0" + struct.pack("<q", len(frames)))
        for at, mol in zip(frames, mols):
            n = len(at)
            E = at.info.get(a.energy)
            if E is None:
                E = at.get_potential_energy()
            F = at.arrays.get(a.forces)
            S = at.info.get(a.stress)
            pbc = bool(at.pbc.all())
            f.write(struct.pack("<i", n))
            f.write(np.asarray(at.numbers, "<i4").tobytes())
            f.write(np.asarray(at.positions, "<f8").tobytes())
            f.write(np.asarray(at.cell.array, "<f8").tobytes())
            f.write(struct.pack("<BBBB", pbc, F is not None, S is not None, mol is not None))
            f.write(struct.pack("<d", float(E)))
            if F is not None:
                f.write(np.asarray(F, "<f8").tobytes())
            if S is not None:
                S = np.asarray(S, float)
                if S.size == 6:  # Voigt xx yy zz yz xz xy
                    S = np.array([[S[0], S[5], S[4]], [S[5], S[1], S[3]], [S[4], S[3], S[2]]])
                f.write(np.asarray(S, "<f8").reshape(9).tobytes())
            if mol is not None:
                f.write(struct.pack("<idd", *mol))
    print(f"{a.out}: {len(frames)} structures")


if __name__ == "__main__":
    main()
