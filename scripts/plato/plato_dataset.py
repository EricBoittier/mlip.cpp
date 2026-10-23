"""extxyz -> the flat binary dataset of mlipcpp-train-plato.

    python scripts/plato/plato_dataset.py train.xyz train.bin --energy e_res --forces f [--stress stress]

Layout (little endian): b"PLATOD1\\0", int64 count, then per structure: int32 n,
int32 Z[n], f64 positions[n][3], f64 cell[3][3] (rows), uint8 pbc, uint8 has_forces,
uint8 has_stress, uint8 0, f64 energy, f64 forces[n][3] if has_forces, f64 stress[3][3]
if has_stress (ASE sign, eV/A^3).
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
    a = ap.parse_args()
    frames = ase.io.read(a.xyz, ":")
    with open(a.out, "wb") as f:
        f.write(b"PLATOD1\0" + struct.pack("<q", len(frames)))
        for at in frames:
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
            f.write(struct.pack("<BBBB", pbc, F is not None, S is not None, 0))
            f.write(struct.pack("<d", float(E)))
            if F is not None:
                f.write(np.asarray(F, "<f8").tobytes())
            if S is not None:
                S = np.asarray(S, float)
                if S.size == 6:  # Voigt xx yy zz yz xz xy
                    S = np.array([[S[0], S[5], S[4]], [S[5], S[1], S[3]], [S[4], S[3], S[2]]])
                f.write(np.asarray(S, "<f8").reshape(9).tobytes())
    print(f"{a.out}: {len(frames)} structures")


if __name__ == "__main__":
    main()
