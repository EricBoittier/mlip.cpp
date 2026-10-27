"""prep_qcell.py arrays -> mlipcpp-train-plato .bin (PLATOD1) and extxyz for `mlipcpp-plato dump`.

    python -I make_bin.py SPLIT.npz OUT.bin [--baseline baseline.json] [--xyz OUT.xyz] [--sample N --seed S]

Target energy e_res = E - b.n (b = baseline.json "composition" {Z: eV}; 0 without --baseline),
forces F (eV/A); cubic box L with PBC; no molecule/charge record (Plato without E_chi takes no
total charge). --sample N writes only N structures drawn with default_rng(S) (for the baseline
fit dumps), and OUT.idx.txt with their positions in SPLIT.npz.
"""
import argparse
import json
import struct

import numpy as np

SYM = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
       "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--xyz", default=None)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.npz)
    n = d["n"]
    off = np.concatenate([[0], np.cumsum(n)])
    b = np.zeros(119)
    if a.baseline:
        for z, v in json.load(open(a.baseline))["composition"].items():
            b[int(z)] = v
    sel = np.arange(len(n))
    if a.sample:
        sel = np.sort(np.random.default_rng(a.seed).choice(len(n), min(a.sample, len(n)), replace=False))
        np.savetxt(a.out + ".idx.txt", sel, fmt="%d")
    Z, R, F, E, L = d["Z"], d["R"], d["F"], d["E"], d["L"]
    xf = open(a.xyz, "w") if a.xyz else None
    with open(a.out, "wb") as f:
        f.write(b"PLATOD1\0" + struct.pack("<q", len(sel)))
        for k in sel:
            s = slice(off[k], off[k + 1])
            z = Z[s]
            e = float(E[k] - b[z].sum())
            cell = np.eye(3) * L[k]
            f.write(struct.pack("<i", len(z)))
            f.write(np.asarray(z, "<i4").tobytes())
            f.write(np.asarray(R[s], "<f8").tobytes())
            f.write(np.asarray(cell, "<f8").tobytes())
            f.write(struct.pack("<BBBB", 1, 1, 0, 0))
            f.write(struct.pack("<d", e))
            f.write(np.asarray(F[s], "<f8").tobytes())
            if xf:
                xf.write(f"{len(z)}\nLattice=\"{L[k]} 0.0 0.0 0.0 {L[k]} 0.0 0.0 0.0 {L[k]}\" "
                         f"Properties=species:S:1:pos:R:3:f:R:3 e_res={e!r} E_total={float(E[k])!r} "
                         f"charge={float(d['charge'][k])} subset={int(d['subset'][k])} qcell_index={int(d['index'][k])} "
                         f"pbc=\"T T T\"\n")
                for zz, r, ff in zip(z, R[s], F[s]):
                    xf.write(f"{SYM[zz]} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f} {ff[0]:.8f} {ff[1]:.8f} {ff[2]:.8f}\n")
    if xf:
        xf.close()
    print(f"{a.out}: {len(sel)} structures, {int(n[sel].sum())} atoms")


if __name__ == "__main__":
    main()
