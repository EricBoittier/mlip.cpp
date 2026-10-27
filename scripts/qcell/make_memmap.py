"""prep_qcell_omol.py / prep_omol25_val.py arrays -> mlip.cpp PET memmap split (+ extxyz for mlipcpp-eval).

    python -I make_memmap.py SPLIT.npz OUTDIR [--baseline baseline.json] [--xyz OUT.xyz] [--sample N --seed S]

Layout of scripts/train/xyz_to_memmap.py (metatrain MemmapWriter): ns.npy, na.npy (int64 offsets),
x.bin (f32 positions), a.bin (i32 Z), c.bin (f32 zero cell: non-periodic), energy.bin (f32),
non_conservative_force.bin and energy_forces.bin (f32 forces, eV/A), charge.bin and
spin_multiplicity.bin (f32 [ns, 1]). energy = E - b.n (eV; b = baseline.json "composition",
0 without --baseline): float32 cannot hold QCell total energies (up to 3e5 eV) to meV.
--xyz also writes the same structures as extxyz (non-periodic) with info E (total, eV),
e_res, charge, spin_multiplicity, and arrays forces. --sample N: N structures, default_rng(S),
positions in SPLIT.npz to OUTDIR/sample_idx.txt.
"""
import argparse
import json
from pathlib import Path

import numpy as np

SYM = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
       "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
       "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()


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
    n, Z, R, F, E = d["n"], d["Z"], d["R"], d["F"], d["E"]
    q, s = d["charge"], d["spin_multiplicity"]
    off = np.concatenate([[0], np.cumsum(n)])
    b = np.zeros(119)
    if a.baseline:
        for z, v in json.load(open(a.baseline))["composition"].items():
            b[int(z)] = v
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ns.npy").exists():
        raise SystemExit(f"{out} already holds a split")
    sel = np.arange(len(n))
    if a.sample:
        sel = np.sort(np.random.default_rng(a.seed).choice(len(n), min(a.sample, len(n)), replace=False))
        np.savetxt(out / "sample_idx.txt", sel, fmt="%d")
    fh = {k: open(out / f"{k}.bin", "wb") for k in ("x", "a", "c", "energy", "non_conservative_force", "energy_forces")}
    xf = open(a.xyz, "w") if a.xyz else None
    zero = np.zeros((3, 3), np.float32)
    for k in sel:
        sl = slice(off[k], off[k + 1]); z = Z[sl]
        e = float(E[k] - b[z].sum())
        R[sl].astype(np.float32).tofile(fh["x"]); z.astype(np.int32).tofile(fh["a"]); zero.tofile(fh["c"])
        np.float32(e).tofile(fh["energy"])
        F[sl].astype(np.float32).tofile(fh["non_conservative_force"]); F[sl].astype(np.float32).tofile(fh["energy_forces"])
        if xf:
            xf.write(f"{len(z)}\nProperties=species:S:1:pos:R:3:forces:R:3 E={float(E[k])!r} e_res={e!r} "
                     f"charge={int(q[k])} spin_multiplicity={int(s[k])} pbc=\"F F F\"\n")
            for zz, r, f in zip(z, R[sl], F[sl]):
                xf.write(f"{SYM[zz]} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f} {f[0]:.8f} {f[1]:.8f} {f[2]:.8f}\n")
    for f in fh.values():
        f.close()
    if xf:
        xf.close()
    q[sel].astype(np.float32).reshape(-1, 1).tofile(out / "charge.bin")
    s[sel].astype(np.float32).reshape(-1, 1).tofile(out / "spin_multiplicity.bin")
    np.save(out / "ns.npy", np.int64(len(sel)))
    np.save(out / "na.npy", np.concatenate([[0], np.cumsum(n[sel])]).astype(np.int64))
    print(f"{out}: {len(sel)} structures, {int(n[sel].sum())} atoms, charge [{q[sel].min()}, {q[sel].max()}], "
          f"spin multiplicity [{s[sel].min()}, {s[sel].max()}]")


if __name__ == "__main__":
    main()
