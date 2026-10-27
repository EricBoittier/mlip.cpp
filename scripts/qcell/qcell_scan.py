"""Stream-parse the QCell extxyz files (no ASE, no eval) and summarise them.

    python -I qcell_scan.py OUT.npz FILE.xyz [FILE.xyz ...]

Per structure: subset id, n atoms, element counts (Z<=100), energy (eV), formation_energy,
charge, pbc flag, max |F| (eV/A), max |pos| extent. Writes OUT.npz and prints a summary.
"""
import re
import sys

import numpy as np

SYM = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
       "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe").split()
Z = {s: i for i, s in enumerate(SYM)}
KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def main():
    out, files = sys.argv[1], sys.argv[2:]
    rows = {k: [] for k in ("subset", "n", "energy", "formation", "charge", "pbc", "fmax", "extent", "lattice")}
    counts = []
    keys = {}
    for si, fn in enumerate(files):
        with open(fn) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                n = int(line)
                hdr = dict(KV.findall(f.readline()))
                ks = tuple(sorted(hdr))
                keys.setdefault((fn, ks, hdr.get("Properties")), 0)
                keys[(fn, ks, hdr.get("Properties"))] += 1
                c = np.zeros(101, np.int32)
                P = np.empty((n, 3)); F = np.empty((n, 3))
                for i in range(n):
                    p = f.readline().split()
                    c[Z[p[0]]] += 1
                    P[i] = p[1:4]; F[i] = p[4:7]
                counts.append(c)
                rows["subset"].append(si); rows["n"].append(n)
                rows["energy"].append(float(hdr["energy"])); rows["formation"].append(float(hdr.get("formation_energy", "nan")))
                rows["charge"].append(float(hdr.get("charge", "nan")))
                rows["pbc"].append("T" in hdr.get("pbc", "F"))
                rows["lattice"].append("Lattice" in hdr)
                rows["fmax"].append(float(np.abs(F).max())); rows["extent"].append(float((P.max(0) - P.min(0)).max()))
    d = {k: np.array(v) for k, v in rows.items()}
    d["counts"] = np.array(counts)
    d["files"] = np.array(files)
    np.savez_compressed(out, **d)
    for (fn, ks, props), m in keys.items():
        print("keys", fn.split("/")[-1], m, ks, props)
    for si, fn in enumerate(files + ["ALL"]):
        s = d["subset"] == si if fn != "ALL" else np.ones(len(d["n"]), bool)
        el = [SYM[z] for z in np.nonzero(d["counts"][s].sum(0))[0]]
        ch = dict(zip(*np.unique(d["charge"][s], return_counts=True)))
        print(f"{fn.split('/')[-1]}: {s.sum()} structures, atoms {d['n'][s].sum()} (n {d['n'][s].min()}-{d['n'][s].max()}, "
              f"mean {d['n'][s].mean():.1f}), elements {el}, charges {ch}, pbc {d['pbc'][s].sum()}, lattice {d['lattice'][s].sum()}, "
              f"fmax max {d['fmax'][s].max():.2f} p99.9 {np.quantile(d['fmax'][s], 0.999):.2f} eV/A, extent max {d['extent'][s].max():.1f} A")


if __name__ == "__main__":
    main()
