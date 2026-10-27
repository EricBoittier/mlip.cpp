"""QCell extxyz (Zenodo 18385612, qcell_xyz.tar.gz) -> seeded split + compact arrays.

    python -I prep_qcell.py OUTDIR NA.xyz LIPIDS.xyz SUGARS.xyz IONS.xyz DIMERS.xyz

No split is published (Zenodo record, GitHub repo, paper), so: per subset, a
numpy default_rng(0) permutation of the structures in file order -> 85% train,
5% val, 10% test. Structures with max |F| > FMAX eV/A or a coordinate extent
> EXTENT A, or with no atom pair closer than the model cutoff (5 A: two-atom DES370K
dimers at long range, no short-range edge at all, which mlip.cpp's graph cannot take) are
excluded from all three sets (listed in OUTDIR/excluded.txt).
Units as distributed: energy (total, PBE0+MBD(-NL), FHI-aims) eV, forces eV/A,
positions A. Every structure is non-periodic in QCell; here each is centred in
a cubic box of side L = max(30, 10 ceil((extent + 15) / 10)) A with PBC, as
the bio_dimers training (30 A box) that the init checkpoint saw, so Plato's
periodic mesh coupling and Ewald LES stay what the model was trained with.

OUTDIR/{train,val,test}.npz: n, Z, R, F (flattened over atoms), E (eV),
charge, L, subset, index (frame index within the subset file);
OUTDIR/{split}_index.txt: "subset index"; OUTDIR/prep.json.
"""
import json
import sys

import numpy as np

SYM = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
       "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe").split()
ZOF = {s: i for i, s in enumerate(SYM)}
NAMES = ["nucleic_acids", "lipids", "sugars", "ions_water", "dimers"]
FMAX, EXTENT, SEED, CUTOFF = 20.0, 60.0, 0, 5.0
FRAC = {"train": 0.85, "val": 0.05, "test": 0.10}


def header(line):
    out, i = {}, 0
    import shlex
    for tok in shlex.split(line):
        k, _, v = tok.partition("=")
        out[k] = v
    return out


def read(fn):
    frames = []
    with open(fn) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line)
            h = header(f.readline())
            assert h["Properties"].startswith("species:S:1:pos:R:3:forces:R:3"), h["Properties"]
            Z = np.empty(n, np.int32)
            A = np.empty((n, 6))
            for i in range(n):
                p = f.readline().split()
                Z[i] = ZOF[p[0]]
                A[i] = p[1:7]
            frames.append((Z, A[:, :3].copy(), A[:, 3:].copy(), float(h["energy"]), float(h["charge"])))
    return frames


def main():
    out, files = sys.argv[1], sys.argv[2:]
    assert len(files) == 5
    sets = {k: [] for k in FRAC}
    excluded = []
    info = {"seed": SEED, "fractions": FRAC, "fmax_cut_ev_a": FMAX, "extent_cut_a": EXTENT, "no_edge_cutoff_a": CUTOFF, "files": files,
            "subsets": NAMES, "counts": {}}
    for si, fn in enumerate(files):
        fr = read(fn)
        perm = np.random.default_rng(SEED).permutation(len(fr))
        ntr, nva = int(round(FRAC["train"] * len(fr))), int(round(FRAC["val"] * len(fr)))
        part = {"train": perm[:ntr], "val": perm[ntr:ntr + nva], "test": perm[ntr + nva:]}
        cnt = {}
        for name, idx in part.items():
            kept = 0
            for i in np.sort(idx):
                Z, R, F, E, q = fr[i]
                ext = float((R.max(0) - R.min(0)).max())
                noedge = len(Z) <= 8 and (np.linalg.norm(R[:, None] - R[None], axis=-1) + 1e9 * np.eye(len(Z))).min() >= CUTOFF
                if np.abs(F).max() > FMAX or ext > EXTENT or noedge:
                    excluded.append((NAMES[si], int(i), name, float(np.abs(F).max()), ext))
                    continue
                L = max(30.0, 10.0 * np.ceil((ext + 15.0) / 10.0))
                R = R - 0.5 * (R.max(0) + R.min(0)) + 0.5 * L
                sets[name].append((Z, R, F, E, q, L, si, int(i)))
                kept += 1
            cnt[name] = kept
        info["counts"][NAMES[si]] = {"total": len(fr), **cnt}
        print(NAMES[si], info["counts"][NAMES[si]], flush=True)
        del fr
    for name, rows in sets.items():
        n = np.array([len(r[0]) for r in rows])
        np.savez(f"{out}/{name}.npz", n=n, Z=np.concatenate([r[0] for r in rows]),
                 R=np.concatenate([r[1] for r in rows]), F=np.concatenate([r[2] for r in rows]),
                 E=np.array([r[3] for r in rows]), charge=np.array([r[4] for r in rows]),
                 L=np.array([r[5] for r in rows]), subset=np.array([r[6] for r in rows]),
                 index=np.array([r[7] for r in rows]))
        np.savetxt(f"{out}/{name}_index.txt", np.array([[r[6], r[7]] for r in rows]), fmt="%d",
                   header="subset(0=nucleic_acids,1=lipids,2=sugars,3=ions_water,4=dimers) frame_index_in_file")
        info["counts"][name] = {"structures": len(rows), "atoms": int(n.sum()),
                                "boxes": {str(int(k)): int(v) for k, v in zip(*np.unique([r[5] for r in rows], return_counts=True))}}
        print(name, info["counts"][name], flush=True)
    with open(f"{out}/excluded.txt", "w") as f:
        f.write("# subset frame_index split max|F|(eV/A) extent(A)\n")
        for e in excluded:
            f.write("%s %d %s %.3f %.2f\n" % e)
    info["excluded"] = len(excluded)
    json.dump(info, open(f"{out}/prep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
