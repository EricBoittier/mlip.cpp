"""QCell (Zenodo 18385612, qcell_xyz.tar.gz) -> split + long-range tags for PET-OMol fine-tuning.

    python -I prep_qcell_omol.py OUTDIR NA.xyz LIPIDS.xyz SUGARS.xyz IONS.xyz DIMERS.xyz

Split: the same seeded assignment as prep_qcell.py (per subset, numpy default_rng(0)
permutation of the file order, first 85% train, next 5% val, rest test). Cuts: max |F| >
20 eV/A or coordinate extent > 60 A (excluded everywhere, OUTDIR/excluded.txt). No cutoff-
based cut: structures without any pair within 5 A are kept and tagged.

Tags per structure: components of the covalent graph (bond if d < 1.2 (r_a + r_b), ASE
covalent radii; He Li Ne Na Mg Ar K Ca Kr Xe never bond, so ions are their own component),
min_inter (smallest distance between atoms of different components, inf if one), max pair
distance, no_pair_5 / no_pair_8 (no atom pair closer than 5 / 8 A). Dimers (DES370K): per-
component formula; monomer charges from the integer least-squares solution of q_A + q_B = Q
over all two-component dimers (one unknown per formula; consistency checked and reported);
charge class nn / cn / cc (rounded to the parity of the monomer's nuclear charge: closed shell); scan id = run of consecutive frames with monomer A fixed and B
rigid (centred coordinates equal), scan coordinate = centroid distance.

Curves holdout: N_CURVES scans of charged dimers (half cc, half cn, >= 6 points, seed 0)
leave train/val (their points form OUTDIR/curves.npz; their test points also stay in test).
"""
import json
import sys
from collections import defaultdict

import numpy as np
from ase.data import covalent_radii

SYM = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
       "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe").split()
ZOF = {s: i for i, s in enumerate(SYM)}
NAMES = ["nucleic_acids", "lipids", "sugars", "ions_water", "dimers"]
NOBOND = {2, 3, 10, 11, 12, 18, 19, 20, 36, 54}
FMAX, EXTENT, SEED, N_CURVES = 20.0, 60.0, 0, 20
FRAC = {"train": 0.85, "val": 0.05, "test": 0.10}


def header(line):
    import shlex
    return dict(tok.partition("=")[::2] for tok in shlex.split(line))


def read(fn):
    out = []
    with open(fn) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line)
            h = header(f.readline())
            Z = np.empty(n, np.int32); A = np.empty((n, 6))
            for i in range(n):
                p = f.readline().split()
                Z[i] = ZOF[p[0]]; A[i] = p[1:7]
            out.append((Z, A[:, :3].copy(), A[:, 3:].copy(), float(h["energy"]), float(h["formation_energy"]),
                        int(round(float(h["charge"])))))
    return out


def components(Z, R):
    n = len(Z)
    D = np.linalg.norm(R[:, None] - R[None], axis=-1)
    rc = covalent_radii[Z]
    B = D < 1.2 * (rc[:, None] + rc[None])
    nb = np.isin(Z, list(NOBOND))
    B[nb, :] = False; B[:, nb] = False
    np.fill_diagonal(B, False)
    lab = -np.ones(n, int); c = 0
    for s in range(n):
        if lab[s] >= 0:
            continue
        st = [s]; lab[s] = c
        while st:
            i = st.pop()
            for j in np.nonzero(B[i] & (lab < 0))[0]:
                lab[j] = c; st.append(j)
        c += 1
    off = D + np.eye(n) * 1e9
    inter = lab[:, None] != lab[None]
    mi = float(D[inter].min()) if inter.any() else np.inf
    return lab, c, mi, float(off.min()) if n > 1 else np.inf, float(D.max())


def formula(Z):
    c = np.bincount(Z, minlength=55)
    return "".join(f"{SYM[z]}{c[z]}" for z in np.nonzero(c)[0])


def main():
    out, files = sys.argv[1], sys.argv[2:]
    rows, excluded = [], []
    info = {"seed": SEED, "fractions": FRAC, "fmax_cut_ev_a": FMAX, "extent_cut_a": EXTENT, "files": files,
            "subsets": NAMES, "counts": {}}
    for si, fn in enumerate(files):
        fr = read(fn)
        perm = np.random.default_rng(SEED).permutation(len(fr))
        ntr, nva = int(round(FRAC["train"] * len(fr))), int(round(FRAC["val"] * len(fr)))
        split = np.empty(len(fr), "<U5")
        split[perm[:ntr]] = "train"; split[perm[ntr:ntr + nva]] = "val"; split[perm[ntr + nva:]] = "test"
        prevA, prevB, scan = None, None, -1
        for i, (Z, R, F, E, Ef, q) in enumerate(fr):
            ext = float((R.max(0) - R.min(0)).max())
            lab, nc, mi, mind, maxd = components(Z, R)
            r = dict(Z=Z, R=R, F=F, E=E, Ef=Ef, q=q, subset=si, index=i, split=str(split[i]), lab=lab, ncomp=nc,
                     min_inter=mi, min_pair=mind, max_pair=maxd, scan=-1, scan_r=np.nan, fA="", fB="")
            if si == 4 and nc == 2:
                a, b = lab == lab[0], lab != lab[0]
                r["fA"], r["fB"] = formula(Z[a]), formula(Z[b])
                RA = R[a]; RB = R[b] - R[b].mean(0)
                same = prevA is not None and prevA.shape == RA.shape and prevB.shape == RB.shape and \
                    np.abs(prevA - RA).max() < 1e-6 and np.abs(prevB - RB).max() < 1e-3
                if not same:
                    scan += 1
                prevA, prevB = RA, RB
                r["scan"] = scan; r["scan_r"] = float(np.linalg.norm(R[b].mean(0) - RA.mean(0)))
            else:
                prevA = prevB = None
            if np.abs(F).max() > FMAX or ext > EXTENT:
                excluded.append((NAMES[si], i, r["split"], float(np.abs(F).max()), ext))
                continue
            rows.append(r)
        print(NAMES[si], len(fr), flush=True)
        del fr
    # monomer charges of two-component dimers: q_A + q_B = Q, one integer unknown per formula
    dim = [r for r in rows if r["subset"] == 4 and r["ncomp"] == 2]
    forms = sorted({r["fA"] for r in dim} | {r["fB"] for r in dim})
    fi = {f: k for k, f in enumerate(forms)}
    eqs = sorted({(fi[r["fA"]], fi[r["fB"]], r["q"]) for r in dim})
    M = np.zeros((len(eqs), len(forms))); y = np.zeros(len(eqs))
    for k, (a, b, q) in enumerate(eqs):
        M[k, a] += 1; M[k, b] += 1; y[k] = q
    qf = np.linalg.lstsq(M, y, rcond=None)[0]
    # closed shell (QCell: spin none, all electron counts even): a monomer's charge has the parity of its
    # nuclear charge sum, so round to the nearest integer of that parity (fixes the min-norm Q/2 splits of
    # formula pairs that only ever occur together)
    zsum = np.array([sum(ZOF[t] * int(c) for t, c in __import__("re").findall(r"([A-Z][a-z]?)(\d+)", f)) for f in forms])
    par = zsum % 2
    qi = (par + 2 * np.round((qf - par) / 2)).astype(int)
    bad = int((np.abs(M @ qi - y) > 0.5).sum())
    info["monomer_charge_solve"] = {"formulas": len(forms), "equations": len(eqs), "inconsistent_equations": bad,
                                    "max_rounding": float(np.abs(qf - qi).max()),
                                    "charged_formulas": {f: int(qi[fi[f]]) for f in forms if qi[fi[f]] != 0}}
    for r in rows:
        r["qA"] = r["qB"] = 0; r["cls"] = ""
        if r["subset"] == 4:
            if r["ncomp"] == 2:
                r["qA"], r["qB"] = int(qi[fi[r["fA"]]]), int(qi[fi[r["fB"]]])
                ok = r["qA"] + r["qB"] == r["q"]
                r["cls"] = ("cc" if r["qA"] and r["qB"] else "cn" if (r["qA"] or r["qB"]) else "nn") if ok else "unresolved"
            else:
                r["cls"] = f"{r['ncomp']}comp"
    # curves holdout
    scans = defaultdict(list)
    for k, r in enumerate(rows):
        if r["scan"] >= 0:
            scans[r["scan"]].append(k)
    rng = np.random.default_rng(SEED)
    chosen = []
    for cls in ("cc", "cn"):
        cand = sorted(s for s, ks in scans.items() if len(ks) >= 6 and all(rows[k]["cls"] == cls for k in ks))
        chosen += [int(s) for s in rng.choice(cand, min(N_CURVES // 2, len(cand)), replace=False)]
        info.setdefault("curves_candidates", {})[cls] = len(cand)
    curve_k = sorted(k for s in chosen for k in scans[s])
    info["curves_scans"] = chosen
    sets = {"train": [], "val": [], "test": [], "curves": curve_k}
    cset = set(curve_k)
    moved = defaultdict(int)
    for k, r in enumerate(rows):
        if k in cset and r["split"] in ("train", "val"):
            moved[r["split"]] += 1
            continue
        sets[r["split"]].append(k)
    info["curves_moved_out_of"] = dict(moved)
    for name, ks in sets.items():
        R_ = [rows[k] for k in ks]
        n = np.array([len(r["Z"]) for r in R_])
        np.savez(f"{out}/{name}.npz", n=n, Z=np.concatenate([r["Z"] for r in R_]), R=np.concatenate([r["R"] for r in R_]),
                 F=np.concatenate([r["F"] for r in R_]), comp=np.concatenate([r["lab"] for r in R_]).astype(np.int16),
                 E=np.array([r["E"] for r in R_]), E_formation=np.array([r["Ef"] for r in R_]),
                 charge=np.array([r["q"] for r in R_]), spin_multiplicity=np.ones(len(R_), int),
                 subset=np.array([r["subset"] for r in R_]), index=np.array([r["index"] for r in R_]),
                 split=np.array([r["split"] for r in R_]), ncomp=np.array([r["ncomp"] for r in R_]),
                 min_inter=np.array([r["min_inter"] for r in R_]), min_pair=np.array([r["min_pair"] for r in R_]),
                 max_pair=np.array([r["max_pair"] for r in R_]), dimer_class=np.array([r["cls"] for r in R_]),
                 qA=np.array([r["qA"] for r in R_]), qB=np.array([r["qB"] for r in R_]),
                 scan=np.array([r["scan"] for r in R_]), scan_r=np.array([r["scan_r"] for r in R_]))
        np.savetxt(f"{out}/{name}_index.txt", np.array([[r["subset"], r["index"]] for r in R_]), fmt="%d",
                   header="subset(0=nucleic_acids,1=lipids,2=sugars,3=ions_water,4=dimers) frame_index_in_file")
        mi = np.array([r["min_inter"] for r in R_]); mp = np.array([r["min_pair"] for r in R_])
        info["counts"][name] = {"structures": len(R_), "atoms": int(n.sum()),
                                "per_subset": {NAMES[s]: int(sum(r["subset"] == s for r in R_)) for s in range(5)},
                                "charged": int(sum(r["q"] != 0 for r in R_)),
                                "no_pair_5A": int((mp >= 5).sum()), "no_pair_8A": int((mp >= 8).sum()),
                                "min_inter_ge_8A": int((mi >= 8).sum() - np.isinf(mi).sum()),
                                "dimer_class": {c: int(sum(r["cls"] == c for r in R_)) for c in sorted({r["cls"] for r in R_}) if c}}
        print(name, json.dumps(info["counts"][name]), flush=True)
    with open(f"{out}/excluded.txt", "w") as f:
        f.write("# subset frame_index split max|F|(eV/A) extent(A)\n")
        for e in excluded:
            f.write("%s %d %s %.3f %.2f\n" % e)
    info["excluded"] = len(excluded)
    json.dump(info, open(f"{out}/prep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
