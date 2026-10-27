"""OMol25 validation split (val.tar.gz, ase-db LMDB shards) -> a stratified charged / open-shell
evaluation set for PET-OMol-S / PET-OMol-LR. Only the public val split is read.

    python -I prep_omol25_val.py scan  VALDIR META.npz [--procs 32]
    python -I prep_omol25_val.py select META.npz SEL.json [--seed 0]
    python -I prep_omol25_val.py write VALDIR META.npz SEL.json OUT.npz

scan: per row of every *.aselmdb shard: shard, row id, data_id (domain), charge, spin, num_electrons,
num_atoms, s_squared, unrestricted, and from positions: components of the covalent graph (1.2 x ASE
covalent radii; alkali, alkaline-earth and noble-gas atoms never bond), min intermolecular distance,
min pair distance, and per component the Lowdin charge sum (largest |sum| over components and the
charge sum of the components that are > 8 A from all others).
spin check: info["spin"] is the multiplicity 2S+1 if num_electrons and spin - 1 have the same
parity for every row (reported).
select (seed): domains grouped as biomolecules / electrolytes / metal_complexes / neutral_control
(every other data_id, neutral singlets only). For the first three, charged or open-shell rows only;
per domain up to 300 "separated" rows (>= 2 components, min intermolecular distance >= 8 A and a
component with |Lowdin charge| >= 0.5), then the rest of the quota (1000 per domain, 600 control)
spread evenly over (charge class 0 / +-1 / |q|>=2) x (multiplicity 1 / 2 / >=3) x (atoms <30 / 30-100
/ >=100) cells, uniform within a cell.
write: the selected rows as arrays (n, Z, R, F, E, charge, spin_multiplicity, domain, data_id, shard,
row, tags) in the layout of prep_qcell_omol.py, for make_memmap.py and eval_omol.py.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from ase.data import covalent_radii

NOBOND = {2, 3, 10, 11, 12, 18, 19, 20, 36, 37, 38, 54, 55, 56, 86}
GROUPS = {"biomolecules": "biomolecules", "elytes": "electrolytes", "metal_complexes": "metal_complexes"}


def comps(Z, R):
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
    inter = lab[:, None] != lab[None]
    mi = float(D[inter].min()) if inter.any() else np.inf
    mp = float((D + 1e9 * np.eye(n)).min()) if n > 1 else np.inf
    # per component: distance to the nearest atom of another component
    far = np.zeros(c, bool)
    if c > 1:
        for k in range(c):
            m = lab == k
            far[k] = D[np.ix_(m, ~m)].min() >= 8.0
    return lab, c, mi, mp, far


def open_db(path):
    sys.path.insert(0, os.environ["OMOL_PYLIB"])
    from ase_db_backends.aselmdb import LMDBDatabase
    return LMDBDatabase(path, readonly=True)


def scan_shard(path):
    db = open_db(path)
    out = []
    for row in db.select():
        Z = np.asarray(row.numbers); R = np.asarray(row.positions)
        d = row.data if hasattr(row, "data") else {}
        kv = row.key_value_pairs
        info = {**kv, **d}
        lab, c, mi, mp, far = comps(Z, R)
        lw = info.get("lowdin_charges")
        qc = np.zeros(c)
        if lw is not None:
            lw = np.asarray(lw, float).reshape(-1)
            np.add.at(qc, lab, lw)
        out.append((os.path.basename(path), int(row.id), str(info.get("data_id")), int(info.get("charge")),
                    int(info.get("spin")), int(info.get("num_electrons", -1)), len(Z), float(info.get("s_squared", np.nan)),
                    bool(info.get("unrestricted", False)), c, mi, mp, float(np.abs(qc).max()) if c else 0.0,
                    float(qc[far].sum()) if far.any() else 0.0, int(far.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "select", "write"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "scan":
        valdir, meta = a.args
        shards = sorted(glob.glob(f"{valdir}/**/*.aselmdb", recursive=True))
        print(len(shards), "shards", flush=True)
        with Pool(a.procs) as p:
            rows = [r for part in p.imap(scan_shard, shards) for r in part]
        cols = list(zip(*rows))
        names = ["shard", "row", "data_id", "charge", "spin", "num_electrons", "natoms", "s_squared", "unrestricted",
                 "ncomp", "min_inter", "min_pair", "max_comp_lowdin", "far_comp_lowdin", "n_far_comp"]
        np.savez(meta, **{k: np.array(v) for k, v in zip(names, cols)})
        sp, ne = np.array(cols[4]), np.array(cols[5])
        ok = (ne >= 0) & ((ne % 2) == ((sp - 1) % 2))
        print(json.dumps({"rows": len(rows), "spin_parity_consistent_as_multiplicity": int(ok.sum()),
                          "rows_with_num_electrons": int((ne >= 0).sum()),
                          "data_id": {k: int(v) for k, v in zip(*np.unique(cols[2], return_counts=True))}}))
    elif a.mode == "select":
        meta, out = a.args
        m = np.load(meta)
        rng = np.random.default_rng(a.seed)
        did, q, sp, na = m["data_id"], m["charge"], m["spin"], m["natoms"]
        grp = np.array([GROUPS.get(x, "neutral_control") for x in did])
        qc = np.where(q == 0, 0, np.where(np.abs(q) == 1, 1, 2))
        sc = np.where(sp == 1, 1, np.where(sp == 2, 2, 3))
        nc = np.where(na < 30, 0, np.where(na < 100, 1, 2))
        sep = (m["ncomp"] >= 2) & (m["min_inter"] >= 8) & np.isfinite(m["min_inter"]) & (m["max_comp_lowdin"] >= 0.5)
        chosen, recipe = [], {}
        for g, quota in (("biomolecules", 1000), ("electrolytes", 1000), ("metal_complexes", 1000), ("neutral_control", 600)):
            if g == "neutral_control":
                pool = np.nonzero((grp == g) & (q == 0) & (sp == 1))[0]
                take_sep = np.zeros(0, int)
            else:
                pool = np.nonzero((grp == g) & ((q != 0) | (sp != 1)))[0]
                s = pool[sep[pool]]
                take_sep = rng.choice(s, min(300, len(s)), replace=False) if len(s) else np.zeros(0, int)
            rest = np.setdiff1d(pool, take_sep)
            cells = defaultdict(list)
            for k in rest:
                cells[(qc[k], sc[k], nc[k])].append(k)
            left = quota - len(take_sep)
            picked = []
            keys = sorted(cells)
            while left > 0 and keys:  # even fill: one round over non-empty cells at a time
                per = max(1, left // len(keys))
                nxt = []
                for c in keys:
                    av = cells[c]
                    t = min(per, len(av), left)
                    if t:
                        idx = rng.choice(len(av), t, replace=False)
                        picked += [av[i] for i in idx]
                        cells[c] = [av[i] for i in range(len(av)) if i not in set(idx.tolist())]
                        left -= t
                    if cells[c]:
                        nxt.append(c)
                    if left <= 0:
                        break
                keys = nxt
            sel = np.concatenate([take_sep, np.array(picked, int)])
            chosen += sel.tolist()
            recipe[g] = {"pool": int(len(pool)), "separated_available": int(sep[pool].sum()), "separated_taken": int(len(take_sep)),
                         "selected": int(len(sel)), "cells": {f"q{a_}s{b_}n{c_}": int(sum((qc[k], sc[k], nc[k]) == (a_, b_, c_) for k in sel))
                                                              for (a_, b_, c_) in sorted({(qc[k], sc[k], nc[k]) for k in sel})}}
        json.dump({"seed": a.seed, "indices": sorted(int(k) for k in chosen), "recipe": recipe}, open(out, "w"), indent=1)
        print(json.dumps(recipe))
    else:
        valdir, meta, selp, out = a.args
        m = np.load(meta)
        sel = json.load(open(selp))["indices"]
        by = defaultdict(list)
        for k in sel:
            by[str(m["shard"][k])].append(k)
        paths = {os.path.basename(p): p for p in glob.glob(f"{valdir}/**/*.aselmdb", recursive=True)}
        rows = {}
        for sh, ks in by.items():
            db = open_db(paths[sh])
            for k in ks:
                row = db._get_row(int(m["row"][k]))
                rows[k] = (np.asarray(row.numbers, np.int32), np.asarray(row.positions, float), np.asarray(row.forces, float),
                           float(row.energy))
        ks = sel
        Z = [rows[k][0] for k in ks]
        grp = np.array([GROUPS.get(str(m["data_id"][k]), "neutral_control") for k in ks])
        lab = [comps(rows[k][0], rows[k][1])[0] for k in ks]
        np.savez(out, n=np.array([len(z) for z in Z]), Z=np.concatenate(Z), R=np.concatenate([rows[k][1] for k in ks]),
                 F=np.concatenate([rows[k][2] for k in ks]), E=np.array([rows[k][3] for k in ks]),
                 comp=np.concatenate(lab).astype(np.int16),
                 charge=m["charge"][ks], spin_multiplicity=m["spin"][ks], domain=grp, data_id=m["data_id"][ks],
                 shard=m["shard"][ks], row=m["row"][ks], ncomp=m["ncomp"][ks], min_inter=m["min_inter"][ks],
                 min_pair=m["min_pair"][ks], max_comp_lowdin=m["max_comp_lowdin"][ks], far_comp_lowdin=m["far_comp_lowdin"][ks])
        print(out, len(ks))


if __name__ == "__main__":
    main()
