#!/usr/bin/env python3
"""OMol25 npz shards (omol_universal/scripts/omol_to_npz.py layout) -> mlipcpp-train splits.

The same structures, energies and forces as the mmml (JAX) SpookyPhysNet trainer
(scripts/train_so3lr_spooky_extxyz.py --cache-path, caches built by
omol_universal/scripts/npz_to_spooky_cache.py):

- selection: --n-structures N --seed S picks exactly the rows npz_to_spooky_cache.py
  --n-structures N --seed S picks (np.random.default_rng(S).choice over the sorted
  shards), in the same order;
- energy: the shards' E (E_DFT - sum E_iso, eV) minus the linear reference
  sum_Z n_Z b_Z + b_Q Q stored in a cache's metadata.json (--linear-ref-from), the
  target the JAX model is trained on; forces unchanged (eV/A);
- split: --valid-fraction f --split-seed s reproduces the JAX trainer's
  split_indices(n, f, s) on the selection -> train/ and val/; without it every
  selected structure goes to --split-name.

Each split is a metatrain MemmapDataset directory as mlipcpp-train reads it
(src/train/dataset.cpp: x.bin a.bin c.bin energy.bin non_conservative_force.bin
float32/int32, ns.npy, na.npy; charge.bin and spin_multiplicity.bin float32 [ns, 1],
the SpookyPhysNet conditioning inputs), plus source.npz (shard index, row in the
shard, row in the selection, the shard's LMDB key "dataNNNN.aselmdb:id") so every
structure traces back to the npz shards and the raw LMDB records, and meta.json.

--aux DIR (multitask): the label shards of npz_aux/<split>/ (same file names as the shards,
rows and atoms aligned with them, checked by key) add, per split, for scheme in lowdin,
mulliken, nbo: <scheme>_charge.bin, <scheme>_spin.bin (float32 per atom, 0 where absent),
<scheme>_charge_mask.bin, <scheme>_spin_mask.bin (float32 per structure, 1 = labelled:
has_q_<scheme>, has_s_<scheme>); homo_energy.bin, homo_lumo_gap.bin (eV), homo_mask.bin,
gap_mask.bin (HOMO in [-20, 2] eV, gap in (0, 25] eV; outliers unlabelled), s_squared.bin
(float32 per structure); source.npz also gets unique_id.

Data files are opened read-only with allow_pickle=False.

    python -I scripts/mmml/omol_npz_to_memmap.py --shards OMOL/npz/train_4M -o DATA/pilot200k \\
        --n-structures 200000 --seed 0 --valid-fraction 0.02 --split-seed 0 \\
        --linear-ref-from OMOL/caches/omol_train4M_pilot200k/metadata.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

SCHEMES = ("lowdin", "mulliken", "nbo")
AUX_ATOM = tuple(f"{sc}_{t}" for sc in SCHEMES for t in ("charge", "spin"))
AUX_SYS = tuple(f"{sc}_{t}_mask" for sc in SCHEMES for t in ("charge", "spin")) + (
    "homo_energy", "homo_lumo_gap", "homo_mask", "gap_mask", "s_squared")


HOMO_RANGE = (-20.0, 2.0)  # eV, labelled when inside [lo, hi]
GAP_RANGE = (0.0, 25.0)    # eV, labelled when inside (lo, hi]


def aux_arrays(z):
    """npz_aux shard (omol aux label extraction) -> the split files' arrays (atoms, structures)"""
    out = {}
    for sc in SCHEMES:
        for t, key in (("charge", "q"), ("spin", "s")):
            v = np.asarray(z[f"{key}_{sc}"], np.float64)
            has = np.asarray(z[f"has_{key}_{sc}"], bool).reshape(-1)
            atom_has = np.repeat(has, np.diff(z["mol_offsets"]))
            out[f"{sc}_{t}"] = np.where(atom_has & np.isfinite(v), v, 0.0)
            out[f"{sc}_{t}_mask"] = has.astype(np.float64)
    # orbital labels: outliers masked (OMol25 has gaps up to 61 eV, HOMO down to -71 eV)
    for name, key, mask, lo, hi in (("homo_energy", "homo", "homo_mask", HOMO_RANGE[0], HOMO_RANGE[1]),
                                    ("homo_lumo_gap", "gap", "gap_mask", GAP_RANGE[0], GAP_RANGE[1])):
        v = np.asarray(z[key], np.float64).reshape(-1)
        ok = np.isfinite(v) & (v >= lo) & (v <= hi) if key == "homo" else np.isfinite(v) & (v > lo) & (v <= hi)
        out[name] = np.where(np.isfinite(v), v, 0.0)
        out[mask] = ok.astype(np.float64)
    out["s_squared"] = np.nan_to_num(np.asarray(z["s2"], np.float64).reshape(-1))
    out["keys"] = z["keys"]
    out["unique_id"] = z["unique_id"]
    return out


def shard_files(paths):
    files = []
    for s in paths:
        files += sorted(glob.glob(os.path.join(s, "*.npz"))) if os.path.isdir(s) else [s]
    return files


def split_indices(n_items: int, valid_fraction: float, seed: int):
    """scripts/train_so3lr_spooky_extxyz.py split_indices"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items).astype(np.int64)
    n_valid = max(1, int(round(n_items * valid_fraction))) if valid_fraction > 0 else 0
    return order[n_valid:], order[:n_valid]


class SplitWriter:
    def __init__(self, out: Path, aux: bool):
        out.mkdir(parents=True, exist_ok=True)
        if (out / "ns.npy").exists():
            raise SystemExit(f"{out} already holds a split")
        self.out = out
        names = ["x", "a", "c", "energy", "non_conservative_force", "charge", "spin_multiplicity"]
        if aux:
            names += list(AUX_ATOM) + list(AUX_SYS)
        self.f = {k: open(out / f"{k}.bin", "wb") for k in names}
        self.n_atoms = [np.zeros(1, np.int64)]
        self.src = {"shard": [], "row": [], "sel": [], "key": []}
        if aux:
            self.src["unique_id"] = []
        self.Q, self.S, self.E = [], [], []

    def add(self, R, Z, F, N, E, Q, S, shard, rows, sel, keys, aux=None):
        """R, Z, F: atoms of the structures (concatenated); per structure: N, E, Q, S, ..."""
        R.astype(np.float32).tofile(self.f["x"])
        Z.astype(np.int32).tofile(self.f["a"])
        np.zeros((len(N), 3, 3), np.float32).tofile(self.f["c"])
        E.astype(np.float32).tofile(self.f["energy"])
        F.astype(np.float32).tofile(self.f["non_conservative_force"])
        Q.astype(np.float32).reshape(-1, 1).tofile(self.f["charge"])
        S.astype(np.float32).reshape(-1, 1).tofile(self.f["spin_multiplicity"])
        if aux is not None:
            for k in AUX_ATOM + AUX_SYS:
                aux[k].astype(np.float32).tofile(self.f[k])
            self.src["unique_id"].append(aux["unique_id"])
        self.n_atoms.append(N.astype(np.int64))
        self.src["shard"].append(np.full(len(N), shard, np.int32))
        self.src["row"].append(rows.astype(np.int64))
        self.src["sel"].append(sel.astype(np.int64))
        self.src["key"].append(keys)
        self.Q.append(Q), self.S.append(S), self.E.append(E)

    def close(self, files, meta):
        for f in self.f.values():
            f.close()
        na = np.cumsum(np.concatenate(self.n_atoms)).astype(np.int64)
        ns = len(na) - 1
        np.save(self.out / "ns.npy", np.int64(ns))
        np.save(self.out / "na.npy", na)
        src = {k: np.concatenate(v) for k, v in self.src.items()}
        np.savez(self.out / "source.npz", shard_files=np.asarray(files), **src)
        Q, S, E = (np.concatenate(v) for v in (self.Q, self.S, self.E))
        N = np.diff(na)
        info = dict(meta, n_structures=int(ns), n_atoms=int(na[-1]), mean_atoms=float(N.mean()) if ns else 0.0,
                    max_atoms=int(N.max()) if ns else 0, charge_range=[int(Q.min()), int(Q.max())] if ns else None,
                    multiplicity_range=[int(S.min()), int(S.max())] if ns else None,
                    energy_mean_eV=float(E.mean()) if ns else None, energy_std_eV=float(E.std()) if ns else None)
        with open(self.out / "meta.json", "w") as fh:
            json.dump(info, fh, indent=1)
        print(f"{self.out}: {ns} structures, {int(na[-1])} atoms, Q {info['charge_range']}, "
              f"S {info['multiplicity_range']}, E std {info['energy_std_eV']:.3f} eV", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True, help="shard dirs or npz files")
    ap.add_argument("-o", "--out", required=True, help="directory to hold the split(s)")
    ap.add_argument("--n-structures", type=int, default=None, help="random subset (npz_to_spooky_cache.py's)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-fraction", type=float, default=0.0, help="the JAX trainer's --valid-fraction")
    ap.add_argument("--split-seed", type=int, default=0, help="the JAX trainer's --seed")
    ap.add_argument("--split-name", default="train", help="split name without --valid-fraction")
    ap.add_argument("--linear-ref-from", default=None, help="metadata.json holding linear_ref (applied, not fitted)")
    ap.add_argument("--max-structures", type=int, default=None, help="first N of the selection only (tests)")
    ap.add_argument("--aux", default=None, help="dir of per-shard aux npz (omol_aux_from_lmdb.py), multitask")
    a = ap.parse_args()

    files = shard_files(a.shards)
    sizes = []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            sizes.append(int(d["N"].shape[0]))
    n_all = int(sum(sizes))
    rng = np.random.default_rng(a.seed)
    if a.n_structures is not None and a.n_structures < n_all:
        pick = np.sort(rng.choice(n_all, size=a.n_structures, replace=False))
    else:
        pick = np.arange(n_all)
    if a.max_structures is not None:
        pick = pick[:a.max_structures]
    n_sel = len(pick)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    # which split each selected structure (position in the selection) goes to
    split_of = np.zeros(n_sel, np.int8)
    names = [a.split_name]
    if a.valid_fraction > 0:
        _, valid = split_indices(n_sel, a.valid_fraction, a.split_seed)
        split_of[valid] = 1
        names = ["train", "val"]
    linref, tab = None, None
    if a.linear_ref_from:
        with open(a.linear_ref_from) as fh:
            linref = json.load(fh)["linear_ref"]
        tab = np.zeros(119)
        tab[np.asarray(linref["Z"])] = linref["b_eV"]
    out = Path(a.out)
    writers = [SplitWriter(out / n, a.aux is not None) for n in names]

    pos = 0  # position in the selection
    for fi, f in enumerate(files):
        sel = pick[(pick >= starts[fi]) & (pick < starts[fi + 1])] - starts[fi]
        if sel.size == 0:
            continue
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in ("R", "Z", "F", "E", "N", "Q", "S", "mol_offsets", "keys")}
        aux = None
        if a.aux:
            with np.load(os.path.join(a.aux, os.path.basename(f)), allow_pickle=False) as z:
                aux = aux_arrays(z)
            if not np.array_equal(aux["keys"], d["keys"]):
                raise SystemExit(f"{f}: aux keys do not match the shard")
        off = d["mol_offsets"]
        E = d["E"].reshape(-1)[sel].astype(np.float64)
        Q = d["Q"].reshape(-1)[sel]
        S = d["S"].reshape(-1)[sel]
        N = d["N"].reshape(-1)[sel]
        if tab is not None:
            shift = np.array([tab[d["Z"][off[i]:off[i + 1]]].sum() for i in sel]) + linref["b_Q_eV_per_e"] * Q
            E = E - shift
        sel_pos = np.arange(pos, pos + sel.size)
        pos += sel.size
        for w_idx, w in enumerate(writers):
            m = split_of[sel_pos] == w_idx
            if not m.any():
                continue
            rows = sel[m]
            atoms = np.concatenate([np.arange(off[i], off[i + 1]) for i in rows])
            ax = None
            if aux is not None:
                ax = {k: aux[k][atoms] for k in AUX_ATOM}
                ax.update({k: aux[k][rows] for k in AUX_SYS})
                ax["unique_id"] = aux["unique_id"][rows]
            w.add(d["R"][atoms], d["Z"][atoms], d["F"][atoms], N[m], E[m], Q[m], S[m], fi, rows, sel_pos[m],
                  d["keys"][rows], ax)
        print(f"  {os.path.basename(f)}: {sel.size} structures", flush=True)
    meta = dict(source=a.shards, n_structures_arg=a.n_structures, seed=a.seed, valid_fraction=a.valid_fraction,
                split_seed=a.split_seed, linear_ref_from=a.linear_ref_from, linear_ref=linref,
                energy="E_DFT - sum E_iso(Z) - (sum_Z n_Z b_Z + b_Q Q) [eV] (float32)", forces="eV/A",
                aux=a.aux)
    for w in writers:
        w.close(files, meta)


if __name__ == "__main__":
    main()
