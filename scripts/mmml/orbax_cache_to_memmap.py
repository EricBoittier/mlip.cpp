#!/usr/bin/env python3
"""An mmml Orbax flat cache (npz_to_spooky_cache.py; the --cache-path of
train_so3lr_spooky_extxyz.py) -> mlipcpp-train splits, for clusters that hold the
caches but not the npz shards (sciCORE).

The cache's E already carries the linear reference, so the targets are exactly the
JAX trainer's. --valid-fraction f --split-seed s reproduces the trainer's
split_indices(n, f, s) -> train/ and val/; without it everything goes to --split-name.
Same split layout as omol_npz_to_memmap.py (source.npz holds the cache row as "sel").

    python scripts/mmml/orbax_cache_to_memmap.py CACHE -o DATA/pilot200k --valid-fraction 0.02 --split-seed 0
    python scripts/mmml/orbax_cache_to_memmap.py VAL_CACHE -o DATA/val50k --split-name val

Needs orbax (the mmml environment).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omol_npz_to_memmap import SplitWriter, split_indices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--valid-fraction", type=float, default=0.0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-name", default="train")
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()
    import orbax.checkpoint as ocp

    d = ocp.PyTreeCheckpointer().restore(a.cache)
    d = {k: np.asarray(d[k]) for k in ("R", "Z", "F", "E", "N", "Q", "S", "mol_offsets")}
    n = d["N"].shape[0]
    split_of = np.zeros(n, np.int8)
    names = [a.split_name]
    if a.valid_fraction > 0:
        _, valid = split_indices(n, a.valid_fraction, a.split_seed)
        split_of[valid] = 1
        names = ["train", "val"]
    out = Path(a.out)
    writers = [SplitWriter(out / s, False) for s in names]
    off = d["mol_offsets"]
    E, Q, S, N = (d[k].reshape(-1) for k in ("E", "Q", "S", "N"))
    for c0 in range(0, n, a.chunk):
        rows_all = np.arange(c0, min(n, c0 + a.chunk))
        for w_idx, w in enumerate(writers):
            rows = rows_all[split_of[rows_all] == w_idx]
            if rows.size == 0:
                continue
            atoms = np.concatenate([np.arange(off[i], off[i + 1]) for i in rows])
            w.add(d["R"][atoms], d["Z"][atoms], d["F"][atoms], N[rows], E[rows], Q[rows], S[rows], 0, rows, rows,
                  np.asarray([f"cache:{i}" for i in rows]))
        print(f"  {min(n, c0 + a.chunk)}/{n}", flush=True)
    meta = dict(source=[a.cache], valid_fraction=a.valid_fraction, split_seed=a.split_seed,
                energy="the cache's E (linear reference applied) [eV] (float32)", forces="eV/A")
    try:
        meta["cache_metadata"] = {k: v for k, v in json.load(open(Path(a.cache) / "metadata.json")).items()
                                  if k != "source_shards"}
    except OSError:
        pass
    for w in writers:
        w.close([a.cache], meta)


if __name__ == "__main__":
    main()
