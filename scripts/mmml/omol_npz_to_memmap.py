#!/usr/bin/env python3
"""OMol25 npz shards (omol_universal/scripts/omol_to_npz.py layout) -> mlipcpp-train data.

Writes DIR/{train,val,test} as metatrain MemmapDataset splits (the layout of
scripts/train/xyz_to_memmap.py) with charge.bin and spin_multiplicity.bin, so a
SpookyPhysNet GGUF trains with mlipcpp-train on the same structures, energies
(E, the referenced energy the mmml trainer fits) and forces as the JAX trainer.

    python scripts/mmml/omol_npz_to_memmap.py OMOL/npz/train_4M/shard_f000*.npz -o omol_pilot \
        --n-train 20000 --n-val 1000 --n-test 1000 [--max-atoms 120] [--max-z 87]

Also writes DIR/structures.npz (the selected R, Z, N, Q, S, E, F) for the JAX benchmark.
"""

import argparse
from pathlib import Path

import numpy as np


def write_split(out: Path, mols):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ns.npy").exists():
        raise SystemExit(f"{out} already holds a split")
    with open(out / "x.bin", "wb") as fx, open(out / "a.bin", "wb") as fa, open(out / "c.bin", "wb") as fc, \
            open(out / "energy.bin", "wb") as fe, open(out / "non_conservative_force.bin", "wb") as ff, \
            open(out / "energy_forces.bin", "wb") as fg:
        for m in mols:
            m["R"].astype(np.float32).tofile(fx)
            m["Z"].astype(np.int32).tofile(fa)
            np.zeros((3, 3), np.float32).tofile(fc)
            np.float32(m["E"]).tofile(fe)
            m["F"].astype(np.float32).tofile(ff)
            m["F"].astype(np.float32).tofile(fg)
    np.array([m["Q"] for m in mols], np.float32).reshape(-1, 1).tofile(out / "charge.bin")
    np.array([m["S"] for m in mols], np.float32).reshape(-1, 1).tofile(out / "spin_multiplicity.bin")
    np.save(out / "ns.npy", np.int64(len(mols)))
    np.save(out / "na.npy", np.cumsum([0] + [len(m["Z"]) for m in mols]).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--max-atoms", type=int, default=120)
    ap.add_argument("--max-z", type=int, default=87)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    need = a.n_train + a.n_val + a.n_test
    mols = []
    for path in a.shards:
        z = np.load(path, allow_pickle=True)
        d = {k: z[k] for k in ("R", "Z", "F", "E", "Q", "S", "N", "mol_offsets")}  # NpzFile reloads per access
        off, N = d["mol_offsets"], d["N"][:, 0]
        for i in range(len(N)):
            if N[i] > a.max_atoms:
                continue
            Z = d["Z"][off[i]:off[i + 1]]
            if Z.max() > a.max_z:
                continue
            mols.append(dict(R=d["R"][off[i]:off[i + 1]], Z=Z, F=d["F"][off[i]:off[i + 1]], E=float(d["E"][i, 0]),
                             Q=int(round(d["Q"][i, 0])), S=int(round(d["S"][i, 0]))))
        if len(mols) >= 2 * need:
            break
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(mols))[:need]
    if len(idx) < need:
        raise SystemExit(f"only {len(idx)} structures available, {need} requested")
    sel = [mols[i] for i in idx]
    out = Path(a.out)
    write_split(out / "train", sel[:a.n_train])
    write_split(out / "val", sel[a.n_train:a.n_train + a.n_val])
    write_split(out / "test", sel[a.n_train + a.n_val:])
    tr = sel[:a.n_train]
    np.savez(out / "structures.npz", N=np.array([len(m["Z"]) for m in tr]), R=np.concatenate([m["R"] for m in tr]),
             Z=np.concatenate([m["Z"] for m in tr]), F=np.concatenate([m["F"] for m in tr]),
             E=np.array([m["E"] for m in tr]), Q=np.array([m["Q"] for m in tr]), S=np.array([m["S"] for m in tr]))
    n_at = np.array([len(m["Z"]) for m in tr])
    print(f"{out}: train {a.n_train} (mean {n_at.mean():.1f} atoms, max {n_at.max()}), val {a.n_val}, test {a.n_test}; "
          f"Q in [{min(m['Q'] for m in sel)}, {max(m['Q'] for m in sel)}], S in [{min(m['S'] for m in sel)}, "
          f"{max(m['S'] for m in sel)}]")


if __name__ == "__main__":
    main()
