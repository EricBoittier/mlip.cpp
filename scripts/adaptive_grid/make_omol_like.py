#!/usr/bin/env python3
"""A small OMol-like set for the "grid" adaptive-cutoff training checks: H/C/N/O
molecules of ase's g2 collection and a few clusters of them (5..60 atoms), rattled,
with a charge and a spin multiplicity of the right electron-count parity, written as
extxyz with OMol's info keys `charge` and `spin` (2S+1). Labels: a model's energy and
forces (mlipcpp-eval of a GGUF, e.g. PET-OMol-S) plus noise, so that a fine-tune has
something to fit; then the memmap splits (scripts/train/xyz_to_memmap.py).

    python make_omol_like.py OUT --gguf pet-omol-s.gguf --eval build/bin/mlipcpp-eval [--n 64]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import ase
import ase.build
import ase.collections
import ase.io
import numpy as np

Z_OK = {1, 6, 7, 8}


def molecules():
    out = []
    for name in ase.collections.g2.names:
        a = ase.collections.g2[name]
        if set(a.numbers) <= Z_OK and len(a) >= 3:
            out.append(a)
    return out


def make(n, rng, pool):
    frames = []
    for k in range(n):
        if k % 4 == 3:  # a cluster of several molecules
            a = ase.Atoms()
            for m in range(rng.integers(3, 7)):
                b = pool[rng.integers(len(pool))].copy()
                b.positions += rng.uniform(-4.5, 4.5, 3)
                if len(a) and np.min(np.linalg.norm(a.positions[:, None] - b.positions[None], axis=-1)) < 2.0:
                    continue
                a += b
        else:
            a = pool[rng.integers(len(pool))].copy()
        a.positions += rng.normal(0, 0.05, a.positions.shape)
        q = int(rng.integers(-2, 3))
        electrons = int(a.numbers.sum()) - q
        s = int(rng.choice([1, 3])) if electrons % 2 == 0 else int(rng.choice([2, 4]))
        a.info = {"charge": q, "spin": s}
        frames.append(a)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pool = molecules()
    here = Path(__file__).resolve().parents[1]
    for split, n in (("train", args.n), ("val", max(8, args.n // 4)), ("test", max(8, args.n // 4))):
        frames = make(n, rng, pool)
        raw = out / f"{split}-raw.xyz"
        ase.io.write(raw, frames, format="extxyz")
        subprocess.run([args.eval, args.gguf, str(raw), str(out / f"{split}-pred.json")], check=True,
                       capture_output=True)
        pred = json.load(open(out / f"{split}-pred.json"))["frames"]
        for a, p in zip(frames, pred):
            a.info["e_ref"] = p["energy"] + 0.02 * len(a) * rng.normal()
            a.arrays["f_ref"] = np.array(p["forces"]).reshape(-1, 3) + rng.normal(0, 0.05, (len(a), 3))
        ase.io.write(out / f"{split}.xyz", frames, format="extxyz")
        subprocess.run([sys.executable, str(here / "train/xyz_to_memmap.py"), str(out / f"{split}.xyz"),
                        str(out / split), "--energy-key", "e_ref", "--forces-key", "f_ref"], check=True)
        print(split, len(frames), "structures,", sum(len(a) for a in frames), "atoms, max",
              max(len(a) for a in frames))


if __name__ == "__main__":
    main()
