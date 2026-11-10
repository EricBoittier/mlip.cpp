#!/usr/bin/env python3
"""Flatten an iris (ipsum/PET-LR, after M. Langer) checkpoint's parameter tree to an npz, read with his own
`marathon.io.read_msgpack`, so the converter needs no JAX/flax. Runs in his env (the oracle).

    python dump_params.py CKPT_DIR OUT.npz

CKPT_DIR holds model/model.msgpack (the checkpoint folder of `Calculator.from_checkpoint`). Keys are the
"/"-joined tree paths (params/sr/backbone/...); dtypes are kept as stored.
"""
import sys

import numpy as np
from marathon.io import read_msgpack


def flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = np.asarray(v)
    return out


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    flat = flatten(read_msgpack(f"{ckpt}/model/model.msgpack"))
    np.savez(out, **flat)
    n = sum(v.size for v in flat.values())
    print(f"{ckpt}: {len(flat)} leaves, {n} parameters -> {out}")


if __name__ == "__main__":
    main()
