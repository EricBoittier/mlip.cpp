#!/usr/bin/env python3
"""Node-feature parity: mlipcpp-feats ID-mode dump (one node, the trunk's final node features [n, d_node])
against the oracle's `node` (ref_trunk.py). Prints max |dh| / max |h| per frame (max and median over frames).

    python compare_feats.py REF.npz FEATS.bin [REF2.npz]
"""
import sys

import numpy as np


def read_bin(path):
    raw = open(path, "rb").read()
    o, out = 0, []
    while o < len(raw):
        ne = np.frombuffer(raw, np.int64, 4, o); o += 32
        n = int(np.prod(ne))
        out.append(np.frombuffer(raw, np.float32, n, o).reshape(ne[1], ne[0]).astype(np.float64)); o += 4 * n
    return out


def frames(path):
    z = np.load(path); a = z["atom_ptr"]
    return [z["node"][a[k]:a[k + 1]].astype(np.float64) for k in range(len(a) - 1)]


def rel(xs, ys):
    r = [np.abs(x - y).max() / np.abs(y).max() for x, y in zip(xs, ys)]
    return f"max {max(r):.3g} median {np.median(r):.3g} (n={len(r)})"


def main():
    ref = frames(sys.argv[1]); cpp = read_bin(sys.argv[2])
    assert len(ref) == len(cpp) and all(x.shape == y.shape for x, y in zip(ref, cpp))
    print("cpp vs ref :", rel(cpp, ref))
    if len(sys.argv) > 3:
        ref2 = frames(sys.argv[3]); print("ref2 vs ref:", rel(ref2, ref))


if __name__ == "__main__":
    main()
