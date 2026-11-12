#!/usr/bin/env python3
"""Two MLIPCPP_NODE_GRAD_DUMP files of the same structures (e.g. the same CUDA run twice): per node, the max relative
difference of the forward value and of dE/d(node), in graph order. Run-to-run differences on one backend are the
unordered (atomic) accumulations: the backward node whose gradient first differs, walking back from the loss, is
where the nondeterminism enters. numpy only.

    python node_diff.py A.bin B.bin [--frame I]
"""
import argparse

import numpy as np


def read(path, frame):
    raw = open(path, "rb").read(); off = 0; f = 0
    while off < len(raw):
        magic, cnt = np.frombuffer(raw, np.int32, 2, off); off += 8
        out = {}

        def block():
            nonlocal off
            nd = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
            ne = np.frombuffer(raw, np.int64, nd, off).copy(); off += 8 * nd
            nb = np.frombuffer(raw, np.int64, nd, off).copy(); off += 8 * nd
            dt = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
            n = int(np.frombuffer(raw, np.int64, 1, off)[0]); off += 8
            buf = raw[off:off + n]; off += n
            if dt != 0 or n == 0 or f != frame:
                return None
            arr = np.frombuffer(buf, np.float32)
            return np.lib.stride_tricks.as_strided(arr, shape=tuple(ne[::-1]), strides=tuple(nb[::-1])).copy()
        for _ in range(cnt):
            i, nl = np.frombuffer(raw, np.int32, 2, off); off += 8
            name = raw[off:off + nl].decode(); off += nl
            v = block()
            has = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
            g = block() if has else None
            if f == frame:
                out[name] = (int(i), v, g)
        if f == frame:
            return out
        f += 1


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()
    A, B = read(a.a, a.frame), read(a.b, a.frame)
    rows = []
    for nm, (i, v, g) in sorted(A.items(), key=lambda kv: kv[1][0]):
        _, v2, g2 = B[nm]
        dv = rel(v, v2) if v is not None and v2 is not None and v.size == v2.size and np.isfinite(v).all() else np.nan
        dg = rel(g, g2) if g is not None and g2 is not None and np.isfinite(g).all() and np.isfinite(g2).all() else np.nan
        rows.append((i, nm, dv, dg))
    print(f"frame {a.frame}: {sum(1 for r in rows if r[2] > 0)} nodes differ in value, "
          f"{sum(1 for r in rows if r[3] > 0)} in dE/d(node)")
    print("value differences (graph order):", [(nm, f"{dv:.1e}") for _, nm, dv, _ in rows if dv > 0][:10])
    print("gradient differences, from the loss backwards (the first is where the nondeterminism enters):")
    for i, nm, dv, dg in reversed(rows):
        if dg > 0:
            print(f"  {i:5d} {nm:28s} d(dE/dnode) {dg:.2e}")
    for nm in ("input_edge_vectors", "input_cutoff_values"):
        if nm in A and A[nm][2] is not None:
            print(f"{nm}: run-to-run d(dE/dnode) {rel(A[nm][2], B[nm][2]):.2e}")


if __name__ == "__main__":
    main()
