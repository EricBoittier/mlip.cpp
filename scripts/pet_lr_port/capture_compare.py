#!/usr/bin/env python3
"""Node-by-node comparison of two mlipcpp-capture dumps (e.g. CPU vs CUDA): relative max difference per node,
in graph order, the first nodes above a threshold, and the largest jumps (a node whose relative difference is
much larger than that of its worst input marks the op that loses precision).

    python capture_compare.py A.bin B.bin [--thr 1e-5] [--top 25]
"""
import argparse

import numpy as np


def read(path):
    raw = open(path, "rb").read(); off = 0; out = {}
    while off < len(raw):
        i, nl = np.frombuffer(raw, np.int32, 2, off); off += 8
        name = raw[off:off + nl].decode(); off += nl
        nd = np.frombuffer(raw, np.int32, 1, off)[0]; off += 4
        shape = tuple(np.frombuffer(raw, np.int64, nd, off)); off += 8 * nd
        dt = np.frombuffer(raw, np.int32, 1, off)[0]; off += 4
        nb = np.frombuffer(raw, np.int64, 1, off)[0]; off += 8
        data = np.frombuffer(raw, np.float32 if dt == 0 else np.int32, nb // 4, off) if dt in (0, 1) else None
        off += nb
        out[int(i)] = (name, shape, data)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--thr", type=float, default=1e-5); ap.add_argument("--top", type=int, default=25)
    x = parse = None
    a_ = ap.parse_args()
    A, B = read(a_.a), read(a_.b)
    rows = []
    for i in sorted(A):
        if i not in B or A[i][2] is None or B[i][2] is None or A[i][2].size != B[i][2].size:
            continue
        a, b = A[i][2].astype(float), B[i][2].astype(float)
        fin = np.isfinite(a) & np.isfinite(b)
        if not fin.any():
            continue
        scale = max(np.abs(a[fin]).max(), 1e-30)
        rel = np.abs(a[fin] - b[fin]).max() / scale
        rows.append((i, A[i][0], A[i][1], rel, scale, int((~fin).sum())))
    print(f"{len(rows)} common nodes")
    print("first nodes above thr:")
    for r in [r for r in rows if r[3] > a_.thr][:a_.top]:
        print(f"  {r[0]:5d} {r[1]:40s} {str(r[2]):22s} rel {r[3]:.2e} scale {r[4]:.2e} nonfinite {r[5]}")
    print("largest relative differences:")
    for r in sorted(rows, key=lambda r: -r[3])[:a_.top]:
        print(f"  {r[0]:5d} {r[1]:40s} {str(r[2]):22s} rel {r[3]:.2e} scale {r[4]:.2e}")


if __name__ == "__main__":
    main()
