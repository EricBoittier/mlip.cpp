#!/usr/bin/env python3
"""Compare two GGUF files key by key and tensor by tensor (numpy only, a minimal GGUF v2/v3 reader).

    python gguf_diff.py OLD.gguf NEW.gguf [--allow-new KEY ...]

Prints keys only in one file, keys whose values differ, and tensors whose shape / type / bytes differ. Exit 0 when the
files are identical apart from the --allow-new keys (present in NEW only), 1 otherwise.
"""
import argparse
import struct
import sys

import numpy as np

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read(path):
    raw = open(path, "rb").read()
    off = 0

    def take(fmt):
        nonlocal off
        v = struct.unpack_from(fmt, raw, off)
        off += struct.calcsize(fmt)
        return v[0]

    def string():
        nonlocal off
        n = take("<Q")
        s = raw[off:off + n].decode("utf-8", "replace")
        off += n
        return s

    def value(t):
        if t in SCALAR:
            return take(SCALAR[t])
        if t == 8:
            return string()
        if t == 9:
            et, n = take("<I"), take("<Q")
            return tuple(value(et) for _ in range(n))
        raise ValueError(f"{path}: unknown GGUF value type {t}")

    assert raw[:4] == b"GGUF", f"{path}: not a GGUF file"
    off = 4
    version = take("<I")
    n_t, n_kv = take("<Q"), take("<Q")
    kv = {}
    for _ in range(n_kv):
        k = string()
        kv[k] = value(take("<I"))
    infos = []
    for _ in range(n_t):
        name = string()
        nd = take("<I")
        dims = tuple(take("<Q") for _ in range(nd))
        infos.append((name, dims, take("<I"), take("<Q")))
    align = int(kv.get("general.alignment", 32))
    base = (off + align - 1) // align * align
    offs = sorted(o for _, _, _, o in infos) + [len(raw) - base]
    tensors = {}
    for name, dims, t, o in infos:
        end = offs[offs.index(o) + 1]
        tensors[name] = (dims, t, raw[base + o:base + end])
    return version, kv, tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--allow-new", nargs="*", default=[])
    a = ap.parse_args()
    _, ka, ta = read(a.old)
    _, kb, tb = read(a.new)
    bad = []
    for k in sorted(set(ka) | set(kb)):
        if k not in kb:
            bad.append(f"key only in old: {k}")
        elif k not in ka:
            if k not in a.allow_new:
                bad.append(f"key only in new: {k} = {str(kb[k])[:80]}")
            else:
                print(f"new key (allowed): {k} = {kb[k]}")
        elif ka[k] != kb[k]:
            bad.append(f"key differs: {k}: {str(ka[k])[:80]} -> {str(kb[k])[:80]}")
    for n in sorted(set(ta) | set(tb)):
        if n not in ta or n not in tb:
            bad.append(f"tensor only in {'new' if n in tb else 'old'}: {n}")
            continue
        (da, xa, ba), (db, xb, bb) = ta[n], tb[n]
        # the data span of a tensor includes alignment padding: compare the tensor's own bytes
        nbytes = int(np.prod(da)) * {0: 4, 1: 2, 26: 4, 28: 8}.get(xa, 4)
        if da != db or xa != xb or ba[:nbytes] != bb[:nbytes]:
            bad.append(f"tensor differs: {n} {da}/{xa} vs {db}/{xb}")
    print("\n".join(bad) if bad else f"IDENTICAL apart from {a.allow_new or 'nothing'}: {len(ka)} keys, {len(ta)} tensors")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
