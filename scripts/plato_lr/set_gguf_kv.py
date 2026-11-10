#!/usr/bin/env python3
"""A copy of a GGUF with scalar metadata keys added or replaced (any type change allowed: the header is rewritten and
the tensor data copied unchanged, re-aligned), e.g. the intra-cluster long range on an lrc model:

    python set_gguf_kv.py IN OUT --str pet.plato_lr.lrc_intra=ewald

--str KEY=VALUE, --i32 KEY=VALUE, --f64 KEY=VALUE; --drop KEY removes a key.
"""
import argparse
import struct

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--str", action="append", default=[])
    ap.add_argument("--i32", action="append", default=[])
    ap.add_argument("--f64", action="append", default=[])
    ap.add_argument("--drop", action="append", default=[])
    a = ap.parse_args()
    data = open(a.input, "rb").read()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string():
        nonlocal pos
        n = take("Q")
        s = data[pos:pos + n].decode()
        pos += n
        return s

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(SC[t])

    enc = lambda s: struct.pack("<Q", len(s.encode())) + s.encode()
    new = {}
    for kv in a.str:
        k, v = kv.split("=", 1)
        new[k] = enc(k) + struct.pack("<I", 8) + enc(v)
    for kv in a.i32:
        k, v = kv.split("=", 1)
        new[k] = enc(k) + struct.pack("<Ii", 5, int(v))
    for kv in a.f64:
        k, v = kv.split("=", 1)
        new[k] = enc(k) + struct.pack("<Id", 12, float(v))
    drop = set(a.drop) | set(new)
    magic, version = take("I"), take("I")
    assert magic == 0x46554747 and version == 3, (hex(magic), version)
    nt, nkv = take("Q"), take("Q")
    kept, alignment = [], 32
    for _ in range(nkv):
        k0 = pos
        k = string()
        t = take("I")
        v = value(t)
        if k == "general.alignment":
            alignment = int(v)
        if k not in drop:
            kept.append(data[k0:pos])
    ti0 = pos
    for _ in range(nt):
        string()
        nd = take("I")
        [take("Q") for _ in range(nd)]
        take("I")
        take("Q")
    ti = data[ti0:pos]
    body = data[(pos + alignment - 1) // alignment * alignment:]
    extra = list(new.values())
    head = struct.pack("<IIQQ", magic, version, nt, len(kept) + len(extra)) + b"".join(kept) + b"".join(extra) + ti
    head += b"\0" * ((-len(head)) % alignment)
    with open(a.output, "wb") as f:
        f.write(head)
        f.write(body)
    print(f"{a.output}: set {sorted(new)}, dropped {sorted(set(a.drop))}")


if __name__ == "__main__":
    main()
