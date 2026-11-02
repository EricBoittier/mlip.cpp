#!/usr/bin/env python3
"""A copy of a GGUF with F32 tensors filled with one value and/or existing scalar metadata values
replaced in place (same type, so offsets do not move), e.g. a Plato-PET-LR model with the LES
within-cutoff subtraction on and the LES gate at 1:

    python patch_gguf.py IN OUT --kv pet.plato_lr.les_tail_subtract_cutoff=1 --tensor plato_les_gate=1
"""
import argparse
import struct

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--kv", action="append", default=[], help="KEY=VALUE (an existing scalar key)")
    ap.add_argument("--tensor", action="append", default=[], help="NAME=VALUE (an F32 tensor, every value)")
    args = ap.parse_args()
    data = open(args.input, "rb").read()
    buf = bytearray(data)
    pos = 4

    def take(fmt):
        nonlocal pos
        out = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return out[0] if len(out) == 1 else out

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

    _, nt, nkv = take("I"), take("Q"), take("Q")
    meta, where = {}, {}
    for _ in range(nkv):
        k = string()
        t = take("I")
        where[k] = (t, pos)
        meta[k] = value(t)
    want = dict(x.split("=", 1) for x in args.kv)
    for k, v in want.items():
        assert k in where, f"{k}: not in {args.input} (only existing keys can be patched in place)"
        t, p = where[k]
        assert t in SC, f"{k}: not a scalar"
        cast = float if SC[t] in "fd" else (lambda s: bool(int(s))) if t == 7 else int
        struct.pack_into("<" + SC[t], buf, p, cast(v))
        print(f"{args.output}: {k} {meta[k]} -> {v}")
    infos = {}
    for _ in range(nt):
        nm = string()
        dims = [take("Q") for _ in range(take("I"))]
        infos[nm] = (dims, take("I"), take("Q"))
    align = meta.get("general.alignment", 32)
    base = (pos + align - 1) // align * align
    for x in args.tensor:
        name, v = x.split("=", 1)
        assert name in infos and infos[name][1] == 0, f"{name} not found as F32"
        dims, _, off = infos[name]
        count = 1
        for d in dims:
            count *= d
        for k in range(count):
            struct.pack_into("<f", buf, base + off + 4 * k, float(v))
        print(f"{args.output}: {name} = {v} ({count} values)")
    open(args.output, "wb").write(buf)


if __name__ == "__main__":
    main()
