#!/usr/bin/env python3
"""A copy of a Plato-PET-LR GGUF with fragment-constrained charges switched on (pet.plato_lr.frag_charges = 1 and
frag_R / frag_t / frag_sigma / frag_scale added; keys already present are replaced). The tensors are unchanged (the
term has no parameters): the model is the same for structures without formal charges.

    python add_frag_keys.py IN.gguf OUT.gguf [--R 8 --t 0.5 --sigma 4 --scale 1]
"""
import argparse
import struct

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--R", type=float, default=8.0); ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=4.0); ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    data = open(a.input, "rb").read()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos); pos += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string():
        nonlocal pos
        n = take("Q"); s = data[pos:pos + n].decode(); pos += n
        return s

    def skip_value(t):
        if t == 8:
            string(); return
        if t == 9:
            et, n = take("I"), take("Q")
            for _ in range(n):
                skip_value(et)
            return
        take(SC[t])

    magic, version = take("I"), take("I")
    assert magic == 0x46554747 and version == 3, (hex(magic), version)
    nt, nkv = take("Q"), take("Q")
    new = {"pet.plato_lr.frag_charges": (5, 1), "pet.plato_lr.frag_R": (6, a.R), "pet.plato_lr.frag_t": (6, a.t),
           "pet.plato_lr.frag_sigma": (6, a.sigma), "pet.plato_lr.frag_scale": (6, a.scale)}
    kept, alignment, plato = [], 32, False
    for _ in range(nkv):
        k0 = pos; k = string(); t = take("I"); v0 = pos
        if k == "general.alignment":
            alignment = struct.unpack_from("<I", data, v0)[0]
        if k == "pet.plato_lr":
            plato = struct.unpack_from("<i", data, v0)[0] != 0
        skip_value(t)
        if k not in new:
            kept.append(data[k0:pos])
    if not plato:
        raise SystemExit(f"{a.input}: not a Plato-PET-LR GGUF (pet.plato_lr)")
    ti0 = pos
    for _ in range(nt):
        string(); nd = take("I")
        for _ in range(nd):
            take("Q")
        take("I"); take("Q")
    ti = data[ti0:pos]
    data_start = (pos + alignment - 1) // alignment * alignment
    enc = lambda s: struct.pack("<Q", len(s.encode())) + s.encode()
    extra = [enc(k) + struct.pack("<I", t) + struct.pack("<" + SC[t], v) for k, (t, v) in new.items()]
    head = struct.pack("<IIQQ", magic, version, nt, len(kept) + len(extra)) + b"".join(kept) + b"".join(extra) + ti
    head += b"\0" * ((-len(head)) % alignment)
    with open(a.output, "wb") as f:
        f.write(head); f.write(data[data_start:])
    print(f"{a.output}: frag_charges on (R {a.R}, t {a.t}, sigma {a.sigma}, scale {a.scale})")


if __name__ == "__main__":
    main()
