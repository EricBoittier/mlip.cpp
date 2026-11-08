#!/usr/bin/env python3
"""A copy of a graph-mode lrc GGUF (add_lrc_keys.py --mode graph) with the prefitted atomic multipoles switched on
(pet.plato_lr.lrc_mp and lrc_mp_*, runtime/lr_combined.h; existing lrc_mp keys are replaced). Tensors unchanged.

    python add_mp_keys.py IN.gguf OUT.gguf MPFIT.json

MPFIT.json (prefit_mp.py): {"rc", "width", "centers": [K], "Z": [E], "theta": [E*E*K], "tau": [E*E*K]}."""
import json
import struct
import sys

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main():
    inp, out, fit = sys.argv[1:4]
    mp = json.load(open(fit))
    E, K = len(mp["Z"]), len(mp["centers"])
    assert len(mp["theta"]) == E * E * K and len(mp["tau"]) == E * E * K
    data = open(inp, "rb").read()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos); pos += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string():
        nonlocal pos
        n = take("Q"); s = data[pos:pos + n].decode(); pos += n
        return s

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(SC[t])

    enc = lambda s: struct.pack("<Q", len(s.encode())) + s.encode()
    P = "pet.plato_lr.lrc_mp"
    new = [enc(P) + struct.pack("<Ii", 5, 1)]
    for k in ("rc", "width"):
        new.append(enc(f"{P}_{k}") + struct.pack("<Id", 12, float(mp[k])))
    for k in ("centers", "Z", "theta", "tau"):
        v = [float(x) for x in mp[k]]
        new.append(enc(f"{P}_{k}") + struct.pack("<IIQ", 9, 12, len(v)) + struct.pack(f"<{len(v)}d", *v))
    magic, version = take("I"), take("I")
    assert magic == 0x46554747 and version == 3
    nt, nkv = take("Q"), take("Q")
    kept, meta = [], {}
    for _ in range(nkv):
        k0 = pos; k = string(); t = take("I")
        v = value(t)
        if k in ("general.alignment", "pet.plato_lr.lrc_q0_mode"):
            meta[k] = v
        if not k.startswith(P):
            kept.append(data[k0:pos])
    if meta.get("pet.plato_lr.lrc_q0_mode") != "graph":
        raise SystemExit(f"{inp}: needs an lrc graph-mode GGUF (add_lrc_keys.py --mode graph)")
    alignment = int(meta.get("general.alignment", 32))
    ti0 = pos
    for _ in range(nt):
        string(); nd = take("I"); [take("Q") for _ in range(nd)]; take("I"); take("Q")
    ti = data[ti0:pos]
    body = data[(pos + alignment - 1) // alignment * alignment:]
    head = struct.pack("<IIQQ", magic, version, nt, len(kept) + len(new)) + b"".join(kept) + b"".join(new) + ti
    head += b"\0" * ((-len(head)) % alignment)
    with open(out, "wb") as f:
        f.write(head); f.write(body)
    print(f"{out}: lrc multipoles on ({E} elements, {K} radial channels, rc {mp['rc']} A)")


if __name__ == "__main__":
    main()
