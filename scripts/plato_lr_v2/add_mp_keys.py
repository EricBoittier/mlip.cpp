#!/usr/bin/env python3
"""A copy of a graph-mode lrc GGUF (add_lrc_keys.py --mode graph) with the prefitted atomic multipoles switched on
(pet.plato_lr.lrc_mp and lrc_mp_*, runtime/lr_combined.h; existing lrc_mp keys are replaced). Tensors unchanged.

    python add_mp_keys.py IN.gguf OUT.gguf FIT.json [FIT2.json]

FIT.json: multipoles (prefit_mp.py): {"rc", "width", "centers": [K], "Z": [E], "theta": [E*E*K], "tau": [E*E*K]} and/or
induced dipoles (prefit_pol.py): {"pol": {"Z": [...], "alpha": [...]}} (pet.plato_lr.lrc_pol*). Existing lrc_mp / lrc_pol
keys are replaced."""
import json
import struct
import sys

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main():
    inp, out = sys.argv[1:3]
    mp, pol = None, None
    for f in sys.argv[3:]:
        d = json.load(open(f))
        if "pol" in d:
            pol = d["pol"]
        if "theta" in d:
            mp = d
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
    arr = lambda key, v: enc(key) + struct.pack("<IIQ", 9, 12, len(v)) + struct.pack(f"<{len(v)}d", *[float(x) for x in v])
    P, PP = "pet.plato_lr.lrc_mp", "pet.plato_lr.lrc_pol"
    new = []
    if mp is not None:
        E, K = len(mp["Z"]), len(mp["centers"])
        assert len(mp["theta"]) == E * E * K and len(mp["tau"]) == E * E * K
        new.append(enc(P) + struct.pack("<Ii", 5, 1))
        for k in ("rc", "width"):
            new.append(enc(f"{P}_{k}") + struct.pack("<Id", 12, float(mp[k])))
        for k in ("centers", "Z", "theta", "tau"):
            new.append(arr(f"{P}_{k}", mp[k]))
    if pol is not None:
        assert len(pol["Z"]) == len(pol["alpha"]) and min(pol["alpha"]) >= 0
        new += [enc(PP) + struct.pack("<Ii", 5, 1), arr(f"{PP}_Z", pol["Z"]), arr(f"{PP}_alpha", pol["alpha"])]
    magic, version = take("I"), take("I")
    assert magic == 0x46554747 and version == 3
    nt, nkv = take("Q"), take("Q")
    kept, meta = [], {}
    for _ in range(nkv):
        k0 = pos; k = string(); t = take("I")
        v = value(t)
        if k in ("general.alignment", "pet.plato_lr.lrc_q0_mode"):
            meta[k] = v
        if not ((mp is not None and k.startswith(P)) or (pol is not None and k.startswith(PP))):
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
    print(f"{out}: lrc" + (f" multipoles ({len(mp['Z'])} elements, {len(mp['centers'])} radial channels)" if mp else "")
          + (f" induced dipoles ({len(pol['Z'])} elements)" if pol else ""))


if __name__ == "__main__":
    main()
