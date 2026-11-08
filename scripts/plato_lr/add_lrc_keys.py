#!/usr/bin/env python3
"""A copy of an SQE GGUF (scripts/plato_lr/add_sqe_keys.py, --q0 uniform, lr_R > 0) with the combined long-range model
switched on (pet.plato_lr.lrc = 1 and the lrc_* keys, runtime/lr_combined.h; keys already present are replaced).
Tensors unchanged: the charge model is float64 metadata (host side, frozen in training).

    python add_lrc_keys.py IN_SQE.gguf OUT.gguf --spec Q0FIT.json [--a-on 5 --a-off 6]

Q0FIT.json: {"spec": {"T", "D", "Z", "IP", "EA", "h", "l", "ap", "am", "bp", "bm"}} (fit_q0.py / lrc_ref.FermiQ0.spec);
the element order must be the GGUF's pet.plato_lr.sqe_Z.
"""
import argparse
import json
import struct

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("output"); ap.add_argument("--spec", default=None)
    ap.add_argument("--a-on", type=float, default=5.0); ap.add_argument("--a-off", type=float, default=6.0)
    ap.add_argument("--mode", choices=["fermi2", "head", "round"], default="fermi2",
                    help="q0: two-level Fermi (needs --spec), the sqe learned logistic head alone, or soft-rounded")
    ap.add_argument("--tau", type=float, default=0.25); ap.add_argument("--merge-w", type=float, default=2.0)
    a = ap.parse_args()
    sp = {"T": 0.25, "D": 5.0}
    if a.mode == "fermi2":
        if not a.spec: raise SystemExit("--mode fermi2 needs --spec")
        sp = json.load(open(a.spec)); sp = sp.get("spec", sp)
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

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(SC[t])

    enc = lambda s: struct.pack("<Q", len(s.encode())) + s.encode()
    P = "pet.plato_lr.lrc"
    new = {P: enc(P) + struct.pack("<Ii", 5, 1)}
    for k, v in (("a_on", a.a_on), ("a_off", a.a_off), ("T", sp["T"]), ("D", sp["D"]), ("tau", a.tau), ("merge_w", a.merge_w)):
        new[f"{P}_{k}"] = enc(f"{P}_{k}") + struct.pack("<Id", 12, float(v))
    new[f"{P}_q0_mode"] = enc(f"{P}_q0_mode") + struct.pack("<I", 8) + enc(a.mode)
    magic, version = take("I"), take("I")
    assert magic == 0x46554747 and version == 3, (hex(magic), version)
    nt, nkv = take("Q"), take("Q")
    kept, alignment, meta = [], 32, {}
    for _ in range(nkv):
        k0 = pos; k = string(); t = take("I"); v0 = pos
        v = value(t)
        if k in ("general.alignment", "pet.plato_lr.sqe", "pet.plato_lr.sqe_Z", "pet.plato_lr.sqe_q0", "pet.plato_lr.sqe_lr_R"):
            meta[k] = v
        if not k.startswith(P):
            kept.append(data[k0:pos])
    alignment = int(meta.get("general.alignment", 32))
    if not meta.get("pet.plato_lr.sqe"):
        raise SystemExit(f"{a.input}: no sqe keys (run add_sqe_keys.py first)")
    if meta.get("pet.plato_lr.sqe_q0") not in ("uniform",) or not meta.get("pet.plato_lr.sqe_lr_R", 0) > 0:
        raise SystemExit(f"{a.input}: needs sqe_q0 uniform and sqe_lr_R > 0")
    Z = [int(z) for z in meta["pet.plato_lr.sqe_Z"]]
    if a.mode != "fermi2":
        Z, sp["Z"] = [], []  # the head lives in the sqe_q0_* tables
    zi = {int(z): i for i, z in enumerate(sp["Z"])}
    E0 = len(sp["Z"])
    for k in (("ip", "ea", "h", "l", "ap", "am") if a.mode == "fermi2" else ()):
        src = sp[{"ip": "IP", "ea": "EA"}.get(k, k)]
        v = [float(src[zi[z]]) for z in Z]
        new[f"{P}_{k}"] = enc(f"{P}_{k}") + struct.pack("<IIQ", 9, 12, len(v)) + struct.pack(f"<{len(v)}d", *v)
    for k in (("bp", "bm") if a.mode == "fermi2" else ()):
        v = [float(sp[k][zi[z] * E0 + zi[z2]]) for z in Z for z2 in Z]
        new[f"{P}_{k}"] = enc(f"{P}_{k}") + struct.pack("<IIQ", 9, 12, len(v)) + struct.pack(f"<{len(v)}d", *v)
    ti0 = pos
    for _ in range(nt):
        string(); nd = take("I"); [take("Q") for _ in range(nd)]; take("I"); take("Q")
    ti = data[ti0:pos]
    data_start = (pos + alignment - 1) // alignment * alignment
    body = data[data_start:]
    extra = list(new.values())
    head = struct.pack("<IIQQ", magic, version, nt, len(kept) + len(extra)) + b"".join(kept) + b"".join(extra) + ti
    head += b"\0" * ((-len(head)) % alignment)
    with open(a.output, "wb") as f:
        f.write(head); f.write(body)
    print(f"{a.output}: lrc on (q0 {a.mode}, clusters {a.a_on}-{a.a_off} A)")


if __name__ == "__main__":
    main()
