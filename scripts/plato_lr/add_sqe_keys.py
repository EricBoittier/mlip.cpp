#!/usr/bin/env python3
"""A copy of a Plato-PET-LR GGUF with split-charge equilibration switched on (pet.plato_lr.sqe = 1 and the sqe_* keys;
keys already present are replaced). Tensors unchanged: the SQE parameters are float64 metadata (host side).

    python add_sqe_keys.py IN.gguf OUT.gguf --spec SPEC.json [--q0 uniform|formal] [--backbone-charge input|neutral]

SPEC.json: {"Z": [...], "chi0", "J0", "rcov", "sigma", "chi_a", "chi_b", "j_z", "kappa_z", "q0_lo", "q0_hi", "q0_a": [per element], "q0_b": [E x E, row-major], "w_raw": x,
"cfg": {"r_ct", "r_ct_width", "dchi_max", "J_lo", "J_hi", "kappa_lo", "kappa_hi", "w_lo", "w_hi", "cn_on", "cn_off",
"ewald_alpha"}}, written by metatrain's ``metatrain.pet.modules.sqe.spec_json`` (tabulated start: a fresh
``SQEParams(atomic_numbers)``).
"""
import argparse
import json
import struct

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
CFG = ["r_ct", "r_ct_width", "dchi_max", "J_lo", "J_hi", "kappa_lo", "kappa_hi", "w_lo", "w_hi", "cn_on", "cn_off",
       "ewald_alpha", "lr_R", "lr_t"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("output"); ap.add_argument("--spec", required=True)
    ap.add_argument("--q0", choices=["uniform", "formal", "learned"], default="uniform")
    ap.add_argument("--backbone-charge", choices=["input", "neutral"], default="input")
    ap.add_argument("--lr-R", type=float, default=None, help="override the spec's lr_R (long-range-only SQE energy)")
    ap.add_argument("--lr-t", type=float, default=None, help="override the spec's lr_t (switch width)")
    a = ap.parse_args()
    sp = json.load(open(a.spec))
    if a.lr_R is not None:
        sp["cfg"]["lr_R"] = a.lr_R
    if a.lr_t is not None:
        sp["cfg"]["lr_t"] = a.lr_t
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

    enc = lambda s: struct.pack("<Q", len(s.encode())) + s.encode()
    P = "pet.plato_lr.sqe"
    new = {P: enc(P) + struct.pack("<Ii", 5, 1),
           P + "_q0": enc(P + "_q0") + struct.pack("<I", 8) + enc(a.q0),
           P + "_backbone_charge": enc(P + "_backbone_charge") + struct.pack("<I", 8) + enc(a.backbone_charge)}
    for k in CFG:
        new[f"{P}_{k}"] = enc(f"{P}_{k}") + struct.pack("<Id", 12, float(sp["cfg"].get(k, {"lr_R": 0.0, "lr_t": 1.0}.get(k, 0.0))))
    new[P + "_w_raw"] = enc(P + "_w_raw") + struct.pack("<Id", 12, float(sp["w_raw"]))
    Z = [int(z) for z in sp["Z"]]
    new[P + "_Z"] = enc(P + "_Z") + struct.pack("<IIQ", 9, 5, len(Z)) + struct.pack(f"<{len(Z)}i", *Z)
    for k in ("chi0", "J0", "rcov", "sigma", "chi_a", "chi_b", "j_z", "kappa_z", "q0_lo", "q0_hi", "q0_a", "q0_b"):
        if k not in sp:
            if k.startswith("q0") and a.q0 != "learned":
                continue
            raise SystemExit(f"{a.spec}: no {k}")
        v = [float(x) for x in sp[k]]; assert len(v) == len(Z) * (len(Z) if k == "q0_b" else 1), k
        new[f"{P}_{k}"] = enc(f"{P}_{k}") + struct.pack("<IIQ", 9, 12, len(v)) + struct.pack(f"<{len(v)}d", *v)
    magic, version = take("I"), take("I")
    assert magic == 0x46554747 and version == 3, (hex(magic), version)
    nt, nkv = take("Q"), take("Q")
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
        string(); nd = take("I"); [take("Q") for _ in range(nd)]; take("I"); take("Q")
    ti = data[ti0:pos]
    data_start = (pos + alignment - 1) // alignment * alignment
    body = data[data_start:]
    extra = list(new.values())
    head = struct.pack("<IIQQ", magic, version, nt, len(kept) + len(extra)) + b"".join(kept) + b"".join(extra) + ti
    head += b"\0" * ((-len(head)) % alignment)
    with open(a.output, "wb") as f:
        f.write(head); f.write(body)
    print(f"{a.output}: sqe on ({len(Z)} elements, q0 {a.q0}, backbone charge {a.backbone_charge}, r_ct {sp['cfg']['r_ct']} A)")


if __name__ == "__main__":
    main()
