#!/usr/bin/env python3
"""A copy of a Plato-PET-LR GGUF with fragment-constrained charges switched on (pet.plato_lr.frag_charges = 1 and
frag_R / frag_t / frag_sigma / frag_scale added; keys already present are replaced). The tensors are unchanged (the
term has no parameters): the model is the same for structures without formal charges.

    python add_frag_keys.py IN.gguf OUT.gguf [--R 8 --t 0.5 --sigma 4 --scale 1]
        [--cond-gate [--gate-value 0 | --gate-random SEED --gate-scale S]]

--cond-gate: the gated unit conditioning (pet.plato_lr.frag_cond_gate = 1) and its per-channel tensor
plato_frag_cond_gate (F32, the conditioning width; zeros by default, so the model is exactly the base one), added to
train.param_names (trainable) and train.state_dict_names (backend.frag.cond_gate).
"""
import argparse
import struct

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--R", type=float, default=8.0); ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=4.0); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cond-gate", action="store_true"); ap.add_argument("--gate-value", type=float, default=0.0)
    ap.add_argument("--gate-random", type=int, default=None); ap.add_argument("--gate-scale", type=float, default=0.5)
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
    if a.cond_gate:
        new["pet.plato_lr.frag_cond_gate"] = (5, 1)
    strings = {}  # train.param_names / train.state_dict_names (rewritten with --cond-gate)
    kept, alignment, plato = [], 32, False
    for _ in range(nkv):
        k0 = pos; k = string(); t = take("I"); v0 = pos
        if k == "general.alignment":
            alignment = struct.unpack_from("<I", data, v0)[0]
        if k == "pet.plato_lr":
            plato = struct.unpack_from("<i", data, v0)[0] != 0
        skip_value(t)
        if a.cond_gate and k in ("train.param_names", "train.state_dict_names"):
            n = struct.unpack_from("<Q", data, v0)[0]; strings[k] = data[v0 + 8:v0 + 8 + n].decode()
        elif k not in new:
            kept.append(data[k0:pos])
    if not plato:
        raise SystemExit(f"{a.input}: not a Plato-PET-LR GGUF (pet.plato_lr)")
    ti0 = pos; width = None
    for _ in range(nt):
        name = string(); nd = take("I"); dims = [take("Q") for _ in range(nd)]
        take("I"); take("Q")
        if name == "system_conditioning_project_2_bias": width = dims[0]
        if name == "plato_frag_cond_gate": raise SystemExit(f"{a.input}: has plato_frag_cond_gate already")
    ti = data[ti0:pos]
    data_start = (pos + alignment - 1) // alignment * alignment
    body = data[data_start:]
    enc = lambda s: struct.pack("<Q", len(s.encode())) + s.encode()
    extra = [enc(k) + struct.pack("<I", t) + struct.pack("<" + SC[t], v) for k, (t, v) in new.items()]
    if a.cond_gate:
        import json, random
        if width is None: raise SystemExit("no system_conditioning_project_2_bias (a conditioned model?)")
        pn = json.loads(strings["train.param_names"]); pn.append("plato_frag_cond_gate")
        sd = json.loads(strings["train.state_dict_names"]); sd["plato_frag_cond_gate"] = "backend.frag.cond_gate"
        for k, v in (("train.param_names", json.dumps(pn)), ("train.state_dict_names", json.dumps(sd))):
            extra.append(enc(k) + struct.pack("<I", 8) + enc(v))
        rng = random.Random(a.gate_random)
        vals = [a.gate_scale * rng.uniform(-1, 1) if a.gate_random is not None else a.gate_value for _ in range(width)]
        off = (len(body) + alignment - 1) // alignment * alignment
        ti += enc("plato_frag_cond_gate") + struct.pack("<I", 1) + struct.pack("<Q", width) + struct.pack("<I", 0) + struct.pack("<Q", off)
        body = body + b"\0" * (off - len(body)) + struct.pack(f"<{width}f", *vals)
        nt += 1
    head = struct.pack("<IIQQ", magic, version, nt, len(kept) + len(extra)) + b"".join(kept) + b"".join(extra) + ti
    head += b"\0" * ((-len(head)) % alignment)
    with open(a.output, "wb") as f:
        f.write(head); f.write(body)
    print(f"{a.output}: frag_charges on (R {a.R}, t {a.t}, sigma {a.sigma}, scale {a.scale})"
          + (f", cond gate width {width}" if a.cond_gate else ""))


if __name__ == "__main__":
    main()
