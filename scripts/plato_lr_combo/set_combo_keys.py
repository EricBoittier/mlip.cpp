#!/usr/bin/env python3
"""A copy of an lrc graph-mode GGUF with the combo components of feat/plato-lr-combo (after M. Langer, ipsum/PET-LR;
~/data/plato-lr-scratch/combo/SPEC.md): replaces / adds pet.plato_lr.lrc_q0_ref and its parameters, the flows, the
conditioning mode and the per-atom charge readout (tensors lrc_readout_weight [2, d] ~ 0.01 N(0, 1/d) and lrc_readout_bias
[2] = (0, -4), appended to train.param_names and graph.weight_shapes). Everything else, tensors included, is unchanged.
Paths are stored as given (relative: next to OUT.gguf, like lrc_qmodel). The link / Coulomb windows are patched with
scripts/plato_lr/patch_gguf.py --kv (in place).
    python set_combo_keys.py IN.gguf OUT.gguf [--ref fukui|fermi|threshold|species_fill|qeq] [--params P.json]
        [--smodel s.gguf] [--fermi-K 3] [--fermi-T 0.1] [--smear 0.9] [--flows 0] [--flow-kernel local|coulomb]
        [--flow-rc 5] [--cond cluster|global|neutral] [--readout] [--seed 13]
P.json (fit_combo.py): {"Z": [...], "theta" | "sf_score" + "sf_scale" | "qeq_eta": [per Z], "kappa": [Z x Z]}; the
element order must equal the GGUF's lrc_fukui_Z (missing elements: theta 0, eta softplus^-1(5), kappa softplus^-1(1e-3))."""
import argparse, json, math, struct
import numpy as np

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("output")
ap.add_argument("--ref", choices=["fukui", "fermi", "threshold", "species_fill", "qeq"], default="fukui")
ap.add_argument("--params"); ap.add_argument("--smodel")
ap.add_argument("--fermi-K", type=int, default=3); ap.add_argument("--fermi-T", type=float, default=0.1)
ap.add_argument("--smear", type=float, default=0.9)
ap.add_argument("--flows", type=int, default=0); ap.add_argument("--flow-kernel", choices=["local", "coulomb"], default="local")
ap.add_argument("--flow-rc", type=float, default=5.0)
ap.add_argument("--cond", choices=["cluster", "global", "neutral"])
ap.add_argument("--readout", action="store_true"); ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--readout-b0", type=float, default=-4.0, help="initial bias of b (softplus(-4) = 0.018 eV/e^2: small initial hardness term)")
ap.add_argument("--readout-scale", type=float, default=0.01, help="W ~ scale N(0, 1/d) (default 0.01: a small non-zero init, not a gate: dE/dW = [h q, h q^2 ...] does not vanish with W; scale 1 perturbs C8q0 forces by ~2 eV/A)")
a = ap.parse_args()
if a.ref in ("fermi", "threshold") and not a.smodel: raise SystemExit("--ref fermi / threshold needs --smodel")
data = open(a.input, "rb").read()
pos = 0


def take(fmt):
    global pos
    v = struct.unpack_from("<" + fmt, data, pos); pos += struct.calcsize("<" + fmt)
    return v[0] if len(v) == 1 else v


def string():
    global pos
    n = take("Q"); s = data[pos:pos + n].decode(); pos += n
    return s


def value(t):
    if t == 8: return string()
    if t == 9:
        et, n = take("I"), take("Q")
        return [value(et) for _ in range(n)]
    return take(SC[t])


enc = lambda s: struct.pack("<Q", len(s.encode())) + s.encode()
P = "pet.plato_lr.lrc"
new = {}
sval = lambda k, v: new.__setitem__(k, enc(k) + struct.pack("<I", 8) + enc(v))
ival = lambda k, v: new.__setitem__(k, enc(k) + struct.pack("<Ii", 5, int(v)))
dval = lambda k, v: new.__setitem__(k, enc(k) + struct.pack("<Id", 12, float(v)))
arr = lambda k, v: new.__setitem__(k, enc(k) + struct.pack("<IIQ", 9, 12, len(v)) + struct.pack(f"<{len(v)}d", *v))

magic, version = take("I"), take("I")
assert magic == 0x46554747 and version == 3, (hex(magic), version)
nt, nkv = take("Q"), take("Q")
kvs = []
meta = {}
for _ in range(nkv):
    k0 = pos; k = string(); t = take("I"); v = value(t)
    meta[k] = v; kvs.append((k, data[k0:pos]))
if meta.get(f"{P}_q0_mode") != "graph": raise SystemExit(f"{a.input}: not an lrc graph-mode GGUF")
FZ = [int(round(z)) for z in meta[f"{P}_fukui_Z"]]
E = len(FZ)
prm = json.load(open(a.params)) if a.params else {}
pz = {int(z): i for i, z in enumerate(prm.get("Z", []))}


def per_z(key, default):
    return [float(prm[key][pz[z]]) if key in prm and z in pz else default for z in FZ]


if a.ref != "fukui":
    sval(f"{P}_q0_ref", a.ref); ival(f"{P}_fermi_K", a.fermi_K); dval(f"{P}_fermi_T", a.fermi_T); dval(f"{P}_smear", a.smear)
    if a.smodel: sval(f"{P}_smodel", a.smodel)
    if a.ref == "threshold": arr(f"{P}_thr_theta", per_z("theta", 0.0))
    if a.ref == "species_fill": arr(f"{P}_sf_score", per_z("sf_score", 0.0)); arr(f"{P}_sf_scale", per_z("sf_scale", math.log(math.e - 1)))
    if a.ref == "qeq": arr(f"{P}_qeq_eta", per_z("qeq_eta", math.log(math.expm1(5.0))))
if a.flows > 0:
    k0 = math.log(math.expm1(1e-3))
    K = [[k0] * E for _ in range(E)]
    if "kappa" in prm:
        for i, zi in enumerate(FZ):
            for j, zj in enumerate(FZ):
                if zi in pz and zj in pz: K[i][j] = float(prm["kappa"][pz[zi]][pz[zj]])
    ival(f"{P}_flow_passes", a.flows); sval(f"{P}_flow_kernel", a.flow_kernel); dval(f"{P}_flow_rc", a.flow_rc)
    dval(f"{P}_smear", a.smear); arr(f"{P}_flow_kappa", [v for r in K for v in r])
if a.cond: sval(f"{P}_cond", a.cond)
alignment = int(meta.get("general.alignment", 32))
ti0 = pos
tinfo = []
for _ in range(nt):
    name = string(); nd = take("I"); dims = [take("Q") for _ in range(nd)]; tt = take("I"); off = take("Q")
    tinfo.append((name, dims, tt, off))
ti = data[ti0:pos]
body = bytearray(data[(pos + alignment - 1) // alignment * alignment:])
new_t = []
if a.readout:
    if any(n.startswith("lrc_readout") for n, *_ in tinfo): raise SystemExit("already has readout tensors")
    ws = json.loads(meta["graph.weight_shapes"])
    heads = sorted(k for k in ws if k.startswith("node_energy_heads_") and k.endswith("_0_weight"))
    d = int(ws[heads[-1]][1])
    rng = np.random.default_rng(a.seed)
    W = (a.readout_scale * rng.standard_normal((2, d)) / math.sqrt(d)).astype(np.float32)
    b = np.array([0.0, a.readout_b0], np.float32)
    ws["lrc_readout_weight"] = [2, d]; ws["lrc_readout_bias"] = [2]
    pn = json.loads(meta["train.param_names"]); pn += ["lrc_readout_weight", "lrc_readout_bias"]
    new["graph.weight_shapes"] = enc("graph.weight_shapes") + struct.pack("<I", 8) + enc(json.dumps(ws))
    new["train.param_names"] = enc("train.param_names") + struct.pack("<I", 8) + enc(json.dumps(pn))
    ival(f"{P}_readout", 1)
    for name, arrv, dims in (("lrc_readout_weight", W, [d, 2]), ("lrc_readout_bias", b, [2])):
        body += b"\0" * ((-len(body)) % alignment)
        new_t.append(enc(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims)
                     + struct.pack("<IQ", 0, len(body)))
        body += arrv.tobytes()
    body += b"\0" * ((-len(body)) % alignment)  # gguf reads the padded size of the last tensor
kept = [raw for k, raw in kvs if k not in new]
head = struct.pack("<IIQQ", magic, version, nt + len(new_t), len(kept) + len(new)) + b"".join(kept) + b"".join(new.values()) + ti + b"".join(new_t)
head += b"\0" * ((-len(head)) % alignment)
with open(a.output, "wb") as f:
    f.write(head); f.write(body)
print(f"{a.output}: ref {a.ref}, flows {a.flows} ({a.flow_kernel}), cond {a.cond or 'unchanged'}, readout {a.readout}")
