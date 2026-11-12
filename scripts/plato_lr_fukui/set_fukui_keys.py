#!/usr/bin/env python3
"""A copy of an lrc graph-mode GGUF (add_lrc_keys.py --mode graph) with the Fukui equilibration arm of
feat/plato-lr-fukui: replaces pet.plato_lr.lrc_fukui_* / lrc_fmodel* (and lrc_qmodel with --qmodel); everything else,
tensors included, unchanged. Paths are stored as given (relative: next to OUT.gguf, like lrc_qmodel).
    python set_fukui_keys.py IN.gguf OUT.gguf --fukui FUKUI.json --mode element|uniform|atom|atom_pm
        [--scope global|cluster] [--qmodel q.gguf] [--fmodel f.gguf] [--fmodel-minus f_minus.gguf]
        [--cluster-w 0.6] [--pm-width 0.05]
FUKUI.json: {"Z", "phi"[, "phi_minus"]} (mk_fukui_ggufs.py)."""
import argparse, json, struct

SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("output"); ap.add_argument("--fukui", required=True)
ap.add_argument("--mode", choices=["element", "uniform", "atom", "atom_pm"], required=True)
ap.add_argument("--scope", choices=["global", "cluster"], default="global")
ap.add_argument("--qmodel"); ap.add_argument("--fmodel"); ap.add_argument("--fmodel-minus")
ap.add_argument("--cluster-w", type=float, default=0.6); ap.add_argument("--pm-width", type=float, default=0.05)
a = ap.parse_args()
if a.mode in ("atom", "atom_pm") and not a.fmodel: raise SystemExit("--mode atom / atom_pm needs --fmodel")
if a.mode == "atom_pm" and not a.fmodel_minus: raise SystemExit("--mode atom_pm needs --fmodel-minus")
if a.mode == "atom_pm" and a.scope == "cluster": raise SystemExit("--scope cluster: uniform / element / atom only")
fk = json.load(open(a.fukui))
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
sval = lambda k, v: new.__setitem__(f"{P}_{k}", enc(f"{P}_{k}") + struct.pack("<I", 8) + enc(v))
dval = lambda k, v: new.__setitem__(f"{P}_{k}", enc(f"{P}_{k}") + struct.pack("<Id", 12, float(v)))
arr = lambda k, v: new.__setitem__(f"{P}_{k}", enc(f"{P}_{k}") + struct.pack("<IIQ", 9, 12, len(v)) + struct.pack(f"<{len(v)}d", *v))
sval("fukui_mode", a.mode); sval("fukui_scope", a.scope)
dval("fukui_cluster_w", a.cluster_w); dval("fukui_pm_width", a.pm_width)
arr("fukui_Z", [float(z) for z in fk["Z"]]); arr("fukui_phi", [float(x) for x in fk["phi"]])
if "phi_minus" in fk: arr("fukui_phi_minus", [float(x) for x in fk["phi_minus"]])
if a.fmodel: sval("fmodel", a.fmodel)
if a.fmodel_minus: sval("fmodel_minus", a.fmodel_minus)
if a.qmodel: sval("qmodel", a.qmodel)
magic, version = take("I"), take("I")
assert magic == 0x46554747 and version == 3, (hex(magic), version)
nt, nkv = take("Q"), take("Q")
kept, meta = [], {}
for _ in range(nkv):
    k0 = pos; k = string(); t = take("I"); v = value(t)
    if k in ("general.alignment", "pet.plato_lr.lrc_q0_mode"): meta[k] = v
    if k not in new and not k.startswith(f"{P}_fukui") and not k.startswith(f"{P}_fmodel"):
        kept.append(data[k0:pos])
if meta.get("pet.plato_lr.lrc_q0_mode") != "graph": raise SystemExit(f"{a.input}: not an lrc graph-mode GGUF")
alignment = int(meta.get("general.alignment", 32))
ti0 = pos
for _ in range(nt):
    string(); nd = take("I"); [take("Q") for _ in range(nd)]; take("I"); take("Q")
ti = data[ti0:pos]
body = data[(pos + alignment - 1) // alignment * alignment:]
head = struct.pack("<IIQQ", magic, version, nt, len(kept) + len(new)) + b"".join(kept) + b"".join(new.values()) + ti
head += b"\0" * ((-len(head)) % alignment)
with open(a.output, "wb") as f:
    f.write(head); f.write(body)
print(f"{a.output}: Fukui {a.mode} / {a.scope}" + (f", f pass {a.fmodel}" if a.fmodel else ""))
