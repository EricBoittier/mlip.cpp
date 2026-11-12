#!/usr/bin/env python3
"""Charge and Fukui passes of the lrc graph mode from a fit_fukui.py fit: GGUF copies of the start model (the
backbone whose last-layer features the fit used, e.g. E8.gguf) whose energy head computes one fitted head, so the
per-atom outputs (divided by pet.energy_scale, no composition) are q_raw (q.gguf), the Fukui logit correction a_i
(f.gguf; f_i = softplus(phi_Z + a_i)) and, for atom_pm, a-_i (f_minus.gguf). Same mapping as petlr-chargehead
fit/mk_qgguf.py: node MLP -> node_energy_heads_0_0 / _0_2 / node_final_layers_0 (input normalisation folded in), the
linear edge term -> edge_final_layers_0 (its normalisation folded into the node bias), the count term -> its bias.
Also FUKUI.json ({"Z", "phi"[, "phi_minus"]}) for set_fukui_keys.py / mlipcpp-qpass.
    python mk_fukui_ggufs.py START.gguf FIT.pt OUTDIR"""
import json, os, shutil, sys
import numpy as np, torch, gguf

START, FIT, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
ck = torch.load(FIT, weights_only=False)
st = {k: v.double().numpy() for k, v in ck["state"].items()}
r = gguf.GGUFReader(START)
es = float(r.fields["pet.energy_scale"].parts[-1][0])
T = {t.name: t for t in r.tensors}
assert float(np.asarray(T["plato_les_gate"].data).ravel()[0]) == 0.0, "LES gate must be 0 in the start model"
mn, sn, me, se = st["mn"], st["sn"], st["me"], st["se"]


def write(prefix, out):
    W1, b1 = st[f"{prefix}.mlp.0.weight"], st[f"{prefix}.mlp.0.bias"]
    W1f = W1 / sn[None, :]; b1f = b1 - W1f @ mn
    W2, b2 = st[f"{prefix}.mlp.2.weight"], st[f"{prefix}.mlp.2.bias"]
    W3, b3 = st[f"{prefix}.mlp.4.weight"], st[f"{prefix}.mlp.4.bias"].copy()
    we, wb, be = st[f"{prefix}.we.weight"][0], float(st[f"{prefix}.we.bias"][0]), float(st[f"{prefix}.be"][0])
    wef = we / se
    b3 = b3 + wb - float(wef @ me)
    new = {"node_energy_heads_0_0_weight": W1f, "node_energy_heads_0_0_bias": b1f,
           "node_energy_heads_0_2_weight": W2, "node_energy_heads_0_2_bias": b2,
           "node_final_layers_0_weight": W3 / es, "node_final_layers_0_bias": b3 / es,
           "edge_final_layers_0_weight": wef[None, :] / es, "edge_final_layers_0_bias": np.array([be / es])}
    shutil.copyfile(START, out)
    mm = np.memmap(out, dtype=np.uint8, mode="r+")
    for k, v in new.items():
        t = T[k]
        assert t.tensor_type == 0 and int(np.prod(t.shape)) == v.size, (k, t.shape, v.shape)
        mm[t.data_offset:t.data_offset + 4 * v.size] = np.ascontiguousarray(v, dtype=np.float32).ravel().view(np.uint8)
    mm.flush()
    print(f"{out}: head {prefix} (energy scale {es})")


write("q", f"{OUT}/q.gguf")
fk = {"Z": [int(z) for z in ck["ZS"]], "phi": [float(x) for x in st["phi"]], "mode": ck["mode"]}
if "fa.mlp.0.weight" in st:
    write("fa", f"{OUT}/f.gguf")
if "fam.mlp.0.weight" in st:
    write("fam", f"{OUT}/f_minus.gguf")
    fk["phi_minus"] = [float(x) for x in st["phim"]]
json.dump(fk, open(f"{OUT}/fukui.json", "w"))
print(f"{OUT}/fukui.json: {len(fk['Z'])} elements, mode {ck['mode']}")
