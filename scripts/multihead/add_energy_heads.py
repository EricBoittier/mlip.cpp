#!/usr/bin/env python3
"""A multi-head (one energy head per level of theory) copy of a PET GGUF, or one head of a multi-head GGUF as a plain
single-head GGUF.

    python add_energy_heads.py add IN.gguf OUT.gguf --names omega_b97m_v,pbe0_mbd [--composition 1:COMP.json]
    python add_energy_heads.py select IN.gguf OUT.gguf --head pbe0_mbd     # or an index

add: head 0 is IN's energy readout (and composition), named by the first --names entry. Every further head h is a copy
of the readout tensors (the per-atom energy MLPs and final layers: the weights the per-atom energy depends on that
neither the backbone nor the direct-force head uses) named "head<h>." + name, with the composition pet.head<h>.
composition_{keys,values} (a copy of head 0's unless --composition h:FILE gives {"Z": eV, ...}). The copies are
trainable (train.param_names; state-dict names "backend.head<h>." + the original without "backend.").
pet.energy_heads = H and pet.energy_head_names (JSON) are set. Readers without head support see head 0 only.

select: OUT predicts with head h for every reader (its tensors and composition in place of head 0's, the head<k>.*
copies and keys dropped). mlip.cpp also selects a head at load time with MLIPCPP_ENERGY_HEAD=<name or index>.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_raw import GGUF  # noqa: E402


def readout_weights(g):
    """Weights between the backbone features and the per-atom energy (runtime/energy_heads.cpp's energy node)."""
    graph = json.loads(g.get_str("graph.json"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    nid = lambda r: int(r.split(":")[1])
    out = nodes[nid(graph["outputs"][0]["node_ref"])]
    others = []
    if out["op"] == "CONCAT":
        others = [nid(r) for r in out["inputs"][1:] if r.startswith("node:")]
        out = nodes[nid(out["inputs"][0])]
    while out["op"] in ("RESHAPE", "CONT", "VIEW"):
        out = nodes[nid(out["inputs"][0])]

    def ancestors(start):
        seen, stack, weights = set(), list(start), {}
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            for r in nodes[i]["inputs"]:
                if r.startswith("node:"):
                    stack.append(nid(r))
                elif r.startswith("weight:"):
                    weights.setdefault(r[7:], set()).add(i)
        return seen, weights

    e_nodes, e_w = ancestors([out["id"]])
    o_nodes, o_w = ancestors(others)
    # a readout weight feeds the energy only; its consumers' other ancestors are the backbone (shared with the force head)
    readout = [w for w in e_w if w not in o_w]
    if not readout:
        raise SystemExit("no energy-only weights found")
    return out["id"], sorted(readout, key=list(g.tensors).index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["add", "select"]); ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--names", default="omega_b97m_v,pbe0_mbd")
    ap.add_argument("--composition", action="append", default=[], help="h:FILE.json ({\"Z\": eV})")
    ap.add_argument("--head", default=None)
    a = ap.parse_args()
    g = GGUF(a.input)
    if a.mode == "add":
        if g.get_int("pet.energy_heads", 1) > 1:
            raise SystemExit(f"{a.input} has energy heads already")
        names = a.names.split(",")
        H = len(names)
        node, ws = readout_weights(g)
        print(f"energy node {node}; readout tensors per head ({len(ws)}): {', '.join(ws)}")
        pn = json.loads(g.get_str("train.param_names")); sd = json.loads(g.get_str("train.state_dict_names"))
        shapes = json.loads(g.get_str("graph.weight_shapes"))
        ck, cv = g.get_array("pet.composition_keys"), g.get_array("pet.composition_values")
        comps = {int(s.split(":")[0]): json.load(open(s.split(":", 1)[1])) for s in a.composition}
        last = list(g.tensors)[-1]
        for h in range(1, H):
            p = f"head{h}."
            for w in ws:
                g.set_tensor(p + w, g.array(w).copy(), after=last); last = p + w
                if w in pn:
                    pn.append(p + w)
                if w in sd:
                    sd[p + w] = "backend." + p + sd[w].removeprefix("backend.")
                if w in shapes:
                    shapes[p + w] = shapes[w]
            if h in comps:
                c = {int(k): float(v) for k, v in comps[h].items()}
                keys = sorted(set(ck) | set(c))
                vals = [c.get(z, cv[ck.index(z)] if z in ck else 0.0) for z in keys]
            else:
                keys, vals = ck, cv
            g.set_array(f"pet.head{h}.composition_keys", keys, 5)
            g.set_array(f"pet.head{h}.composition_values", vals, 6)
        g.set_str("train.param_names", json.dumps(pn)); g.set_str("train.state_dict_names", json.dumps(sd))
        g.set_str("graph.weight_shapes", json.dumps(shapes))
        g.set_i32("pet.energy_heads", H); g.set_str("pet.energy_head_names", json.dumps(names))
        g.write(a.output)
        print(f"wrote {a.output}: {H} heads ({', '.join(names)})")
        return
    H = g.get_int("pet.energy_heads", 1)
    names = json.loads(g.get_str("pet.energy_head_names", "[]"))
    if a.head is None:
        raise SystemExit("--head NAME|INDEX")
    h = names.index(a.head) if a.head in names else int(a.head)
    if not 0 <= h < H:
        raise SystemExit(f"no head {a.head}")
    pn = json.loads(g.get_str("train.param_names")); sd = json.loads(g.get_str("train.state_dict_names"))
    shapes = json.loads(g.get_str("graph.weight_shapes"))
    copies = [t for t in g.tensors if t.startswith("head") and "." in t and t.split(".")[0][4:].isdigit()]
    base = sorted({t.split(".", 1)[1] for t in copies})
    if h > 0:
        for w in base:
            g.tensors[w] = g.tensors[f"head{h}.{w}"]
        g.kv["pet.composition_keys"] = g.kv[f"pet.head{h}.composition_keys"]
        g.kv["pet.composition_values"] = g.kv[f"pet.head{h}.composition_values"]
    for t in copies:
        del g.tensors[t]
        pn = [x for x in pn if x != t]; sd.pop(t, None); shapes.pop(t, None)
    for k in [k for k in g.kv if k.startswith("pet.head") and ".composition_" in k] + ["pet.energy_heads", "pet.energy_head_names"]:
        g.kv.pop(k, None)
    g.set_str("train.param_names", json.dumps(pn)); g.set_str("train.state_dict_names", json.dumps(sd))
    g.set_str("graph.weight_shapes", json.dumps(shapes))
    g.set_str("general.energy_head", names[h] if h < len(names) else str(h))
    g.write(a.output)
    print(f"wrote {a.output}: head {h} ({names[h] if h < len(names) else ''}) as the model's only head")


if __name__ == "__main__":
    main()
