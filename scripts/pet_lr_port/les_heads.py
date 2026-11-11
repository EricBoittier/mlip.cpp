#!/usr/bin/env python3
"""Per-head SIGNED energy error of the les-water per_layer_readout presets (after M. Langer, ipsum-work les-water
LayeredBackbone + per_layer_readout): from MLIPCPP_NODE_GRAD_DUMP files of the same structures (e.g. cpu= and cuda=),
each head's per-atom energy (node part `select`, edge part `sum`, head `add`, gated `mul`) and the running sum, as the
frame sum of (value - fp64) / sum_i |E_i fp64| (the energy metric of lr_parity.py), with the torch fp32 replay of the
same FX graph on the same inputs for scale. The fp64 values come from compare_node_grads_iris.py's promoting
interpreter on the dump's own fp32 inputs, so the input rounding is common to all columns: a systematic per-head
shift that one backend has and the others not names the head (and with --local, the op) that carries it.

    python les_heads.py PARAMS.npz CKPT_DIR --dump cpu=A.bin --dump cuda=B.bin [--frames 0-15] [--local]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import compare_node_grads_iris as c  # noqa: E402


def head_nodes(gm):
    """[(label, node name)] of every head's pieces and the running sum, from the FX graph: a `mul` by a
    p_layer_gate_<l> parameter (gated heads) or the adds of the head sum (ungated)."""
    out = []
    by = {n.name: n for n in gm.graph.nodes}
    final = [n for n in gm.graph.nodes if n.op == "output"][0].all_input_nodes[0]
    # walk the running sum back from the output: add(prev, head) chains
    chain = []
    n = final
    while n.op == "call_function" and "add" in str(n.target):
        chain.append(n)
        a = [x for x in n.args if hasattr(x, "op")]
        if len(a) < 1:
            break
        n = a[0]
    chain = chain[::-1]
    for k, s in enumerate(chain):
        args = [x for x in s.args if hasattr(x, "op")]
        h = args[-1] if len(args) > 1 else args[0]
        out.append((f"sum<= {k}", s.name))
        g = h
        if "mul" in str(h.target):  # the gate
            out.append((f"head{k} gated", h.name))
            g = [x for x in h.args if hasattr(x, "op") and x.op == "call_function"][0]
        out.append((f"head{k}", g.name))
        for x in g.args:
            if hasattr(x, "op") and x.op == "call_function":
                out.append((f"head{k} {'node' if 'select' in x.name else 'edge'}", x.name))
    del by
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt")
    ap.add_argument("--dump", action="append", required=True)
    ap.add_argument("--frames", default="0")
    a = ap.parse_args()
    frames = []
    for tok in a.frames.split(","):
        lo, _, hi = tok.partition("-")
        frames += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    paths = dict(d.split("=", 1) for d in a.dump)
    ep, names = c.export_program(a.params, a.ckpt, "trunk")
    nodes = head_nodes(ep.graph_module)
    labs = list(paths) + ["t32"]
    acc = {(lab, nm): [] for lab in labs for _, nm in nodes}
    for f in frames:
        dumps = {lab: c.read_dump(p, f) for lab, p in paths.items()}
        first = next(iter(dumps.values()))
        rec64 = c.run(ep, names, first, torch.float64, "trunk")
        rec32 = c.run(ep, names, first, torch.float32, "trunk")
        fin = [n for n in ep.graph_module.graph.nodes if n.op == "output"][0].all_input_nodes[0].name
        norm = float(np.abs(rec64.saved[fin].detach().numpy()).sum())
        for lab in labs:
            for _, nm in nodes:
                v64 = rec64.saved[nm].detach().numpy().astype(np.float64)
                if lab == "t32":
                    v = rec32.saved[nm].detach().double().numpy()
                else:
                    v = dumps[lab][nm][1]
                    if v is None or v.size != v64.size:
                        acc[(lab, nm)].append(np.nan); continue
                    v = v.reshape(v64.shape).astype(np.float64)
                acc[(lab, nm)].append(float((v - v64).sum()) / norm)
        print(f"frame {f} done", flush=True)
    print(f"\nsigned frame-sum error / sum|E_i fp64|, mean over {len(frames)} structures (median in brackets):")
    print(f"  {'piece':16s} {'node':12s} " + " ".join(f"{l:>22s}" for l in labs))
    for lab_, nm in nodes:
        print(f"  {lab_:16s} {nm:12s} " + " ".join(
            f"{np.nanmean(acc[(l, nm)]):+.2e} ({np.nanmedian(acc[(l, nm)]):+.1e})" for l in labs))
    if len(paths) > 1:
        l0, l1 = list(paths)[:2]
        print(f"\n{l1} - {l0} per piece (mean, and fraction of structures where it is > 0):")
        for lab_, nm in nodes:
            d = np.asarray(acc[(l1, nm)]) - np.asarray(acc[(l0, nm)])
            print(f"  {lab_:16s} {nm:12s} {np.nanmean(d):+.2e}  P>0 {np.nanmean(d > 0):.2f}")


if __name__ == "__main__":
    main()
