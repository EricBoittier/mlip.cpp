#!/usr/bin/env python3
"""Per-node fp64 oracle for the iris PET-LR graphs (after M. Langer, ipsum/PET-LR): which op loses precision?

mlip.cpp writes, with MLIPCPP_NODE_GRAD_DUMP=<file> (plain path, one record per structure), every GIR node's forward
value and its gradient dE/d(node) from the ggml backward graph, plus the graph inputs. This script re-exports the
same module as iris_to_gguf.py (torch.export, same example shapes, so FX node names = GIR node names), and runs the
exported FX graph on mlip.cpp's own inputs

  * in float64 (a promoting fx.Interpreter: every fp32 tensor and dtype argument becomes fp64): the oracle;
  * in float32 (the same FX graph, torch's kernels): an independent fp32 implementation for scale,

retaining every intermediate gradient of loss = sum of the energy column (the C++ loss). Per node it reports the
global error (C++ vs fp64, torch fp32 vs fp64) of the value and of dE/d(node), and the LOCAL error of each op:

  * forward local: the op replayed in fp64 on mlip.cpp's own fp32 input values, vs mlip.cpp's output
    (torch fp32 replay on the same inputs for scale);
  * backward local: dE/d(node) rebuilt in fp64 as the sum over consumers of the VJP at mlip.cpp's values, seeded
    with mlip.cpp's dE/d(consumer), vs mlip.cpp's dE/d(node).

A node whose local error is far above the fp32 replay's (ratio column) is an op that loses precision; one at ratio
~1 with a large global error only carries its inputs' error (reduction-order noise propagates, it does not amplify).

    python compare_node_grads_iris.py PARAMS.npz CKPT_DIR --kind trunk|lr|pbc --dump cpu=A.bin [--dump cuda=B.bin]
                                      [--frames 0-9] [--top 20] [--table]
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.fx
from torch.utils._pytree import tree_map

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import check_grad64 as cg  # noqa: E402
from iris_torch import IrisPETLRGraph, IrisPETLRGraphPBC  # noqa: E402


# ---------------------------------------------------------------- the C++ dump
def read_dump(path, frame=0):
    raw = open(path, "rb").read()
    off, f = 0, 0
    while off < len(raw):
        magic, cnt = np.frombuffer(raw, np.int32, 2, off); off += 8
        assert magic == 0x454D5246, "not a MLIPCPP_NODE_GRAD_DUMP file"
        out = {}

        def block():
            nonlocal off
            nd = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
            ne = np.frombuffer(raw, np.int64, nd, off).copy(); off += 8 * nd
            nb = np.frombuffer(raw, np.int64, nd, off).copy(); off += 8 * nd
            dt = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
            n = int(np.frombuffer(raw, np.int64, 1, off)[0]); off += 8
            buf = raw[off:off + n]; off += n
            if dt not in (0, 1) or n == 0:
                return None
            arr = np.frombuffer(buf, np.float32 if dt == 0 else np.int32)
            # ggml ne / nb are innermost first: torch shape = reversed; views carry their raw extent and strides
            return np.lib.stride_tricks.as_strided(arr, shape=tuple(ne[::-1]), strides=tuple(nb[::-1])).copy()

        for _ in range(cnt):
            i, nl = np.frombuffer(raw, np.int32, 2, off); off += 8
            name = raw[off:off + nl].decode(); off += nl
            val = block()
            has = int(np.frombuffer(raw, np.int32, 1, off)[0]); off += 4
            grad = block() if has else None
            if f == frame:
                prev = out.get(name)
                if prev is None or int(i) > prev[0]:  # a name split into several GIR nodes: the last carries it
                    out[name] = (int(i), val, grad)
        if f == frame:
            return out
        f += 1
    raise IndexError(f"frame {frame} not in {path}")


# ---------------------------------------------------------------- the export (as iris_to_gguf.py)
def export_program(params, ckpt, kind, n_atoms=7, max_neighbors=11):
    from export_pytorch.export_pet_gguf import add_conditioning_inputs, example_export_args
    cfg, trunk, lr, scale = cg.build(params, ckpt)
    trunk, lr = copy.deepcopy(trunk).float(), copy.deepcopy(lr).float()
    inputs, names, shapes = example_export_args(n_atoms, max_neighbors, cfg["cutoff"], forces=True)
    if trunk.conditioning:
        inputs, names, shapes = add_conditioning_inputs(inputs, names, shapes, n_atoms)
    if kind == "trunk":
        module = trunk
    else:
        n, atom, nk = n_atoms, shapes[0][0], shapes[1]
        g = torch.Generator().manual_seed(0)
        extra = (torch.zeros(n, max_neighbors, dtype=torch.long), torch.randn(n, 3, generator=g),
                 torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(n), torch.zeros(n), torch.zeros(n),
                 torch.zeros(3))
        inputs = inputs + extra
        names = names + ["neighbor_index", "positions", "total_charge", "mu_bisect", "gapped", "m_round", "top",
                         "bottom", "dipole_origin"]
        shapes = shapes[:3] + (shapes[3] + (dict(nk), {0: atom}, None, None, None, {0: atom}, {0: atom}, {0: atom},
                                            None),)
        module = IrisPETLRGraph(trunk, lr, scale).eval()
        if kind == "pbc":
            kl, K = 13, 17
            n_lr = torch.export.Dim("n_lr", min=1, max=100000)
            n_kvec = torch.export.Dim("n_kvec", min=1, max=100000)
            ew_in = (torch.zeros(n, kl, dtype=torch.long), torch.randn(n, kl, 3, generator=g), torch.ones(n, kl),
                     torch.randn(K, 3, generator=g), torch.ones(1) * 1000.0, torch.ones(1) * 2.0,
                     torch.ones(1) * 8.0, torch.zeros(1), torch.tensor([0.0, 0.0, 1.0]), torch.ones(1))
            ew_shapes = ({0: atom, 1: n_lr}, {0: atom, 1: n_lr}, {0: atom, 1: n_lr}, {0: n_kvec}, None, None, None,
                         None, None, None)
            inputs = inputs + ew_in
            names = names + list(IrisPETLRGraphPBC.EWALD)
            shapes = shapes[:3] + (shapes[3] + ew_shapes,)
            module = IrisPETLRGraphPBC(trunk, lr, scale).eval()
    ep = torch.export.export(module, inputs, dynamic_shapes=shapes, strict=False)
    return ep, names


# ---------------------------------------------------------------- interpreters
def promote(x, dt):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.to(dt)
    if isinstance(x, torch.dtype) and x.is_floating_point:
        return dt
    return x


class Recorder(torch.fx.Interpreter):
    """Runs the exported graph with every float tensor / dtype argument cast to `dt`, keeping every node output."""

    def __init__(self, gm, dt):
        super().__init__(gm)
        self.dt, self.saved = dt, {}

    def run_node(self, n):
        if n.op == "call_function":
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            args, kwargs = tree_map(lambda x: promote(x, self.dt), (args, kwargs))
            out = n.target(*args, **kwargs)
        else:
            out = super().run_node(n)
            out = tree_map(lambda x: promote(x, self.dt), out) if n.op != "output" else out
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            if out.requires_grad:
                out.retain_grad()
            self.saved[n.name] = out
        return out


def placeholder_values(ep, names, cpp):
    """Exported-program placeholders: parameters / constants from the program, user inputs from mlip.cpp's dump."""
    vals, user = [], iter(names)
    for spec in ep.graph_signature.input_specs:
        k = spec.kind.name
        if k == "USER_INPUT":
            nm = next(user)
            _, v, _ = cpp["input_" + nm]
            t = torch.from_numpy(np.ascontiguousarray(v))
            if nm in ("species", "neighbor_species", "reverse_neighbor_index", "charge_index", "spin_index",
                      "neighbor_index", "lr_index"):
                t = t.long()
            elif nm == "padding_mask":
                t = t > 0.5
            vals.append(t)
        elif k in ("PARAMETER", "BUFFER"):
            vals.append(ep.state_dict[spec.target].detach() if spec.target in ep.state_dict
                        else ep.constants[spec.target])
        else:
            c = ep.constants[spec.target]
            if c.__class__.__name__ != "Tensor":
                raise RuntimeError(f"fake lifted constant {spec.target}: not materialised")
            vals.append(c)
    return vals


def run(ep, names, cpp, dt, kind):
    gm = ep.graph_module
    vals = placeholder_values(ep, names, cpp)
    user_names = iter(names)
    grads_for = {"edge_vectors", "cutoff_values", "positions"}
    out_vals = []
    for spec, v in zip(ep.graph_signature.input_specs, vals):
        if v.is_floating_point():
            v = v.to(dt)
            if spec.kind.name == "USER_INPUT" and next(user_names) in grads_for:
                v = v.clone().requires_grad_(True)
        elif spec.kind.name == "USER_INPUT":
            next(user_names)
        out_vals.append(v)
    rec = EnvRecorder(gm, dt)
    out = rec.run(*out_vals)
    out = out[0] if isinstance(out, (tuple, list)) else out
    E = out.sum() if kind == "trunk" else out[:, 0].sum()
    E.backward()
    return rec


def rel(a, b):
    s = max(float(np.abs(b).max()), 1e-30)
    return float(np.abs(a - b).max()) / s


def local_errors(gm, rec64, cpp, exact, bad):
    """Local error per op on mlip.cpp's own inputs (see the module doc). `exact`: placeholder names whose values are
    exact in both (parameters, constants, integer inputs); `bad`: nodes whose dumped tensor is not the FX value
    (fused / re-laid-out in the interpreter). A replay that would need the oracle's value for a float input is skipped."""
    fwd, bwd = {}, {}
    consumers = {}
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        for a in n.all_input_nodes:
            consumers.setdefault(a.name, []).append(n)

    def cval(name, ref, d):
        e = cpp.get(name)
        if name not in bad and e is not None and e[1] is not None and e[1].size == ref.numel() \
                and e[1].dtype == np.float32:
            return torch.from_numpy(e[1].reshape(tuple(ref.shape))).to(d), True
        return ref.detach().to(d), name in exact

    def replay(n, d, need_grad=False):
        leaves, pure = [], [True]

        def arg(x):
            if isinstance(x, torch.fx.Node):
                ref = rec64.env_vals.get(x.name)
                if isinstance(ref, torch.Tensor) and ref.is_floating_point():
                    v, ok = cval(x.name, ref, d)
                    pure[0] &= ok
                    if need_grad:
                        v = v.clone().requires_grad_(True)
                        leaves.append((x.name, v))
                    return v
                if isinstance(ref, (list, tuple)) and any(isinstance(t, torch.Tensor) and t.is_floating_point()
                                                          for t in ref):
                    pure[0] = False  # a multi-output op's float results (split / chunk): the oracle's, not mlip.cpp's
                    return type(ref)(t.detach().to(d) if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                                     for t in ref)
                return ref
            return promote(x, d)
        args = tree_map(arg, n.args, is_leaf=lambda x: isinstance(x, torch.fx.Node))
        kwargs = tree_map(arg, n.kwargs, is_leaf=lambda x: isinstance(x, torch.fx.Node))
        return n.target(*args, **kwargs), leaves, pure[0]

    for n in gm.graph.nodes:
        if n.op != "call_function" or n.name not in cpp or n.name in bad or n.name not in rec64.saved:
            continue
        e = cpp[n.name]
        ref = rec64.saved[n.name]
        if e[1] is None or e[1].size != ref.numel() or e[1].dtype != np.float32:
            continue
        try:
            with torch.no_grad():
                y64, _, pure = replay(n, torch.float64)
                y32, _, _ = replay(n, torch.float32)
        except Exception:
            continue
        if not pure or not isinstance(y64, torch.Tensor):
            continue
        c = e[1].reshape(tuple(y64.shape)).astype(np.float64)
        fwd[n.name] = (rel(c, y64.detach().numpy()), rel(y32.detach().double().numpy(), y64.detach().numpy()))

    # backward local: dE/d(v) = sum over consumers c of VJP_c(mlip.cpp values; seed mlip.cpp's dE/dc)
    for v_name, cons in consumers.items():
        e = cpp.get(v_name)
        if e is None or e[2] is None or v_name in bad:
            continue
        acc = {torch.float64: None, torch.float32: None}
        complete = True
        for c in cons:
            ec = cpp.get(c.name)
            ref = rec64.saved.get(c.name)
            if ec is None or ec[2] is None or ref is None or c.name in bad or ref.grad is None:
                complete = False  # a consumer without mlip.cpp's own dE/dc (or no gradient path): skip
                break
            try:
                for d in (torch.float64, torch.float32):
                    y, leaves, pure = replay(c, d, need_grad=True)
                    if not pure or not isinstance(y, torch.Tensor) or ec[2].size != y.numel():
                        raise ValueError
                    seed = torch.from_numpy(ec[2].reshape(tuple(y.shape))).to(d)
                    mine = [lv for nm, lv in leaves if nm == v_name]
                    gs = torch.autograd.grad(y, mine, seed, allow_unused=True)
                    tot = None
                    for g in gs:
                        if g is not None:
                            tot = g if tot is None else tot + g
                    tot = torch.zeros_like(mine[0]) if tot is None else tot
                    tot = tot.detach().double()
                    acc[d] = tot if acc[d] is None else acc[d] + tot
            except Exception:
                complete = False
                break
        if not complete or acc[torch.float64] is None or e[2].size != acc[torch.float64].numel():
            continue
        a64 = acc[torch.float64].numpy()
        g = e[2].reshape(a64.shape).astype(np.float64)
        bwd[v_name] = (rel(g, a64), rel(acc[torch.float32].numpy(), a64), len(cons))
    return fwd, bwd


class EnvRecorder(Recorder):
    """Recorder that also keeps non-float node values (index tensors, sizes) for the replays."""

    def __init__(self, gm, dt):
        super().__init__(gm, dt)
        self.env_vals = {}

    def run_node(self, n):
        out = super().run_node(n)
        self.env_vals[n.name] = out
        return out


FLOOR = 2.0 ** -24  # a local error below one fp32 ulp (relative) is no error


def analyse(ep, names, dumps, kind, frame):
    """One structure: global and local errors per node for every dump (the same structure)."""
    gm = ep.graph_module
    first = next(iter(dumps.values()))
    user = iter(names)
    alias, exact = {}, set()
    for spec in ep.graph_signature.input_specs:
        if spec.kind.name == "USER_INPUT":
            nm = next(user)
            alias[spec.arg.name] = "input_" + nm
            if nm not in ("edge_vectors", "cutoff_values", "positions", "lr_shift", "lr_keep", "kvec", "volume",
                          "smearing", "cutoff", "slab", "slab_normal", "slab_basis", "total_charge", "mu_bisect",
                          "gapped", "m_round", "top", "bottom", "dipole_origin"):
                exact.add(spec.arg.name)
        else:
            exact.add(spec.arg.name)
    for cpp in dumps.values():
        for a_, b_ in alias.items():
            if b_ in cpp:
                cpp[a_] = cpp[b_]
    # float inputs of the C++ run are its own fp32 values: exact for the replays too
    exact |= {a_ for a_ in alias if a_ not in exact}
    rec64 = run(ep, names, first, torch.float64, kind)
    rec32 = run(ep, names, first, torch.float32, kind)
    rows = {}
    order = [n.name for n in gm.graph.nodes if n.name in rec64.saved and n.name in first]
    bad = {lab: set() for lab in dumps}
    for nm in order:
        t64 = rec64.saved[nm]
        r = {}
        t32 = rec32.saved.get(nm)
        if t32 is not None:
            r["v_t32"] = rel(t32.detach().double().numpy(), t64.detach().numpy())
            if t32.grad is not None and t64.grad is not None:
                r["g_t32"] = rel(t32.grad.double().numpy(), t64.grad.numpy())
        for lab, cpp in dumps.items():
            _, v, g = cpp[nm]
            if v is not None and v.size == t64.numel() and v.dtype == np.float32:
                r[f"v_{lab}"] = rel(v.reshape(tuple(t64.shape)).astype(np.float64), t64.detach().numpy())
                if r[f"v_{lab}"] > 0.05 and r.get("v_t32", 0) < 1e-3:
                    bad[lab].add(nm)  # not the FX value (fused / laid out differently in the interpreter)
            else:
                bad[lab].add(nm)
            if g is not None and t64.grad is not None and g.size == t64.numel() and nm not in bad[lab]:
                r[f"g_{lab}"] = rel(g.reshape(tuple(t64.shape)).astype(np.float64), t64.grad.numpy())
                if not np.isfinite(r[f"g_{lab}"]) or (r[f"g_{lab}"] > 0.05 and r.get("g_t32", 1.0) < 1e-3):
                    # a gradient tensor ggml allocated but never computed (unused views): not comparable
                    cpp[nm] = (cpp[nm][0], v, None)
                    del r[f"g_{lab}"]
        rows[nm] = r
    out = {"rows": rows, "order": order, "bad": bad, "local": {}}
    # the output: energy (frame sum, |dE| / sum|E_i|) and, for the LR graphs, charges (max |dq|)
    for lab, cpp in dumps.items():
        out["local"][lab] = local_errors(gm, rec64, cpp, exact, bad[lab])
    fin = None
    for n in gm.graph.nodes:
        if n.op == "output":
            fin = n.all_input_nodes[0].name
    y64 = rec64.saved[fin].detach().numpy()
    y32 = rec32.saved[fin].detach().double().numpy()
    summ = {}
    for lab, cpp in list(dumps.items()) + [("t32", None)]:
        y = y32 if cpp is None else cpp["final_output"][1].reshape(y64.shape).astype(np.float64)
        if kind == "trunk":
            summ[lab] = dict(E=abs(y.sum() - y64.sum()) / np.abs(y64).sum())
        else:
            summ[lab] = dict(E=abs(y[:, 0].sum() - y64[:, 0].sum()) / np.abs(y64[:, 0]).sum(),
                             q=np.abs(y[:, 1] - y64[:, 1]).max())
    out["summary"] = summ
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt")
    ap.add_argument("--kind", choices=("trunk", "lr", "pbc"), required=True)
    ap.add_argument("--dump", action="append", required=True, help="label=path or path (same structures in each)")
    ap.add_argument("--frames", default="0", help="comma list / a-b range of structure indices in the dumps")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--table", action="store_true", help="print the per-node global table of every frame")
    a = ap.parse_args()
    frames = []
    for tok in a.frames.split(","):
        lo, _, hi = tok.partition("-")
        frames += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    paths = {}
    for d in a.dump:
        lab, p = d.split("=", 1) if "=" in d else (Path(d).stem, d)
        paths[lab] = p
    labs = list(paths)
    ep, names = export_program(a.params, a.ckpt, a.kind)
    agg = {lab: {"fwd": {}, "bwd": {}} for lab in labs}
    glob_ = {}
    for f in frames:
        dumps = {lab: read_dump(p, f) for lab, p in paths.items()}
        res = analyse(ep, names, dumps, a.kind, f)
        s = res["summary"]
        print(f"frame {f}: " + "; ".join(f"{lab} " + " ".join(f"{k} {v:.2e}" for k, v in s[lab].items())
                                          for lab in labs + ["t32"]) +
              "; nodes not comparable: " + ", ".join(f"{lab} {len(res['bad'][lab])}" for lab in labs), flush=True)
        for nm in res["order"]:
            for k, v in res["rows"][nm].items():
                glob_.setdefault(nm, {}).setdefault(k, []).append(v)
        if a.table:
            print(f"  {'node':28s} " + " ".join(f"{'v_' + l:>9s}" for l in labs + ["t32"]) + " | " +
                  " ".join(f"{'g_' + l:>9s}" for l in labs + ["t32"]))
            fmt = lambda r, k: f"{r[k]:9.2e}" if k in r else f"{'-':>9s}"
            for nm in res["order"]:
                r = res["rows"][nm]
                print(f"  {nm:28s} " + " ".join(fmt(r, 'v_' + l) for l in labs + ["t32"]) + " | " +
                      " ".join(fmt(r, 'g_' + l) for l in labs + ["t32"]))
        for lab in labs:
            fwd, bwd = res["local"][lab]
            for key, tab in (("fwd", fwd), ("bwd", bwd)):
                for nm, t in tab.items():
                    agg[lab][key].setdefault(nm, []).append(t)
    nf = len(frames)
    print(f"\nGLOBAL error vs fp64 at the graph inputs / outputs (median over {nf} structures):")
    for nm in [x for x in glob_ if x.startswith(("edge_vectors", "cutoff_values", "positions"))] + \
            [x for x in glob_][-3:]:
        print(f"  {nm:28s} " + " ".join(f"{k} {np.median(v):.2e}" for k, v in sorted(glob_[nm].items())))
    for lab in labs:
        print(f"\nLOCAL errors, {lab} (op replayed in fp64 on mlip.cpp's own inputs vs mlip.cpp; fp32 = torch fp32 "
              f"replay on the same inputs; ratio = median over structures of local / max(fp32 replay, 2^-24); "
              f"P = fraction of structures where mlip.cpp's local error > the fp32 replay's):")
        for key, title in (("fwd", "forward"), ("bwd", "backward (dE/d node from all its consumers)")):
            tab = agg[lab][key]
            stats = []
            for nm, ts in tab.items():
                t = np.array([x[:2] for x in ts])
                ratio = np.median(t[:, 0] / np.maximum(t[:, 1], FLOOR))
                stats.append((nm, ratio, np.median(t[:, 0]), np.median(t[:, 1]), float(np.mean(t[:, 0] > t[:, 1])),
                              ts[0][2] if len(ts[0]) > 2 else 0))
            stats.sort(key=lambda x: -x[1])
            print(f"  {title}: {len(tab)} nodes; worst {a.top} by ratio: node, ratio, local med, fp32 med, P")
            for nm, ratio, lm, tm, p, nc in stats[:a.top]:
                print(f"    {nm:28s} {ratio:8.2f} {lm:.2e} {tm:.2e} P {p:.2f}" + (f" ({nc} consumers)" if nc else ""))


if __name__ == "__main__":
    main()
