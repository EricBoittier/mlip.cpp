#!/usr/bin/env python3
"""Iris PET-LR checkpoint (ipsum/PET-LR, after M. Langer) -> mlip.cpp pet-graph GGUF, directly: no metatrain.

    python iris_to_gguf.py PARAMS.npz CKPT_DIR -o OUT.gguf [--no-baseline]

PARAMS.npz: the checkpoint's flat parameter tree (dump_params.py, his marathon.io reader); CKPT_DIR: model/model.yaml,
model/baseline.yaml. Writes the trunk (`params/sr`); the LR branch is added by later phases.

Weights go straight from the iris tree into `iris_torch.IrisTrunk` (a pure-torch transliteration of pet-jax's
trunk, used only to trace the graph with mlip.cpp's FX exporter). Species tables keep iris's Z-indexed layout
(species_map Z -> Z); the conditioning tables are laid out as the C++ side indexes them (charge c at row
c + max_charge, multiplicity m at row m - 1).

Check (bit for bit): every trunk weight is also written under its metatrain state-dict key into a plain dict, and
pet-jax's own metatrain -> Flax converter (`petjax.convert`, c3f216c, imported from his source with jax / marathon
stubbed) must return every `sr` leaf exactly, the three species tables in full (row 0 included); only row 0 of the
spin table (multiplicity 0, never indexed, absent from the C++ layout) is not carried. Each leaf is consumed
exactly once. No metatrain module is imported (asserted).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from petjax_convert import PETJAX_CONVERT, load_petjax_convert  # noqa: E402
from iris_torch import (IrisLR, IrisPETLRGraph, IrisPETLRGraphPBC, IrisQEqApplyGraph, IrisScoresGraph,  # noqa: E402
                        IrisTrunk)

SUPPORTED = {"d_pet", "d_node", "d_head", "d_feedforward", "num_heads", "num_attention_layers", "num_gnn_layers",
             "cutoff", "cutoff_width", "cutoff_width_adaptive", "adaptive_cutoff_method", "num_neighbors_adaptive",
             "attention_temperature", "max_atomic_number", "system_conditioning", "max_charge",
             "max_spin_multiplicity", "conditioning", "final_message_passing", "direct_forces", "direct_stress",
             "inputs"}


# iris.pet.petlr.PETLR field defaults (a2658e3) for keys a model.yaml may leave out
PETLR_DEFAULTS = dict(d_pet=128, d_node=128, d_head=128, d_feedforward=256, num_heads=8, num_attention_layers=1,
                      num_gnn_layers=2, cutoff=5.0, cutoff_width=0.5, cutoff_width_adaptive=0.5,
                      adaptive_cutoff_method="grid", num_neighbors_adaptive=8, attention_temperature=1.0,
                      max_atomic_number=118, direct_forces=False, direct_stress=False, system_conditioning=False,
                      max_charge=10, max_spin_multiplicity=10, conditioning=["charge", "spin_multiplicity"],
                      final_message_passing=True)


def metatrain_key(name):
    """IrisTrunk parameter name -> the metatrain PET state-dict key pet-jax's converter reads."""
    for a, b in (("neighbor_embedder.weight", "edge_embedder.weight"),):
        if name == a:
            return b
    for a, b in (("node_energy_heads.0.", "node_heads.energy.0."), ("edge_energy_heads.0.", "edge_heads.energy.0."),
                 ("node_final_layers.0.", "node_last_layers.energy.0.energy___0."),
                 ("edge_final_layers.0.", "edge_last_layers.energy.0.energy___0.")):
        if name.startswith(a):
            return b + name[len(a):]
    return name


PER_LAYER_HEADS = ("node_energy_heads.", "node_final_layers.", "edge_energy_heads.", "edge_final_layers.")


def per_layer_head(name):
    """(head index, the name with index 0) for a per-layer energy head parameter of IrisTrunk; else None."""
    for pre in PER_LAYER_HEADS:
        if name.startswith(pre):
            l, rest = name[len(pre):].split(".", 1)
            return int(l), pre + "0." + rest
    return None


def fill(model, sr, cfg, conv):
    """Load IrisTrunk from the iris `sr` leaves; return the metatrain-keyed copy for the round trip. With
    per_layer_readout (his iris 38c91c4 PET) head l is the flax subtree energy_head_<l> (the single head's layout
    under that scope) and layer_gate_<l> its gate: those leaves are loaded by name and left out of the metatrain round
    trip (metatrain has no such heads); every leaf is still consumed exactly once."""
    mt, used = {}, {}
    sd = model.state_dict()
    dummy = np.zeros((2, 3), np.float32)
    for name in sd:
        if name.startswith("layer_gate_"):
            v = np.asarray(sr[name]).reshape(tuple(sd[name].shape))
            sd[name] = torch.from_numpy(np.ascontiguousarray(v))
            used.setdefault(name, []).append(name)
            continue
        plh = per_layer_head(name) if model.per_layer else None
        key = metatrain_key(plh[1] if plh else name)
        flax = conv._scope_key(*conv._finalize_key(conv._rename_key(key), dummy)[:1], key)
        transposed = conv._finalize_key(conv._rename_key(key), dummy)[1].shape != dummy.shape
        if plh:
            if not flax.startswith("energy_head."):
                raise KeyError(f"{name}: per-layer head maps to {flax}")
            flax = f"energy_head_{plh[0]}." + flax[len("energy_head."):]
        if flax not in sr:
            raise KeyError(f"{name} ({key}) -> {flax}: not in the iris trunk")
        v = np.asarray(sr[flax])
        if flax.endswith("charge_embedding.embedding"):
            v = np.roll(v, cfg["max_charge"], axis=0)  # value-indexed -> row c + max_charge
        elif flax.endswith("spin_multiplicity_embedding.embedding"):
            v = v[1:]  # row m -> row m - 1
        if transposed:
            v = v.T
        if tuple(v.shape) != tuple(sd[name].shape):
            raise ValueError(f"{name} <- {flax}: {v.shape} vs {tuple(sd[name].shape)}")
        sd[name] = torch.from_numpy(np.ascontiguousarray(v))
        if not plh:
            mt[key] = sd[name]
        used.setdefault(flax, []).append(name)
    model.load_state_dict(sd)
    double = {k: v for k, v in used.items() if len(v) > 1}
    missing = sorted(set(sr) - set(used) - {"energy_scale"})
    if double or missing:
        raise ValueError(f"leaves used twice {double} / not consumed {missing}")
    return mt


def round_trip(mt, sr, cfg, conv):
    back = conv._convert_state_dict(mt)
    n_rows = cfg["max_atomic_number"] + 1
    conv._scatter_species_embeddings(back, list(range(n_rows)), n_rows)  # Z-indexed tables: rows map to themselves
    if cfg["system_conditioning"]:
        conv._layout_conditioning_tables(back, cfg["max_charge"])
    for k, v in back.items():
        ref, v = np.asarray(sr[k]), np.asarray(v)
        if k.endswith("spin_multiplicity_embedding.embedding"):
            ref, v = ref[1:], v[1:]
        if v.shape != ref.shape or not np.array_equal(v, ref):
            raise ValueError(f"round trip differs at {k}")
    lost = {k for k in set(sr) - set(back) - {"energy_scale"}
            if not (k.startswith("energy_head_") or k.startswith("layer_gate_"))}  # per-layer heads: loaded by name
    if lost:
        raise ValueError(f"round trip lost {sorted(lost)}")
    return len(back)


def export_checked(export_torch_model, *a, **kw):
    """export_torch_model, failing on any op the FX converter does not map (it otherwise warns and passes the
    node's first input through, or drops it: torch.stack vanished from a concat that way)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = export_torch_model(*a, **kw)
    text = buf.getvalue()
    # symbolic-size arithmetic (python add / mul on SymInts, sym_size) is shape bookkeeping, not graph maths
    benign = ("<built-in function add>", "<built-in function mul>", "<built-in function floordiv>",
              "aten.sym_size.int")
    bad = [l for l in text.splitlines() if ("Unhandled" in l or "Warning" in l) and not l.strip().endswith(benign)]
    if bad:
        raise ValueError("FX export: " + "; ".join(sorted(set(bad))[:8]))
    return out


def to_sample(a):
    """His training pipeline's to_sample (CKPT_DIR/config.yaml, else --config, else {})."""
    path = Path(f"{a.ckpt}/config.yaml")
    if not path.exists() and a.config:
        path = Path(a.config)
    return yaml.safe_load(open(path))["training_pipeline"]["to_sample"] if path.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--n-atoms", type=int, default=7); ap.add_argument("--max-neighbors", type=int, default=11)
    ap.add_argument("--dump-json", help="also write the graph JSON(s) to this path prefix (inspection)")
    ap.add_argument("--config", help="the training config.yaml (his to_sample: num_k, max_neighbors) when CKPT_DIR has "
                    "none; the seeded presets' is scripts/pet_lr_port/presets/seeded_config.yaml")
    ap.add_argument("--lr", action="store_true",
                    help="trunk + LR branch as one graph (IrisPETLRGraph) plus the scores graph of the two-pass mu")
    a = ap.parse_args()

    (name, cfg), = yaml.safe_load(open(f"{a.ckpt}/model/model.yaml")).items()
    cfg = {**PETLR_DEFAULTS, **cfg}
    if cfg.get("fixed_neighbors") is not None:  # his older fixed-neighbour mode (KnnToSample): the hard cutoff
        cfg["num_neighbors_adaptive"] = None
    # his les-water backbones (NoFinalCombineBackbone / LayeredBackbone; iris_torch.backbone_kind raises on others)
    if cfg.get("charge_layer") is not None and not cfg.get("lr"):
        cfg["charge_layer"] = None  # (charge_layer only picks the LR branch's node features: no effect trunk-only)
    if cfg.get("direct_forces") or cfg.get("direct_stress"):
        raise ValueError("direct heads: not yet supported")
    conditioning = list(cfg.get("conditioning", ["charge", "spin_multiplicity"]))
    if set(conditioning) - {"charge", "spin_multiplicity"}:
        raise ValueError(f"unknown conditioning keys {conditioning}")
    flat = dict(np.load(a.params))
    sr = {k[len("params/sr/"):].replace("/", "."): v for k, v in flat.items() if k.startswith("params/sr/")}
    conv = load_petjax_convert(PETJAX_CONVERT)
    model = IrisTrunk(cfg).eval()
    mt = fill(model, sr, cfg, conv)
    n = round_trip(mt, sr, cfg, conv)
    n_sr = sum(np.asarray(v).size for v in sr.values())
    print(f"{name}: {len(sr)} trunk leaves ({n_sr} parameters incl. energy_scale), round trip bit-identical on {n}")

    from export_pytorch.export_pet_gguf import GGUFWriter, add_conditioning_inputs, example_export_args
    from export_pytorch.fx_converter import export_torch_model

    inputs, names, shapes = example_export_args(a.n_atoms, a.max_neighbors, cfg["cutoff"], forces=True)
    if model.conditioning:
        inputs, names, shapes = add_conditioning_inputs(inputs, names, shapes, a.n_atoms)
    dtypes = {"species": "i32", "neighbor_species": "i32", "reverse_neighbor_index": "i32",
              "charge_index": "i32", "spin_index": "i32", "neighbor_index": "i32"}
    energy_scale = float(np.asarray(sr["energy_scale"]).reshape(()))
    graph_scores = graph_pbc = None
    graph_qeq = {}  # QEq: the host cg's operator graphs (open / pbc)
    if a.lr:
        if not cfg.get("lr"):
            raise ValueError("--lr: the checkpoint has no LR branch (lr: false)")
        from check_lr64 import load_lr
        lr = IrisLR(cfg).eval()
        load_lr(lr, flat)
        qeq = lr.ref == "qeq"
        if lr.fermi or qeq:  # QEq: the scores graph returns chi
            sgraph, sweights = export_checked(export_torch_model, IrisScoresGraph(model, lr).eval(), inputs, output_path=None,
                                                  input_names=names, input_dtypes=dtypes, dynamic_shapes=shapes,
                                                  strict=False)
            graph_scores = sgraph
        export = IrisPETLRGraph(model, lr, energy_scale).eval()
        n = a.n_atoms
        atom = shapes[0][0]
        nk = shapes[1]
        g = torch.Generator().manual_seed(0)
        extra = (torch.zeros(n, a.max_neighbors, dtype=torch.long), torch.randn(n, 3, generator=g),
                 torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(n), torch.zeros(n), torch.zeros(n),
                 torch.zeros(3))
        trunk_inputs, trunk_shapes = inputs, shapes
        inputs = inputs + extra
        names = names + ["neighbor_index", "positions", "total_charge", "mu_bisect", "gapped", "m_round", "top",
                         "bottom", "dipole_origin"]
        shapes = shapes[:3] + (shapes[3] + (dict(nk), {0: atom}, None, None, None, {0: atom}, {0: atom}, {0: atom},
                                            None),)
        if qeq:  # the host cg's solution z (a gradient input) and the cotangent solution lam
            inputs = inputs + (torch.zeros(n), torch.zeros(n))
            names = names + ["qeq_z", "qeq_lambda"]
            shapes = shapes[:3] + (shapes[3] + ({0: atom}, {0: atom}),)
        model_export = export
        # the periodic graph (his jax-pme Ewald), same parameters
        kl, K = 13, 17
        n_lr, n_kvec = torch.export.Dim("n_lr", min=1, max=100000), torch.export.Dim("n_kvec", min=1, max=100000)
        ew_in = (torch.zeros(n, kl, dtype=torch.long), torch.randn(n, kl, 3, generator=g), torch.ones(n, kl),
                 torch.randn(K, 3, generator=g), torch.ones(1) * 1000.0, torch.ones(1) * 2.0, torch.ones(1) * 8.0,
                 torch.zeros(1), torch.tensor([0.0, 0.0, 1.0]), torch.ones(1))
        ew_shapes = ({0: atom, 1: n_lr}, {0: atom, 1: n_lr}, {0: atom, 1: n_lr}, {0: n_kvec}, None, None, None, None,
                     None, None)
        export_pbc = IrisPETLRGraphPBC(model, lr, energy_scale).eval()
        pdtypes = dict(dtypes, lr_index="i32")
        graph_pbc, wpbc = export_checked(
            export_torch_model, export_pbc, inputs + ew_in, output_path=None,
            input_names=names + list(IrisPETLRGraphPBC.EWALD), input_dtypes=pdtypes,
            dynamic_shapes=shapes[:3] + (shapes[3] + ew_shapes,), strict=False)
        if qeq:  # the operator graphs: no trunk (species, geometry, x)
            ti, ts = trunk_inputs, trunk_shapes
            ap_in = (ti[0], ti[2], ti[3], ti[4], extra[0], extra[1], torch.randn(n, generator=g))
            ap_shapes = (ts[0], ts[2], ts[3][0], ts[3][1], dict(nk), {0: atom}, {0: atom})
            for key, periodic in (("graph_qeq_apply.json", False), ("graph_qeq_apply_pbc.json", True)):
                gq, wq = export_checked(
                    export_torch_model, IrisQEqApplyGraph(lr, periodic).eval(), ap_in + (ew_in if periodic else ()),
                    output_path=None, input_names=list(IrisQEqApplyGraph.INPUTS) + (
                        list(IrisPETLRGraphPBC.EWALD) if periodic else []), input_dtypes=pdtypes,
                    dynamic_shapes=ap_shapes + ((ew_shapes,) if periodic else ()), strict=False)
                graph_qeq[key] = (gq, wq)
    else:
        model_export = model
    graph, weights = export_checked(
        export_torch_model, model_export, inputs, output_path=None, input_names=names, input_dtypes=dtypes,
        dynamic_shapes=shapes, strict=False)
    # a symbolic size in arithmetic (x / n with n = species.shape[0]) is traced as an op ON the size's source
    # tensor ("DIV [x, input:species]"): integer / mask inputs must never be arithmetic operands
    gd = graph.to_dict()
    int_inputs = {f"input:{i['name']}" for i in gd["inputs"] if i.get("dtype") in ("i32", "bool")}
    for g_ in [gd] + ([graph_scores.to_dict()] if graph_scores is not None else []):
        for nd in g_["nodes"]:
            if nd["op"] in ("ADD", "SUB", "MUL", "DIV", "POW", "MATMUL") and int_inputs & set(nd["inputs"]):
                raise ValueError(f"node {nd['id']} {nd['op']} {nd['inputs']}: an integer input as an arithmetic operand "
                                 "(a symbolic size?)")
    # the comb_norms' LayerNorm eps as traced: his les-water backbones build them with flax's default 1e-6, pet-jax's
    # Backbone with 1e-5 (invisible in fp32 parity: asserted here, and the torch fp64 check flips with it)
    want_eps = 1e-6 if model.backbone in ("nofinal", "layered") else 1e-5
    for g_ in [graph.to_dict()] + ([graph_pbc.to_dict()] if graph_pbc is not None else []):
        for nd in g_["nodes"]:
            if nd["op"] == "LAYER_NORM" and any("combination_norms" in x for x in nd["inputs"]):
                if abs(nd.get("params", {}).get("eps", -1) - want_eps) > 1e-12:
                    raise ValueError(f"comb_norms LayerNorm {nd['name']}: eps {nd.get('params')} != {want_eps}")
    # the converter writes graph weight refs lower-cased (his flax names MLP_0 / LayerNorm_0 keep their case in
    # the state dict): store the tensors under the names the graph reads
    refs = {x[7:] for nd in graph.to_dict()["nodes"] for x in nd["inputs"] if x.startswith("weight:")}
    lowered = {k.lower(): k for k in weights}
    if len(lowered) != len(weights):
        raise ValueError("weight names collide when lower-cased")
    weights = {(k.lower() if k not in refs and k.lower() in refs else k): v for k, v in weights.items()}
    unresolved = refs - set(weights)
    if unresolved:
        raise ValueError(f"graph reads weights that are not written: {sorted(unresolved)[:5]}")
    if graph_pbc is not None:
        prefs = {x[7:] for nd in graph_pbc.to_dict()["nodes"] for x in nd["inputs"] if x.startswith("weight:")}
        missing = prefs - set(weights)
        if missing:
            raise ValueError(f"periodic graph reads weights the open graph does not write: {sorted(missing)[:5]}")
        for nd in graph_pbc.to_dict()["nodes"]:  # integer inputs never arithmetic operands (as above)
            if nd["op"] in ("ADD", "SUB", "MUL", "DIV", "POW", "MATMUL") and \
                    {"input:lr_index", "input:species", "input:neighbor_index"} & set(nd["inputs"]):
                raise ValueError(f"periodic graph node {nd['id']} {nd['op']} {nd['inputs']}: integer operand")
    for key, (gq, wq) in graph_qeq.items():  # the operator graphs read the full graph's weights (lr.*)
        qrefs = {x[7:] for nd in gq.to_dict()["nodes"] for x in nd["inputs"] if x.startswith("weight:")}
        if qrefs - set(weights):
            raise ValueError(f"{key} reads weights the full graph does not write: {sorted(qrefs - set(weights))[:5]}")
        for nd in gq.to_dict()["nodes"]:
            if nd["op"] in ("ADD", "SUB", "MUL", "DIV", "POW", "MATMUL") and \
                    {"input:lr_index", "input:species", "input:neighbor_index"} & set(nd["inputs"]):
                raise ValueError(f"{key} node {nd['id']} {nd['op']} {nd['inputs']}: integer operand")
    if graph_scores is not None:
        sweights = {(k.lower() if k.lower() in weights and k not in weights else k): v for k, v in sweights.items()}
        missing = set(sweights) - set(weights)
        if missing:
            raise ValueError(f"scores graph weights not in the full graph: {sorted(missing)[:5]}")
    pname = lambda p: p.replace(".", "_") if p.replace(".", "_") in weights else p.replace(".", "_").lower()
    param_names = [pname(p) for p, _ in model_export.named_parameters() if pname(p) in weights]
    baseline = {}
    if not a.no_baseline:
        b = yaml.safe_load(open(f"{a.ckpt}/model/baseline.yaml")) or {}
        baseline = {int(z): float(v) for z, v in (b.get("elemental") or {}).items()}
    nna = cfg.get("num_neighbors_adaptive")

    if a.dump_json:
        json.dump(graph.to_dict(), open(a.dump_json + ".graph.json", "w"))
        if graph_scores is not None:
            json.dump(graph_scores.to_dict(), open(a.dump_json + ".scores.json", "w"))
        if graph_pbc is not None:
            json.dump(graph_pbc.to_dict(), open(a.dump_json + ".pbc.json", "w"))
    w = GGUFWriter()
    w.add_string("general.architecture", "pet-graph")
    w.add_string("general.name", f"iris-petlr:{Path(a.ckpt).resolve()}")
    w.add_string("general.version", "1.0.2")
    w.add_string("pet.source", "iris PET-LR (ipsum/PET-LR, after M. Langer): scripts/pet_lr_port/iris_to_gguf.py")
    w.add_float32("pet.cutoff", float(cfg["cutoff"]))
    w.add_float32("pet.cutoff_width", float(cfg["cutoff_width"]))
    w.add_int32("pet.d_pet", int(cfg["d_pet"]))
    w.add_int32("pet.num_heads", int(cfg["num_heads"]))
    # with --lr the scale is a graph parameter (trunk energy only, as his PETLR): the C++ side applies 1
    w.add_float32("pet.energy_scale", 1.0 if a.lr else energy_scale)
    if a.lr:
        w.add_int32("pet.iris_lr", 1)
        w.add_int32("pet.iris_lr.n_out", IrisPETLRGraph.N_OUT)
        w.add_string("pet.iris_lr.reference", lr.ref)
        w.add_int32("pet.iris_lr.charge_state_min", int(lr.m_min))
        w.add_int32("pet.iris_lr.charge_state_max", int(lr.m_max))
        w.add_float32("pet.iris_lr.charge_temperature", float(lr.T))
        w.add_int32("pet.iris_lr.fermi_iterations", 40)
        w.add_float32("pet.iris_lr.fermi_eps", 1e-6)
        w.add_string("pet.iris_lr.dipole_origin", "center_of_mass")
        if lr.conserving:  # one component per structure (his sample prep raises on more without fragment labels)
            w.add_int32("pet.iris_lr.conserving_charges", 1)
        if lr.lrt:  # his ToSample refuses slabs with the long-range transfer; mlip.cpp too
            w.add_int32("pet.iris_lr.long_range_transfer", 1)
        if graph_scores is not None:
            w.add_string("graph_scores.json", json.dumps(graph_scores.to_dict()))
        if lr.ref == "qeq":  # his qeq_reference cg settings (maxiter 20, tol 1e-5 relative, atol 0)
            w.add_int32("pet.iris_lr.qeq_maxiter", int(lr.qeq_maxiter))
            w.add_float32("pet.iris_lr.qeq_tol", float(lr.qeq_tol))
            for key, (gq, _) in graph_qeq.items():
                w.add_string(key, json.dumps(gq.to_dict()))
        if graph_pbc is not None:  # his calculator's Ewald settings (Calculator.from_checkpoint: the sampler's)
            w.add_string("graph_pbc.json", json.dumps(graph_pbc.to_dict()))
            ts = to_sample(a)
            if not int(ts.get("num_k") or 0):  # (the C++ side would raise on every periodic structure)
                raise ValueError(f"{a.ckpt}: no to_sample.num_k (no config.yaml there and no --config): the periodic graph "
                                 "would be written with pet.iris_lr.num_k = 0; pass --config (seeded presets: "
                                 "presets/seeded_config.yaml)")
            w.add_int32("pet.iris_lr.num_k", int(ts.get("num_k") or 0))
            w.add_float32("pet.iris_lr.smearing", float(ts.get("smearing") or 0.0))  # 0: 2 wavelength
            w.add_int32("pet.iris_lr.halfspace", int(ts.get("halfspace", True)))
            w.add_float32("pet.iris_lr.slab_margin", 2.0)
    # his calculator raises when an atom has more than max_neighbors neighbours within the cutoff (training pipeline
    # to_sample.max_neighbors); mlip.cpp raises there too for iris GGUFs (0: no config.yaml, no check)
    ts_all = to_sample(a)
    # (his older fixed-neighbour mode: KnnToSample keeps the nearest fixed_neighbors, so that is the overflow width)
    w.add_int32("pet.iris.max_neighbors", int(cfg.get("fixed_neighbors") or ts_all.get("max_neighbors") or 0))
    # his XLA keeps libm-accurate transcendentals and IEEE division on the GPU: the CUDA kernels of iris graphs run
    # the precise variants (graph_model.cpp mark_precise_math; the PET-LR graphs get them regardless)
    w.add_int32("pet.precise_math", 1)
    w.add_string("pet.cutoff_function", "bump")
    w.add_int32("pet.forces_mode", 1)
    w.add_int32("pet.nc_forces", 0)
    if model.conditioning:
        w.add_int32("pet.system_conditioning", 1)
        w.add_int32("pet.max_charge", model.max_charge)
        w.add_int32("pet.max_spin_multiplicity", model.max_spin_multiplicity)
        # iris conditions only on the keys listed (absent charge = charge 0): the C++ side pins the rest
        w.add_string("pet.conditioning_keys", ",".join(conditioning))
    w.add_float32("pet.num_neighbors_adaptive", float(nna or 0.0))
    w.add_float32("pet.cutoff_width_adaptive", float(cfg.get("cutoff_width_adaptive", cfg["cutoff_width"])))
    w.add_string("pet.adaptive_cutoff_method", str(cfg.get("adaptive_cutoff_method", "grid")).lower())
    species_map = []
    for z in range(1, cfg["max_atomic_number"] + 1):
        species_map += [z, z]  # Z-indexed tables (row 0 unused)
    w.add_array_int32("pet.species_map", species_map)
    if baseline:
        w.add_array_int32("pet.composition_keys", list(baseline))
        w.add_array_float32("pet.composition_values", list(baseline.values()))
    w.add_string("graph.json", json.dumps(graph.to_dict()))
    w.add_string("train.param_names", json.dumps(param_names))
    w.add_string("train.state_dict_names",
                 json.dumps({pname(p): metatrain_key(p) for p, _ in model_export.named_parameters()}))
    w.add_string("graph.weight_shapes", json.dumps({k: list(t.shape) for k, t in weights.items()}))
    for k, t in weights.items():
        w.add_tensor(k, t, transpose_2d=False)
    w.write(a.output)
    leaked = sorted(m for m in sys.modules if m == "metatrain" or m.startswith("metatrain."))
    assert not leaked, f"metatrain was imported: {leaked[:3]}"
    print(f"wrote {a.output}: {len(graph.nodes)} nodes, {len(weights)} weights, energy_scale {energy_scale!r}, "
          f"final_message_passing={model.final_mp}, layers={len(model.gnn_layers)}, nna={nna}, "
          f"conditioning={conditioning if model.conditioning else None}; metatrain not imported")


if __name__ == "__main__":
    main()
