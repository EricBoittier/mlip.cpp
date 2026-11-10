#!/usr/bin/env python3
"""Iris PET-LR checkpoint (ipsum/PET-LR, after M. Langer) -> mlip.cpp pet-graph GGUF: the PET trunk.

    python convert_iris.py PARAMS.npz CKPT_DIR -o OUT.gguf [--no-baseline] [--keep-pt OUT.pt]

PARAMS.npz is the flat parameter tree of the checkpoint (dump_params.py, read with his own
marathon.io.read_msgpack); CKPT_DIR holds model/model.yaml and model/baseline.yaml.

The trunk (`params/sr`) is pet-jax's UPET, a reimplementation of metatrain's PET. It is converted by
inverting pet-jax's own metatrain -> Flax converter (`petjax.convert`, c3f216c): every metatrain state-dict
key of a PET built with the checkpoint's hypers is pushed through pet-jax's key-renaming functions
(`_rename_key`, `_finalize_key`, `_scope_key`, imported from his source with jax / marathon stubbed out), and
the Flax leaf it names fills it (2-D kernels transposed back, species tables gathered from Z rows to the
species order, the charge table un-rolled and the spin table's unused row 0 dropped). The result is checked
by the forward round trip: his `_convert_state_dict` + `_scatter_species_embeddings` +
`_layout_conditioning_tables` on the filled state dict must give back every `sr` leaf bit for bit (row 0 of
the three species tables excepted: Z = 0 never occurs and metatrain has no row for it). Every `sr` leaf must
be consumed exactly once and every metatrain parameter filled exactly once.

The filled metatrain PET is then exported with mlip.cpp's FX exporter (export_pytorch.export_pet_gguf), and the
GGUF metadata set from the checkpoint: energy_scale (a trained fp32 parameter in iris), the composition
baseline from baseline.yaml (unless --no-baseline), species map Z = 1..max_atomic_number, conditioning.

Not carried by this script (phase 1): the long-range branch (`params/lr`), which is reported and left out.
Conditioning: iris PETLR conditions the trunk on the keys in `conditioning` only (the omol-4m runs:
spin_multiplicity); an absent charge key means pet-jax's default charge 0, so evaluate the trunk at charge 0.
"""
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

PETJAX_CONVERT = Path.home() / "data/plato-lr-scratch/pet_lr_port/src/petjax/convert.py"


def load_petjax_convert(path):
    """pet-jax's convert.py with its JAX / marathon imports stubbed (only numpy is needed by the renames)."""
    jnp = types.ModuleType("jax.numpy")
    jnp.array = np.array
    jnp.asarray = np.asarray
    jax = types.ModuleType("jax")
    jax.numpy = jnp
    mio = types.ModuleType("marathon.io")
    mio.write_msgpack = mio.write_yaml = mio.read_msgpack = mio.read_yaml = None
    marathon = types.ModuleType("marathon")
    marathon.io = mio
    saved = {k: sys.modules.get(k) for k in ("jax", "jax.numpy", "marathon", "marathon.io")}
    sys.modules.update({"jax": jax, "jax.numpy": jnp, "marathon": marathon, "marathon.io": mio})
    try:
        spec = importlib.util.spec_from_file_location("petjax_convert", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def build_metatrain_pet(cfg, atomic_types):
    from metatrain.pet import PET
    from metatrain.utils.architectures import get_default_hypers
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_energy_target_info

    hypers = get_default_hypers("pet")["model"]
    hypers.update(dict(
        cutoff=cfg["cutoff"], cutoff_width=cfg["cutoff_width"],
        cutoff_width_adaptive=cfg["cutoff_width_adaptive"],
        adaptive_cutoff_method=cfg["adaptive_cutoff_method"],
        num_neighbors_adaptive=cfg["num_neighbors_adaptive"],
        d_pet=cfg["d_pet"], d_head=cfg["d_head"], d_node=cfg["d_node"],
        d_feedforward=cfg["d_feedforward"], num_heads=cfg["num_heads"],
        num_attention_layers=cfg["num_attention_layers"], num_gnn_layers=cfg["num_gnn_layers"],
        attention_temperature=cfg["attention_temperature"],
        system_conditioning=cfg["system_conditioning"], max_charge=cfg["max_charge"],
        max_spin_multiplicity=cfg["max_spin_multiplicity"],
        normalization="RMSNorm", activation="SwiGLU", transformer_type="PreLN",
        featurizer_type="feedforward", cutoff_function="Bump", zbl=False,
    ))
    targets = {"energy": get_energy_target_info("energy", {"unit": "eV"}, add_position_gradients=True)}
    info = DatasetInfo(length_unit="angstrom", atomic_types=list(atomic_types), targets=targets)
    model = PET(hypers, info)
    model.eval()
    return model, hypers


def flax_key(conv, key):
    """metatrain state-dict key -> (dotted Flax key of pet-jax's tree, transposed?) via pet-jax's renames."""
    k = conv._strip_backend_prefix({key: torch.zeros(1)})
    (stripped,) = k.keys()
    new = conv._rename_key(stripped)
    dummy = np.zeros((2, 3), dtype=np.float32)
    final, value = conv._finalize_key(new, dummy)
    return conv._scope_key(final, stripped), value.shape != dummy.shape


def convert(flat, cfg, conv):
    sr = {k[len("params/sr/"):].replace("/", "."): v for k, v in flat.items() if k.startswith("params/sr/")}
    Zmax = cfg["max_atomic_number"]
    atomic_types = list(range(1, Zmax + 1))
    model, hypers = build_metatrain_pet(cfg, atomic_types)
    pet = model
    sd = model.state_dict()
    params = dict(model.named_parameters())
    used, filled = {}, {}
    rows = np.asarray(atomic_types)
    for key in sd:
        if key.startswith(conv._SKIP_PREFIXES) or key.removeprefix("backend.").startswith(conv._SKIP_PREFIXES):
            continue
        if not isinstance(sd[key], torch.Tensor) or not sd[key].is_floating_point():
            continue
        fk_tree, transposed = flax_key(conv, key)
        if fk_tree not in sr:
            raise KeyError(f"metatrain {key} -> {fk_tree}: no such leaf in the iris trunk")
        v = np.asarray(sr[fk_tree])
        if fk_tree.endswith(".embedding") and "system_conditioning" not in fk_tree:
            v = v[rows]  # Z rows -> species order
        elif fk_tree.endswith("charge_embedding.embedding"):
            v = np.roll(v, cfg["max_charge"], axis=0)  # value-indexed -> c + max_charge
        elif fk_tree.endswith("spin_multiplicity_embedding.embedding"):
            v = v[1:]  # drop the unused row 0: m at row m - 1
        if transposed:
            v = v.T
        if tuple(v.shape) != tuple(sd[key].shape):
            raise ValueError(f"{key} <- {fk_tree}: shape {v.shape} vs {tuple(sd[key].shape)}")
        if key.removeprefix("backend.") in filled:
            # v16 aliases (conditioning at top level and under backend.): one leaf, both names
            assert np.array_equal(filled[key.removeprefix("backend.")], v)
        filled[key.removeprefix("backend.")] = v
        used.setdefault(fk_tree, []).append(key)
        sd[key] = torch.from_numpy(np.ascontiguousarray(v)).to(sd[key].dtype)
    # every Flax leaf consumed (energy_scale goes to the GGUF metadata), every parameter filled
    missing = sorted(set(sr) - set(used) - {"energy_scale"})
    if missing:
        raise ValueError(f"iris trunk leaves not consumed: {missing}")
    model.load_state_dict(sd)
    unfilled = [n for n, p in params.items()
                if n.removeprefix("backend.") not in filled and not n.startswith(("scaler.", "additive_models."))]
    if unfilled:
        raise ValueError(f"metatrain parameters not filled: {unfilled}")
    return model, hypers, sr, used


def round_trip(model, cfg, sr, conv):
    """pet-jax's forward converter on the filled state dict must return the iris leaves bit for bit."""
    state = conv._strip_backend_prefix(model.state_dict())
    if not cfg["system_conditioning"]:
        state = {k: v for k, v in state.items() if not k.startswith("system_conditioning.")}
    back = conv._convert_state_dict(state)
    n_rows = cfg["max_atomic_number"] + 1
    conv._scatter_species_embeddings(back, list(range(1, n_rows)), n_rows)
    if cfg["system_conditioning"]:
        conv._layout_conditioning_tables(back, cfg["max_charge"])
    worst, checked = 0, 0
    for k, v in back.items():
        ref = np.asarray(sr[k])
        v = np.asarray(v)
        if k.endswith(".embedding") and "system_conditioning" not in k or k.endswith("spin_multiplicity_embedding.embedding"):
            ref, v = ref[1:], v[1:]  # row 0: Z = 0 / multiplicity 0, never indexed
        if v.shape != ref.shape or not np.array_equal(v, ref):
            raise ValueError(f"round trip differs at {k}: max |d| = {np.abs(v - ref).max()}")
        checked += 1
    extra = set(sr) - set(back) - {"energy_scale"}
    if extra:
        raise ValueError(f"round trip lost leaves: {sorted(extra)}")
    return checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--keep-pt")
    ap.add_argument("--n-atoms", type=int, default=7); ap.add_argument("--max-neighbors", type=int, default=11)
    a = ap.parse_args()

    conv = load_petjax_convert(PETJAX_CONVERT)
    flat = dict(np.load(a.params))
    (name, cfg), = yaml.safe_load(open(f"{a.ckpt}/model/model.yaml")).items()
    assert name.endswith("PETLR") or name.endswith("PET"), name
    assert not cfg.get("direct_forces") and not cfg.get("direct_stress")
    if not cfg.get("final_message_passing", True):
        raise ValueError("final_message_passing=False has no metatrain equivalent (not exportable)")
    if cfg["system_conditioning"] and set(cfg.get("conditioning", ["charge", "spin_multiplicity"])) - {
            "charge", "spin_multiplicity"}:
        raise ValueError(f"unknown conditioning keys {cfg['conditioning']}")
    model, hypers, sr, used = convert(flat, cfg, conv)
    n = round_trip(model, cfg, sr, conv)
    n_sr = sum(np.asarray(v).size for v in sr.values())
    n_lr = sum(v.size for k, v in flat.items() if k.startswith("params/lr/"))
    print(f"converted: {len(used)} trunk leaves ({n_sr} parameters, energy_scale incl.), round trip "
          f"bit-identical on {n} leaves; LR branch {n_lr} parameters NOT converted (phase 2)")
    if a.keep_pt:
        torch.save({"model": model, "hypers": hypers}, a.keep_pt)

    energy_scale = float(np.asarray(sr["energy_scale"]).reshape(()))
    baseline = {}
    if not a.no_baseline:
        b = yaml.safe_load(open(f"{a.ckpt}/model/baseline.yaml")) or {}
        baseline = {int(z): float(v) for z, v in (b.get("elemental") or {}).items()}
    export_gguf(model, cfg, energy_scale, baseline, a)


def export_gguf(pet, cfg, energy_scale, baseline, a):
    import json

    from export_pytorch.export_pet_full import PETFullModel, _unwrap_to_pet, get_model_params
    from export_pytorch.export_pet_gguf import (
        GGUFWriter, add_conditioning_inputs, example_export_args, state_dict_names)
    from export_pytorch.fx_converter import export_torch_model

    pet = _unwrap_to_pet(pet)  # metatrain's PET -> its PETBackend (gnn_layers), as load_pet_model does
    params = get_model_params(pet)
    wrapper = PETFullModel(pet, n_atoms=a.n_atoms, max_neighbors=a.max_neighbors, d_pet=params["d_pet"],
                           forces=True, cutoff=params["cutoff"], cutoff_width=params["cutoff_width"],
                           cutoff_function=params["cutoff_function"], nc_forces=False)
    wrapper.eval()
    assert not wrapper.long_range and wrapper.plato_lr is None
    inputs, names, shapes = example_export_args(a.n_atoms, a.max_neighbors, params["cutoff"], forces=True)
    if wrapper.conditioning:
        inputs, names, shapes = add_conditioning_inputs(inputs, names, shapes, a.n_atoms)
    graph, weights = export_torch_model(
        wrapper, inputs, output_path=None, input_names=names,
        input_dtypes={"species": "i32", "neighbor_species": "i32", "reverse_neighbor_index": "i32",
                      "charge_index": "i32", "spin_index": "i32"},
        dynamic_shapes=shapes, strict=False)
    named = list(wrapper.named_parameters(remove_duplicate=False))
    param_names = [n.replace(".", "_") for n, _ in named if n.replace(".", "_") in weights]

    w = GGUFWriter()
    w.add_string("general.architecture", "pet-graph")
    w.add_string("general.name", f"iris-petlr-trunk:{a.ckpt}")
    w.add_string("general.version", "1.0.2")
    w.add_string("pet.source", "iris PET-LR trunk (ipsum/PET-LR, after M. Langer), scripts/pet_lr_port/convert_iris.py")
    w.add_float32("pet.cutoff", params["cutoff"])
    w.add_float32("pet.cutoff_width", params["cutoff_width"])
    w.add_int32("pet.d_pet", params["d_pet"])
    w.add_int32("pet.num_heads", int(pet.gnn_layers[0].trans.layers[0].attention.num_heads))
    w.add_float32("pet.energy_scale", energy_scale)
    w.add_string("pet.cutoff_function", params["cutoff_function"])
    w.add_int32("pet.forces_mode", 1)
    w.add_int32("pet.nc_forces", 0)
    if wrapper.conditioning:
        w.add_int32("pet.system_conditioning", 1)
        w.add_int32("pet.max_charge", wrapper.max_charge)
        w.add_int32("pet.max_spin_multiplicity", wrapper.max_spin_multiplicity)
    w.add_float32("pet.num_neighbors_adaptive", float(cfg["num_neighbors_adaptive"] or 0.0))
    w.add_float32("pet.cutoff_width_adaptive", float(cfg["cutoff_width_adaptive"]))
    w.add_string("pet.adaptive_cutoff_method", str(cfg["adaptive_cutoff_method"]).lower())
    species_map = []
    for idx, Z in enumerate(range(1, cfg["max_atomic_number"] + 1)):
        species_map += [Z, idx]
    w.add_array_int32("pet.species_map", species_map)
    if baseline:
        w.add_array_int32("pet.composition_keys", list(baseline))
        w.add_array_float32("pet.composition_values", list(baseline.values()))
    w.add_string("graph.json", json.dumps(graph.to_dict()))
    w.add_string("train.param_names", json.dumps(param_names))
    w.add_string("train.state_dict_names", json.dumps(state_dict_names(wrapper, pet)))
    w.add_string("graph.weight_shapes", json.dumps({n: list(t.shape) for n, t in weights.items()}))
    for n, t in weights.items():
        w.add_tensor(n, t, transpose_2d=False)
    w.write(a.output)
    print(f"wrote {a.output}: {len(graph.nodes)} nodes, {len(weights)} weights, energy_scale {energy_scale!r}, "
          f"{len(baseline)} baseline species")


if __name__ == "__main__":
    main()
