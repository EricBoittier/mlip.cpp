#!/usr/bin/env python3
"""PET with Plato-PET-LR components (metatrain ``plato_lr`` hypers) for the mlip.cpp
parity and training checks. metatrain builds and converts the model
(``metatrain.pet.convert_plato_lr``); nothing is trained here.

    # a small fresh PET (metatrain init), all components, gates randomised
    python make_pet_plato_lr.py -o small.pt --small feedforward --gates random
    # a pretrained checkpoint (an LLPR wrapper is unwrapped), gates zero
    python make_pet_plato_lr.py -o omol.pt --ckpt pet-omol-s-v1.0.0.ckpt --gates zero
    # ... with charge / spin conditioning added (behind its zero output gate)
    python make_pet_plato_lr.py -o mols.pt --ckpt pet-mols-s-v1.1.0.ckpt --add-conditioning

Writes OUT ({"model": PET}, what export_pet_gguf.py --model reads), OUT with .ckpt
(model.get_checkpoint()) and OUT with .json (the plato_lr hypers, the new tensors, the
gate values). --gates random sets every gate (rope_gate, edge_gate, mesh_gate, les_gate)
to nonzero values (times --gate-scale) so each branch moves the outputs; --gates zero keeps the exact
conversion (the model is the pretrained PET).
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch

ALL = {"frames": "D2", "edge_group_sum": True, "mesh_coupling": True, "couple_at": 0,
       "mesh_spacing": 1.0, "les": True, "total_charge_constraint": True}


def small_pet(featurizer, conditioning=True, nc_forces=True, seed=0, cutoff=4.5, adaptive=None):
    from metatrain.pet import PET
    from metatrain.utils.architectures import get_default_hypers
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_energy_target_info, get_generic_target_info
    torch.manual_seed(seed)
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    hypers.update(dict(cutoff=cutoff, d_pet=32, d_node=48, d_head=32, d_feedforward=64, num_heads=4,
                       num_attention_layers=2, num_gnn_layers=2, featurizer_type=featurizer,
                       system_conditioning=conditioning, max_charge=3, max_spin_multiplicity=4,
                       num_neighbors_adaptive=adaptive))
    if featurizer == "residual":  # PET-MOLS-S-like
        hypers.update(normalization="LayerNorm", activation="SiLU", transformer_type="PostLN",
                      cutoff_function="Cosine")
    targets = {"energy": get_energy_target_info("energy", {"unit": "eV"}, add_position_gradients=True)}
    if nc_forces:
        targets["non_conservative_force"] = get_generic_target_info("non_conservative_force", {
            "quantity": "force", "unit": "eV/A", "sample_kind": "atom", "num_subtargets": 1,
            "type": {"cartesian": {"rank": 1}}, "description": ""})
    info = DatasetInfo(length_unit="angstrom", atomic_types=[1, 6, 7, 8], targets=targets)
    model = PET(hypers, info)
    sc = model.system_conditioning
    if sc is not None:  # a "pretrained" conditioning: the zero-initialised gate randomised
        g = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            gate = sc.project[2]
            gate.weight.copy_(0.5 * torch.randn(gate.weight.shape, generator=g) / gate.weight.shape[1] ** 0.5)
            gate.bias.copy_(0.5 * torch.randn(gate.bias.shape, generator=g))
    with torch.no_grad():  # an energy scale != 1 (the LES charge unit sqrt(s))
        sc = model.scaler
        import metatensor.torch as mts
        for kind, maps in [("", sc.model.scales), ("_per_target", sc.model.per_target_scales)]:
            maps["energy"].block().values.fill_(2.5)
            sc.register_buffer(f"energy{kind}_scaler_buffer", mts.save_buffer(mts.make_contiguous(maps["energy"])))
        sc.sync_tensor_maps()
    return model


def load_ckpt(path):
    import metatomic.torch  # noqa: F401
    from metatrain.pet.convert_plato_lr import load_pet
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("architecture_name") == "llpr":
        ckpt = ckpt["wrapped_model_checkpoint"]
    return load_pet(ckpt), ckpt


def convert(model, plato_lr, extra_hypers=None):
    """metatrain's convert_model, with optional extra model hypers (e.g. conditioning
    added: its new tensors start at metatrain's init, the output gate zero)."""
    from metatrain.pet import PET
    from metatrain.pet.convert_plato_lr import set_charge_unit
    hypers = copy.deepcopy(dict(model.hypers))
    hypers["plato_lr"] = {**hypers.get("plato_lr", {}), **plato_lr}
    hypers.update(extra_hypers or {})
    p = next(model.parameters())
    torch.manual_seed(int(hypers["plato_lr"].get("seed", 0)))
    new = PET(hypers, model.dataset_info).to(dtype=p.dtype, device=p.device)
    old, cur = model.state_dict(), new.state_dict()
    unexpected = [k for k in old if k not in cur]
    if unexpected:
        raise ValueError(f"pretrained tensors not in the new PET: {unexpected[:5]}")
    new.load_state_dict({**cur, **old})
    new.additive_models[0].sync_tensor_maps()
    new.scaler.sync_tensor_maps()
    new.metadata = model.metadata
    new.finetune_config = copy.deepcopy(model.finetune_config)
    set_charge_unit(new)
    return new, [k for k in cur if k not in old]


def set_gates(model, mode, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed + 7)
    out = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.endswith("_gate"):
                continue
            if mode == "random":
                if name.endswith("rope_gate"):
                    p.copy_(scale * (torch.rand(p.shape, generator=g) * 0.6 + 0.2))
                elif name.endswith("les_gate"):
                    p.fill_(0.8 * scale)
                else:  # per-channel gates of edge and mesh updates
                    p.copy_(scale * 0.5 * torch.randn(p.shape, generator=g))
            else:
                p.zero_()
            out[name] = p.flatten()[:4].tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--small", choices=["feedforward", "residual"], help="a fresh small PET")
    src.add_argument("--ckpt", help="a pretrained PET checkpoint")
    ap.add_argument("--no-conditioning", action="store_true", help="--small without conditioning")
    ap.add_argument("--add-conditioning", action="store_true", help="switch system_conditioning on (new, zero gate)")
    ap.add_argument("--plato-lr", default=json.dumps(ALL), help="plato_lr hypers (JSON)")
    ap.add_argument("--gates", choices=["zero", "random"], default="zero")
    ap.add_argument("--gate-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adaptive", type=int, default=None, help="--small: num_neighbors_adaptive")
    ap.add_argument("--save-base", help="also write the unconverted PET here (a {'model': PET} dump)")
    args = ap.parse_args()

    if args.small:
        base = small_pet(args.small, conditioning=not args.no_conditioning, seed=args.seed, adaptive=args.adaptive)
    else:
        base, _ = load_ckpt(args.ckpt)
    if args.save_base:
        torch.save({"model": base.eval(), "hypers": base.hypers}, args.save_base)
    extra = {"system_conditioning": True} if args.add_conditioning else None
    model, new = convert(base, json.loads(args.plato_lr), extra)
    gates = set_gates(model, args.gates, args.seed, args.gate_scale)
    model = model.eval()
    out = Path(args.output)
    torch.save({"model": model, "hypers": model.hypers}, out)
    torch.save(model.get_checkpoint(), out.with_suffix(".ckpt"))
    info = {"plato_lr": model.hypers["plato_lr"], "new_tensors": new, "gates": gates,
            "charge_unit": float(model.backend.les.charge_unit) if model.backend.les is not None else None}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(f"{out}: {sum(p.numel() for p in model.parameters())} params, {len(new)} new tensors, gates {args.gates}")


if __name__ == "__main__":
    main()
