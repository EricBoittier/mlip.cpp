#!/usr/bin/env python3
"""Fresh (metatrain-initialised) Plato-LR PET for the from-scratch ablation ladder (feat/plato-lr-v2).

Plato-D2-att in PET form: D2 (Klein four-group) frames inside the attention only, with the frame path ungated
(rope_init "small" makes rope_gate a constant buffer of ones; rope_eps 1 keeps Plato's full RoPE frequencies, so the
frame attention is the attention from step 0, not a zero-init amplitude), the D2 group-sum edge term with its per-channel
edge_gate started at 1, and charge / spin system conditioning with its output projection randomly initialised (not
zero). No mesh and no LES: the long range of the ladder is the lrc fixed-form inter-cluster Coulomb (add_lrc_keys.py).

    python run.py make_scratch.py -o R0.pt [--d-pet 128 --gnn 2 --att 2 --cutoff 8 --seed 13] [--energy-only-charge]

--charge-model: the stage-1 charge network instead (no conditioning, no direct-force head): its per-atom "energy" is
the raw charge q_raw (the lrc graph mode reads per-atom outputs times pet.energy_scale, no composition), so the energy
scale is 1 and the composition 0.

Writes OUT ({"model": PET}), OUT.ckpt (model.get_checkpoint()) and OUT.json (hypers, parameter count, gate values)."""
import argparse
import copy
import json
from pathlib import Path

import torch

OMOL_TYPES = list(range(1, 84))  # PET-OMol-S species (pet.species_map of the program's GGUFs)


def build(args, charge_model=False):
    from metatrain.pet import PET
    from metatrain.utils.architectures import get_default_hypers
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_energy_target_info, get_generic_target_info
    torch.manual_seed(args.seed)
    h = copy.deepcopy(get_default_hypers("pet")["model"])
    h.update(cutoff=args.cutoff, d_pet=args.d_pet, d_head=args.d_pet, d_node=2 * args.d_pet,
             d_feedforward=2 * args.d_pet, num_heads=args.heads, num_attention_layers=args.att,
             num_gnn_layers=args.gnn, num_neighbors_adaptive=args.adaptive, adaptive_cutoff_method="grid",
             cutoff_width_adaptive=0.5, system_conditioning=not charge_model, max_charge=10,
             max_spin_multiplicity=11)
    h["plato_lr"] = {**h["plato_lr"], "frames": "D2", "frame_mode": "attention", "rope_init": "small",
                     "rope_eps": 1.0, "edge_group_sum": True, "mesh_coupling": False, "les": False,
                     "seed": args.seed}
    targets = {"energy": get_energy_target_info("energy", {"unit": "eV"}, add_position_gradients=not charge_model)}
    if not charge_model:
        targets["non_conservative_force"] = get_generic_target_info("non_conservative_force", {
            "quantity": "force", "unit": "eV/A", "sample_kind": "atom", "num_subtargets": 1,
            "type": {"cartesian": {"rank": 1}}, "description": ""})
    info = DatasetInfo(length_unit="angstrom", atomic_types=OMOL_TYPES, targets=targets)
    model = PET(h, info)
    gates = {}
    g = torch.Generator().manual_seed(args.seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("edge_gate"):
                p.fill_(1.0)
                gates[name] = 1.0
        sc = model.system_conditioning
        if sc is not None:  # metatrain zero-initialises the conditioning's output projection: start it random
            lin = sc.project[2]
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) / lin.weight.shape[1] ** 0.5)
            lin.bias.zero_()
            gates["system_conditioning.project.2"] = "random (1/sqrt(fan_in))"
        if charge_model:  # q_raw = per-atom output: energy scale 1, composition 0
            import metatensor.torch as mts
            scl = model.scaler
            for kind, maps in [("", scl.model.scales), ("_per_target", scl.model.per_target_scales)]:
                maps["energy"].block().values.fill_(1.0)
                scl.register_buffer(f"energy{kind}_scaler_buffer", mts.save_buffer(mts.make_contiguous(maps["energy"])))
            scl.sync_tensor_maps()
    for name, b in model.named_buffers():
        if name.endswith("rope_gate"):
            assert torch.all(b == 1), name  # ungated frame path
    return model, h, gates


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--d-pet", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--gnn", type=int, default=2)
    ap.add_argument("--att", type=int, default=2)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--adaptive", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--charge-model", action="store_true")
    a = ap.parse_args()
    model, h, gates = build(a, a.charge_model)
    model = model.eval()
    out = Path(a.output)
    torch.save({"model": model, "hypers": model.hypers}, out)
    torch.save(model.get_checkpoint(), out.with_suffix(".ckpt"))
    n = sum(p.numel() for p in model.parameters())
    out.with_suffix(".json").write_text(json.dumps({"hypers": h, "params": n, "gates": gates,
                                                    "charge_model": a.charge_model}, indent=1, default=str))
    print(f"{out}: {n} parameters ({'charge model' if a.charge_model else 'energy model'})")


if __name__ == "__main__":
    main()
