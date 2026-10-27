#!/usr/bin/env python3
"""A small metatrain PET with charge / spin conditioning (``system_conditioning:
true``), for the mlip.cpp parity tests. Init only: metatrain builds the model, no
training. The zero-initialised output gate of ``SystemConditioningEmbedding`` (and,
optionally, the whole conditioning branch) is randomised so the conditioning changes
the outputs.

    python make_pet_cond.py -o pet-cond.pt [--featurizer feedforward|residual]
        [--max-charge 3] [--max-spin 4] [--seed 0] [--gate-scale 0.5]

Writes OUT ({"model": PET, "hypers": ...}, what export_pet_gguf.py --model reads),
OUT with .ckpt (a metatrain checkpoint, model.get_checkpoint()) and OUT with
-mta.pt (the exported metatomic AtomisticModel, for evaluation).
"""
import argparse
from pathlib import Path

import torch

from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info, get_generic_target_info


def build(featurizer="feedforward", max_charge=3, max_spin=4, seed=0, gate_scale=0.5,
          types=(1, 6, 7, 8), nc_forces=True, cutoff=4.5, adaptive=None, conditioning=True):
    torch.manual_seed(seed)
    hypers = get_default_hypers("pet")["model"]
    hypers.update(dict(
        cutoff=cutoff, d_pet=32, d_node=48, d_head=32, d_feedforward=64, num_heads=4,
        num_attention_layers=2, num_gnn_layers=2, featurizer_type=featurizer,
        system_conditioning=conditioning, max_charge=max_charge, max_spin_multiplicity=max_spin,
        num_neighbors_adaptive=adaptive,
    ))
    targets = {"energy": get_energy_target_info("energy", {"unit": "eV"}, add_position_gradients=True)}
    if nc_forces:
        targets["non_conservative_force"] = get_generic_target_info("non_conservative_force", {
            "quantity": "force", "unit": "eV/A", "sample_kind": "atom", "num_subtargets": 1,
            "type": {"cartesian": {"rank": 1}}, "description": ""})
    info = DatasetInfo(length_unit="angstrom", atomic_types=list(types), targets=targets)
    model = PET(hypers, info)
    sc = model.system_conditioning
    if not conditioning:
        model.eval()
        return model, hypers
    assert sc is not None and sc is model.backend.system_conditioning
    # the gate is zero-initialised (a no-op); randomise it, and spread the tables, so
    # charge and spin move the energy by a visible amount
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        gate = sc.project[2]
        gate.weight.copy_(gate_scale * torch.randn(gate.weight.shape, generator=g) / gate.weight.shape[1] ** 0.5)
        gate.bias.copy_(gate_scale * torch.randn(gate.bias.shape, generator=g))
        # a fresh scaler / composition: energies of O(1) eV per atom are fine
    model.eval()
    return model, hypers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--featurizer", default="feedforward", choices=["feedforward", "residual"])
    ap.add_argument("--max-charge", type=int, default=3)
    ap.add_argument("--max-spin", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-scale", type=float, default=0.5)
    ap.add_argument("--no-nc-forces", action="store_true")
    ap.add_argument("--adaptive", type=int, default=None, help="num_neighbors_adaptive")
    ap.add_argument("--no-conditioning", action="store_true", help="the same PET without conditioning")
    args = ap.parse_args()
    model, hypers = build(args.featurizer, args.max_charge, args.max_spin, args.seed, args.gate_scale,
                          nc_forces=not args.no_nc_forces, adaptive=args.adaptive,
                          conditioning=not args.no_conditioning)
    out = Path(args.output)
    torch.save({"model": model, "hypers": hypers}, out)
    torch.save(model.get_checkpoint(), out.with_suffix(".ckpt"))
    mta = model.export()
    mta.save(str(out.with_name(out.stem + "-mta.pt")))
    names = [n for n, _ in model.state_dict().items() if "system_conditioning" in n]
    print(f"{out}: {sum(p.numel() for p in model.parameters())} params; conditioning tensors {names}")


if __name__ == "__main__":
    main()
