#!/usr/bin/env python3
"""A PET-Plato-LR checkpoint with other LES settings (the same tensors): the LES tail
(les_tail_sigma, les_tail_subtract_cutoff), the LES gate, and a scale of the latent charges
(the charge heads' output layers times s), for the les_tail diagnostics.

    python les_tail_variant.py --ckpt C.ckpt -o out.pt [--tail-sigma 3] [--subtract]
        [--les-gate 1] [--charge-scale 0.5]
"""
import argparse
import copy

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--tail-sigma", type=float, default=0.0)
    ap.add_argument("--subtract", action="store_true")
    ap.add_argument("--les-gate", type=float, default=None)
    ap.add_argument("--charge-scale", type=float, default=1.0)
    args = ap.parse_args()
    import metatomic.torch  # noqa: F401
    from metatrain.pet import PET
    from metatrain.pet.convert_plato_lr import load_pet
    model = load_pet(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    hypers = copy.deepcopy(dict(model.hypers))
    h = dict(hypers["plato_lr"])
    h["les_tail_sigma"] = args.tail_sigma
    h["les_tail_subtract_cutoff"] = bool(args.subtract)
    hypers["plato_lr"] = h
    new = PET(hypers, model.dataset_info).to(next(model.parameters()).dtype)
    new.load_state_dict(model.state_dict())
    new.additive_models[0].sync_tensor_maps()
    new.scaler.sync_tensor_maps()
    new.metadata = model.metadata
    les = new.backend.les
    with torch.no_grad():
        if args.les_gate is not None:
            les.les_gate.fill_(args.les_gate)
        for kind in ("node", "edge"):
            for head in les.charge_head[kind]:
                head[2].weight.mul_(args.charge_scale)
                head[2].bias.mul_(args.charge_scale)
    new.eval()
    torch.save({"model": new, "hypers": new.hypers}, args.output)
    print(f"{args.output}: les_tail_sigma {args.tail_sigma}, subtract {args.subtract}, les_gate {float(les.les_gate)}, "
          f"charge scale {args.charge_scale}")


if __name__ == "__main__":
    main()
