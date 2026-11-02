#!/usr/bin/env python3
"""A PET-Plato-LR checkpoint with the gated E_split term added (metatrain plato_lr hypers
les_split*, feat/pet-plato-lr-split): every other tensor unchanged, plato_split_gate at
--gate (default 0: the same energies as the input) and the radius R at --radius.

    python split_variant.py --ckpt E-init.ckpt -o E-init-split.pt [--gate 0] [--radius 8]
        [--delta 1] [--radius-max 10]

--ckpt: a checkpoint or another variant's dump (e.g. les_tail_variant.py --tail-sigma 4 first: a
checkpoint from gguf_to_checkpoint.py carries the init dump's hypers, not the GGUF's LES tail).
"""
import argparse
import copy

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--radius-max", type=float, default=10.0)
    args = ap.parse_args()
    import metatomic.torch  # noqa: F401
    from metatrain.pet import PET
    from metatrain.pet.convert_plato_lr import load_pet
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # a checkpoint or a variant's dump
    model = ck["model"] if isinstance(ck.get("model"), torch.nn.Module) else load_pet(ck)
    hypers = copy.deepcopy(dict(model.hypers))
    h = dict(hypers["plato_lr"])
    if not h.get("les", False):
        raise SystemExit("E_split needs les: true")
    h.update(les_split=True, les_split_radius=args.radius, les_split_delta=args.delta,
             les_split_radius_max=args.radius_max)
    hypers["plato_lr"] = h
    new = PET(hypers, model.dataset_info).to(next(model.parameters()).dtype)
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    if unexpected or set(missing) != {"backend.les.split_gate", "backend.les.split_radius"}:
        raise SystemExit(f"state dict: missing {missing}, unexpected {unexpected}")
    new.additive_models[0].sync_tensor_maps()
    new.scaler.sync_tensor_maps()
    new.metadata = model.metadata
    les = new.backend.les
    with torch.no_grad():
        les.split_gate.fill_(args.gate)
        les.split_radius.fill_(args.radius)
    new.eval()
    torch.save({"model": new, "hypers": new.hypers}, args.output)
    print(f"{args.output}: E_split gate {les.split_gate.item()}, R {les.split_radius.item()}, "
          f"delta {les.split_delta}, R_max {les.split_radius_max}")


if __name__ == "__main__":
    main()
