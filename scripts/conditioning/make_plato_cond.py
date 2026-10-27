#!/usr/bin/env python3
"""A Plato-PET-LR checkpoint with charge / spin conditioning (scripts/plato/
plato_conditioning.py) for the parity tests: a trained (or fresh) plato_pet_lr
checkpoint plus PET's SystemConditioningEmbedding, its zero-initialised output gate
randomised so the conditioning changes the outputs. Init only, no training.

    python make_plato_cond.py plato.ckpt out.ckpt [--max-charge 3] [--max-spin 4] [--seed 0] [--gate-scale 0.5]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plato"))
from plato_conditioning import attach_conditioning, load_plato_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--max-charge", type=int, default=3)
    ap.add_argument("--max-spin", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-scale", type=float, default=0.5)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    model = load_plato_model(args.checkpoint)
    attach_conditioning(model, args.max_charge, args.max_spin)
    g = torch.Generator().manual_seed(args.seed + 1)
    gate = model.system_conditioning.project[2]
    with torch.no_grad():
        gate.weight.copy_(args.gate_scale * torch.randn(gate.weight.shape, generator=g) / gate.weight.shape[1] ** 0.5)
        gate.bias.copy_(args.gate_scale * torch.randn(gate.bias.shape, generator=g))
    src = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    ckpt = model.get_checkpoint()
    # keep the source's trainer fields (the exporters read best_model_state_dict / model_state_dict)
    for k in ("trainer_ckpt_version", "train_hypers", "epoch", "best_epoch"):
        if k in src:
            ckpt.setdefault(k, src[k])
    ckpt["best_model_state_dict"] = ckpt["model_state_dict"]
    torch.save(ckpt, args.out)
    n = sum(p.numel() for p in model.system_conditioning.parameters())
    print(f"{args.out}: conditioning {n} parameters, hypers max_charge {args.max_charge} "
          f"max_spin_multiplicity {args.max_spin}")


if __name__ == "__main__":
    main()
