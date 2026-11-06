#!/usr/bin/env python3
"""A PET-Plato-LR checkpoint with other LES settings (the same tensors): the LES tail
(les_tail_sigma, les_tail_subtract_cutoff), the LES gate, and a scale of the latent charges
(the charge heads' output layers times s), for the les_tail diagnostics; and the damped and
screened LES kernel (--damp-screen: metatrain les_damp_screen, new parameters at their start
values w = s = 0, so the energy is unchanged; --damp-w / --screen-s / --damp-a / --screen-kappa
set them, e.g. for parity checks; --thole adds metatrain les_thole, w_T = 0, a_T = --thole-a0,
and --thole-w / --thole-a set them; --frag-charges adds metatrain frag_charges).

    python les_tail_variant.py --ckpt C.ckpt -o out.pt [--tail-sigma 3] [--subtract]
        [--les-gate 1] [--charge-scale 0.5] [--damp-screen [--damp-w 0.5 --screen-s 0.5 ...]]
"""
import math
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
    ap.add_argument("--damp-screen", action="store_true")
    ap.add_argument("--damp-w", type=float, default=None)
    ap.add_argument("--damp-a", type=float, default=None, help="a in A (softplus applied)")
    ap.add_argument("--screen-s", type=float, default=None)
    ap.add_argument("--screen-kappa", type=float, default=None, help="kappa in 1/A (softplus applied)")
    ap.add_argument("--thole", action="store_true", help="Thole damping (needs --damp-screen)")
    ap.add_argument("--thole-a0", type=float, default=0.39, help="les_thole_a hyper (start a_T)")
    ap.add_argument("--thole-w", type=float, default=None)
    ap.add_argument("--thole-a", type=float, default=None, help="a_T (softplus applied)")
    ap.add_argument("--frag-cond-gate", type=float, default=None,
                    help="with --frag-charges: frag_cond_gate, the gate filled with this value (or random, --frag-gate-seed)")
    ap.add_argument("--frag-gate-seed", type=int, default=None, help="gate uniform in +-|--frag-cond-gate| with this seed")
    ap.add_argument("--frag-charges", action="store_true",
                    help="metatrain frag_charges (fragment-constrained charges; no new tensors)")
    args = ap.parse_args()
    import metatomic.torch  # noqa: F401
    from metatrain.pet import PET
    from metatrain.pet.convert_plato_lr import load_pet
    model = load_pet(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    hypers = copy.deepcopy(dict(model.hypers))
    h = dict(hypers["plato_lr"])
    h["les_tail_sigma"] = args.tail_sigma
    h["les_tail_subtract_cutoff"] = bool(args.subtract)
    if args.damp_screen:
        h["les_damp_screen"] = True
    if args.thole:
        if not h.get("les_damp_screen", False):
            raise SystemExit("--thole needs --damp-screen")
        h["les_thole"], h["les_thole_a"] = True, args.thole_a0
    if args.frag_charges:
        h["frag_charges"] = True
        if args.frag_cond_gate is not None:
            h["frag_cond_gate"] = True
    hypers["plato_lr"] = h
    new = PET(hypers, model.dataset_info).to(next(model.parameters()).dtype)
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    allowed = {"backend.frag.cond_gate", "backend.les.les_damp_w", "backend.les.les_damp_a_raw", "backend.les.les_screen_s",
               "backend.les.les_screen_kappa_raw", "backend.les.les_thole_w", "backend.les.les_thole_a_raw",
               "backend.les.les_thole_ia"}
    if unexpected or not set(missing) <= allowed:
        raise RuntimeError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
    new.additive_models[0].sync_tensor_maps()
    new.scaler.sync_tensor_maps()
    new.metadata = model.metadata
    les = new.backend.les
    with torch.no_grad():
        if args.les_gate is not None:
            les.les_gate.fill_(args.les_gate)
        if args.frag_cond_gate is not None:
            g = new.backend.frag.cond_gate
            if args.frag_gate_seed is not None:
                gen = torch.Generator().manual_seed(args.frag_gate_seed)
                g.copy_((2 * torch.rand(g.shape, generator=gen, dtype=torch.float64) - 1).to(g.dtype) * abs(args.frag_cond_gate))
            else:
                g.fill_(args.frag_cond_gate)
        inv = lambda x: x + math.log(-math.expm1(-x))  # noqa: E731  softplus^-1
        if args.damp_w is not None: les.les_damp_w.fill_(args.damp_w)
        if args.screen_s is not None: les.les_screen_s.fill_(args.screen_s)
        if args.damp_a is not None: les.les_damp_a_raw.fill_(inv(args.damp_a))
        if args.screen_kappa is not None: les.les_screen_kappa_raw.fill_(inv(args.screen_kappa))
        if args.thole_w is not None: les.les_thole_w.fill_(args.thole_w)
        if args.thole_a is not None: les.les_thole_a_raw.fill_(inv(args.thole_a))
        for kind in ("node", "edge"):
            for head in les.charge_head[kind]:
                head[2].weight.mul_(args.charge_scale)
                head[2].bias.mul_(args.charge_scale)
    new.eval()
    torch.save({"model": new, "hypers": new.hypers}, args.output)
    print(f"{args.output}: les_tail_sigma {args.tail_sigma}, subtract {args.subtract}, les_gate {float(les.les_gate)}, "
          f"charge scale {args.charge_scale}" + (f", damp/screen {les.damp_screen_params().tolist()}" if args.damp_screen else ""))


if __name__ == "__main__":
    main()
