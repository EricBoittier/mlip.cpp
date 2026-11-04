#!/usr/bin/env python3
"""SOAP-PET variant 1 architecture dump: a make_init.py-style dump (``{"model": PET}``) of a
Plato-PET-LR PET with ``plato_lr.soap_node`` added (metatrain feat/pet-soap-v1,
pet/modules/soap_node.py), every existing weight copied, the new tensors at their
(exact-at-init) start, for gguf_to_checkpoint.py --init and export_pet_gguf.py.

    python run.py soap_v1_init.py SRC.pt OUT.pt [--layers all|input|0,1,2] [--d-hidden 256]
        [--cutoff 5] [--l-max 4] [--n-max 6] [--n-embed 4] [--les-tail-sigma 4] [--random-w S]

--les-tail-sigma: the LES tail of the run being extended (Fa / E-init: 4).
--random-w S: draw the zero-initialised last layers with std S (a parity-test model only).
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--d-hidden", type=int, default=256)
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument("--l-max", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--n-embed", type=int, default=4)
    ap.add_argument("--les-tail-sigma", type=float, default=None)
    ap.add_argument("--random-w", type=float, default=0.0)
    a = ap.parse_args()
    import metatomic.torch  # noqa: F401
    from metatrain.pet.convert_plato_lr import convert_model

    src = torch.load(a.src, map_location="cpu", weights_only=False)
    model = src["model"] if isinstance(src, dict) and isinstance(src.get("model"), torch.nn.Module) else src
    layers = a.layers if a.layers in ("all", "input") else [int(k) for k in a.layers.split(",")]
    cfg = {"layers": layers, "d_hidden": a.d_hidden, "cutoff": a.cutoff, "l_max": a.l_max,
           "n_max": a.n_max, "n_embed": a.n_embed}
    extra = {"soap_node": cfg}
    if a.les_tail_sigma is not None:
        extra["les_tail_sigma"] = a.les_tail_sigma
    new = convert_model(model, extra)
    if a.random_w > 0:
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for mlp in new.backend.soap_node.mlps:
                for t in (mlp[2].weight, mlp[2].bias):
                    t.copy_(torch.randn(t.shape, generator=g) * a.random_w)
    names = [k for k in new.state_dict() if "soap" in k]
    print(f"soap_node {cfg}: points {new.backend.soap_node.layers}, SOAP dim {new.backend.soap.out_dim}, "
          f"{sum(p.numel() for n, p in new.named_parameters() if 'soap' in n)} new parameters in {len(names)} tensors")
    torch.save({"model": new, "hypers": new.hypers}, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
