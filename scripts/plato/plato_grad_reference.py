"""PyTorch reference of one training batch for mlipcpp-train-plato --check: the loss
of metatrain's plato_pet_lr trainer (energies per atom and forces, MSE, targets with
the composition removed and divided by the energy scale; model in training mode, no
augmentation) and its gradient w.r.t. every trainable parameter, by double backward.

    python scripts/plato/plato_grad_reference.py model.ckpt data.xyz OUT --indices 0 1 2 3 \
        [--energy-key e_res --forces-key f] [--energy-weight 1 --force-weight 1] [--dtype float64]

Writes OUT.json (loss parts, the parameter names, shapes and offsets in the order of
the model's named_parameters) and OUT.f64 (the flattened gradients, float64).
"""
import argparse
import json

import ase.io
import metatomic.torch  # noqa: F401
import numpy as np
import torch
from metatomic.torch import ModelOutput, System, systems_to_torch
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from plato_conditioning import load_plato_model as load_model  # also with charge / spin conditioning


def scalar_tmap(value, name, dtype):
    from metatensor.torch import Labels, TensorBlock, TensorMap
    return TensorMap(Labels.single(), [TensorBlock(
        values=torch.tensor([[float(value)]], dtype=dtype), samples=Labels("system", torch.tensor([[0]])),
        components=[], properties=Labels(name, torch.tensor([[0]])))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("xyz")
    ap.add_argument("out")
    ap.add_argument("--indices", type=int, nargs="+", required=True)
    ap.add_argument("--energy-key", default="e_res")
    ap.add_argument("--forces-key", default="f")
    ap.add_argument("--energy-weight", type=float, default=1.0)
    ap.add_argument("--force-weight", type=float, default=1.0)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--plain", action="store_true", help="loss = the sum of the raw energies (a check of dE/dtheta)")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    torch.set_num_threads(8)
    model = load_model(a.checkpoint).to(dt)
    model.train()
    comp = {}
    w = model.additive_models[0].model.weights["energy"].block(0)
    comp = {int(z): float(v) for z, v in zip(w.samples.values[:, 0], w.values.reshape(-1))}
    scale = float(model.scaler.model.scales["energy"].block(0).values.reshape(-1)[0])
    frames = ase.io.read(a.xyz, ":")
    systems, E_t, F_t, nat = [], [], [], []
    for i in a.indices:
        at = frames[i]
        s = systems_to_torch(at, dtype=dt)
        pos = s.positions.detach().requires_grad_(True)
        s = System(types=s.types, positions=pos, cell=s.cell, pbc=s.pbc)
        # charge / spin (conditioned models; the dataset's info, as plato_dataset.py writes it)
        s.add_data("charge", scalar_tmap(at.info.get("charge", 0), "charge", dt))
        s.add_data("spin_multiplicity", scalar_tmap(at.info.get("spin_multiplicity", at.info.get("spin", 1)),
                                                    "spin_multiplicity", dt))
        systems.append(get_system_with_neighbor_lists(s, model.requested_neighbor_lists()))
        E_t.append((at.info[a.energy_key] - sum(comp.get(int(z), 0.0) for z in at.numbers)) / scale)
        F_t.append(torch.tensor(at.arrays[a.forces_key], dtype=dt) / scale)
        nat.append(len(at))
    out = model(systems, {"energy": ModelOutput(sample_kind="system")})["energy"].block().values.reshape(-1)
    grads = torch.autograd.grad(out.sum(), [s.positions for s in systems], create_graph=True)
    n = torch.tensor(nat, dtype=dt)
    le = ((out / n - torch.tensor(E_t, dtype=dt) / n) ** 2).mean()
    g = torch.cat(grads)  # dE/dx, target -F / scale
    lf = ((g + torch.cat(F_t)) ** 2).mean()
    loss = a.energy_weight * le + a.force_weight * lf
    if a.plain:
        loss = out.sum()
    params = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    gp = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    flat, meta, off = [], [], 0
    for (k, p), gg in zip(params, gp):
        gg = torch.zeros_like(p) if gg is None else gg
        flat.append(gg.detach().double().reshape(-1).numpy())
        meta.append({"name": k, "shape": list(p.shape), "offset": off})
        off += p.numel()
    np.concatenate(flat).astype("<f8").tofile(a.out + ".f64")
    json.dump({"loss": loss.item(), "loss_energy": le.item(), "loss_forces": lf.item(), "energies": out.tolist(),
               "scale": scale, "indices": a.indices, "params": meta, "dtype": a.dtype,
               "energy_weight": a.energy_weight, "force_weight": a.force_weight, "plain": a.plain}, open(a.out + ".json", "w"))
    print(f"loss {loss.item():.10g} (E {le.item():.6g}, F {lf.item():.6g}), {len(meta)} parameters, {off} values")


if __name__ == "__main__":
    main()
