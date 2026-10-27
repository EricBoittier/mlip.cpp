"""Fused (torch.ops.mlipcpp.pet_attention) vs manual attention of Plato-PET-LR on CUDA.

    python parity.py OUT.json name=CKPT ...

Per checkpoint, dtype (float64, float32) and system set: energies, forces, stress
(eval mode), and the parameter gradients of one E + F training loss (train mode,
fixed frames, create_graph forces). Errors are max |fused - manual| / max |manual|;
float32 also against the float64 manual result (both backends), to show the fused
float32 error is the same size as the PyTorch float32 error.
"""
import copy
import json
import sys

import numpy as np
import torch
from ase.io import read
from metatomic.torch import systems_to_torch

from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import load_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
import os

dev = torch.device("cuda")
frames = read(os.environ["PLATO_TEST_XYZ"], ":")
cc = next(a for a in frames if a.info["label"] == "CC")
rng = np.random.default_rng(0)
dimers = [frames[i] for i in rng.choice(len(frames), 8, replace=False)]


def open_copy(atoms):
    a = atoms.copy()
    a.pbc = False
    a.cell = np.zeros((3, 3))
    return a


SETS = {
    "dimers8_open": [open_copy(a) for a in dimers],
    "dimers8_pbc": dimers,
    "cell160_pbc": [cc.repeat((2, 2, 2))],
}
INFO = {
    "energy": get_energy_target_info(
        "energy", {"unit": "eV"}, add_position_gradients=True, add_strain_gradients=True
    )
}
INFO_F = {
    "energy": get_energy_target_info("energy", {"unit": "eV"}, add_position_gradients=True)
}


def systems(model, atoms, dtype):
    nl = model.requested_neighbor_lists()
    return [
        get_system_with_neighbor_lists(s, nl)
        for s in systems_to_torch(atoms, dtype=dtype, device=dev)
    ]


def eval_efs(model, sy, periodic):
    model.eval()
    info = INFO if periodic else INFO_F
    b = evaluate_model(model, sy, info, is_training=False)["energy"].block()
    out = {"E": b.values.detach().double(), "F": -b.gradient("positions").values.detach().double()}
    if periodic:
        V = torch.stack([torch.det(s.cell).abs() for s in sy]).double()
        out["S"] = b.gradient("strain").values.detach().double()[..., 0] / V[:, None, None]
    return out


def train_grads(model, sy):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    b = evaluate_model(model, sy, INFO_F, is_training=True)["energy"].block()
    E, G = b.values, b.gradient("positions").values
    loss = (E**2).mean() + (G**2).mean()
    loss.backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
    return {"loss": loss.detach().double().reshape(1), "grad": g.detach().double()}


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


res = {}
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    base = load_model(path)
    res[name] = {"frames": base.hypers["frames"], "frame_mode": base.hypers["frame_mode"]}
    print(name, res[name], flush=True)
    ref64 = {}
    for dtype in (torch.float64, torch.float32):
        dn = str(dtype).split(".")[-1]
        models = {}
        for be in ("manual", "fused"):
            m = copy.deepcopy(base).to(dtype=dtype, device=dev)
            m.set_attention_backend(be)
            assert m.use_fused == (be == "fused")
            models[be] = m
        for sname, atoms in SETS.items():
            periodic = bool(atoms[0].pbc.any())
            out = {}
            for be, m in models.items():
                sy = systems(m, atoms, dtype)
                out[be] = {**eval_efs(m, sy, periodic), **train_grads(m, systems(m, atoms, dtype))}
            r = {k: rel(out["fused"][k], out["manual"][k]) for k in out["manual"]}
            if dtype == torch.float64:
                ref64[sname] = out["manual"]
            else:
                for be in ("manual", "fused"):
                    r.update({f"{k}_{be}_vs_f64": rel(out[be][k], ref64[sname][k]) for k in out[be]})
            res[name][f"{sname}/{dn}"] = r
            print(f"  {sname:13s} {dn}", {k: f"{v:.1e}" for k, v in r.items()}, flush=True)
        del models
        torch.cuda.empty_cache()
    json.dump(res, open(sys.argv[1], "w"), indent=1)
