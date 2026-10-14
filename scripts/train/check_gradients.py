#!/usr/bin/env python3
"""Compare mlipcpp-train's loss and gradients with metatrain's PET in PyTorch.

mlipcpp-train-check dumps the loss and the flat parameter gradients of one
fixed batch; this script evaluates the same structures through metatrain's
own `PET.forward` (training mode: raw outputs, no scaler or composition), in
float64, with the composition and scales the C++ side fitted, and compares.

    python check_gradients.py --init init.pt --data DIR --dump DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from metatomic.torch import ModelOutput, System
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_pytorch.export_pet_full import PETFullModel, _unwrap_to_pet, get_model_params  # noqa: E402
from export_pytorch.export_pet_full import load_local_model  # noqa: E402


def read_split(d: Path):
    na = np.load(d / "na.npy")
    return dict(
        na=na,
        x=np.fromfile(d / "x.bin", np.float32).reshape(-1, 3),
        a=np.fromfile(d / "a.bin", np.int32),
        c=np.fromfile(d / "c.bin", np.float32).reshape(-1, 3, 3),
        e=np.fromfile(d / "energy.bin", np.float32),
        f=np.fromfile(d / "non_conservative_force.bin", np.float32).reshape(-1, 3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="make_init.py dump")
    ap.add_argument("--data", required=True, help="directory holding train/")
    ap.add_argument("--dump", required=True, help="mlipcpp-train-check output")
    ap.add_argument("--tol", type=float, default=1e-3, help="max relative error per tensor")
    args = ap.parse_args()

    meta = json.loads((Path(args.dump) / "check.json").read_text())
    grads_cpp = np.fromfile(Path(args.dump) / "grads.bin", np.float32)
    split = read_split(Path(args.data) / "train")

    model = load_local_model(args.init).to(torch.float64)
    model.train()
    nl_options = model.requested_neighbor_lists()
    comp = {int(k): v for k, v in meta["composition"].items()}
    fscale = {int(k): v for k, v in meta["force_scale"].items()}
    s_e = meta["energy_scale"]

    systems, e_target, f_target = [], [], []
    for s in meta["ids"]:
        lo, hi = split["na"][s], split["na"][s + 1]
        types = torch.tensor(split["a"][lo:hi])
        cell = torch.tensor(split["c"][s], dtype=torch.float64)
        system = System(types=types, positions=torch.tensor(split["x"][lo:hi], dtype=torch.float64),
                        cell=cell, pbc=(cell != 0).any(dim=1))
        systems.append(get_system_with_neighbor_lists(system, nl_options))
        n = hi - lo
        e_target.append((split["e"][s] - sum(comp[int(z)] for z in types)) / (s_e * n))
        f_target.append(split["f"][lo:hi] / np.array([fscale[int(z)] for z in types])[:, None])

    out = model(systems, {"energy": ModelOutput(sample_kind="atom"),
                          "non_conservative_force": ModelOutput(sample_kind="atom")})
    e_atom = out["energy"].block().values.reshape(-1)
    f_atom = out["non_conservative_force"].block().values.reshape(-1, 3)
    counts = [len(s) for s in systems]
    e_pa = torch.stack([chunk.sum() / len(chunk) for chunk in torch.split(e_atom, counts)])
    loss_e = ((e_pa - torch.tensor(e_target, dtype=torch.float64)) ** 2).mean()
    loss_f = ((f_atom - torch.tensor(np.concatenate(f_target), dtype=torch.float64)) ** 2).mean()
    loss = loss_e + loss_f
    loss.backward()

    print(f"loss   torch {loss.item():.8g}  mlipcpp {meta['loss']:.8g}  "
          f"rel {abs(loss.item() - meta['loss']) / abs(loss.item()):.2e}")
    print(f"loss_E torch {loss_e.item():.8g}  mlipcpp {meta['loss_energy']:.8g}")
    print(f"loss_F torch {loss_f.item():.8g}  mlipcpp {meta['loss_force']:.8g}")

    # GGUF tensor names are the export wrapper's parameter names, '.' -> '_'
    pet = _unwrap_to_pet(model)
    p = get_model_params(pet)
    wrapper = PETFullModel(pet, n_atoms=7, max_neighbors=11, d_pet=p["d_pet"], forces=True,
                           cutoff=p["cutoff"], cutoff_width=p["cutoff_width"],
                           cutoff_function=p["cutoff_function"], nc_forces=True)
    by_name = {n.replace(".", "_"): t for n, t in wrapper.named_parameters()}

    worst, failures = 0.0, []
    for entry in meta["params"]:
        g_cpp = grads_cpp[entry["offset"]: entry["offset"] + entry["size"]].astype(np.float64)
        t = by_name[entry["name"]]
        g_ref = t.grad.reshape(-1).numpy() if t.grad is not None else np.zeros_like(g_cpp)
        ref_norm = np.linalg.norm(g_ref)
        rel = np.linalg.norm(g_cpp - g_ref) / ref_norm if ref_norm > 0 else np.linalg.norm(g_cpp)
        worst = max(worst, rel)
        if rel > args.tol:
            failures.append((entry["name"], rel, ref_norm))
    print(f"{len(meta['params'])} tensors, worst relative gradient error {worst:.2e}")
    for name, rel, norm in sorted(failures, key=lambda x: -x[1]):
        print(f"  FAIL {name}: rel {rel:.2e} (|g| {norm:.3e})")
    loss_rel = abs(loss.item() - meta["loss"]) / abs(loss.item())
    ok = not failures and loss_rel < args.tol
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
