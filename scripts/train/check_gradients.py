#!/usr/bin/env python3
"""Compare mlipcpp-train's loss and gradients with metatrain's PET in PyTorch.

mlipcpp-train-check dumps the loss and the flat parameter gradients of one
fixed batch; this script evaluates the same structures through metatrain's
own `PET.forward` (training mode: raw outputs, no scaler or composition), in
float64, with the composition and scales the C++ side fitted, and compares.
With a conservative dump (mlipcpp-train-check --conservative 1) the force loss
is on -dE/dx (autograd through the positions, adaptive cutoff included), the
direct head keeping its term; the parameter gradients are then second
derivatives (double backward).

    python check_gradients.py --init init.pt --data DIR --dump DIR [--fp32-spread 1]

A trained model's mixed derivatives can be sensitive to rounding (sharp
attention), so float32 alone can move them by far more than --tol. With
--fp32-spread 1 the same loss and gradients are also computed in float32 in
PyTorch, and a tensor passes when mlip.cpp's error against float64 is within
--spread-factor times PyTorch-float32's (or below --tol).

Padded neighbour slots are masked by a finite attention bias (-34.5), so a
trained model's gradients depend on what fills them. metatrain copies a real
edge (differentiable in the positions); mlip.cpp uses the fixed edge of
src/core/padding.h (species index 0, vector (0, 0, cutoff)). --pad mlipcpp
(the default) gives the reference mlip.cpp's fill, so both sides compute the
same function; --pad metatrain keeps metatrain's own.
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
    # charge / spin conditioning (metatrain extra_data; absent: 0 and 1)
    opt = lambda name, default: (np.fromfile(d / f"{name}.bin", np.float32).reshape(-1)
                                 if (d / f"{name}.bin").exists() else np.full(len(na) - 1, default, np.float32))
    return dict(
        charge=opt("charge", 0.0),
        spin_multiplicity=opt("spin_multiplicity", 1.0),
        na=na,
        x=np.fromfile(d / "x.bin", np.float32).reshape(-1, 3),
        a=np.fromfile(d / "a.bin", np.int32),
        c=np.fromfile(d / "c.bin", np.float32).reshape(-1, 3, 3),
        e=np.fromfile(d / "energy.bin", np.float32),
        f=np.fromfile(d / "non_conservative_force.bin", np.float32).reshape(-1, 3),
    )


def scalar_tmap(value, name, dtype):
    from metatensor.torch import Labels, TensorBlock, TensorMap
    return TensorMap(Labels.single(), [TensorBlock(
        values=torch.tensor([[value]], dtype=dtype), samples=Labels("system", torch.tensor([[0]])),
        components=[], properties=Labels(name, torch.tensor([[0]])))])


def use_mlipcpp_padding(exact=False):
    """metatrain's batch tensors with padded slots refilled as src/core/padding.h does.

    exact: real edges with cutoff factor 0 become 1e-15 (the clamp they get
    anyway, and no gradient either way), so that with the attention floor at 0
    only the padded slots (factor 0) get -inf, as with MLIPCPP_PAD_EXACT=1."""
    import metatrain.pet.modules.backend as backend

    original = backend.compute_batch_tensors

    def padded(*args, **kwargs):
        out = list(original(*args, **kwargs))
        cutoff = kwargs["cutoff"] if "cutoff" in kwargs else args[8]
        # (element_indices_nodes, element_indices_neighbors, edge_vectors, edge_distances, nef_mask, ...)
        species, vectors, mask = out[1], out[2], out[4]
        fill = torch.zeros_like(vectors)
        fill[..., 2] = cutoff
        out[1] = torch.where(mask, species, torch.zeros_like(species))
        out[2] = torch.where(mask[..., None], vectors, fill)
        out[3] = torch.sqrt(torch.sum(out[2] ** 2, dim=2) + 1e-15)
        if exact:  # (..., padding_mask, reverse_neighbor_index, cutoff_factors, ...)
            cf = out[6]
            out[6] = torch.where(mask & (cf <= 1e-15), torch.full_like(cf, 1e-15), cf)
        return tuple(out)

    backend.compute_batch_tensors = padded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="make_init.py dump")
    ap.add_argument("--data", required=True, help="directory holding train/")
    ap.add_argument("--dump", required=True, help="mlipcpp-train-check output")
    ap.add_argument("--tol", type=float, default=1e-3, help="max relative error per tensor")
    ap.add_argument("--fp32-spread", type=int, default=0,
                    help="1: also run PyTorch in float32 and gate on its spread from float64")
    ap.add_argument("--spread-factor", type=float, default=3.0)
    ap.add_argument("--zero-tol", type=float, default=1e-6,
                    help="tensors with both gradient norms below this pass (identically zero gradients)")
    ap.add_argument("--pad", choices=["mlipcpp", "metatrain"], default="mlipcpp",
                    help="contents of padded neighbour slots in the reference")
    ap.add_argument("--pad-exact", type=int, default=0,
                    help="1: padded keys get -inf (pair with mlip.cpp's MLIPCPP_PAD_EXACT=1); "
                         "real edges with cutoff factor 0 keep -34.5")
    ap.add_argument("--attn-eps", type=float, default=None,
                    help="attention floor of the cutoff factors (metatrain: 1e-15); pair with "
                         "mlip.cpp's MLIPCPP_PAD_FLOOR to move the padded keys' bias on both sides")
    args = ap.parse_args()
    if args.pad == "mlipcpp" or args.pad_exact:
        use_mlipcpp_padding(exact=bool(args.pad_exact))
    if args.pad_exact:  # padded keys at the smallest positive floor: weight ~0, but no log(0) (0 * inf = NaN in backward)
        args.attn_eps = "tiny"

    meta = json.loads((Path(args.dump) / "check.json").read_text())
    for k in ("loss", "loss_energy", "loss_force"):  # a non-finite loss is written as null
        meta[k] = float("nan") if meta.get(k) is None else meta[k]
    grads_cpp = np.fromfile(Path(args.dump) / "grads.bin", np.float32)
    split = read_split(Path(args.data) / "train")
    cons = meta.get("conservative")

    def run(dtype):
        model = load_local_model(args.init).to(dtype)
        model.train()
        if args.attn_eps is not None:
            for mod in model.modules():
                if type(mod).__name__ == "AttentionBlock":
                    mod.epsilon = torch.finfo(dtype).tiny if args.attn_eps == "tiny" else args.attn_eps
        nl_options = model.requested_neighbor_lists()
        comp = {int(k): v for k, v in meta["composition"].items()}
        fscale = {int(k): v for k, v in meta["force_scale"].items()}
        s_e = meta["energy_scale"]

        cons = meta.get("conservative")
        systems, e_target, f_target, f_phys = [], [], [], []
        for s in meta["ids"]:
            lo, hi = split["na"][s], split["na"][s + 1]
            types = torch.tensor(split["a"][lo:hi])
            cell = torch.tensor(split["c"][s], dtype=dtype)
            positions = torch.tensor(split["x"][lo:hi], dtype=dtype, requires_grad=cons is not None)
            system = System(types=types, positions=positions, cell=cell, pbc=(cell != 0).any(dim=1))
            for key in ("charge", "spin_multiplicity"):
                system.add_data(key, scalar_tmap(float(split[key][s]), key, dtype))
            systems.append(get_system_with_neighbor_lists(system, nl_options))
            n = hi - lo
            e_target.append((split["e"][s] - sum(comp[int(z)] for z in types)) / (s_e * n))
            f_target.append(split["f"][lo:hi] / np.array([fscale[int(z)] for z in types])[:, None])
            f_phys.append(split["f"][lo:hi])

        out = model(systems, {"energy": ModelOutput(sample_kind="atom"),
                              "non_conservative_force": ModelOutput(sample_kind="atom")})
        e_atom = out["energy"].block().values.reshape(-1)
        f_atom = out["non_conservative_force"].block().values.reshape(-1, 3)
        counts = [len(s) for s in systems]
        e_pa = torch.stack([chunk.sum() / len(chunk) for chunk in torch.split(e_atom, counts)])
        loss_e = ((e_pa - torch.tensor(e_target, dtype=dtype)) ** 2).mean()
        loss_f = ((f_atom - torch.tensor(np.concatenate(f_target), dtype=dtype)) ** 2).mean()
        if cons:
            # -dE/dx of the scaled energy against F / s_E, kept differentiable
            grads = torch.autograd.grad(e_atom.sum(), [s.positions for s in systems], create_graph=True)
            f_cons = -torch.cat(grads)
            loss_fc = ((f_cons - torch.tensor(np.concatenate(f_phys) / s_e, dtype=dtype)) ** 2).mean()
            loss_e, loss_nc = cons["energy_weight"] * loss_e, cons["nc_force_weight"] * loss_f
            loss_f = cons["force_weight"] * loss_fc
            loss = loss_e + loss_f + loss_nc
        else:
            loss = loss_e + loss_f
        loss.backward()
        return model, loss, loss_e, loss_f, (loss_nc if cons else None)

    model, loss, loss_e, loss_f, loss_nc = run(torch.float64)
    run32 = run(torch.float32) if args.fp32_spread else None
    print(f"loss   torch {loss.item():.8g}  mlipcpp {meta['loss']:.8g}  "
          f"rel {abs(loss.item() - meta['loss']) / abs(loss.item()):.2e}")
    print(f"loss_E torch {loss_e.item():.8g}  mlipcpp {meta['loss_energy']:.8g}")
    print(f"loss_F torch {loss_f.item():.8g}  mlipcpp {meta['loss_force']:.8g}")
    if cons:
        print(f"loss_F_nc torch {loss_nc.item():.8g}  mlipcpp {cons['loss_nc']:.8g}")
    if run32:
        print(f"loss   torch float32 {run32[1].item():.8g}  rel {abs(run32[1].item() - loss.item()) / abs(loss.item()):.2e}")

    # GGUF tensor names are the export wrapper's parameter names, '.' -> '_'
    def grads_by_name(m):
        pet = _unwrap_to_pet(m)
        p = get_model_params(pet)
        wrapper = PETFullModel(pet, n_atoms=7, max_neighbors=11, d_pet=p["d_pet"], forces=True,
                               cutoff=p["cutoff"], cutoff_width=p["cutoff_width"],
                               cutoff_function=p["cutoff_function"], nc_forces=True)
        out = {n.replace(".", "_"): t for n, t in wrapper.named_parameters()}
        plato = getattr(wrapper, "plato", None)
        mesh = plato.__dict__.get("_mesh_module") if plato is not None else None
        if mesh is not None:
            # Plato-PET-LR: the folded mesh projections W_eff = sum_b k_b W_b stand for the
            # FrameLinears: dL/dW_eff = sum_b k_b dL/dW_b / |k|^2, the bias as it is
            from export_pytorch.plato_lr import fold_frame_linear

            class G:
                def __init__(self, g):
                    self.grad = g
            for name in ("down", "up"):
                fl = getattr(mesh.coupling, name)
                _, k = fold_frame_linear(fl)
                gW = None if fl.W.grad is None else torch.einsum("b,boc->oc", k, fl.W.grad.double()) / (k @ k)
                out[f"plato_mesh_{name}_weight"] = G(gW)
                out[f"plato_mesh_{name}_bias"] = G(fl.b.grad)
        return out

    def rel_err(g, ref):
        norm = np.linalg.norm(ref)
        return np.linalg.norm(g - ref) / norm if norm > 0 else np.linalg.norm(g)

    by_name = grads_by_name(model)
    by_name32 = grads_by_name(run32[0]) if run32 else None
    worst, worst32, failures = 0.0, 0.0, []
    for entry in meta["params"]:
        g_cpp = grads_cpp[entry["offset"]: entry["offset"] + entry["size"]].astype(np.float64)
        flat = lambda t: t.grad.reshape(-1).double().numpy() if t.grad is not None else np.zeros_like(g_cpp)
        g_ref = flat(by_name[entry["name"]])
        rel = rel_err(g_cpp, g_ref)
        if np.linalg.norm(g_ref) < args.zero_tol and np.linalg.norm(g_cpp) < args.zero_tol:
            rel = 0.0  # a gradient that vanishes identically (e.g. a bias the LES neutralisation removes)
        worst = max(worst, rel)
        allowed = args.tol
        if by_name32:
            rel32 = rel_err(flat(by_name32[entry["name"]]), g_ref)
            worst32 = max(worst32, rel32)
            allowed = max(allowed, args.spread_factor * rel32)
        if "system_conditioning" in entry["name"]:  # charge / spin conditioning: always shown
            print(f"  {entry['name']}: |g| {np.linalg.norm(g_ref):.4e}, relative error {rel:.2e}")
        if rel > allowed:
            failures.append((entry["name"], rel, allowed, np.linalg.norm(g_ref)))
    print(f"{len(meta['params'])} tensors, worst relative gradient error {worst:.2e}"
          + (f" (PyTorch float32 vs float64: worst {worst32:.2e}; allowed {args.spread_factor:g}x per tensor)"
             if by_name32 else ""))
    for name, rel, allowed, norm in sorted(failures, key=lambda x: -x[1] / x[2]):
        print(f"  FAIL {name}: rel {rel:.2e} > allowed {allowed:.2e} (|g| {norm:.3e})")
    loss_rel = abs(loss.item() - meta["loss"]) / abs(loss.item())
    loss_allowed = max(args.tol, args.spread_factor * abs(run32[1].item() - loss.item()) / abs(loss.item())) \
        if run32 else args.tol
    ok = not failures and loss_rel < loss_allowed
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
