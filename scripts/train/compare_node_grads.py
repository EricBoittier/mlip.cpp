#!/usr/bin/env python3
"""Localise a wrong backward rule: compare the gradient of every IR node.

mlipcpp-train-check dumps its batch inputs and d(loss)/d(node output) for
every GIR node. This script re-exports the same wrapper with torch.export,
runs the FX graph on that batch retaining every intermediate gradient, and
prints the nodes from the output backwards: the first mismatch walking back
from the loss is the op whose backward is wrong.

    python compare_node_grads.py --init init.pt --dump DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_pytorch.export_pet_gguf import example_export_args  # noqa: E402
from export_pytorch.export_pet_full import (  # noqa: E402
    PETFullModel, _unwrap_to_pet, get_model_params, load_local_model,
)


class Recorder(torch.fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.saved = {}

    def run_node(self, n):
        out = super().run_node(n)
        if isinstance(out, torch.Tensor) and out.is_floating_point() and out.requires_grad:
            out.retain_grad()
            self.saved[n.name] = out
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    d = Path(args.dump)
    meta = json.loads((d / "check.json").read_text())
    n, S, m = (meta["batch"][k] for k in ("n_atoms", "n_systems", "max_neighbors"))
    ld = lambda name, dt, shape: torch.from_numpy(np.fromfile(d / f"{name}.bin", dt).reshape(shape))

    inputs = (
        ld("species", np.int32, (n,)).long(),
        ld("neighbor_species", np.int32, (n, m)).long(),
        ld("edge_vectors", np.float32, (n, m, 3)),
        ld("padding_mask", np.float32, (n, m)) > 0.5,
        ld("reverse_neighbor_index", np.int32, (n * m,)).long(),
        ld("cutoff_values", np.float32, (n, m)),
    )
    W = ld("atom_weight", np.float32, (S, n))
    e_t = ld("energy_target", np.float32, (S,))
    f_t = ld("force_target", np.float32, (n, 3))

    pet = _unwrap_to_pet(load_local_model(args.init))
    p = get_model_params(pet)
    wrapper = PETFullModel(pet, n_atoms=n, max_neighbors=m, d_pet=p["d_pet"], forces=True,
                           cutoff=p["cutoff"], cutoff_width=p["cutoff_width"],
                           cutoff_function=p["cutoff_function"], nc_forces=True).eval()
    # export exactly as export_pet_gguf.py does (dynamic shapes, synthetic
    # example inputs), so FX node names match the GIR names; size the dims to
    # this batch so the exported module accepts it
    example, _, dynamic = example_export_args(n, m, p["cutoff"], forces=True)
    gm = torch.export.export(wrapper, example, dynamic_shapes=dynamic, strict=False).module()

    rec = Recorder(gm)
    out = rec.run(*inputs)
    loss_e = ((W @ out[:, 0] - e_t) ** 2).mean()
    loss_f = ((out[:, 1:] - f_t) ** 2).mean()
    loss = loss_e + loss_f
    loss.backward()
    print(f"loss torch {loss.item():.8g} mlipcpp {meta['loss']:.8g}")

    blob = np.fromfile(d / "node_grads.bin", np.float32)
    rows, missing = [], 0
    for e in meta.get("node_grads", []):
        t = rec.saved.get(e["name"])
        if t is None or t.grad is None:
            missing += 1
            continue
        g_t = t.grad.detach().contiguous().reshape(-1).double().numpy()
        g_c = blob[e["offset"]: e["offset"] + e["size"]].astype(np.float64)
        if g_t.size != g_c.size:
            rows.append((e["id"], e["op"], e["name"], float("nan"), f"size {g_c.size} vs {g_t.size}"))
            continue
        norm = np.linalg.norm(g_t)
        rel = np.linalg.norm(g_c - g_t) / norm if norm > 0 else np.linalg.norm(g_c)
        rows.append((e["id"], e["op"], e["name"], rel, f"|g| {norm:.3e}"))

    rows.sort(key=lambda r: -r[0])  # from the output backwards
    print(f"{len(rows)} nodes compared, {missing} without a torch counterpart")
    first_bad = None
    for r in rows:
        bad = not (r[3] <= args.tol)
        if bad and first_bad is None:
            first_bad = r
        if bad or r[0] % 10 == 0:
            print(f"{'BAD ' if bad else '    '}{r[0]:4d} {r[1]:<14} {r[2]:<28} rel {r[3]:.2e}  {r[4]}")
    print("first mismatch walking back from the loss:", first_bad)


if __name__ == "__main__":
    main()
