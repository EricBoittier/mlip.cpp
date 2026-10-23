"""PyTorch reference energies, forces and stress of a plato_pet_lr checkpoint for the
mlip.cpp parity check (mlipcpp-plato check MODEL.gguf OUTDIR).

    python scripts/plato/plato_reference.py model.ckpt test.xyz OUTDIR [--rt model-rt.ckpt]

Writes OUTDIR/structures.xyz (the inputs, positions rounded to float32 so that every
precision sees the same numbers) and OUTDIR/reference.json: per structure, per dtype
(float32, float64) and per variant (full; no_les; no_mesh; sr = neither) the total
energy (eV, scaled + composition, as the exported model), forces and stress
(1/V dE/dstrain, Voigt xx yy zz yz xz xy) by autograd, and the mesh size.
With --rt the round-tripped checkpoint (gguf_to_plato_ckpt.py) is evaluated too and
must give bit-identical results.
"""

import argparse
import json
import math
from pathlib import Path

import ase.io
import metatomic.torch  # noqa: F401
import numpy as np
import torch
from metatomic.torch import ModelOutput, systems_to_torch
from metatrain.utils.io import load_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

D2 = [np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, -1.0, 1.0]), np.diag([-1.0, 1.0, -1.0])]


def structures(path):
    frames = ase.io.read(path, ":")
    by = {}
    for i, a in enumerate(frames):
        by.setdefault(a.info["label"], []).append(i)
    out = []
    cc = frames[by["CC"][0]].copy()
    out.append(("cc0", cc))
    for lab in ("AA", "PP", "CA", "PA", "CP"):
        if lab in by:
            out.append((f"{lab.lower()}0", frames[by[lab][len(by[lab]) // 2]].copy()))
    for k, R in enumerate(D2):  # group elements: identical energy
        a = cc.copy()
        a.positions = a.positions @ R.T
        a.cell = a.cell.array @ R.T
        out.append((f"cc0_d2_{k + 1}", a))
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    a = cc.copy()
    a.positions = a.positions @ q.T
    a.cell = a.cell.array @ q.T
    out.append(("cc0_rot", a))  # a generic rotation: approximate invariance only
    a = cc.copy()
    a.pbc = False
    a.cell = np.zeros((3, 3))
    out.append(("cc0_open", a))
    a = cc.copy()
    a.cell = a.cell.array @ np.array([[1.0, 0.0, 0.0], [0.13, 1.0, 0.0], [-0.07, 0.11, 1.0]])
    a.wrap()
    out.append(("cc0_tric", a))  # triclinic (no Nyquist symmetry of the mesh filter)
    out.append(("cc0_x8", cc.repeat((2, 2, 2))))
    for name, a in out:
        a.positions = a.positions.astype(np.float32).astype(np.float64)
        a.cell = a.cell.array.astype(np.float32).astype(np.float64)
        a.info = {"name": name}
        a.arrays = {k: v for k, v in a.arrays.items() if k in ("numbers", "positions")}
    return out


def evaluate(model, atoms, dtype, variant):
    model.use_mesh = variant in ("full", "no_les")
    model.use_les = variant in ("full", "no_mesh")
    s = systems_to_torch(atoms, dtype=dtype)
    periodic = bool(atoms.pbc.all())
    strain = torch.eye(3, dtype=dtype, requires_grad=True)
    pos = s.positions.detach().requires_grad_(True)
    from metatomic.torch import System
    sys = System(types=s.types, positions=pos @ strain, cell=s.cell @ strain, pbc=s.pbc)
    sys = get_system_with_neighbor_lists(sys, model.requested_neighbor_lists())
    E = model([sys], {"energy": ModelOutput(sample_kind="system")})["energy"].block().values.sum()
    gp, gs = torch.autograd.grad(E, [pos, strain])
    res = {"energy": E.item(), "forces": (-gp).tolist()}
    if periodic:
        V = abs(float(torch.det(s.cell)))
        st = (gs / V).detach()
        res["stress"] = [st[0, 0].item(), st[1, 1].item(), st[2, 2].item(),
                         st[1, 2].item(), st[0, 2].item(), st[0, 1].item()]
        res["mesh"] = model.coupling.mesh_size(s.cell)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("xyz")
    ap.add_argument("out", type=Path)
    ap.add_argument("--rt", help="round-tripped checkpoint: must reproduce the float32 results bit for bit")
    ap.add_argument("--only", nargs="*", help="structure names")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    strucs = [s for s in structures(args.xyz) if not args.only or s[0] in args.only]
    ase.io.write(args.out / "structures.xyz", [a for _, a in strucs], format="extxyz")
    models = {"float32": load_model(args.checkpoint).eval()}
    models["float64"] = load_model(args.checkpoint).eval().to(torch.float64)
    ref = {"checkpoint": str(args.checkpoint), "torch": torch.__version__, "structures": {}}
    for name, atoms in strucs:
        r = {"n_atoms": len(atoms)}
        for dt, m in models.items():
            r[dt] = {v: evaluate(m, atoms, getattr(torch, dt), v) for v in ("full", "no_les", "no_mesh", "sr")}
        ref["structures"][name] = r
        print(name, len(atoms), {dt: r[dt]["full"]["energy"] for dt in models},
              "mesh", r["float32"]["full"].get("mesh"), flush=True)
    if args.rt:
        rt = load_model(args.rt).eval()
        same = True
        for name, atoms in strucs:
            a = evaluate(rt, atoms, torch.float32, "full")
            b = ref["structures"][name]["float32"]["full"]
            same &= a == b
        ref["roundtrip_identical"] = same
        print("round-tripped checkpoint gives identical float32 results:", same)
    json.dump(ref, open(args.out / "reference.json", "w"))


if __name__ == "__main__":
    main()
