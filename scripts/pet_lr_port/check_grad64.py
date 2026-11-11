#!/usr/bin/env python3
"""fp64 derivative parity of the torch PETLR (iris_torch.IrisTrunk + IrisLR, after M. Langer, ipsum/PET-LR) against his
JAX PETLR: autograd forces from ONE positions leaf (full list -> differentiable adaptive grid -> selection -> trunk
edge vectors, cutoff factors, LR SR kernel, all-pair Coulomb, dipole) vs the forces of `ref_petlr.py --x64`
(`iris.pet.predict.predict`, jax.grad through his whole forward incl. the stop-gradients of `fermi_level`).
Open structures only (the periodic LR path is not in IrisLR yet).

    python check_grad64.py PARAMS.npz CKPT_DIR FULL_X64.npz FRAMES.extxyz [--max N] [--hard] [--split]

Prints, per frame aggregate (max / median): energy |dE| / sum|E_i|, forces max|dF| / max|F|, and |F_LR| / |F|
(the LR branch's share of the force, so a missing LR term cannot pass silently). --split also reports the
adaptive-cutoff (cutoff-Jacobian) share of the force.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from check_lr64 import load_lr  # noqa: E402
from check_torch64 import frames_info  # noqa: E402
from iris_to_gguf import PETLR_DEFAULTS, fill  # noqa: E402
from iris_torch import IrisLR, IrisTrunk, bisect_fermi  # noqa: E402
from petjax_convert import PETJAX_CONVERT, load_petjax_convert  # noqa: E402
from torch_grid import open_geometry  # noqa: E402


def build(params, ckpt):
    (_, cfg), = yaml.safe_load(open(f"{ckpt}/model/model.yaml")).items()
    cfg = {**PETLR_DEFAULTS, **cfg}
    flat = dict(np.load(params))
    sr = {k[len("params/sr/"):].replace("/", "."): v for k, v in flat.items() if k.startswith("params/sr/")}
    trunk = IrisTrunk(cfg)
    fill(trunk, sr, cfg, load_petjax_convert(PETJAX_CONVERT))
    lr = IrisLR(cfg)
    load_lr(lr, flat)
    scale = float(np.asarray(sr["energy_scale"]).reshape(()))
    return cfg, trunk.double().eval(), lr.double().eval(), scale


def energy(cfg, trunk, lr, scale, pos, Z, spin, Q, r0, hard=False, lr_on=True, freeze_cut=False):
    """Total energy (SR * scale + LR) and the aux of IrisLR, differentiable in pos."""
    g = open_geometry(pos, cfg, hard=hard)
    n, k = g["neighbor_index"].shape
    t = lambda x, dt=torch.float64: torch.as_tensor(x, dtype=dt)
    Zt = t(Z, torch.long)
    ns = Zt[g["neighbor_index"].reshape(-1)].reshape(n, k)
    cv = g["cutoff_values"].detach() if freeze_cut else g["cutoff_values"]
    args = [Zt, ns, g["edge_vectors"], g["padding_mask"], g["reverse"], cv]
    if trunk.conditioning:
        args += [torch.full((n,), cfg["max_charge"], dtype=torch.long), torch.full((n,), spin - 1, dtype=torch.long)]
    e_sr = trunk(*args) * scale
    if not lr_on:
        return e_sr.sum(), {}
    node, env = trunk._aux["node"], trunk._aux["env"]
    z0 = torch.zeros(n, dtype=torch.float64)
    mu_b, gapped, m_round, top, bottom = 0.0, False, z0, z0, z0
    if lr.ref in ("fermi", "both"):
        mu_b, gapped, m_round, top, bottom = bisect_fermi(lr, lr.scores(node).detach(), Q)
    e_lr = lr(node, Zt, pos, g["neighbor_index"], g["padding_mask"], g["reverse"], g["edge_vectors"], env,
              t(Q), t(mu_b), t(float(gapped)), m_round, top, bottom, t(r0))
    return (e_sr + e_lr).sum(), lr._aux


def forces(*a, **kw):
    pos = a[4]
    E, aux = energy(*a, **kw)
    (gp,) = torch.autograd.grad(E, pos)
    return float(E), -gp.numpy(), aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("full"); ap.add_argument("frames")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--hard", action="store_true", help="plain cutoff (seeded hard-cutoff models)")
    ap.add_argument("--split", action="store_true")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg, trunk, lr, scale = build(a.params, a.ckpt)
    zf = np.load(a.full)
    apf = zf["atom_ptr"]
    info = frames_info(a.frames)
    atoms_all = read(a.frames, ":")
    nf = len(apf) - 1 if not a.max else min(a.max, len(apf) - 1)
    rows = {}

    def add(name, v):
        rows.setdefault(name, []).append(v)

    for f in range(nf):
        Z, spin = info[f]
        atoms = atoms_all[f]
        Q = float(atoms.info.get("charge", 0))
        r0 = np.asarray(atoms.info["dipole_origin"], float) if "dipole_origin" in atoms.info else \
            atoms.get_center_of_mass()
        A = slice(apf[f], apf[f + 1])
        mk = lambda: torch.tensor(atoms.get_positions(), dtype=torch.float64, requires_grad=True)
        E, F, aux = forces(cfg, trunk, lr, scale, mk(), Z, spin, Q, r0, hard=a.hard)
        Fj = zf["forces"][A]
        Ej = zf["atom/energy"][A]
        add("energy |dE|/sum|E_i|", abs(E - Ej.sum()) / np.abs(Ej).sum())
        add("forces max|dF|/max|F|", np.abs(F - Fj).max() / np.abs(Fj).max())
        if "charges" in aux and "atom/charges" in zf.files:
            add("charges", np.abs(aux["charges"].detach().numpy() - zf["atom/charges"][A]).max()
                / max(np.abs(zf["atom/charges"][A]).max(), 1e-300))
        _, Fsr, _ = forces(cfg, trunk, lr, scale, mk(), Z, spin, Q, r0, hard=a.hard, lr_on=False)
        add("share |F_LR|/|F| (max over atoms)", np.abs(F - Fsr).max() / np.abs(F).max())
        if a.split and not a.hard:
            _, Ffz, _ = forces(cfg, trunk, lr, scale, mk(), Z, spin, Q, r0, freeze_cut=True)
            add("share |F_cut|/|F| (cutoff Jacobian)", np.abs(F - Ffz).max() / np.abs(F).max())
        if f % 20 == 0:
            print(f"frame {f}: n={len(Z)} F err {rows['forces max|dF|/max|F|'][-1]:.2e}", flush=True)
    print(f"torch fp64 autograd vs JAX fp64 ({a.full}), {nf} frames: max / median over frames")
    for name, v in rows.items():
        print(f"  {name:40s} {max(v):.3g} / {np.median(v):.3g}")


if __name__ == "__main__":
    main()
