#!/usr/bin/env python3
"""fp64 parity of the PERIODIC torch PETLR (iris_torch.IrisTrunk + IrisLR with ewald_potentials, after M. Langer,
ipsum/PET-LR) against his JAX PETLR (`ref_petlr.py --x64`: his calculator's batch, jax-pme batched_tiled Ewald at
his frozen parameters): energy, forces (autograd from one positions leaf), charges, and for 3-D cells the virial
(autograd of a strain eps applied to positions and cell, x -> x (I + eps), = his sum_i r_a dE/dr_b + (h^T dE/dh)_ab).
It also checks the HOST-SIDE virial mlip.cpp will use (from the gradients of the graph inputs: positions, edge
vectors, real-space shifts S @ cell, k-vectors and the volume) against the strain virial.

    python check_pbc64.py PARAMS.npz CKPT_DIR FULL_X64.npz FRAMES.extxyz [--max N] [--num-k 1024]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from check_grad64 import build  # noqa: E402
from iris_torch import bisect_fermi  # noqa: E402
from torch_grid import ewald_inputs, ewald_params, periodic_geometry  # noqa: E402


def energy(cfg, trunk, lr, scale, pos, cell, Z, spin, Q, prm, hard=False, lr_on=True, taps=None, freeze_cut=False):
    """Total energy and IrisLR's aux; differentiable in pos and cell. taps: dict that receives the graph-input
    tensors (edge vectors, ewald shifts / kvec / volume) as leaves of the energy, for the host-side virial."""
    g = periodic_geometry(pos, cell, prm["pbc"], cfg, hard=hard)
    n, k = g["neighbor_index"].shape
    t = lambda x, dt=torch.float64: torch.as_tensor(x, dtype=dt)
    Zt = t(Z, torch.long)
    ns = Zt[g["neighbor_index"].reshape(-1)].reshape(n, k)
    ev = g["edge_vectors"]
    e = ewald_inputs(pos, cell, prm)
    if taps is not None:  # cut the graph at the inputs mlip.cpp feeds, keep their values
        ev = ev.detach().requires_grad_(True)
        for key in ("lr_shift", "kvec", "volume"):
            e[key] = e[key].detach().requires_grad_(True)
        taps.update(edge_vectors=ev, **{key: e[key] for key in ("lr_shift", "kvec", "volume")})
        freeze_cut = True  # (compared against a strain virial with the cutoffs frozen too)
    if freeze_cut:
        g["cutoff_values"] = g["cutoff_values"].detach()
    args = [Zt, ns, ev, g["padding_mask"], g["reverse"], g["cutoff_values"]]
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
    e_lr = lr(node, Zt, pos, g["neighbor_index"], g["padding_mask"], g["reverse"], ev, env,
              t(Q), t(mu_b), t(float(gapped)), m_round, top, bottom, torch.zeros(3, dtype=torch.float64), ewald=e)
    return (e_sr + e_lr).sum(), lr._aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("full"); ap.add_argument("frames")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--num-k", type=int, default=1024)
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--no-host-virial", action="store_true")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg, trunk, lr, scale = build(a.params, a.ckpt)
    zf = np.load(a.full)
    apf = zf["atom_ptr"]
    frames = read(a.frames, ":")
    nf = len(apf) - 1 if not a.max else min(a.max, len(apf) - 1)
    rows = {}
    add = lambda name, v: rows.setdefault(name, []).append(v)
    for f in range(nf):
        atoms = frames[f]
        Z = atoms.get_atomic_numbers(); spin = int(atoms.info.get("spin_multiplicity", 1))
        Q = float(atoms.info.get("charge", 0))
        prm = ewald_params(atoms.get_positions(), atoms.get_cell().array, atoms.pbc, num_k=a.num_k)
        A = slice(apf[f], apf[f + 1])
        P0 = torch.tensor(atoms.get_positions()); C0 = torch.tensor(atoms.get_cell().array)
        eps = torch.zeros(3, 3, requires_grad=True)
        pos = P0 + P0 @ eps; cell = C0 + C0 @ eps
        pos.retain_grad()
        E, aux = energy(cfg, trunk, lr, scale, pos, cell, Z, spin, Q, prm, hard=a.hard)
        gp, ge = torch.autograd.grad(E, [pos, eps])
        F = -gp.numpy()
        Fj, Ej = zf["forces"][A], zf["atom/energy"][A]
        add("energy |dE|/sum|E_i|", abs(float(E) - Ej.sum()) / np.abs(Ej).sum())
        add("forces max|dF|/max|F|", np.abs(F - Fj).max() / np.abs(Fj).max())
        if "atom/charges" in zf.files:
            add("charges max|dq| (e)", np.abs(aux["charges"].detach().numpy() - zf["atom/charges"][A]).max())
        if atoms.pbc.all():
            Wj = zf["virial"][f]
            add("virial max|dW|/max|W|", np.abs(ge.numpy() - Wj).max() / np.abs(Wj).max())
            if not a.no_host_virial:
                taps = {}
                P1 = P0.clone().requires_grad_(True)
                E1, _ = energy(cfg, trunk, lr, scale, P1, C0, Z, spin, Q, prm, hard=a.hard, taps=taps)
                keys = ["edge_vectors", "lr_shift", "kvec", "volume"]
                gs = torch.autograd.grad(E1, [P1] + [taps[k] for k in keys])
                g = dict(zip(["positions"] + keys, gs))
                W = P1.detach().T @ g["positions"]
                for key in ("edge_vectors", "lr_shift"):  # padded slots (zero vectors) carry no virial
                    x = taps[key].detach().reshape(-1, 3)
                    gk = torch.nan_to_num(g[key].reshape(-1, 3)) * (x.abs().sum(-1, keepdim=True) > 0)
                    W = W + x.T @ gk
                W = W - g["kvec"].T @ taps["kvec"].detach()  # k -> (I + eps)^-T k: d/d eps_ab = -sum_k g_a k_b
                W = W + torch.eye(3) * float(taps["volume"]) * float(g["volume"])
                eps2 = torch.zeros(3, 3, requires_grad=True)
                E2, _ = energy(cfg, trunk, lr, scale, P0 + P0 @ eps2, C0 + C0 @ eps2, Z, spin, Q, prm, hard=a.hard,
                               freeze_cut=True)
                (ge2,) = torch.autograd.grad(E2, [eps2])
                add("host virial vs strain virial (cutoffs frozen)", float((W - ge2).abs().max() / ge2.abs().max()))
        add("LR share |F_LR|/|F|", float("nan"))
        if f % 20 == 0:
            print(f"frame {f}: n={len(Z)} K={len(prm['k_grid'])} F err {rows['forces max|dF|/max|F|'][-1]:.2e}",
                  flush=True)
    print(f"torch fp64 (periodic) vs JAX fp64 ({a.full}), {nf} frames: max / median over frames")
    for name, v in rows.items():
        if np.all(np.isnan(v)):
            continue
        print(f"  {name:40s} {np.nanmax(v):.3g} / {np.nanmedian(v):.3g}")


if __name__ == "__main__":
    main()
