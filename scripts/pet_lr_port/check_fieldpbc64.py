#!/usr/bin/env python3
"""fp64: his field force E . Z0 (iris.pet.predict._field_forces, a98c2f6) on PERIODIC cells, rebuilt in torch the way
his iris.pet.bec.predict_bec does it, and checked against his JAX fp64 field_forces (ref_branch_cc.py --x64 refs on
bulk_f32_field) and, on open frames, against the closed form of check_field64.py.

His Born charge of the pass-1 charges q0 at a frozen Fermi level (frozen_fermi_level: mu / the residual correction out
of the gradient), with the unfolded displacement (every image of an atom moving together), contracted with the field:
    (E . Z0)_k = E q0_k + sum_{e: neighbour(e) = k} (-R_e . E) dQ/dR_e
               + d/dr_k [ sum_gathers sum_rows (-R_row . E) <lambda_row, y_row> ]
with Q = sum_i q0_i, R_e the raw edge vectors (centre -> neighbour, shifts included), y the outputs of every cross-atom
gather of a position-dependent quantity (here the trunk's reverse gathers of the edge messages; the neighbour-species
gather carries no position Jacobian, the LR side has no gather before q0), lambda = dQ/dy and the levers frozen.

    python check_fieldpbc64.py PARAMS.npz CKPT_DIR REF_X64.ref.npz FRAMES.extxyz [--max N] [--num-k 1024]
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
from check_torch64 import frames_info  # noqa: E402
from iris_torch import bisect_fermi  # noqa: E402
from torch_grid import ewald_inputs, ewald_params, open_geometry, periodic_geometry  # noqa: E402


def field_force(cfg, trunk, lr, scale, atoms, num_k):
    """(tapped E . Z0 [n, 3], closed-form E . Z0 on open frames else None) in torch fp64."""
    Z = atoms.get_atomic_numbers(); spin = int(atoms.info.get("spin_multiplicity", 1))
    Q = float(atoms.info.get("charge", 0))
    dt0 = torch.get_default_dtype()
    E = torch.as_tensor(np.asarray(atoms.info["field"], float), dtype=dt0)
    periodic = bool(atoms.pbc.any())
    pos = torch.tensor(atoms.get_positions(), dtype=dt0, requires_grad=True)
    t = lambda x, dt=dt0: torch.as_tensor(x, dtype=dt)
    Zt = t(Z, torch.long)

    def forward(pos, taps):
        if periodic:
            prm = ewald_params(atoms.get_positions(), atoms.get_cell().array, atoms.pbc, num_k=num_k)
            g = periodic_geometry(pos, torch.tensor(atoms.get_cell().array), prm["pbc"], cfg, hard=True)
            ew = ewald_inputs(pos, torch.tensor(atoms.get_cell().array), prm)
        else:
            g, ew = open_geometry(pos, cfg, hard=True), None
        n, k = g["neighbor_index"].shape
        ns = Zt[g["neighbor_index"].reshape(-1)].reshape(n, k)
        ev = g["edge_vectors"]
        args = [Zt, ns, ev, g["padding_mask"], g["reverse"], g["cutoff_values"]]
        if trunk.conditioning:
            args += [torch.full((n,), cfg["max_charge"], dtype=torch.long), torch.full((n,), spin - 1, dtype=torch.long)]
        trunk._gather_taps = taps
        trunk(*args)
        trunk._gather_taps = None
        node, env = trunk._aux["node"], trunk._aux["env"]
        z0 = torch.zeros(n, dtype=dt0)
        mu_b, gapped, m_round, top, bottom = 0.0, False, z0, z0, z0
        if lr.fermi:
            mu_b, gapped, m_round, top, bottom = bisect_fermi(lr, lr.scores(node, Zt).detach(), Q)
            m_round, top, bottom = m_round.to(dt0), top.to(dt0), bottom.to(dt0)
        lr(node, Zt, pos, g["neighbor_index"], g["padding_mask"], g["reverse"], ev, env, t(Q), t(mu_b),
           t(float(gapped)), m_round, top, bottom, torch.zeros(3), ewald=ew, field=E)
        return lr._aux["charges_pass1"], ev, g

    lr.frozen = True
    taps = []
    q0, ev, g = forward(pos, taps)
    Qs = q0.sum()
    lam = torch.autograd.grad(Qs, taps + [ev], retain_graph=True, allow_unused=True)
    lam_y, lam_R = lam[:-1], lam[-1]
    keep = (~g["padding_mask"]).to(dt0)
    lever = (-(ev.detach() * E).sum(-1) * keep)  # -R_row . E per slot [n, k]
    n = len(Z)
    # direct term of the raw edge vectors, to the neighbour atom of each slot
    zed = torch.zeros(n, 3, dtype=dt0)
    if lam_R is not None:
        lR = torch.nan_to_num(lam_R.detach()) * keep[..., None]  # (padded slots: zero vectors, 0 * inf)
        zed.index_add_(0, g["neighbor_index"].reshape(-1), (lever[..., None] * lR).reshape(-1, 3))
    s = sum(((lever[..., None] * torch.nan_to_num(ly.detach()) * keep[..., None]) * y).sum()
            for y, ly in zip(taps, lam_y) if ly is not None)
    (gs,) = torch.autograd.grad(s, pos) if torch.is_tensor(s) and s.requires_grad else (torch.zeros(n, 3),)
    if not torch.isfinite(gs).all():
        print(f"  (non-finite tap gradient on {int((~torch.isfinite(gs)).any(-1).sum())} atoms)", flush=True)
    tapped = (q0.detach()[:, None] * E[None, :] + zed + gs).numpy()
    closed = None
    if not periodic:  # check_field64.py's closed form: E q0 + d/dr [sum_j (E . r_j) q0_j] - (E . r_k) dQ/dr_k
        pos2 = torch.tensor(atoms.get_positions(), dtype=dt0, requires_grad=True)
        q0b, _, _ = forward(pos2, None)
        c = pos2.detach() @ E
        (gG,) = torch.autograd.grad((c * q0b).sum(), pos2, retain_graph=True)
        (gQ,) = torch.autograd.grad(q0b.sum(), pos2)
        closed = (q0b.detach()[:, None] * E[None, :] + gG - c[:, None] * gQ).numpy()
    lr.frozen = False
    return tapped, closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("ref"); ap.add_argument("frames")
    ap.add_argument("--max", type=int, default=0); ap.add_argument("--num-k", type=int, default=1024)
    ap.add_argument("--skip", default="", help="comma-separated frames to skip (fixed-neighbour overflow)")
    ap.add_argument("--f32", action="store_true", help="torch in float32 (the formulas' fp32 conditioning)")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32 if a.f32 else torch.float64)
    cfg, trunk, lr, scale = build(a.params, a.ckpt)
    if a.f32:
        trunk, lr = trunk.float(), lr.float()
    z = np.load(a.ref)
    apr = z["atom_ptr"]
    frames = read(a.frames, ":")
    nf = len(apr) - 1 if not a.max else min(a.max, len(apr) - 1)
    skip = {int(x) for x in a.skip.split(",") if x}
    for f in range(nf):
        if f in skip:
            continue
        at = frames[f]
        if cfg.get("fixed_neighbors") is not None:
            from ase.neighborlist import neighbor_list
            if np.bincount(neighbor_list("i", at, float(cfg["cutoff"]))).max() > int(cfg["fixed_neighbors"]):
                print(f"frame {f}: overflows fixed_neighbors {cfg['fixed_neighbors']} (his KnnToSample truncates), "
                      "skipped", flush=True)
                continue
        A = slice(apr[f], apr[f + 1])
        tapped, closed = field_force(cfg, trunk, lr, scale, at, a.num_k)
        Fh = z["field_forces"][A]
        msg = (f"frame {f} ({'periodic' if at.pbc.any() else 'open'}, n {len(at)}): tapped vs his "
               f"max|d|/max|E.Z0| {np.abs(tapped - Fh).max() / np.abs(Fh).max():.2e} (max|E.Z0| {np.abs(Fh).max():.3g})")
        if closed is not None:
            msg += f"; tapped vs closed form {np.abs(tapped - closed).max() / np.abs(closed).max():.2e}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
