#!/usr/bin/env python3
"""fp64 parity of the torch conserving-charges branch UNDER A UNIFORM FIELD (his feat/conserving-charges a98c2f6
`field` input; iris_torch.IrisLR with field=) against his JAX fp64 (`ref_branch_cc.py --x64` on frames with
info["field"]): energy, the conservative forces -dE/dr (his `forces` minus his `field_forces`), dipole_flow, charges,
and his field force E . Z0 itself, built here from the frozen-Fermi-level pass-1 charges q0 on an OPEN structure as

    (E . Z0)_i = E q0_i + d/dr_i [sum_j (E . r_j) q0_j]  -  (E . r_i) d/dr_i [sum_j q0_j]

(his Born charge Z_i,ab = q_i delta_ab + sum_j (r_j - r_i)_a dq_j/dr_ib contracted with E: two VJPs of q0, with the
lever E . r_j held fixed). Periodic structures: his unfolded displacements (per-gather levers, iris.pet.bec) have no
such closed form; not checked here.

    python check_field64.py PARAMS.npz CKPT_DIR REF_X64.npz FRAMES.extxyz [--max N]
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
from torch_grid import open_geometry  # noqa: E402


def energy(cfg, trunk, lr, scale, pos, Z, spin, Q, r0, field):
    g = open_geometry(pos, cfg, hard=True)
    n, k = g["neighbor_index"].shape
    t = lambda x, dt=torch.float64: torch.as_tensor(x, dtype=dt)
    Zt = t(Z, torch.long)
    ns = Zt[g["neighbor_index"].reshape(-1)].reshape(n, k)
    args = [Zt, ns, g["edge_vectors"], g["padding_mask"], g["reverse"], g["cutoff_values"]]
    if trunk.conditioning:
        args += [torch.full((n,), cfg["max_charge"], dtype=torch.long), torch.full((n,), spin - 1, dtype=torch.long)]
    e_sr = trunk(*args) * scale
    node, env = trunk._aux["node"], trunk._aux["env"]
    z0 = torch.zeros(n, dtype=torch.float64)
    mu_b, gapped, m_round, top, bottom = 0.0, False, z0, z0, z0
    if lr.fermi:
        mu_b, gapped, m_round, top, bottom = bisect_fermi(lr, lr.scores(node, Zt).detach(), Q)
    e_lr = lr(node, Zt, pos, g["neighbor_index"], g["padding_mask"], g["reverse"], g["edge_vectors"], env,
              t(Q), t(mu_b), t(float(gapped)), m_round, top, bottom, t(r0), field=t(field))
    return (e_sr + e_lr).sum(), lr._aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("ref"); ap.add_argument("frames")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg, trunk, lr, scale = build(a.params, a.ckpt)
    z = np.load(a.ref)
    ap_ = z["atom_ptr"]
    err = z["error"] if "error" in z.files else np.zeros(len(ap_) - 1)
    info = frames_info(a.frames)
    frames = read(a.frames, ":")
    nf = len(ap_) - 1 if not a.max else min(a.max, len(ap_) - 1)
    rows = {}
    add = lambda k, v: rows.setdefault(k, []).append(v)
    for f in range(nf):
        if err[f] or frames[f].pbc.any():
            continue
        Z, spin = info[f]
        at = frames[f]
        Q = float(at.info.get("charge", 0))
        E = np.asarray(at.info["field"], float)
        r0 = np.asarray(at.info["dipole_origin"], float) if "dipole_origin" in at.info else at.get_center_of_mass()
        A = slice(ap_[f], ap_[f + 1])
        pos = torch.tensor(at.get_positions(), requires_grad=True)
        lr.frozen = False
        Et, aux = energy(cfg, trunk, lr, scale, pos, Z, spin, Q, r0, E)
        (gp,) = torch.autograd.grad(Et, pos)
        F = -gp.numpy()
        Fj = z["forces"][A] - z["field_forces"][A]  # his conservative part
        add("energy |dE|/|E|", abs(float(Et) - float(z["energy"][f])) / abs(float(z["energy"][f])))
        add("forces -dE/dr  max|dF|/max|F|", np.abs(F - Fj).max() / np.abs(Fj).max())
        add("charges max|dq|", np.abs(aux["charges"].detach().numpy() - z["charges"][A]).max())
        df = aux["dipole_flow_atoms"].sum(0).detach().numpy()
        add("dipole_flow max|d| / max|D|", np.abs(df - z["dipole_flow"][f]).max() / max(np.abs(z["dipole_flow"][f]).max(), 1e-300))
        # the field force from the frozen pass-1 charges
        lr.frozen = True
        pos2 = torch.tensor(at.get_positions(), requires_grad=True)
        _, aux2 = energy(cfg, trunk, lr, scale, pos2, Z, spin, Q, r0, E)
        q0 = aux2["charges_pass1"]
        Et_ = torch.as_tensor(E)
        c = (pos2.detach() @ Et_)  # E . r_j, held fixed
        (gG,) = torch.autograd.grad((c * q0).sum(), pos2, retain_graph=True)
        (gQ,) = torch.autograd.grad(q0.sum(), pos2)
        Ff = (q0.detach()[:, None] * Et_[None, :] + gG - c[:, None] * gQ).numpy()
        Fh = z["field_forces"][A]
        add("field force E.Z0 max|d| / max|E.Z0|", np.abs(Ff - Fh).max() / np.abs(Fh).max())
        add("share max|E.Z0| / max|F|", np.abs(Fh).max() / np.abs(z["forces"][A]).max())
        lr.frozen = False
    print(f"torch fp64 vs his JAX fp64 under a field ({a.ref}), {len(next(iter(rows.values())))} open frames: max / median")
    for k, v in rows.items():
        print(f"  {k:40s} {max(v):.3g} / {np.median(v):.3g}")


if __name__ == "__main__":
    main()
