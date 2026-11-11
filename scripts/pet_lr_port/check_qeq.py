#!/usr/bin/env python3
"""QEq (iris.pet.charges.qeq_reference, a2658e3, after M. Langer) in mlip.cpp's three-stage form, checked in torch:
(1) the host cg (double, jax.scipy cg recurrences and stopping rule) on the operator graph's apply(x) and the chi of
the scores graph; (2) the graph with z = the cg solution as a leaf and lam = 0: d(sum E)/dz = the cotangent zbar;
(3) lam = the host cg on zbar, then the forces from sum E + sum lam r at fixed (z, lam). Compared with the eager
torch transliteration (IrisLR.qeq: his custom_linear_solve) and, when given, his JAX fp64 (ref_petlr.py --x64).
Also prints the cg iteration counts (forward, cotangent) in fp64 and fp32 and the stopping margins.

    python check_qeq.py PARAMS.npz CKPT_DIR FRAMES.extxyz [--ref FULL_X64.npz] [--max N] [--maxiter K]
                        [--cpp MLIPCPP.json]

--maxiter K overrides his maxiter 20 in both torch forms (a binding truncation: the three-stage form must still equal
his custom_linear_solve); --cpp compares an mlipcpp-eval run (made with the same --iris-qeq-maxiter) with the torch
fp64 three-stage result: fp32-sized differences if the truncated semantics match, O(1) if not.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from check_grad64 import build, energy  # noqa: E402
from check_torch64 import frames_info  # noqa: E402
from torch_grid import open_geometry  # noqa: E402


def host_cg(A, b, M, maxiter, tol):
    """jax.scipy.sparse.linalg.cg (x0 = 0, atol 0) in numpy double: (x, iterations, final |r|^2 / atol^2)."""
    x = np.zeros_like(b)
    atol2 = tol * tol * float(b @ b)
    r = b.copy()
    z = M(r)
    p, gamma, k = z, float(r @ z), 0
    while float(r @ r) > atol2 and k < maxiter:
        Ap = A(p)
        alpha = gamma / float(p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M(r)
        g2 = float(r @ z)
        p = z + (g2 / gamma) * p
        gamma, k = g2, k + 1
    return x, k, float(r @ r) / max(atol2, 1e-300)


def three_stage(cfg, trunk, lr, scale, pos, Z, spin, Q, r0):
    g = open_geometry(pos, cfg)
    n, k = g["neighbor_index"].shape
    dt = pos.dtype
    t = lambda x, d=dt: torch.as_tensor(x, dtype=d)
    Zt = t(Z, torch.long)
    ns = Zt[g["neighbor_index"].reshape(-1)].reshape(n, k)
    args = [Zt, ns, g["edge_vectors"], g["padding_mask"], g["reverse"], g["cutoff_values"]]
    if trunk.conditioning:
        args += [torch.full((n,), cfg["max_charge"], dtype=torch.long), torch.full((n,), spin - 1, dtype=torch.long)]
    e_sr = trunk(*args) * scale
    node, env = trunk._aux["node"], trunk._aux["env"]
    z0 = torch.zeros(n, dtype=dt)
    common = (Zt, pos, g["neighbor_index"], g["padding_mask"], g["reverse"], g["edge_vectors"], env, t(Q), t(0.0),
              t(0.0), z0, z0, z0, t(r0))

    def apply(x):
        with torch.no_grad():
            a, d = lr(None, *common, qeq_apply_x=torch.as_tensor(x, dtype=dt))
        return a.double().numpy(), d.double().numpy()

    chi = lr.electronegativity(node)[:, 0].detach().double().numpy()
    P = lambda v: v - v.mean()
    au, d = apply(np.full(n, Q / n))
    b = -P(au + chi)
    A0 = lambda v: P(apply(P(v))[0])
    M0 = lambda v: P(v / d)
    zs, it_f, m_f = host_cg(A0, b, M0, lr.qeq_maxiter, lr.qeq_tol)
    zt = torch.tensor(zs, dtype=dt, requires_grad=True)
    e_lr = lr(node, *common, qeq_z=zt, qeq_lambda=torch.zeros(n, dtype=dt))
    E = (e_sr + e_lr).sum()
    (zbar,) = torch.autograd.grad(E, zt, retain_graph=True)
    lam, it_b, m_b = host_cg(A0, zbar.double().numpy(), M0, lr.qeq_maxiter, lr.qeq_tol)
    e_lr2 = lr(node, *common, qeq_z=zt.detach(), qeq_lambda=torch.tensor(lam, dtype=dt))
    L = (e_sr + e_lr2).sum() + lr._aux["qeq_adjoint"].sum()
    (gp,) = torch.autograd.grad(L, pos)
    return float(E), -gp.double().numpy(), lr._aux["charges"].detach().double().numpy(), (it_f, it_b, m_f, m_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("frames")
    ap.add_argument("--ref"); ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--maxiter", type=int); ap.add_argument("--cpp")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg, trunk, lr, scale = build(a.params, a.ckpt)
    if a.maxiter is not None:
        lr.qeq_maxiter = a.maxiter
    cpp = __import__("json").load(open(a.cpp))["frames"] if a.cpp else None
    trunk32, lr32 = [__import__("copy").deepcopy(m).float() for m in (trunk, lr)]
    info = frames_info(a.frames)
    atoms_all = read(a.frames, ":")
    zf = np.load(a.ref) if a.ref else None
    nf = len(atoms_all) if not a.max else min(a.max, len(atoms_all))
    if zf is not None:
        nf = min(nf, len(zf["atom_ptr"]) - 1)
    rows = {}
    add = lambda k, v: rows.setdefault(k, []).append(v)
    for f in range(nf):
        Z, spin = info[f]
        at = atoms_all[f]
        Q = float(at.info.get("charge", 0))
        r0 = at.get_center_of_mass()
        mk = lambda d=torch.float64: torch.tensor(at.get_positions(), dtype=d, requires_grad=True)
        E3, F3, q3, its = three_stage(cfg, trunk, lr, scale, mk(), Z, spin, Q, r0)
        p = mk()
        Ee, aux = energy(cfg, trunk, lr, scale, p, Z, spin, Q, r0)
        (ge,) = torch.autograd.grad(Ee, p)
        Fe, qe = -ge.numpy(), aux["charges"].detach().numpy()
        ite = (lr._qeq_fwd_iterations, lr._qeq_bwd_iterations)
        p32 = mk(torch.float32)
        E32, aux32 = energy(cfg, trunk32, lr32, scale, p32, Z, spin, Q, r0)
        torch.autograd.grad(E32, p32)
        it32 = (lr32._qeq_fwd_iterations, lr32._qeq_bwd_iterations)
        add("3-stage vs eager: |dE|", abs(E3 - float(Ee)))
        add("3-stage vs eager: max|dF|", np.abs(F3 - Fe).max())
        add("3-stage vs eager: max|dq|", np.abs(q3 - qe).max())
        if zf is not None:
            A = slice(zf["atom_ptr"][f], zf["atom_ptr"][f + 1])
            add("3-stage vs JAX fp64: |dE|", abs(E3 - zf["atom/energy"][A].sum()))
            add("3-stage vs JAX fp64: max|dF|", np.abs(F3 - zf["forces"][A]).max())
            add("3-stage vs JAX fp64: max|dq|", np.abs(q3 - zf["atom/charges"][A]).max())
        add("max|F|", np.abs(Fe).max())
        if cpp is not None and "error" not in cpp[f]:
            x = cpp[f]
            add("mlip.cpp vs 3-stage: |dE|/sum..|E|", abs(x["energy"] - E3) / max(abs(E3), 1e-12))
            add("mlip.cpp vs 3-stage: max|dF|/max|F|", np.abs(np.reshape(x["forces"], (-1, 3)) - F3).max()
                / np.abs(F3).max())
            add("mlip.cpp vs 3-stage: max|dq|", np.abs(np.asarray(x["charges"]) - q3).max())
        print(f"frame {f:2d} n={len(Z):2d}: iterations fwd / cot: host-double {its[0]:2d} / {its[1]:2d} "
              f"(final |r|^2 / atol^2 {its[2]:.2g} / {its[3]:.2g}), eager fp64 {ite[0]:2d} / {ite[1]:2d}, "
              f"eager fp32 {it32[0]:2d} / {it32[1]:2d}", flush=True)
    print(f"{nf} frames: max / median")
    for k_, v in rows.items():
        print(f"  {k_:32s} {max(v):.3g} / {np.median(v):.3g}")


if __name__ == "__main__":
    main()
