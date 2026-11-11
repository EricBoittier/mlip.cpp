#!/usr/bin/env python3
"""Multi-component conserving charges WITHOUT fragment labels (after M. Langer, iris feat/conserving-charges a98c2f6):
the components are the connected components of the model's own neighbour graph (every pair within the hard cutoff:
his KnnToSample list / connected_components of the mutual edge graph; mlip.cpp's iris list), the component charges
Q_c come from mlip.cpp's R-only rule (iris_torch.component_charges_rule: largest-remainder integers of the share at
the structure's single Fermi level, or of the free charges), so the inputs stay R, Z, Q, spin.

    label   PARAMS.npz CKPT_DIR FRAMES.extxyz OUT.extxyz   write fragment / fragment_charge = those components and Q_c
            (torch fp64), for his oracle (ref_branch_cc.py reads them as labels)
    check   PARAMS.npz CKPT_DIR REF_X64.npz LABELLED.extxyz [--max N]   torch fp64 (per-component host bisection +
            the multi graph's forward) vs his JAX fp64 on the labelled frames: E, F, q
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from check_grad64 import build  # noqa: E402
from check_torch64 import frames_info  # noqa: E402
from iris_torch import bisect_fermi, component_charges_rule  # noqa: E402
from torch_grid import open_geometry  # noqa: E402


def components(atoms, cutoff):
    """0-based component per atom of the graph of pairs within `cutoff`, numbered by first atom (his rule)."""
    n = len(atoms)
    i, j = neighbor_list("ij", atoms, cutoff)
    _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
    _, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    return np.argsort(np.argsort(first))[inv].astype(int)


def trunk_node(cfg, trunk, pos, Z, spin):
    g = open_geometry(pos, cfg, hard=True)
    n, k = g["neighbor_index"].shape
    Zt = torch.as_tensor(Z, dtype=torch.long)
    ns = Zt[g["neighbor_index"].reshape(-1)].reshape(n, k)
    args = [Zt, ns, g["edge_vectors"], g["padding_mask"], g["reverse"], g["cutoff_values"]]
    if trunk.conditioning:
        args += [torch.full((n,), cfg["max_charge"], dtype=torch.long), torch.full((n,), spin - 1, dtype=torch.long)]
    e_sr = trunk(*args)
    return g, Zt, e_sr, trunk._aux["node"], trunk._aux["env"]


def rule(cfg, trunk, lr, atoms, Z, spin, Q):
    comp = components(atoms, float(cfg["cutoff"]))
    C = int(comp.max()) + 1
    with torch.no_grad():
        _, Zt, _, node, _ = trunk_node(cfg, trunk, torch.tensor(atoms.get_positions()), Z, spin)
        if lr.ref == "fermi":
            Qc = component_charges_rule(lr, comp, C, Q, s=lr.scores(node, Zt).numpy())
        else:
            Qc = component_charges_rule(lr, comp, C, Q, p=lr.free_charge(node)[:, 0].numpy())
    return comp, Qc


def host_multi(lr, s, comp, Qc):
    """Per-component bisection (bisect_fermi on each component's frozen scores at its Q_c)."""
    n, C = len(comp), len(Qc)
    mu_c, gap_c = np.zeros(C), np.zeros(C)
    m_round, top, bottom = (torch.zeros(n, dtype=torch.float64) for _ in range(3))
    for c in range(C):
        idx = np.flatnonzero(comp == c)
        mu, gapped, m, t, b = bisect_fermi(lr, s[idx], float(Qc[c]))
        mu_c[c], gap_c[c] = mu, float(gapped)
        m_round[idx], top[idx], bottom[idx] = m, t, b
    return mu_c, gap_c, m_round, top, bottom


def energy(cfg, trunk, lr, scale, pos, Z, spin, Q, r0, comp, Qc):
    g, Zt, e_sr, node, env = trunk_node(cfg, trunk, pos, Z, spin)
    n = len(comp)
    t = lambda x: torch.as_tensor(x, dtype=torch.float64)
    M = torch.zeros(n, len(Qc), dtype=torch.float64); M[np.arange(n), comp] = 1.0
    z0 = torch.zeros(n, dtype=torch.float64)
    mu_c, gap_c, m_round, top, bottom = np.zeros(len(Qc)), np.zeros(len(Qc)), z0, z0, z0
    if lr.ref == "fermi":
        mu_c, gap_c, m_round, top, bottom = host_multi(lr, lr.scores(node, Zt).detach(), comp, Qc)
    e_lr = lr(node, Zt, pos, g["neighbor_index"], g["padding_mask"], g["reverse"], g["edge_vectors"], env,
              t(Q), t(0.0), t(0.0), m_round, top, bottom, t(r0), comp=(M, t(Qc), t(mu_c), t(gap_c)))
    return (e_sr * scale + e_lr).sum(), lr._aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("label", "check"))
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg, trunk, lr, scale = build(a.params, a.ckpt)
    if a.mode == "label":
        frames = read(a.a, ":")
        info = frames_info(a.a)
        for f, at in enumerate(frames):
            Z, spin = info[f]
            comp, Qc = rule(cfg, trunk, lr, at, Z, spin, float(at.info.get("charge", 0)))
            at.arrays["fragment"] = comp.astype(float)
            at.arrays["fragment_charge"] = Qc[comp].astype(float)
            print(f"{f}: n {len(at)} C {len(Qc)} Q_c {Qc.astype(int).tolist()}")
        write(a.b, frames)
        return
    z = np.load(a.a)
    ap_ = z["atom_ptr"]
    err = z["error"] if "error" in z.files else np.zeros(len(ap_) - 1)
    frames = read(a.b, ":")
    info = frames_info(a.b)
    nf = len(ap_) - 1 if not a.max else min(a.max, len(ap_) - 1)
    rows = {}
    for f in range(nf):
        if err[f]:
            print(f"frame {f}: his prep raised, skipped"); continue
        at = frames[f]
        Z, spin = info[f]
        comp = np.rint(at.arrays["fragment"]).astype(int)
        C = int(comp.max()) + 1
        Qc = np.zeros(C)
        for c in range(C):
            Qc[c] = at.arrays["fragment_charge"][comp == c][0]
        A = slice(ap_[f], ap_[f + 1])
        pos = torch.tensor(at.get_positions(), requires_grad=True)
        r0 = np.asarray(at.info["dipole_origin"], float) if "dipole_origin" in at.info else at.get_center_of_mass()
        E, aux = energy(cfg, trunk, lr, scale, pos, Z, spin, float(at.info.get("charge", 0)), r0, comp, Qc)
        (gp,) = torch.autograd.grad(E, pos)
        F = -gp.numpy()
        rows.setdefault("energy |dE|/|E|", []).append(abs(float(E) - float(z["energy"][f])) / abs(float(z["energy"][f])))
        rows.setdefault("forces max|dF|/max|F|", []).append(np.abs(F - z["forces"][A]).max() / np.abs(z["forces"][A]).max())
        rows.setdefault("charges max|dq|", []).append(np.abs(aux["charges"].detach().numpy() - z["charges"][A]).max())
        q = aux["charges"].detach().numpy()
        rows.setdefault("component charge |sum_c q - Q_c| max", []).append(
            max(abs(q[comp == c].sum() - Qc[c]) for c in range(C)))
    print(f"torch fp64 (multi-component) vs his JAX fp64 ({a.a}): {len(next(iter(rows.values())))} frames, max / median")
    for k, v in rows.items():
        print(f"  {k:40s} {max(v):.3g} / {np.median(v):.3g}")


if __name__ == "__main__":
    main()
