"""Differentiable torch fp64 transliteration of pet-jax's in-graph geometry for open structures (after M. Langer,
ipsum/PET-LR): the full neighbour list at `cutoff`, `get_adaptive_cutoffs` (the "grid" method, gradients through
everything, as `truncate_edges(no_shadow=False)`), pair cutoffs (a_i + a_j) / 2, the selection r <= pair cutoff,
and the packing into the trunk's [n, k] slots. One positions leaf feeds every distance, so autograd forces carry
the direct, the cutoff-Jacobian and the long-range terms together.

pet-jax c3f216c `petjax.select.get_adaptive_cutoffs` / `_select_edges` / `petjax.utils.cutoff_bump, safe_norm`.
"""
import math

import numpy as np
import torch


def cutoff_bump(r, cutoff, width):
    """petjax.utils.cutoff_bump: 1 for x <= 0, 0 for x >= 1, else the bump at clip(x, 1e-6, 1 - 1e-6)."""
    x = (r - (cutoff - width)) / width
    xs = torch.clamp(x, 1e-6, 1.0 - 1e-6)
    a = math.pi * xs
    bump = 0.5 * (1.0 + torch.tanh(torch.cos(a) / torch.sin(a)))
    one, zero = torch.ones_like(bump), torch.zeros_like(bump)
    return torch.where(x <= 0, one, torch.where(x >= 1, zero, bump))


def adaptive_cutoffs_grid(centers, r, n_atoms, num_neighbors, cutoff, width):
    """petjax.select.get_adaptive_cutoffs (probes arange(0.5, cutoff, width / 4)); r = safe_norm of the full list."""
    probes = torch.arange(0.5, cutoff, width / 4, dtype=r.dtype)
    P = probes.shape[0]
    w = cutoff_bump(r[None, :], probes[:, None], width)  # [P, E]
    eff = torch.zeros(n_atoms, P, dtype=r.dtype).index_add(0, centers, w.T)
    diff = eff - num_neighbors
    x = torch.linspace(0, 1, P, dtype=r.dtype)
    diff = diff + num_neighbors * x ** 3
    gi = (diff[:, 2:] - diff[:, :-2]) / 2
    gl = diff[:, 1:2] - diff[:, 0:1]
    gr = diff[:, -1:] - diff[:, -2:-1]
    width_t = torch.clamp(torch.abs(torch.cat([gl, gi, gr], dim=1)), min=1e-12)
    logw = -0.5 * (diff / width_t) ** 2
    ww = torch.exp(logw - logw.max(dim=-1, keepdim=True).values)
    ww = ww / ww.sum(dim=-1, keepdim=True)
    return ww @ probes


def open_geometry(pos, cfg, hard=False):
    """pos [n, 3] (a leaf, fp64) -> dict of the trunk / LR inputs on the selected, packed list (open boundaries).
    hard: plain cutoff (no adaptive selection), pair cutoffs = cutoff."""
    n = pos.shape[0]
    cutoff = float(cfg["cutoff"])
    with torch.no_grad():
        D = pos[None, :, :] - pos[:, None, :]
        d = torch.sqrt((D * D).sum(-1))
        ii, jj = torch.nonzero((d < cutoff) & ~torch.eye(n, dtype=torch.bool), as_tuple=True)
    R = pos[jj] - pos[ii]
    r = torch.sqrt((R * R).sum(-1) + 1e-15)  # safe_norm
    if hard or not cfg.get("num_neighbors_adaptive"):
        pc = torch.full_like(r, cutoff)
    else:
        a = adaptive_cutoffs_grid(ii, r, n, float(cfg["num_neighbors_adaptive"]), cutoff,
                                  float(cfg.get("cutoff_width_adaptive", cfg["cutoff_width"])))
        pc = 0.5 * (a[ii] + a[jj])
    sel = (r <= pc).detach().numpy()
    c, nb = ii.numpy()[sel], jj.numpy()[sel]
    R, pc = R[torch.from_numpy(sel)], pc[torch.from_numpy(sel)]
    k = max(int(np.bincount(c, minlength=n).max()), 1) if len(c) else 1
    slot = np.zeros(len(c), int)
    cnt = np.zeros(n, int)
    for e in range(len(c)):
        slot[e] = c[e] * k + cnt[c[e]]; cnt[c[e]] += 1
    pair_slot = {(int(c[e]), int(nb[e])): slot[e] for e in range(len(c))}  # open: one image per pair
    rev = np.arange(n * k)
    for e in range(len(c)):
        rev[slot[e]] = pair_slot.get((int(nb[e]), int(c[e])), slot[e])
    st = torch.from_numpy(slot)
    ev = torch.zeros(n * k, 3, dtype=pos.dtype).index_put((st,), R).reshape(n, k, 3)
    cv = torch.full((n * k,), cutoff, dtype=pos.dtype).index_put((st,), pc).reshape(n, k)
    pad = np.ones(n * k, bool); pad[slot] = False
    ni = np.zeros(n * k, int); ni[slot] = nb
    return dict(edge_vectors=ev, cutoff_values=cv, padding_mask=torch.from_numpy(pad.reshape(n, k)),
                reverse=torch.from_numpy(rev), neighbor_index=torch.from_numpy(ni.reshape(n, k)), k=k,
                n_selected=len(c))
