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


def _pack(ii, jj, n, S=None):
    """Slots of a list (ii -> jj [, shifts S]) in an [n, k] layout and the reverse slot of each (j -> i, -S)."""
    k = max(int(np.bincount(ii, minlength=n).max()), 1) if len(ii) else 1
    slot = np.zeros(len(ii), int)
    cnt = np.zeros(n, int)
    for e in range(len(ii)):
        slot[e] = ii[e] * k + cnt[ii[e]]; cnt[ii[e]] += 1
    key = (lambda e: (int(ii[e]), int(jj[e]))) if S is None else (lambda e: (int(ii[e]), int(jj[e]), *S[e]))
    rkey = (lambda e: (int(jj[e]), int(ii[e]))) if S is None else (lambda e: (int(jj[e]), int(ii[e]), *(-S[e])))
    pair_slot = {key(e): slot[e] for e in range(len(ii))}
    rev = np.arange(n * k)
    for e in range(len(ii)):
        rev[slot[e]] = pair_slot.get(rkey(e), slot[e])
    return k, slot, rev


def periodic_geometry(pos, cell, pbc, cfg, hard=False):
    """open_geometry for a periodic cell (3-D or a slab): pos [n, 3] and cell [3, 3] (rows) leaves, the full list
    with lattice shifts (ASE primitive_neighbor_list, detached), R = pos[j] - pos[i] + S @ cell (differentiable in
    both, so a strain of both gives the virial), the adaptive grid and the selection as open_geometry."""
    from ase.neighborlist import primitive_neighbor_list
    n = pos.shape[0]
    cutoff = float(cfg["cutoff"])
    ii, jj, S = primitive_neighbor_list("ijS", np.asarray(pbc), cell.detach().numpy(), pos.detach().numpy(), cutoff)
    St = torch.as_tensor(S, dtype=pos.dtype)
    R = pos[jj] - pos[ii] + St @ cell
    r = torch.sqrt((R * R).sum(-1) + 1e-15)
    keep0 = (r < cutoff).detach().numpy()
    ii, jj, S, R, r = ii[keep0], jj[keep0], S[keep0], R[torch.from_numpy(keep0)], r[torch.from_numpy(keep0)]
    if hard or not cfg.get("num_neighbors_adaptive"):
        pc = torch.full_like(r, cutoff)
    else:
        a = adaptive_cutoffs_grid(torch.from_numpy(ii), r, n, float(cfg["num_neighbors_adaptive"]), cutoff,
                                  float(cfg.get("cutoff_width_adaptive", cfg["cutoff_width"])))
        pc = 0.5 * (a[ii] + a[jj])
    sel = (r <= pc).detach().numpy()
    c, nb, S = ii[sel], jj[sel], S[sel]
    R, pc = R[torch.from_numpy(sel)], pc[torch.from_numpy(sel)]
    k, slot, rev = _pack(c, nb, n, S)
    st = torch.from_numpy(slot)
    ev = torch.zeros(n * k, 3, dtype=pos.dtype).index_put((st,), R).reshape(n, k, 3)
    cv = torch.full((n * k,), cutoff, dtype=pos.dtype).index_put((st,), pc).reshape(n, k)
    pad = np.ones(n * k, bool); pad[slot] = False
    ni = np.zeros(n * k, int); ni[slot] = nb
    return dict(edge_vectors=ev, cutoff_values=cv, padding_mask=torch.from_numpy(pad.reshape(n, k)),
                reverse=torch.from_numpy(rev), neighbor_index=torch.from_numpy(ni.reshape(n, k)), k=k,
                n_selected=len(c))


def ewald_params(positions, cell, pbc, num_k=None, lr_wavelength=None, smearing=None, slab_margin=2.0):
    """His calculator's frozen Ewald parameters (`iris.pet.calculator.Calculator._freeze`, a2658e3) in numpy fp64:
    the effective cell (a slab's shrunk cell, jax-pme `shrink_2d_cell` of a slab `slab_margin` thicker), the
    wavelength from num_k on the effective cell, smearing 2 wavelength, cutoff 8 wavelength, and the halfspace k grid
    sized on the float32 effective cell (`to_lr`)."""
    cell = np.asarray(cell, np.float64); pbc = np.asarray(pbc, bool); positions = np.asarray(positions, np.float64)
    eff, thickness = cell, None
    if pbc.sum() == 2:
        k = int(np.argmax(~pbc))
        nrm = np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])
        nh = nrm / np.linalg.norm(nrm)
        z = positions @ nh
        thickness = float(z.max() - z.min()) + slab_margin
        ends = np.stack([np.zeros(3), thickness * nh])
        zz = ends @ nh
        h_cell = abs(np.dot(cell[k], nh))
        L_max = max(float(np.linalg.norm(cell[(k + 1) % 3])), float(np.linalg.norm(cell[(k + 2) % 3])))
        h_min = (zz.max() - zz.min()) + 1.5 * L_max
        if h_cell > h_min:
            eff = cell.copy(); eff[k] = cell[k] * (h_min / h_cell)
    if lr_wavelength is None:
        lr_wavelength = float((np.prod(np.linalg.norm(eff, axis=-1)) / (2 * num_k)) ** (1 / 3))
    eff32 = eff.astype(np.float32)
    ns = np.ceil(np.linalg.norm(eff32, axis=-1) / lr_wavelength)
    shape = tuple(int(x) for x in ns)
    fx, fy, fz = (np.fft.fftfreq(m) * m for m in shape)
    kx, ky, kz = (a.flatten() for a in np.meshgrid(fx, fy, fz, indexing="ij"))
    m = (kx > 0) | ((kx == 0) & (ky > 0)) | ((kx == 0) & (ky == 0) & (kz > 0))
    k_grid = np.stack([kx[m], ky[m], kz[m]], axis=1)
    return dict(wavelength=lr_wavelength, smearing=smearing or 2.0 * lr_wavelength, cutoff=8.0 * lr_wavelength,
                effective_cell=eff32, thickness=thickness, k_grid=k_grid, pbc=pbc)


def ewald_inputs(pos, cell, prm, skin=0.5):
    """The `ewald` dict of iris_torch.ewald_potentials from pos [n, 3] and cell [3, 3] (leaves; the effective cell's
    non-periodic row is a constant, as jax-pme's compose_cell), at the frozen parameters `prm` (ewald_params)."""
    from ase.neighborlist import primitive_neighbor_list
    n = pos.shape[0]
    pbc = prm["pbc"]
    pm = torch.as_tensor(pbc, dtype=pos.dtype)[:, None]
    cellc = cell * pm + torch.as_tensor(prm["effective_cell"], dtype=pos.dtype) * (1.0 - pm)
    rc = float(prm["cutoff"])
    ii, jj, S = primitive_neighbor_list("ijS", pbc, cell.detach().numpy(), pos.detach().numpy(), rc + skin,
                                       self_interaction=False)
    k, slot, _ = _pack(ii, jj, n, S)
    st = torch.from_numpy(slot)
    shift = torch.zeros(n * k, 3, dtype=pos.dtype).index_put((st,), torch.as_tensor(S, dtype=pos.dtype) @ cellc)
    keep = torch.zeros(n * k, dtype=pos.dtype); keep[st] = 1.0
    ni = np.zeros(n * k, int); ni[slot] = jj
    recip = 2.0 * math.pi * torch.linalg.inv(cellc).T
    kvec = torch.as_tensor(prm["k_grid"], dtype=pos.dtype) @ recip
    V = torch.abs(torch.linalg.det(cellc)).reshape(1)
    slab = float(pbc.sum() == 2)
    if slab:
        kk = int(np.argmax(~pbc))
        nrm = torch.linalg.cross(cellc[(kk + 1) % 3], cellc[(kk + 2) % 3])
        nh = nrm / torch.sqrt((nrm * nrm).sum())
        basis = torch.abs((cellc[kk] * nh).sum()).reshape(1)
    else:
        nh, basis = torch.zeros(3, dtype=pos.dtype), torch.zeros(1, dtype=pos.dtype)
    t = lambda x: torch.tensor([float(x)], dtype=pos.dtype)
    return dict(lr_index=torch.from_numpy(ni.reshape(n, k)), lr_shift=shift.reshape(n, k, 3),
                lr_keep=keep.reshape(n, k), kvec=kvec, volume=V, smearing=t(np.float32(prm["smearing"])),
                cutoff=t(np.float32(rc)), slab=t(slab), slab_normal=nh, slab_basis=basis)
