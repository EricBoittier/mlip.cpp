#!/usr/bin/env python3
"""Torch reference of the lrc graph-mode Fukui arms (U / E / A / A+- / AC) on the analytic toy charge and Fukui passes
of tests/test_lr_combined.cpp, against the C++ dump (MLIPCPP_FUKUI_PARITY=out.json test_lr_combined "[fukui]").
Same weights, independent code: charges (q0 and the long-range charges after the plateau projection), the long-range
energy and the forces by autograd (two clusters, no weak link, Coulomb switched 8 -> 10 A).
    python parity_toy.py out.json"""
import json, sys
import torch

torch.set_default_dtype(torch.float64)
KE = 14.399645
d = json.load(open(sys.argv[1]))
Z = d["Z"]
n = len(Z)
FZ = {1: 0.3, 8: -0.2, 11: 1.1, 17: 0.5}            # graph_spec fukui_phi
FZm = {1: 0.1, 8: 0.4, 11: 0.2, 17: -0.3}           # fukui_phi_minus (A+-)
ARMS = {"U": ("uniform", False, .05), "E": ("element", False, .05), "A": ("atom", False, .05), "A+-": ("atom_pm", False, .05),
        "A+-hard": ("atom_pm", False, 0.0), "UC": ("uniform", True, .05), "EC": ("element", True, .05), "AC": ("atom", True, .05)}


def dist(x, i, j):
    return torch.linalg.norm(x[i] - x[j])


def toyq(x, Q, spin):  # ToyQ + 0.03 (spin - 1), charge pass at Q = 0 (qpass_neutral)
    a = lambda z: 0.8 if z == 11 else -0.7 if z == 17 else -0.5 if z == 8 else 0.25
    b = lambda zi, zj: 0.05 * ((zi * 7 + zj * 3) % 5) - 0.1
    return torch.stack([a(Z[i]) + 0.1 * 0.0 + 0.03 * (spin - 1) + sum(b(Z[i], Z[j]) * torch.exp(-dist(x, i, j) / 2)
                                                                     for j in range(n) if j != i) for i in range(n)])


def toya(x, Q, spin, k):
    c = lambda z: 0.9 if z == 11 else -0.4 if z == 17 else 0.3 if z == 8 else -0.6
    dd = lambda zi, zj: 0.3 * ((zi * 5 + zj * 11) % 7) / 7.0 - 0.1 + k
    return torch.stack([c(Z[i]) + 0.05 * 0.0 + 0.02 * (spin - 1) + k + sum(dd(Z[i], Z[j]) * torch.exp(-dist(x, i, j) / 3)
                                                                          for j in range(n) if j != i) for i in range(n)])


def sstep(t):
    t = torch.clamp(t, 0.0, 1.0)
    return t ** 3 * (t * (t * 6 - 15) + 10)


def plateau(L, w):
    k = torch.floor(L.detach())
    return k + sstep((L - k - 0.5) / w + 0.5)


def fukui(x, mode, spin, minus=False):
    if mode == "uniform":
        return torch.ones(n)
    phi = torch.tensor([(FZm if minus else FZ)[z] for z in Z])
    a = toya(x, 0.0, spin, 0.35 if minus else 0.0) if mode in ("atom", "atom_pm") else 0.0
    return torch.nn.functional.softplus(phi + a) + 1e-6


def project(q, f, cl, w, resid, Q):  # separated clusters (each group alone, probability 1)
    K = int(cl.max()) + 1
    Q0 = torch.zeros(K).index_add(0, cl, q); F = torch.zeros(K).index_add(0, cl, f)
    s = plateau(Q0, w) - Q0
    u = (Q0 - torch.floor(Q0.detach()) - 0.5) / w + 0.5
    dP = torch.where((u > 0) & (u < 1), 30 * u * u * (u - 1) ** 2, torch.zeros_like(u)) / w
    g = F * (1e-3 + dP)
    T = ((Q - q.sum()) if resid else 0.0) - s.sum()
    return q + f / F[cl] * (s + T * g / g.sum())[cl]


def model(x, arm, Q, spin):
    mode, cluster, pmw = ARMS[arm]
    qr = toyq(x, Q, spin)
    cl = torch.tensor([0, 0, 0, 0, 1])  # Na + water | Cl (window 5 / 6 A)
    fp = fukui(x, mode, spin)
    r = Q - qr.sum()
    if cluster:
        q0 = project(qr, fp, cl, 0.6, True, Q)
    else:
        w = fp / fp.sum()
        if mode == "atom_pm":
            fm = fukui(x, mode, spin, True)
            if pmw > 0:
                s = sstep(r / pmw + 0.5)
            else:
                s = (r > 0).to(r.dtype)
            w = (1 - s) * w + s * fm / fm.sum()
        q0 = qr + w * r
    fe = torch.ones(n) if mode == "uniform" else torch.nn.functional.softplus(torch.tensor([FZ[z] for z in Z])) + 1e-6
    q = q0 if cluster else project(q0, fe, cl, 0.6, False, Q)  # lrc_qproj with element shares (AC: not applied again)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if cl[i] == cl[j]:
                continue
            R = dist(x, i, j)
            if R <= 8.0:
                continue
            E = E + KE * q[i] * q[j] * sstep((R - 8.0) / 2.0) / R
    return q0, q, E


worst = {"q0": 0.0, "q": 0.0, "E": 0.0, "F": 0.0}
for c in d["cases"]:
    x = torch.tensor(d["x"]).reshape(n, 3).requires_grad_(True)
    q0, q, E = model(x, c["arm"], c["Q"], c["spin"])
    F = -torch.autograd.grad(E, x)[0].reshape(-1)
    errs = {"q0": (q0 - torch.tensor(c["q0"])).abs().max().item(), "q": (q - torch.tensor(c["q"])).abs().max().item(),
            "E": abs(E.item() - c["E"]), "F": (F - torch.tensor(c["F"])).abs().max().item()}
    for k in worst:
        worst[k] = max(worst[k], errs[k])
    print(f"{c['arm']:8s} Q {c['Q']:+.0f} spin {c['spin']}: E {c['E']:+.6f}  |dq0| {errs['q0']:.1e} |dq| {errs['q']:.1e} "
          f"|dE| {errs['E']:.1e} |dF| {errs['F']:.1e}")
print("max over", len(d["cases"]), "cases:", {k: f"{v:.1e}" for k, v in worst.items()})
ok = worst["q0"] < 1e-12 and worst["q"] < 1e-12 and worst["E"] < 1e-12 and worst["F"] < 1e-11
print("PARITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
