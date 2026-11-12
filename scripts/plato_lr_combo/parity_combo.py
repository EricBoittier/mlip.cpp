#!/usr/bin/env python3
"""Independent torch autograd reference of the combo charge models of feat/plato-lr-combo (after M. Langer, ipsum/PET-LR;
~/data/plato-lr-scratch/combo/SPEC.md) on the analytic toy passes of tests/test_lr_combo.cpp, against the C++ dump
(MLIPCPP_COMBO_PARITY=out.json test_lr_combo "[combo]"). Same weights, independent code: q0 (reference + flows), the
long-range charges after lrc_qproj, E_lr, the forces by autograd, the metallicity and the Born charges (jacobian of
P = sum q x). The Fermi level: exact root (bisection + Newton, detached), then one live Newton step, whose first
derivatives are the implicit-function ones at the root.
    python parity_combo.py out.json"""
import json, math, sys
import torch

torch.set_default_dtype(torch.float64)
KE = 14.399645
d = json.load(open(sys.argv[1]))
Z = d["Z"]; n = len(Z)
ZL = [1, 8, 11, 17]; E_ = {z: i for i, z in enumerate(ZL)}
PHI = [0.3, -0.2, 1.1, 0.5]
THETA = [0.2, -0.3, -0.9, 0.4]; SFS = [0.1, -0.6, 1.2, -1.0]; SFC = [0.2, 0.5, -0.3, 0.1]; ETA = [1.0, 1.5, 0.5, 1.2]
KAP = [-3.0, -2.5, -3.5, -2.8, -2.2, -3.1, -2.9, -2.6, -3.3, -2.7, -3.0, -3.4, -2.4, -2.9, -3.2, -3.0]
K, T, SIG, RC = 3, 0.1, 0.9, 5.0
B = K + 2.0
ARMS = {"fermi": ("fermi", "element", 0, False), "thresh": ("threshold", "element", 0, False),
        "threshA": ("threshold", "atom", 0, False), "sfill": ("species_fill", "element", 0, False),
        "qeq": ("qeq", "element", 0, False), "E+fl": ("fukui", "element", 1, False), "A+fl2": ("fukui", "atom", 2, False),
        "fermi+fl": ("fermi", "element", 1, False), "E+flC": ("fukui", "element", 1, True), "qeq+fl": ("qeq", "element", 1, False)}
sp = torch.nn.functional.softplus


def dist(x, i, j):
    return torch.linalg.norm(x[i] - x[j])


def toy(x, spin, variant, k, L):  # tests/test_lr_combo.cpp Toy
    if variant == 0:
        c = lambda z: 0.8 if z == 11 else -0.7 if z == 17 else -0.5 if z == 8 else 0.25
        dd = lambda zi, zj: 0.05 * ((zi * 7 + zj * 3) % 5) - 0.1
    else:
        c = lambda z: 0.9 if z == 11 else -0.4 if z == 17 else 0.3 if z == 8 else -0.6
        dd = lambda zi, zj: 0.3 * ((zi * 5 + zj * 11) % 7) / 7.0 - 0.1 + k
    return torch.stack([c(Z[i]) + 0.03 * (spin - 1) + k + sum(dd(Z[i], Z[j]) * torch.exp(-dist(x, i, j) / L)
                                                            for j in range(n) if j != i) for i in range(n)])


def stair(xv):
    ks = torch.arange(-K, K, dtype=torch.float64) + 0.5
    s = torch.sigmoid((xv[:, None] - ks) / T)
    return s.sum(1) - K, (s * (1 - s)).sum(1) / T


def fermi_mu(s, c, target):
    with torch.no_grad():
        sd = s.detach(); lo = sd.min() - K - 1 - 40 * T; hi = sd.max() + K + 1 + 40 * T
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if (c * stair(sd - mid)[0]).sum() - target > 0: lo = mid
            else: hi = mid
        mu = 0.5 * (lo + hi)
        for _ in range(60):
            o, dn = stair(sd - mu)
            mu = mu + ((c * o).sum() - target) / (c * dn).sum()
    o, dn = stair(s - mu)
    return mu + ((c * o).sum() - target) / (c * dn).sum().detach()


def gk(r):
    return KE * torch.erf(r / (math.sqrt(2) * SIG)) / r


def fweights(x, mode, spin):
    if mode == "uniform": return torch.ones(n)
    phi = torch.tensor([PHI[E_[z]] for z in Z])
    a = toy(x, spin, 1, 0.0, 3.0) if mode == "atom" else 0.0
    return sp(phi + a) + 1e-6


def reference(x, ref, mode, Q, spin):
    qr = toy(x, spin, 0, 0.0, 2.0); r = Q - qr.sum(); M = torch.tensor(0.0)
    if ref == "fukui":
        f = fweights(x, mode, spin); return qr + f / f.sum() * r, M
    if ref in ("fermi", "threshold"):
        s = B * torch.tanh(toy(x, spin, 1, 0.15, 2.5) / B)
    if ref == "fermi":
        mu = fermi_mu(s, torch.ones(n), r); o, dn = stair(s - mu); return qr + o, dn.sum()
    if ref == "threshold":
        o, dn = stair(s - torch.tensor([THETA[E_[z]] for z in Z])); f = fweights(x, mode, spin)
        return qr + o + f / f.sum() * (r - o.sum()), dn.sum()
    if ref == "species_fill":
        s = B * torch.tanh(torch.tensor([SFS[E_[z]] for z in Z]) / B); c = sp(torch.tensor([SFC[E_[z]] for z in Z]))
        mu = fermi_mu(s, c, r); o, dn = stair(s - mu); return qr + c * o, (c * dn).sum()
    if ref == "qeq":
        J = sp(torch.tensor([ETA[E_[z]] for z in Z]))
        A = torch.diag(J + KE * math.sqrt(2 / math.pi) / SIG)
        Kij = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i != j: Kij = Kij + torch.nn.functional.one_hot(torch.tensor(i), n)[:, None].double() * torch.nn.functional.one_hot(torch.tensor(j), n)[None, :].double() * gk(dist(x, i, j))
        A = A + Kij
        Mk = torch.zeros(n + 1, n + 1); Mk[:n, :n] = A; Mk[:n, n] = 1; Mk[n, :n] = 1
        rhs = torch.cat([J * qr, torch.tensor([float(Q)])])
        return torch.linalg.solve(Mk, rhs)[:n], M


def flows(x, q, P, coulomb):
    for _ in range(P):
        phi = torch.stack([sum(gk(dist(x, i, j)) * q[j] for j in range(n) if j != i) for i in range(n)])
        dq = [torch.tensor(0.0) for _ in range(n)]
        for i in range(n):
            for j in range(i):
                r = dist(x, i, j)
                if not coulomb and r >= RC: continue
                kk = sp(torch.tensor(0.5 * (KAP[E_[Z[i]] * 4 + E_[Z[j]]] + KAP[E_[Z[j]] * 4 + E_[Z[i]]])))
                w = torch.erf(r / (math.sqrt(2) * SIG)) / r if coulomb else 1 - sstep(r - RC + 1)
                t = kk * w * (phi[j] - phi[i]); dq[i] = dq[i] + t; dq[j] = dq[j] - t
        q = q + torch.stack(dq)
    return q


def sstep(t):
    t = torch.clamp(t, 0.0, 1.0)
    return t ** 3 * (t * (t * 6 - 15) + 10)


def project(q, f, cl, w):  # lrc_qproj, separated clusters, element shares
    Kc = int(cl.max()) + 1
    Q0 = torch.zeros(Kc).index_add(0, cl, q); F = torch.zeros(Kc).index_add(0, cl, f)
    kf = torch.floor(Q0.detach()); u = (Q0 - kf - 0.5) / w + 0.5
    s = kf + sstep(u) - Q0
    dP = torch.where((u > 0) & (u < 1), 30 * u * u * (u - 1) ** 2, torch.zeros_like(u)) / w
    g = F * (1e-3 + dP)
    return q + f / F[cl] * (s - s.sum() * g / g.sum())[cl]


def model(x, arm, Q, spin):
    ref, mode, P, coul = ARMS[arm]
    q0, M = reference(x, ref, mode, Q, spin)
    q0 = flows(x, q0, P, coul)
    cl = torch.tensor([0, 0, 0, 0, 1])
    fe = sp(torch.tensor([PHI[E_[z]] for z in Z])) + 1e-6
    q = project(q0, fe, cl, 0.6)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if cl[i] == cl[j]: continue
            R = dist(x, i, j)
            if R <= 8.0: continue
            E = E + KE * q[i] * q[j] * sstep((R - 8.0) / 2.0) / R
    return q0, q, E, M


worst = {"q0": 0.0, "q": 0.0, "E": 0.0, "F": 0.0, "M": 0.0, "bec": 0.0}
for c in d["cases"]:
    x = torch.tensor(d["x"]).reshape(n, 3).requires_grad_(True)
    q0, q, E, M = model(x, c["arm"], c["Q"], c["spin"])
    F = -torch.autograd.grad(E, x)[0].reshape(-1)
    xf = torch.tensor(d["x"])
    P = lambda xx: (model(xx.reshape(n, 3), c["arm"], c["Q"], c["spin"])[0][:, None] * xx.reshape(n, 3)).sum(0)
    Zb = torch.autograd.functional.jacobian(P, xf)  # [3 (a), 3n]: dP_a/dx_(i, b)
    zb = Zb.reshape(3, n, 3).permute(1, 0, 2).reshape(-1)  # [n][a][b]
    errs = {"q0": (q0 - torch.tensor(c["q0"])).abs().max().item(), "q": (q - torch.tensor(c["q"])).abs().max().item(),
            "E": abs(float(E) - c["E"]), "F": (F - torch.tensor(c["F"])).abs().max().item(), "M": abs(float(M) - c["M"]),
            "bec": (zb - torch.tensor(c["bec"])).abs().max().item()}
    for k in worst: worst[k] = max(worst[k], errs[k])
    print(f"{c['arm']:9s} Q {c['Q']:+.0f} spin {c['spin']}: E {c['E']:+.6f} M {c['M']:.3f} " +
          " ".join(f"|d{k}| {v:.1e}" for k, v in errs.items()))
print("max over", len(d["cases"]), "cases:", {k: f"{v:.1e}" for k, v in worst.items()})
ok = all(worst[k] < 1e-11 for k in ("q0", "q", "E", "M")) and worst["F"] < 1e-10 and worst["bec"] < 1e-10
print("PARITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
