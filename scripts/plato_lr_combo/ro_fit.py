#!/usr/bin/env python3
"""Ridge fit of the per-atom charge readout init on frozen C8q0 features (user: "fit it first, then train").
Target: per-structure residual r = E_DFT - E_model of the frozen variant (eV). Model:
  r ~ sum_i [a_i q_i + 1/2 J_i q_i^2] + nuisance,  a_i = w_a.h_i + beta_a,  J_i = softplus(b_i) ~ c0 + c_h.h_i (linearised),
with q the reference charges (qpass --combo), h the node-347 features, nuisance = element counts (unpenalised; the
readout must not absorb the composition baseline). Ridge lambda on the held-out split (val; grid). Then
b = (w_b, beta_b): beta_b = softplus^-1(max(c0, 1e-3)), w_b = c_h / sigmoid(beta_b). Graph units = eV / energy_scale.
Reports R^2 (train, held-out) of the linear fit and of the exact softplus readout with the mapped parameters.
    python ro_fit.py VARIANT_DIR DATA_DIR MODEL.gguf OUT.npz"""
import json, sys, struct
import numpy as np
from ase.io import read
import gguf

vd, dd, model, out = sys.argv[1:5]
RESTRICT = "--j-const" in sys.argv  # hardness J = softplus(beta_b) constant (W_b = 0, fitted): the linearised h-dependent J does not map onto softplus
es = float(gguf.GGUFReader(model).fields["pet.energy_scale"].parts[-1][0])


def feats(path):
    b = open(path, "rb").read(); o = 0; L = []
    while o < len(b):
        ne = struct.unpack_from("<4q", b, o); o += 32
        n = ne[0] * ne[1] * ne[2] * ne[3]
        L.append(np.frombuffer(b, np.float32, n, o).reshape(ne[1], ne[0]).astype(np.float64)); o += 4 * n
    return L


def load(split):
    H = feats(f"{vd}/{split}.feats.bin")
    ev = json.load(open(f"{vd}/{split}.eval.json"))["frames"]
    qs = json.load(open(f"{vd}/{split}.q.json"))["frames"]
    fr = read(f"{dd}/{split}.xyz", ":")
    assert len(H) == len(ev) == len(qs) == len(fr), (len(H), len(ev), len(qs), len(fr))
    rows, Zc, r, Q = [], [], [], []
    for h, e, q, a in zip(H, ev, qs, fr):
        q = np.asarray(q["q"], float); assert h.shape[0] == len(q) == len(a)
        rows.append(np.concatenate([q @ h, [q.sum()]] + ([] if RESTRICT else [0.5 * (q * q) @ h]) + [[0.5 * (q * q).sum()]]))
        Zc.append(np.bincount(a.numbers, minlength=119)); r.append(a.get_potential_energy() - e["energy"]); Q.append(a.info["charge"])
    return np.array(rows), np.array(Zc, float), np.array(r), np.array(Q), H, qs, ev, fr


Xt, Ct, rt, Qt, Ht, qt, evt, frt = load("train")
Xh, Ch, rh, Qh, Hh, qh, evh, frh = load("heldout")
keep = Ct.sum(0) > 0; Ct, Ch = Ct[:, keep], Ch[:, keep]
d = Xt.shape[1] - 2 if RESTRICT else (Xt.shape[1] - 2) // 2
mu, sd = Xt.mean(0), Xt.std(0) + 1e-8


def fit(lam):
    A = np.hstack([(Xt - mu) / sd, Ct]); p = Xt.shape[1]
    P = np.zeros(A.shape[1]); P[:p] = lam * len(rt)
    w = np.linalg.solve(A.T @ A + np.diag(P) + 1e-9 * np.eye(A.shape[1]), A.T @ rt)
    return w, p


def pred(w, p, X, C):
    return np.hstack([(X - mu) / sd, C]) @ w


def r2(y, yh):
    return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()


res = {}
for lam in 10.0 ** np.arange(-6, 3):
    w, p = fit(lam); res[lam] = (np.mean((rh - pred(w, p, Xh, Ch)) ** 2), w)
best = min(res, key=lambda k: res[k][0])
for a_ in sys.argv:
    if a_.startswith("--lam="): best = float(a_.split("=")[1]); res[best] = (np.mean((rh - pred(fit(best)[0], Xt.shape[1], Xh, Ch)) ** 2), fit(best)[0])
w, p = res[best][1], Xt.shape[1]
# baseline: nuisance only (what the readout adds over composition)
w0 = np.linalg.lstsq(Ct, rt, rcond=None)[0]
coef = w[:p] / sd; c_int = -(mu / sd) @ w[:p]  # the centring constant goes to the nuisance (counts sum >= 1)
wa, ba = coef[:d], coef[d]
ch, c0 = (np.zeros(d), coef[d + 1]) if RESTRICT else (coef[d + 1:2 * d + 1], coef[2 * d + 1])
J0 = max(c0, 1e-3); bb = np.log(np.expm1(J0)); wb = ch / (1 / (1 + np.exp(-bb)))
W = np.stack([wa, wb]) / es; b = np.array([ba, bb])
b_graph = np.array([ba / es, np.log(np.expm1(J0 / es))]); W_graph = np.stack([wa / es, ch / es / (1 / (1 + np.exp(-b_graph[1])))])


def exact(H, qs):  # the mapped readout with the real softplus, eV
    E = []
    for h, q in zip(H, qs):
        q = np.asarray(q["q"], float); a = h @ wa + ba; bg = h @ W_graph[1] + b_graph[1]
        E.append(((a * q) + 0.5 * es * np.logaddexp(0, bg) * q * q).sum())
    return np.array(E)


Et, Eh = exact(Ht, qt), exact(Hh, qh)
cn = np.linalg.lstsq(Ct, rt - Et, rcond=None)[0]  # nuisance refit for the exact readout
out_j = {"variant": vd, "j_const": RESTRICT, "energy_scale": es, "lambda": best, "n_train": len(rt), "n_heldout": len(rh),
         "r2_linear": {"train": r2(rt, pred(w, p, Xt, Ct)), "heldout": r2(rh, pred(w, p, Xh, Ch))},
         "r2_nuisance_only": {"train": r2(rt, Ct @ w0), "heldout": r2(rh, Ch @ w0)},
         "r2_exact_softplus": {"train": r2(rt, Et + Ct @ cn), "heldout": r2(rh, Eh + Ch @ cn)},
         "mae_meV_per_struct": {"nuisance_only_heldout": float(np.abs(rh - Ch @ w0).mean() * 1e3),
                                "readout_heldout": float(np.abs(rh - Eh - Ch @ cn).mean() * 1e3)},
         "J0_eV_per_e2": J0, "c0_raw": float(c0), "lam_grid_mse": {f"{k:g}": float(v[0]) for k, v in res.items()},
         "by_Q_heldout_mean_residual_meV": {int(qq): [float(rh[Qh == qq].mean() * 1e3), float((rh - Eh - Ch @ cn)[Qh == qq].mean() * 1e3), int((Qh == qq).sum())] for qq in np.unique(Qh)}}
np.savez(out, W=W_graph.astype(np.float32), b=b_graph.astype(np.float32))
json.dump(out_j, open(out.replace(".npz", ".json"), "w"), indent=1)
print(json.dumps(out_j, indent=1))
