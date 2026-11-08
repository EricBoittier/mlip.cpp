#!/usr/bin/env python3
"""Prefit of the lrc atomic multipoles (R2 of the ladder): element-pair x radial-channel coefficients of the atomic
dipoles and traceless quadrupoles (runtime/lr_combined.h, lrc_mp*), fitted to DFT monomer moments with the stage-1
charges held fixed, then frozen in training.

For each monomer (merge-aug set A, train side only: train_aug of ~/data/lr-augment/split.json), about its centre of mass:
  mu_DFT    = sum_i q_i r_i + sum_i mu_i(theta)
  Theta_DFT = sum_i q_i B(r_i) + sum_i [3/2 (r_i mu_i^T + mu_i r_i^T) - (r_i . mu_i) I] + sum_i Theta_i(tau)
B(r) = 3/2 r r^T - 1/2 r^2 I (Buckingham), q the stage-1 charges (charge pass at Q = 0 + Fukui to the monomer charge).
Both are linear in (theta, tau): one weighted ridge solve (scales 0.05 e A and 0.15 e A^2, the kernel-lr prefit's), the
ridge strength chosen on a held-out 10 % of the training monomers. Reports the dipole / quadrupole errors with charges
only and with the fitted moments, on that 10 % and on the set A val monomers (heldout side; never fitted).

    python run.py prefit_mp.py MONOMERS.jsonl CHARGE OUT.json [--rc 4.5 --width 0.6 --centers 1.0,2.0,3.0,4.0]
CHARGE: a stage-1 prefix (CHARGE.pt + CHARGE_fukui.json, PyTorch), or QPASS.json:KEYS.json (mlipcpp-qpass output for a
monomer xyz and the matching "id:which" keys), e.g. for a GGUF charge pass such as head_q0."""
import argparse
import json
import os
import sys

import numpy as np
import torch

BOHR = 0.529177210903


def radial(r, centers, width, rc):
    fc = np.where(r < rc, 0.5 * (1 + np.cos(np.pi * r / rc)), 0.0)
    return np.exp(-0.5 * ((r[..., None] - centers) / width) ** 2) * fc[..., None]


def charges(model, phi, Z, R, Q):
    from metatomic.torch import ModelOutput, System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
    s = System(types=torch.tensor(Z, dtype=torch.int32), positions=torch.tensor(R, dtype=torch.float32),
               cell=torch.zeros(3, 3), pbc=torch.zeros(3, dtype=torch.bool))
    s = get_system_with_neighbor_lists(s, model.requested_neighbor_lists())
    # a water batched alongside (the PyTorch frame attention fails on a batch without any neighbour pair, e.g. a single ion)
    w = System(types=torch.tensor([8, 1, 1], dtype=torch.int32),
               positions=torch.tensor([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], dtype=torch.float32),
               cell=torch.zeros(3, 3), pbc=torch.zeros(3, dtype=torch.bool))
    w = get_system_with_neighbor_lists(w, model.requested_neighbor_lists())
    with torch.no_grad():
        raw = model([s, w], {"energy": ModelOutput(per_atom=True)})["energy"].block().values[: len(Z), 0].double().numpy()
    f = np.log1p(np.exp(phi[np.asarray(Z)])) + 1e-6
    return raw + f * (Q - raw.sum()) / f.sum()


def design(Z, R, com, zi, centers, width, rc):
    """rows: mu (3) and Theta (9) of the molecule; columns: theta (E*E*K) then tau (E*E*K)"""
    E, K = len(zi), len(centers)
    n = len(Z)
    P = E * E * K
    A = np.zeros((12, 2 * P))
    d = R[None, :, :] - R[:, None, :]  # x_j - x_i
    r = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(r, np.inf)
    g = radial(np.where(np.isinf(r), 1e9, r), centers, width, rc)  # n x n x K
    for i in range(n):
        ei = zi.get(int(Z[i]))
        if ei is None:
            continue
        ri = R[i] - com
        for j in range(n):
            ej = zi.get(int(Z[j]))
            if ej is None or j == i or r[i, j] >= rc:
                continue
            u = d[i, j] / r[i, j]
            off = (ei * E + ej) * K
            for k in range(K):
                v = g[i, j, k] * u  # d mu_i / d theta
                A[0:3, off + k] += v
                T = 1.5 * (np.outer(ri, v) + np.outer(v, ri)) - np.dot(ri, v) * np.eye(3)
                A[3:12, off + k] += T.ravel()
                Bq = g[i, j, k] * (1.5 * np.outer(u, u) - 0.5 * np.eye(3))
                A[3:12, P + off + k] += Bq.ravel()
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("monomers"); ap.add_argument("charge"); ap.add_argument("out")
    ap.add_argument("--rc", type=float, default=4.5); ap.add_argument("--width", type=float, default=0.6)
    ap.add_argument("--centers", default="1.0,2.0,3.0,4.0")
    ap.add_argument("--split", default=os.path.expanduser("~/data/lr-augment/split.json"))
    a = ap.parse_args()
    from ase.data import atomic_masses
    centers = np.array([float(c) for c in a.centers.split(",")])
    if a.charge.endswith(".json") or ".json:" in a.charge:
        qj, kj = a.charge.split(":")
        pre = dict(zip(json.load(open(kj)), [np.array(f["q"]) for f in json.load(open(qj))["frames"]]))
        model = phi = None
    else:
        pre = None
        model = torch.load(a.charge + ".pt", weights_only=False, map_location="cpu")["model"].eval()
        fk = json.load(open(a.charge + "_fukui.json"))
        phi = np.zeros(128); phi[np.array(fk["Z"])] = fk["phi"]
    side = {}
    sp = json.load(open(a.split))
    for p in sp["pairs"]:
        if p.get("source", "").startswith("setA") or str(p.get("pair_id", "")).startswith("A-"):
            for sid in p.get("scan_ids", []) or [p["pair_id"]]:
                side[str(sid)] = p["side"]
    mons = [json.loads(l) for l in open(a.monomers)]
    mons = [m for m in mons if m.get("conv", True) and m.get("mu_au") is not None and m.get("theta_au") is not None]
    rows = {"train": [], "val": []}
    for m in mons:
        s = side.get(m["id"], "train_aug" if m["id"].startswith("A-train") else "heldout")
        rows["train" if s == "train_aug" else "val"].append(m)
    Zall = sorted({int(z) for m in rows["train"] for z in __import__("ase").Atoms(m["sym"]).numbers})
    zi = {z: k for k, z in enumerate(Zall)}
    from ase import Atoms

    def build(ms):
        As, bs, base = [], [], []
        for m in ms:
            at = Atoms(m["sym"], positions=m["R"])
            Z, R = at.numbers, np.array(m["R"], float)
            com = (atomic_masses[Z][:, None] * R).sum(0) / atomic_masses[Z].sum()
            q = pre[f"{m['id']}:{m['which']}"] if pre is not None else charges(model, phi, Z, R, float(m["charge"]))
            rr = R - com
            mu_q = (q[:, None] * rr).sum(0)
            th_q = sum(q[i] * (1.5 * np.outer(rr[i], rr[i]) - 0.5 * rr[i] @ rr[i] * np.eye(3)) for i in range(len(Z)))
            mu = np.array(m["mu_au"]) * BOHR
            th = np.array(m["theta_au"]) * BOHR ** 2
            As.append(design(Z, R, com, zi, centers, a.width, a.rc))
            bs.append(np.concatenate([mu - mu_q, (th - th_q).ravel()]))
            base.append(np.concatenate([mu, th.ravel()]))
        return np.array(As), np.array(bs), np.array(base)

    rng = np.random.default_rng(7)
    tr = rows["train"]
    perm = rng.permutation(len(tr))
    nh = len(tr) // 10
    hold, fit = [tr[i] for i in perm[:nh]], [tr[i] for i in perm[nh:]]
    Af, bf, _ = build(fit); Ah, bh, _ = build(hold); Av, bv, _ = build(rows["val"])
    w = np.concatenate([np.full(3, 1 / 0.05), np.full(9, 1 / 0.15)])

    def solve(A, b, lam):
        X = (A * w[None, :, None]).reshape(-1, A.shape[-1]); y = (b * w[None]).ravel()
        return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

    def err(A, b, p):
        res = b - A @ p if p is not None else b
        return float(np.linalg.norm(res[:, :3], axis=1).mean()), float(np.abs(res[:, 3:]).mean())

    best = None
    for lam in (1e-3, 1e-2, 1e-1, 1, 10, 100):
        p = solve(Af, bf, lam)
        e = err(Ah, bh, p)
        score = e[0] / 0.05 + e[1] / 0.15
        if best is None or score < best[0]:
            best = (score, lam)
    lam = best[1]
    Aall = np.concatenate([Af, Ah]); ball = np.concatenate([bf, bh])
    p = solve(Aall, ball, lam)
    P = len(Zall) ** 2 * len(centers)
    rep = {"lambda": lam, "n_fit": len(fit), "n_hold": len(hold), "n_val": len(rows["val"]),
           "hold_charges_only": err(Ah, bh, None), "hold_fit_before_refit": err(Ah, bh, solve(Af, bf, lam)),
           "val_charges_only": err(Av, bv, None), "val_fitted": err(Av, bv, p),
           "units": "mean |mu error| e A, mean |Theta component error| e A^2 (COM, Buckingham)"}
    json.dump({"rc": a.rc, "width": a.width, "centers": centers.tolist(), "Z": Zall, "theta": p[:P].tolist(),
               "tau": p[P:].tolist(), "report": rep}, open(a.out, "w"))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
