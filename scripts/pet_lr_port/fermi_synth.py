#!/usr/bin/env python3
"""Synthetic score sets through his iris.pet.charges.fermi_level (fp64), covering its three branches: ordinary
(Newton step), gapped (midpoint), saturated (density below eps: the eps clamp). Run in his env; writes OUT.npz
with the scores, Q, mu and the branch flags. check: python fermi_synth.py --check OUT.npz (torch side, iris_torch).

    python fermi_synth.py OUT.npz | python fermi_synth.py --check OUT.npz
"""
import sys

import numpy as np

CASES = [  # (name, scores, Q)
    ("ordinary_q0", [0.31, -0.42, 1.73, -1.12, 0.05, 0.66], 0.0),
    ("ordinary_q1", [0.31, -0.42, 1.73, -1.12, 0.05, 0.66], 1.0),
    ("ordinary_qm2", [0.9, 2.1, -0.3, -1.7, 0.4], -2.0),
    ("gapped_q0", [0.02, 1.01, -0.98, 0.0, 2.03], 2.0),
    ("gapped_q1", [3.0, 1.0, -1.0], 3.0),
    ("saturated", [10.0, -10.0, 9.0], 3.0),
    ("tie", [0.0, 0.0, 0.0, 0.0], 1.0),
    # T = 0.02 (his biotrimers temperatures): the gapped midpoint is reachable only at low T
    ("T002_gapped", [0.02, 1.01, -0.98, 0.0, 2.03], 2.0, 0.02),
    ("T002_gapped_b", [3.0, 1.0, -1.0, 0.1], 3.0, 0.02),
    ("T002_ordinary", [0.31, -0.42, 1.73, -1.12, 0.05, 0.66], 1.0, 0.02),
]
CONFIG = (-3, 3, 0.1)


def config(case):
    return (-3, 3, case[3]) if len(case) > 3 else CONFIG


def run_his(out):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from iris.pet.charges import fermi_level, occupation, occupation_density
    rows = {}
    for case in CASES:
        name, s, Q = case[:3]
        cf = config(case)
        s = jnp.asarray(s, jnp.float64)
        n = s.shape[0]
        mu = fermi_level(s, jnp.zeros(n, jnp.int32), jnp.asarray([Q]), jnp.ones(n, bool), cf)
        g = jax.grad(lambda x: fermi_level(x, jnp.zeros(n, jnp.int32), jnp.asarray([Q]), jnp.ones(n, bool), cf)[0])(s)
        rows[name] = (np.asarray(s), Q, float(mu[0]), np.asarray(g))
    np.savez(out, **{f"{k}/s": v[0] for k, v in rows.items()}, **{f"{k}/Q": v[1] for k, v in rows.items()},
             **{f"{k}/mu": v[2] for k, v in rows.items()}, **{f"{k}/dmu": v[3] for k, v in rows.items()})
    for k, v in rows.items():
        print(k, v[2])


def check(path):
    import torch
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    from iris_torch import IrisLR, bisect_fermi
    lr = IrisLR.__new__(IrisLR)
    z = np.load(path)
    for case in CASES:
        name = case[0]
        lr.m_min, lr.m_max, lr.T = config(case)
        s = torch.tensor(z[f"{name}/s"], dtype=torch.float64, requires_grad=True)
        Q = float(z[f"{name}/Q"])
        mu_b, gapped, m, top, bottom = bisect_fermi(lr, s.detach(), Q)
        mu_n = mu_b + (lr.occupation(s - mu_b).sum() - Q) / torch.clamp(lr.density(s - mu_b).sum(), min=1e-6)
        d = s - m
        mu = 0.5 * ((d * top).sum() + (d * bottom).sum()) if gapped else mu_n
        (g,) = torch.autograd.grad(mu, s)
        ref, gref = float(z[f"{name}/mu"]), z[f"{name}/dmu"]
        branch = "gapped" if gapped else ("eps-clamped" if float(lr.density(s.detach() - mu_b).sum()) < 1e-6 else "newton")
        print(f"{name:13s} {branch:11s} mu {float(mu):+.15f} his {ref:+.15f} |d| {abs(float(mu) - ref):.1e}  "
              f"dmu/ds max|d| {np.abs(g.numpy() - gref).max():.1e}")


if __name__ == "__main__":
    check(sys.argv[2]) if sys.argv[1] == "--check" else run_his(sys.argv[1])
