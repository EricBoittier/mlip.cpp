#!/usr/bin/env python3
"""Stage 1 of the ladder (physics pretraining): the charge network from scratch, MACE-POLAR-1-style.

A Plato-D2-att PET without charge/spin conditioning (inputs R and Z only; the lrc graph mode runs the charge pass at
Q = 0, lrc_qpass_charge "zero") gives raw per-atom charges q_raw; Fukui equilibration to the total charge Q,
    q_i = q_raw_i + f_i (Q - sum_j q_raw_j) / sum_j f_j,   f_i = softplus(phi_Z),
so sum q = Q exactly with no uniform shift (the correction is weighted per element). Supervised on DFT Loewdin charges:
    L = mean_atoms (q - q_DFT)^2 + w_c mean_{clusters of multi-cluster structures} (sum_c q - sum_c q_DFT)^2
with clusters the single-link 8 A components (the data audit's cluster rule). Inputs R, Z and Q only (DFT emulation).

Data: a MemmapSplit (ns.npy na.npy x.bin a.bin charge.bin lowdin_charge.bin), rows in EXCL/train.npy dropped;
validation on VAL. Writes OUT.pt ({"model": PET}), OUT.ckpt, OUT_fukui.json ({"Z", "phi"}), OUT_log.jsonl.

    python run.py train_charge.py TRAIN_DIR EXCL_DIR VAL_DIR OUT [--epochs 6 --batch 32 --lr 1e-3 --wc 1.0 --hours 1.0]"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_scratch import build  # noqa: E402


class Split:
    def __init__(self, d, excl=None):
        self.na = np.load(f"{d}/na.npy")
        self.ns = len(self.na) - 1
        self.x = np.fromfile(f"{d}/x.bin", np.float32).reshape(-1, 3)
        self.a = np.fromfile(f"{d}/a.bin", np.int32)
        self.Q = np.fromfile(f"{d}/charge.bin", np.float32)
        self.q = np.fromfile(f"{d}/lowdin_charge.bin", np.float32)
        drop = set(np.load(excl).tolist()) if excl else set()
        ok = []
        for s in range(self.ns):
            if s in drop:
                continue
            q = self.q[self.na[s]:self.na[s + 1]]
            if np.isfinite(q).all():
                ok.append(s)
        self.rows = np.array(ok)
        self.cl = {}

    def clusters(self, s):
        if s not in self.cl:
            from scipy.sparse.csgraph import connected_components
            from scipy.spatial.distance import cdist
            x = self.x[self.na[s]:self.na[s + 1]].astype(float)
            A = cdist(x, x) < 8.0
            self.cl[s] = connected_components(A, directed=False)[1].astype(np.int64)
        return self.cl[s]


def haar(rng):
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def batch(sp, rows, rng, dev, nl_opts, rotate):
    from metatomic.torch import System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
    systems, qd, Q, ci, sid = [], [], [], [], []
    off = 0
    for k, s in enumerate(rows):
        i0, i1 = sp.na[s], sp.na[s + 1]
        x = sp.x[i0:i1].astype(np.float64)
        if rotate:
            x = x @ haar(rng).T
        sysm = System(types=torch.tensor(sp.a[i0:i1], dtype=torch.int32), positions=torch.tensor(x, dtype=torch.float32),
                      cell=torch.zeros(3, 3), pbc=torch.zeros(3, dtype=torch.bool))
        systems.append(get_system_with_neighbor_lists(sysm, nl_opts).to(dev))
        qd.append(sp.q[i0:i1])
        Q.append(float(sp.Q[s]))
        c = sp.clusters(s)
        ci.append(c + off)
        off += int(c.max()) + 1
        sid.append(np.full(i1 - i0, k))
    t = lambda v, dt=torch.float32: torch.tensor(np.concatenate(v), dtype=dt, device=dev)
    return systems, t(qd), torch.tensor(Q, device=dev), t(ci, torch.long), t(sid, torch.long), off


def forward(model, phi, systems, Q, sid, dev):
    from metatomic.torch import ModelOutput
    out = model(systems, {"energy": ModelOutput(per_atom=True)})
    raw = out["energy"].block().values[:, 0]
    Z = torch.cat([s.types for s in systems]).long()
    f = torch.nn.functional.softplus(phi[Z])
    G = len(systems)
    sraw = torch.zeros(G, device=dev).index_add_(0, sid, raw)
    sf = torch.zeros(G, device=dev).index_add_(0, sid, f)
    return raw + f * ((Q - sraw) / sf)[sid]


def losses(q, qd, ci, nclu, sid, G, dev):
    la = ((q - qd) ** 2).mean()
    cs = torch.zeros(nclu, device=dev).index_add_(0, ci, q - qd)
    # clusters of multi-cluster structures only
    first = torch.zeros(nclu, dtype=torch.long, device=dev).index_put_((ci,), sid)
    per = torch.zeros(G, device=dev).index_add_(0, first, torch.ones(nclu, device=dev))
    m = per[first] > 1
    lc = (cs[m] ** 2).mean() if m.any() else torch.zeros((), device=dev)
    return la, lc, cs[m].abs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("excl")
    ap.add_argument("val")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wc", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--d-pet", type=int, default=128)
    ap.add_argument("--gnn", type=int, default=2)
    ap.add_argument("--att", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    ns = argparse.Namespace(d_pet=a.d_pet, heads=8, gnn=a.gnn, att=a.att, cutoff=8.0, adaptive=16.0, seed=a.seed)
    model, hyp, _ = build(ns, charge_model=True)
    model = model.to(dev).train()
    phi = torch.nn.Parameter(torch.zeros(84, device=dev))
    tr, va = Split(a.train, f"{a.excl}/train.npy"), Split(a.val, f"{a.excl}/val.npy")
    nl_opts = model.requested_neighbor_lists()
    params = [p for p in model.parameters() if p.requires_grad] + [phi]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    spe = len(tr.rows) // a.batch
    total = spe * a.epochs if not a.max_steps else a.max_steps
    warm = max(1, total // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    log = open(a.out + "_log.jsonl", "a")
    t0, step = time.time(), 0
    vrows = va.rows[:: max(1, len(va.rows) // 1500)]

    def validate():
        model.eval()
        ea, n, ec, nc = 0.0, 0, [], 0
        with torch.no_grad():
            for k in range(0, len(vrows), a.batch):
                sy, qd, Q, ci, sid, nclu = batch(va, vrows[k:k + a.batch], rng, dev, nl_opts, False)
                q = forward(model, phi, sy, Q, sid, dev)
                ea += (q - qd).abs().sum().item(); n += len(q)
                _, _, c = losses(q, qd, ci, nclu, sid, len(sy), dev)
                ec += c.tolist()
        model.train()
        return ea / n, float(np.mean(ec)) if ec else float("nan"), len(ec)

    done = False
    for ep in range(a.epochs):
        perm = rng.permutation(tr.rows)
        for k in range(spe):
            sy, qd, Q, ci, sid, nclu = batch(tr, perm[k * a.batch:(k + 1) * a.batch], rng, dev, nl_opts, True)
            q = forward(model, phi, sy, Q, sid, dev)
            la, lc, _ = losses(q, qd, ci, nclu, sid, len(sy), dev)
            L = la + a.wc * lc
            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"step {step}/{total} ep {ep} L {L.item():.4g} atom {la.item():.4g} clu {lc.item():.4g} "
                      f"{el / step * 1e3:.0f} ms/step", flush=True)
            if (a.max_steps and step >= a.max_steps) or time.time() - t0 > a.hours * 3600:
                done = True
                break
        mae, cmae, ncl = validate()
        rec = dict(epoch=ep, step=step, val_atom_mae_e=mae, val_cluster_sum_mae_e=cmae, val_clusters=ncl,
                   s=time.time() - t0, ms_per_step=(time.time() - t0) / step * 1e3)
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        save(model, phi, hyp, a.out)
        if done:
            break


def save(model, phi, hyp, out):
    import copy
    m = copy.deepcopy(model).cpu().eval()
    torch.save({"model": m, "hypers": m.hypers}, out + ".pt")
    torch.save(m.get_checkpoint(), out + ".ckpt")
    json.dump({"Z": list(range(1, 84)), "phi": phi.detach().cpu()[1:84].tolist()}, open(out + "_fukui.json", "w"))


if __name__ == "__main__":
    main()
