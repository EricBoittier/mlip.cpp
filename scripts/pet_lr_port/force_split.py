#!/usr/bin/env python3
"""Ion-F bisection, step 1: split mlip.cpp's force into its pieces and compare each with torch fp64 (= his JAX fp64 to
5e-14, check_grad64.py). Pieces (all from mlip.cpp's MLIPCPP_FORCE_SPLIT dump, unscaled + energy_scale):
  edge  : the scatter of dE/d edge_vectors (the ggml backward of the graph)
  cut   : the adaptive-cutoff chain term (dE/d cutoff_values from the graph, host Jacobian of the grid)
  pos   : dE/d positions (iris LR: Coulomb, SR kernel, dipole lever)
plus per-slot dE/d edge_vectors and dE/d cutoff_values against torch (matched by (centre, neighbour); open frames).

    python force_split.py PARAMS.npz CKPT_DIR SPLIT.bin FRAMES.extxyz [--lr] [--max N] [--ref REF_CPU32.npz ...]
                          [--fp64 REF_CPU64.npz] [--save OUT.npz]

--ref: his fp32 runs; for each, P(worse) of "hybrid" forces that carry mlip.cpp's error in ONE piece only
(the other pieces from torch fp64), to locate a systematic excess in a piece.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_grad64 as cg  # noqa: E402
from check_torch64 import frames_info  # noqa: E402
from ase.io import read  # noqa: E402


def read_split(path, v2=True):
    raw = open(path, "rb").read()
    off, out = 0, []
    while off < len(raw):
        n, k = np.frombuffer(raw, np.int32, 2, off); off += 8
        s = n * k
        take = lambda cnt, dt: np.frombuffer(raw, dt, cnt, off)
        nb = take(s, np.int32).reshape(n, k); off += 4 * s
        pm = take(s, np.float32).reshape(n, k); off += 4 * s
        gev = take(3 * s, np.float32).reshape(n, k, 3); off += 12 * s
        gcv = take(s, np.float32).reshape(n, k); off += 4 * s
        gpos = take(3 * n, np.float32).reshape(n, 3); off += 12 * n
        fcut = take(3 * n, np.float32).reshape(n, 3); off += 12 * n
        sc = float(take(1, np.float32)[0]); off += 4
        rec = dict(n=n, k=k, nb=nb, pm=pm, gev=gev, gcv=gcv, gpos=gpos, fcut=fcut, scale=sc)
        if v2:  # dumps from 2026-11-11 05:00 on also carry the graph inputs edge_vectors / cutoff_values
            rec["ev"] = take(3 * s, np.float32).reshape(n, k, 3); off += 12 * s
            rec["cv"] = take(s, np.float32).reshape(n, k); off += 4 * s
        out.append(rec)
    return out


def scatter_edges(nb, valid, g):
    """F[c] += g, F[nb] -= g (mlip.cpp's convention: dE/d(r_nb - r_c))."""
    n = nb.shape[0]
    F = np.zeros((n, 3))
    for c in range(n):
        for s in np.nonzero(valid[c])[0]:
            F[c] += g[c, s]; F[nb[c, s]] -= g[c, s]
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("split"); ap.add_argument("frames")
    ap.add_argument("--lr", action="store_true", help="the GGUF has the LR branch (else trunk only)")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--fp64", default=None, help="his fp64 npz (sanity: torch vs JAX fp64)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--save-torch", default=None, help="npz of the torch per-frame pieces (split_compare.py input)")
    ap.add_argument("--torch32", action="store_true", help="also the traced trunk in torch fp32 (trunk-only)")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    cfg, trunk, lr, scale = cg.build(a.params, a.ckpt)
    trunk32 = None
    if a.torch32:
        import copy
        trunk32 = copy.deepcopy(trunk).float()
        torch.set_num_threads(max(torch.get_num_threads(), 1))
    dumps = read_split(a.split)
    info = frames_info(a.frames)
    atoms_all = read(a.frames, ":")
    nf = len(dumps) if not a.max else min(a.max, len(dumps))
    keep = {}
    orig = cg.open_geometry

    def hooked(*args, **kw):
        g = orig(*args, **kw)
        g["edge_vectors"].retain_grad(); g["cutoff_values"].retain_grad()
        keep["g"] = g
        return g

    cg.open_geometry = hooked
    rows = {k: [] for k in ("F", "edge", "cut", "pos", "gev_slot", "gcv_slot")}
    per = []
    for f in range(nf):
        d = dumps[f]
        Z, spin = info[f]
        at = atoms_all[f]
        Q = float(at.info.get("charge", 0))
        r0 = np.asarray(at.info["dipole_origin"], float) if "dipole_origin" in at.info else at.get_center_of_mass()
        pos = torch.tensor(at.get_positions(), dtype=torch.float64, requires_grad=True)
        E, aux = cg.energy(cfg, trunk, lr, scale, pos, Z, spin, Q, r0, lr_on=a.lr)
        E.backward()
        g = keep["g"]
        F_t = -pos.grad.numpy()
        gev_t, gcv_t = g["edge_vectors"].grad.numpy(), g["cutoff_values"].grad.numpy()
        nb_t, pad_t = g["neighbor_index"].numpy(), g["padding_mask"].numpy()
        n = len(Z)
        # torch's own pieces (same formulas as mlip.cpp)
        F_edge_t = scatter_edges(nb_t, ~pad_t, gev_t)
        pos2 = torch.tensor(at.get_positions(), dtype=torch.float64, requires_grad=True)
        cg.open_geometry = orig
        _, F_fz, _ = cg.forces(cfg, trunk, lr, scale, pos2, Z, spin, Q, r0, lr_on=a.lr, freeze_cut=True)
        cg.open_geometry = hooked
        F_cut_t = F_t - F_fz
        F_pos_t = F_fz - F_edge_t
        # mlip.cpp pieces
        sc = d["scale"]
        valid = d["pm"] < 0.5
        F_edge_c = scatter_edges(d["nb"], valid, d["gev"].astype(np.float64)) * sc
        F_cut_c = d["fcut"].astype(np.float64) * sc
        F_pos_c = -d["gpos"].astype(np.float64) * sc
        F_c = F_edge_c + F_cut_c + F_pos_c
        Fmax = np.abs(F_t).max()
        rows["F"].append(np.abs(F_c - F_t).max() / Fmax)
        rows["edge"].append(np.abs(F_edge_c - F_edge_t).max() / Fmax)
        rows["cut"].append(np.abs(F_cut_c - F_cut_t).max() / Fmax)
        rows["pos"].append(np.abs(F_pos_c - F_pos_t).max() / Fmax)
        # per slot, matched by pair
        tmap = {(c, int(nb_t[c, s])): (c, s) for c in range(n) for s in np.nonzero(~pad_t[c])[0]}
        dg, dc, gscale, cscale = [], [], 0.0, 0.0
        for c in range(n):
            for s in np.nonzero(valid[c])[0]:
                ct, st = tmap[(c, int(d["nb"][c, s]))]
                dg.append(np.abs(d["gev"][c, s] * sc - gev_t[ct, st]).max())
                dc.append(abs(d["gcv"][c, s] * sc - gcv_t[ct, st]))
        assert len(dg) == len(tmap), "edge sets differ"
        rows["gev_slot"].append(max(dg) / max(np.abs(gev_t[~pad_t]).max(), 1e-300))
        rows["gcv_slot"].append(max(dc) / max(np.abs(gcv_t[~pad_t]).max(), 1e-300))
        if trunk32 is not None:  # the traced trunk in torch fp32 / fp64 on fp32 edge vectors (trunk-only)
            k_ = nb_t.shape[1]
            Zt = torch.as_tensor(Z, dtype=torch.long)
            ns = Zt[g["neighbor_index"].reshape(-1)].reshape(n, k_)
            p32 = torch.tensor(at.get_positions(), dtype=torch.float32)
            ev_p32 = g["edge_vectors"].detach().float().clone()
            vm = torch.from_numpy(~pad_t)
            ev_p32[vm] = (p32[g["neighbor_index"]] - p32[:, None, :])[vm]  # fp32 subtraction, like mlip.cpp / JAX

            def run(model, ev, dt):
                ev = ev.to(dt).clone().requires_grad_(True)
                args = [Zt, ns, ev, g["padding_mask"], g["reverse"], g["cutoff_values"].detach().to(dt)]
                if model.conditioning:
                    args += [torch.full((n,), cfg["max_charge"], dtype=torch.long),
                             torch.full((n,), spin - 1, dtype=torch.long)]
                (model(*args) * scale).sum().backward()
                return scatter_edges(nb_t, ~pad_t, ev.grad.double().numpy())

            F_edge_32 = run(trunk32, g["edge_vectors"].detach(), torch.float32)      # exact geometry, fp32 maths
            F_edge_32p = run(trunk32, ev_p32, torch.float32)                         # fp32 geometry, fp32 maths
            F_edge_in = run(trunk, ev_p32, torch.float64)                            # fp32 geometry, exact maths
            for nm, v in (("edge_torch32", F_edge_32), ("edge_torch32_p32", F_edge_32p), ("edge_input_only", F_edge_in)):
                rows.setdefault(nm, []).append(np.abs(v - F_edge_t).max() / Fmax)
        per.append(dict(F_e32=F_edge_32 if trunk32 is not None else None, F_e32p=F_edge_32p if trunk32 is not None else None, F_ein=F_edge_in if trunk32 is not None else None, F_t=F_t, F_edge_t=F_edge_t, F_cut_t=F_cut_t, F_pos_t=F_pos_t,
                        F_edge_c=F_edge_c, F_cut_c=F_cut_c, F_pos_c=F_pos_c))
        if f % 40 == 0:
            print(f"frame {f}: n={n} F {rows['F'][-1]:.2e} edge {rows['edge'][-1]:.2e} cut {rows['cut'][-1]:.2e} "
                  f"pos {rows['pos'][-1]:.2e}", flush=True)
    print(f"mlip.cpp pieces vs torch fp64, {nf} frames (max|d piece| / max|F|): max / median / mean")
    for k, v in rows.items():
        print(f"  {k:9s} {max(v):.3e} / {np.median(v):.3e} / {np.mean(v):.3e}")
    if a.fp64:
        z = np.load(a.fp64); ap_ = z["atom_ptr"]
        e = [np.abs(per[f]["F_t"] - z["forces"][ap_[f]:ap_[f + 1]]).max() for f in range(nf)]
        print(f"  torch fp64 vs his JAX fp64 F: max {max(e):.2e}")
    for ref in a.ref:
        z = np.load(ref); ap_ = z["atom_ptr"]
        his, hyb = [], {k: [] for k in ("all", "edge", "cut", "pos")}
        for f in range(nf):
            p = per[f]
            Fj = z["forces"][ap_[f]:ap_[f + 1]].astype(np.float64)
            Fmax = np.abs(p["F_t"]).max()
            his.append(np.abs(Fj - p["F_t"]).max() / Fmax)
            pieces = {"edge": p["F_edge_c"] - p["F_edge_t"], "cut": p["F_cut_c"] - p["F_cut_t"],
                      "pos": p["F_pos_c"] - p["F_pos_t"]}
            hyb["all"].append(np.abs(sum(pieces.values())).max() / Fmax)
            for k, v in pieces.items():
                hyb[k].append(np.abs(v).max() / Fmax)
        his = np.array(his)
        if a.torch32:
            for key, nm in (("F_e32", "torch fp32, exact geometry"), ("F_e32p", "torch fp32, fp32 geometry"),
                            ("F_ein", "torch fp64 on fp32 geometry")):
                t = np.array([np.abs(per[f][key] - per[f]["F_edge_t"]).max() / np.abs(per[f]["F_t"]).max()
                              for f in range(nf)])
                print(f"  {nm:28s} vs {Path(ref).name}: P(worse) {np.mean(t > his):.2f} (med {np.median(t):.2e});"
                      f" mlip.cpp worse than it: P {np.mean(np.array(hyb['all']) > t):.2f}")
        if a.torch32:  # arithmetic error only: against the exact maths on the same fp32 edge vectors
            ar_c, ar_h = [], []
            for f in range(nf):
                p = per[f]
                Fj = z["forces"][ap_[f]:ap_[f + 1]].astype(np.float64)
                Fmax = np.abs(p["F_t"]).max()
                ar_c.append(np.abs(p["F_edge_c"] + p["F_cut_c"] - p["F_ein"] - p["F_cut_t"]).max() / Fmax)
                ar_h.append(np.abs(Fj - p["F_ein"] - p["F_cut_t"]).max() / Fmax)
            ar_c, ar_h = np.array(ar_c), np.array(ar_h)
            print(f"  arithmetic-only error (vs torch fp64 on fp32 geometry): mlip.cpp med {np.median(ar_c):.2e} "
                  f"max {ar_c.max():.2e}; his med {np.median(ar_h):.2e} max {ar_h.max():.2e}; P(cpp worse) "
                  f"{np.mean(ar_c > ar_h):.2f}")
        s = " ".join(f"{k} P {np.mean(np.array(v) > his):.2f} (med {np.median(v):.2e})" for k, v in hyb.items())
        print(f"  vs {Path(ref).name} (his med {np.median(his):.2e}): {s}")
    if a.save_torch:
        keys = ("F_t", "F_edge_t", "F_cut_t", "F_pos_t", "F_ein", "F_e32p")
        np.savez(a.save_torch, nf=nf, **{f"{k}_{f}": per[f][k] for f in range(nf) for k in keys
                                         if per[f].get(k) is not None})
    if a.save:
        np.savez(a.save, **{k: np.array(v) for k, v in rows.items()})


if __name__ == "__main__":
    main()
