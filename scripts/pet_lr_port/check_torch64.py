#!/usr/bin/env python3
"""fp64 check of the transliteration iris_torch.IrisTrunk == pet-jax trunk, on the oracle's own inputs: the selected
pair list, R_ij and per-pair cutoffs of a ref_trunk.py --x64 dump are packed into the [n, k] graph layout and run
through IrisTrunk in float64 (weights from the iris tree, as iris_to_gguf.py loads them). Compares per-atom energies
(max |dE_i| / max |E_i| per frame). Network maths only: list building and adaptive selection are the dump's.

    python check_torch64.py PARAMS.npz CKPT_DIR REF_X64.npz FRAMES.extxyz [--max N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from petjax_convert import PETJAX_CONVERT, load_petjax_convert  # noqa: E402
from iris_to_gguf import fill  # noqa: E402
from iris_torch import IrisTrunk  # noqa: E402


def frames_info(path):
    out, lines, i = [], open(path).read().splitlines(), 0
    from ase.data import atomic_numbers
    while i < len(lines):
        n = int(lines[i]); head = lines[i + 1]
        spin = int(head.split("spin_multiplicity=")[1].split()[0]) if "spin_multiplicity=" in head else 1
        Z = [atomic_numbers[l.split()[0]] for l in lines[i + 2:i + 2 + n]]
        out.append((np.array(Z), spin)); i += n + 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("ref"); ap.add_argument("frames")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    (_, cfg), = yaml.safe_load(open(f"{a.ckpt}/model/model.yaml")).items()
    flat = dict(np.load(a.params))
    sr = {k[len("params/sr/"):].replace("/", "."): v for k, v in flat.items() if k.startswith("params/sr/")}
    model = IrisTrunk(cfg)
    fill(model, sr, cfg, load_petjax_convert(PETJAX_CONVERT))
    model = model.double().eval()
    scale = float(np.asarray(sr["energy_scale"]).reshape(()))
    z = np.load(a.ref)
    assert json.loads(str(z["meta"]))["x64"], "needs an --x64 dump"
    ap_, pp = z["atom_ptr"], z["pair_ptr"]
    info = frames_info(a.frames)
    worst, worst_h = [], []
    cap = {}
    model.node_energy_heads[0].register_forward_hook(lambda m, i, o: cap.__setitem__("h", i[0].detach()))
    nf = len(ap_) - 1 if not a.max else min(a.max, len(ap_) - 1)
    for f in range(nf):
        Z, spin = info[f]
        n = len(Z)
        c = z["pair_c"][pp[f]:pp[f + 1]]; nb = z["pair_n"][pp[f]:pp[f + 1]]
        R = z["pair_R"][pp[f]:pp[f + 1]]; cut = z["pair_cut"][pp[f]:pp[f + 1]]
        k = max(int(np.bincount(c, minlength=n).max()), 1) if len(c) else 1
        slot = np.zeros(len(c), int); cnt = np.zeros(n, int)
        for e in range(len(c)):
            slot[e] = c[e] * k + cnt[c[e]]; cnt[c[e]] += 1
        ev = np.zeros((n * k, 3)); ns = np.zeros(n * k, int); pad = np.ones(n * k, bool)
        cv = np.full(n * k, cfg["cutoff"], float); rev = np.arange(n * k)
        ev[slot] = R; ns[slot] = Z[nb]; pad[slot] = False; cv[slot] = cut
        # reverse: the pair (j, i) with R_ji = -R_ij (else the pad sentinel, as pet-jax's masked reverse)
        key = {(int(c[e]), int(nb[e]), *np.round(R[e], 9)): slot[e] for e in range(len(c))}
        for e in range(len(c)):
            rev[slot[e]] = key.get((int(nb[e]), int(c[e]), *np.round(-R[e], 9)), slot[e])
        t = lambda x, dt=torch.float64: torch.as_tensor(x, dtype=dt)
        args = [t(Z, torch.long), t(ns.reshape(n, k), torch.long), t(ev.reshape(n, k, 3)), t(pad.reshape(n, k), torch.bool),
                t(rev, torch.long), t(cv.reshape(n, k))]
        if model.conditioning:
            args += [torch.full((n,), cfg["max_charge"], dtype=torch.long), torch.full((n,), spin - 1, dtype=torch.long)]
        with torch.no_grad():
            e = model(*args).numpy() * scale
        ref = z["energy"][ap_[f]:ap_[f + 1]]
        worst.append(np.abs(e - ref).max() / np.abs(ref).max())
        h, href = cap["h"].numpy(), z["node"][ap_[f]:ap_[f + 1]]
        worst_h.append(np.abs(h - href).max() / np.abs(href).max())
    print(f"IrisTrunk fp64 vs JAX fp64, per-atom E: max {max(worst):.3g} median {np.median(worst):.3g} (n={nf}); node features: max {max(worst_h):.3g} median {np.median(worst_h):.3g}")


if __name__ == "__main__":
    main()
