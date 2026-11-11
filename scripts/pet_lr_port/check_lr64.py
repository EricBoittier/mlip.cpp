#!/usr/bin/env python3
"""fp64 check of the torch PETLR (iris_torch.IrisTrunk + IrisLR) against his JAX PETLR, component by component, on
the oracle's own inputs: the trunk's selected pairs / R_ij / pair cutoffs from a ref_trunk.py --x64 dump, positions,
total charge, spin and dipole origin from the frames, and every sown value of a ref_petlr.py --x64 dump of the same
frames. μ: the host bisection (iris_torch.bisect_fermi) on the torch scores, then the in-graph Newton / midpoint.

    python check_lr64.py PARAMS.npz CKPT_DIR TRUNK_X64.npz FULL_X64.npz FRAMES.extxyz [--max N]
Prints max |torch - JAX| / max |JAX| per component (per frame, max and median over frames).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from check_torch64 import frames_info  # noqa: E402
from convert_iris import PETJAX_CONVERT, load_petjax_convert  # noqa: E402
from iris_to_gguf import fill  # noqa: E402
from iris_torch import IrisLR, IrisTrunk, bisect_fermi  # noqa: E402


def load_lr(lr, flat):
    """flax params/lr -> IrisLR (Dense kernel [in, out] -> weight [out, in]; MLP Dense_i -> Sequential index 2i)."""
    sd = lr.state_dict()
    used = set()
    for key, v in flat.items():
        if not key.startswith("params/lr/"):
            continue
        parts = key[len("params/lr/"):].split("/")
        leaf = parts[-1]
        mods = [p if not p.startswith("Dense_") else str(2 * int(p[6:])) for p in parts[:-1]]
        if leaf == "kernel":
            name, val = ".".join(mods + ["weight"]), np.asarray(v).T
        elif leaf == "scale":
            name, val = ".".join(mods + ["weight"]), np.asarray(v)
        elif leaf == "bias":
            name, val = ".".join(mods + ["bias"]), np.asarray(v)
        else:
            name, val = ".".join(mods + [leaf]) if mods else leaf, np.asarray(v)
        if name not in sd:
            raise KeyError(f"{key} -> {name}: not in IrisLR")
        if tuple(val.shape) != tuple(sd[name].shape):
            raise ValueError(f"{key} -> {name}: {val.shape} vs {tuple(sd[name].shape)}")
        sd[name] = torch.from_numpy(np.ascontiguousarray(val))
        used.add(name)
    missing = set(sd) - used
    if missing:
        raise ValueError(f"IrisLR parameters not in the checkpoint: {sorted(missing)}")
    lr.load_state_dict(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("params"); ap.add_argument("ckpt"); ap.add_argument("trunk"); ap.add_argument("full")
    ap.add_argument("frames"); ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    (_, cfg), = yaml.safe_load(open(f"{a.ckpt}/model/model.yaml")).items()
    flat = dict(np.load(a.params))
    sr = {k[len("params/sr/"):].replace("/", "."): v for k, v in flat.items() if k.startswith("params/sr/")}
    trunk = IrisTrunk(cfg)
    fill(trunk, sr, cfg, load_petjax_convert(PETJAX_CONVERT))
    trunk = trunk.double().eval()
    lr = IrisLR(cfg)
    load_lr(lr, flat)
    lr = lr.double().eval()
    scale = float(np.asarray(sr["energy_scale"]).reshape(()))
    zt, zf = np.load(a.trunk), np.load(a.full)
    assert json.loads(str(zt["meta"]))["x64"] and json.loads(str(zf["meta"]))["x64"]
    apt, ppt, apf = zt["atom_ptr"], zt["pair_ptr"], zf["atom_ptr"]
    info = frames_info(a.frames)
    atoms_all = read(a.frames, ":")
    nf = len(apf) - 1 if not a.max else min(a.max, len(apf) - 1)
    comps = {}

    def rel(name, x, y):
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        comps.setdefault(name, []).append(np.abs(x - y).max() / max(np.abs(y).max(), 1e-300))

    for f in range(nf):
        Z, spin = info[f]
        atoms = atoms_all[f]
        n = len(Z)
        c = zt["pair_c"][ppt[f]:ppt[f + 1]]; nb = zt["pair_n"][ppt[f]:ppt[f + 1]]
        R = zt["pair_R"][ppt[f]:ppt[f + 1]]; cut = zt["pair_cut"][ppt[f]:ppt[f + 1]]
        k = max(int(np.bincount(c, minlength=n).max()), 1) if len(c) else 1
        slot = np.zeros(len(c), int); cnt = np.zeros(n, int)
        for e in range(len(c)):
            slot[e] = c[e] * k + cnt[c[e]]; cnt[c[e]] += 1
        ev = np.zeros((n * k, 3)); ns = np.zeros(n * k, int); pad = np.ones(n * k, bool); ni = np.zeros(n * k, int)
        cv = np.full(n * k, cfg["cutoff"], float); rev = np.arange(n * k)
        ev[slot] = R; ns[slot] = Z[nb]; pad[slot] = False; cv[slot] = cut; ni[slot] = nb
        key = {(int(c[e]), int(nb[e]), *np.round(R[e], 9)): slot[e] for e in range(len(c))}
        for e in range(len(c)):
            rev[slot[e]] = key.get((int(nb[e]), int(c[e]), *np.round(-R[e], 9)), slot[e])
        t = lambda x, dt=torch.float64: torch.as_tensor(x, dtype=dt)
        targs = [t(Z, torch.long), t(ns.reshape(n, k), torch.long), t(ev.reshape(n, k, 3)), t(pad.reshape(n, k), torch.bool),
                 t(rev, torch.long), t(cv.reshape(n, k))]
        if trunk.conditioning:
            targs += [torch.full((n,), cfg["max_charge"], dtype=torch.long), torch.full((n,), spin - 1, dtype=torch.long)]
        Q = float(atoms.info.get("charge", 0))
        r0 = np.asarray(atoms.info["dipole_origin"], float) if "dipole_origin" in atoms.info else atoms.get_center_of_mass()
        with torch.no_grad():
            e_sr = trunk(*targs) * scale
            node, env = trunk._aux["node"], trunk._aux["env"]
            mu_b, gapped, m_round, top, bottom = 0.0, False, torch.zeros(n, dtype=torch.float64), 0, 0
            if lr.ref in ("fermi", "both"):
                mu_b, gapped, m_round, top, bottom = bisect_fermi(lr, lr.scores(node), Q)
            e_lr = lr(node, t(Z, torch.long), t(atoms.get_positions()), t(ni.reshape(n, k), torch.long),
                      t(pad.reshape(n, k), torch.bool), t(rev, torch.long), t(ev.reshape(n, k, 3)), env,
                      t(Q), t(mu_b), t(float(gapped)), m_round, top, bottom, t(r0))
        aux = lr._aux
        A = slice(apf[f], apf[f + 1])
        rel("energy (per atom, SR + LR)", (e_sr + e_lr).numpy(), zf["atom/energy"][A])
        for name, jkey in (("scores", "int/scores"), ("charges_base", "int/charges_base"), ("transfers", "int/transfers"),
                           ("charges_pass1", "int/charges_pass1"), ("delta_charges", "int/delta_charges"),
                           ("charges", "atom/charges")):
            if name in aux and jkey in zf.files:
                rel(name, aux[name].numpy(), zf[jkey][A])
        if "fermi_level" in aux:
            rel("fermi_level", [float(aux["fermi_level"])], [zf["int/fermi_level"][A][0]])
            rel("metallicity", [float(aux["metallicity"])], [zf["struct/metallicity"][f]])
        rel("dipole", aux["dipole"].numpy(), zf["struct/dipole"][f])
    print(f"torch fp64 vs JAX fp64 over {nf} frames (max / median of max|d|/max|ref| per frame):")
    for name, v in comps.items():
        print(f"  {name:28s} {max(v):.3g} / {np.median(v):.3g}")


if __name__ == "__main__":
    main()
