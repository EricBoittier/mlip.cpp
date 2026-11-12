#!/usr/bin/env python3
"""Trunk parity: mlip.cpp (mlipcpp-eval JSON) against the iris/pet-jax oracle (ref_trunk.py npz).

    python compare_trunk.py REF.npz MLIPCPP.json [--ref2 OTHER_REF.npz] [--json OUT.json]

Metrics (energies carry no composition baseline on either side: the GGUF is exported --no-baseline):
  E_rel   max over frames |E_cpp - E_ref| / |E_ref|                (total SR energy per frame)
  E_atom  max over frames |E_cpp - E_ref| / n_atoms                 (eV/atom)
  F_rel   max over frames ||F_cpp - F_ref||_max / ||F_ref||_max     (largest component error over the
          frame's largest force component)
  F_abs   max |F_cpp - F_ref| over all components                  (eV/A)
  F_rms   RMS over all components
  V_rel   max over periodic frames ||sym V_cpp - sym V_ref||_max / ||sym V_ref||_max (virial = mlip.cpp stress x
          volume; symmetric parts: mlip.cpp emits Voigt stress); V_antisym_ref = the reference's antisymmetric part
With --ref2 (e.g. the oracle's fp64 run, or its GPU fp32 run) the same metrics are printed for REF vs REF2:
his own floor at that precision.
"""
import argparse
import json

import numpy as np


def load_ref(path):
    z = np.load(path)
    a, p = z["atom_ptr"], z["pair_ptr"]
    frames = []
    for k in range(len(a) - 1):
        frames.append(dict(E=float(np.sum(z["energy"][a[k]:a[k + 1]].astype(np.float64))),
                           F=z["forces"][a[k]:a[k + 1]].astype(np.float64),
                           V=z["virial"][k].astype(np.float64)))
    return frames, json.loads(str(z["meta"]))


def volumes(frames_path):
    """Cell volume per frame of an extxyz (nan for frames without a Lattice)."""
    out, lines, i = [], open(frames_path).read().splitlines(), 0
    while i < len(lines):
        n = int(lines[i]); head = lines[i + 1]
        if 'Lattice="' in head:
            c = np.array(head.split('Lattice="')[1].split('"')[0].split(), dtype=np.float64).reshape(3, 3)
            out.append(abs(np.linalg.det(c)))
        else:
            out.append(np.nan)
        i += n + 2
    return out


def load_cpp(path, vols=None):
    d = json.load(open(path))
    out = []
    for k, f in enumerate(d["frames"]):
        n = f["n_atoms"]
        V = np.full((3, 3), np.nan)
        if "stress" in f and vols is not None and np.isfinite(vols[k]):
            s = np.asarray(f["stress"], dtype=np.float64)  # Voigt xx yy zz yz xz xy, eV/A^3
            V = np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]]) * vols[k]
        full = np.asarray(f["virial"], dtype=np.float64).reshape(3, 3) if "virial" in f else None
        out.append(dict(E=float(f["energy"]), F=np.asarray(f["forces"], dtype=np.float64).reshape(n, 3), V=V, W=full))
    return out


def metrics(a, b):
    e_rel, e_atom, f_rel, f_abs, f_sq, n_f, v_rel, v_anti, w_rel = [], [], [], [], 0.0, 0, [], [], []
    for x, y in zip(a, b):
        n = len(y["F"])
        e_rel.append(abs(x["E"] - y["E"]) / max(abs(y["E"]), 1e-300))
        e_atom.append(abs(x["E"] - y["E"]) / n)
        d = np.abs(x["F"] - y["F"])
        f_abs.append(d.max())
        f_rel.append(d.max() / max(np.abs(y["F"]).max(), 1e-300))
        f_sq += float((d ** 2).sum()); n_f += d.size
        if np.isfinite(x["V"]).all() and np.isfinite(y["V"]).all():
            # mlip.cpp emits the symmetric (Voigt) stress; PET is not exactly rotation invariant, so the
            # oracle's virial has an antisymmetric part: compare sym(V), report the antisymmetric size
            sx, sy = 0.5 * (x["V"] + x["V"].T), 0.5 * (y["V"] + y["V"].T)
            v_rel.append(np.abs(sx - sy).max() / max(np.abs(sy).max(), 1e-300))
            v_anti.append(np.abs(y["V"] - y["V"].T).max() / 2 / max(np.abs(sy).max(), 1e-300))
            if x.get("W") is not None:  # mlip.cpp's full virial against the oracle's, unsymmetrised
                w_rel.append(np.abs(x["W"] - y["V"]).max() / max(np.abs(y["V"]).max(), 1e-300))
    r = dict(n=len(a), E_rel=max(e_rel), E_rel_median=float(np.median(e_rel)), E_atom=max(e_atom),
             F_rel=max(f_rel), F_rel_median=float(np.median(f_rel)), F_abs=max(f_abs),
             F_rms=(f_sq / n_f) ** 0.5)
    if v_rel:
        r.update(V_rel=max(v_rel), V_rel_median=float(np.median(v_rel)), V_antisym_ref=max(v_anti))
        if w_rel:
            r.update(W_full_rel=max(w_rel), W_full_rel_median=float(np.median(w_rel)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("cpp")
    ap.add_argument("--ref2"); ap.add_argument("--json")
    ap.add_argument("--frames", help="the extxyz evaluated (cell volumes for the stress -> virial)")
    a = ap.parse_args()
    ref, meta = load_ref(a.ref)
    cpp = load_cpp(a.cpp, volumes(a.frames) if a.frames else None)
    assert len(ref) == len(cpp), (len(ref), len(cpp))
    res = {"cpp_vs_ref": metrics(cpp, ref), "ref_meta": meta}
    if a.ref2:
        ref2, meta2 = load_ref(a.ref2)
        res["ref2_vs_ref"] = metrics(ref2, ref)
        res["cpp_vs_ref2"] = metrics(cpp, ref2)
        res["ref2_meta"] = meta2
    for k, v in res.items():
        if not k.endswith("meta"):
            print(k, " ".join(f"{m}={x:.3g}" if isinstance(x, float) else f"{m}={x}" for m, x in v.items()))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
