#!/usr/bin/env python3
"""MLIPCPP_PRECISE_SQRT decision table: the same CUDA binary with the flag off (A) and on (B), every reference cell,
against his fp64 and his fp32 runs (lr_parity.per_frame metrics). Per cell and output: max over frames / max of his
runs, paired P(mlip.cpp worse than his run) per run (ties count 1/2), his own pairwise P range (independent pairs:
bitwise-identical runs are dropped), and the status A -> B.
Status: "in" = max ratio <= 1 and every P <= bound, bound = max(top of his pairwise range, 0.5 + 1.96 * 0.5 / sqrt(n))
(the binomial 95 % band of a fair coin over n frames; it also covers cells with one run of his); "OUT" otherwise. A cell that goes in -> OUT under B blocks a default flip.

    python sqrt_decision.py --root PARITY_DIR --a TAG_A --b TAG_B [--keys E F W q ...]
Cells are hard-wired below (H100 layout of cuda_sqrtdec_h100.sbatch: {d} = the tag directory suffix).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_parity import from_cpp, from_ref  # noqa: E402

CK6 = ("readout-pol", "control", "readout", "readout-1m", "readout-2m", "readout-4m")


def cells():
    c = []  # (name, json pattern with {t}, ref prefix, conservative)
    for g in CK6:
        for f in ("bulk", "slab"):
            c.append((f"{g} {f} (pbc 60)", f"out/pbc/td_h100_{{t}}/{g}_lr_nobase_{f}_cuda.json", f"ref/{g}/full_{f}_", False))
    for g in CK6:
        for f in ("ion", "cluster", "mol"):
            c.append((f"{g} {f} (open)", f"out/lr/td_h100_{{t}}/{g}_{f}_cuda.json", f"ref/{g}/full_{f}_", False))
    N = "out/new_cuda_td_h100_{t}"
    S = "out/sqrtdec_h100_{t}"
    for p in ("les_ladder", "les_ladder_c0", "les_local2", "les_local3"):
        for f in ("mol_z18", "bulk_f32"):
            c.append((f"{p} {f}", f"{S}/{p}_{f}_cuda.json", f"ref/seeded_les/{p}_{f}_", False))
    for p in ("les_ladder_lr", "les_ladder_lr_c0"):
        c.append((f"{p} mol_z18", f"{N}/{p}_mol_z18_cuda.json", f"ref/seeded_les/{p}_mol_z18_", False))
        c.append((f"{p} bulk_f32", f"{N}/{p}_bulk_f32_12_cuda.json", f"ref/seeded_les/{p}_bulk_f32_", False))
    for g in ("lr_species", "lr_both", "lr_fermi_cold"):
        for f in ("ion", "cluster"):
            c.append((f"seeded {g} {f}", f"{S}/seeded_{g}_{f}_cuda.json", f"ref/seeded/{g}_{f}_", False))
    for p in ("lrt_weighted", "lrt_fill", "lrt_fill_cold", "lrt_species_fill", "lrt_species_weighted"):
        for f in ("cluster", "mol_z18"):
            c.append((f"branch {p} {f}", f"{S}/{p}_{f}_cuda.json", f"ref/seeded_branch/{p}_{f}_", False))
    for p in ("lrt_species_fill", "lrt_species_weighted"):
        for f in ("bulk_f32", "bulknacl_f32"):
            c.append((f"branch {p} {f}", f"{S}/{p}_{f}_cuda.json", f"ref/seeded_branch/{p}_{f}_", False))
    for p in ("cc_fermi", "cc_residual", "cc_residual_plain"):
        for f in ("mol_z18", "bulk_f32"):
            c.append((f"{p} {f}", f"{S}/{p}_{f}_cuda.json", f"ref/seeded_cc/{p}_{f}_", False))
    for f in ("mol_z18", "bulk_f32"):
        c.append((f"tiny_qeq {f}", f"{S}/tiny_qeq_{f}_cuda.json", f"ref/seeded/tiny_qeq_{f}_", False))
    for p in ("cc_fermi_field", "cc_residual_field"):
        c.append((f"{p} mol_z18_field", f"{N}/{p}_mol_z18_field_cuda.json", f"ref/seeded_cc/{p}_mol_z18_field_", False))
        c.append((f"{p} bulk (conservative F)", f"{N}/{p}_bulk_f32_field_cuda.json", f"ref/seeded_cc/{p}_bulk_f32_field_", True))
        c.append((f"{p} bulk (full F, E.Z0)", f"{N}/{p}_bulk_f32_field_full_cuda.json", f"ref/seeded_cc/{p}_bulk_f32_field_", False))
    for p in ("cc_fermi", "cc_residual"):
        for f in ("multi", "scan_nacl"):
            c.append((f"{p} {f}", f"{N}/{p}_{f}_cuda.json", f"ref/seeded_cc/{p}_{f}_", False))
    for p in ("thr", "thr_cold_lrt"):
        for f in ("mol_z18", "cluster", "bulk_f32"):
            c.append((f"{p} {f}", f"{N}/{p}_{f}_cuda.json", f"ref/seeded_thr/{p}_{f}_", False))
    for p in ("cc_ind_field", "cc_ind_full"):
        for f in ("mol_z18_field", "mol_z18_field0"):
            c.append((f"{p} {f}", f"{N}/{p}_{f}_cuda.json", f"ref/seeded_cc/{p}_{f}_", False))
        c.append((f"{p} bulk (conservative F)", f"{N}/{p}_bulk_f32_field_cuda.json", f"ref/seeded_cc/{p}_bulk_f32_field_", True))
    return c


def pick(pre, tag):
    for ext in (".ref.npz", ".npz"):
        if os.path.exists(pre + tag + ext):
            return pre + tag + ext
    return None


def pworse(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if not m.any():
        return np.nan
    return float(np.mean((a[m] > b[m]) + 0.5 * (a[m] == b[m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--keys", nargs="+", default=["E", "F", "W", "q", "mu", "metallicity", "charge_residual",
                                                   "qeq_residual", "dipole_flow", "dipole_ind", "alpha"])
    ap.add_argument("--only", default="", help="substring filter on cell names")
    a = ap.parse_args()
    os.chdir(a.root)
    blockers, improved, worse_out = [], [], []
    for name, pat, pre, cons in cells():
        if a.only and a.only not in name:
            continue
        r64 = pick(pre, "cpu64")
        ja, jb = pat.format(t=a.a), pat.format(t=a.b)
        if not r64 or not os.path.exists(ja) or not os.path.exists(jb):
            print(f"-- {name}: missing ({'ref' if not r64 else ja if not os.path.exists(ja) else jb})"); continue
        z = np.load(r64)
        tags = sorted({os.path.basename(p)[len(os.path.basename(pre)):].split(".")[0]
                       for p in glob.glob(pre + "*32*.npz")})
        runs = {}
        for t in tags:
            p = pick(pre, t)
            try:
                runs[t] = from_ref(p, z, cons)
            except Exception:  # noqa: BLE001
                pass
        try:
            ca, cb = from_cpp(ja, z), from_cpp(jb, z)
        except KeyError as e:
            print(f"-- {name}: skipped ({e})"); continue
        lines = []
        for k in a.keys:
            hr = {t: r[k] for t, r in runs.items() if k in r and np.isfinite(r[k]).any()}
            if k not in ca or not hr or not np.isfinite(ca[k]).any():
                continue
            hm = max(np.nanmax(v) for v in hr.values())
            # his pairwise P over distinct (not bitwise-identical) runs
            ts = [t for t in hr]
            pp = [pworse(hr[x], hr[y]) for x in ts for y in ts if x != y and not np.array_equal(hr[x], hr[y], equal_nan=True)]
            n = int(np.sum(np.isfinite(ca[k])))
            band = 0.5 + 1.96 * 0.5 / np.sqrt(max(n, 1))  # binomial 95 % band of P for n frames
            top = max(max(pp) if pp else 0.5, band)
            st, desc = [], []
            for c in (ca, cb):
                v = c[k][:len(next(iter(hr.values())))]
                ratio = np.nanmax(v) / hm
                ps = {t: pworse(v[:len(hr[t])], hr[t][:len(v)]) for t in hr}
                ok = ratio <= 1.0 + 5e-3 and max(ps.values()) <= top
                st.append("in" if ok else "OUT")
                desc.append(f"{ratio:.2f}x P " + "/".join(f"{ps[t]:.2f}" for t in hr))
            flag = ""
            if st == ["in", "OUT"]:
                flag = "  <== BLOCKS"; blockers.append(f"{name} {k}")
            elif st == ["OUT", "in"]:
                flag = "  (fixed)"; improved.append(f"{name} {k}")
            elif st == ["OUT", "OUT"]:
                worse_out.append(f"{name} {k}: {desc[0]} -> {desc[1]}")
            lines.append(f"  {k:15s} n{n:3d} off {desc[0]:32s} [{st[0]:3s}] | on {desc[1]:32s} [{st[1]:3s}]"
                         f" | his pairwise {min(pp) if pp else float('nan'):.2f}-{max(pp) if pp else float('nan'):.2f}, bound {top:.2f}{flag}")
        print(f"-- {name}  (runs: {', '.join(runs)})")
        print("\n".join(lines))
    print("\nBLOCKERS (in -> OUT with the flag on):", "; ".join(blockers) or "none")
    print("FIXED (OUT -> in):", "; ".join(improved) or "none")
    print("OUT both ways:\n  " + "\n  ".join(worse_out))


if __name__ == "__main__":
    main()
