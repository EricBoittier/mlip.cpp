#!/usr/bin/env python3
"""The "mlip.cpp fp64" column: mlip.cpp's fp64 executor (mlipcpp-eval --fp64) against his JAX fp64 dumps, for every
fp64 reference under ref/ (the six omol-4m checkpoints' full / trunk refs and every seeded preset). Frames are cut
from the frame files as TEXT (the first N frames, exact digits) and checked against the ref's atom counts.

    python fp64_sweep.py [--root ~/data/plato-lr-scratch/pet_lr_port] [--eval BIN] [--n 12] [--only REGEX]
                         [--out out/fp64/sweep.txt] [--json-dir out/fp64/json] [--env-f32]

Per ref, max over frames of: E = |dE| / sum|E_i|, F = max|dF| / max|F|, W = max|dW| / max|W| (3-D), q = max|dq| /
max|q|, dipole = max|dD| / max|D| (open), mu, metallicity, charge_residual, dipole_flow (relative). --env-f32: run the
same executor on float32-rounded inputs (MLIPCPP_EXEC=fp64, no --fp64), the input-rounding share of the fp32 error.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

CKPTS = ("control", "readout", "readout-pol", "readout-1m", "readout-2m", "readout-4m")


def frames_text(path, n=None):
    """the first n frames of an extxyz as text blocks"""
    lines = Path(path).read_text().splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines) and (n is None or len(out) < n):
        if not lines[i].strip():
            i += 1
            continue
        na = int(lines[i].split()[0])
        out.append("".join(lines[i:i + na + 2]))
        i += na + 2
    return out


def locate(root, ref):
    """(gguf, frames file) of a ref npz, or None"""
    rel = ref.relative_to(root / "ref")
    d, name = rel.parts[0], rel.name
    z = np.load(ref)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    nf = len(z["atom_ptr"]) - 1
    g = root / "gguf"
    fam_files = {"mol_z18": "set_mol_z18", "bulk_f32": "set_bulk_f32_12", "bulknacl_f32": "set_bulknacl_f32_12",
                 "multi": "set_multi", "scan_nacl": "set_scan_nacl", "mol_z18_field": "set_mol_z18_field",
                 "bulk_f32_field": "set_bulk_f32_field"}
    if d in CKPTS:
        m = re.match(r"(full_)?(\w+?)_cpu64(_first30)?\.npz$", name)
        if not m:
            return None
        full, fam = m.group(1), m.group(2)
        gg = g / "pbc" / f"{d}_lr_nobase.gguf" if full else g / f"{d}_direct_nobase.gguf"
        fr = root / meta["frames"]
        return gg, fr, nf
    if d == "seeded":
        p = meta.get("ckpt", "").split("/")[-1]
        fr = root / meta["frames"]
        if p == "tiny_qeq":
            return g / "qeq" / "tiny_qeq.gguf", fr, nf
        for cand in (g / "pbc" / f"seeded_{p}.gguf", g / f"seeded_{p}.gguf", g / "pbc" / f"{p}.gguf", g / f"{p}.gguf"):
            if cand.exists():
                return cand, fr, nf
        return None
    sub = {"seeded_branch": "branch", "seeded_cc": "cc", "seeded_thr": "thr", "seeded_les": None}.get(d, "skip")
    if sub == "skip":
        return None
    m = re.match(r"(.+?)_(mol_z18_field|bulk_f32_field|mol_z18|bulknacl_f32|bulk_f32|cluster|ion|multi|scan_nacl)"
                 r"_cpu64(\.ref)?\.npz$", name)
    if not m:
        return None
    p, fam = m.group(1), m.group(2)
    if sub is None:
        sub = "les_lr" if "_lr" in p else "les"
    gg = g / sub / f"{p}.gguf"
    if fam == "multi" and (g / sub / f"{p}_mc.gguf").exists():  # several components: the re-export with graph_multi
        gg = g / sub / f"{p}_mc.gguf"
    if "frames" in meta:
        fr = root / meta["frames"]
    else:
        base = fam_files.get(fam, f"set_{fam}")
        cand = root / "frames" / f"{base}_{nf}.extxyz"
        fr = cand if cand.exists() else root / "frames" / f"{base}.extxyz"
    return gg, fr, nf


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def compare(cpp, z):
    ap = z["atom_ptr"]
    out = {}

    def put(k, v):
        if v is not None and np.isfinite(v):
            out.setdefault(k, []).append(v)

    for f, c in enumerate(cpp):
        if f >= len(ap) - 1:
            break
        if "error" in c:
            out.setdefault("errors", []).append(1.0)
            continue
        A = slice(ap[f], ap[f + 1])
        e = z["atom/energy"][A] if "atom/energy" in z.files else z["energy"][A]
        put("E", abs(c["energy"] - e.sum()) / np.abs(e).sum())
        if "forces" in c and "forces" in z.files:
            put("F", rel(np.reshape(c["forces"], (-1, 3)), z["forces"][A]))
        periodic = "virial" in z.files and np.all(np.isfinite(z["virial"][f]))
        if "virial" in c and periodic:
            put("W", rel(np.reshape(c["virial"], (3, 3)), z["virial"][f]))
        qk = "atom/charges" if "atom/charges" in z.files else None
        if "charges" in c and qk:
            put("q", rel(c["charges"], z[qk][A]))
            if "struct/dipole" in z.files and not periodic:
                put("dipole", rel(c["dipole"], z["struct/dipole"][f]))
            if "int/fermi_level" in z.files:
                mu = z["int/fermi_level"][A][0]
                put("mu", abs(c["fermi_level"] - mu) / max(abs(mu), 1e-300))
            if "struct/metallicity" in z.files:
                mref = z["struct/metallicity"][f]
                put("metal", abs(c["metallicity"] - mref) / max(abs(mref), 1e-300))
            if "struct/charge_residual" in z.files and "charge_residual" in c:
                put("R", abs(c["charge_residual"] - z["struct/charge_residual"][f]) /
                    max(abs(z["struct/charge_residual"][f]), 1e-300))
            if "struct/dipole_flow" in z.files and "dipole_flow" in c:
                put("Dflow", rel(c["dipole_flow"], z["struct/dipole_flow"][f]))
    res = {k: (float(np.max(v)), float(np.median(v)), len(v)) for k, v in out.items() if k != "errors"}
    if "errors" in out:  # frames the model rejects (species outside its map, multi-component conserving charges)
        res["errors"] = (float(len(out["errors"])), 0.0, len(out["errors"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "data/plato-lr-scratch/pet_lr_port"))
    ap.add_argument("--eval", default=str(Path(__file__).resolve().parents[2] / "build-cpu/bin/mlipcpp-eval"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="out/fp64/sweep.txt")
    ap.add_argument("--json-dir", default="out/fp64/json")
    ap.add_argument("--env-f32", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    jd = root / a.json_dir
    jd.mkdir(parents=True, exist_ok=True)
    outp = root / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    refs = sorted((root / "ref").glob("*/*cpu64*.npz"))
    # a converted .ref.npz replaces its raw dump
    refs = [r for r in refs if not (r.name.endswith("_cpu64.npz") and r.with_name(r.name[:-4] + ".ref.npz").exists())]
    refs = [r for r in refs if "first30" not in r.name and "_20f" not in r.name]
    with open(outp, "a") as log:
        for r in refs:
            tag = str(r.relative_to(root / "ref"))
            if a.only and not re.search(a.only, tag):
                continue
            loc = locate(root, r)
            if not loc:
                print(f"skip {tag}", flush=True)
                continue
            gg, fr, nf = loc
            if not gg.exists() or not fr.exists():
                print(f"skip {tag}: missing {gg if not gg.exists() else fr}", flush=True)
                continue
            n = min(a.n, nf) if a.n > 0 else nf
            blocks = frames_text(fr, n)
            z = np.load(r)
            counts = np.diff(z["atom_ptr"])[:n]
            got = [int(b.split("\n", 1)[0]) for b in blocks]
            if list(counts) != got:
                print(f"skip {tag}: frame atom counts differ ({fr.name})", flush=True)
                continue
            tmp = jd / (tag.replace("/", "__") + ".extxyz")
            tmp.write_text("".join(blocks))
            js = jd / (tag.replace("/", "__") + (".env32" if a.env_f32 else "") + ".json")
            extra = list(a.extra)
            if "field" in tag and "bulk" in tag:  # his field force E . Z0 is open-structure only (raises otherwise)
                extra.append("--no-forces")
            cmd = [a.eval, str(gg), str(tmp), str(js)] + ([] if a.env_f32 else ["--fp64"]) + extra
            env = dict(os.environ)
            if a.env_f32:
                env["MLIPCPP_EXEC"] = "fp64"
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if p.returncode:
                line = f"{tag}: eval failed: {p.stderr.strip()[-300:]}"
            else:
                res = compare(json.load(open(js))["frames"], z)
                line = f"{tag} [{gg.relative_to(root)} n={n}] " + "  ".join(
                    (f"errors {int(v[0])}" if k == "errors" else f"{k} {v[0]:.1e}") for k, v in res.items())
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


if __name__ == "__main__":
    main()
