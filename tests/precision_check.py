#!/usr/bin/env python3
"""Per-model CUDA matmul precision in one process: runs mlipcpp-precision-check in
several processes on one structure and compares them.

  precision_check.py <mlipcpp-precision-check> <model.gguf> <structure.xyz> <work dir>
                     [--backend cuda] [--reps R] [--json out.json] [--ragged]

Processes (each loads every listed instance before its first predict):
  fp32ref   env clean, [default]                    pure FP32 process (today's default)
  tf32ref   MLIPCPP_TF32=1, [default]               the TF32 process of today
  mixed     env clean, [fp32, tf32]                 the two precisions side by side
  default2  env clean, [default, default] + a TF32 load after the predicts (must fail on CUDA)
  strict    MLIPCPP_TF32=1, [fp32]                  an FP32 model in a TF32 process (one instance)

Checks: mixed.fp32 = fp32ref and strict.fp32 = fp32ref to <= 1e-6 eV/atom and <= 1e-5 eV/A;
mixed.tf32 = tf32ref to the same; default2 = fp32ref (bitwise expected); on CUDA the late TF32
load fails. On the CPU every process must agree (precision is a CUDA setting).
Exit 0 on success, 1 on a failed check, 77 (skip) without the binary or the model.
"""
import argparse
import json
import os
import subprocess
import sys

try:
    import numpy as np
except ImportError:  # the ctest entry skips
    print("skip: numpy missing")
    sys.exit(77)

TOL_E, TOL_F = 1e-6, 1e-5  # eV/atom, eV/A


def run(exe, model, xyz, work, tag, precs, env_add, args, extra=()):
    env = {k: v for k, v in os.environ.items() if k not in ("MLIPCPP_TF32", "GGML_CUDA_NO_TF32",
                                                            "NVIDIA_TF32_OVERRIDE")}
    if args.ragged:
        env["MLIPCPP_RAGGED"] = "1"
    else:
        env.pop("MLIPCPP_RAGGED", None)
    env.update(env_add)
    prefix = os.path.join(work, tag)
    cmd = [exe, model, xyz, prefix, ",".join(precs), "--backend", args.backend, "--reps", str(args.reps),
           "--warmup", str(args.warmup), *extra]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise RuntimeError(f"{tag}: exit {p.returncode}")
    lines = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    out = {"rows": [l for l in lines if "instance" in l], "extra": [l for l in lines if "instance" not in l]}
    out["data"] = []
    for k, prec in enumerate(precs):
        with open(f"{prefix}.{k}.{prec}.txt") as f:
            n = int(f.readline())
            e = float(f.readline())
            F = np.loadtxt(f).reshape(n, 3)
        out["data"].append((n, e, F))
    return out


def diff(a, b):
    n = a[0]
    return abs(a[1] - b[1]) / n, float(np.abs(a[2] - b[2]).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe")
    ap.add_argument("model")
    ap.add_argument("xyz")
    ap.add_argument("work")
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ragged", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    if not (os.path.exists(args.exe) and os.path.exists(args.model) and os.path.exists(args.xyz)):
        print("skip: binary, model or structure missing")
        return 77
    os.makedirs(args.work, exist_ok=True)
    R = {}
    R["fp32ref"] = run(args.exe, args.model, args.xyz, args.work, "fp32ref", ["default"], {}, args)
    R["tf32ref"] = run(args.exe, args.model, args.xyz, args.work, "tf32ref", ["default"], {"MLIPCPP_TF32": "1"}, args)
    R["mixed"] = run(args.exe, args.model, args.xyz, args.work, "mixed", ["fp32", "tf32"], {}, args)
    R["default2"] = run(args.exe, args.model, args.xyz, args.work, "default2", ["default", "default"], {}, args,
                        ["--late-tf32"])
    R["strict"] = run(args.exe, args.model, args.xyz, args.work, "strict", ["fp32"], {"MLIPCPP_TF32": "1"}, args)
    cuda = args.backend == "cuda" or "CUDA" in R["fp32ref"]["rows"][0]["backend"]

    ref32, ref_tf = R["fp32ref"]["data"][0], R["tf32ref"]["data"][0]
    checks = []

    def check(name, a, b, tol_e=TOL_E, tol_f=TOL_F):
        de, df = diff(a, b)
        ok = de <= tol_e and df <= tol_f
        checks.append({"check": name, "dE_eV_per_atom": de, "dF_max_eV_per_A": df, "ok": ok})

    check("mixed.fp32 vs fp32ref", R["mixed"]["data"][0], ref32)
    check("strict.fp32 vs fp32ref", R["strict"]["data"][0], ref32)
    check("mixed.tf32 vs tf32ref", R["mixed"]["data"][1], ref_tf)
    check("default2[0] vs fp32ref", R["default2"]["data"][0], ref32, 0.0, 0.0)
    check("default2[1] vs fp32ref", R["default2"]["data"][1], ref32, 0.0, 0.0)
    # information: the TF32 error itself (CUDA ~0.2-0.7 meV/atom; the CPU 0)
    de, df = diff(ref_tf, ref32)
    checks.append({"check": "tf32ref vs fp32ref (TF32 error, info)", "dE_eV_per_atom": de, "dF_max_eV_per_A": df,
                   "ok": True})
    if not cuda:
        check("tf32ref vs fp32ref (CPU: no-op)", ref_tf, ref32, 0.0, 0.0)
    late = R["default2"]["extra"][0] if R["default2"]["extra"] else {}
    if cuda:
        checks.append({"check": "late TF32 load fails in an FP32 process", "ok": bool(late.get("late_tf32_failed")),
                       "error": late.get("error", "")})
    else:
        checks.append({"check": "late TF32 load is a no-op on the CPU", "ok": not late.get("late_tf32_failed", True)})

    timing = {tag: [{"precision": r["precision"], "ms_median": r["ms_median"]} for r in R[tag]["rows"]] for tag in R}
    for c in checks:
        extra = f" dE={c['dE_eV_per_atom']:.3e} eV/atom dFmax={c['dF_max_eV_per_A']:.3e} eV/A" \
            if "dE_eV_per_atom" in c else f" {c.get('error', '')[:120]}"
        print(f"[{'ok' if c['ok'] else 'FAIL'}] {c['check']}:{extra}")
    for tag, rows in timing.items():
        print(f"  {tag}: " + ", ".join(f"{r['precision']} {r['ms_median']:.2f} ms" for r in rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"checks": checks, "timing": timing, "n_atoms": ref32[0], "ragged": args.ragged,
                       "backend": R["fp32ref"]["rows"][0]["backend"]}, f, indent=1)
    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
