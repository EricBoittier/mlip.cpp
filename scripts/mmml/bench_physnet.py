#!/usr/bin/env python3
"""Inference speed: mlip.cpp (ggml) vs mmml (JAX, jit) on the same PhysNet / Spooky weights.

Workloads (energy + forces):
  single   one water, methanol and octanol molecule per call
  dimers   N water-methanol dimers per call (mmml ML/MM: all pairs within each dimer)
  omol     a batch of OMol25 molecules with charge / spin (Spooky checkpoints)

    MLIPCPP_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=1 \
    taskset -c 0-7 python scripts/mmml/bench_physnet.py --checkpoint student.json --backend cpu --json out.json
    python scripts/mmml/bench_physnet.py --checkpoint student.json --backend cuda   # GPU node (JAX on GPU too)

JAX: one jitted call over fixed index arrays (the pairs are built once, as in
mmml's ML/MM); timed with block_until_ready. mlip.cpp: Predictor.predict /
predict_batch from numpy positions, so its time includes the neighbour list,
input packing, the graph and the force gather. Median of --reps calls after warm-up.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mmml"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))


def timeit(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t)), 1e3 * float(np.min(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n-dimers", type=int, nargs="+", default=[1, 10, 100, 1000])
    ap.add_argument("--n-omol", type=int, default=256)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-jax", action="store_true")
    ap.add_argument("--no-cpp", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.backend == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import test_physnet_parity as tp  # dev-package path setup, structure builders
    from convert_mmml import to_gguf
    from mmml_ref import JaxRef

    rng = np.random.default_rng(0)
    mons = {k: tp._read_xyz(tp.MONOMERS / f"{k}.xyz") for k in ("water", "methanol", "octanol")}

    def dimer():
        (Za, Ra), (Zb, Rb) = mons["water"], mons["methanol"]
        Ra = (Ra - Ra.mean(0)) @ tp._rotation(rng).T
        Rb = (Rb - Rb.mean(0)) @ tp._rotation(rng).T + rng.normal(size=3) / np.sqrt(3) * rng.uniform(2.8, 5.0)
        return dict(Z=np.concatenate([Za, Zb]), R=np.concatenate([Ra, Rb]))

    res = {"checkpoint": a.checkpoint, "backend": a.backend, "threads": os.environ.get("MLIPCPP_NUM_THREADS"),
           "single": {}, "dimers": {}, "omol": {}}
    model_type = None
    with tempfile.TemporaryDirectory() as td:
        gguf = Path(td) / "m.gguf"
        info = to_gguf(a.checkpoint, str(gguf))
        model_type = info["model_type"]
        cpp = None
        if not a.no_cpp:
            import mlipcpp

            opts = mlipcpp.ModelOptions()
            opts.backend = mlipcpp.Backend.CUDA if a.backend == "cuda" else mlipcpp.Backend.CPU
            cpp = mlipcpp.Predictor(str(gguf), opts)
        ref = None if a.no_jax else JaxRef(a.checkpoint)
        if ref is not None:
            import jax
            import jax.numpy as jnp

        def jax_fn(mols):
            Z, R, seg, dst, src, Q, S = ref._batch(mols)
            f = ref.fn(len(Z), len(dst), len(mols), True)
            args = [ref.params, jnp.asarray(Z), jnp.asarray(R, ref.dtype), jnp.asarray(seg), jnp.asarray(dst),
                    jnp.asarray(src), jnp.asarray(Q, ref.dtype), jnp.asarray(S, ref.dtype)]
            return lambda: jax.block_until_ready(f(*args))

        def cpp_batch(mols):
            pos = np.ascontiguousarray(np.concatenate([m["R"] for m in mols]).astype(np.float32))
            Z = np.concatenate([m["Z"] for m in mols]).astype(np.int32)
            n = np.array([len(m["Z"]) for m in mols], np.int32)
            q = np.array([m.get("Q", 0) for m in mols], np.int32)
            s = np.array([m.get("S", 1) for m in mols], np.int32)
            return lambda: cpp.predict_batch(pos, Z, n, q, s, True)

        for name, (Z, R) in mons.items():
            m = dict(Z=Z, R=R)
            row = {"n_atoms": len(Z)}
            if cpp:
                pos = np.ascontiguousarray(R.astype(np.float32))
                row["cpp_ms"] = timeit(lambda: cpp.predict(pos, Z.astype(np.int32)), a.reps)
            if ref:
                row["jax_ms"] = timeit(jax_fn([m]), a.reps)
            res["single"][name] = row
            print("single", name, row, flush=True)
        for nd in a.n_dimers:
            mols = [dimer() for _ in range(nd)]
            row = {"n_atoms": int(sum(len(m["Z"]) for m in mols))}
            reps = max(3, a.reps // (1 + nd // 100))
            if cpp:
                row["cpp_ms"] = timeit(cpp_batch(mols), reps)
            if ref:
                row["jax_ms"] = timeit(jax_fn(mols), reps)
            res["dimers"][nd] = row
            print("dimers", nd, row, flush=True)
        if model_type == "spooky" and a.n_omol:
            mols = tp.omol_structures(a.n_omol, max_atoms=60)
            row = {"n_mols": len(mols), "n_atoms": int(sum(len(m["Z"]) for m in mols))}
            if cpp:
                row["cpp_ms"] = timeit(cpp_batch(mols), max(3, a.reps // 4))
            if ref:
                row["jax_ms"] = timeit(jax_fn(mols), max(3, a.reps // 4))
            res["omol"] = row
            print("omol", row, flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
