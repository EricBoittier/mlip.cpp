#!/usr/bin/env python3
"""Training-step throughput of mmml's SpookyPhysNet in JAX (energy + force loss, AdamW).

The reference for mlipcpp-train's struct/s on the same structures and model size.
Batching follows the OMol pilot (configs/pilot.yaml): structures bucketed by
atom count in widths of 8 (pad = 8, 16, ..., 120), per-device batch
B = min(256, max_pairs // pad^2) - 1, all pairs within each padded molecule
(padding pairs masked), one jitted value_and_grad + AdamW update per step.
Steps are drawn from the buckets in proportion to their structure counts, so
struct/s is the data-weighted mean of what an epoch would see.

    python scripts/mmml/bench_train_jax.py --data omol_pilot/structures.npz --steps 200 [--max-pairs 210000]

Needs mmml (JAX, e3x, flax, optax) on PYTHONPATH. Prints JSON.
"""

import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="structures.npz from omol_npz_to_memmap.py")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=2, help="untimed steps per bucket (compilation)")
    ap.add_argument("--max-pairs", type=int, default=210000)
    ap.add_argument("--batch-cap", type=int, default=256)
    ap.add_argument("--bucket-width", type=int, default=8)
    ap.add_argument("--features", type=int, default=64)
    ap.add_argument("--max-degree", type=int, default=1)
    ap.add_argument("--num-iterations", type=int, default=3)
    ap.add_argument("--num-basis-functions", type=int, default=32)
    ap.add_argument("--n-res", type=int, default=2)
    ap.add_argument("--forces-weight", type=float, default=52.91)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import optax

    from mmml_ref import build_model

    d = np.load(a.data)
    N = d["N"].astype(int)
    off = np.concatenate([[0], np.cumsum(N)])
    config = dict(features=a.features, max_degree=a.max_degree, num_iterations=a.num_iterations,
                  num_basis_functions=a.num_basis_functions, cutoff=6.0, max_atomic_number=87, charges=False,
                  n_res=a.n_res, zbl=True, use_energy_bias=True, efa=False, model_type="spooky")
    model = build_model(config, "spooky", natoms=8)

    buckets = {}
    for i, n in enumerate(N):
        pad = int(np.ceil(n / a.bucket_width) * a.bucket_width)
        buckets.setdefault(pad, []).append(i)
    pads = sorted(buckets)

    def batch_size(pad):
        return max(1, min(a.batch_cap, a.max_pairs // (pad * pad)) - 1)

    def make_batch(pad, idx):
        B = len(idx)
        Z = np.zeros((B, pad), np.int32)
        R = np.zeros((B, pad, 3))
        F = np.zeros((B, pad, 3))
        am = np.zeros((B, pad))
        for b, i in enumerate(idx):
            n = N[i]
            Z[b, :n], R[b, :n], F[b, :n], am[b, :n] = d["Z"][off[i]:off[i + 1]], d["R"][off[i]:off[i + 1]], \
                d["F"][off[i]:off[i + 1]], 1
            R[b, n:] = 100.0 + 10.0 * np.arange(pad - n)[:, None]  # padding atoms far away
        ii, jj = np.nonzero(~np.eye(pad, dtype=bool))
        dst = (np.arange(B)[:, None] * pad + ii[None]).reshape(-1)
        src = (np.arange(B)[:, None] * pad + jj[None]).reshape(-1)
        amf = am.reshape(-1)
        pm = amf[dst] * amf[src]
        Q = np.repeat(d["Q"][idx].astype(float), pad)
        S = np.repeat(d["S"][idx].astype(float), pad)
        return dict(Z=Z.reshape(-1), R=R.reshape(-1, 3).astype(np.float32), F=F.reshape(-1, 3).astype(np.float32),
                    E=d["E"][idx].astype(np.float32), am=amf.astype(np.float32), pm=pm.astype(np.float32),
                    dst=dst.astype(np.int32), src=src.astype(np.int32), seg=np.repeat(np.arange(B), pad).astype(np.int32),
                    Q=Q.astype(np.float32), S=S.astype(np.float32), B=B)

    key = jax.random.PRNGKey(0)
    b0 = make_batch(pads[0], buckets[pads[0]][:batch_size(pads[0])])
    params = model.init(key, b0["Z"], b0["Q"], b0["S"], b0["R"], b0["dst"], b0["src"], batch_segments=b0["seg"],
                        batch_size=b0["B"], batch_mask=b0["pm"], atom_mask=b0["am"])
    opt = optax.adamw(5e-4)
    state = opt.init(params)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    def loss_fn(p, bt, B):
        out = model.apply(p, bt["Z"], bt["Q"], bt["S"], bt["R"], bt["dst"], bt["src"], batch_segments=bt["seg"],
                          batch_size=B, batch_mask=bt["pm"], atom_mask=bt["am"], compute_forces=True)
        e = out["energy"].reshape(-1)
        le = jnp.mean((e - bt["E"]) ** 2)
        lf = jnp.sum(((out["forces"] - bt["F"]) * bt["am"][:, None]) ** 2) / (3 * jnp.sum(bt["am"]))
        return le + a.forces_weight * lf

    steps = {}

    def step_fn(B):
        if B not in steps:
            def f(p, s, bt):
                l, g = jax.value_and_grad(loss_fn)(p, bt, B)
                u, s = opt.update(g, s, p)
                return optax.apply_updates(p, u), s, l
            steps[B] = jax.jit(f)
        return steps[B]

    rng = np.random.default_rng(0)
    weights = np.array([len(buckets[p]) for p in pads], float)
    weights /= weights.sum()
    plan = rng.choice(len(pads), a.steps, p=weights)
    # compile (untimed) every bucket that is drawn
    t_compile = time.perf_counter()
    for k in sorted(set(plan)):
        pad = pads[k]
        B = min(batch_size(pad), len(buckets[pad]))
        bt = {kk: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for kk, v in make_batch(pad, buckets[pad][:B]).items()}
        for _ in range(a.warmup):
            params, state, l = step_fn(B)(params, state, {kk: v for kk, v in bt.items() if kk != "B"})
            jax.block_until_ready(l)
    t_compile = time.perf_counter() - t_compile
    n_struct = n_atoms = 0
    t_step = 0.0
    for k in plan:
        pad = pads[k]
        B = min(batch_size(pad), len(buckets[pad]))
        idx = rng.choice(buckets[pad], B, replace=False)
        bt = {kk: jnp.asarray(v) for kk, v in make_batch(pad, idx).items() if kk != "B"}  # host prep untimed
        t0 = time.perf_counter()
        params, state, l = step_fn(B)(params, state, bt)
        jax.block_until_ready(l)
        t_step += time.perf_counter() - t0
        n_struct += B
        n_atoms += int(N[idx].sum())
    res = dict(backend=jax.default_backend(), device=str(jax.devices()[0]), n_params=int(n_params), steps=a.steps,
               structures=n_struct, atoms=n_atoms, seconds=t_step, struct_per_s=n_struct / t_step,
               atoms_per_s=n_atoms / t_step, mean_batch=n_struct / a.steps, compile_s=t_compile,
               max_pairs=a.max_pairs, final_loss=float(l))
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
