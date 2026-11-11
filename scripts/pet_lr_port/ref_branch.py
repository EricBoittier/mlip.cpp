#!/usr/bin/env python3
"""Oracle driver for M. Langer's iris BRANCHES (ipsum/PET-LR: stack/water-screening, feat/conserving-charges,
feat/long-range-transfer), which have no ASE calculator: one structure per batch through his own training-path
batch builder, iris.pet.sample.ToSample(...).map(atoms) + iris.pet.batching.pet_batcher(...) + records (as his
tests' make_batch, iris/pet/tests/test_flows_allocation.py), then iris.pet.predict.predict. Run in the branch's
env (kuma /scratch/boittier/ipsum/envs/iris-<branch>/bin/python), read-only on the checkpoint.

    python ref_branch.py CKPT_DIR FRAMES.extxyz OUT.npz [--x64] [--precision float32] [--max N]
                         [--cutoffs hard|differentiable] [--max-neighbors 59]

--x64: the batch is cast to float64 and the exact positions put back (his batch builder stores float32, so without
this the "fp64" oracle sees fp32-rounded positions: 4e-7 relative in E on the seeded lr_both ions; as ref_petlr.py).
Validated against the calculator oracle (ref_petlr.py, same env, same checkpoint): see PARITY.md.
Writes energy [S], forces [N, 3], atom_ptr, every per-atom / per-structure array predict returns (charges, ...),
and overflow [S] (his calculator would raise there; the batch trims to max_neighbors as training does).
Open structures (no cell) only for now.
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("frames"); ap.add_argument("out")
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--cutoffs", default="hard")
    ap.add_argument("--max-neighbors", type=int, default=59)
    a = ap.parse_args()
    import jax
    if a.x64:
        jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", a.precision)
    import jax.numpy as jnp
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read
    from marathon.data.properties import DEFAULT_PROPERTIES
    from marathon.grain import Record, RecordMetadata  # noqa: F401  (his tests' to_records)
    from marathon.io import from_dict, read_msgpack, read_yaml

    from iris.pet.batching import pet_batcher
    from iris.pet.predict import predict
    from iris.pet.sample import ToSample

    model = from_dict(read_yaml(f"{a.ckpt}/model/model.yaml"))
    params = read_msgpack(f"{a.ckpt}/model/model.msgpack")
    if a.x64:
        params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float64) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
                              else x, params)
    inputs = tuple(k for k in ("charge", "spin_multiplicity") if k in getattr(model, "inputs", ("charge",))) or ("charge",)
    props = {**DEFAULT_PROPERTIES, **{k: {"shape": (1,), "storage": "atoms.info"} for k in inputs}}
    keys = ("energy", "forces")
    nna = getattr(model, "num_neighbors_adaptive", None)
    prep = ToSample(cutoff=model.cutoff, cutoffs=a.cutoffs, max_neighbors=a.max_neighbors, inputs=inputs, keys=keys,
                    properties=props, with_lr=True, num_k=16,
                    **({"num_neighbors_adaptive": nna} if a.cutoffs != "hard" else {}))
    from petjax.utils import cast_floats
    # device arrays: a numpy leaf indexed by a traced array fails under jit (the species tables; as ref_petlr.py)
    params = jax.tree.map(jnp.asarray, params)
    run = jax.jit(lambda b: predict(model, params, b))
    frames = read(a.frames, ":")
    if a.max:
        frames = frames[:a.max]
    out, ptr = {}, [0]
    t0 = time.time()
    for f, at in enumerate(frames):
        if at.pbc.any():
            raise ValueError("ref_branch.py: open structures only")
        at.info.setdefault("charge", 0.0)
        at.info.setdefault("spin_multiplicity", 1.0)
        n = len(at)
        at.calc = SinglePointCalculator(at, energy=0.0, forces=np.zeros((n, 3)))
        batcher = pet_batcher(inputs=inputs, keys=keys, properties=props, n_structures=2, n_atoms=n + 1,
                              n_pairs=max(64, n * a.max_neighbors + 1), max_neighbors=a.max_neighbors,
                              cutoffs=a.cutoffs,
                              lr={"n_pairs": 64, "n_structures_pbc": 1, "n_atoms_pbc": 8,
                                  "n_pairs_nonpbc": max(64, n * (n - 1) // 2 + 1), "num_k": 16, "BM": 8, "BK": 16})
        from marathon.grain import Record as R, RecordMetadata as M
        (rec,) = list(batcher(iter([R(metadata=M(index=0, record_key=0), data=prep.map(at))])))
        batch = jax.tree.map(jnp.asarray, rec.data)
        if a.x64:  # ToSample / pet_batcher build float32 batches: cast, then put back the exact positions (open)
            batch = cast_floats(batch, jnp.float64)
            pos = np.array(np.asarray(batch["positions"]), dtype=np.float64)
            pos[:n] = at.get_positions()
            batch = {**batch, "positions": jnp.asarray(pos)}
        res = run(batch)
        res = jax.tree.map(np.asarray, res)
        for k, v in res.items():
            v = np.asarray(v)
            if v.ndim >= 1 and v.shape[0] >= n and k not in ("energy",):
                out.setdefault(k, []).append(v[:n])
            else:
                out.setdefault(k, []).append(v.reshape(-1)[:1])
        ptr.append(ptr[-1] + n)
        if f % 10 == 0:
            print(f"{f}/{len(frames)} n={n} {time.time() - t0:.1f}s", flush=True)
    np.savez(a.out, atom_ptr=np.array(ptr), **{k: np.concatenate(v) for k, v in out.items()})
    print("keys:", sorted(out))


if __name__ == "__main__":
    main()
