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
Periodic structures (3-D cells; his ToSample raises on a slab with the long-range transfer): the jax-pme tiled
Ewald LR batch at --num-k (default: the checkpoint's config.yaml to_sample.num_k, else 1024), and the stress (his
predict's virial dU/d eps, [S, 3, 3]) as `virial`. His batch builder stores float32 positions and cells on BOTH the SR
and the LR side, so for an exact --x64 oracle on periodic frames the frame is rounded to float32 first (positions and
cell; the script rounds them itself): the float32 batch is then exact.
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
    ap.add_argument("--num-k", type=int, default=0)
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
    num_k = a.num_k
    if not num_k:
        import os
        cfgp = f"{a.ckpt}/config.yaml"
        num_k = (read_yaml(cfgp).get("to_sample", {}) or {}).get("num_k", 1024) if os.path.exists(cfgp) else 1024
    # model-owned prep parameters (iris.train.model_owned_params: ToSample.MODEL_PARAMS read off the model)
    owned = {k: getattr(model, k) for k in getattr(ToSample, "MODEL_PARAMS", ()) if hasattr(model, k)
             and k not in ("cutoff", "num_neighbors_adaptive")}
    if a.cutoffs == "hard":
        owned.pop("cutoff_width_adaptive", None), owned.pop("adaptive_cutoff_method", None)
    prep = ToSample(cutoff=model.cutoff, cutoffs=a.cutoffs, max_neighbors=a.max_neighbors, inputs=inputs, keys=keys,
                    properties=props, with_lr=True, num_k=num_k,
                    **({"num_neighbors_adaptive": nna} if a.cutoffs != "hard" else {}), **owned)
    from petjax.utils import cast_floats
    # device arrays: a numpy leaf indexed by a traced array fails under jit (the species tables; as ref_petlr.py)
    params = jax.tree.map(jnp.asarray, params)
    run = jax.jit(lambda b: predict(model, params, b))
    run_pbc = jax.jit(lambda b: predict(model, params, b, stress=True))
    frames = read(a.frames, ":")
    if a.max:
        frames = frames[:a.max]
    out, ptr = {}, [0]
    t0 = time.time()
    for f, at in enumerate(frames):
        pbc = bool(at.pbc.any())
        if pbc:  # his batch stores float32 positions / cells (SR and LR side): the oracle sees the float32 rounding
            # of both in every precision (mlip.cpp reads them as float32 too), so the --x64 batch is exact
            at.set_cell(at.get_cell().array.astype(np.float32).astype(np.float64))
            at.set_positions(at.get_positions().astype(np.float32).astype(np.float64))
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
        if pbc:  # the tiled-Ewald LR budget sized on this frame (real-space pairs, atoms, k vectors; BM / BK multiples)
            smp = prep.map(at)
            ls = smp.lr_structure
            n_lr = len(np.asarray(ls["centers"]))
            up = lambda x, m: int(-(-x // m) * m)
            batcher = pet_batcher(inputs=inputs, keys=keys, properties=props, n_structures=2, n_atoms=n + 1,
                                  n_pairs=max(64, n * a.max_neighbors + 1), max_neighbors=a.max_neighbors,
                                  cutoffs=a.cutoffs,
                                  lr={"n_pairs": up(n_lr + 1, 4096), "n_structures_pbc": 2, "n_atoms_pbc": up(n, 8) + 16,
                                      "n_pairs_nonpbc": 64, "num_k": up(4 * num_k, 16), "BM": 8, "BK": 16})
            data = smp
        else:
            data = prep.map(at)
        (rec,) = list(batcher(iter([R(metadata=M(index=0, record_key=0), data=data)])))
        batch = jax.tree.map(jnp.asarray, rec.data)
        if a.x64:  # ToSample / pet_batcher build float32 batches: cast, then put back the exact positions (open)
            batch = cast_floats(batch, jnp.float64)
            pos = np.array(np.asarray(batch["positions"]), dtype=np.float64)
            pos[:n] = at.get_positions()
            batch = {**batch, "positions": jnp.asarray(pos)}
        res = (run_pbc if pbc else run)(batch)
        if pbc:
            st = res.pop("stress")
            res = {**res, "virial": np.asarray(st)[:1]}
        elif "stress" in res:
            res.pop("stress")
        res = jax.tree.map(np.asarray, res)
        for k, v in res.items():
            v = np.asarray(v)
            if k == "virial":
                out.setdefault(k, []).append(v.reshape(1, 3, 3))
            elif v.ndim >= 1 and v.shape[0] >= n and k not in ("energy",):
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
