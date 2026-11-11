#!/usr/bin/env python3
"""Oracle driver for M. Langer's iris branch feat/conserving-charges (a98c2f6; stack/water-screening bb5a5ea has the
same model code): one structure per batch through that branch's own training-path batch builder (KnnToSample +
pet_batcher, as its tests' make_batch in iris/pet/tests/test_conserving.py), then iris.pet.predict.predict. Run in
the branch env (kuma /scratch/boittier/ipsum/envs/iris-conserving-charges/bin/python), read-only on the checkpoint.

    python ref_branch_cc.py CKPT_DIR FRAMES.extxyz OUT.npz [--x64] [--precision float32] [--max N] [--num-k N]

conserving_charges needs fixed neighbour lists (the model's fixed_neighbors: KnnToSample at that width, every pair
within the cutoff unless an atom has more) and component charges: his sample prep builds the connected components of
the mutual edge graph and raises on a structure of more than one component without fragment labels; such frames are
recorded as NaN (error [S] = 1). --x64: cast to float64 and the exact positions put back (open structures); periodic
frames are rounded to float32 first (his batch stores float32 positions and cells on both the SR and the LR side).
Writes energy [S], forces [N, 3], atom_ptr, every per-atom / per-structure array predict returns, virial [S, 3, 3] on
periodic frames (his predict's stress, dU/d eps).
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
    ap.add_argument("--num-k", type=int, default=1024)
    a = ap.parse_args()
    import jax
    if a.x64:
        jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", a.precision)
    import jax.numpy as jnp
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read
    from marathon.data.properties import DEFAULT_PROPERTIES
    from marathon.grain import Record as R, RecordMetadata as M
    from marathon.io import from_dict, read_msgpack, read_yaml
    from petjax.utils import cast_floats

    from iris.pet.batching import pet_batcher
    from iris.pet.predict import predict
    from iris.pet.sample import KnnToSample

    model = from_dict(read_yaml(f"{a.ckpt}/model/model.yaml"))
    params = read_msgpack(f"{a.ckpt}/model/model.msgpack")
    if a.x64:
        params = cast_floats(params, jnp.float64)
    params = jax.tree.map(jnp.asarray, params)
    if not model.conserving_charges or model.fixed_neighbors is None:
        raise ValueError("ref_branch_cc.py: a conserving_charges model with fixed_neighbors")
    width = int(model.fixed_neighbors)
    # the uniform external field (his `field` input, atoms.info["field"] [3], V/A) when the model reads it
    inputs = ("charge", "field") if "field" in (getattr(model, "inputs", None) or ()) else ("charge",)
    props = {**DEFAULT_PROPERTIES, "charge": {"shape": (1,), "storage": "atoms.info"},
             "field": {"shape": (3,), "storage": "atoms.info"}}
    keys = ("energy", "forces")
    prep = KnnToSample(cutoff=model.cutoff, num_neighbors=width, inputs=inputs, keys=keys, properties=props,
                       with_lr=True, num_k=a.num_k, conserving_charges=True,
                       charge_state_min=model.charge_state_min, charge_state_max=model.charge_state_max)
    run = jax.jit(lambda b: predict(model, params, b))
    run_pbc = jax.jit(lambda b: predict(model, params, b, stress=True))
    field_forces = None
    if "field" in inputs:  # his field force E . Z0 on the pass-1 charges (predict adds it to `forces`), on its own
        from iris.pet.predict import _field_forces
        field_forces = jax.jit(lambda b: _field_forces(model, params, b, b["inputs"]["field"]))
    frames = read(a.frames, ":")
    if a.max:
        frames = frames[:a.max]
    out, ptr, err = {}, [0], []
    t0 = time.time()
    up = lambda x, m: int(-(-x // m) * m)
    for f, at in enumerate(frames):
        at.info.setdefault("charge", 0.0)
        if "field" in inputs:
            at.info.setdefault("field", np.zeros(3))
        n = len(at)
        pbc = bool(at.pbc.any())
        if pbc:
            at.set_cell(at.get_cell().array.astype(np.float32).astype(np.float64))
            at.set_positions(at.get_positions().astype(np.float32).astype(np.float64))
        at.calc = SinglePointCalculator(at, energy=0.0, forces=np.zeros((n, 3)))
        try:
            smp = prep.map(at)
        except ValueError as e:  # e.g. more than one component without fragment labels
            print(f"{f}: {e}", flush=True)
            err.append(1); ptr.append(ptr[-1] + n)
            for k, v in out.items():
                v.append(np.full((n,) + v[-1].shape[1:], np.nan) if v[-1].shape[0] > 1 or k in ("forces", "charges")
                         else np.full(v[-1].shape, np.nan))
            continue
        lr = {"n_pairs": 64, "n_structures_pbc": 1, "n_atoms_pbc": 8, "n_pairs_nonpbc": max(64, n * (n - 1) // 2 + 1),
              "num_k": 16, "BM": 8, "BK": 16}
        if pbc:
            n_lr = len(np.asarray(smp.lr_structure["centers"]))
            lr = {"n_pairs": up(n_lr + 1, 4096), "n_structures_pbc": 2, "n_atoms_pbc": up(n, 8) + 16,
                  "n_pairs_nonpbc": 64, "num_k": up(4 * a.num_k, 16), "BM": 8, "BK": 16}
        batcher = pet_batcher(inputs=inputs, keys=keys, properties=props, n_structures=2, n_atoms=n + 1,
                              n_pairs=max(64, n * width + 1), k_sel=width, lr=lr)
        (rec,) = list(batcher(iter([R(metadata=M(index=0, record_key=0), data=smp)])))
        batch = jax.tree.map(jnp.asarray, rec.data)
        if a.x64:
            batch = cast_floats(batch, jnp.float64)
            if not pbc:
                pos = np.array(np.asarray(batch["positions"]), dtype=np.float64)
                pos[:n] = at.get_positions()
                batch = {**batch, "positions": jnp.asarray(pos)}
        res = (run_pbc if pbc else run)(batch)
        if pbc:
            st = res.pop("stress")
            res = {**res, "virial": np.asarray(st)[:1]}
        else:
            res.pop("stress", None)
        if field_forces is not None:
            res = {**res, "field_forces": field_forces(batch)}
        res = jax.tree.map(np.asarray, res)
        err.append(0)
        for k, v in res.items():
            v = np.asarray(v)
            if k in ("dipole_flow", "dipole_induced"):  # per-structure vectors [S, 3] (before: the first component only)
                out.setdefault(k, []).append(v.reshape(-1, 3)[:1])
            elif k == "virial":
                out.setdefault(k, []).append(v.reshape(1, 3, 3))
            elif v.ndim >= 1 and v.shape[0] >= n and k != "energy":
                out.setdefault(k, []).append(v[:n])
            else:
                out.setdefault(k, []).append(v.reshape(-1)[:1])
        ptr.append(ptr[-1] + n)
        if f % 10 == 0:
            print(f"{f}/{len(frames)} n={n} {time.time() - t0:.1f}s", flush=True)
    np.savez(a.out, atom_ptr=np.array(ptr), error=np.array(err), **{k: np.concatenate(v) for k, v in out.items()})
    print("keys:", sorted(out), "errors:", int(np.sum(err)))


if __name__ == "__main__":
    main()
