#!/usr/bin/env python3
"""Oracle dump of a full iris PETLR (ipsum/PET-LR, after M. Langer) with his own code, for the LR-branch port:
the batch of `iris.pet.calculator.Calculator` (its neighbour lists, Ewald parameters and masking), then
`iris.pet.predict.predict` for energy / forces / stress and a direct `model.apply` capturing every sown value.
Runs in his env (kuma /scratch/boittier/ipsum/env).

    python ref_petlr.py CKPT_DIR FRAMES.extxyz OUT.npz [--x64] [--precision float32] [--max N]

Per frame (atom axis concatenated, offsets `atom_ptr`): total energy (no composition baseline), per-atom energies,
forces, virial (periodic), and every value the model sows / emits:
  per_atom: charges; per_structure: charge, dipole, metallicity, delta_charge, qeq_residual, ...;
  intermediates: scores, fermi_level, charges_base, transfers, charges_pass1, delta_charges, chi, hardness, ...
(names as sown, prefixed `int/`, `atom/`, `struct/`). --x64 casts the calculator's batch and the parameters
to float64 (the calculator itself builds float32 batches).
"""
import argparse
import json
import platform
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("frames"); ap.add_argument("out")
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--precision", default="float32")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()

    import jax
    if a.x64:
        jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", a.precision)
    import jax.numpy as jnp
    from ase.io import read
    from flax.traverse_util import flatten_dict
    from petjax.utils import cast_floats

    from iris.pet.calculator import Calculator
    from iris.pet.predict import model_inputs, predict

    # discretize: padded shape buckets (his option), so compilations are reused across frames
    calc = Calculator.from_checkpoint(a.ckpt, discretize=True, info_keys={"spin_multiplicity": "spin_multiplicity"})
    model = calc.model
    params = calc.params
    fdt = jnp.float64 if a.x64 else jnp.float32
    if a.x64:
        params = cast_floats(params, jnp.float64)

    def run(batch):
        res = predict(model, params, batch, stress=True)
        truncated, lr, _ = model_inputs(model, batch)
        out, state = model.apply(params, truncated, lr,
                                 mutable=["intermediates", "per_atom", "per_structure"])
        return res, out, state

    run = jax.jit(run)
    frames = read(a.frames, ":")
    if a.max:
        frames = frames[: a.max]
    rows, atom_ptr, t0 = {}, [0], time.time()
    for k, atoms in enumerate(frames):
        if "spin_multiplicity" not in atoms.info:
            atoms.info["spin_multiplicity"] = 1
        calc._reset()
        calc._system = None
        if calc._lr and calc._lr_params is None:
            calc._lr_params = calc._freeze(atoms)
        calc._rebuild(atoms)
        lr_cutoff = jnp.float32((calc._lr_params or {}).get("cutoff", 0.0))
        batch = calc._masked(calc._batch, lr_cutoff)
        batch = cast_floats(batch, fdt)
        if a.x64:  # the calculator builds float32 batches: put back the exact positions, cell and inputs
            pos = np.array(np.asarray(batch["positions"]), dtype=np.float64)
            pos[: len(atoms)] = atoms.get_positions()
            cell = np.array(np.asarray(batch["cell"]), dtype=np.float64)
            if atoms.pbc.any():
                cell[0] = atoms.get_cell().array
            inputs = dict(batch.get("inputs", {}))
            if "dipole_origin" in inputs:
                d0 = np.array(np.asarray(inputs["dipole_origin"]), dtype=np.float64)
                d0[0] = atoms.info["dipole_origin"] if "dipole_origin" in atoms.info else atoms.get_center_of_mass()
                inputs["dipole_origin"] = jnp.asarray(d0)
            batch = {**batch, "positions": jnp.asarray(pos), "cell": jnp.asarray(cell), "inputs": inputs}
        res, out, state = run(batch)
        n = len(atoms)
        NA, NS = batch["atom_mask"].shape[0], batch["structure_mask"].shape[0]
        per_atom = np.asarray(out["energy"] if isinstance(out, dict) else out)
        frame = {"energy": np.asarray(res["energy"])[:1], "atom/energy": per_atom[:n],
                 "forces": np.asarray(res["forces"])[:n],
                 "virial": np.asarray(res["stress"])[:1] if atoms.pbc.any() else np.full((1, 3, 3), np.nan)}
        for coll, tag in (("per_atom", "atom/"), ("per_structure", "struct/"), ("intermediates", "int/")):
            for path, values in flatten_dict(state.get(coll, {})).items():
                frame[tag + path[-1]] = np.asarray(values[0])
        for key, v in frame.items():
            if v.ndim and v.shape[0] == NA:      # per atom: real atoms only, concatenated over frames
                v, axis = v[:n], "atom"
            elif v.ndim and v.shape[0] == NS:    # per structure: the frame's row
                v, axis = v[:1], "frame"
            else:
                axis = "atom" if key in ("atom/energy", "forces") else "frame"
            rows.setdefault(key, ([], axis))[0].append(v)
        atom_ptr.append(atom_ptr[-1] + n)
        if k % 50 == 0:
            print(f"{k}/{len(frames)} n={n} {time.time() - t0:.1f}s", flush=True)
    out = {key: np.concatenate(v) if axis == "atom" or v[0].ndim else np.stack(v) for key, (v, axis) in rows.items()}
    meta = dict(ckpt=a.ckpt, frames=a.frames, x64=a.x64, matmul_precision=str(jax.config.jax_default_matmul_precision),
                device=str(jax.devices()[0]), jax=jax.__version__, host=platform.node(),
                lr_params={k: v for k, v in (calc._lr_params or {}).items() if k != "effective_cell"})
    np.savez(a.out, atom_ptr=np.asarray(atom_ptr), meta=json.dumps(meta, default=str), **out)
    print(json.dumps({k: list(v.shape) for k, v in out.items()}), f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
