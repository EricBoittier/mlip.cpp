#!/usr/bin/env python3
"""Oracle dump of the PET trunk (the `sr` subtree) of an iris PET-LR checkpoint (ipsum/PET-LR, after
M. Langer), with his own code: pet-jax's structure builder and in-forward adaptive selection
(`petjax.select.truncate_edges`), and `iris.pet.model.PET` on the checkpoint's `sr` parameters.
Runs in his env (kuma /scratch/boittier/ipsum/env), the reference for the mlip.cpp port.

    python ref_trunk.py CKPT_DIR FRAMES.extxyz OUT.npz [--x64] [--precision float32]

This is the calculator's inference path (`iris.pet.calculator`): the full list within `cutoff` (no
kNN trim, no Verlet skin), the adaptive grid selection in the forward (differentiable cutoffs),
conservative forces = -dE/dr through the selection, virial = sum_i r_i (x) dE/dr_i + cell^T dE/dcell.
Conditioning inputs as `PETLR` passes them: only the keys in the model's `conditioning` (absent charge
-> 0, absent multiplicity -> 1, pet-jax's defaults).

Per frame (concatenated over frames, offsets in `atom_ptr` / `pair_ptr`): per-atom SR energy
(energy_scale applied, no composition baseline), node features [n, d_node] (the trunk's `node`, which
the LR branch consumes), forces, virial (periodic frames), and the selected pair list the network saw
(center, neighbor, R_ij, pair_cutoff), for bisecting list/selection vs network maths.

--x64 casts parameters and inputs to float64 (jax_enable_x64; RMSNorm eps becomes finfo(float64).eps
there, as in pet-jax). --precision sets jax_default_matmul_precision (his training configs: float32).
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
    # floor-spread knobs: padding changes the reduction shapes (as training batches do), not the maths
    ap.add_argument("--kpad", type=int, default=8, help="k_sel rounded up to a multiple of this")
    ap.add_argument("--bucket", default="multiples", help="atom / pair padding strategy (marathon next_size)")
    a = ap.parse_args()

    import jax
    if a.x64:
        jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", a.precision)
    import jax.numpy as jnp
    from ase.io import read
    from marathon.io import from_dict, read_msgpack, read_yaml
    from petjax.select import select_edges, truncate_edges
    from petjax.structure import to_structure
    from petjax.utils import cast_floats, edge_displacements

    from iris.pet.model import PET

    lr = from_dict(read_yaml(f"{a.ckpt}/model/model.yaml"))
    fdt = np.float64 if a.x64 else np.float32
    trunk = PET(
        d_pet=lr.d_pet, d_node=lr.d_node, d_head=lr.d_head, d_feedforward=lr.d_feedforward,
        num_heads=lr.num_heads, num_attention_layers=lr.num_attention_layers,
        num_gnn_layers=lr.num_gnn_layers, cutoff=lr.cutoff, cutoff_width=lr.cutoff_width,
        num_neighbors_adaptive=lr.num_neighbors_adaptive,
        attention_temperature=lr.attention_temperature, max_atomic_number=lr.max_atomic_number,
        system_conditioning=lr.system_conditioning, max_charge=lr.max_charge,
        max_spin_multiplicity=lr.max_spin_multiplicity, conditioning=tuple(lr.conditioning),
        final_message_passing=lr.final_message_passing,
    )
    assert not lr.direct_forces and not lr.direct_stress
    params = {"params": read_msgpack(f"{a.ckpt}/model/model.msgpack")["params"]["sr"]}
    params = cast_floats(params, jnp.float64 if a.x64 else jnp.float32)
    cfg = dict(nna=lr.num_neighbors_adaptive, cutoff=lr.cutoff, cwa=lr.cutoff_width_adaptive,
               method=lr.adaptive_cutoff_method)

    def forward(positions, cell, s, k_sel, cond):
        R_ij = edge_displacements(positions, s["centers"], s["others"], s["cell_shifts"], cell)
        t, overflow = truncate_edges(
            R_ij, s["centers"], s["others"], s["reverse"], s["pair_mask"], s["atomic_numbers"],
            s["atom_mask"], k_sel, cfg["nna"], cfg["cutoff"], cfg["cwa"], method=cfg["method"])
        e, node = trunk.apply(params, **t, return_features=True, inputs=cond,
                              atom_to_structure=jnp.zeros(s["atomic_numbers"].shape[0], jnp.int32))
        return jnp.sum(e), (e, node, t, overflow)

    grad = jax.jit(jax.value_and_grad(forward, argnums=(0, 1), has_aux=True), static_argnums=(3,))

    frames = read(a.frames, ":")
    if a.max:
        frames = frames[: a.max]
    keys = ("energy", "node", "forces", "virial", "pair_c", "pair_n", "pair_R", "pair_cut")
    out = {k: [] for k in keys}
    atom_ptr, pair_ptr, overflow_any, t0 = [0], [0], False, time.time()
    for k, atoms in enumerate(frames):
        # bucketed (masked) padding so compilations are reused across frames
        s = to_structure(atoms, lr.cutoff, skin=0.0, bucket_strategy=a.bucket,
                         n_atoms_bucket_strategy=a.bucket, int_dtype=np.int32, float_dtype=fdt)
        sel = select_edges(s, cfg["nna"], cfg["cutoff"], cfg["cwa"], method=cfg["method"])
        k_sel = -(-max(int(sel.counts.max()), 1) // a.kpad) * a.kpad
        Np = s["positions"].shape[0]
        cond = {}
        for key in lr.conditioning:
            default = 0 if key == "charge" else 1
            cond[key] = jnp.asarray([int(atoms.info.get(key, default))], dtype=jnp.int32)
        js = {kk: jnp.asarray(v) for kk, v in s.items()}
        (_, (e, node, t, ovf)), (g_pos, g_cell) = grad(js["positions"], js["cell"], js, k_sel, cond)
        n = len(atoms)
        out["energy"].append(np.asarray(e)[:n])
        out["node"].append(np.asarray(node)[:n])
        out["forces"].append(-np.asarray(g_pos)[:n])
        vir = np.einsum("ia,ib->ab", np.asarray(js["positions"]), np.asarray(g_pos)) \
            + np.asarray(js["cell"]).T @ np.asarray(g_cell)
        out["virial"].append(vir if atoms.pbc.any() else np.full((3, 3), np.nan))
        m = np.asarray(t["pair_mask"])
        out["pair_c"].append(np.asarray(t["centers"])[m])
        out["pair_n"].append(np.asarray(t["neighbors"])[m])
        out["pair_R"].append(np.asarray(t["R_ij"])[m])
        out["pair_cut"].append(np.asarray(t["pair_cutoffs"])[m])
        overflow_any |= bool(ovf)
        atom_ptr.append(atom_ptr[-1] + n)
        pair_ptr.append(pair_ptr[-1] + int(m.sum()))
        if k % 50 == 0:
            print(f"{k}/{len(frames)} n={n} k_sel={k_sel} {time.time() - t0:.1f}s", flush=True)
    res = {k: np.concatenate(v) if k != "virial" else np.stack(v) for k, v in out.items()}
    meta = dict(ckpt=a.ckpt, frames=a.frames, x64=a.x64,
                matmul_precision=str(jax.config.jax_default_matmul_precision),
                device=str(jax.devices()[0]), jax=jax.__version__, host=platform.node(),
                overflow=overflow_any, kpad=a.kpad, bucket=a.bucket,
                xla_flags=__import__('os').environ.get('XLA_FLAGS', ''), conditioning=list(lr.conditioning), **cfg)
    np.savez(a.out, atom_ptr=np.asarray(atom_ptr), pair_ptr=np.asarray(pair_ptr),
             meta=json.dumps(meta), **res)
    print(json.dumps(meta), f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
