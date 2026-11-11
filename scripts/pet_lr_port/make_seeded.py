#!/usr/bin/env python3
"""A seeded-init iris PETLR checkpoint folder (model/model.yaml, model/model.msgpack, model/baseline.yaml) for trunk
options no omol-4m checkpoint has, run in his env: e.g. his biotrimers / multivalent trunk (final_message_passing
False, 1 GNN layer, hard cutoff, spin-only conditioning). The LR branch is off (`lr: False`): trunk tests only.

    python make_seeded.py OUT_DIR --preset multivalent|biotrimers [--seed 0] [--scale 0.5]

--scale multiplies every parameter after init by a seeded factor in [1 - s, 1 + s] per leaf and randomises the
zero-initialised leaves, so no branch is trivially zero.
"""
import argparse
from pathlib import Path

PRESETS = {
    # multivalent/configs.py trunk
    "multivalent": dict(d_pet=128, d_node=512, d_head=128, d_feedforward=256, num_heads=8, num_attention_layers=1,
                        num_gnn_layers=1, cutoff=5.0, cutoff_width=0.5, final_message_passing=False,
                        num_neighbors_adaptive=None, max_atomic_number=27, system_conditioning=True,
                        conditioning=["spin_multiplicity"], max_charge=10, max_spin_multiplicity=11),
    # biotrimers-physical/configs.py trunk
    "biotrimers": dict(d_pet=128, d_node=512, d_head=128, d_feedforward=256, num_heads=8, num_attention_layers=1,
                       num_gnn_layers=1, cutoff=4.0, cutoff_width=0.5, final_message_passing=False,
                       num_neighbors_adaptive=None, max_atomic_number=18),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from marathon.io import write_msgpack, write_yaml

    from iris.pet.petlr import PETLR
    cfg = {**PRESETS[a.preset], "lr": False}
    model = PETLR(**cfg)
    key = jax.random.PRNGKey(a.seed)
    params = model.init(key, *model.dummy_inputs())
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(a.seed + 1), len(leaves))
    out = []
    for x, k in zip(leaves, keys):
        k1, k2 = jax.random.split(k)
        if jnp.all(x == 0) and x.ndim:
            x = 0.1 * jax.random.normal(k1, x.shape, x.dtype)
        out.append(x * (1.0 + a.scale * jax.random.uniform(k2, (), x.dtype, -1.0, 1.0)))
    params = jax.tree.unflatten(tree, out)
    d = Path(a.out) / "model"
    d.mkdir(parents=True, exist_ok=True)
    write_msgpack(d / "model.msgpack", jax.tree.map(lambda v: jnp.asarray(v), params))
    write_yaml(d / "model.yaml", {"iris.pet.petlr.PETLR": cfg})
    write_yaml(d / "baseline.yaml", {})
    print(a.preset, cfg, sum(x.size for x in out), "parameters ->", d)


if __name__ == "__main__":
    main()
