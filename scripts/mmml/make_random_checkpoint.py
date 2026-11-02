"""Write a randomly initialised mmml SpookyPhysNet / PhysNet as a portable JSON.

Used by the parity tests while no trained Spooky checkpoint exists. Flax's
initialisation zeroes some kernels (the energy head); every parameter is then
perturbed (zero-initialised leaves: N(0, noise^2); others: 10% relative noise) so that
every path of the model is exercised with realistic magnitudes.

    PYTHONPATH=$MMML python scripts/mmml/make_random_checkpoint.py --model-type spooky \
        --features 64 --max-degree 1 --num-iterations 3 --num-basis-functions 32 --n-res 2 \
        --max-atomic-number 87 -o spooky_random.json
"""

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from mmml_ref import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["physnet", "spooky"], default="spooky")
    ap.add_argument("--features", type=int, default=64)
    ap.add_argument("--max-degree", type=int, default=1)
    ap.add_argument("--num-iterations", type=int, default=3)
    ap.add_argument("--num-basis-functions", type=int, default=32)
    ap.add_argument("--n-res", type=int, default=2)
    ap.add_argument("--max-atomic-number", type=int, default=87)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--no-zbl", action="store_true")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--filter-scale", type=float, default=0.3,
                    help="multiplies the MessagePass filter kernels: flax's init sums ~N_neighbours x "
                         "num_basis O(1) terms per message, which makes untrained features (and float32 "
                         "round-off) grow by orders of magnitude over the iterations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    config = dict(
        features=a.features, max_degree=a.max_degree, num_iterations=a.num_iterations,
        num_basis_functions=a.num_basis_functions, cutoff=a.cutoff, max_atomic_number=a.max_atomic_number,
        charges=False, n_res=a.n_res, n_refinement_blocks=a.n_res, zbl=not a.no_zbl, trainable_zbl=False,
        zbl_cuton=0.1, zbl_cutoff=0.6, efa=False, use_energy_bias=True, use_pbc=False,
        include_electrostatics=False, max_padded_atoms=8,
    )
    if a.model_type == "spooky":
        config["model_type"] = "spooky"
    model = build_model(config, a.model_type, natoms=8)
    Z = jnp.array([8, 1, 1, 6, 1, 1, 1, 7], jnp.int32)
    R = jax.random.normal(jax.random.PRNGKey(1), (8, 3)) * 1.5
    i, j = np.nonzero(~np.eye(8, dtype=bool))
    kw = dict(batch_segments=jnp.zeros(8, jnp.int32), batch_size=1, batch_mask=jnp.ones(len(i)),
              atom_mask=jnp.ones(8))
    key = jax.random.PRNGKey(a.seed)
    if a.model_type == "spooky":
        params = model.init(key, Z, jnp.zeros(8), jnp.ones(8), R, jnp.asarray(i), jnp.asarray(j), **kw)
    else:
        params = model.init(key, Z, R, jnp.asarray(i), jnp.asarray(j), **kw)
    leaves, tree = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(a.seed + 7), len(leaves))
    # zero-initialised leaves (energy head, biases, energy_bias) get N(0, noise^2), the
    # others a 10% relative perturbation: realistic magnitudes, every path exercised
    def perturb(x, k):
        x = np.asarray(x)
        eps = np.asarray(jax.random.normal(k, x.shape))
        if not np.any(x):
            return a.noise * eps
        return x * (1 + 0.1 * eps)
    leaves = [perturb(x, k) for x, k in zip(leaves, keys)]
    params = jax.tree_util.tree_unflatten(tree, leaves)
    for name, sub in params["params"].items():
        if name.startswith("MessagePass_"):
            for deg in sub["filter"]:
                sub["filter"][deg]["kernel"] = sub["filter"][deg]["kernel"] * a.filter_scale
    tolist = lambda x: {k: tolist(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x).tolist()
    with open(a.output, "w") as f:
        json.dump({"params": tolist(params["params"]), "config": config}, f)
    print(a.output, sum(np.asarray(x).size for x in leaves), "parameters")


if __name__ == "__main__":
    main()
