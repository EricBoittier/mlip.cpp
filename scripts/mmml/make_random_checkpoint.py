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
    ap.add_argument("--pristine", action="store_true",
                    help="flax's initialisation as is (no perturbation, --filter-scale 1): the parameters "
                         "train_so3lr_spooky_extxyz.py starts from with --seed, for training")
    ap.add_argument("--multitask", default="", help="comma list of charges,spins,orbitals: mmml multitask heads "
                    "(proposals/SPOOKY_MULTITASK.md); initialised by mmml if it has them, else here with the same "
                    "scheme (kernels N(0, 0.01^2) / lecun normal, biases and the orbital output 0)")
    ap.add_argument("--orbital-hidden", type=int, default=None, help="H of the orbital MLP (default: features)")
    ap.add_argument("--homo-offset", type=float, default=0.0, help="eV; the training set's mean HOMO energy")
    ap.add_argument("--gap-offset", type=float, default=0.0, help="eV; the training set's mean HOMO-LUMO gap")
    ap.add_argument("--offsets-from", default=None, help="mlipcpp-train split dir: --homo-offset / --gap-offset = the "
                    "means of its labelled homo_energy.bin / homo_lumo_gap.bin (homo_mask.bin, gap_mask.bin)")
    ap.add_argument("--feature-rmsnorm", action="store_true",
                    help="SpookyPhysNet feature_rmsnorm (RMS-normalised features after each refinement stage)")
    ap.add_argument("--charge-spin-per-atom", action="store_true",
                    help="SpookyPhysNet charge_spin_per_atom (charge / spin inputs Q / N and (M - 1) / N per atom)")
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
        if a.feature_rmsnorm:
            config["feature_rmsnorm"] = True
        if a.charge_spin_per_atom:
            config["charge_spin_per_atom"] = True
    if a.offsets_from:
        from pathlib import Path
        sp = Path(a.offsets_from)
        for key, val, msk in (("homo_offset", "homo_energy", "homo_mask"), ("gap_offset", "homo_lumo_gap", "gap_mask")):
            v, m = np.fromfile(sp / f"{val}.bin", np.float32), np.fromfile(sp / f"{msk}.bin", np.float32) > 0.5
            setattr(a, key, float(v[m].astype(np.float64).mean()))
        print(f"offsets from {sp}: HOMO {a.homo_offset:.4f} eV, gap {a.gap_offset:.4f} eV")
    mt = {x for x in a.multitask.split(",") if x}
    if mt - {"charges", "spins", "orbitals"}:
        raise SystemExit(f"--multitask: unknown heads {mt - {'charges', 'spins', 'orbitals'}}")
    for h in mt:
        config[f"multitask_{h}"] = True
    if "orbitals" in mt:
        config.update(multitask_orbital_hidden=a.orbital_hidden or a.features, multitask_homo_offset=a.homo_offset,
                      multitask_gap_offset=a.gap_offset)
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
    if not a.pristine:
        leaves = [perturb(x, k) for x, k in zip(leaves, keys)]
    params = jax.tree_util.tree_unflatten(tree, leaves)
    for name, sub in params["params"].items() if not a.pristine else ():
        if name.startswith("MessagePass_"):
            for deg in sub["filter"]:
                sub["filter"][deg]["kernel"] = sub["filter"][deg]["kernel"] * a.filter_scale
    # heads mmml does not know (the multitask patch not applied): the same initialisation here
    rng = np.random.default_rng(a.seed + 11)
    P, F, Zn = params["params"], a.features, a.max_atomic_number + 1
    for key, name in (("charge", "charges"), ("spin", "spins")):
        if name in mt and f"mt_{key}_dense" not in P:
            P[f"mt_{key}_dense"] = {"kernel": rng.normal(0.0, 0.01, (F, 1))}
            P[f"mt_{key}_zbias"] = np.zeros(Zn)
    if "orbitals" in mt and "mt_orb_hidden" not in P:
        H = config["multitask_orbital_hidden"]
        P["mt_orb_hidden"] = {"kernel": rng.normal(0.0, 1.0 / np.sqrt(F), (F, H)), "bias": np.zeros(H)}
        P["mt_orb_out"] = {"kernel": np.zeros((H, 2)), "bias": np.zeros(2)}
    tolist = lambda x: {k: tolist(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x).tolist()
    with open(a.output, "w") as f:
        json.dump({"params": tolist(params["params"]), "config": config}, f)
    print(a.output, sum(np.asarray(x).size for x in leaves), "parameters")


if __name__ == "__main__":
    main()
