#!/usr/bin/env python3
"""A seeded-init iris PETLR checkpoint folder (model/model.yaml, model/model.msgpack, model/baseline.yaml) for trunk
options no omol-4m checkpoint has, run in his env: e.g. his biotrimers / multivalent trunk (final_message_passing
False, 1 GNN layer, hard cutoff, spin-only conditioning). The trunk presets keep the LR branch off (`lr: False`); the lr_* presets mount it.

    python make_seeded.py OUT_DIR --preset multivalent|biotrimers|lr_species|lr_both|lr_fermi_cold [--seed 0] [--scale 0.5]

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

# LR-branch presets on the multivalent trunk (adaptive off, so the oracle and the torch check share the plain list),
# covering the IrisLR rows no omol-4m checkpoint exercises; zero-initialised leaves (incl. scalars: transfer gates,
# lr_scale) are randomised, so every LR term carries force.
_LR = dict(PRESETS["multivalent"], max_atomic_number=83, lr=True, physical_charges=True, charge_allocation="flows",
           num_charges=1, inputs=["charge", "spin_multiplicity", "dipole_origin"])
PRESETS.update({
    # species reference, softplus conductance, 3 unshared polarisation passes with hardness, charge readout
    "lr_species": dict(_LR, charge_reference="species", conductance_form="softplus", polarisation_passes=3,
                       polarisation_shared=False, hardness=True, readout_charges=True, transfer_passes=2),
    # both references, layerscale, 2 shared passes, Gaussian charges (erf / erfc), no hardness
    "lr_both": dict(_LR, charge_reference="both", conductance_form="layerscale", polarisation_passes=2,
                    polarisation_shared=True, hardness=False, smearing=1.0, readout_charges=True),
    # fermi at T = 0.02 (gapped midpoint reachable), single pass, no readout
    "lr_fermi_cold": dict(_LR, charge_reference="fermi", charge_temperature=0.02, polarisation_passes=1),
})

# tiny models for the committed test GGUFs (tests/resources/petlr_tiny_*.gguf): a small trunk with the adaptive cutoff
# (cutoff Jacobian) and spin-only conditioning, and one LR model per reference / pass type, so the C++ FD tests run by
# default on every IrisLR row: fermi (T = 0.1), gapped fermi (T = 0.02), species (softplus, 3 unshared passes, hardness,
# 2 transfer passes), both (Gaussian charges, 2 shared passes)
_TINY = dict(d_pet=16, d_node=32, d_head=16, d_feedforward=32, num_heads=2, num_attention_layers=1, num_gnn_layers=2,
             cutoff=5.0, cutoff_width=0.5, final_message_passing=True, num_neighbors_adaptive=8,
             max_atomic_number=18, system_conditioning=True, conditioning=["spin_multiplicity"], max_charge=10,
             max_spin_multiplicity=11)
_TLR = dict(_TINY, lr=True, physical_charges=True, charge_allocation="flows", num_charges=1,
            inputs=["charge", "spin_multiplicity", "dipole_origin"])
PRESETS.update({
    "tiny_trunk": dict(_TINY),
    "tiny_fermi": dict(_TLR, charge_reference="fermi", polarisation_passes=1, readout_charges=True, transfer_passes=1),
    "tiny_cold": dict(_TLR, charge_reference="fermi", charge_temperature=0.02, polarisation_passes=1),
    # damped (2026-11-11, 4th session): conductance_init -6 and the conductance heads' last layer x 0.2 keep the
    # charges O(1) (undamped: 3 passes amplified them to ~100 e and max|F| 1e4-1e5 eV/A, which made the FD tolerance
    # of the C++ test weak) and put softplus in its log1p regime (z ~ -6). NOT a reproducer of the CUDA __logf defect
    # (fast-math tiny_species is inside his floor too, E 0.30x): the seeded lr_species preset is (5.1x), and
    # test_pet_lr_interp checks the flagged log op on the GPU directly
    "tiny_species": dict(_TLR, charge_reference="species", conductance_form="softplus", polarisation_passes=3,
                         polarisation_shared=False, hardness=True, readout_charges=True, transfer_passes=2,
                         conductance_init=-6.0, _damp={"pol_conductance": 0.2}),
    "tiny_both": dict(_TLR, charge_reference="both", conductance_form="layerscale", polarisation_passes=2,
                      polarisation_shared=True, hardness=False, smearing=1.0, readout_charges=True),
})

# his branch feat/long-range-transfer (3ed96b0; env kuma /scratch/boittier/ipsum/envs/iris-long-range-transfer): no
# readout_charges there; species_fill / species_weighted references and the long-range transfer passes (Gaussian
# Coulomb kernel of width `smearing`)
_LRT = {k: v for k, v in _TLR.items()}
PRESETS.update({
    "lrt_species_fill": dict(_LRT, charge_reference="species_fill", long_range_transfer=True, smearing=1.0,
                             transfer_passes=1, polarisation_passes=1),
    "lrt_species_weighted": dict(_LRT, charge_reference="species_weighted", long_range_transfer=True, smearing=1.0,
                                 transfer_passes=2, polarisation_passes=2, polarisation_shared=False,
                                 conductance_form="softplus", hardness=True),
    # one feature at a time (2026-11-11, sixth session): the two new references WITHOUT the long-range transfer
    "lrt_weighted": dict(_LRT, charge_reference="species_weighted", transfer_passes=1, polarisation_passes=2,
                         polarisation_shared=True, conductance_form="layerscale", hardness=False),
    "lrt_fill": dict(_LRT, charge_reference="species_fill", transfer_passes=1, polarisation_passes=1),
    "lrt_fill_cold": dict(_LRT, charge_reference="species_fill", charge_temperature=0.02, transfer_passes=1,
                          polarisation_passes=1),
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", help="a training config.yaml to copy (Calculator.from_checkpoint reads its sampler)")
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from marathon.io import write_msgpack, write_yaml

    from iris.pet.petlr import PETLR
    cfg = {"lr": False, **PRESETS[a.preset]}
    damp = cfg.pop("_damp", {})  # {path substring: factor} on the last Dense of matching heads, after the randomisation
    lr_on = bool(cfg["lr"])
    model = PETLR(**cfg)
    key = jax.random.PRNGKey(a.seed)
    params = model.init(key, *model.dummy_inputs())
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(a.seed + 1), len(leaves))
    out = []
    for x, k in zip(leaves, keys):
        k1, k2 = jax.random.split(k)
        if jnp.all(x == 0) and (x.ndim or lr_on):
            x = 0.1 * jax.random.normal(k1, x.shape, x.dtype)
        out.append(x * (1.0 + a.scale * jax.random.uniform(k2, (), x.dtype, -1.0, 1.0)))
    params = jax.tree.unflatten(tree, out)
    if damp:
        from flax.traverse_util import flatten_dict, unflatten_dict
        flat = flatten_dict(params)
        last = {}
        for path in flat:  # the last Dense_k of each matching module
            name = "/".join(map(str, path))
            for sub, f in damp.items():
                if sub in name and any(str(p).startswith("Dense_") for p in path):
                    mod = path[:[str(p).startswith("Dense_") for p in path].index(True)]
                    k = max(int(str(p)[6:]) for q in flat for p in q[len(mod):len(mod) + 1]
                            if q[:len(mod)] == mod and str(p).startswith("Dense_"))
                    last[mod + (f"Dense_{k}",)] = f
        for path in list(flat):
            for mod, f in last.items():
                if path[:len(mod)] == mod:
                    flat[path] = flat[path] * f
        print("damped:", sorted("/".join(map(str, m)) + f" x{f}" for m, f in last.items()))
        params = unflatten_dict(flat)
    d = Path(a.out) / "model"
    d.mkdir(parents=True, exist_ok=True)
    write_msgpack(d / "model.msgpack", jax.tree.map(lambda v: jnp.asarray(v), params))
    write_yaml(d / "model.yaml", {"iris.pet.petlr.PETLR": cfg})
    write_yaml(d / "baseline.yaml", {})
    if a.config:
        import shutil
        shutil.copy(a.config, Path(a.out) / "config.yaml")
    print(a.preset, cfg, sum(x.size for x in out), "parameters ->", d)


if __name__ == "__main__":
    main()
