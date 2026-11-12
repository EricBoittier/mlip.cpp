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
    # QEq (iris.pet.charges.qeq_reference, a2658e3: cg + custom_linear_solve), Gaussian charges, 2 shared passes
    "tiny_qeq": dict(_TLR, charge_reference="qeq", smearing=1.0, polarisation_passes=2, polarisation_shared=True,
                     hardness=False, transfer_passes=1),
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

# his branch feat/threshold-reference (101575d, two commits on 3ed96b0; run in the long-range-transfer env with
# PYTHONPATH=<a worktree at 101575d>, kuma /scratch/boittier/ipsum/src/iris-threshold): charge_reference "threshold",
# q0_i = c_Z + f(s_i - theta_Z) - w_i R / sum w, R = sum_i (c_Z + f_i) - Q (emitted as charge_residual), no Fermi level.
# The init randomises theta_Z and c_Z (zero-initialised leaves) and jitters the softplus weights per species (else they
# stay equal and cancel in the spread), so every term carries
PRESETS.update({
    "thr": dict(_LRT, charge_reference="threshold", transfer_passes=1, polarisation_passes=2,
                polarisation_shared=True, conductance_form="layerscale", hardness=True, _jitter={"species_weight": 0.8}),
    # gapped placement (T 0.02) with the long-range transfer, Gaussian charges, one pass
    "thr_cold_lrt": dict(_LRT, charge_reference="threshold", charge_temperature=0.02, long_range_transfer=True,
                         smearing=1.0, transfer_passes=1, polarisation_passes=1, _jitter={"species_weight": 0.8}),
})

# his branch feat/conserving-charges (a98c2f6; env kuma /scratch/boittier/ipsum/envs/iris-conserving-charges; the
# water-screening branch bb5a5ea carries the same model code): conserving_charges (per-component Fermi level, or the
# residual allocation, plus one edge transfer, then the polarisation passes). That PETLR has no `conditioning` list
# (system_conditioning only) and needs fixed neighbour lists (fixed_neighbors: his KnnToSample, hard cutoff); the tiny
# trunk otherwise
_CC = dict({k: v for k, v in _TINY.items() if k != "conditioning"}, system_conditioning=False, fixed_neighbors=63,
           lr=True, physical_charges=True, num_charges=1, conserving_charges=True, inputs=["charge"])
PRESETS.update({
    # per-component Fermi level (T 0.1), 2 shared layerscale passes with hardness
    "cc_fermi": dict(_CC, charge_allocation="fermi", polarisation_passes=2, hardness=True),
    # residual allocation, both information-flow switches on (Q_c basis into the weights, q0 into the transfer),
    # 3 unshared softplus passes, no hardness
    "cc_residual": dict(_CC, charge_allocation="residual", weights_see_charge=True, transfers_see_charges=True,
                        charge_head_init=0.3, polarisation_passes=3, polarisation_shared=False,
                        conductance_form="softplus", hardness=False),
    # residual allocation, plain heads, one pass
    "cc_residual_plain": dict(_CC, charge_allocation="residual", polarisation_passes=1),
    # with his `field` input (a uniform external field per structure: the pass drives' -E . R / 2 and the enthalpy
    # -E . P_flow; dipole_flow): fermi with 2 shared passes, and residual with the transfer alone (one pass)
    "cc_fermi_field": dict(_CC, charge_allocation="fermi", polarisation_passes=2, hardness=True,
                           inputs=["charge", "field"]),
    "cc_residual_field": dict(_CC, charge_allocation="residual", polarisation_passes=1, inputs=["charge", "field"]),
    # induced dipoles (his a98c2f6 induced_dipoles; jax-pme EwaldDipole of lab-cosmo/jax-pme#38 rebased onto 1c26495 by
    # us): the gate (init -6, softplus 2.5e-3: dipoles all but off) set to 1.0 (softplus 1.31) so alpha_i is O(1) A^3
    # and the induced terms carry weight. "field" (one-shot, variational) on the fermi preset with 2 passes (the field
    # of the last pass's charges); "full" (charge + dipole Coulomb with the dipoles as sources) on residual, one pass,
    # Gaussian charges (smearing 1.0: the smeared-field and dipole-potential corrections on the SR edges)
    "cc_ind_field": dict(_CC, charge_allocation="fermi", polarisation_passes=2, hardness=True, inputs=["charge", "field"],
                         induced_dipoles="field", _set={"polarisability_gate": 1.0}),
    "cc_ind_full": dict(_CC, charge_allocation="residual", polarisation_passes=1, inputs=["charge", "field"],
                        induced_dipoles="full", smearing=1.0, _set={"polarisability_gate": 1.0}),
})

# his les-water backbones (ipsum-work play/les-water/backbones.py; iris branch explore/alessandro-strict-local-20261012-v1
# 1d412cf carries the same trunk() and NoFinalCombineBackbone, and the PET per_layer_readout / learn_layer_gates /
# feature_layer (charge_layer) of iris 38c91c4): NoFinalCombineBackbone (no combine after the last GNN layer) and
# LayeredBackbone (the standard trunk returning per-layer features: entry l before layer l's combine, the last entry
# after the final combine, L + 1 energy heads); both build the comb_norms with flax's default LayerNorm eps 1e-6
# (pet-jax's Backbone: 1e-5). Trunk-only (lr: False; that branch's LR side predates the ported flows allocation).
# Run in the main env with PYTHONPATH=<that worktree>:<a dir holding his backbones.py> (kuma ref_les.sbatch). No
# conditioning (that PET has no `conditioning` list)
_LES = dict({k: v for k, v in _TINY.items() if k not in ("conditioning", "final_message_passing")},
            system_conditioning=False)
PRESETS.update({
    # his configs.py local_backbone at two and three layers (1 resp. 2 intermediate comb_norms at eps 1e-6)
    "les_local2": dict(_LES, num_gnn_layers=2, backbone="backbones.NoFinalCombineBackbone"),
    "les_local3": dict(_LES, num_gnn_layers=3, backbone="backbones.NoFinalCombineBackbone"),
    # his ladder (layered, two layers, per_layer_readout; charges from layer 1's features) with the layer gates on
    "les_ladder": dict(_LES, num_gnn_layers=2, backbone="backbones.LayeredBackbone", per_layer_readout=True,
                       learn_layer_gates=True, charge_layer=1),
    # the ladder's charge_layer 0 variant, no gates
    "les_ladder_c0": dict(_LES, num_gnn_layers=2, backbone="backbones.LayeredBackbone", per_layer_readout=True,
                          charge_layer=0),
    # WITH that branch's LR side (the pre-flows physical charge head; his configs.py PETLR_PHYSICAL: one physical
    # charge, constant charge_scale 0.1, lr_scale_init 1): the charge_layer oracle. As his ladder runs (charges from
    # layer 1, gated heads, one pass), and charge_layer 0 with a learned scale and 2 unshared passes with hardness
    "les_ladder_lr": dict(_LES, num_gnn_layers=2, backbone="backbones.LayeredBackbone", per_layer_readout=True,
                          learn_layer_gates=True, charge_layer=1, lr=True, physical_charges=True, num_charges=1,
                          lr_scale_init=1.0, learn_charge_scale=False, charge_scale_init=0.1),
    "les_ladder_lr_c0": dict(_LES, num_gnn_layers=2, backbone="backbones.LayeredBackbone", per_layer_readout=True,
                             charge_layer=0, lr=True, physical_charges=True, num_charges=1, lr_scale_init=1.0,
                             polarisation_passes=3, polarisation_shared=False, hardness=True),
})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "presets" / "seeded_config.yaml"),
                    help="a training config.yaml to copy (Calculator.from_checkpoint reads its sampler; default the "
                         "seeded presets' presets/seeded_config.yaml, num_k 1024, max_neighbors 59; '' for none)")
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    from marathon.io import write_msgpack, write_yaml

    from iris.pet.petlr import PETLR
    cfg = {"lr": False, **PRESETS[a.preset]}
    damp = cfg.pop("_damp", {})  # {path substring: factor} on the last Dense of matching heads, after the randomisation
    # {leaf name: s}: an ELEMENTWISE factor in [1 - s, 1 + s] on those leaves (a constant-initialised per-species leaf,
    # e.g. species_weight, otherwise stays uniform under the per-leaf factor and its species dependence is untested)
    jitter = cfg.pop("_jitter", {})
    setv = cfg.pop("_set", {})  # {leaf name: value}: set those scalar leaves after the randomisation
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
    if setv:
        from flax.traverse_util import flatten_dict, unflatten_dict
        flat = flatten_dict(params)
        for path in flat:
            if str(path[-1]) in setv:
                flat[path] = jnp.full_like(flat[path], setv[str(path[-1])])
                print("set:", "/".join(map(str, path)), setv[str(path[-1])])
        params = unflatten_dict(flat)
    if jitter:
        from flax.traverse_util import flatten_dict, unflatten_dict
        flat = flatten_dict(params)
        jk = jax.random.PRNGKey(a.seed + 2)
        for path in sorted(flat, key=str):
            if str(path[-1]) in jitter:
                jk, k1 = jax.random.split(jk)
                x = flat[path]
                flat[path] = x * (1.0 + jitter[str(path[-1])] * jax.random.uniform(k1, x.shape, x.dtype, -1.0, 1.0))
                print("jittered:", "/".join(map(str, path)))
        params = unflatten_dict(flat)
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
    if a.config and Path(a.config).exists():
        import shutil
        shutil.copy(a.config, Path(a.out) / "config.yaml")
    print(a.preset, cfg, sum(x.size for x in out), "parameters ->", d)


if __name__ == "__main__":
    main()
