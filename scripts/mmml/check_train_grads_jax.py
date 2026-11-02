#!/usr/bin/env python3
"""mlipcpp-train's loss and weight gradients vs mmml (JAX) autograd on the same batch.

    mlipcpp-train-check --data DATA --init model.gguf --out GC --n 8 --gpus 0 \\
        --conservative 1 --energy-weight 1 --force-weight 52.91 --nc-force-weight 0 \\
        --model-baselines 1 --energy-per-structure 1
    PYTHONPATH=$MMML python scripts/mmml/check_train_grads_jax.py --data DATA --gguf model.gguf --check GC

The JAX loss is the one of scripts/train_so3lr_spooky_extxyz.py: w_E mean_s (E_s - E_s^ref)^2
+ w_F mean over force components (F - F^ref)^2 (--per-atom: the energy term on
(E_s - E_s^ref) / N_s, mlipcpp-train's default and mmml's --energy-loss-per-atom), with
forces -dE/dx, all pairs within each molecule, float64. The JAX gradient tree is mapped
onto the GGUF tensors with convert_mmml.params_to_tensors (a linear relayout). Prints the
relative difference per tensor and in total; exit 1 above --tol.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mmml"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir holding train/ (as given to mlipcpp-train-check)")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--check", required=True, help="mlipcpp-train-check --out dir (check.json, grads.bin)")
    ap.add_argument("--per-atom", action="store_true", help="energy term per atom (no --energy-per-structure)")
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import convert_mmml as cm
    from mmml_ref import JaxRef, hypers

    meta = json.load(open(Path(a.check) / "check.json"))
    ids = meta["ids"]
    cons = meta.get("conservative", {})
    w_e, w_f = cons.get("energy_weight", 1.0), cons.get("force_weight", 1.0)
    grads = np.fromfile(Path(a.check) / "grads.bin", np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, "m.json")
        cm.to_json(a.gguf, js)
        ref = JaxRef(js, x64=True)
    kv, _ = cm.read_gguf(a.gguf)
    max_q = int(kv.get("physnet.max_charge", 10)) if isinstance(kv.get("physnet.max_charge", 10), (int, float)) else 10

    sp = Path(a.data) / "train"
    na = np.load(sp / "na.npy")
    X = np.fromfile(sp / "x.bin", np.float32).reshape(-1, 3)
    Zs = np.fromfile(sp / "a.bin", np.int32)
    Fs = np.fromfile(sp / "non_conservative_force.bin", np.float32).reshape(-1, 3)
    Es = np.fromfile(sp / "energy.bin", np.float32)
    Qs = np.fromfile(sp / "charge.bin", np.float32) if (sp / "charge.bin").exists() else np.zeros(len(Es))
    Ss = np.fromfile(sp / "spin_multiplicity.bin", np.float32) if (sp / "spin_multiplicity.bin").exists() \
        else np.ones(len(Es))
    mols = [dict(Z=Zs[na[i]:na[i + 1]], R=X[na[i]:na[i + 1]].astype(np.float64), Q=float(Qs[i]), S=float(Ss[i]))
            for i in ids]
    Eref = jnp.asarray(Es[ids].astype(np.float64))
    Fref = jnp.asarray(np.concatenate([Fs[na[i]:na[i + 1]] for i in ids]).astype(np.float64))
    N = jnp.asarray([len(m["Z"]) for m in mols], jnp.float64)
    Z, R, seg, dst, src, Q, S = ref._batch(mols)
    f = ref.fn(len(Z), len(dst), len(mols), True)
    args = (jnp.asarray(Z), jnp.asarray(R), jnp.asarray(seg), jnp.asarray(dst), jnp.asarray(src),
            jnp.asarray(Q, jnp.float64), jnp.asarray(S, jnp.float64))

    def loss(params):
        E, F = f(params, *args)
        dE = (E - Eref) / (N if a.per_atom else 1.0)
        le = w_e * jnp.mean(dE ** 2)
        lf = w_f * jnp.mean((F.reshape(-1, 3) - Fref) ** 2)
        return le + lf, (le, lf)

    (L, (le, lf)), g = jax.value_and_grad(loss, has_aux=True)(ref.params)
    g = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), g)["params"]
    tg = cm.params_to_tensors(ref.config, g, ref.model_type, max_q, 10)
    print(f"loss: mlipcpp {meta['loss']:.8g} (E {meta['loss_energy']:.6g}, F {meta['loss_force']:.6g}) | "
          f"JAX float64 {float(L):.8g} (E {float(le):.6g}, F {float(lf):.6g})")
    num = den = 0.0
    worst = (0.0, "")
    for p in meta["params"]:
        name = p["name"]
        mine = grads[p["offset"]:p["offset"] + p["size"]].astype(np.float64)
        if name not in tg:
            print(f"  {name}: no JAX counterpart")
            continue
        theirs = np.asarray(tg[name], np.float64)
        if theirs.size != mine.size:
            print(f"  {name}: size {mine.size} vs {theirs.size}")
            continue
        theirs = theirs.reshape(-1)
        # the GGUF stores numpy-layout arrays; ggml's flat order is the numpy C order
        d = np.linalg.norm(mine - theirs)
        n = np.linalg.norm(theirs)
        num += d * d
        den += n * n
        rel = d / max(n, 1e-30)
        if rel > worst[0] and n > 1e-12 * np.sqrt(den + 1e-300):
            worst = (rel, name)
        print(f"  {name:22s} |g| {n:10.4g}  rel {rel:.2e}")
    total = np.sqrt(num / max(den, 1e-300))
    lrel = abs(meta["loss"] - float(L)) / abs(float(L))
    print(f"total gradient rel {total:.2e} (worst tensor {worst[1]} {worst[0]:.2e}); loss rel {lrel:.2e}")
    sys.exit(0 if total < a.tol and lrel < a.tol else 1)


if __name__ == "__main__":
    main()
