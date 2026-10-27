"""A freshly initialised plato_pet_lr checkpoint with the hypers, dataset info,
composition and energy scale of an existing one (fitted on the same training set):
the from-scratch start of a mlipcpp-train-plato run. Optionally with the
electronegativity term (metatrain branch feat/plato-chi-frozen: checkpoint v4).

    python scripts/plato/make_init.py trained.ckpt init.ckpt --seed 0 \
        [--electronegativity linear --chi0-frozen --dchi-bound 1.0] [--keep-weights] [--random-dchi 0.3]
        [--frames identity]
    python scripts/plato/export_plato_gguf.py init.ckpt -o init.gguf

--keep-weights carries every weight of the source over (a parity-test model);
--frames replaces the frame group (identity: the same model without frames; frame_mode stays
attention, which with one frame is plain RoPE attention).
--random-dchi s draws the last layer of the dchi head with std s (it starts at zero).
--mesh-normalization: physical (default for fresh weights) or per_point; with --keep-weights the
source's (a v5 or older source is per_point). Needs metatrain's checkpoint v6 (the hyper).
charge_unit = sqrt(energy scale), as metatrain's trainer sets it when a run starts.
"""
import argparse
import math

import metatomic.torch  # noqa: F401
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--electronegativity", default=None)
    ap.add_argument("--chi0-frozen", action="store_true")
    ap.add_argument("--dchi-bound", type=float, default=0.0)
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--random-dchi", type=float, default=0.0)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--mesh-normalization", choices=["physical", "per_point"], default=None)
    a = ap.parse_args()
    from metatrain.experimental.plato_pet_lr.model import PlatoPETLR
    from metatrain.experimental.plato_pet_lr.trainer import Trainer
    from metatrain.utils.io import load_model

    ck = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    src_model = load_model(a.checkpoint)
    scale = float(src_model.scaler.model.scales["energy"].block(0).values.reshape(-1)[0])
    hypers = dict(src_model.hypers)
    if a.electronegativity:
        hypers.update(electronegativity=a.electronegativity, chi0_trainable=not a.chi0_frozen, dchi_bound=a.dchi_bound)
    if a.frames:
        hypers.update(frames=a.frames)
    if "mesh_normalization" in hypers:  # metatrain with the hyper (checkpoint v6)
        if a.mesh_normalization or not a.keep_weights:
            hypers.update(mesh_normalization=a.mesh_normalization or "physical")
    elif a.mesh_normalization not in (None, "per_point"):
        raise SystemExit("--mesh-normalization physical needs metatrain's plato_pet_lr checkpoint v6")
    torch.manual_seed(a.seed)
    model = PlatoPETLR(hypers, ck["model_data"]["dataset_info"])
    state = model.state_dict()
    src = ck["best_model_state_dict"] or ck["model_state_dict"]
    for k in state:  # composition model and scaler (fitted), and with --keep-weights the rest
        if k in src and (a.keep_weights or k.startswith("additive_models.") or k.startswith("scaler.")):
            state[k] = src[k]
    g = torch.Generator().manual_seed(a.seed + 1)
    for chi_k in [k for k in state if k.startswith("electronegativity.0.")]:
        if chi_k.endswith("charge_unit"):
            state[chi_k] = torch.tensor(math.sqrt(scale), dtype=state[chi_k].dtype)
        if a.random_dchi > 0 and chi_k.endswith("head.l2.W"):
            state[chi_k] = a.random_dchi * torch.randn(state[chi_k].shape, generator=g, dtype=state[chi_k].dtype)
    model.load_state_dict(state)
    out = model.get_checkpoint()
    out["model_state_dict"] = out["best_model_state_dict"] = state
    out["metadata"] = ck.get("metadata")
    out.update(trainer_ckpt_version=Trainer.__checkpoint_version__, train_hypers=None, optimizer_state_dict=None,
               scheduler_state_dict=None, best_metric=None, best_optimizer_state_dict=None, epoch=None, best_epoch=None)
    torch.save(out, a.out)
    print(f"{a.out}: seed {a.seed}, {'source weights' if a.keep_weights else 'fresh weights'}, "
          f"frames {hypers.get('frames')}, mesh normalization {hypers.get('mesh_normalization', 'per_point')}, electronegativity {hypers.get('electronegativity', 'none')}, charge unit {math.sqrt(scale):.4f}")


if __name__ == "__main__":
    main()
