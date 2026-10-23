"""A freshly initialised plato_pet_lr checkpoint with the hypers, dataset info,
composition and energy scale of an existing one (fitted on the same training set):
the from-scratch start of a mlipcpp-train-plato run.

    python scripts/plato/make_init.py trained.ckpt init.ckpt --seed 0
    python scripts/plato/export_plato_gguf.py init.ckpt -o init.gguf
"""
import argparse

import metatomic.torch  # noqa: F401
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from metatrain.experimental.plato_pet_lr.model import PlatoPETLR
    from metatrain.experimental.plato_pet_lr.trainer import Trainer

    ck = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    torch.manual_seed(a.seed)
    model = PlatoPETLR(ck["model_data"]["model_hypers"], ck["model_data"]["dataset_info"])
    state = model.state_dict()
    src = ck["best_model_state_dict"] or ck["model_state_dict"]
    for k in state:  # composition model and scaler: the fitted ones
        if k.startswith("additive_models.") or k.startswith("scaler."):
            state[k] = src[k]
    model.load_state_dict(state)
    out = model.get_checkpoint()
    out["model_state_dict"] = out["best_model_state_dict"] = {k: (src[k] if (k.startswith("additive_models.") or k.startswith("scaler.")) else v)
                                                               for k, v in model.state_dict().items()}
    out["metadata"] = ck.get("metadata")
    out.update(trainer_ckpt_version=Trainer.__checkpoint_version__, train_hypers=None, optimizer_state_dict=None,
               scheduler_state_dict=None, best_metric=None, best_optimizer_state_dict=None, epoch=None, best_epoch=None)
    torch.save(out, a.out)
    print(f"{a.out}: fresh weights (seed {a.seed}), composition and scale of {a.checkpoint}")


if __name__ == "__main__":
    main()
