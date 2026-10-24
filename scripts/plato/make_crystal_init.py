"""A freshly initialised plato_pet_lr checkpoint for a new element set: the hypers and target
info of a template checkpoint, the atomic types of a fitted baseline (plato_dataset.py
--fit-baseline), optional hyper overrides. The composition energies and the energy scale go
into the GGUF at export (export_plato_gguf.py --baseline), not into the checkpoint.

    python scripts/plato/make_crystal_init.py template.ckpt baseline.json init.ckpt --seed 0 \
        [--frames identity|D2|T] [--set d_model=128 --set num_layers=4 ...]
    python scripts/plato/export_plato_gguf.py init.ckpt --baseline baseline.json -o init.gguf
"""
import argparse
import json

import metatomic.torch  # noqa: F401
import torch
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template")
    ap.add_argument("baseline")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--set", action="append", default=[], help="hyper=value (YAML value)")
    a = ap.parse_args()
    from metatrain.experimental.plato_pet_lr.model import PlatoPETLR
    from metatrain.experimental.plato_pet_lr.trainer import Trainer
    from metatrain.utils.data import DatasetInfo

    from metatrain.utils.architectures import get_default_hypers

    ck = torch.load(a.template, weights_only=False, map_location="cpu")
    # the architecture's defaults for hypers newer than the template (e.g. electronegativity)
    hypers = dict(get_default_hypers("experimental.plato_pet_lr")["model"])
    hypers.update(ck["model_data"]["model_hypers"])
    if a.frames:
        hypers["frames"] = a.frames
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in hypers:
            raise SystemExit(f"unknown hyper {k}")
        hypers[k] = yaml.safe_load(v)
    types = sorted(int(z) for z in json.load(open(a.baseline))["composition"])
    src = ck["model_data"]["dataset_info"]
    info = DatasetInfo(length_unit=src.length_unit, atomic_types=types, targets=src.targets)
    torch.manual_seed(a.seed)
    model = PlatoPETLR(hypers, info)
    out = model.get_checkpoint()
    out["model_state_dict"] = out["best_model_state_dict"] = model.state_dict()
    out["metadata"] = ck.get("metadata")
    out.update(trainer_ckpt_version=Trainer.__checkpoint_version__, train_hypers=None, optimizer_state_dict=None,
               scheduler_state_dict=None, best_metric=None, best_optimizer_state_dict=None, epoch=None, best_epoch=None)
    torch.save(out, a.out)
    n = sum(p.numel() for p in model.parameters())
    print(f"{a.out}: seed {a.seed}, {len(types)} elements, frames {hypers['frames']} ({hypers['frame_mode']}), "
          f"d_model {hypers['d_model']}, {n} parameters")


if __name__ == "__main__":
    main()
