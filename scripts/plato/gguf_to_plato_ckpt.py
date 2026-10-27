"""GGUF (export_plato_gguf.py, or one mlipcpp-train wrote) back to a metatrain
``experimental.plato_pet_lr`` checkpoint.

    python scripts/plato/gguf_to_plato_ckpt.py model.gguf -o model.ckpt [--check orig.ckpt]

The model is rebuilt from the stored hypers and dataset info and every state-dict
tensor is loaded from the GGUF bits. The checkpoint has no optimizer or scheduler state
and no epoch: ``mtt train options.yaml --restart model.ckpt`` trains from these
weights with a fresh Adam and schedule (a fine-tune). ``--check`` compares every tensor with the one in
the original checkpoint's best state dict, bit for bit.
"""

import argparse
import json

import gguf
import metatomic.torch  # noqa: F401
import numpy as np
import torch

NP_TO_TORCH = {np.float32: torch.float32, np.float64: torch.float64, np.int64: torch.int64,
               np.int32: torch.int32, np.int8: torch.uint8, np.uint8: torch.uint8}


def read_gguf(path):
    r = gguf.GGUFReader(path)
    meta = {}
    for f in r.fields.values():
        if f.name.startswith("plato.") or f.name.startswith("train."):
            v = f.contents()
            meta[f.name] = v
    tensors = {}
    for t in r.tensors:
        a = np.array(t.data)
        shape = [int(x) for x in reversed(t.shape.tolist())]
        tensors[t.name] = a.reshape(shape)
    return meta, tensors


def build_model(meta):
    from metatrain.experimental.plato_pet_lr.model import PlatoPETLR
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_energy_target_info

    info = json.loads(meta["plato.dataset_info"])
    targets = {
        n: get_energy_target_info(n, {"unit": t["unit"], "description": t["description"]},
                                  add_position_gradients="positions" in t["gradients"],
                                  add_strain_gradients="strain" in t["gradients"])
        for n, t in info["targets"].items()
    }
    ds = DatasetInfo(length_unit=info["length_unit"], atomic_types=info["atomic_types"], targets=targets)
    # hypers of an older checkpoint version: the model's own upgrade path
    ck = {"model_ckpt_version": int(meta.get("plato.checkpoint_version", PlatoPETLR.__checkpoint_version__)),
          "model_data": {"model_hypers": json.loads(meta["plato.hypers"])}}
    # charge / spin conditioning (plato_conditioning.py): built without, then attached
    hy = ck["model_data"]["model_hypers"]
    cond = {k: hy.pop(k) for k in ("system_conditioning", "max_charge", "max_spin_multiplicity") if k in hy}
    PlatoPETLR.upgrade_checkpoint(ck)
    model = PlatoPETLR(ck["model_data"]["model_hypers"], ds)
    if cond.get("system_conditioning"):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plato_conditioning import attach_conditioning
        attach_conditioning(model, cond.get("max_charge", 10), cond.get("max_spin_multiplicity", 10))
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--check", help="original checkpoint to compare with, bit for bit")
    args = ap.parse_args()

    meta, tensors = read_gguf(args.gguf)
    model = build_model(meta)
    names = json.loads(meta["plato.state_dict_names"])
    ref = model.state_dict()
    state = {}
    for n in names:
        a = tensors[n]
        dt = ref[n].dtype
        state[n] = torch.from_numpy(a.view(np.uint8) if a.dtype == np.int8 else a.copy()).reshape(ref[n].shape)
    model.load_state_dict(state)  # checks names and shapes
    model.additive_models[0].sync_tensor_maps()
    model.scaler.sync_tensor_maps()
    ckpt = model.get_checkpoint()
    # the GGUF's tensors as they are (load_state_dict casts float64 buffers into the
    # fresh model's float32 ones); load_checkpoint reads the best state dict
    ckpt["model_state_dict"] = ckpt["best_model_state_dict"] = state
    ckpt["epoch"] = ckpt["best_epoch"] = None
    # trainer state for `mtt train --restart`: none, so the restart starts at epoch 0 with a
    # fresh optimizer and schedule from these weights
    from metatrain.experimental.plato_pet_lr.trainer import Trainer

    ckpt.update(trainer_ckpt_version=Trainer.__checkpoint_version__, train_hypers=None, optimizer_state_dict=None,
                scheduler_state_dict=None, best_metric=None, best_optimizer_state_dict=None)
    torch.save(ckpt, args.output)
    print(f"{args.output}: {len(state)} tensors")

    if args.check:
        orig = torch.load(args.check, weights_only=False, map_location="cpu")
        osd = orig["best_model_state_dict"] or orig["model_state_dict"]
        bad = [n for n in osd if n not in state or osd[n].dtype != state[n].dtype or osd[n].shape != state[n].shape
               or not torch.equal(osd[n].reshape(-1).view(torch.uint8), state[n].reshape(-1).view(torch.uint8))]
        extra = [n for n in state if n not in osd]
        print(f"bit-identical: {len(osd) - len(bad)}/{len(osd)} tensors; differ: {bad[:10]}; extra: {extra}")
        if bad or extra:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
