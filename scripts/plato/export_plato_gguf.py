"""Export a metatrain ``experimental.plato_pet_lr`` checkpoint to GGUF for mlip.cpp.

    python scripts/plato/export_plato_gguf.py model.ckpt -o model.gguf

Every entry of the model's state dict (the best model, as ``mtt export`` takes it) is
written verbatim: same name, dtype and bits (float32 weights, the int64 / float64 /
uint8 buffers of the composition model and the scaler). GGUF dimensions are the
reversed torch shape, so a torch ``Linear`` weight ``[out, in]`` is the ggml
``mul_mat`` operand ``[in, out]``. The hypers and the dataset info are stored as JSON,
the composition energies and the energy scale also as float64 metadata for the C++
side. ``gguf_to_plato_ckpt.py`` inverts this (bit-identical tensors).

mlip.cpp folds each ``FrameLinear`` with one token frame (``frame_mode: attention``)
into a plain linear map at load time; the raw ``W`` / ``phi`` stay the parameters, so
a GGUF written by a trainer that updates them converts back without loss.
"""

import argparse
import json

import gguf
import metatomic.torch  # noqa: F401  (registers the TorchScript classes of the checkpoint)
import numpy as np
import torch

ARCH = "plato-pet-lr"
TORCH_TO_NP = {torch.float32: np.float32, torch.float64: np.float64, torch.int64: np.int64,
               torch.int32: np.int32, torch.uint8: np.uint8, torch.bool: np.uint8}
NP_TO_GGML = {np.float32: gguf.GGMLQuantizationType.F32, np.float64: gguf.GGMLQuantizationType.F64,
              np.int64: gguf.GGMLQuantizationType.I64, np.int32: gguf.GGMLQuantizationType.I32,
              np.uint8: gguf.GGMLQuantizationType.I8}


def dataset_info_json(info) -> dict:
    targets = {}
    for name, t in info.targets.items():
        targets[name] = {"quantity": t.quantity, "unit": t.unit, "description": t.description,
                         "gradients": t.gradients, "per_atom": t.per_atom}
    return {"length_unit": info.length_unit, "atomic_types": list(info.atomic_types), "targets": targets}


def baselines(model):
    comp = {}
    w = model.additive_models[0].model.weights
    if "energy" in w:
        b = w["energy"].block(0)
        comp = {int(z): float(v) for z, v in zip(b.samples.values[:, 0], b.values.reshape(-1))}
    scales = model.scaler.model.scales
    b = scales["energy"].block(0)
    if b.values.numel() != 1:
        raise ValueError("per-type energy scales are not supported")
    return comp, float(b.values.reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    from metatrain.utils.io import load_model

    model = load_model(args.checkpoint)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    if ckpt["architecture_name"] != "experimental.plato_pet_lr":
        raise ValueError(f"not a plato_pet_lr checkpoint: {ckpt['architecture_name']}")
    hypers = dict(model.hypers)
    if hypers["frame_mode"] != "attention" or hypers["num_framed_layers"] not in (None, hypers["num_layers"]):
        raise ValueError("mlip.cpp builds frame_mode: attention with every layer framed")
    comp, scale = baselines(model)

    w = gguf.GGUFWriter(args.output, ARCH)
    w.add_string("plato.hypers", json.dumps(hypers))
    w.add_string("plato.dataset_info", json.dumps(dataset_info_json(model.dataset_info)))
    w.add_uint32("plato.checkpoint_version", int(ckpt["model_ckpt_version"]))
    w.add_string("plato.source", json.dumps({"epoch": ckpt.get("epoch"), "best_epoch": ckpt.get("best_epoch")}))
    w.add_array("plato.composition_types", list(comp))
    w.add_key_value("plato.composition_energies", list(comp.values()), gguf.GGUFValueType.ARRAY,
                    sub_type=gguf.GGUFValueType.FLOAT64)
    w.add_key_value("plato.energy_scale", scale, gguf.GGUFValueType.FLOAT64)
    w.add_float32("plato.cutoff", float(hypers["cutoff"]))
    names = []
    # the checkpoint's own tensors (load_model casts a few float64 buffers to float32)
    state = ckpt["best_model_state_dict"] or ckpt["model_state_dict"]
    for name, t in state.items():
        t = t.detach().cpu().contiguous()
        a = t.numpy().astype(TORCH_TO_NP[t.dtype], copy=False)
        if a.ndim == 0:
            a = a.reshape(1)
        w.add_tensor(name, a, raw_dtype=NP_TO_GGML[a.dtype.type])
        names.append(name)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    w.add_string("train.param_names", json.dumps(trainable))
    w.add_string("plato.state_dict_names", json.dumps(names))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.output}: {len(names)} tensors ({len(trainable)} trainable), scale {scale:.6g}, "
          f"composition {comp}")


if __name__ == "__main__":
    main()
