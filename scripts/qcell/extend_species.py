"""A plato_pet_lr checkpoint with more atomic types than its source (QCell's 20 elements from
bio_dimers' H, C, N, O), as the start of a mlipcpp-train-plato fine-tune (or, with --fresh,
of a from-scratch run with the same hypers, scaler and composition handling).

    python extend_species.py source.ckpt out.ckpt --types 1,2,3,6,... [--fresh --seed S]

The model is rebuilt with the new dataset_info (same hypers). Every tensor of the source is
copied except the species-dependent ones:
  embed.sp_center.weight, embed.sp_nbr.weight [n_types, d]: rows of the source types are
    copied; a new type gets the row of a same-group element of the source if there is one
    (P <- N, S <- O, As <- N, Se <- O, Si <- C) and otherwise the mean of the source rows;
  species_to_species_index, *.type_to_index: the new model's;
  additive_models.0.energy_composition_buffer: the new model's (all zero; the source's is
    zero to 1e-12: bio_dimers' per-element baseline lives outside the model, in the data);
  scaler buffers: the source's energy scale (a single value) written into the new layout.
--fresh keeps the new model's own random weights (torch.manual_seed(seed)) everywhere except
the scaler and composition, as scripts/plato/make_init.py does.
"""
import argparse
import json

import metatensor.torch as mts
import metatomic.torch  # noqa: F401
import torch

ANALOG = {15: 7, 16: 8, 33: 7, 34: 8, 14: 6}
SPECIES_ROWS = ("embed.sp_center.weight", "embed.sp_nbr.weight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--types", required=True)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from metatrain.experimental.plato_pet_lr.model import PlatoPETLR
    from metatrain.experimental.plato_pet_lr.trainer import Trainer
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.io import load_model

    ck = torch.load(a.source, weights_only=False, map_location="cpu")
    src_model = load_model(a.source)
    src = ck["best_model_state_dict"] or ck["model_state_dict"]
    old_types = [int(z) for z in src_model.dataset_info.atomic_types]
    new_types = sorted(set(int(z) for z in a.types.split(",")) | set(old_types))
    info = src_model.dataset_info
    new_info = DatasetInfo(length_unit=info.length_unit, atomic_types=new_types, targets=info.targets)
    torch.manual_seed(a.seed)
    model = PlatoPETLR(dict(src_model.hypers), new_info)
    state = model.state_dict()
    report = {"old_types": old_types, "new_types": new_types, "fresh": a.fresh, "seed": a.seed, "rows": {}}
    for k in state:
        if k in SPECIES_ROWS:
            if a.fresh:
                continue
            old = src[k]
            assert old.shape[0] == len(old_types) and state[k].shape[0] == len(new_types), (k, old.shape, state[k].shape)
            new = torch.empty_like(state[k])
            for i, z in enumerate(new_types):
                if z in old_types:
                    new[i] = old[old_types.index(z)]
                    how = "copied"
                elif ANALOG.get(z) in old_types:
                    new[i] = old[old_types.index(ANALOG[z])]
                    how = f"from Z={ANALOG[z]}"
                else:
                    new[i] = old.mean(0)
                    how = "mean of source rows"
                report["rows"][str(z)] = how
            state[k] = new
        elif k.startswith("scaler.") and k.endswith("_buffer") and k != "scaler.dummy_buffer":
            tm_new = mts.load_buffer(state[k].cpu())
            tm_old = mts.load_buffer(src[k].cpu())
            vals = [b.values.reshape(-1) for b in tm_old.blocks()]
            assert all(v.numel() == 1 for v in vals), (k, [v.shape for v in vals])
            blocks = []
            for b, v in zip(tm_new.blocks(), vals):
                assert b.values.numel() == 1, (k, b.values.shape)
                blocks.append(mts.TensorBlock(values=v.reshape(b.values.shape).to(b.values.dtype), samples=b.samples,
                                              components=b.components, properties=b.properties))
            state[k] = mts.save_buffer(mts.make_contiguous(mts.TensorMap(tm_new.keys, blocks)))
            report.setdefault("scaler", {})[k] = [float(v) for v in vals]
        elif k.startswith("additive_models.") or k.endswith("type_to_index") or k == "species_to_species_index":
            continue  # the new model's (types layout); composition all zero
        elif not a.fresh:
            assert src[k].shape == state[k].shape, (k, src[k].shape, state[k].shape)
            state[k] = src[k]
    model.load_state_dict(state)
    model.additive_models[0].sync_tensor_maps()
    model.scaler.sync_tensor_maps()
    comp = model.additive_models[0].model.weights["energy"].block(0).values
    assert float(comp.abs().max()) == 0.0, comp
    out = model.get_checkpoint()
    out["model_state_dict"] = out["best_model_state_dict"] = state
    out["metadata"] = ck.get("metadata")
    out.update(trainer_ckpt_version=Trainer.__checkpoint_version__, train_hypers=None, optimizer_state_dict=None,
               scheduler_state_dict=None, best_metric=None, best_optimizer_state_dict=None, epoch=None, best_epoch=None)
    torch.save(out, a.out)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
