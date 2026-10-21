#!/usr/bin/env python3
"""Turn an mlipcpp-train GGUF back into a metatrain PET checkpoint (fine-tuning).

The GGUF holds every trainable tensor under the export wrapper's parameter
name ('.' -> '_', row-major PyTorch layout, see export_pet_gguf.py) and the
baselines mlipcpp-train fitted (composition, energy scale, per-species
non-conservative force scales). The architecture comes from the make_init.py
dump the run started from; its own (unfitted) baselines are ignored.

Writes OUT.ckpt (`model.get_checkpoint()`, for `finetune.read_from`) with the
GGUF's baselines in its composition and scaler buffers (a fine-tune keeps
those for existing targets and ignores atomic_baseline / fixed_scaling_weights),
and OUT.baselines.yaml with the same values for reference.

    python gguf_to_checkpoint.py --init init.pt --gguf model.gguf -o big24 [--check DATA/test --n 64]

--check: energies and direct forces of metatrain's PET (evaluation mode, with
the baselines applied) against the GGUF's, through GraphModel's CPU reference
values written by `mlipcpp-md check --csv` if given (--csv), else printed
against the labels only.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_pytorch.export_pet_full import PETFullModel, _unwrap_to_pet, get_model_params  # noqa: E402
from export_pytorch.export_pet_full import load_local_model  # noqa: E402

GGML_F32 = 0
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_gguf(path):
    """(metadata, {name: float32 array}) of a GGUF v3 file with F32 tensors."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0

    def take(fmt):
        nonlocal pos
        out = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return out[0] if len(out) == 1 else out

    def string():
        nonlocal pos
        n = take("Q")
        s = data[pos: pos + n].decode()
        pos += n
        return s

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(SCALAR[t])

    assert data[:4] == b"GGUF"
    pos = 4
    _, n_tensors, n_kv = take("I"), take("Q"), take("Q")
    meta = {}
    for _ in range(n_kv):
        key = string()
        meta[key] = value(take("I"))
    infos = []
    for _ in range(n_tensors):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        infos.append((name, dims, take("I"), take("Q")))
    align = meta.get("general.alignment", 32)
    base = (pos + align - 1) // align * align
    tensors = {}
    for name, dims, typ, off in infos:
        if typ != GGML_F32:
            raise ValueError(f"{name}: tensor type {typ}, only F32 is supported")
        count = int(np.prod(dims))
        tensors[name] = np.frombuffer(data, np.float32, count, base + off)
    return meta, tensors


def baselines(meta):
    comp = dict(zip(meta["pet.composition_keys"], meta["pet.composition_values"]))
    fixed = {"energy": float(meta["pet.energy_scale"])}
    if "pet.nc_force_scale_keys" in meta:
        fixed["non_conservative_force"] = {
            int(z): float(v) for z, v in zip(meta["pet.nc_force_scale_keys"], meta["pet.nc_force_scale_values"])}
    return {"atomic_baseline": {"energy": {int(z): float(v) for z, v in comp.items()}},
            "fixed_scaling_weights": fixed}


def load_weights(model, tensors, param_names):
    pet = _unwrap_to_pet(model)
    p = get_model_params(pet)
    wrapper = PETFullModel(pet, n_atoms=7, max_neighbors=11, d_pet=p["d_pet"], forces=True,
                           cutoff=p["cutoff"], cutoff_width=p["cutoff_width"],
                           cutoff_function=p["cutoff_function"], nc_forces=True)
    named = {n.replace(".", "_"): t for n, t in wrapper.named_parameters()}
    missing = [n for n in param_names if n not in named or n not in tensors]
    if missing:
        raise ValueError(f"{len(missing)} trained tensors without a match, e.g. {missing[:3]}")
    untrained = sorted(set(named) - set(param_names))
    with torch.no_grad():
        for name in param_names:
            t = named[name]
            src = tensors[name]
            if src.size != t.numel():
                raise ValueError(f"{name}: {src.size} values in the GGUF, {t.numel()} in the model")
            t.copy_(torch.from_numpy(src.reshape(tuple(t.shape)).copy()))
    return len(param_names), untrained


def set_baselines(model, b):
    """Write the GGUF's baselines into the model, through metatrain's own fits
    (all weights fixed, so no data is needed) and buffer layout (as its
    composition / scaler trainers do). A fine-tune keeps the checkpoint's
    baselines for targets it already has, ignoring atomic_baseline and
    fixed_scaling_weights, so they must be right in the checkpoint."""
    import metatensor.torch as mts

    comp = model.additive_models[0]
    comp.model.fit(fixed_weights=b["atomic_baseline"], targets_to_fit=list(b["atomic_baseline"]))
    for t in comp.model.weights.keys():
        comp.register_buffer(t + "_composition_buffer",
                             mts.save_buffer(mts.make_contiguous(comp.model.weights[t].to("cpu", torch.float64))))
    sc = model.scaler
    sc.model.fit(fixed_weights=b["fixed_scaling_weights"], targets_to_fit=list(b["fixed_scaling_weights"]))
    for t in sc.model.scales.keys():
        sc.register_buffer(t + "_scaler_buffer",
                           mts.save_buffer(mts.make_contiguous(sc.model.scales[t].to("cpu", torch.float64))))
        sc.register_buffer(t + "_per_target_scaler_buffer",
                           mts.save_buffer(mts.make_contiguous(sc.model.per_target_scales[t].to("cpu", torch.float64))))
    # read back the way export_pet_gguf.py does
    from export_pytorch.export_pet_gguf import metatrain_baselines
    comp_r, e_r, f_r = metatrain_baselines(model)
    want_c, want_f = b["atomic_baseline"]["energy"], b["fixed_scaling_weights"].get("non_conservative_force", {})
    dc = max(abs(comp_r.get(z, float("nan")) - v) for z, v in want_c.items())
    df = max((abs(f_r.get(z, float("nan")) - v) for z, v in want_f.items()), default=0.0)
    de = abs(e_r - b["fixed_scaling_weights"]["energy"])
    print(f"baselines set: |composition diff| {dc:.2e}, |energy scale diff| {de:.2e}, |force scale diff| {df:.2e}")
    if not (dc < 1e-5 and de < 1e-6 and df < 1e-5):
        raise RuntimeError("the baselines did not take")


def check(model, split_dir, n, csv):
    from metatomic.torch import ModelOutput, System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    d = Path(split_dir)
    na = np.load(d / "na.npy")
    x = np.fromfile(d / "x.bin", np.float32).reshape(-1, 3)
    a = np.fromfile(d / "a.bin", np.int32)
    c = np.fromfile(d / "c.bin", np.float32).reshape(-1, 3, 3)
    e = np.fromfile(d / "energy.bin", np.float32)
    ref = {}
    if csv:
        for row in np.genfromtxt(csv, delimiter=",", names=True):
            ref[int(row["index"])] = (row["E_cpu"], row["E_trainer"])
    ids = sorted(ref)[:n] if ref else list(range(0, len(na) - 1, max(1, (len(na) - 1) // n)))[:n]
    model = model.to(torch.float64).eval()
    nl = model.requested_neighbor_lists()
    rows = []
    for s in ids:
        lo, hi = na[s], na[s + 1]
        cell = torch.tensor(c[s], dtype=torch.float64)
        sys_ = System(types=torch.tensor(a[lo:hi]), positions=torch.tensor(x[lo:hi], dtype=torch.float64),
                      cell=cell, pbc=(cell != 0).any(dim=1))
        out = model([get_system_with_neighbor_lists(sys_, nl)], {"energy": ModelOutput()})
        E = float(out["energy"].block().values.sum())
        rows.append((s, hi - lo, E, float(e[s])) + ref.get(s, (np.nan, np.nan)))
    r = np.array(rows, dtype=float)
    per = lambda col: 1e3 * (r[:, 2] - r[:, col]) / r[:, 1]
    rms = lambda v: float(np.sqrt(np.nanmean(v ** 2)))
    print(f"{len(rows)} structures: metatrain E vs labels {rms(per(3)):.2f} meV/atom"
          + (f", vs GraphModel CPU {rms(per(4)):.3f} (max {np.nanmax(np.abs(per(4))):.3f}),"
             f" vs trainer {rms(per(5)):.3f} meV/atom" if ref else ""))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="make_init.py dump of the run's architecture")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("-o", "--output", required=True, help="prefix: OUT.ckpt, OUT.baselines.yaml")
    ap.add_argument("--check", help="memmap split for a forward check")
    ap.add_argument("--csv", help="mlipcpp-md check --csv output (GraphModel CPU energies)")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    meta, tensors = read_gguf(args.gguf)
    if meta.get("pet.padding_mask") == "exact":
        print("note: trained with the exact padding mask (pet.padding_mask = exact); in metatrain set "
              "METATRAIN_PET_PAD_EXACT=1 with pet_exact_mask.py to reproduce it")
    param_names = json.loads(meta["train.param_names"])
    model = load_local_model(args.init)
    n, untrained = load_weights(model, tensors, param_names)
    print(f"loaded {n} trained tensors ({sum(t.size for k, t in tensors.items() if k in param_names)} values)")
    if untrained:
        print(f"  parameters not in train.param_names (kept from --init): {untrained}")
    b = baselines(meta)
    set_baselines(model, b)
    Path(args.output + ".baselines.yaml").write_text(yaml.safe_dump(b, sort_keys=False))
    ckpt = model.get_checkpoint()
    ckpt["epoch"] = ckpt["best_epoch"] = 0
    torch.save(ckpt, args.output + ".ckpt")
    print(f"wrote {args.output}.ckpt and {args.output}.baselines.yaml "
          f"({len(b['atomic_baseline']['energy'])} composition energies, energy scale {b['fixed_scaling_weights']['energy']:.6f})")
    if args.check:
        check(model, args.check, args.n, args.csv)


if __name__ == "__main__":
    main()
