"""Charge / spin conditioning for metatrain's ``experimental.plato_pet_lr`` (PyTorch
reference of mlip.cpp's Plato conditioning).

metatrain's Plato-PET-LR has no system conditioning (metatrain 76a1fbf7). This adds
PET's own module, ``metatrain.pet.modules.conditioning.SystemConditioningEmbedding``
(d_out = d_model), as ``model.system_conditioning``, with PET's semantics: the
projected embedding of each structure's charge and spin multiplicity is added to the
centre tokens (slot 0, every token frame) after every layer, i.e. before the mesh
coupling of layer ``couple_at`` (PET adds it to the node features after every GNN
layer, before the next one and the readout). Data and checks as PET: the systems'
``charge`` / ``spin_multiplicity`` data (absent: 0 and 1), integers within
[-max_charge, max_charge] and [1, max_spin_multiplicity], an error otherwise.

Checkpoints carry it as hypers ``system_conditioning`` (true), ``max_charge``,
``max_spin_multiplicity`` and state-dict entries ``system_conditioning.*``; a stock
metatrain does not load them, so use :func:`load_plato_model` (which strips them,
loads the rest with metatrain and attaches the module again).

    model = attach_conditioning(model, max_charge=3, max_spin_multiplicity=4)
    model = load_plato_model("cond.ckpt")
    torch.save(model.get_checkpoint(), "cond.ckpt")   # hypers and tensors included
"""

import os
import tempfile
from typing import List

import torch

KEYS = ("system_conditioning", "max_charge", "max_spin_multiplicity")
PREFIX = "system_conditioning."


def _per_atom(model, systems: List) -> torch.Tensor:
    from metatrain.pet.model import _extract_charge_spin_multiplicity

    sc = model.system_conditioning
    device = systems[0].positions.device
    q, s = _extract_charge_spin_multiplicity(systems, device)
    sc.validate(q, s)
    counts = torch.tensor([len(x) for x in systems], device=device)
    index = torch.repeat_interleave(torch.arange(len(systems), device=device), counts)
    return sc(q, s, index)  # [A, d]


def _pre_forward(model, args, kwargs):
    systems = args[0] if args else kwargs["systems"]
    model._cond_atoms = _per_atom(model, systems)


def _after_layer(model):
    def hook(layer, args, tok):
        cond = model._cond_atoms
        if cond is None:
            return tok
        # tok: [A, K+1, F, d]; the centre token of every frame
        return torch.cat([tok[:, :1] + cond.to(tok.dtype)[:, None, None, :], tok[:, 1:]], 1)
    return hook


def attach_conditioning(model, max_charge=10, max_spin_multiplicity=10, state=None):
    """Add the conditioning module (zero output gate, as PET) and its hooks."""
    from metatrain.pet.modules.conditioning import SystemConditioningEmbedding

    if getattr(model, "system_conditioning", None) is not None:
        return model
    d = int(model.hypers["d_model"])
    dtype = next(model.parameters()).dtype
    model.system_conditioning = SystemConditioningEmbedding(d, int(max_charge), int(max_spin_multiplicity)).to(dtype)
    if state:
        model.system_conditioning.load_state_dict(state)
    model.hypers["system_conditioning"] = True
    model.hypers["max_charge"] = int(max_charge)
    model.hypers["max_spin_multiplicity"] = int(max_spin_multiplicity)
    model._cond_atoms = None
    model.register_forward_pre_hook(_pre_forward, with_kwargs=True)
    for layer in model.layers:
        layer.register_forward_hook(_after_layer(model))
    return model


def split_checkpoint(ckpt):
    """(checkpoint without the conditioning, (max_charge, max_spin), its state dict or None)."""
    hy = ckpt["model_data"]["model_hypers"]
    if not hy.get("system_conditioning"):
        return ckpt, None, None
    ckpt = dict(ckpt)
    ckpt["model_data"] = dict(ckpt["model_data"])
    hy = dict(hy)
    cfg = (hy.pop("max_charge", 10), hy.pop("max_spin_multiplicity", 10))
    hy.pop("system_conditioning")
    ckpt["model_data"]["model_hypers"] = hy
    cond = None
    for key in ("model_state_dict", "best_model_state_dict"):
        sd = ckpt.get(key)
        if sd is None:
            continue
        sd = dict(sd)
        part = {k[len(PREFIX):]: sd.pop(k) for k in list(sd) if k.startswith(PREFIX)}
        if key == "best_model_state_dict" or cond is None:
            cond = part or cond
        ckpt[key] = sd
    return ckpt, cfg, cond


def load_plato_model(path):
    """metatrain's load_model for a plato_pet_lr checkpoint, with or without conditioning."""
    import metatomic.torch  # noqa: F401
    from metatrain.utils.io import load_model

    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    stripped, cfg, cond = split_checkpoint(ckpt)
    if cfg is None:
        return load_model(path)
    fd, tmp = tempfile.mkstemp(suffix=".ckpt")
    os.close(fd)
    try:
        torch.save(stripped, tmp)
        model = load_model(tmp)
    finally:
        os.unlink(tmp)
    return attach_conditioning(model, cfg[0], cfg[1], state=cond)


def checkpoint_state(path):
    """(raw checkpoint, the state dict export_plato_gguf.py writes) incl. conditioning."""
    import metatomic.torch  # noqa: F401
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    return ckpt, ckpt["best_model_state_dict"] or ckpt["model_state_dict"]
