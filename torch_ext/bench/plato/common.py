"""Shared helpers: load a Plato checkpoint eagerly, build systems, E+F and train step."""
import os
import sys
import types

try:  # not needed to read local checkpoints; absent from the GPU venv
    import huggingface_hub  # noqa: F401
except ImportError:
    _hf = types.ModuleType("huggingface_hub")
    _hf.hf_hub_download = None
    sys.modules["huggingface_hub"] = _hf
import numpy as np
import torch
from ase.io import read
from metatomic.torch import ModelOutput, systems_to_torch
from metatrain.utils.io import load_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.environ.get("PLATO_TEST_XYZ", os.path.join(HERE, "..", "test.xyz"))
_frames = None


def frames():
    global _frames
    if _frames is None:
        _frames = read(TEST, ":")
    return _frames


def cc():
    return next(a for a in frames() if a.info["label"] == "CC")


def batch64():
    rng = np.random.default_rng(0)
    f = frames()
    return [f[i] for i in rng.choice(len(f), 64, replace=False)]


def train_batch(n=32, seed=1):
    rng = np.random.default_rng(seed)
    f = frames()
    return [f[i] for i in rng.choice(len(f), n, replace=False)]


def load(ckpt, dtype=torch.float32, device="cpu"):
    m = load_model(ckpt)
    return m.to(dtype=dtype, device=device)


def make_systems(model, atoms, dtype, device):
    nl = model.requested_neighbor_lists()
    return [
        get_system_with_neighbor_lists(s, nl)
        for s in systems_to_torch(
            atoms, dtype=dtype, device=device, positions_requires_grad=True
        )
    ]


OUT = {"energy": ModelOutput(per_atom=False)}


def energy_forces(model, systems):
    E = model(systems, OUT)["energy"].block().values
    g = torch.autograd.grad(E.sum(), [s.positions for s in systems])
    return E.detach(), [-x for x in g]


def train_step(model, systems, e_ref, f_ref):
    """forward + forces (create_graph) + MSE(E) + MSE(F) + backward, like the trainer."""
    E = model(systems, OUT)["energy"].block().values[:, 0]
    pos = [s.positions for s in systems]
    g = torch.autograd.grad(E.sum(), pos, create_graph=True)
    F = -torch.cat(g)
    loss = ((E - e_ref) ** 2).mean() + ((F - f_ref) ** 2).mean()
    loss.backward()
    return loss.detach()


def refs(atoms, dtype, device):
    e = torch.tensor([a.info["e_res"] for a in atoms], dtype=dtype, device=device)
    f = torch.tensor(np.concatenate([a.arrays["f"] for a in atoms]), dtype=dtype, device=device)
    return e, f
