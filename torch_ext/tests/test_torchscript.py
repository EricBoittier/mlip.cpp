"""TorchScript: a module calling torch.ops.mlipcpp.pet_attention is scripted,
saved, and loaded in a fresh process that only loads the shared library (no
mlipcpp_torch Python package, no JIT build), as metatomic / LAMMPS would; the
loaded module gives the same energy, forces and force-training gradients."""

import math
import os
import subprocess
import sys
import tempfile

import torch

import mlipcpp_torch as mt


class PetLikeAttention(torch.nn.Module):
    def __init__(self, D: int = 64, H: int = 4):
        super().__init__()
        self.H = H
        self.scale = 1.0 / math.sqrt(D // H)
        self.emb = torch.nn.Linear(4, D)
        self.qkv = torch.nn.Linear(D, 3 * D)
        self.out = torch.nn.Linear(D, 1)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        dist = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
        x = self.emb(torch.cat([r, dist], -1))
        fc = 0.5 * (torch.cos(math.pi * (dist / 5.0).clamp(max=1.0)) + 1.0)
        bias = torch.log(fc.clamp_min(1e-15)).transpose(1, 2).expand(-1, r.shape[1], -1)
        o, lse = torch.ops.mlipcpp.pet_attention(self.qkv(x), bias.contiguous(), self.H, self.scale)
        return self.out(o).sum()


def energy_forces_grads(model, r):
    r = r.clone().requires_grad_()
    e = model(r)
    (f,) = torch.autograd.grad(e, r, create_graph=True)
    loss = (f**2).sum()
    gs = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    return [e.detach(), f.detach()] + [torch.zeros(()) if g is None else g.detach().cpu() for g in gs]


CHILD = r"""
import sys, torch
torch.ops.load_library(sys.argv[1])
m = torch.jit.load(sys.argv[2])
r = torch.load(sys.argv[3])
r.requires_grad_()
e = m(r)
(f,) = torch.autograd.grad(e, r, create_graph=True)
gs = torch.autograd.grad((f ** 2).sum(), list(m.parameters()), allow_unused=True)
torch.save([e.detach(), f.detach()] + [torch.zeros(()) if g is None else g.detach().cpu() for g in gs], sys.argv[4])
"""

if __name__ == "__main__":
    lib = mt.load()
    torch.manual_seed(0)
    model = PetLikeAttention().cuda()
    r = 2.0 * torch.randn(50, 16, 3, device="cuda")
    eager = energy_forces_grads(model, r)
    scripted = torch.jit.script(model)
    print(scripted.code)
    ref = energy_forces_grads(scripted, r)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        scripted.save(path)
        torch.save(r, os.path.join(tmp, "r.pt"))
        out = os.path.join(tmp, "out.pt")
        subprocess.run([sys.executable, "-c", CHILD, lib, path, os.path.join(tmp, "r.pt"), out], check=True,
                       env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
        loaded = torch.load(out)
    worst = 0.0
    for a, b, c in zip(eager, ref, loaded):
        a, b, c = a.cpu(), b.cpu(), c.cpu()
        worst = max(worst, (a - b).abs().max().item(), (a - c).abs().max().item())
    print(f"energy {eager[0].item():.6f}; max |eager - scripted|, |eager - reloaded| over E, F, dL/dparams: {worst:.2e}")
    print("libraries a metatomic export would collect:", torch.ops.loaded_libraries)
    sys.exit(0 if worst < 1e-5 else 1)
