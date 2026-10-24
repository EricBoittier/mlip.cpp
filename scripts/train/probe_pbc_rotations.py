#!/usr/bin/env python3
"""Does an exported model see rotated periodic copies correctly when they are batched?

o3diag's probe_pet3.py for any model.pt: energies of one test structure over the L=11
rotation grid, with and without the periodic cell, batched (256 per call) and one at a
time. The study's metatrain PET exports gave wrong batched periodic energies (up to
0.3 eV off), so their symmetry diagnostics were run without the cell (--nopbc).

    o3diag/venv/bin/python probe_pbc_rotations.py MODEL.pt [IDX ...]
"""
import sys

sys.path.insert(0, "/work/cosmo/boittier/lyra/plato/o3diag/scripts")
import ase.io  # noqa: E402
import numpy as np  # noqa: E402
import o3diag  # noqa: E402
import torch  # noqa: E402
from metatensor.torch import TensorBlock  # noqa: E402
from metatomic.torch import ModelEvaluationOptions, ModelOutput, System, register_autograd_neighbors  # noqa: E402
from metatomic.torch.o3._quadrature import choose_quadrature, get_rotation_quadrature  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
path, ids = sys.argv[1], [int(i) for i in sys.argv[2:]] or [45, 2]
dt = "float32"
opts = ModelEvaluationOptions(length_unit="angstrom", outputs={"energy": ModelOutput(sample_kind="system")})
rots, w = get_rotation_quadrature(*choose_quadrature(11))
model = o3diag.load_model(path, dt, "cuda")
for i in ids:
    fr = ase.io.read("/work/cosmo/boittier/lyra/plato/study/data/test.xyz", str(i))
    res = {}
    for pbc in [True, False]:
        a = fr.copy()
        if not pbc:
            a.pbc = False
            a.cell = np.zeros((3, 3))
        s0 = o3diag.make_system(a, dt, "cuda", model)
        o = s0.known_neighbor_lists()[0]
        nl = s0.get_neighbor_list(o)

        def build(Q):
            s = System(types=s0.types, positions=s0.positions @ Q.T, cell=s0.cell @ Q.T, pbc=s0.pbc)
            blk = TensorBlock((nl.values.squeeze(-1) @ Q.T).unsqueeze(-1), nl.samples, nl.components, nl.properties)
            register_autograd_neighbors(s, blk)
            s.add_neighbor_list(o, blk)
            return s

        Qs = torch.tensor(rots, dtype=s0.positions.dtype, device="cuda")
        with torch.no_grad():
            Eb = []
            for st in range(0, len(Qs), 256):
                Eb.append(model([build(Q) for Q in Qs[st:st + 256]], opts, check_consistency=False)["energy"].block().values.reshape(-1))
            Eb = torch.cat(Eb).double().cpu().numpy()
            Es = np.array([model([build(Q)], opts, check_consistency=False)["energy"].block().values.item() for Q in Qs[:600:5]])
        res[pbc] = Eb
        print(f"struct {i} N={len(fr)}", "pbc" if pbc else "nopbc",
              "batched: mean %.6f std %.3e | single (every 5th): std %.3e | max |batched-single| %.3e"
              % (Eb.mean(), Eb.std(), Es.std(), np.abs(Eb[::5] - Es).max()))
    print(f"struct {i}: max |E_pbc - E_nopbc| over the batched rotations {np.abs(res[True] - res[False]).max():.3e} eV")
