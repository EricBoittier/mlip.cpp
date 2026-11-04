#!/usr/bin/env python3
"""metatrain PET (checkpoint) energies and conservative forces on an extended XYZ, against an
mlipcpp-eval JSON of the exported GGUF: max |dE| (eV) and max |dF| (eV/A), per frame.

    python run.py torch_parity.py MODEL.ckpt IN.xyz EVAL.json
"""
import json
import sys

import numpy as np
import torch


def main():
    ckpt, xyz, ev = sys.argv[1:4]
    import ase.io
    import metatensor.torch as mts
    import metatomic.torch  # noqa: F401
    from metatomic.torch import ModelOutput, System
    from metatrain.pet import PET
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    model = PET.load_checkpoint(torch.load(ckpt, map_location="cpu", weights_only=False), "export")
    model = model.to(torch.float64).eval()
    nl = model.requested_neighbor_lists()
    frames = json.load(open(ev))["frames"]

    def scalar(v):
        return mts.TensorMap(mts.Labels.single(), [mts.TensorBlock(
            torch.tensor([[float(v)]], dtype=torch.float64), mts.Labels.single(),
            [], mts.Labels("property", torch.tensor([[0]])))])

    worst_e = worst_f = 0.0
    for at, fr in zip(ase.io.read(xyz, ":"), frames):
        pos = torch.tensor(at.positions, dtype=torch.float64, requires_grad=True)
        s = System(types=torch.tensor(at.numbers), positions=pos, cell=torch.zeros(3, 3, dtype=torch.float64),
                   pbc=torch.zeros(3, dtype=torch.bool))
        s.add_data("charge", scalar(at.info.get("charge", 0)))
        s.add_data("spin_multiplicity", scalar(at.info.get("spin_multiplicity", 1)))
        s = get_system_with_neighbor_lists(s, nl)
        E = model([s], {"energy": ModelOutput()})["energy"].block().values.sum()
        (g,) = torch.autograd.grad(E, pos)
        dE = abs(float(E) - fr["energy"])
        dF = float(np.abs(-g.numpy() - np.array(fr["forces"]).reshape(-1, 3)).max())
        worst_e, worst_f = max(worst_e, dE), max(worst_f, dF)
        print(f"{len(at):4d} atoms q {at.info.get('charge', 0):+d}: E torch {float(E):.6f} mlipcpp {fr['energy']:.6f} "
              f"|dE| {dE:.2e} eV, max|dF| {dF:.2e} eV/A")
    print(f"max |dE| {worst_e:.3e} eV, max |dF| {worst_f:.3e} eV/A")


if __name__ == "__main__":
    main()
