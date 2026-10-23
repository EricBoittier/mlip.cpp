"""PyTorch reference timing on the same node as the mlip.cpp runs, the study's method
(evaluate.py / cost10k.py): exported TorchScript model.pt, float32, CUDA, energy +
forces (autograd w.r.t. positions), neighbour lists excluded, median of N after 3
warmups. Structures: the first test CC dimer (20 atoms, 30 A box) tiled K^3.

    python time_torch.py RUNDIR K [K ...]
"""
import json
import sys
import time

import numpy as np
import torch
from ase.io import read
from metatomic.torch import ModelEvaluationOptions, ModelOutput, load_atomistic_model, systems_to_torch
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

run = sys.argv[1]
dev = torch.device("cuda")
frames = read("/work/cosmo/boittier/lyra/plato/study/data/test.xyz", ":")
cc = [a for a in frames if a.info["label"] == "CC"][0]
m = load_atomistic_model(f"{run}/model.pt").to(dev)
nl = m.requested_neighbor_lists()
opts = ModelEvaluationOptions(length_unit="angstrom", outputs={"energy": ModelOutput(sample_kind="system")})
for K in map(int, sys.argv[2:]):
    atoms = cc.copy().repeat((K, K, K))
    s = [get_system_with_neighbor_lists(systems_to_torch(atoms, dtype=torch.float32, device=dev, positions_requires_grad=True), nl)]
    res = {"run": run.rstrip("/").split("/")[-1], "atoms": len(atoms), "gpu": torch.cuda.get_device_name()}
    try:
        def ef():
            E = m(s, opts, check_consistency=False)["energy"].block().values.sum()
            return E, torch.autograd.grad(E, [s[0].positions])
        for _ in range(3):
            ef()
        ts = []
        n = 30 if len(atoms) < 1000 else 10
        torch.cuda.reset_peak_memory_stats()
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            E, _ = ef()
            torch.cuda.synchronize()
            ts.append(1e3 * (time.perf_counter() - t))
        res.update(ms=float(np.median(ts)), ms_min=float(min(ts)), energy=float(E), peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    print(json.dumps(res), flush=True)
