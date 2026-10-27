"""Plato-PET-LR attention backends on one GPU (eager, float32): energy + forces
calls and E + F training steps (create_graph forces, Adam), wall time (CUDA-synced
median) and peak CUDA memory.

    python bench_fused.py CKPT OUT.json backend [backend ...]

Systems: one test CC dimer; 8 and 64 test dimers; the CC dimer tiled 2x2x2 (160
atoms, periodic: mesh + Ewald). Training: 8 and 32 random rotated training-set-like
dimers (test.xyz), and the 160-atom cell.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402
from metatrain.experimental.plato_pet_lr.modules.frames import haar_rotations  # noqa: E402
from metatrain.utils.evaluate_model import evaluate_model  # noqa: E402

ckpt, out, backends = sys.argv[1], sys.argv[2], sys.argv[3:]
dev, dtype = torch.device("cuda"), torch.float32
base = common.load(ckpt, dtype, dev)


def timeit(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1e3 * (time.perf_counter() - t))
    return {"ms": float(np.median(ts)), "ms_min": float(min(ts)),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}


def rotated(atoms, seed):
    atoms = [a.copy() for a in atoms]
    R = haar_rotations(len(atoms), torch.float64, generator=torch.Generator().manual_seed(seed)).numpy()
    for a, r in zip(atoms, R):
        a.positions = a.positions @ r.T
        a.cell = np.asarray(a.cell) @ r.T
    return atoms


cc = common.cc()
EVAL = {"dimer": [cc], "batch8": common.batch64()[:8], "batch64": common.batch64(),
        "cell160": [cc.repeat((2, 2, 2))]}
TRAIN = {"train8": 8, "train32": 32, "train_cell160": 0}
res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "ckpt": ckpt,
       "hypers": {k: base.hypers[k] for k in ("frames", "frame_mode", "d_model", "num_heads", "num_layers")}}
for be in backends:
    model = copy.deepcopy(base)
    model.set_attention_backend(be)
    r = res[be] = {"use_fused": model.use_fused}
    model.eval()
    for name, atoms in EVAL.items():
        try:
            systems = common.make_systems(model, atoms, dtype, dev)
            r[name] = timeit(lambda: common.energy_forces(model, systems))
            r[name]["n_atoms"] = sum(len(a) for a in atoms)
        except torch.OutOfMemoryError:
            r[name] = {"error": "OOM"}
        torch.cuda.empty_cache()
        print(be, name, r[name], flush=True)
    model.train()
    tinfo = {"energy": model.dataset_info.targets["energy"]}
    opt = torch.optim.Adam(model.parameters(), lr=1e-7)
    for name, n in TRAIN.items():
        batches = []
        for seed in range(4):
            atoms = rotated(common.train_batch(n, seed) if n else [cc.repeat((2, 2, 2))], seed)
            e, f = common.refs(atoms, dtype, dev) if n else (None, None)
            batches.append((common.make_systems(model, atoms, dtype, dev), e, f))
        it = [0]

        def step():
            systems, e_ref, f_ref = batches[it[0] % len(batches)]
            it[0] += 1
            opt.zero_grad(set_to_none=True)
            b = evaluate_model(model, systems, tinfo, is_training=True)["energy"].block()
            E, G = b.values[:, 0], b.gradient("positions").values[:, :, 0]
            loss = (E**2).mean() + (G**2).mean() if e_ref is None else ((E - e_ref) ** 2).mean() + ((G + f_ref) ** 2).mean()
            loss.backward()
            opt.step()

        try:
            r[name] = timeit(step, n=16, warm=4)
        except torch.OutOfMemoryError:
            r[name] = {"error": "OOM"}
        torch.cuda.empty_cache()
        print(be, name, r[name], flush=True)
    del model, opt
    torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1)
