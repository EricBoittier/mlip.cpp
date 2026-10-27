"""PET attention backends (fused mlip.cpp op vs SDPA vs manual), float32, CUDA.

Model: PET-S trained on bio_dimers (d_pet 64, 4 heads of 16, 2 GNN x 2 attention
layers, cutoff 5 A), the weights of ~/bench/models/pet-s.pt loaded into eager PET
models that differ only in attention_backend.

Cases (neighbour lists built beforehand, excluded):
- infer: energy + forces (one backward), model.eval(); "auto" is SDPA here;
- train: energy + forces loss, backward through the forces (double backward),
  Adam step, model.train(); "auto" is the manual attention here, SDPA cannot.
Systems: bio_dimers test frames (12-25 atoms, isolated in a 30 A box) at batch
1 / 8 / 64, and a periodic 160-atom rattled diamond cell (2 x 2 x 5 cubic cells).
Median wall time over ITERS calls after WARMUP calls (CUDA-synchronised); peak
CUDA memory of one call above the memory held before it.

    python bench_pet_fused.py OUT.json
"""

import json
import os
import sys
import time

import ase.build
import numpy as np
import torch
from ase.io import read
from metatomic.torch import ModelOutput, load_atomistic_model, systems_to_torch

from metatrain.pet import PET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

DATA = os.environ.get("PET_FUSED_DATA", "/home/boittier/bench")
WARMUP, ITERS = 3, 10
dev = "cuda"
MODEL = os.path.join(DATA, "pet-s.pt" if os.path.exists(os.path.join(DATA, "pet-s.pt")) else "models/pet-s.pt")
HYPERS = dict(
    cutoff=5.0, cutoff_width=0.5, cutoff_function="Bump", d_pet=64, d_node=128,
    d_head=64, d_feedforward=128, num_heads=4, num_gnn_layers=2,
    num_attention_layers=2, normalization="RMSNorm", activation="SwiGLU",
    attention_temperature=1.0, transformer_type="PreLN", featurizer_type="feedforward",
)  # fmt: skip


def pet(backend, state):
    from metatrain.utils.architectures import get_default_hypers

    hypers = get_default_hypers("pet")["model"]
    hypers.update(HYPERS, attention_backend=backend)
    info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})},
    )
    m = PET(hypers, info)
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    # composition / scaler keep their weights in TensorMaps, filled from buffers
    m.additive_models[0].sync_tensor_maps()
    m.scaler.sync_tensor_maps()
    return m.to(dev)


frames = read(os.path.join(DATA, "test.xyz"), ":")
rng = np.random.default_rng(0)
cell = ase.build.bulk("C", "diamond", a=3.567, cubic=True) * (2, 2, 5)
cell.positions += 0.1 * rng.standard_normal(cell.positions.shape)
cell.numbers[rng.choice(len(cell), 40, replace=False)] = 1
SYSTEMS = {
    "dimers_b1": [frames[0]],
    "dimers_b8": [frames[i] for i in rng.choice(len(frames), 8, replace=False)],
    "dimers_b64": [frames[i] for i in rng.choice(len(frames), 64, replace=False)],
    "cell160_pbc": [cell],
}


def prepare(atoms, model):
    systems = systems_to_torch(atoms, dtype=torch.float32, device=dev, positions_requires_grad=True)
    return [get_system_with_neighbor_lists(s, model.requested_neighbor_lists()) for s in systems]


def make_step(model, systems, mode):
    outputs = {"energy": ModelOutput(sample_kind="system")}
    pos = [s.positions for s in systems]
    opt = torch.optim.Adam(model.parameters(), lr=1e-6)
    if mode == "infer":
        model.eval()

        def step():
            e = model(systems, outputs)["energy"].block().values.sum()
            f = torch.autograd.grad(e, pos)
            return e, f
    else:
        model.train()

        def step():
            opt.zero_grad(set_to_none=True)
            e = model(systems, outputs)["energy"].block().values
            f = torch.autograd.grad(e.sum(), pos, create_graph=True)
            loss = (e**2).mean() + sum((g**2).sum() for g in f)
            loss.backward()
            opt.step()
            return loss, f
    return step


def bench(step):
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return float(np.median(times)), float(peak)


def main():
    state = load_atomistic_model(MODEL).module.state_dict()
    rows = []
    only = os.environ.get("BENCH_ONLY")  # e.g. "train:cell160_pbc:fused"
    for mode in ("infer", "train"):
        for name, atoms in SYSTEMS.items():
            for backend in ("fused", "auto", "manual"):
                if only and f"{mode}:{name}:{backend}" not in only.split(","):
                    continue
                if mode == "train" and backend == "auto":
                    continue  # == manual
                model = pet(backend, state)
                systems = prepare(atoms, model)
                label = {"auto": "sdpa"}.get(backend, backend)
                try:
                    t, mem = bench(make_step(model, systems, mode))
                    r = dict(mode=mode, system=name, backend=label, ms=1e3 * t, peak_mib=mem / 2**20)
                except torch.cuda.OutOfMemoryError as e:
                    r = dict(mode=mode, system=name, backend=label, error=str(e).splitlines()[0])
                r.update(atoms=sum(len(a) for a in atoms), structures=len(atoms))
                print(json.dumps(r), flush=True)
                rows.append(r)
                del model, systems
                torch.cuda.empty_cache()
    meta = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, host=os.uname().nodename, arch=os.uname().machine,
        dbl_naive=os.environ.get("MLIPCPP_ATTN_DBL_NAIVE", "0"), dtype="float32",
        model=MODEL, hypers=HYPERS, warmup=WARMUP, iters=ITERS, doc=__doc__,
    )  # fmt: skip
    json.dump(dict(meta=meta, rows=rows), open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
