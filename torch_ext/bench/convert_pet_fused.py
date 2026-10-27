"""Convert the trained PET-S (bio_dimers) export to attention_backend "fused",
export it with the mlip.cpp op library as a metatomic extension, reload it in a
fresh process and compare energies / forces with the original export (SDPA).

    python convert_pet_fused.py OUT_DIR
"""

import json
import os
import subprocess
import sys

import ase.build
import numpy as np
import torch
from ase.io import read
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    load_atomistic_model,
    systems_to_torch,
)

from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

DATA = os.environ.get("PET_FUSED_DATA", "/home/boittier/bench")
ORIG = os.path.join(DATA, "pet-s.pt" if os.path.exists(os.path.join(DATA, "pet-s.pt")) else "models/pet-s.pt")


def systems(model, dtype=torch.float32):
    frames = read(os.path.join(DATA, "test.xyz"), ":")
    rng = np.random.default_rng(0)
    cell = ase.build.bulk("C", "diamond", a=3.567, cubic=True) * (2, 2, 5)
    cell.positions += 0.1 * rng.standard_normal(cell.positions.shape)
    cell.numbers[rng.choice(len(cell), 40, replace=False)] = 1
    atoms = [frames[i] for i in rng.choice(len(frames), 64, replace=False)] + [cell]
    s = systems_to_torch(atoms, dtype=dtype, device="cuda", positions_requires_grad=True)
    return [get_system_with_neighbor_lists(x, model.requested_neighbor_lists()) for x in s]


def evaluate(path, ext=None):
    model = load_atomistic_model(path, extensions_directory=ext).to("cuda")
    sys_ = systems(model)
    opts = ModelEvaluationOptions(
        length_unit="Angstrom", outputs={"energy": ModelOutput(sample_kind="system")}
    )
    e = model(sys_, opts, check_consistency=False)["energy"].block().values.reshape(-1)
    f = torch.autograd.grad(e.sum(), [s.positions for s in sys_])
    return e.detach().cpu(), torch.cat([-g for g in f]).cpu()


if __name__ == "__main__" and sys.argv[1] == "child":
    e, f = evaluate(sys.argv[2], sys.argv[3] if len(sys.argv) > 4 else None)
    torch.save((e, f), sys.argv[-1])
    sys.exit(0)

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_pet_fused import pet  # noqa: E402

state = load_atomistic_model(ORIG).module.state_dict()
model = pet("fused", state).to("cpu")
assert all(
    l.attention.use_fused_attention for g in model.backend.gnn_layers for l in g.trans.layers
)
path = os.path.join(out, "pet-s-fused.pt")
# the op library lives outside site-packages, where metatomic's collect_extensions
# refuses to copy it: the export records its absolute path instead
model.export().save(path)


def child(*args):
    res = os.path.join(out, f"res-{os.path.basename(args[0])}")
    env = {k: v for k, v in os.environ.items() if k not in ("MLIPCPP_TORCH_LIB",)}
    subprocess.run([sys.executable, __file__, "child", *args, res], check=True, env=env)
    return torch.load(res)


e0, f0 = child(ORIG)  # original export: SDPA
e1, f1 = child(path)  # fused export, op loaded by metatomic from the recorded path
rows = dict(
    gpu=torch.cuda.get_device_name(), torch=torch.__version__, host=os.uname().nodename,
    systems="64 bio_dimers test frames + 160-atom periodic diamond cell, float32",
    energy_max_abs=(e1 - e0).abs().max().item(),
    energy_max_rel=((e1 - e0).abs() / e0.abs().clamp_min(1e-12)).max().item(),
    forces_max_abs=(f1 - f0).abs().max().item(),
    forces_max_rel_to_max=((f1 - f0).abs().max() / f0.abs().max()).item(),
    energy_scale=e0.abs().max().item(),
    forces_scale=f0.abs().max().item(),
)
print(json.dumps(rows, indent=1))
json.dump(rows, open(os.path.join(out, "convert_pet_s.json"), "w"), indent=1)
