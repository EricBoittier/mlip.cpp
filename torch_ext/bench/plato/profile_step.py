"""CUDA time of one Plato E + F train step (32 dimers) and one batch-64 E + F call
per backend, by kernel group: attention (softmax / bmm of the scores / mlipcpp),
everything else; plus the GPU-busy fraction of the wall time.

    python profile_step.py CKPT OUT.json backend ...
"""
import copy
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402
from metatrain.utils.evaluate_model import evaluate_model  # noqa: E402

ckpt, out, backends = sys.argv[1], sys.argv[2], sys.argv[3:]
dev, dtype = torch.device("cuda"), torch.float32
base = common.load(ckpt, dtype, dev)
res = {"gpu": torch.cuda.get_device_name()}
for be in backends:
    m = copy.deepcopy(base)
    m.set_attention_backend(be)
    tinfo = {"energy": m.dataset_info.targets["energy"]}
    tr = common.make_systems(m, common.train_batch(32, 0), dtype, dev)
    ev = common.make_systems(m, common.batch64(), dtype, dev)

    def train():
        m.train()
        m.zero_grad(set_to_none=True)
        b = evaluate_model(m, tr, tinfo, is_training=True)["energy"].block()
        ((b.values**2).mean() + (b.gradient("positions").values ** 2).mean()).backward()

    def infer():
        m.eval()
        common.energy_forces(m, ev)

    for name, fn in (("train32", train), ("batch64", infer)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            wall = 1e3 * (time.perf_counter() - t)
        groups, total, n = {}, 0.0, 0
        for e in p.key_averages():
            ms = e.device_time_total / 1e3 if hasattr(e, "device_time_total") else e.cuda_time_total / 1e3
            if e.device_type != torch.autograd.DeviceType.CUDA or ms == 0:
                continue
            k = e.key.lower()
            g = ("mlipcpp" if any(s in k for s in ("dbl_bwd", "jvp_", "v1_", "v2_", "bwd_bias")) else
                 "softmax" if "softmax" in k else
                 "gemm" if any(s in k for s in ("gemm", "sgemm", "cutlass", "matmul", "xmma")) else "other")
            groups[g] = groups.get(g, 0.0) + ms
            total += ms
            n += e.count
        top = sorted(((e.key[:80], e.device_time_total / 1e3) for e in p.key_averages()
                      if e.device_type == torch.autograd.DeviceType.CUDA), key=lambda x: -x[1])[:12]
        res[f"{be}/{name}"] = {"wall_ms": wall, "gpu_kernel_ms": total, "kernels": n, "groups_ms": groups, "top": top}
        print(be, name, f"wall {wall:.1f} ms, kernels {total:.1f} ms ({n})", {k: round(v, 2) for k, v in groups.items()}, flush=True)
json.dump(res, open(out, "w"), indent=1)
