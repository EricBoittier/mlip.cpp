"""GPU timings: mlipcpp::pet_attention vs manual softmax attention (metatrain
PET/Plato training path) vs torch SDPA, float32.

Modes (inputs qkv [N, L, 3*D] and bias [N, L, L], as PET builds them):
  fwd    forward only, no autograd
  fb     forward + backward of <u, o> to qkv and bias
  force  force-training pattern: g = grad(<u, o>, (qkv, bias), create_graph),
         then backward of <wq, gq> + <wb, gb> to qkv, bias and u (u stands in
         for the downstream network that makes dE/do depend on parameters)
Each method includes the reshapes it needs from the packed qkv layout.
"""

import argparse
import json
import os
import math

import torch
import torch.nn.functional as F

import mlipcpp_torch as mt

mt.load()
dev = "cuda"


def custom(qkv, bias, H, scale):
    return torch.ops.mlipcpp.pet_attention(qkv, bias, H, scale)[0]


def manual(qkv, bias, H, scale):  # metatrain PET manual_attention
    N, L = qkv.shape[:2]
    x = qkv.reshape(N, L, 3, H, -1).permute(2, 0, 3, 1, 4)
    q, k, v = x[0], x[1], x[2]
    a = (q @ k.transpose(-2, -1)) * scale + bias[:, None]
    o = a.softmax(dim=-1) @ v
    return o.transpose(1, 2).reshape(N, L, -1)


def sdpa(qkv, bias, H, scale):
    N, L = qkv.shape[:2]
    x = qkv.reshape(N, L, 3, H, -1).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(x[0], x[1], x[2], attn_mask=bias[:, None], scale=scale)
    return o.transpose(1, 2).reshape(N, L, -1)


def custom_ggml_jvp(qkv, bias, H, scale):  # double backward through ggml's JVP kernels
    os.environ["MLIPCPP_ATTN_DBL_NAIVE"] = "1"
    return custom(qkv, bias, H, scale)


METHODS = {"mlipcpp": custom, "mlipcpp_ggmljvp": custom_ggml_jvp, "manual": manual, "sdpa": sdpa}


def make_step(fn, mode, qkv, bias, u, wq, wb, H, scale):
    if mode == "fwd":
        def step():
            with torch.no_grad():
                fn(qkv, bias, H, scale)
    elif mode == "fb":
        def step():
            o = fn(qkv, bias, H, scale)
            torch.autograd.backward(o, u)
    else:
        def step():
            o = fn(qkv, bias, H, scale)
            gq, gb = torch.autograd.grad((o * u).sum(), (qkv, bias), create_graph=True)
            ((gq * wq).sum() + (gb * wb).sum()).backward()
    return step


def timeit(step, reps, warm=5):
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        step()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def run_case(N, L, H, d, reps):
    D = H * d
    scale = 1.0 / math.sqrt(d)
    torch.manual_seed(0)
    qkv = torch.randn(N, L, 3 * D, device=dev, requires_grad=True)
    # PET-like bias: log of a cutoff, with ~25% padded (masked) neighbours
    fc = torch.rand(N, 1, L, device=dev).clamp_min(1e-3)
    nv = torch.randint(L // 2, L + 1, (N, 1, 1), device=dev)
    fc = torch.where(torch.arange(L, device=dev)[None, None] < nv, fc, torch.full_like(fc, 1e-15))
    bias = torch.log(fc).expand(N, L, L).contiguous().requires_grad_()
    u = torch.randn(N, L, D, device=dev, requires_grad=True)
    wq, wb = torch.randn_like(qkv), torch.randn_like(bias)
    out = []
    for mode in ["fwd", "fb", "force"]:
        for name, fn in METHODS.items():
            if name == "mlipcpp_ggmljvp" and mode != "force":
                continue
            os.environ["MLIPCPP_ATTN_DBL_NAIVE"] = "0"
            rec = dict(N=N, L=L, H=H, d=d, mode=mode, method=name)
            try:
                step = make_step(fn, mode, qkv, bias, u, wq, wb, H, scale)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                step()
                torch.cuda.synchronize()
                rec["peak_MiB"] = (torch.cuda.max_memory_allocated() - base) / 2**20
                rec["ms"] = timeit(step, reps)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
            for t in (qkv, bias, u):
                t.grad = None
            out.append(rec)
            print(json.dumps(rec), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench.json")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    cases = [
        # bio_dimers batch of 32 dimers (~650 atoms), per-centre sequences
        (650, 24, 4, 16), (650, 32, 4, 32), (650, 32, 8, 16), (650, 40, 8, 32),
        # 10k-atom periodic system
        (10000, 32, 4, 32), (10000, 40, 8, 32),
    ]
    res = []
    for c in cases:
        res += run_case(*c, reps=args.reps)
    with open(args.out, "w") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, results=res), f, indent=1)


if __name__ == "__main__":
    main()
