"""Correctness of torch.ops.mlipcpp.pet_attention: forward, gradients and
second derivatives against manual softmax attention (float64), gradcheck /
gradgradcheck on the float64 kernels, the float32 (fast v2) kernels against
the float64 reference, masked keys and padded sequences, and the
force-training pattern on a toy model."""

import math
import os
import sys

import torch

import mlipcpp_torch as mt

mt.load()
op = torch.ops.mlipcpp.pet_attention
dev = "cuda"
torch.manual_seed(0)
results = []


def report(name, err, tol):
    ok = err <= tol
    results.append((name, err, tol, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:60s} {err:.3e}  (tol {tol:.0e})", flush=True)


def make(N, L, H, d, dtype, n_valid=None, mask_mode="ninf"):
    """qkv, a learnable bias and a constant mask; n_valid[n] = tokens of
    sequence n (the rest is padding: keys masked, queries kept, like PET)."""
    qkv = torch.randn(N, L, 3 * H * d, dtype=dtype, device=dev)
    bias = 0.5 * torch.randn(N, L, L, dtype=dtype, device=dev)
    mask = torch.zeros(N, L, L, dtype=dtype, device=dev)
    if n_valid is not None:
        for n, nv in enumerate(n_valid):
            val = -math.inf if mask_mode == "ninf" else -1e4
            mask[n, :, nv:] = val
    return qkv, bias, mask


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def derivs(fn, qkv, bias, mask, H, scale, u, ul, wq, wb):
    """o, lse; first derivatives along (u, ul); second derivatives of
    <(wq, wb), J^T (u, ul)> w.r.t. (qkv, bias, u, ul)."""
    qkv = qkv.detach().requires_grad_()
    bias = bias.detach().requires_grad_()
    u = u.detach().requires_grad_()
    ul = ul.detach().requires_grad_()
    o, lse = fn(qkv, bias + mask, H, scale)
    s1 = (o * u).sum() + (lse * ul).sum()
    gq, gb = torch.autograd.grad(s1, (qkv, bias), create_graph=True)
    s2 = (gq * wq).sum() + (gb * wb).sum()
    hh = torch.autograd.grad(s2, (qkv, bias, u, ul))
    return [o, lse, gq, gb, *hh]


NAMES = ["o", "lse", "dqkv", "dbias", "d2/dqkv", "d2/dbias", "d2/du", "d2/dulse"]


def compare(tag, N, L, H, d, n_valid=None, mask_mode="ninf", tol32=2e-4, v1=False, naive=False, dbl="auto",
            first_key_masked=False, bias_path="auto"):
    scale = 1.0 / math.sqrt(d)
    qkv, bias, mask = make(N, L, H, d, torch.float64, n_valid, mask_mode)
    if first_key_masked:
        mask[:, :, 0] = -math.inf
    u = torch.randn(N, L, H * d, dtype=torch.float64, device=dev)
    ul = torch.randn(N, L, H, dtype=torch.float64, device=dev)
    wq = torch.randn_like(qkv)
    wb = torch.randn_like(bias)
    if n_valid is not None:  # padded queries: no loss on them (as with PET padding)
        for n, nv in enumerate(n_valid):
            u[n, nv:] = 0
            ul[n, nv:] = 0
    ref = derivs(mt.reference_attention, qkv, bias, mask, H, scale, u, ul, wq, wb)
    os.environ["MLIPCPP_ATTN_V1"] = "1" if v1 else "0"
    os.environ["MLIPCPP_ATTN_DBL_NAIVE"] = "1" if naive else "0"
    os.environ["MLIPCPP_ATTN_DBL"] = dbl
    # seq and dq are float32 (v2) paths: the float64 run keeps the default
    os.environ["MLIPCPP_ATTN_BIAS"] = "auto" if bias_path in ("seq", "dq") else bias_path
    if naive:
        tag += " (ggml JVP kernels)"
    elif dbl != "auto":
        tag += f" ({dbl})"
    got64 = derivs(op, qkv, bias, mask, H, scale, u, ul, wq, wb)
    c = lambda t: t.float()
    os.environ["MLIPCPP_ATTN_BIAS"] = bias_path
    got32 = derivs(op, c(qkv), c(bias), c(mask), H, scale, c(u), c(ul), c(wq), c(wb))
    os.environ["MLIPCPP_ATTN_V1"] = "0"
    os.environ["MLIPCPP_ATTN_DBL_NAIVE"] = "0"
    os.environ["MLIPCPP_ATTN_DBL"] = "auto"
    os.environ["MLIPCPP_ATTN_BIAS"] = "auto"
    for name, r, g64, g32 in zip(NAMES, ref, got64, got32):
        assert torch.isfinite(g32).all() and torch.isfinite(g64).all(), (tag, name)
        report(f"{tag} f64 {name}", rel(g64, r), 1e-11)
        report(f"{tag} f32{' v1' if v1 else ''} {name}", rel(g32.double(), r), tol32)


def gradchecks(N, L, H, d, n_valid=None, dbl="auto", fast=False):
    scale = 1.0 / math.sqrt(d)
    qkv, bias, mask = make(N, L, H, d, torch.float64, n_valid)
    qkv.requires_grad_()
    bias.requires_grad_()

    def f(qkv, bias):
        return op(qkv, bias + mask, H, scale)

    tag = f"N{N} L{L} H{H} d{d}" + (" padded" if n_valid else "") + (f" ({dbl})" if dbl != "auto" else "")
    tag += " fast_mode" if fast else ""
    os.environ["MLIPCPP_ATTN_DBL"] = dbl
    kw = dict(eps=1e-6, atol=1e-7, rtol=1e-5, fast_mode=fast)
    ok1 = torch.autograd.gradcheck(f, (qkv, bias), **kw)
    ok2 = torch.autograd.gradgradcheck(f, (qkv, bias), **kw)
    os.environ["MLIPCPP_ATTN_DBL"] = "auto"
    report(f"gradcheck {tag}", 0.0 if ok1 else 1.0, 0.0)
    report(f"gradgradcheck {tag}", 0.0 if ok2 else 1.0, 0.0)


# --- gradcheck / gradgradcheck (float64 kernels: v1 + JVP kernels) ---
gradchecks(2, 5, 2, 8)
gradchecks(3, 7, 2, 8, n_valid=[7, 4, 2])
gradchecks(2, 4, 1, 16)
os.environ["MLIPCPP_ATTN_DBL_NAIVE"] = "1"
gradchecks(3, 7, 2, 8, n_valid=[7, 4, 2])  # with the ggml JVP kernels
os.environ["MLIPCPP_ATTN_DBL_NAIVE"] = "0"

# --- float64 and float32 kernels against the float64 reference ---
compare("dense L5 H2 d8", 3, 5, 2, 8)
compare("dense L33 H4 d16", 8, 33, 4, 16)
compare("dense L40 H8 d32", 16, 40, 8, 32)
compare("dense L40 H8 d32", 16, 40, 8, 32, v1=True)
compare("dense L24 H2 d64", 8, 24, 2, 64)
compare("padded -inf L37 H4 d32", 6, 37, 4, 32, n_valid=[37, 20, 9, 1, 30, 2])
compare("padded -1e4 L37 H4 d32", 6, 37, 4, 32, n_valid=[37, 20, 9, 1, 30, 2], mask_mode="neg")
compare("long L70 H4 d16 (bias rows kernel)", 4, 70, 4, 16)
compare("long L120 H4 d32 (no fused dbl bwd)", 2, 120, 4, 32)
compare("dense L40 H8 d32", 16, 40, 8, 32, naive=True)
compare("padded -inf L37 H4 d32", 6, 37, 4, 32, n_valid=[37, 20, 9, 1, 30, 2], naive=True)

# --- tiled double backward (any L): gradgradcheck at long sequences ---
# float64 fused kernel: fits up to L ~ 50 (99 KiB) / ~ 75 (227 KiB) with d = 8,
# so L >= 97 runs tiled in auto mode; fast_mode (random projections of the
# Jacobians) above L = 40, where the full numerical Jacobian of the L x L bias
# would take too long
gradchecks(2, 40, 2, 8, n_valid=[40, 23], dbl="tiled")
gradchecks(2, 40, 2, 8, n_valid=[40, 23], dbl="fused")
gradchecks(2, 97, 2, 8, n_valid=[97, 61], fast=True)
gradchecks(1, 160, 2, 8, fast=True)
gradchecks(2, 300, 1, 8, n_valid=[300, 211], fast=True)

# --- tiled double backward against the float64 reference ---
for d_ in (8, 16, 32, 64):
    H_ = 4 if d_ < 64 else 2
    compare(f"tiled L97 H{H_} d{d_}", 3, 97, H_, d_, dbl="tiled")
compare("tiled L40 H8 d16", 8, 40, 8, 16, dbl="tiled")
compare("tiled L160 H8 d16 padded -inf", 4, 160, 8, 16, n_valid=[160, 97, 33, 1], dbl="tiled")
compare("tiled L160 H8 d16 padded -1e4", 4, 160, 8, 16, n_valid=[160, 97, 33, 1], mask_mode="neg", dbl="tiled")
compare("tiled L300 H4 d16 padded -inf", 2, 300, 4, 16, n_valid=[300, 150], dbl="tiled")
compare("tiled L97 H4 d16 first key -inf", 3, 97, 4, 16, first_key_masked=True, dbl="tiled")
compare("fused L33 H4 d16 first key -inf", 3, 33, 4, 16, first_key_masked=True, dbl="fused")
compare("tiled L33 H4 d16 first key -inf", 3, 33, 4, 16, first_key_masked=True, dbl="tiled")
compare("auto L256 H8 d16 padded -inf", 2, 256, 8, 16, n_valid=[256, 130])
# first-backward bias gradient paths at long L (tiled is the default above L = 64)
compare("bias rows L97 H4 d16", 3, 97, 4, 16, bias_path="rows")
compare("bias tiled L50 H4 d16", 3, 50, 4, 16, bias_path="tiled")
compare("bias tiled L130 H2 d64 padded", 3, 130, 2, 64, n_valid=[130, 70, 3], bias_path="tiled")
compare("bias seq L40 H8 d16", 4, 40, 8, 16, bias_path="seq")
compare("bias dq L40 H8 d16 padded", 4, 40, 8, 16, n_valid=[40, 30, 2, 1], bias_path="dq")
compare("bias dq L200 H32 d8", 2, 200, 32, 8, bias_path="dq")
compare("auto L70 H3 d16 (tiled bias, 3 heads)", 3, 70, 3, 16)
compare("auto L23 H1 d32", 5, 23, 1, 32)

# --- tiled vs fused double backward where both run (float32 and float64) ---
def dbl_outputs(dtype, mode, N, L, H, d, n_valid):
    torch.manual_seed(5)
    qkv, bias, mask = make(N, L, H, d, dtype, n_valid)
    go = torch.randn(N, L, H * d, dtype=dtype, device=dev)
    gl = torch.randn(N, L, H, dtype=dtype, device=dev)
    tq, tb = torch.randn_like(qkv), torch.randn_like(bias)
    os.environ["MLIPCPP_ATTN_DBL"] = mode
    r = torch.ops.mlipcpp.pet_attention_double_backward(qkv, bias + mask, go, gl, tq, tb, H, 1 / math.sqrt(d))
    os.environ["MLIPCPP_ATTN_DBL"] = "auto"
    return r


for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 2e-5)):
    for N_, L_, H_, d_, nv in ((8, 24, 4, 16, None), (6, 33, 4, 32, [33, 20, 9, 1, 30, 2]), (4, 48, 8, 16, None),
                               (4, 40, 2, 64, None)):
        if dtype == torch.float64 and (L_ > 40 or d_ > 32):
            continue  # float64 fused kernel does not fit on 99 KiB GPUs
        a = dbl_outputs(dtype, "fused", N_, L_, H_, d_, nv)
        b = dbl_outputs(dtype, "tiled", N_, L_, H_, d_, nv)
        for name, x, y in zip(["to", "tlse", "gqkv", "gbias"], a, b):
            report(f"tiled vs fused {str(dtype)[6:]} L{L_} H{H_} d{d_} {name}", rel(y.double(), x.double()), tol)

# --- the mask at -inf on the FIRST key (online softmax starts at -inf) ---
qkv, bias, mask = make(2, 6, 2, 8, torch.float32)
mask[:, :, 0] = -math.inf
o, _ = op(qkv, bias + mask, 2, 0.3)
o_ref, _ = mt.reference_attention(qkv.double(), (bias + mask).double(), 2, 0.3)
report("first key masked (-inf) f32 forward", rel(o.double(), o_ref), 1e-5)


# --- force-training pattern on a toy PET-like model ---
class Toy(torch.nn.Module):
    def __init__(self, attn, H=4, d=16, L=12):
        super().__init__()
        D = H * d
        self.attn, self.H, self.d = attn, H, d
        self.emb = torch.nn.Linear(4, D)
        self.qkv = torch.nn.Linear(D, 3 * D)
        self.out = torch.nn.Sequential(torch.nn.Linear(D, D), torch.nn.SiLU(), torch.nn.Linear(D, 1))

    def forward(self, r):  # r [N, L, 3] neighbour vectors of N centres
        dist = r.norm(dim=-1, keepdim=True)
        x = self.emb(torch.cat([r, dist], -1))
        fc = 0.5 * (torch.cos(math.pi * (dist / 5.0).clamp(max=1.0)) + 1)  # [N, L, 1]
        bias = torch.log(fc.clamp_min(1e-15)).transpose(1, 2).expand(-1, r.shape[1], -1)
        o, _ = self.attn(self.qkv(x), bias.contiguous(), self.H, 1.0 / math.sqrt(self.d))
        return self.out(o).sum()


def force_train_grads(attn, dtype):
    torch.manual_seed(1)
    m = Toy(attn).to(dev, dtype)
    torch.manual_seed(2)
    r = (2.0 * torch.randn(10, 12, 3, device=dev)).to(dtype).requires_grad_()
    e = m(r)
    (f,) = torch.autograd.grad(e, r, create_graph=True)
    loss = (f**2).sum() + e
    loss.backward()
    return [e.detach(), f.detach()] + [p.grad.detach() for p in m.parameters()]


ref = force_train_grads(mt.reference_attention, torch.float64)
g64 = force_train_grads(op, torch.float64)
g32 = force_train_grads(op, torch.float32)
for i, (a, b, c) in enumerate(zip(ref, g64, g32)):
    name = ["energy", "forces"][i] if i < 2 else f"param grad {i - 2}"
    report(f"force training f64 {name}", rel(b, a), 1e-10)
    report(f"force training f32 {name}", rel(c.double(), a), 1e-3)

# --- third derivatives raise ---
try:
    qkv, bias, _ = make(2, 5, 2, 8, torch.float64)
    qkv.requires_grad_()
    o, _ = op(qkv, bias, 2, 0.3)
    (g,) = torch.autograd.grad(o.sum(), qkv, create_graph=True)
    (g2,) = torch.autograd.grad((g**2).sum(), qkv, create_graph=True)
    g2.sum().backward()
    report("third derivative raises", 1.0, 0.0)
except RuntimeError as e:
    report("third derivative raises", 0.0 if "third derivatives" in str(e) else 1.0, 0.0)

# --- torch.compile traces around the op (Meta kernels), fwd + bwd ---
try:
    qkv, bias, _ = make(4, 9, 2, 16, torch.float32)
    qkv.requires_grad_()

    def g(qkv, bias):
        o, lse = op(qkv * 1.5, bias, 2, 0.25)
        return (o.sin().sum() + lse.sum())

    gc = torch.compile(g, fullgraph=True)
    (a,) = torch.autograd.grad(gc(qkv, bias), qkv)
    (b,) = torch.autograd.grad(g(qkv, bias), qkv)
    report("torch.compile(fullgraph) fwd+bwd vs eager", rel(a, b), 1e-6)
except Exception as e:  # noqa: BLE001
    print("torch.compile failed:", type(e).__name__, str(e)[:2000])
    report("torch.compile(fullgraph) fwd+bwd vs eager", 1.0, 0.0)

n_fail = sum(not r[3] for r in results)
print(f"\n{len(results) - n_fail}/{len(results)} passed")
sys.exit(1 if n_fail else 0)
