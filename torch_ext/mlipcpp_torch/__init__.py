"""PyTorch bindings of mlip.cpp's fused PET attention (CUDA).

``load()`` JIT-builds (or loads) ``libmlipcpp_torch.so`` and registers
``torch.ops.mlipcpp.pet_attention`` and its helper ops. The library has no
Python dependency: a TorchScript model calling the op only needs
``torch.ops.load_library(lib_path())`` (or metatomic's extension mechanism).
"""

import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(os.path.dirname(_HERE), "csrc")
_lib = None


def load(verbose: bool = False) -> str:
    """Build if needed and load the extension; returns the library path."""
    global _lib
    if _lib is not None:
        return _lib
    prebuilt = os.environ.get("MLIPCPP_TORCH_LIB")
    if prebuilt:
        torch.ops.load_library(prebuilt)
        _lib = prebuilt
        return _lib
    from torch.utils.cpp_extension import load as _load

    build_dir = os.environ.get(
        "MLIPCPP_TORCH_BUILD", os.path.join(os.path.dirname(_HERE), "build")
    )
    os.makedirs(build_dir, exist_ok=True)
    # torch >= 2.14 headers require C++20 (ATen.h #error, std::strong_ordering)
    std = "-std=c++20" if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 14) else "-std=c++17"
    _load(
        name="mlipcpp_torch",
        sources=[
            os.path.join(_CSRC, "pet_attn_op.cpp"),
            os.path.join(_CSRC, "pet_attn_kernels.cu"),
        ],
        extra_cflags=["-O3", std],
        extra_cuda_cflags=["-O3", std, "-lineinfo"],
        build_directory=build_dir,
        is_python_module=False,
        verbose=verbose,
    )
    _lib = os.path.join(build_dir, "mlipcpp_torch.so")
    return _lib


def lib_path() -> str:
    return load()


def pet_attention(qkv: torch.Tensor, bias: torch.Tensor, heads: int, scale: float):
    """Fused attention: qkv [N, L, 3*D] (q | k | v, heads contiguous in D),
    bias [N, L, L] added to the scores of every head. Returns (o [N, L, D],
    lse [N, L, heads]). Differentiable twice (forces with force training)."""
    return torch.ops.mlipcpp.pet_attention(qkv, bias, heads, scale)


def reference_attention(qkv: torch.Tensor, bias: torch.Tensor, heads: int, scale: float):
    """Manual softmax attention with the same conventions (any dtype/device)."""
    N, L = qkv.shape[0], qkv.shape[1]
    x = qkv.reshape(N, L, 3, heads, -1).permute(2, 0, 3, 1, 4)  # [3, N, H, L, d]
    q, k, v = x[0], x[1], x[2]
    s = scale * (q @ k.transpose(-1, -2)) + bias.reshape(N, 1, L, L)
    lse = torch.logsumexp(s, -1)  # [N, H, L]
    o = torch.softmax(s, -1) @ v  # [N, H, L, d]
    return o.transpose(1, 2).reshape(N, L, -1), lse.transpose(1, 2)
