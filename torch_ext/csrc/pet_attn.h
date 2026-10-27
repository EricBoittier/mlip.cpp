// Raw-pointer entry points of mlip.cpp's fused PET attention CUDA kernels
// (ported from the patched ggml, src/ggml-cuda/pet-attn.cu; see
// cmake/patches/ggml-pet-attention*.patch). Layouts (all contiguous):
//   qkv  [N, L, 3, D]   D = H*d, head h in channels h*d .. h*d + d - 1
//   bias [N, L, L]      bias[n][query][key], shared by the heads
//   o    [N, L, D]      lse [N, L, H]
// S = scale K^T Q + bias, P = softmax_keys(S), o = V P per head and sequence.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace mlipcpp_attn {

// variant: 0 = default (float: register-blocked v2 kernels, double: v1),
//          1 = the one-row-per-thread v1 kernels
template <typename T>
void forward(const T *qkv, const T *bias, T *o, T *lse, int64_t N, int L, int H,
             int d, double scale, cudaStream_t stream, int variant);

// gradients of (o, lse) given (go, glse); glse may be null (zero). dbias may
// be null (not computed). dqkv [N, L, 3, D], dbias [N, L, L] (sum over heads)
template <typename T>
void backward(const T *qkv, const T *bias, const T *o, const T *lse,
              const T *go, const T *glse, T *dqkv, T *dbias, int64_t N, int L,
              int H, int d, double scale, cudaStream_t stream, int variant);

// forward-mode derivative along (tqkv, tbias), either may be null (zero):
// to [N, L, D], tlse [N, L, H]
template <typename T>
void jvp(const T *qkv, const T *bias, const T *tqkv, const T *tbias, T *to,
         T *tlse, int64_t N, int L, int H, int d, double scale,
         cudaStream_t stream);

// gradients of the JVP's result (to, tlse) given (go, glse; glse may be null)
// w.r.t. qkv and tqkv: gx [N, L, 6, D] slots q, k, v, tq, tk, tv; and w.r.t.
// bias and tbias: gb [N, 2, L, L]. `stats` holds jvp_stats_bytes(N*H*L).
template <typename T>
void jvp_backward(const T *qkv, const T *bias, const T *go, const T *glse,
                  const T *tqkv, const T *tbias, void *stats, T *gx, T *gb,
                  int64_t N, int L, int H, int d, double scale,
                  cudaStream_t stream);

// fused double backward, one block per sequence (needs the sequence's tiles
// and four L x L maps in shared memory: double_backward_fused_ok): with
// u = (go, glse) and w = (tqkv, tbias), to/tlse = J w and
// gx [N, L, 3, D], gb [N, L, L] = d/d(qkv, bias) <u, J w>
template <typename T>
void double_backward(const T *qkv, const T *bias, const T *go, const T *glse,
                     const T *tqkv, const T *tbias, T *to, T *tlse, T *gx,
                     T *gb, int64_t N, int L, int H, int d, double scale,
                     cudaStream_t stream);
template <typename T> bool double_backward_fused_ok(int L, int d);

// tiled double backward (any L, shared memory independent of L): the same
// outputs as double_backward; `stats` holds dbl_tiled_stats_bytes(N*L*H) of
// per-row statistics (written by the first kernel, read by the second)
template <typename T>
void double_backward_tiled(const T *qkv, const T *bias, const T *go, const T *glse, const T *tqkv,
                           const T *tbias, void *stats, T *to, T *tlse, T *gx, T *gb, int64_t N, int L, int H,
                           int d, double scale, cudaStream_t stream);
size_t dbl_tiled_stats_bytes(int64_t rows, bool dbl);

// which double backward to run for (L, d): the fused per-sequence kernel where
// it fits (and L <= its measured break-even), else the tiled kernels.
// MLIPCPP_ATTN_DBL = auto | fused | tiled | jvp forces one (jvp: the ggml JVP
// kernels, debug only; MLIPCPP_ATTN_DBL_NAIVE=1 is the old spelling of jvp).
enum { DBL_FUSED = 0, DBL_TILED = 1, DBL_JVP = 2 };
template <typename T> int double_backward_path(int L, int d);

size_t jvp_stats_bytes(int64_t rows, bool dbl);

bool head_dim_supported(int d);

} // namespace mlipcpp_attn
