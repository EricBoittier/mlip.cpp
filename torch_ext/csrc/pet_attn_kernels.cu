// mlip.cpp's fused PET attention kernels, ported from the patched ggml
// (src/ggml-cuda/pet-attn.cu of mlip.cpp's ggml, cmake/patches/
// ggml-pet-attention*.patch) to raw pointers. Changes against ggml:
//  - o and lse are separate arrays ([N, L, D], [N, L, H]) instead of one
//    [D + H, L, N] result; the incoming gradients likewise (go, glse);
//  - the backward kernels take the gradient of lse too (glse; ggml ignores
//    it): every use of Di = dO . O becomes Di - glse (dlse/dS = P);
//  - the v1, bias-row and JVP kernels are templated on the scalar type, so
//    float64 runs for gradcheck; v2 and the per-sequence bias kernel are F32;
//  - a key with bias -inf contributes p = 0 in the online softmax (ggml:
//    exp(-inf - -inf) = NaN when it is the first key of a row);
//  - the F16 / bf16 tensor-core variants are not ported.
#include "pet_attn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mlipcpp_attn {

#define PA_CHECK(x)                                                            \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess)                                                     \
      throw std::runtime_error(std::string("mlipcpp pet_attn CUDA error: ") +  \
                               cudaGetErrorString(e_));                        \
  } while (0)

static constexpr int MAX_SMEM = 48 * 1024;

__device__ __forceinline__ float ex(float x) { return expf(x); }
__device__ __forceinline__ double ex(double x) { return exp(x); }
__device__ __forceinline__ float lg(float x) { return logf(x); }
__device__ __forceinline__ double lg(double x) { return log(x); }

// ---------------------------------------------------------------------------
// v1: one thread per (query or key, head); a block holds Tt positions x H heads
// and streams the other side in tiles of Tt rows through shared memory. Rows
// are head-padded to d+1. Probabilities are recomputed from lse in backward.

template <int d, typename T>
__global__ void v1_fwd(const T *__restrict__ qkv, const T *__restrict__ bias,
                       T *__restrict__ out, T *__restrict__ lse, const int H,
                       const int L, const int Tt, const T scale) {
  extern __shared__ __align__(16) unsigned char smraw[];
  T *sm = reinterpret_cast<T *>(smraw);
  const int D = H * d, P = d + 1;
  T *Ks = sm;
  T *Vs = Ks + Tt * H * P;
  T *Bs = Vs + Tt * H * P;

  const int64_t n = blockIdx.x;
  const int q0 = blockIdx.y * Tt, t = threadIdx.x, h = t % H, qi = t / H,
            i = q0 + qi;
  const bool active = i < L;
  const T *base = qkv + n * L * 3 * D;

  T q[d], acc[d], m = -INFINITY, l = 0;
#pragma unroll
  for (int c = 0; c < d; c++)
    q[c] = active ? scale * base[(int64_t)i * 3 * D + h * d + c] : T(0),
    acc[c] = 0;

  for (int k0 = 0; k0 < L; k0 += Tt) {
    const int nk = min(Tt, L - k0);
    for (int e = t; e < nk * D; e += blockDim.x) {
      const int kk = e / D, c = e % D, o = (kk * H + c / d) * P + c % d;
      Ks[o] = base[(int64_t)(k0 + kk) * 3 * D + D + c];
      Vs[o] = base[(int64_t)(k0 + kk) * 3 * D + 2 * D + c];
    }
    for (int e = t; e < Tt * nk; e += blockDim.x) {
      const int a = e / nk, kk = e % nk;
      Bs[a * Tt + kk] = q0 + a < L ? bias[(n * L + q0 + a) * L + k0 + kk] : T(0);
    }
    __syncthreads();
    if (active) {
      for (int kk = 0; kk < nk; kk++) {
        const T *kr = Ks + (kk * H + h) * P;
        T s = Bs[qi * Tt + kk];
#pragma unroll
        for (int c = 0; c < d; c++) s += q[c] * kr[c];
        if (s > m) {
          const T corr = ex(m - s);
          l *= corr;
#pragma unroll
          for (int c = 0; c < d; c++) acc[c] *= corr;
          m = s;
        }
        const T p = s == -INFINITY ? T(0) : ex(s - m);
        l += p;
        const T *vr = Vs + (kk * H + h) * P;
#pragma unroll
        for (int c = 0; c < d; c++) acc[c] += p * vr[c];
      }
    }
    __syncthreads();
  }
  if (active) {
    T *o = out + (n * L + i) * D;
#pragma unroll
    for (int c = 0; c < d; c++) o[h * d + c] = acc[c] / l;
    lse[(n * L + i) * H + h] = m + lg(l);
  }
}

// dq = scale * sum_k p (dp - Di) k, Di = dO . O - glse
template <int d, typename T>
__global__ void v1_bwd_dq(const T *__restrict__ qkv, const T *__restrict__ bias,
                          const T *__restrict__ O, const T *__restrict__ LSE,
                          const T *__restrict__ GO, const T *__restrict__ GL,
                          T *__restrict__ dst, const int H, const int L,
                          const int Tt, const T scale) {
  extern __shared__ __align__(16) unsigned char smraw[];
  T *sm = reinterpret_cast<T *>(smraw);
  const int D = H * d, P = d + 1;
  T *Ks = sm;
  T *Vs = Ks + Tt * H * P;
  T *Bs = Vs + Tt * H * P;

  const int64_t n = blockIdx.x;
  const int q0 = blockIdx.y * Tt, t = threadIdx.x, h = t % H, qi = t / H,
            i = q0 + qi;
  const bool active = i < L;
  const T *base = qkv + n * L * 3 * D;
  const int64_t row = n * L + (active ? i : 0);

  T q[d], dO[d], dq[d], Di = 0;
#pragma unroll
  for (int c = 0; c < d; c++) {
    q[c] = active ? base[(int64_t)i * 3 * D + h * d + c] : T(0);
    dO[c] = active ? GO[row * D + h * d + c] : T(0);
    Di += dO[c] * (active ? O[row * D + h * d + c] : T(0));
    dq[c] = 0;
  }
  if (GL && active) Di -= GL[row * H + h];
  const T lse = active ? LSE[row * H + h] : T(0);

  for (int k0 = 0; k0 < L; k0 += Tt) {
    const int nk = min(Tt, L - k0);
    for (int e = t; e < nk * D; e += blockDim.x) {
      const int kk = e / D, c = e % D, o = (kk * H + c / d) * P + c % d;
      Ks[o] = base[(int64_t)(k0 + kk) * 3 * D + D + c];
      Vs[o] = base[(int64_t)(k0 + kk) * 3 * D + 2 * D + c];
    }
    for (int e = t; e < Tt * nk; e += blockDim.x) {
      const int a = e / nk, kk = e % nk;
      Bs[a * Tt + kk] = q0 + a < L ? bias[(n * L + q0 + a) * L + k0 + kk] : T(0);
    }
    __syncthreads();
    if (active) {
      for (int kk = 0; kk < nk; kk++) {
        const T *kr = Ks + (kk * H + h) * P;
        const T *vr = Vs + (kk * H + h) * P;
        T s = 0, dp = 0;
#pragma unroll
        for (int c = 0; c < d; c++) s += q[c] * kr[c], dp += dO[c] * vr[c];
        const T p = ex(scale * s + Bs[qi * Tt + kk] - lse);
        const T ds = p * (dp - Di);
#pragma unroll
        for (int c = 0; c < d; c++) dq[c] += ds * kr[c];
      }
    }
    __syncthreads();
  }
  if (active) {
#pragma unroll
    for (int c = 0; c < d; c++) dst[row * 3 * D + h * d + c] = scale * dq[c];
  }
}

// dk = scale * sum_q p (dp - Di) q, dv = sum_q p dO
template <int d, typename T>
__global__ void v1_bwd_dkv(const T *__restrict__ qkv, const T *__restrict__ bias,
                           const T *__restrict__ O, const T *__restrict__ LSE,
                           const T *__restrict__ GO, const T *__restrict__ GL,
                           T *__restrict__ dst, const int H, const int L,
                           const int Tt, const T scale) {
  extern __shared__ __align__(16) unsigned char smraw[];
  T *sm = reinterpret_cast<T *>(smraw);
  const int D = H * d, P = d + 1;
  T *Qs = sm;
  T *dOs = Qs + Tt * H * P;
  T *Bs = dOs + Tt * H * P;
  T *Ls = Bs + Tt * Tt;
  T *Ds = Ls + Tt * H;

  const int64_t n = blockIdx.x;
  const int k0 = blockIdx.y * Tt, t = threadIdx.x, h = t % H, kj = t / H,
            j = k0 + kj;
  const bool active = j < L;
  const int nk = min(Tt, L - k0);
  const T *base = qkv + n * L * 3 * D;

  T k[d], v[d], dk[d], dv[d];
#pragma unroll
  for (int c = 0; c < d; c++) {
    k[c] = active ? base[(int64_t)j * 3 * D + D + h * d + c] : T(0);
    v[c] = active ? base[(int64_t)j * 3 * D + 2 * D + h * d + c] : T(0);
    dk[c] = dv[c] = 0;
  }

  for (int q0 = 0; q0 < L; q0 += Tt) {
    const int nq = min(Tt, L - q0);
    for (int e = t; e < nq * D; e += blockDim.x) {
      const int qq = e / D, c = e % D, o = (qq * H + c / d) * P + c % d;
      const int64_t row = n * L + q0 + qq;
      Qs[o] = base[(int64_t)(q0 + qq) * 3 * D + c];
      dOs[o] = GO[row * D + c];
    }
    for (int e = t; e < nq * nk; e += blockDim.x) {
      const int qq = e / nk, kk = e % nk;
      Bs[qq * Tt + kk] = bias[(n * L + q0 + qq) * L + k0 + kk];
    }
    for (int e = t; e < nq * H; e += blockDim.x)
      Ls[e] = LSE[(n * L + q0 + e / H) * H + e % H];
    __syncthreads();
    for (int e = t; e < nq * H; e += blockDim.x) {
      const int qq = e / H, hh = e % H;
      const int64_t row = n * L + q0 + qq;
      const T *o = O + row * D + hh * d;
      T s = 0;
      for (int c = 0; c < d; c++) s += dOs[(qq * H + hh) * P + c] * o[c];
      Ds[e] = GL ? s - GL[row * H + hh] : s;
    }
    __syncthreads();
    if (active) {
      for (int qq = 0; qq < nq; qq++) {
        const T *qr = Qs + (qq * H + h) * P;
        const T *gr = dOs + (qq * H + h) * P;
        T s = 0, dp = 0;
#pragma unroll
        for (int c = 0; c < d; c++) s += qr[c] * k[c], dp += gr[c] * v[c];
        const T p = ex(scale * s + Bs[qq * Tt + kj] - Ls[qq * H + h]);
        const T ds = p * (dp - Ds[qq * H + h]);
#pragma unroll
        for (int c = 0; c < d; c++) dv[c] += p * gr[c], dk[c] += ds * qr[c];
      }
    }
    __syncthreads();
  }
  if (active) {
    T *o = dst + (n * L + j) * 3 * D;
#pragma unroll
    for (int c = 0; c < d; c++)
      o[D + h * d + c] = scale * dk[c], o[2 * D + h * d + c] = dv[c];
  }
}

// ---------------------------------------------------------------------------
// v2 (F32): register-blocked. Each thread owns R rows of one head; shared rows
// padded to d+4 (128-bit loads). See ggml's pet-attn.cu for the full notes.

template <int d>
__device__ __forceinline__ void load_row(const float *__restrict__ s,
                                         float (&r)[d]) {
#pragma unroll
  for (int c = 0; c < d; c += 4) {
    const float4 v = *reinterpret_cast<const float4 *>(s + c);
    r[c] = v.x, r[c + 1] = v.y, r[c + 2] = v.z, r[c + 3] = v.w;
  }
}

// nr rows of D values (row stride `stride`) into shared [nr][H][P]
__device__ __forceinline__ void stage_rows(float *__restrict__ dst,
                                           const float *__restrict__ src,
                                           const int64_t stride, const int nr,
                                           const int D, const int d,
                                           const int P, const int t,
                                           const int nt) {
  const int H = D / d;
  if ((stride & 3) == 0 && (reinterpret_cast<uintptr_t>(src) % 16) == 0) {
    const int D4 = D / 4;
    for (int e = t; e < nr * D4; e += nt) {
      const int r = e / D4, c = (e % D4) * 4;
      *reinterpret_cast<float4 *>(dst + (r * H + c / d) * P + c % d) =
          *reinterpret_cast<const float4 *>(src + r * stride + c);
    }
  } else {
    for (int e = t; e < nr * D; e += nt) {
      const int r = e / D, c = e % D;
      dst[(r * H + c / d) * P + c % d] = src[r * stride + c];
    }
  }
}

template <int d, int R>
__global__ void v2_fwd(const float *__restrict__ qkv,
                       const float *__restrict__ bias, float *__restrict__ out,
                       float *__restrict__ lse, const int H, const int L,
                       const int TP, const int TS, const float scale) {
  extern __shared__ float4 sm4[];
  float *sm = reinterpret_cast<float *>(sm4);
  const int D = H * d, P = d + 4, G = TP / R;
  float *Ks = sm;
  float *Vs = Ks + TS * H * P;
  float *Bs = Vs + TS * H * P;

  const int64_t n = blockIdx.x;
  const int q0 = blockIdx.y * TP, t = threadIdx.x, h = t % H, g = t / H;
  const float *base = qkv + n * L * 3 * D;

  float q[R][d], acc[R][d], m[R], l[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = q0 + g + r * G;
    m[r] = -INFINITY, l[r] = 0.0f;
#pragma unroll
    for (int c = 0; c < d; c++)
      q[r][c] = i < L ? scale * base[(int64_t)i * 3 * D + h * d + c] : 0.0f,
      acc[r][c] = 0.0f;
  }

  for (int k0 = 0; k0 < L; k0 += TS) {
    const int nk = min(TS, L - k0);
    stage_rows(Ks, base + (int64_t)k0 * 3 * D + D, 3 * D, nk, D, d, P, t, blockDim.x);
    stage_rows(Vs, base + (int64_t)k0 * 3 * D + 2 * D, 3 * D, nk, D, d, P, t, blockDim.x);
    for (int e = t; e < TP * nk; e += blockDim.x) {
      const int a = e / nk, kk = e % nk;
      Bs[a * TS + kk] = q0 + a < L ? bias[(n * L + q0 + a) * L + k0 + kk] : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < nk; kk++) {
      float kr[d], vr[d];
      load_row<d>(Ks + (kk * H + h) * P, kr);
      load_row<d>(Vs + (kk * H + h) * P, vr);
#pragma unroll
      for (int r = 0; r < R; r++) {
        float s = Bs[(g + r * G) * TS + kk];
#pragma unroll
        for (int c = 0; c < d; c++) s += q[r][c] * kr[c];
        if (s > m[r]) {
          const float corr = __expf(m[r] - s);
          l[r] *= corr;
#pragma unroll
          for (int c = 0; c < d; c++) acc[r][c] *= corr;
          m[r] = s;
        }
        const float p = s == -INFINITY ? 0.0f : __expf(s - m[r]);
        l[r] += p;
#pragma unroll
        for (int c = 0; c < d; c++) acc[r][c] += p * vr[c];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = q0 + g + r * G;
    if (i >= L) continue;
    float *o = out + (n * L + i) * D;
#pragma unroll
    for (int c = 0; c < d; c++) o[h * d + c] = acc[r][c] / l[r];
    lse[(n * L + i) * H + h] = m[r] + logf(l[r]);
  }
}

template <int d, int R>
__global__ void v2_bwd_dq(const float *__restrict__ qkv,
                          const float *__restrict__ bias,
                          const float *__restrict__ O,
                          const float *__restrict__ LSE,
                          const float *__restrict__ GO,
                          const float *__restrict__ GL, float *__restrict__ dst,
                          const int H, const int L, const int TP, const int TS,
                          const float scale) {
  extern __shared__ float4 sm4[];
  float *sm = reinterpret_cast<float *>(sm4);
  const int D = H * d, P = d + 4, G = TP / R;
  float *Ks = sm;
  float *Vs = Ks + TS * H * P;
  float *Bs = Vs + TS * H * P;

  const int64_t n = blockIdx.x;
  const int q0 = blockIdx.y * TP, t = threadIdx.x, h = t % H, g = t / H;
  const float *base = qkv + n * L * 3 * D;

  float q[R][d], dO[R][d], dq[R][d], Di[R], lse[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = q0 + g + r * G;
    const bool ok = i < L;
    const int64_t row = n * L + (ok ? i : 0);
    Di[r] = 0.0f;
#pragma unroll
    for (int c = 0; c < d; c++) {
      q[r][c] = ok ? base[(int64_t)i * 3 * D + h * d + c] : 0.0f;
      dO[r][c] = ok ? GO[row * D + h * d + c] : 0.0f;
      Di[r] += dO[r][c] * (ok ? O[row * D + h * d + c] : 0.0f);
      dq[r][c] = 0.0f;
    }
    if (GL && ok) Di[r] -= GL[row * H + h];
    lse[r] = ok ? LSE[row * H + h] : 0.0f;
  }

  for (int k0 = 0; k0 < L; k0 += TS) {
    const int nk = min(TS, L - k0);
    stage_rows(Ks, base + (int64_t)k0 * 3 * D + D, 3 * D, nk, D, d, P, t, blockDim.x);
    stage_rows(Vs, base + (int64_t)k0 * 3 * D + 2 * D, 3 * D, nk, D, d, P, t, blockDim.x);
    for (int e = t; e < TP * nk; e += blockDim.x) {
      const int a = e / nk, kk = e % nk;
      Bs[a * TS + kk] = q0 + a < L ? bias[(n * L + q0 + a) * L + k0 + kk] : 0.0f;
    }
    __syncthreads();
    for (int kk = 0; kk < nk; kk++) {
      float kr[d], vr[d];
      load_row<d>(Ks + (kk * H + h) * P, kr);
      load_row<d>(Vs + (kk * H + h) * P, vr);
#pragma unroll
      for (int r = 0; r < R; r++) {
        float s = 0.0f, dp = 0.0f;
#pragma unroll
        for (int c = 0; c < d; c++) s += q[r][c] * kr[c], dp += dO[r][c] * vr[c];
        const float p = __expf(scale * s + Bs[(g + r * G) * TS + kk] - lse[r]);
        const float ds = p * (dp - Di[r]);
#pragma unroll
        for (int c = 0; c < d; c++) dq[r][c] += ds * kr[c];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int i = q0 + g + r * G;
    if (i >= L) continue;
#pragma unroll
    for (int c = 0; c < d; c++)
      dst[(n * L + i) * 3 * D + h * d + c] = scale * dq[r][c];
  }
}

template <int d, int R>
__global__ void v2_bwd_dkv(const float *__restrict__ qkv,
                           const float *__restrict__ bias,
                           const float *__restrict__ O,
                           const float *__restrict__ LSE,
                           const float *__restrict__ GO,
                           const float *__restrict__ GL, float *__restrict__ dst,
                           const int H, const int L, const int TP, const int TS,
                           const float scale) {
  extern __shared__ float4 sm4[];
  float *sm = reinterpret_cast<float *>(sm4);
  const int D = H * d, P = d + 4, G = TP / R;
  float *Qs = sm;
  float *dOs = Qs + TS * H * P;
  float *Bs = dOs + TS * H * P; // [TS queries][TP keys]
  float *Ls = Bs + TS * TP;
  float *Ds = Ls + TS * H;

  const int64_t n = blockIdx.x;
  const int k0 = blockIdx.y * TP, t = threadIdx.x, h = t % H, g = t / H;
  const float *base = qkv + n * L * 3 * D;

  float k[R][d], v[R][d], dk[R][d], dv[R][d];
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int j = k0 + g + r * G;
#pragma unroll
    for (int c = 0; c < d; c++) {
      k[r][c] = j < L ? base[(int64_t)j * 3 * D + D + h * d + c] : 0.0f;
      v[r][c] = j < L ? base[(int64_t)j * 3 * D + 2 * D + h * d + c] : 0.0f;
      dk[r][c] = dv[r][c] = 0.0f;
    }
  }

  for (int q0 = 0; q0 < L; q0 += TS) {
    const int nq = min(TS, L - q0);
    stage_rows(Qs, base + (int64_t)q0 * 3 * D, 3 * D, nq, D, d, P, t, blockDim.x);
    stage_rows(dOs, GO + (n * L + q0) * D, D, nq, D, d, P, t, blockDim.x);
    for (int e = t; e < nq * TP; e += blockDim.x) {
      const int qq = e / TP, a = e % TP;
      Bs[qq * TP + a] = k0 + a < L ? bias[(n * L + q0 + qq) * L + k0 + a] : 0.0f;
    }
    for (int e = t; e < nq * H; e += blockDim.x)
      Ls[e] = LSE[(n * L + q0 + e / H) * H + e % H];
    __syncthreads();
    for (int e = t; e < nq * H; e += blockDim.x) {
      const int qq = e / H, hh = e % H;
      const int64_t row = n * L + q0 + qq;
      const float *o = O + row * D + hh * d;
      float s = 0.0f;
      for (int c = 0; c < d; c++) s += dOs[(qq * H + hh) * P + c] * o[c];
      Ds[e] = GL ? s - GL[row * H + hh] : s;
    }
    __syncthreads();
    for (int qq = 0; qq < nq; qq++) {
      float qr[d], gr[d];
      load_row<d>(Qs + (qq * H + h) * P, qr);
      load_row<d>(dOs + (qq * H + h) * P, gr);
      const float lq = Ls[qq * H + h], dq = Ds[qq * H + h];
#pragma unroll
      for (int r = 0; r < R; r++) {
        float s = 0.0f, dp = 0.0f;
#pragma unroll
        for (int c = 0; c < d; c++) s += qr[c] * k[r][c], dp += gr[c] * v[r][c];
        const float p = __expf(scale * s + Bs[qq * TP + g + r * G] - lq);
        const float ds = p * (dp - dq);
#pragma unroll
        for (int c = 0; c < d; c++) dv[r][c] += p * gr[c], dk[r][c] += ds * qr[c];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int j = k0 + g + r * G;
    if (j >= L) continue;
    float *o = dst + (n * L + j) * 3 * D;
#pragma unroll
    for (int c = 0; c < d; c++)
      o[D + h * d + c] = scale * dk[r][c], o[2 * D + h * d + c] = dv[r][c];
  }
}

// ---------------------------------------------------------------------------
// bias gradient: dbias[n][i][j] = sum_h p_hij (dO_hi . v_hj - Di_hi)

// one block per (query row i, sequence n), threads over the keys j
template <int d, typename T>
__global__ void bwd_bias_rows(const T *__restrict__ qkv, const T *__restrict__ bias,
                              const T *__restrict__ O, const T *__restrict__ LSE,
                              const T *__restrict__ GO, const T *__restrict__ GL,
                              T *__restrict__ dbias, const int H, const int L,
                              const T scale) {
  extern __shared__ __align__(16) unsigned char smraw[];
  T *sm = reinterpret_cast<T *>(smraw);
  const int D = H * d, i = blockIdx.x;
  const int64_t n = blockIdx.y;
  T *q = sm, *dO = sm + D, *lse = sm + 2 * D, *Di = lse + H;
  const T *x = qkv + n * L * 3 * D;
  const int64_t row = n * L + i;
  const T *oi = O + row * D;
  const T *gi = GO + row * D;
  for (int c = threadIdx.x; c < D; c += blockDim.x)
    q[c] = x[(int64_t)i * 3 * D + c], dO[c] = gi[c];
  for (int h = threadIdx.x; h < H; h += blockDim.x) lse[h] = LSE[row * H + h];
  __syncthreads();
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    T s = 0;
#pragma unroll
    for (int c = 0; c < d; c++) s += dO[h * d + c] * oi[h * d + c];
    Di[h] = GL ? s - GL[row * H + h] : s;
  }
  __syncthreads();
  const T *b = bias + row * L;
  T *db = dbias + row * L;
  for (int j = threadIdx.x; j < L; j += blockDim.x) {
    const T *k = x + ((int64_t)j * 3 + 1) * D, *v = k + D;
    T acc = 0;
    for (int h = 0; h < H; h++) {
      T s = 0, dp = 0;
#pragma unroll
      for (int c = 0; c < d; c++) s += q[h * d + c] * k[h * d + c], dp += dO[h * d + c] * v[h * d + c];
      acc += ex(scale * s + b[j] - lse[h]) * (dp - Di[h]);
    }
    db[j] = acc;
  }
}

// F32, one block per sequence: per head, q, k, v and dO staged in shared
// memory; each thread owns up to PT entries (i, j) of the L x L map
template <int d, int PT>
__global__ void __launch_bounds__(256)
    bwd_bias_seq(const float *__restrict__ qkv, const float *__restrict__ bias,
                 const float *__restrict__ O, const float *__restrict__ LSE,
                 const float *__restrict__ GO, const float *__restrict__ GL,
                 float *__restrict__ dbias, const int H, const int L,
                 const float scale) {
  extern __shared__ float sm[];
  constexpr int S = d + 1;
  const int D = H * d, LL = L * L, nt = blockDim.x, t = threadIdx.x;
  const int64_t n = blockIdx.x;
  float *q = sm, *k = q + L * S, *v = k + L * S, *g = v + L * S, *lse = g + L * S,
        *Di = lse + L;
  const float *x = qkv + n * L * 3 * D;
  const float *o_ = O + n * L * D;
  const float *gr = GO + n * L * D;
  float acc[PT], bb[PT];
#pragma unroll
  for (int p = 0; p < PT; p++) {
    const int e = t + p * nt;
    acc[p] = 0.0f, bb[p] = e < LL ? bias[n * LL + e] : 0.0f;
  }
  for (int h = 0; h < H; h++) {
    __syncthreads();
    for (int e = t; e < L * d; e += nt) {
      const int row = e / d, c = e % d;
      const float *xr = x + (int64_t)row * 3 * D + h * d + c;
      q[row * S + c] = xr[0], k[row * S + c] = xr[D], v[row * S + c] = xr[2 * D];
      g[row * S + c] = gr[(int64_t)row * D + h * d + c];
    }
    for (int row = t; row < L; row += nt) lse[row] = LSE[(n * L + row) * H + h];
    __syncthreads();
    for (int row = t; row < L; row += nt) {
      const float *o = o_ + (int64_t)row * D + h * d;
      float s = 0.0f;
#pragma unroll
      for (int c = 0; c < d; c++) s += g[row * S + c] * o[c];
      Di[row] = GL ? s - GL[(n * L + row) * H + h] : s;
    }
    __syncthreads();
#pragma unroll
    for (int p = 0; p < PT; p++) {
      const int e = t + p * nt;
      if (e >= LL) break;
      const int i = e / L, j = e % L;
      float s = 0.0f, dp = 0.0f;
#pragma unroll
      for (int c = 0; c < d; c++) s += q[i * S + c] * k[j * S + c], dp += g[i * S + c] * v[j * S + c];
      acc[p] += expf(scale * s + bb[p] - lse[i]) * (dp - Di[i]);
    }
  }
#pragma unroll
  for (int p = 0; p < PT; p++) {
    const int e = t + p * nt;
    if (e < LL) dbias[n * LL + e] = acc[p];
  }
}

// ---------------------------------------------------------------------------
// JVP and its gradients (the maths: ggml-cpu/ops.cpp of mlip.cpp's ggml). One
// thread per (sequence, head, row); P recomputed from per-row statistics.

template <int d, typename T> struct pet_jvp {
  const T *x, *dx; // the sequence's qkv / dqkv [L][3][D]; dx may be null
  int D, h;
  __device__ const T *row(const T *t, int j, int s) const {
    return t + ((int64_t)j * 3 + s) * D + h * d;
  }
  __device__ void load(const T *t, int j, int s, T *r) const {
#pragma unroll
    for (int c = 0; c < d; c++) r[c] = t ? row(t, j, s)[c] : T(0);
  }
  __device__ static T dot(const T *r, const T *g) {
    T s = 0;
#pragma unroll
    for (int c = 0; c < d; c++) s += r[c] * g[c];
    return s;
  }
  __device__ T S(const T *q, int j, T scale, const T *b) const {
    return scale * dot(q, row(x, j, 1)) + b[j];
  }
  __device__ T dS(const T *q, const T *dq, int j, T scale, const T *db) const {
    T t = db ? db[j] : T(0);
    if (dx) t += scale * (dot(dq, row(x, j, 1)) + dot(q, row(dx, j, 1)));
    return t;
  }
};

// per row: max of S, 1 / sum exp(S - max), c = sum P dS, D = sum P gv, E
template <typename T> struct jvp_stats { T mx, inv, c, D, E; };

template <int d, typename T>
__device__ jvp_stats<T> jvp_row_stats(const pet_jvp<d, T> &s, int L, T scale,
                                      const T *b, const T *db, const T *q,
                                      const T *dq, const T *G, T g) {
  jvp_stats<T> r{-INFINITY, 0, 0, 0, 0};
  for (int j = 0; j < L; j++) r.mx = max(r.mx, s.S(q, j, scale, b));
  T sum = 0, c = 0, D = 0;
  for (int j = 0; j < L; j++) {
    const T e = ex(s.S(q, j, scale, b) - r.mx);
    sum += e, c += e == T(0) ? T(0) : e * s.dS(q, dq, j, scale, db);
    if (G) D += e * s.dot(G, s.row(s.x, j, 2));
  }
  r.inv = T(1) / sum, r.c = c * r.inv, r.D = D * r.inv;
  if (!G) return r;
  T E = 0;
  for (int j = 0; j < L; j++) {
    const T P = ex(s.S(q, j, scale, b) - r.mx) * r.inv;
    if (P == T(0)) continue;
    const T dS = s.dS(q, dq, j, scale, db);
    const T gv = s.dot(G, s.row(s.x, j, 2)),
            gdv = s.dx ? s.dot(G, s.row(s.dx, j, 2)) : T(0);
    E += P * (dS * (gv - r.D + g) - r.c * gv + gdv);
  }
  r.E = E;
  return r;
}

template <int d, typename T>
__global__ void jvp_fwd(const T *__restrict__ qkv, const T *__restrict__ bias,
                        const T *__restrict__ dqkv, const T *__restrict__ dbias,
                        T *__restrict__ to, T *__restrict__ tlse, const int H,
                        const int L, const int64_t N, const T scale) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * H * L) return;
  const int i = t % L, h = (t / L) % H;
  const int64_t n = t / ((int64_t)L * H);
  const int D = H * d;
  const pet_jvp<d, T> s{qkv + n * L * 3 * D, dqkv ? dqkv + n * L * 3 * D : nullptr, D, h};
  const T *b = bias + (n * L + i) * L, *db = dbias ? dbias + (n * L + i) * L : nullptr;
  T q[d], dq[d], o[d], O[d], odv[d];
  s.load(s.x, i, 0, q), s.load(s.dx, i, 0, dq);
#pragma unroll
  for (int c = 0; c < d; c++) o[c] = O[c] = odv[c] = 0;
  T mx = -INFINITY;
  for (int j = 0; j < L; j++) mx = max(mx, s.S(q, j, scale, b));
  T sum = 0, cs = 0;
  for (int j = 0; j < L; j++) {
    const T e = ex(s.S(q, j, scale, b) - mx);
    if (e == T(0)) continue; // masked key: also skips a non-finite tangent there
    const T w = e * s.dS(q, dq, j, scale, db);
    const T *v = s.row(s.x, j, 2), *dv = s.dx ? s.row(s.dx, j, 2) : nullptr;
    sum += e, cs += w;
#pragma unroll
    for (int c = 0; c < d; c++)
      o[c] += w * v[c], O[c] += e * v[c], odv[c] += dv ? e * dv[c] : T(0);
  }
  const T inv = T(1) / sum, c = cs * inv;
  T *out = to + (n * L + i) * D;
#pragma unroll
  for (int k = 0; k < d; k++) out[h * d + k] = (o[k] - c * O[k] + odv[k]) * inv;
  tlse[(n * L + i) * H + h] = c;
}

// statistics of every (sequence, head, row), and with gx the query-side
// gradients (q, tq: slots 0, 3 of gx [N, L, 6, D])
template <int d, typename T>
__global__ void jvp_bwd_rows(const T *__restrict__ qkv, const T *__restrict__ bias,
                             const T *__restrict__ GO, const T *__restrict__ GL,
                             const T *__restrict__ dqkv, const T *__restrict__ dbias,
                             jvp_stats<T> *__restrict__ stats, T *__restrict__ dst,
                             const int H, const int L, const int64_t N, const T scale) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * H * L) return;
  const int i = t % L, h = (t / L) % H;
  const int64_t n = t / ((int64_t)L * H);
  const int D = H * d;
  const pet_jvp<d, T> s{qkv + n * L * 3 * D, dqkv ? dqkv + n * L * 3 * D : nullptr, D, h};
  const T *b = bias + (n * L + i) * L, *db = dbias ? dbias + (n * L + i) * L : nullptr;
  const T *gi = GO + (n * L + i) * D;
  T q[d], dq[d], G[d];
  s.load(s.x, i, 0, q), s.load(s.dx, i, 0, dq);
#pragma unroll
  for (int c = 0; c < d; c++) G[c] = gi[h * d + c];
  const T g = GL ? GL[(n * L + i) * H + h] : T(0);
  const jvp_stats<T> r = jvp_row_stats<d, T>(s, L, scale, b, db, q, dq, G, g);
  stats[(n * H + h) * L + i] = r;
  if (!dst) return;
  T gq[d], gdq[d];
#pragma unroll
  for (int c = 0; c < d; c++) gq[c] = gdq[c] = 0;
  for (int j = 0; j < L; j++) {
    const T P = ex(s.S(q, j, scale, b) - r.mx) * r.inv;
    if (P == T(0)) continue;
    const T dS = s.dS(q, dq, j, scale, db);
    const T gv = s.dot(G, s.row(s.x, j, 2)),
            gdv = s.dx ? s.dot(G, s.row(s.dx, j, 2)) : T(0);
    const T Gam = P * (gv - r.D + g),
            Sig = P * (dS * (gv - r.D + g) - r.c * gv + gdv - r.E);
    const T *k = s.row(s.x, j, 1), *dk = s.dx ? s.row(s.dx, j, 1) : nullptr;
#pragma unroll
    for (int c = 0; c < d; c++)
      gq[c] += scale * (Sig * k[c] + (dk ? Gam * dk[c] : T(0))),
          gdq[c] += scale * Gam * k[c];
  }
  T *out = dst + ((n * L + i) * 6) * D + h * d;
#pragma unroll
  for (int c = 0; c < d; c++) out[c] = gq[c], out[3 * D + c] = gdq[c];
}

// key side: k, v, tk, tv (slots 1, 2, 4, 5), one thread per (sequence, head, key)
template <int d, typename T>
__global__ void jvp_bwd_keys(const T *__restrict__ qkv, const T *__restrict__ bias,
                             const T *__restrict__ GO, const T *__restrict__ GL,
                             const T *__restrict__ dqkv, const T *__restrict__ dbias,
                             const jvp_stats<T> *__restrict__ stats, T *__restrict__ dst,
                             const int H, const int L, const int64_t N, const T scale) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * H * L) return;
  const int j = t % L, h = (t / L) % H;
  const int64_t n = t / ((int64_t)L * H);
  const int D = H * d;
  const pet_jvp<d, T> s{qkv + n * L * 3 * D, dqkv ? dqkv + n * L * 3 * D : nullptr, D, h};
  T k[d], dk[d], v[d], dv[d], gk[d], gdk[d], gv[d], gdv[d];
  s.load(s.x, j, 1, k), s.load(s.dx, j, 1, dk), s.load(s.x, j, 2, v), s.load(s.dx, j, 2, dv);
#pragma unroll
  for (int c = 0; c < d; c++) gk[c] = gdk[c] = gv[c] = gdv[c] = 0;
  for (int i = 0; i < L; i++) {
    const jvp_stats<T> r = stats[(n * H + h) * L + i];
    const T *q = s.row(s.x, i, 0), *dq = s.dx ? s.row(s.dx, i, 0) : nullptr;
    const T *G = GO + (n * L + i) * D + h * d;
    const T b = bias[(n * L + i) * L + j];
    const T P = ex(scale * s.dot(k, q) + b - r.mx) * r.inv;
    if (P == T(0)) continue;
    const T db = dbias ? dbias[(n * L + i) * L + j] : T(0);
    const T dS = db + (dq ? scale * (s.dot(k, dq) + s.dot(dk, q)) : T(0));
    const T g = GL ? GL[(n * L + i) * H + h] : T(0), gvv = s.dot(v, G),
            gdvv = dq ? s.dot(dv, G) : T(0);
    const T Gam = P * (gvv - r.D + g),
            Sig = P * (dS * (gvv - r.D + g) - r.c * gvv + gdvv - r.E);
    const T w = P * (dS - r.c);
#pragma unroll
    for (int c = 0; c < d; c++) {
      gk[c] += scale * (Sig * q[c] + (dq ? Gam * dq[c] : T(0)));
      gdk[c] += scale * Gam * q[c];
      gv[c] += w * G[c];
      gdv[c] += P * G[c];
    }
  }
  T *out = dst + ((n * L + j) * 6) * D + h * d;
#pragma unroll
  for (int c = 0; c < d; c++)
    out[D + c] = gk[c], out[2 * D + c] = gv[c], out[4 * D + c] = gdk[c], out[5 * D + c] = gdv[c];
}

// bias and tbias gradients (planes 0, 1 of gb [N, 2, L, L]), summed over the
// heads: one thread per (sequence, row, key)
template <int d, typename T>
__global__ void jvp_bwd_bias(const T *__restrict__ qkv, const T *__restrict__ bias,
                             const T *__restrict__ GO, const T *__restrict__ GL,
                             const T *__restrict__ dqkv, const T *__restrict__ dbias,
                             const jvp_stats<T> *__restrict__ stats, T *__restrict__ dst,
                             const int H, const int L, const int64_t N, const T scale) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N * L * L) return;
  const int j = t % L, i = (t / L) % L;
  const int64_t n = t / ((int64_t)L * L);
  const int D = H * d;
  const T b = bias[(n * L + i) * L + j];
  const T db = dbias ? dbias[(n * L + i) * L + j] : T(0);
  const T *gi = GO + (n * L + i) * D;
  T sb = 0, sdb = 0;
  for (int h = 0; h < H; h++) {
    const pet_jvp<d, T> s{qkv + n * L * 3 * D, dqkv ? dqkv + n * L * 3 * D : nullptr, D, h};
    const jvp_stats<T> r = stats[(n * H + h) * L + i];
    const T *q = s.row(s.x, i, 0), *k = s.row(s.x, j, 1), *G = gi + h * d;
    const T P = ex(scale * s.dot(q, k) + b - r.mx) * r.inv;
    if (P == T(0)) continue;
    const T dS = db + (s.dx ? scale * (s.dot(s.row(s.dx, i, 0), k) + s.dot(q, s.row(s.dx, j, 1))) : T(0));
    const T g = GL ? GL[(n * L + i) * H + h] : T(0), gvv = s.dot(G, s.row(s.x, j, 2)),
            gdvv = s.dx ? s.dot(G, s.row(s.dx, j, 2)) : T(0);
    sdb += P * (gvv - r.D + g);
    sb += P * (dS * (gvv - r.D + g) - r.c * gvv + gdvv - r.E);
  }
  dst[((n * 2 + 0) * L + i) * L + j] = sb;
  dst[((n * 2 + 1) * L + i) * L + j] = sdb;
}

// ---------------------------------------------------------------------------
// Fused double backward (new, not in ggml): for one sequence per block and its
// heads in turn, everything the backward of the backward needs,
//   to, tlse = J w                    (w = (tq, tk, tv, tbias): the cotangent
//   gx, gb   = d/d(qkv, bias) <u, J w> of the backward's outputs; u = (go, glse))
// with the L x L maps of the head (P, P (dS - c), Gamma, Sigma) in shared
// memory, so every dot product is computed once (the JVP kernels above
// recompute them in each of their passes) and the tangent gradients (slots
// tq, tk, tv, tbias of jvp_backward) are not computed: d/dw <u, J w> = J^T u is
// the backward itself.

template <typename T> __device__ __forceinline__ T warp_sum(T v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
template <typename T> __device__ __forceinline__ T warp_max(T v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

template <int d, typename T>
__global__ void __launch_bounds__(256)
    dbl_bwd_seq(const T *__restrict__ qkv, const T *__restrict__ bias, const T *__restrict__ GO,
                const T *__restrict__ GL, const T *__restrict__ tqkv, const T *__restrict__ tbias,
                T *__restrict__ to, T *__restrict__ tlse, T *__restrict__ gx, T *__restrict__ gb, const int H,
                const int L, const T scale) {
  extern __shared__ __align__(16) unsigned char smraw[];
  T *sm = reinterpret_cast<T *>(smraw);
  constexpr int S = d + 1;
  const int D = H * d, LL = L * L, nt = blockDim.x, t = threadIdx.x, lane = t % 32, warp = t / 32,
            nw = nt / 32;
  const int64_t n = blockIdx.x;
  T *q = sm, *k = q + L * S, *v = k + L * S, *dq = v + L * S, *dk = dq + L * S, *dv = dk + L * S,
    *G = dv + L * S;
  T *M0 = G + L * S, *M1 = M0 + LL, *M2 = M1 + LL, *M3 = M2 + LL;
  const T *x = qkv + n * L * 3 * D, *tx = tqkv ? tqkv + n * L * 3 * D : nullptr;
  const T *b = bias + n * LL, *db = tbias ? tbias + n * LL : nullptr;
  const T *go = GO + n * L * D;

  for (int h = 0; h < H; h++) {
    __syncthreads();
    for (int e = t; e < L * d; e += nt) {
      const int r = e / d, c = e % d;
      const int64_t o = (int64_t)r * 3 * D + h * d + c;
      q[r * S + c] = x[o], k[r * S + c] = x[o + D], v[r * S + c] = x[o + 2 * D];
      dq[r * S + c] = tx ? tx[o] : T(0), dk[r * S + c] = tx ? tx[o + D] : T(0),
      dv[r * S + c] = tx ? tx[o + 2 * D] : T(0);
      G[r * S + c] = go[(int64_t)r * D + h * d + c];
    }
    __syncthreads();
    // scores, their tangents and the products with the incoming gradient
    for (int e = t; e < LL; e += nt) {
      const int i = e / L, j = e % L;
      T s = 0, ds = 0, gv = 0, gdv = 0;
#pragma unroll
      for (int c = 0; c < d; c++) {
        const T qi = q[i * S + c], kj = k[j * S + c], Gi = G[i * S + c];
        s += qi * kj, ds += dq[i * S + c] * kj + qi * dk[j * S + c];
        gv += Gi * v[j * S + c], gdv += Gi * dv[j * S + c];
      }
      M0[e] = scale * s + b[e];
      M1[e] = scale * ds + (db ? db[e] : T(0));
      M2[e] = gv, M3[e] = gdv;
    }
    __syncthreads();
    // per row (one warp each): softmax, the row statistics, and the maps
    for (int i = warp; i < L; i += nw) {
      T *Si = M0 + i * L, *dSi = M1 + i * L, *gvi = M2 + i * L, *gdvi = M3 + i * L;
      T mx = -INFINITY;
      for (int j = lane; j < L; j += 32) mx = max(mx, Si[j]);
      mx = warp_max(mx);
      T sum = 0;
      for (int j = lane; j < L; j += 32) sum += ex(Si[j] - mx);
      const T inv = T(1) / warp_sum(sum);
      T c = 0, Dr = 0;
      for (int j = lane; j < L; j += 32) {
        const T P = ex(Si[j] - mx) * inv;
        Si[j] = P;
        if (P != T(0)) c += P * dSi[j], Dr += P * gvi[j];
      }
      c = warp_sum(c), Dr = warp_sum(Dr);
      const T g = GL ? GL[(n * L + i) * H + h] : T(0);
      T E = 0;
      for (int j = lane; j < L; j += 32) {
        const T P = Si[j];
        if (P != T(0)) E += P * (dSi[j] * (gvi[j] - Dr + g) - c * gvi[j] + gdvi[j]);
      }
      E = warp_sum(E);
      for (int j = lane; j < L; j += 32) {
        const T P = Si[j];
        if (P == T(0)) {
          dSi[j] = gvi[j] = gdvi[j] = 0;
        } else {
          const T a = gvi[j] - Dr + g;
          const T sig = P * (dSi[j] * a - c * gvi[j] + gdvi[j] - E);
          dSi[j] = P * (dSi[j] - c); // W
          gvi[j] = P * a;            // Gamma
          gdvi[j] = sig;             // Sigma
        }
      }
      if (lane == 0) tlse[(n * L + i) * H + h] = c;
    }
    __syncthreads();
    // bias gradient: sum over the heads of Sigma (this block owns sequence n)
    for (int e = t; e < LL; e += nt) gb[n * LL + e] = (h ? gb[n * LL + e] : T(0)) + M3[e];
    // to = W v + P dv; gq = scale (Sigma k + Gamma dk); gk = scale (Sigma^T q +
    // Gamma^T dq); gv = W^T G
    for (int e = t; e < L * d; e += nt) {
      const int r = e / d, c = e % d;
      T a_to = 0, a_q = 0, a_k = 0, a_v = 0;
      for (int j = 0; j < L; j++) {
        const int rj = r * L + j, jr = j * L + r;
        a_to += M1[rj] * v[j * S + c] + M0[rj] * dv[j * S + c];
        a_q += M3[rj] * k[j * S + c] + M2[rj] * dk[j * S + c];
        a_k += M3[jr] * q[j * S + c] + M2[jr] * dq[j * S + c];
        a_v += M1[jr] * G[j * S + c];
      }
      const int64_t row = n * L + r;
      to[row * D + h * d + c] = a_to;
      T *gr = gx + row * 3 * D + h * d + c;
      gr[0] = scale * a_q, gr[D] = scale * a_k, gr[2 * D] = a_v;
    }
  }
}

// ---------------------------------------------------------------------------
// host side

bool head_dim_supported(int d) { return d == 8 || d == 16 || d == 32 || d == 64; }

size_t jvp_stats_bytes(int64_t rows, bool dbl) {
  return (size_t)rows * (dbl ? sizeof(jvp_stats<double>) : sizeof(jvp_stats<float>));
}

// d as a compile-time constant
template <typename F> static void with_d(int d, F &&f) {
  switch (d) {
  case 8: f(std::integral_constant<int, 8>{}); break;
  case 16: f(std::integral_constant<int, 16>{}); break;
  case 32: f(std::integral_constant<int, 32>{}); break;
  case 64: f(std::integral_constant<int, 64>{}); break;
  default: throw std::runtime_error("mlipcpp pet_attn: head dim must be 8, 16, 32 or 64");
  }
}

// v1 positions per block: <= 256 threads, shrunk until the tiles fit
template <typename T> static int v1_tile(int H, int d, bool back) {
  int Tt = std::max(1, 256 / H);
  auto smem = [&](int Tt) {
    return (2 * (size_t)Tt * H * (d + 1) + (size_t)Tt * Tt + (back ? 2 * (size_t)Tt * H : 0)) * sizeof(T);
  };
  while (Tt > 1 && smem(Tt) > MAX_SMEM) Tt /= 2;
  if (smem(Tt) > MAX_SMEM) throw std::runtime_error("mlipcpp pet_attn: H*d too large for v1 shared memory");
  return Tt;
}

struct v2_shape { int TP, TS; size_t smem; };

static v2_shape v2_shape_of(int H, int d, int R, int L, bool dkv) {
  const int G = std::max(1, std::min(256 / H, (L + R - 1) / R)), TP = G * R;
  auto smem = [&](int TS) {
    return (2 * (size_t)TS * H * (d + 4) + (size_t)TP * TS + (dkv ? 2 * (size_t)TS * H : 0)) * sizeof(float);
  };
  int TS = 32;
  while (TS > 1 && (TS / 2 >= L || smem(TS) > MAX_SMEM)) TS /= 2;
  if (smem(TS) > MAX_SMEM) throw std::runtime_error("mlipcpp pet_attn: H*d too large for v2 shared memory");
  return {TP, TS, smem(TS)};
}

static void check_grid(int64_t N) {
  if (N > INT_MAX) throw std::runtime_error("mlipcpp pet_attn: too many sequences");
}

template <typename T>
void forward(const T *qkv, const T *bias, T *o, T *lse, int64_t N, int L, int H,
             int d, double scale, cudaStream_t stream, int variant) {
  if (N == 0 || L == 0) return;
  check_grid(N);
  with_d(d, [&](auto dc) {
    constexpr int D_ = decltype(dc)::value;
    if (std::is_same<T, float>::value && variant == 0) {
      constexpr int RF = 64 / D_ > 1 ? 64 / D_ : 1;
      const v2_shape s = v2_shape_of(H, D_, RF, L, false);
      const dim3 grid((unsigned)N, (L + s.TP - 1) / s.TP), block(s.TP / RF * H);
      v2_fwd<D_, RF><<<grid, block, s.smem, stream>>>((const float *)qkv, (const float *)bias, (float *)o,
                                                      (float *)lse, H, L, s.TP, s.TS, (float)scale);
    } else {
      const int Tt = v1_tile<T>(H, D_, false);
      const dim3 grid((unsigned)N, (L + Tt - 1) / Tt), block(Tt * H);
      const size_t smem = (2 * (size_t)Tt * H * (D_ + 1) + (size_t)Tt * Tt) * sizeof(T);
      v1_fwd<D_, T><<<grid, block, smem, stream>>>(qkv, bias, o, lse, H, L, Tt, (T)scale);
    }
  });
  PA_CHECK(cudaGetLastError());
}

template <typename T>
void backward(const T *qkv, const T *bias, const T *o, const T *lse, const T *go,
              const T *glse, T *dqkv, T *dbias, int64_t N, int L, int H, int d,
              double scale, cudaStream_t stream, int variant) {
  if (N == 0 || L == 0) return;
  check_grid(N);
  with_d(d, [&](auto dc) {
    constexpr int D_ = decltype(dc)::value;
    if (dqkv) {
      if (std::is_same<T, float>::value && variant == 0) {
        constexpr int RB = 32 / D_ > 1 ? 32 / D_ : 1;
        const v2_shape sq = v2_shape_of(H, D_, RB, L, false), sk = v2_shape_of(H, D_, RB, L, true);
        v2_bwd_dq<D_, RB><<<dim3((unsigned)N, (L + sq.TP - 1) / sq.TP), sq.TP / RB * H, sq.smem, stream>>>(
            (const float *)qkv, (const float *)bias, (const float *)o, (const float *)lse, (const float *)go,
            (const float *)glse, (float *)dqkv, H, L, sq.TP, sq.TS, (float)scale);
        v2_bwd_dkv<D_, RB><<<dim3((unsigned)N, (L + sk.TP - 1) / sk.TP), sk.TP / RB * H, sk.smem, stream>>>(
            (const float *)qkv, (const float *)bias, (const float *)o, (const float *)lse, (const float *)go,
            (const float *)glse, (float *)dqkv, H, L, sk.TP, sk.TS, (float)scale);
      } else {
        const int Tt = v1_tile<T>(H, D_, true);
        const dim3 grid((unsigned)N, (L + Tt - 1) / Tt), block(Tt * H);
        const size_t smem_q = (2 * (size_t)Tt * H * (D_ + 1) + (size_t)Tt * Tt) * sizeof(T);
        const size_t smem_k = smem_q + 2 * (size_t)Tt * H * sizeof(T);
        v1_bwd_dq<D_, T><<<grid, block, smem_q, stream>>>(qkv, bias, o, lse, go, glse, dqkv, H, L, Tt, (T)scale);
        v1_bwd_dkv<D_, T><<<grid, block, smem_k, stream>>>(qkv, bias, o, lse, go, glse, dqkv, H, L, Tt, (T)scale);
      }
      PA_CHECK(cudaGetLastError());
    }
    if (dbias) {
      const size_t smem_seq = (4 * (size_t)L * (D_ + 1) + 2 * (size_t)L) * sizeof(float);
      const int64_t LL = (int64_t)L * L;
      if (std::is_same<T, float>::value && variant == 0 && smem_seq <= MAX_SMEM && LL <= 16 * 256) {
        auto launch = [&](auto kernel) {
          kernel<<<(unsigned)N, 256, smem_seq, stream>>>((const float *)qkv, (const float *)bias, (const float *)o,
                                                         (const float *)lse, (const float *)go, (const float *)glse,
                                                         (float *)dbias, H, L, (float)scale);
        };
        if (LL <= 4 * 256) launch(bwd_bias_seq<D_, 4>);
        else if (LL <= 8 * 256) launch(bwd_bias_seq<D_, 8>);
        else launch(bwd_bias_seq<D_, 16>);
      } else {
        const size_t smem = (2 * (size_t)H * D_ + 2 * H) * sizeof(T);
        if (smem > MAX_SMEM || N > 65535)
          throw std::runtime_error("mlipcpp pet_attn: bias gradient shape not supported (N > 65535 or H*d too large)");
        const int threads = std::min(128, (L + 31) / 32 * 32);
        bwd_bias_rows<D_, T><<<dim3((unsigned)L, (unsigned)N), threads, smem, stream>>>(
            qkv, bias, o, lse, go, glse, dbias, H, L, (T)scale);
      }
      PA_CHECK(cudaGetLastError());
    }
  });
}

template <typename T>
void jvp(const T *qkv, const T *bias, const T *tqkv, const T *tbias, T *to, T *tlse,
         int64_t N, int L, int H, int d, double scale, cudaStream_t stream) {
  if (N == 0 || L == 0) return;
  const int threads = 128;
  const int64_t rows = N * H * L;
  if ((rows + threads - 1) / threads > INT_MAX) throw std::runtime_error("mlipcpp pet_attn_jvp: too large");
  const unsigned blocks = (unsigned)((rows + threads - 1) / threads);
  with_d(d, [&](auto dc) {
    constexpr int D_ = decltype(dc)::value;
    jvp_fwd<D_, T><<<blocks, threads, 0, stream>>>(qkv, bias, tqkv, tbias, to, tlse, H, L, N, (T)scale);
  });
  PA_CHECK(cudaGetLastError());
}

template <typename T>
void jvp_backward(const T *qkv, const T *bias, const T *go, const T *glse, const T *tqkv,
                  const T *tbias, void *stats_, T *gx, T *gb, int64_t N, int L, int H, int d,
                  double scale, cudaStream_t stream) {
  if (N == 0 || L == 0) return;
  const int threads = 128;
  const int64_t rows = N * H * L, cells = N * (int64_t)L * L;
  if ((std::max(rows, cells) + threads - 1) / threads > INT_MAX)
    throw std::runtime_error("mlipcpp pet_attn_jvp_backward: too large");
  const unsigned blocks = (unsigned)((rows + threads - 1) / threads);
  auto *stats = (jvp_stats<T> *)stats_;
  with_d(d, [&](auto dc) {
    constexpr int D_ = decltype(dc)::value;
    jvp_bwd_rows<D_, T><<<blocks, threads, 0, stream>>>(qkv, bias, go, glse, tqkv, tbias, stats, gx, H, L, N, (T)scale);
    if (gx)
      jvp_bwd_keys<D_, T><<<blocks, threads, 0, stream>>>(qkv, bias, go, glse, tqkv, tbias, stats, gx, H, L, N,
                                                          (T)scale);
    if (gb)
      jvp_bwd_bias<D_, T><<<(unsigned)((cells + threads - 1) / threads), threads, 0, stream>>>(
          qkv, bias, go, glse, tqkv, tbias, stats, gb, H, L, N, (T)scale);
  });
  PA_CHECK(cudaGetLastError());
}

template <typename T> static size_t dbl_bwd_smem(int L, int d) {
  return (7 * (size_t)L * (d + 1) + 4 * (size_t)L * L) * sizeof(T);
}

static int smem_optin() {
  static int v = [] {
    int dev = 0, s = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&s, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return s;
  }();
  return v;
}

template <typename T> bool double_backward_fused_ok(int L, int d) {
  const char *e = std::getenv("MLIPCPP_ATTN_DBL_NAIVE");
  if (e && std::atoi(e) != 0) return false;
  return dbl_bwd_smem<T>(L, d) <= (size_t)smem_optin();
}

template <typename T>
void double_backward(const T *qkv, const T *bias, const T *go, const T *glse, const T *tqkv,
                     const T *tbias, T *to, T *tlse, T *gx, T *gb, int64_t N, int L, int H, int d,
                     double scale, cudaStream_t stream) {
  if (N == 0 || L == 0) return;
  check_grid(N);
  const size_t smem = dbl_bwd_smem<T>(L, d);
  with_d(d, [&](auto dc) {
    constexpr int D_ = decltype(dc)::value;
    auto kernel = dbl_bwd_seq<D_, T>;
    static size_t set = 0; // per instantiation: raise the dynamic smem limit once
    if (smem > 48 * 1024 && smem > set) {
      PA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      set = smem;
    }
    kernel<<<(unsigned)N, 256, smem, stream>>>(qkv, bias, go, glse, tqkv, tbias, to, tlse, gx, gb, H, L,
                                                (T)scale);
  });
  PA_CHECK(cudaGetLastError());
}

#define INST(T)                                                                                     \
  template void forward<T>(const T *, const T *, T *, T *, int64_t, int, int, int, double,          \
                           cudaStream_t, int);                                                      \
  template void backward<T>(const T *, const T *, const T *, const T *, const T *, const T *, T *,  \
                            T *, int64_t, int, int, int, double, cudaStream_t, int);                \
  template void jvp<T>(const T *, const T *, const T *, const T *, T *, T *, int64_t, int, int, int, \
                       double, cudaStream_t);                                                       \
  template void jvp_backward<T>(const T *, const T *, const T *, const T *, const T *, const T *,   \
                                void *, T *, T *, int64_t, int, int, int, double, cudaStream_t);   \
  template void double_backward<T>(const T *, const T *, const T *, const T *, const T *, const T *, \
                                   T *, T *, T *, T *, int64_t, int, int, int, double, cudaStream_t); \
  template bool double_backward_fused_ok<T>(int, int);
INST(float)
INST(double)

} // namespace mlipcpp_attn
