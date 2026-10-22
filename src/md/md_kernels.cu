#include "md/md_kernels.h"

#include "core/adaptive_cutoff.h"
#include "core/padding.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace mlipcpp::md::gpu {

namespace {

// 1 eV / (Å amu) in Å / fs^2
constexpr double ACC = 9.64853321233100e-3;
constexpr int BLOCK = 128;

void check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("md: ") + what + ": " + cudaGetErrorString(e));
}
#define LAUNCH(kernel, n, ...)                                                   \
  do {                                                                           \
    if ((n) > 0) kernel<<<((n) + BLOCK - 1) / BLOCK, BLOCK>>>(__VA_ARGS__);      \
    check(cudaGetLastError(), #kernel);                                          \
  } while (0)

struct Cell {
  double m[9];
  int pbc[3];
};

__device__ inline bool primary(int i, int j, const Shift &s) {
  if (i != j) return i < j;
  for (int c = 0; c < 3; c++)
    if (s.s[c] != 0) return s.s[c] > 0;
  return false;
}

// x_b - x_a + shift . cell, always evaluated for the primary orientation of
// the pair so that i -> j and j -> i are exact negatives of each other
__device__ inline void pair_vector(const double *x, const Cell &cell, int i, int j, const Shift &s,
                                   double v[3]) {
  const bool p = primary(i, j, s);
  const int a = p ? i : j, b = p ? j : i;
  const double sg = p ? 1.0 : -1.0;
  for (int c = 0; c < 3; c++) {
    const double t = sg * s.s[0] * cell.m[c] + sg * s.s[1] * cell.m[3 + c] + sg * s.s[2] * cell.m[6 + c];
    v[c] = sg * (x[3 * b + c] - x[3 * a + c] + t);
  }
}

// periodic fractional coordinates into [0, 1); x = f . cell (rows), f = x . inv
struct Inverse {
  double m[9];
};
__global__ void wrap_kernel(int n, double *x, Cell cell, Inverse inverse) {
  const double *inv = inverse.m;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double f[3];
  for (int d = 0; d < 3; d++) {
    f[d] = x[3 * i] * inv[d] + x[3 * i + 1] * inv[3 + d] + x[3 * i + 2] * inv[6 + d];
    if (cell.pbc[d]) f[d] -= floor(f[d]);
  }
  for (int c = 0; c < 3; c++) x[3 * i + c] = f[0] * cell.m[c] + f[1] * cell.m[3 + c] + f[2] * cell.m[6 + c];
}

// one warp per atom i, a lane per candidate j: each lane counts its hits
// over the periodic images, a warp prefix sum gives its first slot, so the
// candidates stay in (j, shift) order as with one thread per atom
__global__ void build_kernel(int n, int K, const double *x, Cell cell, int3 img, double rc2,
                             int32_t *cand_j, Shift *cand_s, int32_t *count, int32_t *overflow) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return; // whole warps
  float cf[9];
  for (int q = 0; q < 9; q++) cf[q] = float(cell.m[q]);
  const float screen2 = float(rc2) * 1.001f + 1e-3f;
  auto visit = [&](int j, auto &&hit) {
    for (int a = -img.x; a <= img.x; a++)
      for (int b = -img.y; b <= img.y; b++)
        for (int c = -img.z; c <= img.z; c++) {
          const Shift s{{int8_t(a), int8_t(b), int8_t(c)}};
          if (i == j && a == 0 && b == 0 && c == 0) continue;
          // float screen with a margin (FP64 is slow on most GPUs); the
          // decision stays the double test of pair_vector
          float d2 = 0;
          for (int q = 0; q < 3; q++) {
            const float t = float(x[3 * j + q] - x[3 * i + q]) + a * cf[q] + b * cf[3 + q] + c * cf[6 + q];
            d2 += t * t;
          }
          if (d2 > screen2) continue;
          double v[3];
          pair_vector(x, cell, i, j, s, v);
          if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < rc2) hit(s);
        }
  };
  int base = 0;
  for (int j0 = 0; j0 < n; j0 += 32) {
    const int j = j0 + lane;
    int hits = 0;
    if (j < n) visit(j, [&](const Shift &) { hits++; });
    int incl = hits;
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += y;
    }
    if (hits) {
      int k = base + incl - hits;
      visit(j, [&](const Shift &s) {
        if (k < K) cand_j[i * K + k] = j, cand_s[i * K + k] = s;
        k++;
      });
    }
    base += __shfl_sync(0xffffffffu, incl, 31);
  }
  if (lane == 0) {
    count[i] = base;
    if (base > K) atomicMax(overflow, base);
  }
}

struct Grid {
  int nc[3];
  bool periodic;  // fractional cells over the periodic cell, else Cartesian over a box
  double lo[3], scale[3]; // open: cell index = (x - lo) * scale
};

__device__ inline int3 cell_of_atom(const double *x, int i, const Grid &g, const Inverse &inv) {
  int c[3];
  for (int d = 0; d < 3; d++) {
    double f;
    if (g.periodic) {
      f = x[3 * i] * inv.m[d] + x[3 * i + 1] * inv.m[3 + d] + x[3 * i + 2] * inv.m[6 + d];
      f = (f - floor(f)) * g.nc[d];
    } else {
      f = (x[3 * i + d] - g.lo[d]) * g.scale[d];
    }
    c[d] = min(max(int(f), 0), g.nc[d] - 1);
  }
  return make_int3(c[0], c[1], c[2]);
}

__global__ void bin_kernel(int n, const double *x, Grid g, Inverse inv, int32_t *cell_of, int32_t *cell_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int3 c = cell_of_atom(x, i, g, inv);
  const int id = (c.z * g.nc[1] + c.y) * g.nc[0] + c.x;
  cell_of[i] = id;
  atomicAdd(cell_count + id, 1);
}

// exclusive scan of the cell counts in one block: start[0..ncell]
__global__ void scan_kernel(int ncell, const int32_t *count, int32_t *start) {
  __shared__ int32_t part[1024];
  const int t = threadIdx.x, chunk = (ncell + 1023) / 1024, b = t * chunk, e = min(b + chunk, ncell);
  int32_t s = 0;
  for (int k = b; k < e; k++) s += count[k];
  part[t] = s;
  __syncthreads();
  if (t == 0) {
    int32_t acc = 0;
    for (int k = 0; k < 1024; k++) {
      const int32_t v = part[k];
      part[k] = acc, acc += v;
    }
    start[ncell] = acc;
  }
  __syncthreads();
  s = part[t];
  for (int k = b; k < e; k++) start[k] = s, s += count[k];
}

__global__ void fill_kernel(int n, const int32_t *cell_of, const int32_t *start, int32_t *fill, int32_t *atoms) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = cell_of[i];
  atoms[start[c] + atomicAdd(fill + c, 1)] = i;
}

__device__ inline bool before(int ja, const Shift &a, int jb, const Shift &b) {
  if (ja != jb) return ja < jb;
  for (int c = 0; c < 3; c++)
    if (a.s[c] != b.s[c]) return a.s[c] < b.s[c];
  return false;
}

// atom i's candidates from the 27 cells around its own, then sorted by
// (j, shift): the order of the all-pairs kernel
__global__ void gather_kernel(int n, int K, const double *x, Cell cell, Grid g, Inverse inv, const int32_t *cell_of,
                              const int32_t *start, const int32_t *atoms, double rc2, int32_t *cand_j, Shift *cand_s,
                              int32_t *count, int32_t *overflow) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float cf[9];
  for (int q = 0; q < 9; q++) cf[q] = float(cell.m[q]);
  const float screen2 = float(rc2) * 1.001f + 1e-3f;
  const int3 ci = cell_of_atom(x, i, g, inv);
  const int me[3] = {ci.x, ci.y, ci.z};
  int k = 0;
  for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) {
        const int dd[3] = {dx, dy, dz};
        int nb[3];
        Shift s{{0, 0, 0}};
        bool inside = true;
        for (int d = 0; d < 3; d++) {
          nb[d] = me[d] + dd[d];
          if (nb[d] < 0 || nb[d] >= g.nc[d]) {
            if (!g.periodic) inside = false;
            const int w = nb[d] < 0 ? -1 : 1;
            s.s[d] = int8_t(w), nb[d] -= w * g.nc[d];
          }
        }
        if (!inside) continue;
        const int c = (nb[2] * g.nc[1] + nb[1]) * g.nc[0] + nb[0];
        for (int p = start[c]; p < start[c + 1]; p++) {
          const int j = atoms[p];
          if (i == j && s.s[0] == 0 && s.s[1] == 0 && s.s[2] == 0) continue;
          float d2 = 0;
          for (int q = 0; q < 3; q++) {
            const float t = float(x[3 * j + q] - x[3 * i + q]) + s.s[0] * cf[q] + s.s[1] * cf[3 + q] + s.s[2] * cf[6 + q];
            d2 += t * t;
          }
          if (d2 > screen2) continue;
          double v[3];
          pair_vector(x, cell, i, j, s, v);
          if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] >= rc2) continue;
          if (k < K) cand_j[i * K + k] = j, cand_s[i * K + k] = s;
          k++;
        }
      }
  count[i] = k;
  if (k > K) {
    atomicMax(overflow, k);
    return;
  }
  int32_t *cj = cand_j + i * K;
  Shift *cs = cand_s + i * K;
  for (int a = 1; a < k; a++) {
    const int j = cj[a];
    const Shift t = cs[a];
    int b = a - 1;
    for (; b >= 0 && before(j, t, cj[b], cs[b]); b--) cj[b + 1] = cj[b], cs[b + 1] = cs[b];
    cj[b + 1] = j, cs[b + 1] = t;
  }
}

// bounding box of the positions, one block: lo[3], hi[3]
__global__ void bbox_kernel(int n, const double *x, double *out) {
  __shared__ double lo[3][256], hi[3][256];
  const int t = threadIdx.x;
  double l[3] = {1e300, 1e300, 1e300}, h[3] = {-1e300, -1e300, -1e300};
  for (int i = t; i < n; i += 256)
    for (int d = 0; d < 3; d++) l[d] = fmin(l[d], x[3 * i + d]), h[d] = fmax(h[d], x[3 * i + d]);
  for (int d = 0; d < 3; d++) lo[d][t] = l[d], hi[d][t] = h[d];
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (t < o)
      for (int d = 0; d < 3; d++) lo[d][t] = fmin(lo[d][t], lo[d][t + o]), hi[d][t] = fmax(hi[d][t], hi[d][t + o]);
    __syncthreads();
  }
  if (t == 0)
    for (int d = 0; d < 3; d++) out[d] = lo[d][0], out[3 + d] = hi[d][0];
}

__global__ void reverse_kernel(int n, int K, const int32_t *cand_j, const Shift *cand_s,
                               const int32_t *count, int32_t *rev) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = e / K, k = e % K;
  if (i >= n || k >= count[i]) return;
  const int j = cand_j[e];
  const Shift s = cand_s[e];
  int r = -1;
  for (int q = 0; q < count[j]; q++) {
    const Shift t = cand_s[j * K + q];
    if (cand_j[j * K + q] == i && t.s[0] == -s.s[0] && t.s[1] == -s.s[1] && t.s[2] == -s.s[2]) {
      r = j * K + q;
      break;
    }
  }
  rev[e] = r;
}

__global__ void edge_kernel(int n, int K, const double *x, Cell cell, const int32_t *cand_j,
                            const Shift *cand_s, const int32_t *count, double *cv, double *cd) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = e / K, k = e % K;
  if (i >= n || k >= count[i]) return;
  double v[3];
  pair_vector(x, cell, i, cand_j[e], cand_s[e], v);
  for (int c = 0; c < 3; c++) cv[3 * e + c] = v[c];
  // + 1e-15 as in training (train/dataset.cpp)
  cd[e] = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + 1e-15;
}

// a warp per atom: the lanes share the edge sums of each Newton step
__global__ void radii_kernel(int n, int K, const int32_t *count, const double *cd, double cutoff,
                             double n_target, double width, double *r, double *root, double *dn,
                             int8_t *clamped) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (i >= n) return; // whole warps
  if (n_target <= 0) {
    if (lane == 0) r[i] = cutoff, root[i] = cutoff, dn[i] = 1, clamped[i] = 1;
    return;
  }
  const double *d = cd + i * K;
  auto warp_sum = [](double &a, double &b) {
    for (int o = 16; o > 0; o >>= 1)
      a += __shfl_xor_sync(0xffffffffu, a, o), b += __shfl_xor_sync(0xffffffffu, b, o);
    // lane 0's totals everywhere, so all lanes take the same Newton steps
    a = __shfl_sync(0xffffffffu, a, 0), b = __shfl_sync(0xffffffffu, b, 0);
  };
  const auto a = adaptive::solve(
      count[i], [&](int k) { return d[k] < cutoff ? d[k] : 1e30; }, cutoff, n_target, width, lane, 32, warp_sum);
  if (lane == 0) r[i] = a.r, root[i] = a.root, dn[i] = a.dn_root, clamped[i] = a.clamped;
}

// atom i's kept edges, in candidate order, into slots [0, nkept); the rest padded
__global__ void pack_kernel(int n, int K, int M, const int32_t *cand_j, const int32_t *count,
                            const int32_t *cand_rev, const double *cv, const double *cd,
                            const double *r, double cutoff, bool adaptive, bool reverse, const int32_t *species,
                            int32_t *cand_slot, int32_t *nkept, int32_t *max_kept, int32_t *g_ns,
                            float *g_ev, float *g_pm, float *g_cv, int32_t *g_rev) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // d of i -> j and j -> i are identical (pair_vector), and so is pc
  auto keep = [&](int e) {
    const double d = cd[e], pc = adaptive ? 0.5 * (r[i] + r[cand_j[e]]) : cutoff;
    return d < cutoff && (!adaptive || d <= pc) && cand_rev[e] >= 0;
  };
  int nk = 0; // `reverse` (MLIPCPP_MD_REVERSE_SLOTS, a test of order invariance): last kept edge first
  if (reverse)
    for (int k = 0; k < count[i]; k++) nk += keep(i * K + k);
  int idx = 0;
  for (int k = 0; k < count[i]; k++) {
    const int e = i * K + k, j = cand_j[e];
    if (!keep(e)) {
      cand_slot[e] = -1;
      continue;
    }
    const int slot = reverse ? nk - 1 - idx : idx;
    idx++;
    const double pc = adaptive ? 0.5 * (r[i] + r[j]) : cutoff;
    cand_slot[e] = slot;
    if (slot < M) {
      const int f = i * M + slot;
      g_ns[f] = species[j];
      for (int c = 0; c < 3; c++) g_ev[3 * f + c] = static_cast<float>(cv[3 * e + c]);
      g_pm[f] = 0.0f;
      g_cv[f] = static_cast<float>(pc);
    }
  }
  nkept[i] = idx;
  atomicMax(max_kept, idx);
  for (int s = idx; s < M; s++) {
    const int f = i * M + s;
    // core/padding.h: an ordinary edge at the cutoff, as metatrain
    g_ns[f] = padding::kSpecies, g_pm[f] = 1.0f, g_cv[f] = static_cast<float>(cutoff), g_rev[f] = 0;
    padding::edge(static_cast<float>(cutoff), &g_ev[3 * f]);
  }
}

__global__ void reverse_slot_kernel(int n, int K, int M, const int32_t *cand_j, const int32_t *count,
                                    const int32_t *cand_rev, const int32_t *cand_slot, int32_t *g_rev) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = e / K, k = e % K;
  if (i >= n || k >= count[i]) return;
  const int slot = cand_slot[e];
  if (slot < 0 || slot >= M) return;
  const int j = cand_j[e], rs = cand_slot[cand_rev[e]];
  g_rev[i * M + slot] = rs >= 0 && rs < M ? j * M + rs : 0;
}

// ---- ragged layout: atom i's kept edges are rows [off[i], off[i + 1]) ----

// the edges atom i keeps (as pack_kernel), and their largest count
__global__ void count_kept_kernel(int n, int K, const int32_t *cand_j, const int32_t *count, const int32_t *cand_rev,
                                  const double *cd, const double *r, double cutoff, bool adaptive, int32_t *nkept,
                                  int32_t *max_kept) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int nk = 0;
  for (int k = 0; k < count[i]; k++) {
    const int e = i * K + k;
    const double d = cd[e], pc = adaptive ? 0.5 * (r[i] + r[cand_j[e]]) : cutoff;
    nk += d < cutoff && (!adaptive || d <= pc) && cand_rev[e] >= 0;
  }
  nkept[i] = nk;
  atomicMax(max_kept, nk);
}

// atom i's kept edges, in candidate order, into rows off[i] + slot; the row's centre atom too
__global__ void pack_ragged_kernel(int n, int K, const int32_t *cand_j, const int32_t *count, const int32_t *cand_rev,
                                   const double *cv, const double *cd, const double *r, double cutoff, bool adaptive,
                                   const int32_t *species, const int32_t *off, int32_t *cand_slot, int32_t *g_ns,
                                   float *g_ev, float *g_cv, int32_t *g_centers) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int idx = 0;
  for (int k = 0; k < count[i]; k++) {
    const int e = i * K + k, j = cand_j[e];
    const double d = cd[e], pc = adaptive ? 0.5 * (r[i] + r[j]) : cutoff;
    if (!(d < cutoff && (!adaptive || d <= pc) && cand_rev[e] >= 0)) {
      cand_slot[e] = -1;
      continue;
    }
    cand_slot[e] = idx;
    const int64_t f = off[i] + idx++;
    g_ns[f] = species[j], g_centers[f] = i;
    for (int c = 0; c < 3; c++) g_ev[3 * f + c] = static_cast<float>(cv[3 * e + c]);
    g_cv[f] = static_cast<float>(pc);
  }
}

__global__ void reverse_ragged_kernel(int n, int K, const int32_t *cand_j, const int32_t *count,
                                      const int32_t *cand_rev, const int32_t *cand_slot, const int32_t *off,
                                      int32_t *g_rev) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = e / K, k = e % K;
  if (i >= n || k >= count[i]) return;
  const int slot = cand_slot[e];
  if (slot < 0) return;
  const int rs = cand_slot[cand_rev[e]]; // kept as well: the pair cutoff is symmetric
  g_rev[off[i] + slot] = rs >= 0 ? off[cand_j[e]] + rs : off[i] + slot;
}

// rows [off[n], e_cap): edges of no atom, at the cutoff (fc = 0), their own reverse
__global__ void tail_ragged_kernel(int n, int64_t e_cap, const int32_t *off, float cutoff, int32_t *g_ns,
                                   float *g_ev, float *g_cv, int32_t *g_rev, int32_t *g_centers) {
  const int64_t f = off[n] + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (f >= e_cap) return;
  g_ns[f] = padding::kSpecies, g_cv[f] = cutoff, g_rev[f] = static_cast<int32_t>(f), g_centers[f] = 0;
  padding::edge(cutoff, &g_ev[3 * f]);
}

// F_k = sum over k's slots of (G[k, s] - G[reverse]); dE/dr_k = sum of
// (C[k, s] + C[reverse]) / 2 (each pair cutoff is the mean of two radii)
__global__ void edge_force_kernel(int n, int M, const int32_t *nkept, const int32_t *off, const int32_t *g_rev,
                                  const float *G, const float *C, double scale, double *f, double *dEdr) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  double F[3] = {0, 0, 0}, S = 0;
  const int64_t a0 = off ? off[k] : int64_t(k) * M;
  for (int s = 0; s < nkept[k]; s++) {
    const int64_t a = a0 + s, b = g_rev[a];
    for (int c = 0; c < 3; c++) F[c] += double(G[3 * a + c]) - double(G[3 * b + c]);
    if (C) S += 0.5 * (double(C[a]) + double(C[b]));
  }
  for (int c = 0; c < 3; c++) f[3 * k + c] = scale * F[c];
  dEdr[k] = S;
}

// chain term of the adaptive cutoff on each candidate edge k -> l:
// dE/dv += dE/dr_k * (dr_k/dd) * v / d
__global__ void chain_kernel(int n, int K, const int32_t *count, const double *cv, const double *cd,
                             double cutoff, double width, const double *r, const double *root,
                             const double *dn, const int8_t *clamped, const double *dEdr, double *chain) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = e / K, k = e % K;
  if (i >= n || k >= count[i]) return;
  const double d = cd[e];
  double coef = 0;
  if (d < cutoff) {
    adaptive::Radius a;
    a.r = r[i], a.root = root[i], a.dn_root = dn[i], a.clamped = clamped[i];
    coef = dEdr[i] * adaptive::ddist(d, a, width) / d;
  }
  for (int c = 0; c < 3; c++) chain[3 * e + c] = coef * cv[3 * e + c];
}

__global__ void chain_gather_kernel(int n, int K, const int32_t *count, const int32_t *cand_rev,
                                    const double *chain, double scale, double *f) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double F[3] = {0, 0, 0};
  for (int k = 0; k < count[i]; k++) {
    const int e = i * K + k, b = cand_rev[e];
    if (b < 0) continue; // (lists are symmetric: never)
    for (int c = 0; c < 3; c++) F[c] += chain[3 * e + c] - chain[3 * b + c];
  }
  for (int c = 0; c < 3; c++) f[3 * i + c] += scale * F[c];
}

__device__ inline void add_virial(double w[6], const double r[3], const double g[3]) {
  w[0] += r[0] * g[0], w[1] += r[1] * g[1], w[2] += r[2] * g[2];
  w[3] += 0.5 * (r[1] * g[2] + r[2] * g[1]);
  w[4] += 0.5 * (r[0] * g[2] + r[2] * g[0]);
  w[5] += 0.5 * (r[0] * g[1] + r[1] * g[0]);
}

// atom k's edges: the slots with their input vectors and gradients, and the
// adaptive chain term of every candidate within the cutoff
__global__ void virial_kernel(int n, int K, int M, const int32_t *nkept, const int32_t *off, const float *ev,
                              const float *G, const int32_t *count, const double *cv, const double *cd, double cutoff,
                              const double *chain, double *out) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  double w[6] = {0, 0, 0, 0, 0, 0}, r[3], g[3];
  const int64_t a0 = off ? off[k] : int64_t(k) * M;
  for (int s = 0; s < nkept[k]; s++) {
    const int64_t a = a0 + s;
    for (int c = 0; c < 3; c++) r[c] = ev[3 * a + c], g[c] = G[3 * a + c];
    add_virial(w, r, g);
  }
  if (chain)
    for (int q = 0; q < count[k]; q++) {
      const int e = k * K + q;
      if (cd[e] < cutoff) add_virial(w, cv + 3 * e, chain + 3 * e);
    }
  for (int c = 0; c < 6; c++) out[6 * k + c] = w[c];
}

__global__ void nc_kernel(int n, const float *nc, const float *scale, double *f) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int c = 0; c < 3; c++) f[3 * i + c] = double(nc[3 * i + c]) * scale[i];
}

__global__ void kick_kernel(int n, double *v, const double *inv_m, const double *f1, double a1,
                            const double *f2, double a2, double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double h = dt * ACC * inv_m[i];
  for (int c = 0; c < 3; c++)
    v[3 * i + c] += h * (a1 * f1[3 * i + c] + (f2 ? a2 * f2[3 * i + c] : 0.0));
}

__global__ void drift_kernel(int n, double *x, const double *v, const double *x_ref, double dt,
                             int32_t *max_disp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double d2 = 0;
  for (int c = 0; c < 3; c++) {
    x[3 * i + c] += dt * v[3 * i + c];
    const double u = x[3 * i + c] - x_ref[3 * i + c];
    d2 += u * u;
  }
  atomicMax(max_disp, __float_as_int(static_cast<float>(d2))); // non-negative floats order as ints
}

Cell make_cell(const State &s) {
  Cell c;
  for (int k = 0; k < 9; k++) c.m[k] = s.cell[k];
  for (int d = 0; d < 3; d++) c.pbc[d] = s.pbc[d];
  return c;
}

} // namespace

struct CellList {
  int32_t *cell_of = nullptr, *atoms = nullptr; // [n]
  int32_t *count = nullptr, *start = nullptr;   // [ncell], [ncell + 1]
  double *bbox = nullptr;                       // [6]
  int n_cap = 0, cell_cap = 0;
};

void free_cells(State &s) {
  if (!s.cells) return;
  for (void *p : {(void *)s.cells->cell_of, (void *)s.cells->atoms, (void *)s.cells->count, (void *)s.cells->start,
                  (void *)s.cells->bbox})
    if (p) cudaFree(p);
  delete s.cells;
  s.cells = nullptr;
}

namespace {
template <class T> void grow(T *&p, int &cap, int want) {
  if (want <= cap) return;
  if (p) check(cudaFree(p), "free");
  check(cudaMalloc(&p, sizeof(T) * size_t(want)), "malloc");
  cap = want;
}

// the cell grid for build_list, or false for the all-pairs kernel
bool make_grid(State &s, double rc, Grid &g) {
  static const char *mode = std::getenv("MLIPCPP_GPU_NL");
  const bool force_cells = mode && std::string(mode) == "cells", force_pairs = mode && std::string(mode) == "pairs";
  if (force_pairs || (!force_cells && s.n < 256)) return false;
  const bool all = s.pbc[0] && s.pbc[1] && s.pbc[2], none = !s.pbc[0] && !s.pbc[1] && !s.pbc[2];
  if (!all && !none) return false;
  g.periodic = all;
  if (all) {
    const double *m = s.cell;
    auto cross = [&](int a, int b, double out[3]) {
      out[0] = m[3 * a + 1] * m[3 * b + 2] - m[3 * a + 2] * m[3 * b + 1];
      out[1] = m[3 * a + 2] * m[3 * b] - m[3 * a] * m[3 * b + 2];
      out[2] = m[3 * a] * m[3 * b + 1] - m[3 * a + 1] * m[3 * b];
    };
    double c12[3];
    cross(1, 2, c12);
    const double volume = std::abs(m[0] * c12[0] + m[1] * c12[1] + m[2] * c12[2]);
    for (int d = 0; d < 3; d++) {
      double w[3];
      cross((d + 1) % 3, (d + 2) % 3, w);
      const double height = volume / std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
      g.nc[d] = int(std::min(height / rc, 1024.0));
      if (g.nc[d] < 3) return false; // the 27 neighbour cells would repeat
    }
  } else {
    if (!s.cells) s.cells = new CellList();
    int one = 0;
    if (!s.cells->bbox) grow(s.cells->bbox, one, 6);
    bbox_kernel<<<1, 256>>>(s.n, s.x, s.cells->bbox);
    check(cudaGetLastError(), "bbox_kernel");
    double b[6];
    check(cudaMemcpy(b, s.cells->bbox, sizeof b, cudaMemcpyDeviceToHost), "bbox");
    for (int d = 0; d < 3; d++) {
      const double extent = b[3 + d] - b[d];
      g.nc[d] = std::max(1, int(std::min(extent / rc, 1024.0)));
      g.lo[d] = b[d];
      g.scale[d] = extent > 0 ? g.nc[d] / extent : 0.0;
    }
  }
  // no more cells than ~ atoms: coarser cells (still at least rc) for sparse systems
  while (double(g.nc[0]) * g.nc[1] * g.nc[2] > 4.0 * s.n + 64) {
    int &d = g.nc[0] >= g.nc[1] && g.nc[0] >= g.nc[2] ? g.nc[0] : g.nc[1] >= g.nc[2] ? g.nc[1] : g.nc[2];
    if (g.periodic && d / 2 < 3) break;
    const int old = d;
    d = std::max(1, d / 2);
    if (!g.periodic) {
      const int k = int(&d - g.nc);
      g.scale[k] *= double(d) / old;
    }
  }
  return true;
}
} // namespace

namespace {

Inverse make_inverse(const State &s) {
  if (!(s.pbc[0] || s.pbc[1] || s.pbc[2])) return Inverse{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
  const double *m = s.cell;
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  return Inverse{{(m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det,
                  (m[1] * m[5] - m[2] * m[4]) / det, (m[5] * m[6] - m[3] * m[8]) / det,
                  (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
                  (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det,
                  (m[0] * m[4] - m[1] * m[3]) / det}};
}

__global__ void displacement_kernel(int n, const double *x, const double *x_ref, int32_t *max_disp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double d2 = 0;
  for (int c = 0; c < 3; c++) {
    const double u = x[3 * i + c] - x_ref[3 * i + c];
    d2 += u * u;
  }
  atomicMax(max_disp, __float_as_int(static_cast<float>(d2))); // non-negative floats order as ints
}

} // namespace

bool list_valid(State &s, double skin) {
  const bool any_pbc = s.pbc[0] || s.pbc[1] || s.pbc[2];
  if (any_pbc) LAUNCH(wrap_kernel, s.n, s.n, s.x, make_cell(s), make_inverse(s));
  check(cudaMemset(s.scalars + 2, 0, sizeof(int32_t)), "memset");
  LAUNCH(displacement_kernel, s.n, s.n, s.x, s.x_ref, s.scalars + 2);
  int32_t bits = 0;
  check(cudaMemcpy(&bits, s.scalars + 2, sizeof(int32_t), cudaMemcpyDeviceToHost), "max displacement");
  float d2;
  std::memcpy(&d2, &bits, sizeof d2);
  return d2 <= 0.25 * skin * skin;
}

void build_list(State &s, const Cutoff &cut) {
  const Cell cell = make_cell(s);
  const bool any_pbc = s.pbc[0] || s.pbc[1] || s.pbc[2];
  const Inverse inv = make_inverse(s);
  if (any_pbc) LAUNCH(wrap_kernel, s.n, s.n, s.x, cell, inv);
  check(cudaMemcpy(s.x_ref, s.x, sizeof(double) * 3 * s.n, cudaMemcpyDeviceToDevice), "x_ref");
  const double rc = cut.cutoff + cut.skin;
  Grid g{};
  if (make_grid(s, rc, g)) {
    if (!s.cells) s.cells = new CellList();
    CellList &c = *s.cells;
    const int ncell = g.nc[0] * g.nc[1] * g.nc[2];
    int n_cap = c.n_cap, cell_cap = c.cell_cap;
    grow(c.cell_of, c.n_cap, s.n), grow(c.atoms, n_cap, s.n);
    grow(c.count, c.cell_cap, ncell + 1), grow(c.start, cell_cap, ncell + 1);
    check(cudaMemset(c.count, 0, sizeof(int32_t) * ncell), "memset");
    LAUNCH(bin_kernel, s.n, s.n, s.x, g, inv, c.cell_of, c.count);
    scan_kernel<<<1, 1024>>>(ncell, c.count, c.start);
    check(cudaGetLastError(), "scan_kernel");
    check(cudaMemset(c.count, 0, sizeof(int32_t) * ncell), "memset");
    LAUNCH(fill_kernel, s.n, s.n, c.cell_of, c.start, c.count, c.atoms);
    LAUNCH(gather_kernel, s.n, s.n, s.K, s.x, cell, g, inv, c.cell_of, c.start, c.atoms, rc * rc, s.cand_j, s.cand_s,
           s.count, s.scalars);
  } else {
    LAUNCH(build_kernel, 32 * s.n, s.n, s.K, s.x, cell, int3{s.images[0], s.images[1], s.images[2]}, rc * rc,
           s.cand_j, s.cand_s, s.count, s.scalars);
  }
  LAUNCH(reverse_kernel, s.n * s.K, s.n, s.K, s.cand_j, s.cand_s, s.count, s.cand_rev);
}

void prepare_inputs(State &s, const Cutoff &c, const Graph &g) {
  const Cell cell = make_cell(s);
  const int E = s.n * s.K;
  LAUNCH(edge_kernel, E, s.n, s.K, s.x, cell, s.cand_j, s.cand_s, s.count, s.cand_v, s.cand_d);
  LAUNCH(radii_kernel, 32 * s.n, s.n, s.K, s.count, s.cand_d, c.cutoff, c.n_target, c.width, s.r, s.root,
         s.dn_root, s.clamped);
  check(cudaMemset(s.scalars + 1, 0, sizeof(int32_t)), "memset");
  LAUNCH(pack_kernel, s.n, s.n, s.K, s.M, s.cand_j, s.count, s.cand_rev, s.cand_v, s.cand_d, s.r,
         c.cutoff, c.n_target > 0, std::getenv("MLIPCPP_MD_REVERSE_SLOTS") != nullptr, s.species, s.cand_slot, s.nkept, s.scalars + 1, g.neighbor_species,
         g.edge_vectors, g.padding_mask, g.cutoff_values, g.reverse);
  LAUNCH(reverse_slot_kernel, E, s.n, s.K, s.M, s.cand_j, s.count, s.cand_rev, s.cand_slot, g.reverse);
  check(cudaMemcpy(g.species, s.species, sizeof(int32_t) * s.n, cudaMemcpyDeviceToDevice), "species");
}

void count_kept(State &s, const Cutoff &c) {
  const Cell cell = make_cell(s);
  const int E = s.n * s.K;
  LAUNCH(edge_kernel, E, s.n, s.K, s.x, cell, s.cand_j, s.cand_s, s.count, s.cand_v, s.cand_d);
  LAUNCH(radii_kernel, 32 * s.n, s.n, s.K, s.count, s.cand_d, c.cutoff, c.n_target, c.width, s.r, s.root,
         s.dn_root, s.clamped);
  check(cudaMemset(s.scalars + 1, 0, sizeof(int32_t)), "memset");
  LAUNCH(count_kept_kernel, s.n, s.n, s.K, s.cand_j, s.count, s.cand_rev, s.cand_d, s.r, c.cutoff, c.n_target > 0,
         s.nkept, s.scalars + 1);
  scan_kernel<<<1, 1024>>>(s.n, s.nkept, s.offsets);
  check(cudaGetLastError(), "scan_kernel");
}

void pack_ragged(State &s, const Cutoff &c, const Graph &g) {
  const int E = s.n * s.K;
  check(cudaMemcpy(g.offsets, s.offsets, sizeof(int32_t) * (s.n + 1), cudaMemcpyDeviceToDevice), "offsets");
  LAUNCH(pack_ragged_kernel, s.n, s.n, s.K, s.cand_j, s.count, s.cand_rev, s.cand_v, s.cand_d, s.r, c.cutoff,
         c.n_target > 0, s.species, s.offsets, s.cand_slot, g.neighbor_species, g.edge_vectors, g.cutoff_values,
         g.centers);
  LAUNCH(reverse_ragged_kernel, E, s.n, s.K, s.cand_j, s.count, s.cand_rev, s.cand_slot, s.offsets, g.reverse);
  // the tail is at most e_cap rows (fewer: blocks past it return)
  const int64_t tail_blocks = (g.e_cap + BLOCK - 1) / BLOCK;
  if (tail_blocks > 0)
    tail_ragged_kernel<<<(unsigned)tail_blocks, BLOCK>>>(s.n, g.e_cap, s.offsets, static_cast<float>(c.cutoff),
                                                         g.neighbor_species, g.edge_vectors, g.cutoff_values,
                                                         g.reverse, g.centers);
  check(cudaGetLastError(), "tail_ragged_kernel");
  check(cudaMemcpy(g.species, s.species, sizeof(int32_t) * s.n, cudaMemcpyDeviceToDevice), "species");
}

void conservative_forces(State &s, const Cutoff &c, const Graph &g, double energy_scale) {
  const bool adaptive = c.n_target > 0 && g.grad_cv;
  LAUNCH(edge_force_kernel, s.n, s.n, s.M, s.nkept, g.offsets, g.reverse, g.grad_ev, adaptive ? g.grad_cv : nullptr,
         energy_scale, s.f_cons, s.dEdr);
  if (!adaptive) return;
  LAUNCH(chain_kernel, s.n * s.K, s.n, s.K, s.count, s.cand_v, s.cand_d, c.cutoff, c.width, s.r, s.root,
         s.dn_root, s.clamped, s.dEdr, s.chain);
  LAUNCH(chain_gather_kernel, s.n, s.n, s.K, s.count, s.cand_rev, s.chain, energy_scale, s.f_cons);
}

void virial(State &s, const Cutoff &c, const Graph &g, double *w) {
  const bool adaptive = c.n_target > 0 && g.grad_cv;
  LAUNCH(virial_kernel, s.n, s.n, s.K, s.M, s.nkept, g.offsets, g.edge_vectors, g.grad_ev, s.count, s.cand_v, s.cand_d,
         c.cutoff, adaptive ? s.chain : nullptr, w);
}

void nc_forces(State &s, const Graph &g) { LAUNCH(nc_kernel, s.n, s.n, g.nc, s.nc_scale, s.f_nc); }

void kick(State &s, const double *f1, double a1, const double *f2, double a2, double dt) {
  LAUNCH(kick_kernel, s.n, s.n, s.v, s.inv_mass, f1, a1, f2, a2, dt);
}

void drift(State &s, double dt) { LAUNCH(drift_kernel, s.n, s.n, s.x, s.v, s.x_ref, dt, s.scalars + 2); }

void reset_scalars(State &s) { check(cudaMemset(s.scalars, 0, 3 * sizeof(int32_t)), "memset"); }

void synchronize() { check(cudaDeviceSynchronize(), "synchronize"); }

} // namespace mlipcpp::md::gpu
