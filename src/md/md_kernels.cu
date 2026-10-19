#include "md/md_kernels.h"

#include "core/adaptive_cutoff.h"
#include "core/padding.h"

#include <cuda_runtime.h>

#include <cstdlib>
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
__global__ void wrap_kernel(int n, double *x, Cell cell, const double *inv) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double f[3];
  for (int d = 0; d < 3; d++) {
    f[d] = x[3 * i] * inv[d] + x[3 * i + 1] * inv[3 + d] + x[3 * i + 2] * inv[6 + d];
    if (cell.pbc[d]) f[d] -= floor(f[d]);
  }
  for (int c = 0; c < 3; c++) x[3 * i + c] = f[0] * cell.m[c] + f[1] * cell.m[3 + c] + f[2] * cell.m[6 + c];
}

__global__ void build_kernel(int n, int K, const double *x, Cell cell, int3 img, double rc2,
                             int32_t *cand_j, Shift *cand_s, int32_t *count, int32_t *overflow) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int k = 0;
  for (int j = 0; j < n; j++)
    for (int a = -img.x; a <= img.x; a++)
      for (int b = -img.y; b <= img.y; b++)
        for (int c = -img.z; c <= img.z; c++) {
          const Shift s{{int8_t(a), int8_t(b), int8_t(c)}};
          if (i == j && a == 0 && b == 0 && c == 0) continue;
          double v[3];
          pair_vector(x, cell, i, j, s, v);
          if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] >= rc2) continue;
          if (k < K) cand_j[i * K + k] = j, cand_s[i * K + k] = s;
          k++;
        }
  count[i] = k;
  if (k > K) atomicMax(overflow, k);
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

__global__ void radii_kernel(int n, int K, const int32_t *count, const double *cd, double cutoff,
                             double n_target, double width, double *r, double *root, double *dn,
                             int8_t *clamped) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (n_target <= 0) {
    r[i] = cutoff, root[i] = cutoff, dn[i] = 1, clamped[i] = 1;
    return;
  }
  const double *d = cd + i * K;
  const auto a = adaptive::solve(
      count[i], [&](int k) { return d[k] < cutoff ? d[k] : 1e30; }, cutoff, n_target, width);
  r[i] = a.r, root[i] = a.root, dn[i] = a.dn_root, clamped[i] = a.clamped;
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

// F_k = sum over k's slots of (G[k, s] - G[reverse]); dE/dr_k = sum of
// (C[k, s] + C[reverse]) / 2 (each pair cutoff is the mean of two radii)
__global__ void edge_force_kernel(int n, int M, const int32_t *nkept, const int32_t *g_rev,
                                  const float *G, const float *C, double scale, double *f, double *dEdr) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  double F[3] = {0, 0, 0}, S = 0;
  for (int s = 0; s < nkept[k]; s++) {
    const int a = k * M + s, b = g_rev[a];
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

void build_list(State &s, const Cutoff &cut) {
  const Cell cell = make_cell(s);
  const bool any_pbc = s.pbc[0] || s.pbc[1] || s.pbc[2];
  if (any_pbc) {
    const double *m = s.cell;
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    double inv[9] = {(m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det,
                     (m[1] * m[5] - m[2] * m[4]) / det, (m[5] * m[6] - m[3] * m[8]) / det,
                     (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
                     (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det,
                     (m[0] * m[4] - m[1] * m[3]) / det};
    double *d_inv;
    check(cudaMalloc(&d_inv, sizeof inv), "malloc");
    check(cudaMemcpy(d_inv, inv, sizeof inv, cudaMemcpyHostToDevice), "copy");
    LAUNCH(wrap_kernel, s.n, s.n, s.x, cell, d_inv);
    check(cudaFree(d_inv), "free");
  }
  check(cudaMemcpy(s.x_ref, s.x, sizeof(double) * 3 * s.n, cudaMemcpyDeviceToDevice), "x_ref");
  const double rc = cut.cutoff + cut.skin;
  LAUNCH(build_kernel, s.n, s.n, s.K, s.x, cell, int3{s.images[0], s.images[1], s.images[2]}, rc * rc,
         s.cand_j, s.cand_s, s.count, s.scalars);
  LAUNCH(reverse_kernel, s.n * s.K, s.n, s.K, s.cand_j, s.cand_s, s.count, s.cand_rev);
}

void prepare_inputs(State &s, const Cutoff &c, const Graph &g) {
  const Cell cell = make_cell(s);
  const int E = s.n * s.K;
  LAUNCH(edge_kernel, E, s.n, s.K, s.x, cell, s.cand_j, s.cand_s, s.count, s.cand_v, s.cand_d);
  LAUNCH(radii_kernel, s.n, s.n, s.K, s.count, s.cand_d, c.cutoff, c.n_target, c.width, s.r, s.root,
         s.dn_root, s.clamped);
  check(cudaMemset(s.scalars + 1, 0, sizeof(int32_t)), "memset");
  LAUNCH(pack_kernel, s.n, s.n, s.K, s.M, s.cand_j, s.count, s.cand_rev, s.cand_v, s.cand_d, s.r,
         c.cutoff, c.n_target > 0, std::getenv("MLIPCPP_MD_REVERSE_SLOTS") != nullptr, s.species, s.cand_slot, s.nkept, s.scalars + 1, g.neighbor_species,
         g.edge_vectors, g.padding_mask, g.cutoff_values, g.reverse);
  LAUNCH(reverse_slot_kernel, E, s.n, s.K, s.M, s.cand_j, s.count, s.cand_rev, s.cand_slot, g.reverse);
  check(cudaMemcpy(g.species, s.species, sizeof(int32_t) * s.n, cudaMemcpyDeviceToDevice), "species");
}

void conservative_forces(State &s, const Cutoff &c, const Graph &g, double energy_scale) {
  const bool adaptive = c.n_target > 0 && g.grad_cv;
  LAUNCH(edge_force_kernel, s.n, s.n, s.M, s.nkept, g.reverse, g.grad_ev, adaptive ? g.grad_cv : nullptr,
         energy_scale, s.f_cons, s.dEdr);
  if (!adaptive) return;
  LAUNCH(chain_kernel, s.n * s.K, s.n, s.K, s.count, s.cand_v, s.cand_d, c.cutoff, c.width, s.r, s.root,
         s.dn_root, s.clamped, s.dEdr, s.chain);
  LAUNCH(chain_gather_kernel, s.n, s.n, s.K, s.count, s.cand_rev, s.chain, energy_scale, s.f_cons);
}

void nc_forces(State &s, const Graph &g) { LAUNCH(nc_kernel, s.n, s.n, g.nc, s.nc_scale, s.f_nc); }

void kick(State &s, const double *f1, double a1, const double *f2, double a2, double dt) {
  LAUNCH(kick_kernel, s.n, s.n, s.v, s.inv_mass, f1, a1, f2, a2, dt);
}

void drift(State &s, double dt) { LAUNCH(drift_kernel, s.n, s.n, s.x, s.v, s.x_ref, dt, s.scalars + 2); }

void reset_scalars(State &s) { check(cudaMemset(s.scalars, 0, 3 * sizeof(int32_t)), "memset"); }

void synchronize() { check(cudaDeviceSynchronize(), "synchronize"); }

} // namespace mlipcpp::md::gpu
