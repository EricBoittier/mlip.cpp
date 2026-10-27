#pragma once
// PET's adaptive cutoff, "solver" and "grid" methods (the grid: see grid_solve
// below). "solver" method (metatrain
// pet/modules/adaptive_cutoff.py, get_adaptive_cutoffs_solver), shared by
// training, inference and the CUDA MD kernels so all three agree bit for bit
// on the host.
//
// Per atom i, with the smoothed neighbour count
//   n_i(r) = sum_k bump(d_ik; r, w) + N (r / cutoff)^3,
// 10 Newton-bisection steps (exact 0/1 outside the taper) find r with
// n_i(r) = N, then the IFT step r - (n_i(r) - N) / n_i'(r) (bump clamped
// everywhere) is clamped to [cutoff/16, cutoff]. Its gradient, with r and
// n_i'(r) held constant as in metatrain, is
//   d r_i / d d_ik = bump_r(d_ik; r) / n_i'(r)
// (zero when the clamp is active or d_ik is outside the taper).

#include <cmath>
#include <vector>

#if defined(__CUDACC__)
#define MLIPCPP_HD __host__ __device__
#else
#define MLIPCPP_HD
#endif

namespace mlipcpp::adaptive {

constexpr double PI = 3.14159265358979323846;

struct Radius {
  double r = 0;       // adapted cutoff
  double root = 0;    // Newton-bisection root, before the IFT step
  double dn_root = 1; // max(n'(root), 1e-6): the IFT denominator
  bool clamped = false;
};

/// No reduction: one thread visits all of an atom's edges.
struct Serial {
  MLIPCPP_HD void operator()(double &, double &) const {}
};

/// `dist(k)` for k in [0, n_edges): the distances of atom i's edges (any
/// distance >= cutoff contributes exactly nothing). Cooperating threads (a GPU
/// warp) each visit k = k0, k0 + stride, ...; `sum` reduces the partial counts
/// to the same totals on every one of them.
template <class Dist, class Sum = Serial>
MLIPCPP_HD inline Radius solve(int n_edges, Dist dist, double cutoff, double n_target, double width,
                               int k0 = 0, int stride = 1, Sum sum = {}) {
  const double N = n_target, w = width, inv = 1.0 / cutoff;
  // (n_total, dn/dr) at r; `loop`: the solver's exact 0/1 outside the taper,
  // else cutoff_func_bump (clamped) everywhere, as in the IFT residual
  auto count = [&](double r, bool loop, double &cnt, double &dcnt) {
    cnt = 0, dcnt = 0;
    for (int k = k0; k < n_edges; k += stride) {
      const double sc = (dist(k) - (r - w)) / w;
      const double s = PI * fmin(fmax(sc, 1e-6), 1.0 - 1e-6);
      const double sn = sin(s), th = tanh(cos(s) / sn);
      if (!loop) cnt += 0.5 * (1.0 + th);
      else if (sc > 0.0 && sc < 1.0) cnt += 0.5 * (1.0 + th), dcnt += 0.5 * PI / w * (1.0 - th * th) / (sn * sn);
      else if (sc <= 0.0) cnt += 1.0;
    }
    sum(cnt, dcnt);
    const double x = r * inv;
    cnt += N * x * x * x;
    dcnt += 3.0 * N * x * x * inv;
  };
  double lo = 0.0, hi = cutoff, r = 0.5 * cutoff, cnt, dcnt;
  for (int it = 0; it < 10; it++) {
    count(r, true, cnt, dcnt);
    const double f = cnt - N;
    if (f <= 0) lo = r;
    else hi = r;
    const double rn = r - f / fmax(dcnt, 1e-6);
    r = rn >= lo && rn <= hi ? rn : 0.5 * (lo + hi);
  }
  double dn_root, residual, unused;
  count(r, true, unused, dn_root);
  count(r, false, residual, unused);
  residual -= N;
  Radius out;
  out.root = r;
  out.dn_root = fmax(dn_root, 1e-6);
  const double raw = r - residual / out.dn_root, lo_c = cutoff / 16.0;
  out.clamped = !(raw > lo_c && raw < cutoff);
  out.r = fmin(fmax(raw, lo_c), cutoff);
  return out;
}

/// d r_i / d d_ik for one edge of atom i (see the header comment): the
/// residual's bump is evaluated at the root, and is flat where its clamp acts.
MLIPCPP_HD inline double ddist(double d, const Radius &a, double width) {
  const double sc = (d - (a.root - width)) / width;
  if (a.clamped || !(sc > 1e-6 && sc < 1.0 - 1e-6)) return 0.0;
  const double s = PI * sc, sn = sin(s), th = tanh(cos(s) / sn);
  return 0.5 * PI / width * (1.0 - th * th) / (sn * sn) / a.dn_root;
}

// ---- "grid" (metatrain get_adaptive_cutoffs_grid, the method of PET-OMol and
// the older PET checkpoints) ----
//
// Probes p_q = 0.5 + q h, h = w / 4, q in [0, P), P = ceil((cutoff - 0.5) / h)
// (torch.arange). Per atom i, with the clamped bump of width w,
//   D_q = sum_k bump(d_ik; p_q, w) - N + N (q / (P - 1))^3,
//   g = torch.gradient(D) (centred inside, one-sided at the ends),
//   s_q = max(|g_q|, 1e-12), l_q = -(D_q / s_q)^2 / 2,
//   r_i = sum_q p_q softmax(l)_q.
// metatrain differentiates through all of it (positions -> d_ik -> D -> s, l
// -> r_i), and so does this: X_q = d r_i / d D_q (through l_q directly and
// through the widths s of the neighbouring probes), and
//   d r_i / d d_ik = sum_q X_q dbump(d_ik; p_q, w) / dd.
// (metatrain's softmax shift is the largest logit of the whole batch, here of
// the atom: the same weights, and the shift carries no gradient.)

constexpr double GRID_MIN_PROBE = 0.5;
constexpr double GRID_EPS = 1e-12;

/// the number of probes (torch.arange(0.5, cutoff, width / 4))
MLIPCPP_HD inline int grid_probes(double cutoff, double width) {
  const double n = ceil((cutoff - GRID_MIN_PROBE) / (0.25 * width));
  return n > 0 ? static_cast<int>(n) : 0;
}

/// metatrain cutoff_func_bump (clamped to [1e-6, 1 - 1e-6], where it is exactly 1 / 0)
MLIPCPP_HD inline double grid_bump(double d, double probe, double w) {
  const double sc = (d - (probe - w)) / w;
  if (sc <= 1e-6) return 1.0;
  if (sc >= 1.0 - 1e-6) return 0.0;
  const double s = PI * sc;
  return 0.5 * (1.0 + tanh(cos(s) / sin(s)));
}

/// d bump / d d (zero where the clamp acts)
MLIPCPP_HD inline double grid_dbump(double d, double probe, double w) {
  const double sc = (d - (probe - w)) / w;
  if (!(sc > 1e-6 && sc < 1.0 - 1e-6)) return 0.0;
  const double s = PI * sc, sn = sin(s), th = tanh(cos(s) / sn);
  return -0.5 * PI / w * (1.0 - th * th) / (sn * sn);
}

/// Host reductions (one thread): nothing to combine.
struct GridSerial {
  MLIPCPP_HD void max(double &) const {}
  MLIPCPP_HD void sum(double &, double &) const {}
  MLIPCPP_HD void sync() const {}
};

/// Atom i's grid radius. `dist(k)`, k in [0, n_edges): its edge distances
/// (>= cutoff: no contribution); `buf`: 3 P doubles of scratch; `jac(k, J)`
/// receives d r_i / d d_ik for every edge (0 when it does not move r_i).
/// Cooperating threads (a GPU warp) take probes and edges q = k0, k0 + stride,
/// ... and share buf; `red` reduces across them (max, sum) and syncs the
/// buffer. P == 0 (cutoff <= 0.5): the cutoff, no derivative.
template <class Dist, class Jac, class Red = GridSerial>
MLIPCPP_HD inline double grid_solve(int n_edges, Dist dist, double cutoff, double n_target, double width,
                                    double *buf, Jac jac, int k0 = 0, int stride = 1, Red red = {}) {
  const int P = grid_probes(cutoff, width);
  const double N = n_target, w = width, h = 0.25 * width;
  if (P == 0) {
    for (int k = k0; k < n_edges; k += stride) jac(k, 0.0);
    return cutoff;
  }
  double *D = buf, *T = buf + P, *X = buf + 2 * P;
  auto probe = [&](int q) { return GRID_MIN_PROBE + q * h; };
  // 1. D_q
  for (int q = k0; q < P; q += stride) {
    double c = 0;
    const double pq = probe(q);
    for (int k = 0; k < n_edges; k++) c += grid_bump(dist(k), pq, w);
    const double x = P > 1 ? double(q) / double(P - 1) : 0.0;
    D[q] = c - N + N * x * x * x;
  }
  red.sync();
  // d s_q / d (the D it is made of): the gradient's sign (s = |g| clamped), or
  // for a single probe s = |D| / 2 + eps
  auto width_of = [&](int q, double &s, double &ds) {
    if (P == 1) {
      s = 0.5 * fabs(D[0]) + GRID_EPS;
      ds = D[0] > 0 ? 0.5 : D[0] < 0 ? -0.5 : 0.0;
      return;
    }
    const double g = q == 0 ? D[1] - D[0] : q == P - 1 ? D[P - 1] - D[P - 2] : 0.5 * (D[q + 1] - D[q - 1]);
    const double a = fabs(g);
    s = a >= GRID_EPS ? a : GRID_EPS;
    ds = a >= GRID_EPS ? (g > 0 ? 1.0 : g < 0 ? -1.0 : 0.0) : 0.0;
  };
  auto logit = [&](int q, double &s, double &ds) {
    width_of(q, s, ds);
    const double z = D[q] / s;
    return -0.5 * z * z;
  };
  // 2. softmax
  double lmax = -1e300, s, ds;
  for (int q = k0; q < P; q += stride) lmax = fmax(lmax, logit(q, s, ds));
  red.max(lmax);
  double S = 0, R = 0;
  for (int q = k0; q < P; q += stride) {
    const double e = exp(logit(q, s, ds) - lmax);
    S += e, R += e * probe(q);
  }
  red.sum(S, R);
  const double r = R / S;
  // 3. X_q = d r / d D_q: dr/dl_q = w_q (p_q - r); l_q through D_q directly,
  // and through s_t (T_t = dr/ds_t ds_t/dg_t) for the probes t whose gradient
  // g_t holds D_q
  for (int q = k0; q < P; q += stride) {
    const double l = logit(q, s, ds);
    const double c = exp(l - lmax) / S * (probe(q) - r);
    X[q] = -c * D[q] / (s * s);
    T[q] = c * D[q] * D[q] / (s * s * s) * ds;
  }
  red.sync();
  for (int q = k0; q < P; q += stride) {
    double x = X[q];
    if (P == 1) {
      x += T[0];
    } else {
      if (q == 0) x -= T[0];
      if (q == P - 1) x += T[P - 1];
      if (q > 0) x += q - 1 == 0 ? T[0] : 0.5 * T[q - 1];          // t = q - 1 < P - 1 always
      if (q < P - 1) x -= q + 1 == P - 1 ? T[P - 1] : 0.5 * T[q + 1]; // t = q + 1 > 0 always
    }
    X[q] = x;
  }
  red.sync();
  // 4. d r / d d_k: the probes whose taper holds d_k, (d - w, d) around p_q - w
  for (int k = k0; k < n_edges; k += stride) {
    const double d = dist(k);
    double J = 0;
    if (d < cutoff) {
      int q0 = static_cast<int>(floor((d - GRID_MIN_PROBE) / h)), q1 = static_cast<int>(ceil((d + w - GRID_MIN_PROBE) / h));
      q0 = q0 < 0 ? 0 : q0, q1 = q1 > P - 1 ? P - 1 : q1;
      for (int q = q0; q <= q1; q++) J += X[q] * grid_dbump(d, probe(q), w);
    }
    jac(k, J);
  }
  return r;
}

/// Either method on the host: atom i's radius, and (if jac) d r_i / d d_ik per edge.
template <class Dist>
inline double radius(bool grid, int n_edges, Dist dist, double cutoff, double n_target, double width,
                     double *jac = nullptr) {
  if (grid) {
    thread_local std::vector<double> buf;
    buf.resize(3 * size_t(grid_probes(cutoff, width)) + 1);
    return grid_solve(n_edges, dist, cutoff, n_target, width, buf.data(), [&](int k, double J) {
      if (jac) jac[k] = J;
    });
  }
  const Radius a = solve(n_edges, dist, cutoff, n_target, width);
  if (jac)
    for (int k = 0; k < n_edges; k++) jac[k] = ddist(dist(k), a, width);
  return a.r;
}

} // namespace mlipcpp::adaptive
