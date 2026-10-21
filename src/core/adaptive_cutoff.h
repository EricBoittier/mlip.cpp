#pragma once
// PET's adaptive cutoff, "solver" method (metatrain
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

} // namespace mlipcpp::adaptive
