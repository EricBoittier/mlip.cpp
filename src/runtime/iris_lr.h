#pragma once
// Host side of iris PET-LR (ipsum/PET-LR, after M. Langer; iris.pet.charges.fermi_level, a2658e3) for the
// in-graph LR branch of GraphModel (pet.iris_lr): the bisection of sum_i f(s_i - mu) = Q on the frozen scores
// (his stop_gradient), the rounded occupations, the gapped decision and the tie-split indicator weights of the
// extremes of s - m. The Newton step and the gapped midpoint are in the graph, so their derivatives are his.
// Transliterated from scripts/pet_lr_port/iris_torch.py bisect_fermi (checked against his fermi_level by
// scripts/pet_lr_port/fermi_synth.py).

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace mlipcpp::runtime::iris {

struct FermiConfig {
  int m_min = -3, m_max = 3;
  double temperature = 0.1;
  int iterations = 40;
  double eps = 1e-6;
};

inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

/// f(x) = m_min + sum_{c = m - 1/2, m = m_min + 1 .. m_max} sigmoid((x - c) / T)
inline double occupation(double x, const FermiConfig &c) {
  double q = c.m_min;
  for (int m = c.m_min + 1; m <= c.m_max; m++) q += sigmoid((x - (m - 0.5)) / c.temperature);
  return q;
}

/// f'(x)
inline double density(double x, const FermiConfig &c) {
  double d = 0;
  for (int m = c.m_min + 1; m <= c.m_max; m++) {
    const double s = sigmoid((x - (m - 0.5)) / c.temperature);
    d += s * (1.0 - s);
  }
  return d / c.temperature;
}

struct FermiHost {
  double mu = 0;               // bisected mu (the Newton step's base point)
  bool gapped = false;
  std::vector<float> m_round;  // rounded occupations at mu
  std::vector<float> top, bottom; // indicator weights of max / min of s - m, divided by their tie counts
};

inline FermiHost fermi_host(const std::vector<double> &s, double Q, const FermiConfig &c) {
  FermiHost h;
  const size_t n = s.size();
  h.m_round.assign(n, 0.f), h.top.assign(n, 0.f), h.bottom.assign(n, 0.f);
  if (n == 0) return h;
  const double margin = 1.0 + 40.0 * c.temperature;
  double lo = *std::min_element(s.begin(), s.end()) - c.m_max - margin;
  double hi = *std::max_element(s.begin(), s.end()) - c.m_min + margin;
  auto residual = [&](double mu) {
    double q = 0;
    for (double x : s) q += occupation(x - mu, c);
    return q - Q;
  };
  for (int it = 0; it < c.iterations; it++) {
    const double mid = 0.5 * (lo + hi);
    if (residual(mid) > 0) lo = mid; else hi = mid;
  }
  h.mu = 0.5 * (lo + hi);
  std::vector<double> d(n);
  double msum = 0;
  for (size_t i = 0; i < n; i++) {
    const double m = std::nearbyint(occupation(s[i] - h.mu, c)); // round half to even, as jnp.round
    h.m_round[i] = static_cast<float>(m);
    msum += m;
    d[i] = s[i] - m;
  }
  const double dmax = *std::max_element(d.begin(), d.end()), dmin = *std::min_element(d.begin(), d.end());
  int ntop = 0, nbot = 0;
  for (size_t i = 0; i < n; i++) ntop += d[i] == dmax, nbot += d[i] == dmin;
  for (size_t i = 0; i < n; i++) {
    h.top[i] = d[i] == dmax ? static_cast<float>(1.0 / ntop) : 0.f;
    h.bottom[i] = d[i] == dmin ? static_cast<float>(1.0 / nbot) : 0.f;
  }
  const double mid = 0.5 * (dmax + dmin);
  double dens = 0;
  for (double x : s) dens += density(x - mid, c);
  h.gapped = std::fabs(msum - Q) < 0.5 && dmax - dmin < 1.0 && dens < c.eps;
  return h;
}

/// His calculator's frozen Ewald parameters (iris.pet.calculator.Calculator._freeze, a2658e3, and jax-pme
/// batched_mixed to_lr / shrink_2d_cell, 1c26495), per structure as his oracle runs them: for a slab (two periodic
/// directions) the effective cell with the non-periodic vector shrunk to the slab thickness + margin + 1.5 L_max,
/// the wavelength from num_k on the effective cell, smearing 2 wavelength (or the checkpoint's), cutoff 8
/// wavelength, the halfspace k grid sized on the float32 effective cell. Transliterated from
/// scripts/pet_lr_port/torch_grid.py ewald_params (checked there against his JAX to 1e-14).
struct EwaldSetup {
  double cellc[3][3];            // the cell the Ewald maths uses: periodic rows raw, the slab's normal row effective (float32)
  double wavelength = 0, smearing = 0, cutoff = 0;
  bool slab = false;
  int slab_axis = -1;
  std::vector<std::array<int, 3>> k_grid; // halfspace integer k (lexicographic first of each +-k pair, k = 0 out)
};

inline EwaldSetup ewald_setup(const double cell[3][3], const bool pbc[3], const double *pos, int n, int num_k,
                              double smearing_cfg, double slab_margin) {
  EwaldSetup e;
  double eff[3][3];
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) eff[a][b] = cell[a][b];
  const int np = int(pbc[0]) + int(pbc[1]) + int(pbc[2]);
  if (np != 3 && np != 2) throw std::runtime_error("iris PET-LR: periodic in 3 or 2 directions only");
  auto norm = [](const double *v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); };
  if (np == 2) {
    const int k = !pbc[0] ? 0 : !pbc[1] ? 1 : 2;
    e.slab = true, e.slab_axis = k;
    const double *v1 = cell[(k + 1) % 3], *v2 = cell[(k + 2) % 3];
    double nh[3] = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
    const double nn = norm(nh);
    for (double &x : nh) x /= nn;
    double zmin = 1e300, zmax = -1e300;
    for (int i = 0; i < n; i++) {
      const double z = pos[3 * i] * nh[0] + pos[3 * i + 1] * nh[1] + pos[3 * i + 2] * nh[2];
      zmin = std::min(zmin, z), zmax = std::max(zmax, z);
    }
    const double thickness = (zmax - zmin) + slab_margin;
    const double h_cell = std::fabs(cell[k][0] * nh[0] + cell[k][1] * nh[1] + cell[k][2] * nh[2]);
    const double h_min = thickness + 1.5 * std::max(norm(v1), norm(v2));
    if (h_cell > h_min)
      for (int b = 0; b < 3; b++) eff[k][b] = cell[k][b] * (h_min / h_cell);
  }
  if (num_k <= 0) throw std::runtime_error("iris PET-LR: periodic structure but pet.iris_lr.num_k is not set");
  e.wavelength = std::cbrt(norm(eff[0]) * norm(eff[1]) * norm(eff[2]) / (2.0 * num_k));
  e.smearing = smearing_cfg > 0 ? smearing_cfg : 2.0 * e.wavelength;
  e.cutoff = 8.0 * e.wavelength;
  int ns[3];
  for (int a = 0; a < 3; a++) { // numpy on the float32 effective cell: norm and the division in float32
    const float x = float(eff[a][0]), y = float(eff[a][1]), z = float(eff[a][2]);
    const float l = std::sqrt(x * x + y * y + z * z);
    ns[a] = int(std::ceil(l / float(e.wavelength)));
  }
  auto freq = [](int m, int i) { return i < (m + 1) / 2 ? i : i - m; }; // np.fft.fftfreq(m) * m
  for (int i = 0; i < ns[0]; i++)
    for (int j = 0; j < ns[1]; j++)
      for (int l = 0; l < ns[2]; l++) {
        const int kx = freq(ns[0], i), ky = freq(ns[1], j), kz = freq(ns[2], l);
        if (kx > 0 || (kx == 0 && ky > 0) || (kx == 0 && ky == 0 && kz > 0)) e.k_grid.push_back({kx, ky, kz});
      }
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) e.cellc[a][b] = pbc[a] ? cell[a][b] : double(float(eff[a][b]));
  return e;
}

} // namespace mlipcpp::runtime::iris
