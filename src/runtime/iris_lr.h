#pragma once
// Host side of iris PET-LR (ipsum/PET-LR, after M. Langer; iris.pet.charges.fermi_level, a2658e3) for the
// in-graph LR branch of GraphModel (pet.iris_lr): the bisection of sum_i f(s_i - mu) = Q on the frozen scores
// (his stop_gradient), the rounded occupations, the gapped decision and the tie-split indicator weights of the
// extremes of s - m. The Newton step and the gapped midpoint are in the graph, so their derivatives are his.
// Transliterated from scripts/pet_lr_port/iris_torch.py bisect_fermi (checked against his fermi_level by
// scripts/pet_lr_port/fermi_synth.py).

#include <algorithm>
#include <cmath>
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

} // namespace mlipcpp::runtime::iris
