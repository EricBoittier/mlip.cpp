#include "train/batch.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mlipcpp::train {

double TargetScaling::comp(const int32_t *Z, int n) const {
  double e = 0.0;
  for (int i = 0; i < n; i++) {
    auto it = composition.find(Z[i]);
    if (it != composition.end()) e += it->second;
  }
  return e;
}

TargetScaling TargetScaling::fit(const MemmapSplit &train) {
  // composition: least squares E ≈ Σ_Z count_Z c_Z, normal equations in
  // float64 with metatrain's 1e-14 relative ridge, solved by Cholesky
  std::map<int, int> col;
  for (int64_t a = 0; a < train.total_atoms(); a++) col.emplace(train.numbers(0)[a], 0);
  int T = 0;
  std::vector<int> types;
  for (auto &[Z, c] : col) c = T++, types.push_back(Z);

  std::vector<double> XtX(T * T, 0.0), Xty(T, 0.0), x(T);
  for (int64_t s = 0; s < train.size(); s++) {
    std::fill(x.begin(), x.end(), 0.0);
    const int32_t *Z = train.numbers(s);
    for (int i = 0; i < train.n_atoms(s); i++) x[col[Z[i]]] += 1.0;
    for (int a = 0; a < T; a++) {
      if (x[a] == 0.0) continue;
      Xty[a] += x[a] * train.energy(s);
      for (int b = 0; b < T; b++) XtX[a * T + b] += x[a] * x[b];
    }
  }
  double mean_diag = 0.0;
  for (int a = 0; a < T; a++) mean_diag += XtX[a * T + a] / T;
  for (int a = 0; a < T; a++) XtX[a * T + a] += 1e-14 * mean_diag;

  std::vector<double> L(T * T, 0.0); // XtX = L L^T
  for (int i = 0; i < T; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = XtX[i * T + j];
      for (int k = 0; k < j; k++) sum -= L[i * T + k] * L[j * T + k];
      if (i == j) {
        if (sum <= 0.0) throw std::runtime_error("composition fit: matrix not positive definite");
        L[i * T + i] = std::sqrt(sum);
      } else {
        L[i * T + j] = sum / L[j * T + j];
      }
    }
  }
  std::vector<double> y(T), w(T);
  for (int i = 0; i < T; i++) {
    double sum = Xty[i];
    for (int k = 0; k < i; k++) sum -= L[i * T + k] * y[k];
    y[i] = sum / L[i * T + i];
  }
  for (int i = T - 1; i >= 0; i--) {
    double sum = y[i];
    for (int k = i + 1; k < T; k++) sum -= L[k * T + i] * w[k];
    w[i] = sum / L[i * T + i];
  }

  TargetScaling sc;
  for (int a = 0; a < T; a++) sc.composition[types[a]] = w[a];

  // scales: uncentred RMS of the per-atom residual energy, and of the force
  // components of each species
  double e2 = 0.0;
  std::map<int, std::pair<double, int64_t>> f2;
  for (int64_t s = 0; s < train.size(); s++) {
    const int n = train.n_atoms(s);
    const int32_t *Z = train.numbers(s);
    const double r = (train.energy(s) - sc.comp(Z, n)) / n;
    e2 += r * r;
    const float *F = train.forces(s);
    for (int i = 0; i < n; i++) {
      auto &[sum, count] = f2[Z[i]];
      for (int k = 0; k < 3; k++) sum += double(F[3 * i + k]) * F[3 * i + k];
      count += 3;
    }
  }
  sc.energy_scale = std::sqrt(e2 / train.size());
  for (auto &[Z, sc2] : f2) sc.force_scale[Z] = sc2.second ? std::sqrt(sc2.first / sc2.second) : 1.0;
  return sc;
}

std::array<float, 9> random_o3(std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> U(0.0, 1.0);
  const double u1 = U(rng), u2 = U(rng), u3 = U(rng);
  const double tau = 2.0 * std::numbers::pi;
  // uniform unit quaternion (Shoemake)
  const double a = std::sqrt(1 - u1) * std::sin(tau * u2), b = std::sqrt(1 - u1) * std::cos(tau * u2),
               c = std::sqrt(u1) * std::sin(tau * u3), d = std::sqrt(u1) * std::cos(tau * u3);
  const double sign = U(rng) < 0.5 ? -1.0 : 1.0; // inversion
  const double R[9] = {1 - 2 * (c * c + d * d), 2 * (b * c - a * d),     2 * (b * d + a * c),
                       2 * (b * c + a * d),     1 - 2 * (b * b + d * d), 2 * (c * d - a * b),
                       2 * (b * d - a * c),     2 * (c * d + a * b),     1 - 2 * (b * b + c * c)};
  std::array<float, 9> out;
  for (int k = 0; k < 9; k++) out[k] = static_cast<float>(sign * R[k]);
  return out;
}

Batch make_batch(const MemmapSplit &split, const NeighborCache &nl,
                 const std::vector<int64_t> &ids,
                 const std::vector<std::array<float, 9>> &rotations,
                 const std::map<int, int> &species_to_index,
                 const TargetScaling &scaling, float cutoff) {
  Batch b;
  b.n_systems = static_cast<int>(ids.size());
  for (int64_t s : ids) {
    b.n_atoms += split.n_atoms(s);
    b.max_neighbors = std::max(b.max_neighbors, nl.max_neighbors[s]);
  }
  b.max_neighbors = std::max(b.max_neighbors, 1);
  const int n = b.n_atoms, m = b.max_neighbors;

  auto index_of = [&](int Z) {
    auto it = species_to_index.find(Z);
    if (it == species_to_index.end())
      throw std::runtime_error("atomic number " + std::to_string(Z) + " not in the model");
    return it->second;
  };

  b.species.resize(n);
  b.neighbor_species.assign(n * m, 0);
  b.edge_vectors.assign(n * m * 3, 0.0f);
  b.padding_mask.assign(n * m, 1.0f);
  b.reverse_neighbor_index.assign(n * m, 0);
  b.cutoff_values.assign(n * m, cutoff);
  b.atom_weight.assign(static_cast<size_t>(b.n_systems) * n, 0.0f);
  b.energy_target.resize(b.n_systems);
  b.force_target.resize(n * 3);
  b.force_scale.resize(n);
  b.n_per_system.resize(b.n_systems);

  int offset = 0;
  for (int k = 0; k < b.n_systems; k++) {
    const int64_t s = ids[k];
    const int ns = split.n_atoms(s);
    const int32_t *Z = split.numbers(s);
    const float *F = split.forces(s);
    const float *R = rotations.empty() ? nullptr : rotations[k].data();
    auto rotate = [R](const float *v, float *out) {
      if (!R) return void(std::copy(v, v + 3, out));
      for (int r = 0; r < 3; r++) out[r] = R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2];
    };

    for (int i = 0; i < ns; i++) {
      const int a = offset + i;
      b.species[a] = index_of(Z[i]);
      b.atom_weight[static_cast<size_t>(k) * n + a] = 1.0f / ns;
      const double fs = scaling.fscale(Z[i]);
      b.force_scale[a] = static_cast<float>(fs);
      float f[3];
      rotate(F + 3 * i, f);
      for (int r = 0; r < 3; r++) b.force_target[3 * a + r] = static_cast<float>(f[r] / fs);
    }
    b.energy_target[k] = static_cast<float>(
        (split.energy(s) - scaling.comp(Z, ns)) / (scaling.energy_scale * ns));
    b.n_per_system[k] = ns;

    const int64_t e0 = nl.edge_offset[s], e1 = nl.edge_offset[s + 1];
    for (int64_t e = e0; e < e1; e++) {
      const int flat = (offset + nl.center[e]) * m + nl.slot[e];
      b.neighbor_species[flat] = index_of(Z[nl.neighbor[e]]);
      rotate(nl.vector[e].data(), &b.edge_vectors[3 * flat]);
      b.padding_mask[flat] = 0.0f;
      const int32_t r = nl.reverse[e];
      b.reverse_neighbor_index[flat] =
          r >= 0 ? (offset + nl.center[e0 + r]) * m + nl.slot[e0 + r] : flat;
    }
    offset += ns;
  }
  return b;
}

std::vector<Batch> make_microbatches(const MemmapSplit &split, const NeighborCache &nl,
                                     const std::vector<int64_t> &ids,
                                     const std::vector<std::array<float, 9>> &rotations,
                                     const std::map<int, int> &species_to_index,
                                     const TargetScaling &scaling, float cutoff, double budget,
                                     double growth) {
  std::vector<size_t> order(ids.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return nl.max_neighbors[ids[a]] < nl.max_neighbors[ids[b]];
  });
  int64_t n_total = 0;
  for (int64_t s : ids) n_total += split.n_atoms(s);

  std::vector<Batch> out;
  std::vector<int64_t> g_ids;
  std::vector<std::array<float, 9>> g_rot;
  int64_t g_atoms = 0;
  double g_min = 0;
  auto flush = [&] {
    if (g_ids.empty()) return;
    out.push_back(make_batch(split, nl, g_ids, g_rot, species_to_index, scaling, cutoff));
    out.back().energy_norm = 1.0f / ids.size();
    out.back().force_norm = 1.0f / (3.0f * n_total);
    g_ids.clear(), g_rot.clear(), g_atoms = 0;
  };
  for (size_t k : order) {
    const int64_t s = ids[k];
    const double m1 = nl.max_neighbors[s] + 1.0; // sorted: the group's largest so far
    const int64_t atoms = g_atoms + split.n_atoms(s);
    // never close a single-atom micro-batch: a leading PyTorch dim of 1 is
    // invisible to ggml_n_dims, which some interpreter ops rely on
    if (g_atoms >= 2 && (m1 > growth * g_min || atoms * m1 * m1 > budget)) flush();
    if (g_ids.empty()) g_min = m1;
    g_ids.push_back(s);
    if (!rotations.empty()) g_rot.push_back(rotations[k]);
    g_atoms += split.n_atoms(s);
  }
  flush();
  return out;
}

} // namespace mlipcpp::train
