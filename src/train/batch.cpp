#include "train/batch.h"
#include "core/padding.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
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
  // a species whose forces are all zero (MPtrj's Ne and Ar: single-element fcc cells) has RMS 0,
  // and dividing the targets by it gives NaN; such species keep scale 1
  for (auto &[Z, sc2] : f2)
    sc.force_scale[Z] = sc2.second && sc2.first > 0.0 ? std::sqrt(sc2.first / sc2.second) : 1.0;
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

int Tiling::bucket(int x, int lo, double ratio) {
  int v = lo;
  while (v < x) v = std::max(v + 8, static_cast<int>(std::ceil(v * ratio / 8.0)) * 8);
  return v;
}

Batch make_batch(const MemmapSplit &split, const NeighborCache &nl,
                 const std::vector<int64_t> &ids,
                 const std::vector<std::array<float, 9>> &rotations,
                 const std::map<int, int> &species_to_index,
                 const TargetScaling &scaling, float cutoff, int pad_atoms,
                 int pad_neighbors, int pad_systems) {
  Batch b;
  b.n_real_systems = static_cast<int>(ids.size());
  b.ids.assign(ids.begin(), ids.end());
  int n_real = 0;
  for (int64_t s : ids) {
    n_real += split.n_atoms(s);
    b.max_neighbors = std::max(b.max_neighbors, nl.max_neighbors[s]);
  }
  b.max_neighbors = std::max({b.max_neighbors, 1, pad_neighbors});
  b.n_atoms = std::max(n_real, pad_atoms);
  b.n_systems = std::max(b.n_real_systems, pad_systems);
  const int n = b.n_atoms, m = b.max_neighbors;

  auto index_of = [&](int Z) {
    auto it = species_to_index.find(Z);
    if (it == species_to_index.end())
      throw std::runtime_error("atomic number " + std::to_string(Z) + " not in the model");
    return it->second;
  };

  b.neighbor_species.assign(n * m, padding::kSpecies);
  // padded slots: core/padding.h (an ordinary edge at the cutoff, as metatrain)
  b.edge_vectors.resize(static_cast<size_t>(n) * m * 3);
  for (int f = 0; f < n * m; f++) padding::edge(cutoff, &b.edge_vectors[3 * static_cast<size_t>(f)]);
  b.padding_mask.assign(n * m, 1.0f);
  b.reverse_neighbor_index.assign(n * m, 0);
  b.cutoff_values.assign(n * m, cutoff);
  b.atom_weight.assign(static_cast<size_t>(b.n_systems) * n, 0.0f);
  b.energy_target.assign(b.n_systems, 0.0f);
  b.force_target.assign(n * 3, 0.0f);
  b.force_scale.assign(n, 0.0f);
  b.force_mask.assign(n, 0.0f);
  b.n_per_system.assign(b.n_systems, 0);
  b.species.assign(n, 0);
  // padding atoms: no neighbours, so every slot is padded and points to itself
  for (int a = n_real; a < n; a++)
    for (int j = 0; j < m; j++) b.reverse_neighbor_index[a * m + j] = a * m + j;

  int offset = 0;
  for (int k = 0; k < b.n_real_systems; k++) {
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
      b.force_mask[a] = 1.0f;
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
      if (!nl.pair_cutoff.empty()) b.cutoff_values[flat] = nl.pair_cutoff[e];
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
                                     double growth, const Tiling &tiling) {
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
    int pad_atoms = 0, pad_nb = 0;
    if (tiling.enabled()) {
      int atoms = 0, nb = 1;
      for (int64_t s : g_ids) atoms += split.n_atoms(s), nb = std::max(nb, nl.max_neighbors[s]);
      pad_atoms = Tiling::bucket(atoms, 32, tiling.atom_ratio);
      pad_nb = Tiling::bucket(nb, 8, tiling.nb_ratio);
    }
    out.push_back(make_batch(split, nl, g_ids, g_rot, species_to_index, scaling, cutoff, pad_atoms,
                             pad_nb, tiling.enabled() ? tiling.systems : 0));
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

std::vector<std::vector<Batch>> make_step_batches(const MemmapSplit &split, const NeighborCache &nl,
                                                  const std::vector<int64_t> &ids,
                                                  const std::vector<std::array<float, 9>> &rotations,
                                                  const std::map<int, int> &species_to_index,
                                                  const TargetScaling &scaling, float cutoff,
                                                  int n_workers, const CostModel &cost,
                                                  double budget,
                                                  const Tiling &tiling) {
  const size_t K = ids.size();
  const int W = std::max(1, n_workers);
  std::vector<size_t> order(K);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return nl.max_neighbors[ids[a]] < nl.max_neighbors[ids[b]];
  });
  std::vector<int64_t> atoms(K + 1, 0); // prefix sums in sorted order
  for (size_t k = 0; k < K; k++) atoms[k + 1] = atoms[k] + split.n_atoms(ids[order[k]]);
  // group = sorted structures [i, j): its largest neighbour count is the last one's
  auto nb = [&](size_t j) { return std::max(1, nl.max_neighbors[ids[order[j - 1]]]); };
  auto group_cost = [&](size_t i, size_t j) {
    double n = static_cast<double>(atoms[j] - atoms[i]), m = nb(j);
    if (tiling.enabled())
      n = Tiling::bucket(static_cast<int>(n), 32, tiling.atom_ratio), m = Tiling::bucket(nb(j), 8, tiling.nb_ratio);
    return cost(n, m);
  };

  using Groups = std::vector<std::pair<size_t, size_t>>;
  auto plan = [&](double cap) -> Groups {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> best(K + 1, inf);
    std::vector<size_t> cut(K + 1, 0);
    best[0] = 0;
    for (size_t j = 1; j <= K; j++) {
      const double m1 = nb(j) + 1.0;
      for (size_t i = j; i-- > 0;) {
        const int64_t n = atoms[j] - atoms[i];
        const double c = group_cost(i, j);
        // a single structure is always allowed; groups grow monotonically in n and cost
        if (j - i > 1 && (n * m1 * m1 > budget || c > cap)) break;
        // never a single-atom micro-batch (see make_microbatches)
        if (n >= 2 && best[i] + c < best[j]) best[j] = best[i] + c, cut[j] = i;
      }
    }
    if (!std::isfinite(best[K])) return {};
    Groups g;
    for (size_t j = K; j > 0; j = cut[j]) g.emplace_back(cut[j], j);
    return g;
  };
  // longest first onto the least loaded worker; returns the makespan
  auto deal = [&](const Groups &groups, std::vector<int> &owner) {
    std::vector<std::pair<double, size_t>> by_cost;
    for (size_t g = 0; g < groups.size(); g++) by_cost.emplace_back(group_cost(groups[g].first, groups[g].second), g);
    std::sort(by_cost.rbegin(), by_cost.rend());
    std::vector<double> load(W, 0.0);
    owner.assign(groups.size(), 0);
    for (auto [c, g] : by_cost) {
      const int w = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
      owner[g] = w, load[w] += c;
    }
    return *std::max_element(load.begin(), load.end());
  };

  // the cap on a micro-batch's cost trades balance (small caps) against the
  // fixed cost of each extra micro-batch: keep the plan that ends first
  Groups groups = plan(std::numeric_limits<double>::infinity());
  if (groups.empty()) groups = {{0, K}};
  std::vector<int> owner;
  double best = deal(groups, owner);
  if (W > 1) {
    double total = 0;
    for (auto [i, j] : groups) total += group_cost(i, j);
    for (double f : {1.0, 1.05, 1.15, 1.3, 1.5, 1.75}) {
      Groups g = plan(f * total / W);
      std::vector<int> own;
      double span;
      if (!g.empty() && (span = deal(g, own)) < best) best = span, groups = std::move(g), owner = std::move(own);
    }
  }

  std::vector<std::future<Batch>> built;
  for (const auto &grp : groups)
    built.push_back(std::async(std::launch::async, [&, i = grp.first, j = grp.second] {
      std::vector<int64_t> g_ids;
      std::vector<std::array<float, 9>> g_rot;
      for (size_t k = i; k < j; k++) {
        g_ids.push_back(ids[order[k]]);
        if (!rotations.empty()) g_rot.push_back(rotations[order[k]]);
      }
      int pad_atoms = 0, pad_nb = 0;
      if (tiling.enabled())
        pad_atoms = Tiling::bucket(static_cast<int>(atoms[j] - atoms[i]), 32, tiling.atom_ratio),
        pad_nb = Tiling::bucket(nb(j), 8, tiling.nb_ratio);
      Batch b = make_batch(split, nl, g_ids, g_rot, species_to_index, scaling, cutoff, pad_atoms,
                           pad_nb, tiling.enabled() ? tiling.systems : 0);
      b.energy_norm = static_cast<float>(W) / K;
      b.force_norm = static_cast<float>(W) / (3.0f * atoms[K]);
      return b;
    }));
  std::vector<std::vector<Batch>> out(W);
  for (size_t g = 0; g < groups.size(); g++) out[owner[g]].push_back(built[g].get());
  return out;
}

} // namespace mlipcpp::train
