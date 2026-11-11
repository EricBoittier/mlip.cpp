#include "train/batch.h"
#include "core/padding.h"

#include <algorithm>
#include <atomic>
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

std::vector<std::vector<int32_t>> StatePlanner::groups(int n, const float *x) const {
  if (!lrc || !lrc->states || n < 2) return {};
  const std::vector<double> xd(x, x + 3 * size_t(n));
  const runtime::LrcClusters cl = runtime::lrc_clusters(*lrc, n, xd.data());
  const runtime::LrcStates st = runtime::lrc_states(*lrc, cl);
  std::vector<std::vector<int32_t>> out;
  for (size_t v = 0; v < st.copies.size(); v++) out.push_back(st.drop_groups(cl, v));
  return out;
}

int StatePlanner::count(const MemmapSplit &split, int64_t s, const float *R) const {
  if (!lrc || !lrc->states) return 1;
  const int n = split.n_atoms(s);
  const float *X = split.positions(s), *C = split.cell(s);
  for (int r = 0; r < 3; r++)
    if (C[3 * r] != 0 || C[3 * r + 1] != 0 || C[3 * r + 2] != 0) return 1; // open structures only
  std::vector<float> x(3 * size_t(n));
  for (int i = 0; i < n; i++)
    for (int r = 0; r < 3; r++)
      x[3 * i + r] = R ? R[3 * r] * X[3 * i] + R[3 * r + 1] * X[3 * i + 1] + R[3 * r + 2] * X[3 * i + 2] : X[3 * i + r];
  return std::max<int>(1, static_cast<int>(groups(n, x.data()).size()));
}

namespace {
Batch make_batch_impl(const MemmapSplit &split, const NeighborCache &nl, const std::vector<int64_t> &ids,
                      const std::vector<std::array<float, 9>> &rotations, const std::map<int, int> &species_to_index,
                      const TargetScaling &scaling, float cutoff, int pad_atoms, int pad_neighbors, int pad_systems,
                      const StatePlanner *planner, const std::vector<int> *part);
} // namespace

Batch make_batch(const MemmapSplit &split, const NeighborCache &nl,
                 const std::vector<int64_t> &ids,
                 const std::vector<std::array<float, 9>> &rotations,
                 const std::map<int, int> &species_to_index,
                 const TargetScaling &scaling, float cutoff, int pad_atoms,
                 int pad_neighbors, int pad_systems, const StatePlanner *planner) {
  return make_batch_impl(split, nl, ids, rotations, species_to_index, scaling, cutoff, pad_atoms, pad_neighbors,
                         pad_systems, planner, nullptr);
}

Batch make_part_batch(const MemmapSplit &split, const NeighborCache &nl, int64_t id, const float *R,
                      const std::vector<int> &copies, const std::map<int, int> &species_to_index,
                      const TargetScaling &scaling, float cutoff, const StatePlanner &planner) {
  std::vector<std::array<float, 9>> rot;
  if (R) rot.push_back({}), std::copy(R, R + 9, rot[0].begin());
  return make_batch_impl(split, nl, {id}, rot, species_to_index, scaling, cutoff, 0, 0, 0, &planner, &copies);
}

int part_copies(int c, int n, int m, double budget) {
  static const int forced = [] { const char *e = std::getenv("MLIPCPP_VS_SPLIT_COPIES"); return e && *e ? std::atoi(e) : 0; }();
  if (c <= 1) return c;
  if (forced > 0) return std::min(c, forced);
  const double one = double(n) * (m + 1.0) * (m + 1.0);
  if (c * one <= budget) return c;
  return std::max(1, static_cast<int>(budget / one));
}

namespace {
Batch make_batch_impl(const MemmapSplit &split, const NeighborCache &nl, const std::vector<int64_t> &ids,
                      const std::vector<std::array<float, 9>> &rotations, const std::map<int, int> &species_to_index,
                      const TargetScaling &scaling, float cutoff, int pad_atoms, int pad_neighbors, int pad_systems,
                      const StatePlanner *planner, const std::vector<int> *part) {
  Batch b;
  if (part && (ids.size() != 1 || part->empty())) throw std::logic_error("make_part_batch: one structure, some copies");
  const int K = static_cast<int>(ids.size());
  // rotated positions of every structure; lrc_states: the evaluations of each from them (copies after all structures)
  std::vector<std::vector<float>> xr(K);
  std::vector<std::vector<std::vector<int32_t>>> plan(K);
  int n_copies = 0, copy_atoms = 0;
  for (int k = 0; k < K; k++) {
    const int64_t s = ids[k];
    const int ns = split.n_atoms(s);
    const float *R = rotations.empty() ? nullptr : rotations[k].data();
    const float *X = split.positions(s);
    xr[k].resize(3 * size_t(ns));
    for (int i = 0; i < ns; i++) {
      const float *v = X + 3 * i;
      float *out = &xr[k][3 * size_t(i)];
      if (!R) std::copy(v, v + 3, out);
      else
        for (int r = 0; r < 3; r++) out[r] = R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2];
    }
    if (planner && planner->lrc && planner->lrc->states) {
      const float *C = split.cell(s);
      bool open = true;
      for (int r = 0; r < 3; r++) open = open && C[3 * r] == 0 && C[3 * r + 1] == 0 && C[3 * r + 2] == 0;
      if (open) plan[k] = planner->groups(ns, xr[k].data());
      if (plan[k].size() > 1) n_copies += int(plan[k].size()) - 1, copy_atoms += (int(plan[k].size()) - 1) * ns;
    }
  }
  const bool states = planner && planner->lrc && planner->lrc->states;
  if (part) { // the listed copies of the one structure only
    if (plan[0].size() < 2) throw std::logic_error("make_part_batch: not a multi-state structure");
    for (int v : *part)
      if (v < 0 || size_t(v) >= plan[0].size()) throw std::logic_error("make_part_batch: copy index");
    n_copies = int(part->size()) - 1, copy_atoms = n_copies * split.n_atoms(ids[0]);
  }
  b.n_real_systems = K + n_copies;
  b.n_primary = states ? K : 0;
  b.ids.assign(ids.begin(), ids.end());
  if (part) {
    for (int c = 0; c < n_copies; c++) b.ids.push_back(ids[0]);
  } else {
    for (int k = 0; k < K; k++)
      for (size_t v = 1; v < plan[k].size(); v++) b.ids.push_back(ids[k]);
  }
  int n_real = copy_atoms;
  for (int64_t s : ids) {
    n_real += split.n_atoms(s);
    b.max_neighbors = std::max(b.max_neighbors, nl.max_neighbors[s]);
  }
  b.max_neighbors = std::max({b.max_neighbors, 1, pad_neighbors});
  b.n_atoms = std::max(n_real, pad_atoms);
  // copies beyond the tile's systems: the next multiple of it (few graph shapes)
  b.n_systems = std::max(b.n_real_systems, pad_systems);
  if (pad_systems > 0 && b.n_real_systems > pad_systems)
    b.n_systems = (b.n_real_systems + pad_systems - 1) / pad_systems * pad_systems;
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
  b.neighbor_atom.assign(n * m, -1);
  b.atom_weight.assign(static_cast<size_t>(b.n_systems) * n, 0.0f);
  b.energy_target.assign(b.n_systems, 0.0f);
  b.force_target.assign(n * 3, 0.0f);
  b.force_scale.assign(n, 0.0f);
  b.force_mask.assign(n, 0.0f);
  b.n_per_system.assign(b.n_systems, 0);
  b.species.assign(n, 0);
  b.atom_charge.assign(n, 0);
  b.atom_spin.assign(n, 1);
  if (split.has_frag_charges() && split.frag_spec().enabled && split.frag_spec().cond_gate) b.atom_unit_charge.assign(n, 0);
  b.atomic_number.assign(n, 0);
  if (split.sqe_q0_labels())
    b.sqe_q0.assign(n, 0.0), b.sqe_q0_ok.assign(b.n_real_systems, 0), b.sqe_unit.assign(n, 0), b.sqe_unitq.assign(n, 0);
  b.positions.assign(static_cast<size_t>(n) * 3, 0.0f);
  b.cells.assign(b.n_real_systems, {});
  b.periodic.assign(b.n_real_systems, 0);
  b.system_charge.assign(b.n_real_systems, 0);
  if (states) b.vs_of.assign(b.n_real_systems, 0), b.vs_drop.assign(n, -1), b.vs_active.assign(K, 0);
  // padding atoms: no neighbours, so every slot is padded and points to itself
  for (int a = n_real; a < n; a++)
    for (int j = 0; j < m; j++) b.reverse_neighbor_index[a * m + j] = a * m + j;

  // one system: structure k (copy v of it: no loss rows, its own edge drops) at batch system `sys`, atoms from `offset`
  auto emit = [&](int k, int sys, int offset, size_t v) {
    const int64_t s = ids[k];
    const int ns = split.n_atoms(s);
    const int32_t *Z = split.numbers(s);
    const float *F = split.forces(s);
    const float *R = rotations.empty() ? nullptr : rotations[k].data();
    const bool copy = v > 0 || part; // a part: no system has loss rows
    const std::vector<int32_t> *drop = plan[k].empty() ? nullptr : &plan[k][v];
    auto rotate = [R](const float *v, float *out) {
      if (!R) return void(std::copy(v, v + 3, out));
      for (int r = 0; r < 3; r++) out[r] = R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2];
    };
    if (states) {
      b.vs_of[sys] = k;
      if (sys == k) b.vs_active[k] = !plan[k].empty();
      if (!b.vs_copy.empty()) b.vs_copy[sys] = static_cast<int32_t>(v);
      if (drop) std::copy(drop->begin(), drop->end(), b.vs_drop.begin() + offset);
    }

    for (int i = 0; i < ns; i++) {
      const int a = offset + i;
      b.species[a] = index_of(Z[i]);
      b.atomic_number[a] = Z[i];
      b.atom_charge[a] = split.sqe_backbone_neutral() ? 0 : split.charge(s);
      if (!b.atom_unit_charge.empty()) b.atom_unit_charge[a] = split.charge(s);
      b.atom_spin[a] = split.spin_multiplicity(s);
      if (!copy) b.atom_weight[static_cast<size_t>(sys) * n + a] = 1.0f / ns;
      const double fs = scaling.fscale(Z[i]);
      float f[3];
      rotate(F + 3 * i, f);
      for (int r = 0; r < 3; r++) b.force_target[3 * a + r] = static_cast<float>(f[r] / fs);
      if (part) b.force_scale[a] = static_cast<float>(fs); // (the host loss of run_parts)
      if (copy) continue; // no force loss and no metrics on a copy's atoms
      b.force_scale[a] = static_cast<float>(fs);
      b.force_mask[a] = 1.0f;
    }
    {
      const float *C = split.cell(s);
      std::copy(xr[k].begin(), xr[k].end(), b.positions.begin() + 3 * size_t(offset));
      int np = 0;
      for (int r = 0; r < 3; r++) {
        rotate(C + 3 * r, &b.cells[sys][3 * r]);
        np += C[3 * r] != 0 || C[3 * r + 1] != 0 || C[3 * r + 2] != 0;
      }
      b.periodic[sys] = np == 3;
      b.system_charge[sys] = split.charge(s);
      if (!copy && split.has_partial_charges()) {
        if (b.ref_charge.empty()) b.ref_charge.assign(n, 0.0f), b.ref_ok.assign(b.n_real_systems, 0.0f);
        b.ref_ok[sys] = split.partial_charges_ok(s) ? 1.0f : 0.0f;
        b.n_ref_skipped += b.ref_ok[sys] == 0.0f;
        if (b.ref_ok[sys] != 0.0f)
          std::copy(split.partial_charges(s), split.partial_charges(s) + ns, b.ref_charge.begin() + offset);
      }
      if (!copy && split.has_fragments()) {
        if (b.frag_col.empty()) b.frag_col.assign(n, -1), b.frag_target.assign(size_t(b.n_systems) * Batch::kFragCap, 0.0f);
        if (const int cols = split.fragment_columns(s, Batch::kFragCap)) {
          const int32_t *fr = split.fragments(s);
          const float *pq = split.partial_charges(s);
          for (int i = 0; i < ns; i++) {
            const int c = sys * Batch::kFragCap + std::min(fr[i], cols - 1);
            b.frag_col[offset + i] = c, b.frag_target[c] += pq[i];
          }
          b.n_frag += cols;
        }
      }
    }
    if (!copy)
      b.energy_target[sys] = static_cast<float>(
          (split.energy(s) - scaling.comp(Z, ns)) / (scaling.energy_scale * ns));
    if (part && sys == 0)
      b.part_energy_target = static_cast<float>((split.energy(s) - scaling.comp(Z, ns)) / (scaling.energy_scale * ns));
    if (split.sqe_q0_labels() && split.has_frag_charges() && split.frag_ok(s)) {
      const std::vector<double> q = runtime::frag_atom_charges(ns, split.frag_units(s), split.frag_unit_charge(s));
      std::copy(q.begin(), q.end(), b.sqe_q0.begin() + offset);
      std::copy_n(split.frag_units(s), ns, b.sqe_unit.begin() + offset);
      std::copy_n(split.frag_unit_charge(s), ns, b.sqe_unitq.begin() + offset);
      b.sqe_q0_ok[sys] = 1;
    }
    if (split.has_frag_charges() && split.frag_spec().enabled && split.frag_ok(s)) {
      if (states) throw std::runtime_error("lrc_states: fragment-constrained charges are not supported");
      // fragment-constrained charges: each atom conditioned on its bonded unit's charge, and the
      // beyond-R Coulomb term of the unit charges (no weights) taken out of the targets, i.e. the
      // model's prediction plus the term is fitted (same residuals, loss and gradients)
      const int32_t *U = split.frag_units(s), *QU = split.frag_unit_charge(s);
      for (int i = 0; i < ns; i++) (split.frag_spec().cond_gate ? b.atom_unit_charge : b.atom_charge)[offset + i] = QU[i];
      const std::vector<double> q = runtime::frag_atom_charges(ns, U, QU);
      std::vector<double> FF(3 * size_t(ns));
      const double Ef = runtime::frag_coulomb(split.frag_spec(), ns, split.positions(s), q.data(), FF.data());
      if (Ef != 0) {
        b.energy_target[sys] = static_cast<float>((double(split.energy(s)) - scaling.comp(Z, ns) - Ef) / (scaling.energy_scale * ns));
        for (int i = 0; i < ns; i++) {
          const float f[3] = {float(FF[3 * i]), float(FF[3 * i + 1]), float(FF[3 * i + 2])};
          float fr[3];
          rotate(f, fr);
          const double fs = scaling.fscale(Z[i]);
          for (int r = 0; r < 3; r++) b.force_target[3 * size_t(offset + i) + r] -= static_cast<float>(fr[r] / fs);
        }
      }
    }
    b.n_per_system[sys] = ns;

    // edges; a copy drops those between groups its link state separates (same weak component, different group)
    auto dropped = [&](int i, int j) {
      if (!drop) return false;
      const int32_t ga = (*drop)[i], gb = (*drop)[j];
      return ga >= 0 && gb >= 0 && ga != gb && (ga >> 20) == (gb >> 20);
    };
    const int64_t e0 = nl.edge_offset[s], e1 = nl.edge_offset[s + 1];
    for (int64_t e = e0; e < e1; e++) {
      if (dropped(nl.center[e], nl.neighbor[e])) continue;
      const int flat = (offset + nl.center[e]) * m + nl.slot[e];
      b.neighbor_species[flat] = index_of(Z[nl.neighbor[e]]);
      rotate(nl.vector[e].data(), &b.edge_vectors[3 * flat]);
      b.padding_mask[flat] = 0.0f;
      if (!nl.pair_cutoff.empty()) b.cutoff_values[flat] = nl.pair_cutoff[e];
      b.neighbor_atom[flat] = offset + nl.neighbor[e];
      const int32_t r = nl.reverse[e];
      b.reverse_neighbor_index[flat] =
          r >= 0 ? (offset + nl.center[e0 + r]) * m + nl.slot[e0 + r] : flat;
    }
    // (a dropped edge's slot stays as every unused slot of a real atom: padded, reverse index 0)
    if (nl.taper) { // the radius chain: every edge of the full structure (the radii are those of the full structure)
      const TaperEdges t = taper_edges(split, s, nl.cutoff, nl.adaptive);
      for (size_t e = 0; e < t.center.size(); e++) {
        b.taper_center.push_back(offset + t.center[e]);
        b.taper_neighbor.push_back(offset + t.neighbor[e]);
        float v[3];
        rotate(t.vector[e].data(), v);
        b.taper_vector.insert(b.taper_vector.end(), v, v + 3);
        b.taper_jacobian.push_back(t.jacobian[e]);
      }
    }
  };
  int offset = 0, sys = 0;
  if (states && part) b.vs_copy.assign(b.n_real_systems, 0);
  if (part) {
    for (int v : *part) emit(0, sys++, offset, size_t(v)), offset += split.n_atoms(ids[0]);
    return b;
  }
  if (states) {
    b.vs_copy.assign(b.n_real_systems, 0);
  }
  for (int k = 0; k < K; k++) emit(k, sys++, offset, 0), offset += split.n_atoms(ids[k]);
  for (int k = 0; k < K; k++)
    for (size_t v = 1; v < plan[k].size(); v++) emit(k, sys++, offset, v), offset += split.n_atoms(ids[k]);
  return b;
}
} // namespace

std::vector<Batch> make_microbatches(const MemmapSplit &split, const NeighborCache &nl,
                                     const std::vector<int64_t> &ids,
                                     const std::vector<std::array<float, 9>> &rotations,
                                     const std::map<int, int> &species_to_index,
                                     const TargetScaling &scaling, float cutoff, double budget,
                                     double growth, const Tiling &tiling, const StatePlanner *planner) {
  int n_ok = 0;
  for (int64_t s : ids) n_ok += split.partial_charges_ok(s);
  const float charge_norm = n_ok ? 1.0f / n_ok : 0.0f;
  int n_cols = 0;
  for (int64_t s : ids) n_cols += split.fragment_columns(s, Batch::kFragCap);
  const float frag_norm = n_cols ? 1.0f / n_cols : 0.0f;
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
  int64_t g_atoms = 0, g_packed = 0; // real atoms; packed atoms (lrc_states copies included)
  double g_min = 0;
  // lrc_states: the evaluations of each structure (its packed size is that many copies)
  std::vector<int> copies(ids.size(), 1);
  if (planner)
    for (size_t k = 0; k < ids.size(); k++) copies[k] = planner->count(split, ids[k], rotations.empty() ? nullptr : rotations[k].data());
  auto flush = [&] {
    if (g_ids.empty()) return;
    int pad_atoms = 0, pad_nb = 0;
    if (tiling.enabled()) {
      int nb = 1;
      for (int64_t s : g_ids) nb = std::max(nb, nl.max_neighbors[s]);
      pad_atoms = Tiling::bucket(static_cast<int>(g_packed), 32, tiling.atom_ratio);
      pad_nb = Tiling::bucket(nb, 8, tiling.nb_ratio);
    }
    out.push_back(make_batch(split, nl, g_ids, g_rot, species_to_index, scaling, cutoff, pad_atoms,
                             pad_nb, tiling.enabled() ? tiling.systems : 0, planner));
    out.back().energy_norm = 1.0f / ids.size();
    out.back().force_norm = 1.0f / (3.0f * n_total);
    out.back().charge_norm = charge_norm;
    out.back().frag_norm = frag_norm;
    g_ids.clear(), g_rot.clear(), g_atoms = 0, g_packed = 0;
  };
  std::vector<size_t> splits; // lrc_states: structures whose copies exceed the budget, packed as parts below
  for (size_t k : order) {
    const int64_t s = ids[k];
    if (planner && part_copies(copies[k], split.n_atoms(s), nl.max_neighbors[s], budget) < copies[k]) {
      splits.push_back(k);
      continue;
    }
    const double m1 = nl.max_neighbors[s] + 1.0; // sorted: the group's largest so far
    const int64_t atoms = g_packed + int64_t(copies[k]) * split.n_atoms(s);
    // never close a single-atom micro-batch: a leading PyTorch dim of 1 is
    // invisible to ggml_n_dims, which some interpreter ops rely on
    if (g_atoms >= 2 && (m1 > growth * g_min || atoms * m1 * m1 > budget)) flush();
    if (g_ids.empty()) g_min = m1;
    g_ids.push_back(s);
    if (!rotations.empty()) g_rot.push_back(rotations[k]);
    g_atoms += split.n_atoms(s), g_packed += int64_t(copies[k]) * split.n_atoms(s);
  }
  flush();
  static std::atomic<int> next_group{1 << 20}; // (distinct from make_step_batches' group ids)
  for (size_t k : splits) {
    const int64_t s = ids[k];
    const int c = copies[k], pc = part_copies(c, split.n_atoms(s), nl.max_neighbors[s], budget);
    const int n_parts = (c + pc - 1) / pc, gid = next_group++;
    for (int p = 0; p < n_parts; p++) {
      std::vector<int> cp;
      for (int v = p * pc; v < std::min(c, (p + 1) * pc); v++) cp.push_back(v);
      out.push_back(make_part_batch(split, nl, s, rotations.empty() ? nullptr : rotations[k].data(), cp, species_to_index,
                                    scaling, cutoff, *planner));
      Batch &b = out.back();
      b.energy_norm = 1.0f / ids.size();
      b.force_norm = 1.0f / (3.0f * n_total);
      b.charge_norm = charge_norm, b.frag_norm = frag_norm;
      b.part_group = gid, b.part_count = n_parts, b.part_index = p;
    }
  }
  return out;
}

std::vector<std::vector<Batch>> make_step_batches(const MemmapSplit &split, const NeighborCache &nl,
                                                  const std::vector<int64_t> &ids,
                                                  const std::vector<std::array<float, 9>> &rotations,
                                                  const std::map<int, int> &species_to_index,
                                                  const TargetScaling &scaling, float cutoff,
                                                  int n_workers, const CostModel &cost,
                                                  double budget,
                                                  const Tiling &tiling, const StatePlanner *planner) {
  const size_t K_all = ids.size();
  const int W = std::max(1, n_workers);
  int n_ok = 0;
  for (int64_t s : ids) n_ok += split.partial_charges_ok(s);
  const float charge_norm = n_ok ? static_cast<float>(W) / n_ok : 0.0f;
  int n_cols = 0;
  for (int64_t s : ids) n_cols += split.fragment_columns(s, Batch::kFragCap);
  const float frag_norm = n_cols ? static_cast<float>(W) / n_cols : 0.0f;
  std::vector<size_t> order;
  // lrc_states: the evaluations of each structure; a structure whose copies exceed the budget goes into parts (a group
  // of part batches, run together on one worker) instead of the packing below
  std::vector<int> copies(K_all, 1);
  struct SplitUnit {
    size_t q;
    int c, pc;
  };
  std::vector<SplitUnit> splits;
  int64_t real_atoms = 0;
  for (size_t q = 0; q < K_all; q++) {
    real_atoms += split.n_atoms(ids[q]);
    if (planner) copies[q] = planner->count(split, ids[q], rotations.empty() ? nullptr : rotations[q].data());
    const int pc = planner ? part_copies(copies[q], split.n_atoms(ids[q]), nl.max_neighbors[ids[q]], budget) : copies[q];
    if (pc < copies[q]) splits.push_back({q, copies[q], pc});
    else order.push_back(q);
  }
  const size_t K = order.size();
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return nl.max_neighbors[ids[a]] < nl.max_neighbors[ids[b]];
  });
  // prefix sums in sorted order of the packed atoms (lrc_states: every evaluation of a structure counts); the
  // force-loss normalisation counts the real atoms of the whole step
  std::vector<int64_t> atoms(K + 1, 0);
  for (size_t k = 0; k < K; k++) atoms[k + 1] = atoms[k] + int64_t(copies[order[k]]) * split.n_atoms(ids[order[k]]);
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
  Groups groups = K ? plan(std::numeric_limits<double>::infinity()) : Groups{};
  if (groups.empty() && K) groups = {{0, K}};
  std::vector<int> owner;
  double best = deal(groups, owner);
  if (W > 1 && K) {
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
                           pad_nb, tiling.enabled() ? tiling.systems : 0, planner);
      b.energy_norm = static_cast<float>(W) / K_all;
      b.force_norm = static_cast<float>(W) / (3.0f * real_atoms);
      b.charge_norm = charge_norm;
      b.frag_norm = frag_norm;
      return b;
    }));
  std::vector<std::vector<Batch>> out(W);
  std::vector<double> load(W, 0.0);
  for (size_t g = 0; g < groups.size(); g++)
    out[owner[g]].push_back(built[g].get()), load[owner[g]] += group_cost(groups[g].first, groups[g].second);
  // the split structures: each a group of consecutive parts, on the least loaded worker
  static std::atomic<int> next_group{0};
  for (const SplitUnit &u : splits) {
    const int wk = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
    const int n_parts = (u.c + u.pc - 1) / u.pc, gid = next_group++;
    for (int p = 0; p < n_parts; p++) {
      std::vector<int> cp;
      for (int v = p * u.pc; v < std::min(u.c, (p + 1) * u.pc); v++) cp.push_back(v);
      Batch b = make_part_batch(split, nl, ids[u.q], rotations.empty() ? nullptr : rotations[u.q].data(), cp,
                                species_to_index, scaling, cutoff, *planner);
      b.energy_norm = static_cast<float>(W) / K_all;
      b.force_norm = static_cast<float>(W) / (3.0f * real_atoms);
      b.charge_norm = charge_norm, b.frag_norm = frag_norm;
      b.part_group = gid, b.part_count = n_parts, b.part_index = p;
      load[wk] += cost(double(b.n_atoms), double(b.max_neighbors));
      out[wk].push_back(std::move(b));
    }
  }
  return out;
}

} // namespace mlipcpp::train
