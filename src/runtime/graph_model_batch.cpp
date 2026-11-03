// GraphModel::predict_batch: many independent molecules in one graph
// computation (mmml ML/MM evaluates thousands of monomers and dimers per MD
// step). The structures are packed into the cached graph of their total atom
// count (graph_model_cached.cpp) with a block-diagonal neighbour list, the
// charge / spin rows per atom; energies are summed per structure and the edge
// gradients scattered to forces as in predict_cached.

#include "core/padding.h"
#include "graph_cache.h"
#include "graph_model.h"

#include <ggml-backend.h>
#include <ggml.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mlipcpp::runtime {

std::vector<ModelResult> GraphModel::predict_batch(const std::vector<AtomicSystem> &systems, bool compute_forces) {
  std::vector<ModelResult> results(systems.size());
  if (systems.empty()) return results;
  bool loop = !cached_path_supported() || num_neighbors_adaptive_ > 0.0f || plato_lr_.enabled || long_range_ ||
              nc_forces_;
  for (const auto &s : systems) {
    const Cell *c = s.cell();
    if (c && (c->periodic[0] || c->periodic[1] || c->periodic[2])) loop = true;
  }
  if (loop) { // periodic structures, PET variants: one at a time
    for (size_t k = 0; k < systems.size(); k++) results[k] = predict(systems[k], compute_forces);
    return results;
  }
  GraphProfile profile;
  const int S = static_cast<int>(systems.size());
  std::vector<int64_t> atom0(S + 1, 0);
  for (int k = 0; k < S; k++) atom0[k + 1] = atom0[k] + systems[k].num_atoms();
  const int64_t N = atom0[S];
  if (N > std::numeric_limits<int32_t>::max() / 64) throw std::runtime_error("predict_batch: too many atoms");
  const int n = static_cast<int>(N);
  const int nt = std::max(1, backend_provider_->n_threads());
  const double rc = cutoff_, rc2 = rc * rc;

  // species, neighbour counts (all pairs within each structure, j ascending)
  std::vector<int32_t> species(n), nkept(n, 0);
  for (int k = 0; k < S; k++) {
    const int32_t *Z = systems[k].atomic_numbers();
    for (int i = 0; i < systems[k].num_atoms(); i++) {
      auto it = species_to_index_.find(Z[i]);
      if (it == species_to_index_.end())
        throw std::runtime_error("Atomic number " + std::to_string(Z[i]) + " not in species map");
      species[atom0[k] + i] = it->second;
    }
  }
#pragma omp parallel for num_threads(nt) schedule(dynamic, 8)
  for (int k = 0; k < S; k++) {
    const int na = systems[k].num_atoms();
    const float *x = systems[k].positions();
    for (int i = 0; i < na; i++) {
      int c = 0;
      for (int j = 0; j < na; j++) {
        if (j == i) continue;
        const double dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        if (dx * dx + dy * dy + dz * dz < rc2) c++;
      }
      nkept[atom0[k] + i] = c;
    }
  }
  const int max_kept = n ? *std::max_element(nkept.begin(), nkept.end()) : 0;
  profile.mark("neighbor_list");
  const bool no_edges = max_kept == 0, forces_requested = compute_forces;
  if (no_edges) compute_forces = false;
  if (fit_cached_graph(n, max_kept)) profile.mark("build_graph");
  CachedGraph &g = *cached_;
  const int M = g.M;
  const size_t SL = size_t(n) * M;

  std::vector<int32_t> ns(SL, padding::kSpecies), rev(SL, 0);
  std::vector<float> ev(3 * SL), pm(SL, 1.0f), cv(SL, static_cast<float>(rc));
  for (size_t f = 0; f < SL; f++) padding::edge(static_cast<float>(rc), &ev[3 * f]);
#pragma omp parallel for num_threads(nt) schedule(dynamic, 8)
  for (int k = 0; k < S; k++) {
    const int na = systems[k].num_atoms();
    const float *x = systems[k].positions();
    const int64_t a0 = atom0[k];
    std::vector<int32_t> slot(size_t(na) * na, -1); // slot of j in i's list
    for (int i = 0; i < na; i++) {
      int s = 0;
      for (int j = 0; j < na; j++) {
        if (j == i) continue;
        const double v[3] = {double(x[3 * j]) - x[3 * i], double(x[3 * j + 1]) - x[3 * i + 1],
                             double(x[3 * j + 2]) - x[3 * i + 2]};
        if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] >= rc2) continue;
        const size_t f = size_t(a0 + i) * M + s;
        slot[size_t(i) * na + j] = s++;
        ns[f] = species[a0 + j];
        for (int c = 0; c < 3; c++) ev[3 * f + c] = static_cast<float>(v[c]);
        pm[f] = 0.0f;
      }
    }
    for (int i = 0; i < na; i++)
      for (int j = 0; j < na; j++) {
        const int s = slot[size_t(i) * na + j];
        if (s < 0) continue;
        rev[size_t(a0 + i) * M + s] = static_cast<int32_t>(size_t(a0 + j) * M + slot[size_t(j) * na + i]);
      }
  }
  ggml_backend_tensor_set(g.species, species.data(), 0, ggml_nbytes(g.species));
  ggml_backend_tensor_set(g.neighbor_species, ns.data(), 0, ggml_nbytes(g.neighbor_species));
  ggml_backend_tensor_set(g.edge_vectors, ev.data(), 0, ggml_nbytes(g.edge_vectors));
  ggml_backend_tensor_set(g.padding_mask, pm.data(), 0, ggml_nbytes(g.padding_mask));
  ggml_backend_tensor_set(g.reverse, rev.data(), 0, ggml_nbytes(g.reverse));
  ggml_backend_tensor_set(g.cutoff_values, cv.data(), 0, ggml_nbytes(g.cutoff_values));
  if (g.charge_index) { // per structure; the single-system path re-uploads after this
    std::vector<int32_t> ci(n), si(n);
    for (int k = 0; k < S; k++) {
      const auto [c, s] = conditioning_.rows(systems[k].charge(), systems[k].spin_multiplicity());
      std::fill(ci.begin() + atom0[k], ci.begin() + atom0[k + 1], c);
      std::fill(si.begin() + atom0[k], si.begin() + atom0[k + 1], s);
    }
    ggml_backend_tensor_set(g.charge_index, ci.data(), 0, ggml_nbytes(g.charge_index));
    ggml_backend_tensor_set(g.spin_index, si.data(), 0, ggml_nbytes(g.spin_index));
    if (g.system_index) {
      std::vector<int32_t> sys(n);
      for (int k = 0; k < S; k++) std::fill(sys.begin() + atom0[k], sys.begin() + atom0[k + 1], k);
      ggml_backend_tensor_set(g.system_index, sys.data(), 0, ggml_nbytes(g.system_index));
    }
    g.cond_set = false;
  }
  profile.mark("inputs");
  if (backend_provider_->compute(compute_forces ? g.graph : g.forward) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("Graph computation failed");
  profile.mark("compute");

  std::vector<float> atomic(n);
  ggml_backend_tensor_get(g.energy, atomic.data(), 0, atomic.size() * sizeof(float));
  std::vector<float> G;
  if (compute_forces) {
    G.resize(3 * SL);
    ggml_backend_tensor_get(g.grad_ev, G.data(), 0, G.size() * sizeof(float));
  }
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int k = 0; k < S; k++) {
    ModelResult &r = results[k];
    const int32_t *Z = systems[k].atomic_numbers();
    double model = 0, composition = 0;
    for (int64_t a = atom0[k]; a < atom0[k + 1]; a++) {
      model += atomic[a];
      if (auto c = composition_energies_.find(Z[a - atom0[k]]); c != composition_energies_.end())
        composition += c->second;
    }
    r.energy = static_cast<float>(energy_scale_ * model + composition);
    if (compute_forces || forces_requested) {
      r.forces.assign(3 * size_t(systems[k].num_atoms()), 0.0f);
      r.has_forces = true;
    }
    if (!compute_forces) continue;
    for (int64_t a = atom0[k]; a < atom0[k + 1]; a++) {
      double F[3] = {0, 0, 0};
      for (int s = 0; s < nkept[a]; s++) {
        const size_t f = size_t(a) * M + s, b = rev[f];
        for (int c = 0; c < 3; c++) F[c] += double(G[3 * f + c]) - double(G[3 * b + c]);
      }
      for (int c = 0; c < 3; c++) r.forces[3 * (a - atom0[k]) + c] = static_cast<float>(energy_scale_ * F[c]);
    }
  }
  profile.mark("readback");
  return results;
}

} // namespace mlipcpp::runtime
