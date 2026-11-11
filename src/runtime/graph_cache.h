#pragma once
// Internals shared by GraphModel's two prediction paths (graph_model.cpp,
// graph_model_cached.cpp).

#include "pet_plato_lr.h"
#include <memory>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace mlipcpp::runtime {

// Optional wall-clock breakdown, including backend synchronization in compute.
struct GraphProfile {
  bool enabled = std::getenv("MLIPCPP_PROFILE_GRAPH") != nullptr;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now(), previous = start;
  std::vector<std::pair<const char *, double>> stages;
  void mark(const char *name) {
    if (!enabled) return;
    const auto now = std::chrono::steady_clock::now();
    stages.emplace_back(name, std::chrono::duration<double, std::milli>(now - previous).count());
    previous = now;
  }
  ~GraphProfile() {
    if (!enabled) return;
    mark("cleanup");
    std::fprintf(stderr, "GRAPH_PROFILE");
    for (auto [name, ms] : stages) std::fprintf(stderr, " %s=%.3f", name, ms);
    std::fprintf(stderr, " total=%.3f ms\n", std::chrono::duration<double, std::milli>(previous - start).count());
  }
};

/// The graph of a forces-mode model for one shape (n_atoms, M neighbour
/// slots): forward, and backward to the edge vectors and (adaptive cutoff)
/// the pair cutoffs. Built once, computed for every system of that shape.
/// Inputs and constants have a buffer of their own, the rest of the graph
/// goes through GraphModel's ggml_gallocr.
struct CachedGraph {
  int n_atoms = 0, M = 0;
  bool exact_mask = false, backward = false;
  // packed edges (pet_ragged.h): rows [offsets[i], offsets[i + 1]) of atom i,
  // then padding rows up to e_cap; sequences of at most max_len tokens
  bool ragged = false;
  int64_t e_cap = 0;
  int max_len = 0;
  ggml_tensor *offsets = nullptr, *centers = nullptr;
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr; // forward (+ backward)
  ggml_cgraph *forward = nullptr; // the forward nodes of `graph`, in the same order
  ggml_backend_buffer_t inputs = nullptr;

  ggml_tensor *species = nullptr, *neighbor_species = nullptr, *edge_vectors = nullptr,
              *padding_mask = nullptr, *reverse = nullptr, *cutoff_values = nullptr;
  // charge / spin conditioning: I32 [n] table rows (nullptr: an unconditioned model),
  // and the (charge, spin multiplicity) last uploaded to them
  ggml_tensor *charge_index = nullptr, *spin_index = nullptr, *charge_index_unit = nullptr;
  ggml_tensor *cond_blend = nullptr; // F32 [n]: the combined model's local-charge blend
  int cond_charge = 0, cond_spin = 0;
  bool cond_set = false;
  ggml_tensor *energy = nullptr, *nc = nullptr;            // [1, n], [3, n] (unscaled)
  ggml_tensor *loss = nullptr;                             // what the backward pass differentiates
  ggml_tensor *grad_ev = nullptr, *grad_cv = nullptr;      // dE / d edge_vectors, cutoff_values
  // Plato-PET-LR: the long-range ops' inputs (positions, cell constants) and dE / d positions, strain
  std::unique_ptr<PetPlatoLR> plato;
  ggml_tensor *grad_pos = nullptr, *grad_eps = nullptr;
  ggml_tensor *latent_q = nullptr; // [Q, n] latent charges (model units), an output
  // predict_device_step: 1 when every node runs on the compute backend, so the
  // graph goes to it directly and asynchronously (-1: not checked yet)
  int direct = -1;

  CachedGraph() = default;
  CachedGraph(const CachedGraph &) = delete;
  CachedGraph &operator=(const CachedGraph &) = delete;
  ~CachedGraph() {
    if (inputs) ggml_backend_buffer_free(inputs);
    if (ctx) ggml_free(ctx);
  }
};

/// The graph of GraphModel's general path (predict_single_impl: per-atom outputs, atom energy weights, dE/d blend,
/// lrc_states edge drops, structures without edges), kept for one exact shape: the same graph (same nodes, same
/// kernels) as a fresh build, without the per-call build, allocations and constant uploads. A kept graph goes through
/// its own ggml_gallocr (memory reused along the graph), its constants in const_buffer.
struct PlainGraph {
  struct Key {
    int n_atoms = 0, M = 0, periodic = 0;
    bool forces = false, backward = false, blend_grad = false, atom_w = false, no_edges = false;
    bool operator==(const Key &) const = default;
  } key;
  ggml_context *input_ctx = nullptr, *compute_ctx = nullptr;
  ggml_backend_buffer_t input_buffer = nullptr, compute_buffer = nullptr, const_buffer = nullptr;
  ggml_gallocr_t galloc = nullptr; // a kept graph's allocator (compute_buffer: a graph built for one call)
  ggml_tensor *species = nullptr, *neighbor_species = nullptr, *edge_vectors = nullptr, *padding_mask = nullptr,
              *reverse = nullptr, *charge_index = nullptr, *spin_index = nullptr, *charge_index_unit = nullptr,
              *cond_blend = nullptr, *atom_w = nullptr, *edge_distances = nullptr, *cutoff_factors = nullptr,
              *cutoff_values = nullptr, *output = nullptr, *nc_output = nullptr, *latent_q = nullptr;
  ggml_cgraph *cgraph = nullptr;
  bool forces = false; // dE/d edge vectors in the graph
  std::unique_ptr<PetPlatoLR> plato;
  size_t bytes = 0; // device bytes held (inputs + compute)

  PlainGraph() = default;
  PlainGraph(const PlainGraph &) = delete;
  PlainGraph &operator=(const PlainGraph &) = delete;
  ~PlainGraph() {
    plato.reset();
    if (galloc) ggml_gallocr_free(galloc);
    if (const_buffer) ggml_backend_buffer_free(const_buffer);
    if (compute_buffer) ggml_backend_buffer_free(compute_buffer);
    if (input_buffer) ggml_backend_buffer_free(input_buffer);
    if (compute_ctx) ggml_free(compute_ctx);
    if (input_ctx) ggml_free(input_ctx);
  }
};

} // namespace mlipcpp::runtime
