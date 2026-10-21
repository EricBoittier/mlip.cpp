#pragma once
// Internals shared by GraphModel's two prediction paths (graph_model.cpp,
// graph_model_cached.cpp).

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
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr; // forward (+ backward)
  ggml_cgraph *forward = nullptr; // the forward nodes of `graph`, in the same order
  ggml_backend_buffer_t inputs = nullptr;

  ggml_tensor *species = nullptr, *neighbor_species = nullptr, *edge_vectors = nullptr,
              *padding_mask = nullptr, *reverse = nullptr, *cutoff_values = nullptr;
  ggml_tensor *energy = nullptr, *nc = nullptr;            // [1, n], [3, n] (unscaled)
  ggml_tensor *grad_ev = nullptr, *grad_cv = nullptr;      // dE / d edge_vectors, cutoff_values

  CachedGraph() = default;
  CachedGraph(const CachedGraph &) = delete;
  CachedGraph &operator=(const CachedGraph &) = delete;
  ~CachedGraph() {
    if (inputs) ggml_backend_buffer_free(inputs);
    if (ctx) ggml_free(ctx);
  }
};

} // namespace mlipcpp::runtime
