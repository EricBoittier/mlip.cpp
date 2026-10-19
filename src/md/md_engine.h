#pragma once
// GPU molecular dynamics with a PET GGUF (forces-mode export, optionally with
// the direct "nc" force head). The model graph (forward + backward to the edge
// vectors and pair cutoffs) is built once per neighbour capacity M and reused;
// neighbour list, input packing, adaptive cutoffs, force gathering and the
// integrator are CUDA kernels writing straight into the ggml tensors.
//
// Forces: conservative F = -dE/dx (including the adaptive cutoff's own
// position dependence), and the direct head's forces (nc) when present.
// Integrators: velocity Verlet on either force, or RESPA (inner steps on the
// cheap nc forces, outer correction F_cons - F_nc), which keeps the
// conservative Hamiltonian's energy while running most steps forward-only.

#include "md/md_kernels.h"
#include "mlipcpp/system.h"
#include "runtime/graph_model.h"

#include <ggml-alloc.h>
#include <ggml-impl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mlipcpp::md {

class Engine {
public:
  Engine(const std::string &gguf, int device = 0, double skin = 1.0);
  ~Engine();
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  void set_system(const AtomicSystem &system); // also builds the neighbour list
  void set_positions(const std::vector<double> &x); // [n, 3], Å
  void set_velocities(const std::vector<double> &v); // [n, 3], Å/fs
  void maxwell_boltzmann(double temperature, uint64_t seed); // exact T, zero momentum

  /// Model evaluation at the current positions: nc forces always (if the
  /// model has the head), conservative forces when `conservative`.
  /// Returns the energy (eV) if `want_energy`, else NaN.
  double evaluate(bool conservative, bool want_energy = true);

  /// One step; the forces of the current positions must be up to date
  /// (evaluate(true) once after set_system). `inner` = RESPA inner steps.
  enum class Integrator { Conservative, Direct, Respa };
  void step(Integrator integ, double dt, int inner = 1, bool want_energy = false);

  double energy() const { return energy_; } // of the last evaluation that computed it
  double kinetic_energy() const;            // eV
  double temperature() const;               // K, 3n - 3 degrees of freedom
  std::vector<double> positions() const, velocities() const, forces_conservative() const,
      forces_direct() const;

  int n_atoms() const { return st_.n; }
  bool has_direct() const { return model_.nc_forces(); }
  int capacity_neighbors() const { return st_.M; }
  int max_neighbors() const { return max_kept_; }
  int64_t n_edges() const; // kept edges of the last evaluation
  std::vector<int32_t> kept_per_atom() const;
  struct Inputs { // the graph inputs of the last evaluation, [n, M] slots
    int M = 0;
    std::vector<int32_t> species, neighbor_species, reverse;
    std::vector<float> edge_vectors, padding_mask, cutoff_values;
  };
  Inputs inputs() const;
  int64_t list_builds() const { return list_builds_; }
  int64_t graph_builds() const { return graph_builds_; }
  const std::vector<int32_t> &numbers() const { return Z_; }
  const double *cell() const { return st_.cell; }
  const int *pbc() const { return st_.pbc; }
  runtime::GraphModel &model() { return model_; }

private:
  void build_graph(int M);
  void free_graph();
  void alloc_atoms();
  void alloc_candidates(int K);
  void build_list();
  void maybe_rebuild_list();
  void prepare(); // inputs; grows / shrinks the graph if needed

  runtime::GraphModel model_;
  gpu::State st_;
  gpu::Cutoff cut_;
  gpu::Graph g_;
  double energy_scale_ = 1, energy_ = 0;
  std::vector<int32_t> Z_;
  std::vector<double> composition_; // per atom
  int max_kept_ = 0;
  int64_t list_builds_ = 0, graph_builds_ = 0;

  // graph
  ggml_context *ctx_ = nullptr;
  ggml_cgraph *gf_ = nullptr;
  ggml_cgraph fwd_{};
  ggml_backend_buffer_t persist_ = nullptr, flat_ = nullptr;
  ggml_gallocr_t galloc_ = nullptr;
  ggml_tensor *energy_t_ = nullptr;
};

} // namespace mlipcpp::md
