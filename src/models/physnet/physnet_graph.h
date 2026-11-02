#pragma once
// PhysNet and SpookyPhysNet (mmml physnetjax, e3x) as a ggml graph on PET's
// neighbour-slot inputs, so the same GGUF runs in GraphModel (inference),
// mlipcpp-train (energy + conservative force loss: the graph only uses ops with
// forward-mode tangent rules, runtime/graph_jvp.h) and the C/C++/Python APIs.
//
// GGUF: architecture "pet-graph" whose graph.json is a single "PHYSNET" node
// (scripts/convert_mmml.py). Its params are the hyperparameters below; its
// inputs are the graph inputs species, neighbor_species, edge_vectors,
// padding_mask, reverse_neighbor_index (+ charge_index, spin_index for Spooky:
// core/conditioning.h rows). The output is the per-atom energy [n_atoms] in eV
// (atomic energies + per-element bias + ZBL repulsion), as mmml's `energy`.
//
// Math (mmml models/model.py, spooky_model.py; e3x 1.0, Cartesian order,
// Racah-normalised spherical harmonics, no pseudotensors, max_degree <= 1):
//   basis_lm,k(r_ij) = Y_lm(u) T_k(2 e^{-r} - 1) exp(1 - 1/(1 - (r/rc)^2))
//   x_i = Embed[Z_i] (+ Wq Q + bq + Ws S + bs for Spooky)
//   per iteration t: filter f_l = basis_l W_l (MessagePass/filter),
//     m_i = sum_j tensor(f_ij, x_j) with the paths 00->0, 11->0 (dot / sqrt 3),
//     01->1, 10->1 (MessagePass/tensor kernel), the last iteration to degree 0,
//     then n_res residual blocks x += Dense(x + silu(x)), y = silu(Dense(y_last))
//     with the e3x gated silu (sigmoid of the scalar channel times every degree)
//   E_i = (x_i W_e) w + energy_bias[Z_i] + ZBL_i
#include <functional>
#include <map>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace mlipcpp::physnet {

struct PhysNetHypers {
  bool spooky = false;
  int features = 32;
  int max_degree = 1;
  int num_iterations = 2;
  int num_basis = 16;
  int n_res = 3;
  int max_atomic_number = 118;
  float cutoff = 6.0f;
  bool zbl = true;
  float zbl_cutoff = 0.6f;
  float zbl_cuton = 0.1f;
  bool energy_bias = false;
};

struct PhysNetInputs {
  ggml_tensor *species = nullptr;          // I32 [n]  (= Z)
  ggml_tensor *neighbor_species = nullptr; // I32 [m, n]
  ggml_tensor *edge_vectors = nullptr;     // F32 [3, m, n]  r_j - r_i
  ggml_tensor *padding_mask = nullptr;     // F32 [m, n]  1 = padded slot
  ggml_tensor *reverse_index = nullptr;    // I32 [m n]  flat slot of the edge j -> i
  ggml_tensor *charge_index = nullptr;     // I32 [n]  (Spooky)
  ggml_tensor *spin_index = nullptr;       // I32 [n]  (Spooky)
};

using WeightFn = std::function<ggml_tensor *(const std::string &)>;
/// a tensor of the given ggml shape filled with `value` (an input set after allocation)
using ConstantFn = std::function<ggml_tensor *(ggml_context *, std::vector<int64_t>, float)>;

/// per-atom energies [n]
ggml_tensor *build_physnet(ggml_context *ctx, const PhysNetHypers &h, const PhysNetInputs &in,
                           const WeightFn &weight, const ConstantFn &constant);

} // namespace mlipcpp::physnet
