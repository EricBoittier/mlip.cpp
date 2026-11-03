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
//   x_i = Embed[Z_i] (+ Wq Q + bq + Ws S + bs for Spooky; charge_spin_per_atom: Q / N, (S - 1) / N,
//     N the atom count of the structure, S = 2S+1 the multiplicity)
//   per iteration t: filter f_l = basis_l W_l (MessagePass/filter),
//     m_i = sum_j tensor(f_ij, x_j) with the paths 00->0, 11->0 (dot / sqrt 3),
//     01->1, 10->1 (MessagePass/tensor kernel), the last iteration to degree 0,
//     then n_res residual blocks x += Dense(x + silu(x)), y = silu(Dense(y_last))
//     with the e3x gated silu (sigmoid of the scalar channel times every degree)
//     (feature_rmsnorm: then y /= sqrt(mean of y^2 over all components + 1e-6) per atom)
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
  // mmml SpookyPhysNet feature_rmsnorm: after each refinement stage's final silu,
  // y <- y / sqrt(mean_{degree, feature}(y^2) + 1e-6), one scalar per atom (no parameters)
  bool feature_rmsnorm = false;
  // mmml SpookyPhysNet charge_spin_per_atom: the conditioning inputs are Q / N and (M - 1) / N per atom,
  // N the atom count of its structure (system_index; nullptr: all atoms), M the multiplicity 2S+1
  bool charge_spin_per_atom = false;
  // multitask heads (mmml SpookyPhysNet multitask_*, proposals/SPOOKY_MULTITASK.md) on the
  // final scalar features x_i, the energy head's input:
  //   charges q_i = q~_i + (Q - sum_s q~) / N_s, q~_i = x_i w_q + b_q[Z_i]  (mt.charge.w, mt.charge.zbias)
  //   spins   s_i = s~_i + (M - sum_s s~) / N_s, M = multiplicity - 1      (mt.spin.w, mt.spin.zbias)
  //   orbitals: h_i = silu(W_h x_i + c_h), (homo_i, gap_i) = W_o h_i + c_o + offsets
  //            (mt.orb.hidden.w/b, mt.orb.out.w/b); the molecular values are the structure means
  // sums and means over the atom's structure (system_index); Q, M: the Spooky conditioning values
  bool head_charges = false, head_spins = false, head_orbitals = false;
  float homo_offset = 0.0f, gap_offset = 0.0f;
  bool any_head() const { return head_charges || head_spins || head_orbitals; }
};

/// per-atom outputs of the heads ([n] each, nullptr when absent)
struct PhysNetHeads {
  ggml_tensor *charges = nullptr, *spins = nullptr, *homo = nullptr, *gap = nullptr;
};

struct PhysNetInputs {
  ggml_tensor *species = nullptr;          // I32 [n]  (= Z)
  ggml_tensor *neighbor_species = nullptr; // I32 [m, n]
  ggml_tensor *edge_vectors = nullptr;     // F32 [3, m, n]  r_j - r_i
  ggml_tensor *padding_mask = nullptr;     // F32 [m, n]  1 = padded slot
  ggml_tensor *reverse_index = nullptr;    // I32 [m n]  flat slot of the edge j -> i
  ggml_tensor *charge_index = nullptr;     // I32 [n]  (Spooky)
  ggml_tensor *spin_index = nullptr;       // I32 [n]  (Spooky)
  // I32 [n]: the atom's structure in [0, n) (heads: per-structure sums); nullptr = one structure
  ggml_tensor *system_index = nullptr;
};

using WeightFn = std::function<ggml_tensor *(const std::string &)>;
/// a tensor of the given ggml shape filled with `value` (an input set after allocation)
using ConstantFn = std::function<ggml_tensor *(ggml_context *, std::vector<int64_t>, float)>;

/// per-atom energies [n]; the heads' per-atom outputs in *heads (if given)
ggml_tensor *build_physnet(ggml_context *ctx, const PhysNetHypers &h, const PhysNetInputs &in,
                           const WeightFn &weight, const ConstantFn &constant, PhysNetHeads *heads = nullptr);

} // namespace mlipcpp::physnet
