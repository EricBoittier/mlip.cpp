#pragma once
// PET with Plato-PET-LR components (metatrain pet/modules/plato_lr.py, the plato_lr
// hypers): native builders of the MLIPCPP_* graph ops that export_pytorch/plato_lr.py
// leaves in an exported PET graph, and the per-call inputs they read.
//
//   MLIPCPP_ROPE_FRAME_ATTN(q, k, v, bias, pos, omega, frames)  mean_f softmax(RoPE_f q .
//       RoPE_f k scale + bias) v, one fused PET attention (ggml_pet_attn) per frame
//   MLIPCPP_EDGE_GROUP_SUM(x, W, frames)  mean_f SiLU(W R_f^T x[:3])
//   MLIPCPP_PLATO_MESH(v, filt..., k_quad, w_quad)  the learned mesh potential of the
//       latent channels (periodic: B-spline spread, rfftn, filter, irfftn, gather, times
//       the mesh points; open: the real-space kernel erf(r/sqrt2 s)/r + sum_n w_n f(k_n)
//       j0(k_n r) over the atom pairs of each structure, i = j included)
//   MLIPCPP_PLATO_LES(q, charge_unit)  per-atom LES energies of the latent charges
//       (periodic: Ewald k-sum or smooth PME, the structure's energy spread evenly over
//       its atoms; open: the erf pair sum per atom), neutralised per structure or shifted
//       to the structure's total charge
//
// The long-range ops depend on the absolute positions (input `positions`, [3, n]) and,
// for periodic structures, a strain tensor (input `strain`, [3, 3, G], zeros): their
// gradients complete the forces (-dE/dx) and the virial (-dE/d eps) of the graph's
// edge-vector terms, as in runtime/plato_graph.

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace mlipcpp {
class GGUFLoader;
namespace runtime {

struct GIRNode;

/// the plato_lr hypers of a GGUF ("pet.plato_lr.*"; absent: off)
struct PetPlatoLRSpec {
  bool enabled = false;
  bool rope = false, edge_group_sum = false, mesh = false, les = false;
  std::string frames = "identity";
  int n_frames = 1, couple_at = -1, mesh_channels = 0, n_quad = 0, num_charges = 0;
  float mesh_spacing = 1.0f, mesh_sigma = 1.5f, les_sigma = 1.0f, les_kcut = 6.0f;
  bool mesh_realspace = true, total_charge_constraint = false;
  // periodic LES: "ewald" | "pme" | "auto" (PME from les_pme_atoms atoms per structure);
  // GGUF "pet.plato_lr.les_method", MLIPCPP_PLATO_LES_METHOD overrides
  std::string les_method = "auto";
  int les_pme_order = 8, les_pme_atoms = 128;
  double les_pme_spacing = 0.5; // in units of les_sigma
  bool long_range() const { return mesh || les; }
  static PetPlatoLRSpec from_gguf(const GGUFLoader &loader);
};

/// one structure of a batch: atoms [a0, a1) of the graph's atom order
struct PetPlatoLRStructure {
  int a0 = 0, a1 = 0;
  bool periodic = false;
  double cell[3][3] = {};  // lattice vectors as rows
  double charge = 0;       // total charge (e), for total_charge_constraint
};

/// Per-call inputs and the builders. Use: create(...) in the graph's input context
/// (before it is allocated), set the interpreter's custom_op to build(...), allocate,
/// upload(positions).
class PetPlatoLR {
public:
  /// `grad`: positions and strain become ggml params (forces, virial). mesh_fixed: a
  /// fixed periodic mesh (0: max(8, ceil(|a| / spacing)) per axis, all structures equal).
  void create(ggml_context *ctx, const PetPlatoLRSpec &spec, const std::vector<PetPlatoLRStructure> &structures,
              int n_atoms, bool grad);
  /// writes the host data into the created tensors; positions [n][3]
  void upload(const float *positions) const;
  /// the MLIPCPP_* op of `node`
  ggml_tensor *build(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &inputs);

  ggml_tensor *positions = nullptr; // F32 [3, n]
  ggml_tensor *strain = nullptr;    // F32 [3, 3, G] (periodic), else nullptr
  bool periodic() const { return periodic_; }
  int n_struct() const { return static_cast<int>(st_.size()); }
  const int *mesh() const { return mesh_; }
  const int *les_mesh() const { return les_mesh_; }
  int64_t n_k() const { return nk_; }
  /// the input tensors (tangent seeds of the conservative trainer: positions)
  const std::vector<ggml_tensor *> &tensors() const { return all_; }

private:
  PetPlatoLRSpec spec_;
  std::vector<PetPlatoLRStructure> st_;
  int n_ = 0;
  bool periodic_ = false;
  int mesh_[3] = {0, 0, 0}, les_mesh_[3] = {0, 0, 0};
  int64_t nk_ = 0, n_pairs_ = 0;
  std::vector<ggml_tensor *> all_;
  // inputs
  ggml_tensor *struct_index_ = nullptr, *onehot_ = nullptr, *inv_count_ = nullptr, *charge_per_atom_ = nullptr;
  ggml_tensor *atom_inv_cell_t_ = nullptr, *inv_volume_ = nullptr;
  ggml_tensor *mesh_k_ = nullptr, *mesh_k0_pad_ = nullptr, *mesh_deconv_ = nullptr;
  ggml_tensor *les_k_ = nullptr, *les_n_ = nullptr, *les_K_ = nullptr, *les_tab_ = nullptr;
  ggml_tensor *pair_i_ = nullptr, *pair_j_ = nullptr; // open: off-diagonal pairs within structures
  // host data
  std::vector<int32_t> h_sidx_, h_pi_, h_pj_;
  std::vector<float> h_onehot_, h_invc_, h_qpa_, h_aict_, h_invv_, h_mk_, h_mpad_, h_mdec_, h_lk_, h_ln_, h_lK_, h_tab_;
  ggml_tensor *frac_ = nullptr; // [3, n], built once per graph

  ggml_tensor *frac(ggml_context *ctx);
  ggml_tensor *rope_attn(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in);
  ggml_tensor *edge_sum(ggml_context *ctx, const std::vector<ggml_tensor *> &in);
  ggml_tensor *mesh_op(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in);
  ggml_tensor *les_op(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in);
  ggml_tensor *pair_r(ggml_context *ctx); // [1, P] distances of the pairs
  ggml_tensor *pair_r_ = nullptr;
};

} // namespace runtime
} // namespace mlipcpp
