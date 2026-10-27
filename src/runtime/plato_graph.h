#pragma once
// Plato-PET-LR (metatrain experimental.plato_pet_lr, frame_mode: attention) built
// directly in ggml on packed tokens: the n centre tokens, then the E edge tokens
// grouped by centre (the sequences of ggml_pet_attn_ragged). Every FrameLinear
// has one token frame, i.e. it is a linear map W_eff = sum_b phi_b(I) W_b; the
// group frames enter only through the RoPE angles of the attention
// (ggml_plato_attn) and the edge embedding sum_f w_f SiLU(W R_f^T r).
// Long range: the learned mesh coupling after layer couple_at (B-spline spread,
// rfftn, learned k filter, irfftn, gather) and the LES Ewald energy of the latent
// charges (or their erf pair sum for open systems). The energy is a function of
// the edge vectors, the positions (through fractional coordinates) and a strain
// tensor, so ggml's backward pass gives forces and the virial.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace mlipcpp::runtime {

struct PlatoSpec {
  int layers = 4, heads = 4, d = 64, n_rbf = 16, n_basis = 8, frames = 4, n_charges = 1;
  int couple_at = 1, mesh_channels = 8, n_types = 0;
  float cutoff = 5.0f, mesh_sigma = 1.5f, les_sigma = 1.0f, les_kcut = 6.0f, ln_eps = 1e-5f;
  bool mesh = true, les = true;
  // electronegativity term E_chi = sum_i chi_i q_i on the first charge channel (metatrain
  // feat/plato-chi-frozen): chi = chi0[Z] + dchi, dchi = b tanh(head(h)) when b > 0
  bool chi = false;
  float chi_bound = 0.0f;
  // charge / spin conditioning (metatrain's PET SystemConditioningEmbedding on Plato,
  // hypers system_conditioning / max_charge / max_spin_multiplicity): the projected
  // embedding of the structure's charge and spin multiplicity, added to the centre
  // tokens after every layer (before the mesh coupling of layer couple_at), weights
  // system_conditioning.{charge_embedding,spin_multiplicity_embedding,project.0,project.2}
  bool conditioning = false;
  int max_charge = 10, max_spin_multiplicity = 10;
  std::vector<float> frame_w;        // [F]
  std::vector<double> frame_R;       // [F][3][3], R_f
  std::vector<double> omega;         // [d/h/2][3]
  // phase 2 (training): every FrameLinear folded in the graph from its raw W and
  // phi (gradients reach the checkpoint's own parameters), LayerNorms of
  // primitive ops (gradients w.r.t. their weights). Otherwise the folds are
  // weights "fold.<prefix>" computed once on the host (plato_fold_weights)
  bool fold_in_graph = false, decomposed_layer_norm = false;
  int head_dim() const { return d / heads; }
};

/// the graph's per-call inputs (shapes for n atoms, E edges, T = n + E tokens)
struct PlatoInputs {
  ggml_tensor *species = nullptr;          // I32 [n], species index
  ggml_tensor *edge_center_species = nullptr; // I32 [E]
  ggml_tensor *neighbor_species = nullptr; // I32 [E]
  ggml_tensor *edge_vectors = nullptr;     // F32 [3, E] (param: forces, virial)
  ggml_tensor *reverse = nullptr;          // I32 [T], token row of the reverse edge (centres: their own)
  ggml_tensor *offsets = nullptr;          // I32 [n + 1]
  ggml_tensor *centers = nullptr;          // I32 [E]
  ggml_tensor *zero_ang = nullptr;         // F32 [d/h/2 F, n], zeros (the centres' RoPE angles)
  ggml_tensor *rbf0 = nullptr;             // F32 [n_rbf], the radial basis of the centre token (d = 1e-6)
  ggml_tensor *rel_identity = nullptr;     // F32 [9] (fold_in_graph)
  ggml_tensor *charge_index = nullptr;     // I32 [n] (conditioning): charge + max_charge of the atom's structure
  ggml_tensor *spin_index = nullptr;       // I32 [n] (conditioning): 2S+1 - 1
  // structures of the batch: atoms of structure g are consecutive
  int n_struct = 1;
  // every structure of the batch with the same |k| sets and no strain derivative
  // (rotated copies of one cell, no stress loss): one mesh filter and one Ewald g(k)
  // for all, from mesh_k / les_k / strain / inv_volume of one structure ([.., 1])
  bool shared_filter = false;
  ggml_tensor *struct_index = nullptr;     // I32 [n], structure of each atom
  ggml_tensor *energy_weights = nullptr;   // F32 [1, G]: total = sum_g w_g E_g
  ggml_tensor *charge_weights = nullptr;   // F32 [n, 1] (training, optional): total += sum_i c_i q_i,
                                           // first latent charge channel
  // long range (periodic batches: every structure on the same mesh and the same
  // integer k set; open systems: one structure)
  ggml_tensor *positions = nullptr;        // F32 [3, n] (param)
  ggml_tensor *strain = nullptr;           // F32 [3, 3, G], row-major eps per structure (param, zeros)
  ggml_tensor *atom_inv_cell_t = nullptr;  // F32 [3, 3, n]: frac_i = mul_mat(inv_cell_t of its structure, x_i)
  ggml_tensor *inv_volume = nullptr;       // F32 [1, G], 1 / |det cell|
  ggml_tensor *onehot = nullptr;           // F32 [n, G], atom in structure
  ggml_tensor *onehot_t = nullptr;         // F32 [G, n]
  ggml_tensor *inv_count = nullptr;        // F32 [1, G], 1 / atoms
  int mesh_size[3] = {0, 0, 0};            // periodic mesh coupling
  ggml_tensor *mesh_k = nullptr;           // F32 [3, M0 M1 (M2/2+1), G], 2 pi m inv(cell)^T
  ggml_tensor *mesh_k0_pad = nullptr;      // F32 [1, ...], 1 at k = 0, else 0
  ggml_tensor *mesh_deconv = nullptr;      // F32 [1, ...], (k != 0) / prod_a ((4 + 2 cos(2 pi m_a / M_a)) / 6)^2
  ggml_tensor *les_k = nullptr;            // F32 [3, nk, G], half-space k vectors with |k| < kcut / sigma
  ggml_tensor *les_n = nullptr;            // F32 [3, nk], 2 pi n (phases 2 pi n . frac)
  ggml_tensor *les_taper = nullptr;        // F32 [1, nk], unused (the window is clamped); kept for callers
  // LES by smooth PME instead (les_mesh[0] > 0; one charge channel): order-les_order B-spline
  // spread of q on the mesh, rfftn, ggml_plato_pme with the metric A = K'^T K', K' = (1 - eps) K
  int les_mesh[3] = {0, 0, 0};
  int les_order = 8;
  ggml_tensor *les_K = nullptr;            // F32 [3, 3, G]: column b = 2 pi inv(cell)^T e_b (k of m = e_b)
  ggml_tensor *les_tab = nullptr;          // F32 [M0 + M1 + M2/2 + 1], plato::pme_axis_table
  ggml_tensor *offdiag = nullptr;          // F32 [n, n], open-system LES (one structure): 1 - I
};

struct PlatoOutputs {
  ggml_tensor *energies = nullptr;    // [1, G] unscaled energy per structure
  ggml_tensor *total = nullptr;       // [1] sum_g w_g E_g (+ sum_i c_i q_i): differentiate this
  ggml_tensor *short_range = nullptr; // [1, G] energy head part
  ggml_tensor *les = nullptr;         // [1, G] LES part (or nullptr)
  ggml_tensor *chi = nullptr;         // [1, G] electronegativity part (or nullptr)
  ggml_tensor *chi_atoms = nullptr;   // [1, n] chi_i (eV / e)
  ggml_tensor *charges = nullptr;     // [n, Q] centred latent charges (or nullptr): an
                                      // electronegativity term sum_i chi_i q_i adds here
};

/// weights by checkpoint (state dict) name, plus "fold.<prefix>" for the host folds
PlatoOutputs build_plato(ggml_context *ctx, const PlatoSpec &spec,
                         const std::map<std::string, ggml_tensor *> &weights, const PlatoInputs &in, int n,
                         int64_t E, int max_len, bool periodic);

/// host folds W_eff = sum_b phi_b(I) W_b (double) of every FrameLinear prefix in
/// `raw` (name -> values, torch shapes), as "fold.<prefix>" [c_out][c_in]
std::map<std::string, std::vector<float>> plato_fold_weights(
    const std::map<std::string, std::vector<float>> &raw,
    const std::map<std::string, std::vector<int64_t>> &shapes, std::vector<std::string> &prefixes);

} // namespace mlipcpp::runtime
