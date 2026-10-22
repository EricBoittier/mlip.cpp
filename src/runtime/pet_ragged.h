#pragma once
// PET (metatrain's feedforward featurizer: one PreLN transformer layer with
// RMSNorm and a sigmoid-gated MLP per GNN layer, the bump cutoff) built
// directly in ggml on packed edges instead of n_atoms x max_neighbours slots.
// Node rows are [d, n]; edge rows [d, e_cap] hold atom i's kept edges at
// [offsets[i], offsets[i + 1]), then rows of no atom (edges at the cutoff,
// fc = 0) up to e_cap. Attention runs per atom over its central token and
// its edges (ggml_pet_attn_ragged). Forces come from ggml's backward pass.

#include <cstdint>
#include <map>
#include <string>

struct ggml_context;
struct ggml_tensor;

namespace mlipcpp::runtime {

struct PetRaggedSpec {
  int layers = 0, heads = 0;
  int64_t d_pet = 0, d_node = 0;
  float rms_eps = 1e-8f, ln_eps = 1e-5f;
  float cutoff_width = 1.0f; // of the bump function
};

struct PetRaggedInputs {
  ggml_tensor *species = nullptr;          // I32 [n], model species index
  ggml_tensor *neighbor_species = nullptr; // I32 [e_cap]
  ggml_tensor *edge_vectors = nullptr;     // F32 [3, e_cap]
  ggml_tensor *cutoff_values = nullptr;    // F32 [e_cap], pair cutoffs
  ggml_tensor *reverse = nullptr;          // I32 [e_cap], row of the reverse edge
  ggml_tensor *offsets = nullptr;          // I32 [n + 1]
  ggml_tensor *centers = nullptr;          // I32 [e_cap], atom of each row
};

struct PetRaggedOutputs {
  ggml_tensor *atom_energy = nullptr; // [1, n], unscaled (forward only)
  ggml_tensor *total = nullptr;       // [1], unscaled sum: differentiate this
};

/// The architecture of a PET model's weights (by GGUF name). False, with the
/// reason, when they are not the feedforward-featurizer PET built here.
bool pet_ragged_spec(const std::map<std::string, ggml_tensor *> &weights, int64_t d_pet, int heads,
                     float cutoff_width, PetRaggedSpec &spec, std::string &why);

PetRaggedOutputs build_pet_ragged(ggml_context *ctx, const PetRaggedSpec &spec,
                                  const std::map<std::string, ggml_tensor *> &weights, const PetRaggedInputs &in,
                                  int n_atoms, int64_t e_cap, int max_len);

} // namespace mlipcpp::runtime
