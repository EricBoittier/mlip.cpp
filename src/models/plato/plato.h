#pragma once
// Plato-PET-LR (metatrain experimental.plato_pet_lr, frame_mode: attention):
// GGUFs from scripts/plato/export_plato_gguf.py. Energy, forces and stress from
// one ggml graph and its backward pass (runtime/plato_graph.h).

#include "core/backend.h"
#include "mlipcpp/model.h"
#include "runtime/plato_graph.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;
struct ggml_cgraph;
struct ggml_gallocr;

namespace mlipcpp::plato {

struct PlatoOptions {
  BackendPreference backend = BackendPreference::Auto;
  int device = 0;
  // the mesh coupling's mesh; 0: PyTorch's rule max(8, ceil(|a_i| / spacing)) per
  // cell vector, which jumps under rotation or strain where |a_i| / spacing crosses
  // an integer (an integer-length box rotated by a generic rotation: one more point).
  // MLIPCPP_PLATO_MESH="M" or "M0,M1,M2" sets it too. Off by default (parity).
  int fixed_mesh[3] = {0, 0, 0};
  bool mesh = true, les = true; // the model's own switches (parity of the parts)
  bool fold_in_graph = false;   // training layout (MLIPCPP_PLATO_FOLD_IN_GRAPH=1)
  bool decomposed_layer_norm = false; // LayerNorms of primitive ops (weight gradients)
};

/// full neighbour list grouped by centre and sorted, with reverse edges
struct Edges {
  std::vector<int64_t> start;            // [n + 1]
  std::vector<int32_t> j, rev;           // rev: the edge j -> i with the opposite shift
  std::vector<std::array<int32_t, 3>> shift;
  std::vector<float> v;                  // [E, 3], x_j - x_i + shift . cell
};
Edges neighbor_edges(const AtomicSystem &system, double cutoff, void *&vesin_state);
void free_vesin(void *&vesin_state);
/// mesh size by PyTorch's rule from a cell row (|a| in double)
int mesh_points(const float *cell_row, double spacing);

class PlatoModel : public Model {
public:
  explicit PlatoModel(PlatoOptions options = {});
  ~PlatoModel() override;

  bool load_from_gguf(const std::string &path);
  ModelResult predict(const AtomicSystem &system) override { return predict(system, true); }
  ModelResult predict(const AtomicSystem &system, bool compute_forces) override;
  std::string model_type() const override { return "plato-pet-lr"; }
  float cutoff() const override { return spec_.cutoff; }

  PlatoOptions &options() { return options_; }
  const runtime::PlatoSpec &spec() const { return spec_; }
  // parts of the last energy (eV, scaled): energy head, LES; and the mesh it used
  struct Parts {
    double short_range = 0, les = 0, composition = 0;
    int mesh[3] = {0, 0, 0};
    int64_t n_k = 0, edges = 0;
    int max_len = 0;
    bool rebuilt = false;
    double nl_ms = 0, prep_ms = 0, compute_ms = 0, read_ms = 0; // host neighbour list, inputs (+ graph build), graph, outputs
  };
  const Parts &last_parts() const { return parts_; }
  const char *backend_name() const;

  // training access (src/train/plato_train.cpp)
  const std::map<std::string, ggml_tensor *> &weights() const { return weights_; }
  std::shared_ptr<BackendProvider> backend() const { return backend_; }
  double energy_scale() const { return energy_scale_; }
  double mesh_spacing() const { return mesh_spacing_; }
  const std::map<int, double> &composition() const { return composition_; }
  const std::map<int, int> &species_index() const { return species_index_; }

private:
  struct Graph;
  PlatoOptions options_;
  runtime::PlatoSpec spec_;
  double energy_scale_ = 1.0, mesh_spacing_ = 1.0;
  std::map<int, double> composition_;
  std::map<int, int> species_index_;
  std::shared_ptr<BackendProvider> backend_;
  ggml_context *ctx_w_ = nullptr;
  ggml_backend_buffer *buf_w_ = nullptr;
  std::map<std::string, ggml_tensor *> weights_;
  std::unique_ptr<Graph> graph_;
  ggml_gallocr *galloc_ = nullptr;
  void *vesin_ = nullptr;
  Parts parts_;
  // the cell-dependent long-range constants of the last cell (k vectors of the
  // mesh and of LES, deconvolution); rebuilt and uploaded only when the cell changes
  struct CellConsts {
    float cell[9] = {};
    bool valid = false, periodic = false;
    int mesh[3] = {0, 0, 0};
    double inv[3][3] = {}, volume = 0;
    std::vector<float> mesh_k, mesh_pad, mesh_deconv, les_k, les_n, les_taper;
  } consts_;
  bool consts_uploaded_ = false;
};

} // namespace mlipcpp::plato
