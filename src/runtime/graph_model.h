#pragma once

#include "core/backend.h"
#include "core/conditioning.h"
#include "core/coulomb.h"
#include "graph_interpreter.h"
#include "pet_ragged.h"
#include "mlipcpp/model.h"
#include "mlipcpp/neighbor_list.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_backend;
struct ggml_backend_buffer;

typedef struct ggml_backend *ggml_backend_t;
typedef struct ggml_backend_buffer *ggml_backend_buffer_t;
typedef struct ggml_gallocr *ggml_gallocr_t;

namespace mlipcpp::runtime {

struct CachedGraph;

/**
 * Model implementation using auto-exported computation graphs.
 *
 * Wraps GraphInterpreter to provide the standard Model interface,
 * enabling automatic PyTorch -> GGML model conversion.
 *
 * Supports dynamic system sizes: the graph is exported with symbolic
 * dimensions (n_atoms, max_neighbors) that are resolved at runtime.
 *
 * Usage:
 *   GraphModel model;
 *   model.load_from_gguf("model.gguf");
 *   ModelResult result = model.predict(system);
 *   ModelResult result_f = model.predict(system, true); // with forces
 */
class GraphModel : public Model {
public:
  GraphModel();
  ~GraphModel() override;

  // Model interface
  ModelResult predict(const AtomicSystem &system) override;
  ModelResult predict(const AtomicSystem &system, bool compute_forces) override;
  std::string model_type() const override { return "graph"; }
  float cutoff() const override { return cutoff_; }
  bool conditioned() const override { return conditioning_.enabled; }
  int max_charge() const override { return conditioning_.max_charge; }
  int max_spin_multiplicity() const override { return conditioning_.max_spin_multiplicity; }

  /**
   * Load model from GGUF file.
   *
   * The GGUF file must contain:
   * - Weights as tensors
   * - Graph JSON in metadata field "graph.json"
   * - Model hyperparameters (cutoff, species map, etc.)
   */
  bool load_from_gguf(const std::string &path);

  /**
   * Load graph from separate JSON file (for testing).
   */
  void load_graph_file(const std::string &path);

  /**
   * Set a weight tensor manually (for testing).
   */
  void set_weight(const std::string &name, ggml_tensor *tensor);

  /**
   * Set backend preference.
   */
  void set_backend_preference(BackendPreference pref) {
    backend_preference_ = pref;
  }

  /**
   * Get the underlying graph interpreter for inspection.
   */
  const GraphInterpreter &interpreter() const { return interp_; }

  // --- Training access (src/train) ---

  /// GPU index among those matching the backend preference; set before load.
  void set_device(int device) { device_ = device; }

  /// One MD step on arrays that live on the GPU (mlipcpp_compute_device, for
  /// LAMMPS's KOKKOS package): positions and types read and forces added on the
  /// device, on the caller's stream; one synchronisation per call (the totals),
  /// more only when the neighbour list is rebuilt. CUDA builds, forces-mode
  /// models of the cached path (predict_cached).
  struct DeviceStep {
    int n_atoms = 0;
    const double *positions = nullptr;    // device; (i, c) at positions[i * s0 + c * s1]
    int64_t position_strides[2] = {3, 1};
    const int32_t *types = nullptr;       // device [n_atoms], in [0, n_types]
    int n_types = 0;
    const int32_t *type_atomic_numbers = nullptr; // host [n_types + 1]; 0: no element
    const double *cell = nullptr;         // host [9], lattice vectors as rows; nullptr: none
    bool pbc[3] = {false, false, false};
    double *forces = nullptr;             // device, added into; (i, c) at forces[i * s0 + c * s1]
    int64_t force_strides[2] = {3, 1};
    void *stream = nullptr;               // cudaStream_t; nullptr: the legacy default stream
    bool rebuild = false;                 // the atoms may have been reordered or remapped
    double skin = -1;                     // Verlet skin (A); < 0: MLIPCPP_NL_SKIN or 0.5
    bool virial = false;
    int charge = 0, spin_multiplicity = 1; // conditioned models (GGUF pet.system_conditioning)
  };
  struct DeviceTotals {
    double energy = 0;
    double virial[6] = {0, 0, 0, 0, 0, 0}; // -sum over edges v (x) dE/dv: xx yy zz yz xz xy (eV)
    bool rebuilt = false;                  // the neighbour list was built in this call
    int attempts = 1;                      // > 1: computed again (graph too small, list stale)
  };
  bool device_step_supported() const;
  DeviceTotals predict_device_step(const DeviceStep &step);

  GraphInterpreter &interpreter() { return interp_; }
  ggml_backend_t backend() const { return compute_backend_; }

  /// Weight tensors by GGUF name, in GGUF order (all F32, on the backend).
  std::vector<std::pair<std::string, ggml_tensor *>> weights() const;
  /// Conservative forces and d(total physical energy)/d(parameter), flattened
  /// in param_names() order. Includes the P3M charge adjoint. Intended for
  /// experimental native training; parameter flags are restored on return.
  ModelResult predict_with_parameter_gradients(const AtomicSystem &system,
                                              std::vector<float> &gradients);
  /// The backend buffer holding every weight tensor (one allocation).
  ggml_backend_buffer_t weight_buffer() const { return weight_buffer_; }

  const std::map<int, int> &species_to_index() const { return species_to_index_; }
  const std::map<int, float> &composition_energies() const { return composition_energies_; }
  const std::map<int, float> &nc_force_scales() const { return nc_force_scales_; }
  float energy_scale() const { return energy_scale_; }
  float cutoff_width() const { return cutoff_width_; }
  /// PET adaptive cutoff: target neighbour count (0 = fixed cutoff), the
  /// solver's taper width and the method ("solver" or "grid"; GGUFs written
  /// before these keys existed: the cutoff width and "grid")
  float num_neighbors_adaptive() const { return num_neighbors_adaptive_; }
  float cutoff_width_adaptive() const { return cutoff_width_adaptive_; }
  const std::string &adaptive_cutoff_method() const { return adaptive_cutoff_method_; }
  bool forces_mode() const { return forces_mode_; }
  /// Graph output is [n_atoms, 4] = (energy | non-conservative force).
  bool uses_p3m() const { return p3m_; }
  const CoulombParameters &coulomb_parameters() const { return coulomb_parameters_; }
  bool long_range() const { return long_range_; }
  bool nc_forces() const { return nc_forces_; }
  const std::string &gguf_path() const { return gguf_path_; }
  /// Names of the trainable tensors (GGUF "train.param_names"; buffers and
  /// constants excluded). Empty for exports that predate it.
  const std::vector<std::string> &param_names() const { return param_names_; }
  /// GGUF tensor name -> metatrain state-dict name ("train.state_dict_names";
  /// empty for exports that predate it)
  const std::map<std::string, std::string> &state_dict_names() const { return state_dict_names_; }
  /// charge / spin conditioning (GGUF pet.system_conditioning): the graph has the
  /// inputs "charge_index" and "spin_index" (I32 [n_atoms], table rows)
  const Conditioning &conditioning() const { return conditioning_; }

  /**
   * Run inference and invoke `cb` after the graph compute completes, while
   * the compute buffer is still alive. The callback typically calls
   * interp_.capture_all_outputs(...) to read intermediate tensors. Forces
   * are not computed in this path (the proposed visualisation contract is
   * energy-only; see ACTIVATION_CAPTURE.md).
   */
  ModelResult predict_with_capture(
      const AtomicSystem &system,
      const std::function<void(GraphInterpreter &)> &cb);

private:
  GraphInterpreter interp_;

  // Model hyperparameters
  float cutoff_ = 4.5f;
  float cutoff_width_ = 0.2f;
  float energy_scale_ = 1.0f;
  std::string cutoff_function_ = "cosine";
  bool forces_mode_ = false;
  bool nc_forces_ = false;
  bool long_range_ = false;
  bool p3m_ = false;
  CoulombParameters coulomb_parameters_;
  int device_ = 0;
  std::string gguf_path_;
  std::vector<std::string> weight_names_;
  std::vector<std::string> param_names_;
  std::map<std::string, std::string> state_dict_names_;
  Conditioning conditioning_;
  float num_neighbors_adaptive_ = 0.0f;
  float cutoff_width_adaptive_ = 0.0f;
  std::string adaptive_cutoff_method_ = "grid";

  BackendPreference backend_preference_ = BackendPreference::Auto;

  // GGML contexts and backend
  ggml_context *ctx_weights_ = nullptr;
  std::shared_ptr<BackendProvider> backend_provider_;
  ggml_backend_buffer_t weight_buffer_ = nullptr;
  ggml_backend_t compute_backend_ = nullptr;

  // Species mapping (atomic number -> index)
  std::map<int, int> species_to_index_;

  // Composition energies (atomic reference energies)
  std::map<int, float> composition_energies_;

  // Per-species scale of the non-conservative force head
  std::map<int, float> nc_force_scales_;

  // Neighbor list builder
  NeighborListBuilder neighbor_builder_;

  // Forces-mode models without long-range terms (fixed or "solver"
  // adaptive cutoff) go through predict_cached: a graph built once per
  // (n_atoms, neighbour slots) and reused, with the neighbour list (vesin when
  // available), adaptive cutoffs and force gathering threaded over atoms.
  // MLIPCPP_NO_GRAPH_CACHE=1 forces the general path below.
  bool cached_path_supported() const;
  ModelResult predict_cached(const AtomicSystem &system, bool compute_forces);
  // the same on a CUDA backend, with the inputs built and the forces
  // gathered on the device (MLIPCPP_HOST_INPUTS=1: on the host)
  ModelResult predict_cached_device(const AtomicSystem &system, bool compute_forces,
                                    const std::vector<int32_t> &species);
  void build_cached_graph(int n_atoms, int M, bool exact_mask, bool backward);

  // Packed edges instead of n_atoms x M slots (pet_ragged.h): PET models of
  // the architecture it builds, with MLIPCPP_RAGGED=1. num_heads_ is
  // pet.num_heads, or the exported attention's scale gives it.
  int num_heads_ = 0;
  int ragged_state_ = -1; // -1 unchecked, 0 no, 1 yes
  PetRaggedSpec ragged_spec_;
  std::map<std::string, ggml_tensor *> ragged_weights_;
  ggml_context *ctx_ragged_fold_ = nullptr; // pet_ragged_fold_weights
  ggml_backend_buffer_t ragged_fold_buffer_ = nullptr;
  bool use_ragged();
  void build_ragged_graph(int n_atoms, int64_t e_cap, int max_len, bool backward);
  // rebuilds cached_ unless it holds n_atoms, `edges` packed edges and sequences of
  // 1 + max_kept tokens; true if rebuilt
  bool fit_ragged_graph(int n_atoms, int64_t edges, int max_kept);
  // rebuilds cached_ unless it fits n_atoms with max_kept neighbours; true if rebuilt
  bool fit_cached_graph(int n_atoms, int max_kept);
  // energy and non-conservative forces of the last compute of cached_
  void read_cached_outputs(const AtomicSystem &system, ModelResult &result) const;
  // the conditioning inputs of g for (charge, spin multiplicity), when they changed
  void upload_conditioning(CachedGraph &g, int charge, int spin_multiplicity) const;
  struct CachedGraphDeleter {
    void operator()(CachedGraph *graph) const;
  };
  std::unique_ptr<CachedGraph, CachedGraphDeleter> cached_;
  // allocates g's inputs, constants and compute buffer, then makes it cached_
  void finish_cached_graph(std::unique_ptr<CachedGraph, CachedGraphDeleter> g,
                           const std::vector<ggml_tensor *> &inputs,
                           std::vector<std::pair<ggml_tensor *, float>> constants);
  ggml_gallocr_t galloc_ = nullptr;
  void *vesin_ = nullptr;          // VesinNeighborList, reused between calls
  void *device_inputs_ = nullptr;  // predict_cached_device's buffers
  void release_cached_resources(); // galloc_, vesin_, device_inputs_

  // Predict a single system (all logic lives here).
  // post_compute_hook (if set) runs after a successful graph compute and
  // before the compute buffer is freed — this is the window during which
  // intermediate tensor data is live on the backend buffer.
  ModelResult predict_single(
      const AtomicSystem &system, bool compute_forces,
      const std::function<void()> &post_compute_hook = nullptr,
      std::vector<float> *parameter_gradients = nullptr);
};

} // namespace mlipcpp::runtime
