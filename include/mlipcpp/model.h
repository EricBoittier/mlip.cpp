#pragma once

#include "system.h"
#include <memory>
#include <string>
#include <vector>

namespace mlipcpp {

/// Result from model inference
struct ModelResult {
  float energy = 0.0f;
  std::vector<float> forces; // [n_atoms * 3], flattened
  std::vector<float> stress; // [6] in Voigt notation (xx, yy, zz, yz, xz, xy)

  std::vector<float> nc_forces; // [n_atoms * 3], direct force head
  // Plato-PET-LR with LES: the latent charges per atom (e: model units times charge_unit,
  // after the neutralisation / total-charge shift), first channel; empty otherwise
  std::vector<float> latent_charges;

  bool has_forces = false;
  bool has_stress = false;
  bool has_nc_forces = false;
};

/// Matrix-multiply precision of one model on CUDA (no effect on other backends).
///
/// TF32 is a process-wide CUDA setting (GGML_CUDA_NO_TF32, read once by ggml at
/// the first CUDA matmul); within a TF32 process a model in FP32 has every matrix
/// product of its graphs (and their gradients) tagged GGML_PREC_F32, which runs it
/// in full FP32. The process runs TF32 when MLIPCPP_TF32 is set or some model asked
/// for TF32 before the first graph was built; otherwise it runs in full FP32
/// (GGML_CUDA_NO_TF32), as without this option. So set the precision of every model
/// (load them) before the first predict: asking for TF32 once the process has
/// committed to full FP32 throws.
enum class MatmulPrecision {
  Default, ///< MLIPCPP_TF32 set: TF32 (geometry products pinned to FP32); otherwise FP32
  FP32,    ///< every product in full FP32, also in a TF32 process
  TF32,    ///< TF32 tensor cores except the FP32-pinned geometry products
};

/// Base interface for all MLIP models
class Model {
public:
  virtual ~Model() = default;

  /// Run inference on a single system (energy only)
  virtual ModelResult predict(const AtomicSystem &system) = 0;

  /// Run inference with optional force computation
  virtual ModelResult predict(const AtomicSystem &system, bool compute_forces) {
    (void)compute_forces; // Default ignores this flag
    return predict(system);
  }

  /// Get model metadata
  virtual std::string model_type() const = 0;
  virtual float cutoff() const = 0;

  /// Charge / spin conditioning: whether the model reads AtomicSystem::charge()
  /// and spin_multiplicity(), and the ranges it accepts ([-max_charge,
  /// max_charge], [1, max_spin_multiplicity]; others are an error)
  virtual bool conditioned() const { return false; }
  virtual int max_charge() const { return 0; }
  virtual int max_spin_multiplicity() const { return 0; }

  /// CUDA matmul precision of this model (see MatmulPrecision). TF32 throws
  /// std::runtime_error if the process already committed to full FP32.
  void set_matmul_precision(MatmulPrecision precision);
  MatmulPrecision matmul_precision() const { return matmul_precision_; }

protected:
  MatmulPrecision matmul_precision_ = MatmulPrecision::Default;
};

/// Load a model from GGUF file
std::unique_ptr<Model> load_model(const std::string &path);

} // namespace mlipcpp
