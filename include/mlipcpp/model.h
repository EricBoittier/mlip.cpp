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

  bool has_forces = false;
  bool has_stress = false;
  bool has_nc_forces = false;
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
};

/// Load a model from GGUF file
std::unique_ptr<Model> load_model(const std::string &path);

} // namespace mlipcpp
