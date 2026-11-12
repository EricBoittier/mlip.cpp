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
  // periodic graph models: the full virial dE/d(strain) [9], row-major, eV, before symmetrisation (the backbone
  // and its adaptive-cutoff chain; the Voigt stress is sym(virial) / V plus any long-range terms); empty otherwise
  std::vector<double> virial;

  std::vector<float> nc_forces; // [n_atoms * 3], direct force head
  // Plato-PET-LR with LES: the latent charges per atom (e: model units times charge_unit,
  // after the neutralisation / total-charge shift), first channel; empty otherwise
  std::vector<float> latent_charges;
  // split-charge equilibration (pet.plato_lr.sqe): the SQE charges per atom (e); empty otherwise
  std::vector<float> sqe_charges;
  // graph models: the per-atom model outputs (scaled to eV, without the composition energies)
  std::vector<float> atomic_energies;

  /// iris PET-LR (pet.iris_lr): physical charges (e) [n_atoms], dipole (e A) about the dipole origin,
  /// Fermi level mu and metallicity sum_i f'(s_i - mu) (fermi / both references; 0 otherwise)
  std::vector<float> charges;
  std::vector<double> dipole;
  double fermi_level = 0.0, metallicity = 0.0;
  bool has_charges = false;
  /// threshold reference (his feat/threshold-reference): the residual its species spread supplied,
  /// R = sum_i (c_Z + f_i) - Q (e)
  double charge_residual = 0.0;
  bool has_charge_residual = false;
  /// conserving charges (his feat/conserving-charges): the dipole of the edge flows, P_flow = -1/2 sum_e F_e R_e (e A)
  std::vector<double> dipole_flow;
  /// induced dipoles (his induced_dipoles, feat/conserving-charges): sum_i mu_i (e A) and the polarisabilities
  /// alpha_i (A^3) per atom
  std::vector<double> dipole_induced;
  std::vector<float> polarisabilities;
  /// under a nonzero field (field-input models): his field force E . Z0 [3 n] (eV/A), already in `forces`
  std::vector<double> field_forces;
  /// conserving charges on several components: each atom's component and their charges Q_c (mlip.cpp's R-only rule)
  std::vector<int> components;
  std::vector<double> component_charges;

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
