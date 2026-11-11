#pragma once
// fp64 evaluation of iris GGUFs (trunk-only and PET-LR, open / 3-D / slab, every ported reference) on the fp64
// executor (gir64.h): the host side of GraphModel::predict_single_impl's iris path, in double from end to end
// (positions, cell, edge vectors, adaptive cutoffs, pair cutoffs, Ewald inputs, mu, the tie-split weights, QEq cg,
// the field force passes). The GGUF's weights (float32) are cast exactly to double; float32 GGUF metadata that
// enters continuous host maths (charge temperature, fermi eps, QEq tol) is read as the decimal it rounds
// (f32_intent: 0.1f -> 0.1). The graphs' RMS_NORM eps finfo(float32).eps runs as finfo(float64).eps.
//
// Selection: MLIPCPP_EXEC=fp64 (GraphModel on any caller: float positions promoted exactly, results rounded to the
// API's float), or mlipcpp-eval --fp64 (double positions / cell from the extxyz text, double JSON).

#include "gir64.h"
#include "graph_model.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlipcpp::runtime::fp64 {

struct System64 {
  int n = 0;
  std::vector<double> positions; // [n, 3]
  std::vector<int32_t> numbers;
  bool has_cell = false;
  double cell[3][3] = {};
  bool pbc[3] = {false, false, false};
  int charge = 0, spin_multiplicity = 1;
  std::optional<std::array<double, 3>> dipole_origin, field;
};

struct Result64 {
  double energy = 0;
  std::vector<double> atomic_energies, forces, stress, virial; // stress Voigt (xx yy zz yz xz xy) eV/A^3
  bool has_forces = false, has_stress = false;
  // iris PET-LR
  bool has_charges = false;
  std::vector<double> charges, dipole, dipole_flow;
  double fermi_level = 0, metallicity = 0, charge_residual = 0;
  bool has_charge_residual = false;
  int qeq_forward_iterations = -1, qeq_cotangent_iterations = -1;
  /// the requested column gradients (IrisFp64::request_column_gradient): d(sum_i w_i out[i, column])/d r [n, 3]
  /// (edge scatter, explicit positions and the adaptive-cutoff chain), one per request
  std::vector<std::vector<double>> column_gradients;
  /// conserving charges on several components: each atom's component and the Q_c of mlip.cpp's R-only rule
  std::vector<int> components;
  std::vector<double> component_charges;
  /// induced dipoles: sum_i mu_i (e A) and alpha_i (A^3) per atom; under a nonzero field his E . Z0 (in forces too)
  std::vector<double> dipole_induced, polarisabilities, field_forces;
};

/// the full graph, its inputs and dimensions of the last predict (tests: forward mode on the real graphs)
struct Snapshot {
  const GIRGraph *graph = nullptr;
  std::map<std::string, Tensor<double>> inputs;
  std::map<std::string, int64_t> dims;
};

class IrisDriver;

/// The fp64 evaluator of one GraphModel (its GGUF re-read for the weights and graphs; the rest from the model's
/// own settings, so ModelOptions / set_iris_* apply). Not thread safe (the frozen Ewald state).
class IrisFp64 {
public:
  explicit IrisFp64(const GraphModel &model);
  ~IrisFp64();
  Result64 predict(const System64 &s, bool compute_forces);
  /// RMS_NORM eps substitution (default true: finfo(float64).eps; false keeps the trace's finfo(float32).eps)
  void set_fp64_rms_eps(bool on);
  /// with forces: also d(sum_i w_i out[i, column])/d r (column 1 = charges, 0 = energies; not for QEq, whose
  /// charges need the implicit cg term), in Result64::column_gradients
  void request_column_gradient(int column, std::vector<double> w);
  void clear_column_gradients();
  const Snapshot &last() const;
  const std::map<std::string, Tensor<double>> &weights() const;

private:
  std::unique_ptr<IrisDriver> d_;
};

/// MLIPCPP_EXEC=fp64
bool exec_fp64_requested();
/// AtomicSystem -> System64 (exact float -> double)
System64 to_system64(const AtomicSystem &s);
/// Result64 -> ModelResult (rounded to the API's float)
ModelResult to_model_result(const Result64 &r);

} // namespace mlipcpp::runtime::fp64
