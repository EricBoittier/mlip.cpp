#pragma once

#include "mlipcpp/neighbor_list.h"
#include <vector>

namespace mlipcpp {

// PET's excluded Coulomb potential, including torch-pme's factor of 1/2.
// A dense reference solver for small systems; periodic calculations use direct
// Ewald, not the P3M mesh approximation used by metatrain in evaluation mode.
struct CoulombParameters {
  double cutoff = 4.5;
  double smearing = 1.4;
  double wavelength = 1.33;
};

struct CoulombResult {
  std::vector<float> matrix;  // row-major [n_atoms, n_atoms]
  std::vector<double> forces; // -adjoint : d(matrix)/d(positions)
  std::vector<double> stress; // adjoint : d(matrix)/d(strain) / volume, Voigt
};

// adjoint is dE/d(matrix), with no symmetry assumed. When supplied, contract
// analytic derivatives without storing an N^2-by-N Jacobian. The full neighbor
// list must be at parameters.cutoff (before any PET adaptive filtering).
CoulombResult coulomb_matrix(const AtomicSystem &system,
                             const NeighborList &neighbors,
                             const CoulombParameters &parameters,
                             const std::vector<float> *adjoint = nullptr);

} // namespace mlipcpp
