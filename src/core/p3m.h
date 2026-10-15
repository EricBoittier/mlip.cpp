#pragma once
#include "coulomb.h"
namespace mlipcpp {
struct P3MResult {
  std::vector<float> potentials, charge_gradient;
  std::vector<double> forces, stress;
};
// Atom-major charge channels. The optional adjoint is dE/d(potentials).
// Periodic mode uses a power-of-two FFT mesh and fifth-order assignment;
// isolated systems use the excluded direct potential without a dense matrix.
P3MResult p3m(const AtomicSystem &system, const NeighborList &neighbors,
              const CoulombParameters &parameters, int channels,
              const std::vector<float> &charges,
              const std::vector<float> *adjoint = nullptr);
} // namespace mlipcpp
