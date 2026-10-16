#pragma once
#include "coulomb.h"
namespace mlipcpp {
struct P3MResult {
  std::vector<float> potentials, charge_gradient, potential_tangent;
  std::vector<double> forces, stress;
};
// Atom-major charge channels. The optional adjoint is dE/d(potentials).
// position_direction computes (D_R K)[direction] charges at fixed charges/cell.
// This symmetric linear operator also supplies its transpose for force
// training. Periodic mode uses a power-of-two FFT mesh and fifth-order
// assignment; isolated systems use the excluded direct potential without a
// dense matrix.
P3MResult p3m(const AtomicSystem &system, const NeighborList &neighbors,
              const CoulombParameters &parameters, int channels,
              const std::vector<float> &charges,
              const std::vector<float> *adjoint = nullptr,
              const std::vector<double> *position_direction = nullptr,
              int threads = 1);
} // namespace mlipcpp
