#pragma once
// Fragment-constrained charges (metatrain plato_lr frag_charges, PETFragCharges; GGUF
// pet.plato_lr.frag_charges, frag_R, frag_t, frag_sigma, frag_scale). Host side, float64.
//
// Units are the bonded fragments of an open structure: two atoms are bonded when closer than
// f (r_i + r_j), r the ASE covalent radii, f = 1.2, or 1.3 when either atom is a transition,
// lanthanide or post-transition metal; alkali metals (Li-Cs) never bond. Units are the connected
// components, numbered by their first atom. Unit u carries the integer charge Q_u, the sum of the
// per-atom formal charges over the unit (only the sums matter).
//   - conditioning: every atom of unit u is conditioned on Q_u (instead of the structure's charge);
//   - charges q_i = Q_u / n_u;
//   - energy (eV) 1/2 KE scale sum_{i != j} q_i q_j erf(r / (sqrt 2 sigma)) / r T(r), T the
//     smootherstep from 0 at R - t to 1 at R: zero inside the receptive field, no parameters.
// Without formal charges nothing changes. Periodic structures: not supported (the caller leaves
// them to whole-structure conditioning and no term).

#include <cstdint>
#include <vector>

namespace mlipcpp::runtime {

struct FragChargesSpec {
  bool enabled = false;
  double R = 8.0, t = 0.5, sigma = 4.0, scale = 1.0;
};

/// unit label per atom (0..U-1, numbered by first atom) by the bond rule; positions [n][3]
std::vector<int32_t> frag_bonded_units(int n, const int32_t *Z, const float *positions, int *n_units = nullptr);

/// per-atom unit charge Q_u (rounded sums of `formal` over each unit), and per-atom q = Q_u / n_u
void frag_unit_charges(int n, const std::vector<int32_t> &units, const float *formal, std::vector<int32_t> &unit_charge,
                       std::vector<double> &q);
/// per-atom q = Q_u / n_u from given units and per-atom unit charges (the training labels)
std::vector<double> frag_atom_charges(int n, const int32_t *units, const int32_t *unit_charge);

/// beyond-R Coulomb energy (eV) of charges q (e) at positions [n][3] (A), forces -dE/dx (eV/A) into
/// `forces` [n][3] when non-null (overwritten)
double frag_coulomb(const FragChargesSpec &spec, int n, const float *positions, const double *q, double *forces);

} // namespace mlipcpp::runtime
