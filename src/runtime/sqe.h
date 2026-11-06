#pragma once
// Split-charge equilibration (metatrain plato_lr sqe, modules/sqe.py; GGUF pet.plato_lr.sqe*). Host side, float64.
//
//   E_SQE(q) = sum_i [chi_i q_i + J_i q_i^2 / 2] + sum_{i<j} q_i q_j gamma_ij + sum_pairs kappa_ij p_ij^2 / 2,
//   q = q0 + sum_j p_ij (p_ij = -p_ji on the pairs closer than r_ct), q0 = Q / N or per-atom formal charges.
//
// Bounded parameters (no free amplitude): chi_i = chi0_Z + dchi_max tanh(a_Z + b_Z (sqrt(CN_i + 1) - 1)) (chi0 the
// Mulliken electronegativity, CN the D3 coordination number); J_i = J0_Z exp(lJ + hJ tanh(c_Z)) in [J_lo, J_hi] J0
// (J0 = IP - EA); bond softness s = 1 / kappa = exp(-(r - r0_ij) / w) f_c(r) / sqrt(kappa_i kappa_j) (kappa_Z
// log-uniform in [kappa_lo, kappa_hi], w in [w_lo, w_hi], f_c the C2 switch to 0 at r_ct); gamma = KE erf(r / sqrt(
// sigma_i^2 + sigma_j^2)) / r with sigma_i = KE sqrt(2 / pi) / (J_lo J0_Z), so the minimum is unique.
//
// The minimum over u = p / sqrt(s) solves (I + B^T A B) u = -B^T (chi + A q0), A = diag(J) + gamma, B the pair
// incidence scaled by sqrt(s) (Jacobi-preconditioned CG). Forces are Hellmann-Feynman (dE/dx at fixed u). Training:
// dE*/dtheta at fixed u (envelope); a force-loss contraction sum_a w_a . F_a gets its theta gradient -D_x(dE*/dtheta)[w]
// from one more (tangent) solve, H du = -dg/dx[w], with dual numbers along w: the implicit-function (adjoint) term.
// Open structures only (periodic: Ewald matrix, inference only).

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace mlipcpp {
class GGUFLoader;
}

namespace mlipcpp::runtime {

struct SqeSpec {
  bool enabled = false;
  bool q0_formal = false;        // pet.plato_lr.sqe_q0 == "formal"
  bool backbone_neutral = false; // pet.plato_lr.sqe_backbone_charge == "neutral": the backbone sees charge 0
  double r_ct = 6.0, r_ct_width = 1.0, dchi_max = 2.0, J_lo = 0.75, J_hi = 2.0, kappa_lo = 1.0, kappa_hi = 100.0,
         w_lo = 0.2, w_hi = 2.0, cn_on = 4.0, cn_off = 5.0, ewald_alpha = 0.0;
  // per element (sqe_Z order): tabulated data (fixed) and the raw learned parameters
  std::vector<int32_t> Z;
  std::vector<double> chi0, J0, rcov, sigma;
  std::vector<double> chi_a, chi_b, j_z, kappa_z;
  double w_raw = 0.0;
  double cg_tol = 1e-12; // relative residual
  int index_of(int z) const; // -1: not in the table
  size_t n_params() const { return 4 * Z.size() + 1; }
  // flat order: chi_a, chi_b, j_z, kappa_z (each per element), w_raw
  std::vector<double> flat() const;
  void set_flat(const std::vector<double> &v);
  std::vector<std::string> flat_names() const; // "sqe.chi_a.<Z>", ...
  static SqeSpec from_gguf(const GGUFLoader &L);
  void to_gguf(gguf_context *g) const;
  // bounded maps
  double J(int e) const;
  double kappa0(int e) const;
  double w() const;
  /// ranges against the bounds: tanh(a) (chi offset / dchi_max at CN 0), b, J / J0, kappa0 (eV), w (A)
  std::string summary() const;
};

struct SqeOutput {
  double energy = 0;          // eV
  std::vector<double> q;      // e
  std::vector<double> forces; // -dE/dx, eV/A [3n] (when asked)
  std::vector<double> chi, J; // per atom
  int cg_iters = 0;
};

/// one structure. x [3n] A (float or double via the overloads), Z atomic numbers, Q total charge, q0 per-atom reference
/// (nullptr: Q / N). cell: 9 values (rows = lattice vectors) or nullptr for open.
SqeOutput sqe_compute(const SqeSpec &spec, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                      bool forces, const double *cell = nullptr);
SqeOutput sqe_compute(const SqeSpec &spec, int n, const int32_t *Z, const float *x, double Q, const double *q0,
                      bool forces, const double *cell = nullptr);

/// training: adds to grad (size n_params) dE/dtheta * dL_dE + d(sum_a w_a . F_a)/dtheta with w = dL_dF [3n] (nullptr:
/// no force term); also returns the forward output (energy, forces). Open structures only.
SqeOutput sqe_train_grad(const SqeSpec &spec, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                         double dL_dE, const double *dL_dF, std::vector<double> &grad);

} // namespace mlipcpp::runtime
