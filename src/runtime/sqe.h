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
  // pet.plato_lr.sqe_q0 == "learned": q0_i = lo + (hi - lo) sigmoid(f_i + t), f_i = q0_a[Z_i] + sum_Z' q0_b[Z_i, Z']
  // CN_{i,Z'} (element-resolved D3 coordination numbers), t the common shift with sum q0 = Q (bisection + Newton,
  // implicit derivatives); formal charges are then never an input (only optional training supervision)
  bool q0_learned = false;
  bool backbone_neutral = false; // pet.plato_lr.sqe_backbone_charge == "neutral": the backbone sees charge 0
  double r_ct = 6.0, r_ct_width = 1.0, dchi_max = 2.0, J_lo = 0.75, J_hi = 2.0, kappa_lo = 1.0, kappa_hi = 100.0,
         w_lo = 0.2, w_hi = 2.0, cn_on = 4.0, cn_off = 5.0, ewald_alpha = 0.0;
  // lr_R > 0: only the long-range part E*(gamma) - E*(gamma (1 - T)), T the smootherstep from 0 at lr_R - lr_t to 1
  // at lr_R (zero for structures without pairs beyond lr_R - lr_t); charges from the full minimum
  double lr_R = 0.0, lr_t = 1.0;
  // per element (sqe_Z order): tabulated data (fixed) and the raw learned parameters
  std::vector<int32_t> Z;
  std::vector<double> chi0, J0, rcov, sigma;
  std::vector<double> chi_a, chi_b, j_z, kappa_z;
  double w_raw = 0.0;
  std::vector<double> q0_lo, q0_hi, q0_a, q0_b; // per element; q0_b E x E (row: the atom's element)
  double cg_tol = 1e-12; // relative residual
  int index_of(int z) const; // -1: not in the table
  size_t n_params() const { return 5 * Z.size() + 1 + Z.size() * Z.size(); }
  // flat order: chi_a, chi_b, j_z, kappa_z (each per element), w_raw, q0_a (per element), q0_b (E x E)
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
  std::vector<double> q0;     // reference charges (given or learned)
  double q0_loss = 0;         // sqe_train_grad with supervision: weight * mean over units of (sum q0 - Q_u)^2
  int cg_iters = 0;
  std::vector<double> dE_dq0; // dE/dq0_i at fixed geometry (given q0): mu_i (lr_R: the difference of the two minima)
  std::vector<double> dE_dR;  // sqe_compute_masked: dE/dR_ab (s x s, split evenly over ab and ba)
};

/// cluster mask of the long-range-only energy: the local functional uses gamma (1 - T(r) (1 - R_{sn_i sn_j})), so only
/// pairs in different clusters (R < 1) contribute beyond lr_R (runtime/lr_combined.h)
struct SqeClusterMask {
  const int32_t *sn = nullptr; // super-node per atom
  int s = 0;
  const double *R = nullptr; // s x s
};

/// one structure. x [3n] A (float or double via the overloads), Z atomic numbers, Q total charge, q0 per-atom reference
/// (nullptr: Q / N). cell: 9 values (rows = lattice vectors) or nullptr for open.
SqeOutput sqe_compute(const SqeSpec &spec, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                      bool forces, const double *cell = nullptr);
SqeOutput sqe_compute(const SqeSpec &spec, int n, const int32_t *Z, const float *x, double Q, const double *q0,
                      bool forces, const double *cell = nullptr);

/// lr_R > 0 required: E*(gamma) - E*(gamma (1 - T (1 - R))) with the given q0 (not learned); Hellmann-Feynman forces at
/// fixed q0 and R, plus dE_dq0 and dE_dR for the caller's chain rules. Inference / frozen use only.
SqeOutput sqe_compute_masked(const SqeSpec &spec, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                             bool forces, const SqeClusterMask &mask);

/// optional supervision of a learned q0 (training only): unit index per atom (0..n_units-1), the units' integer
/// charges, the loss weight, and `scale` (the loss normalisation, e.g. 1 / structures of the step) on the gradient
struct SqeQ0Supervision {
  const int32_t *unit = nullptr;
  const double *Qu = nullptr;
  int n_units = 0;
  double weight = 0, scale = 1;
};

/// training: adds to grad (size n_params) dE/dtheta * dL_dE + d(sum_a w_a . F_a)/dtheta with w = dL_dF [3n] (nullptr:
/// no force term); also returns the forward output (energy, forces). Open structures only.
SqeOutput sqe_train_grad(const SqeSpec &spec, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                         double dL_dE, const double *dL_dF, std::vector<double> &grad,
                         const SqeQ0Supervision *sup = nullptr);

} // namespace mlipcpp::runtime
