#pragma once
// Combined long-range model (GGUF pet.plato_lr.lrc*, with pet.plato_lr.sqe*). Host side, float64. Three pieces, none
// of them gated or zero-initialised, no bonding topology and no formal charges as inputs:
//
// 1. Reference charges q0, two modes (pet.plato_lr.lrc_q0_mode):
//    "round" (option, ~/data/lr-sqe-pol RECOMMENDATION FINAL; imposes integers, so not used under the DFT-emulation rule): s = the learned logistic head of sqe.cpp (sqe_q0_a / _b,
//      bounded per element, one shift to Q); components c = connected at r < r_ct; merge weight w_c = prod over the pairs
//      (i in c, j not in c) of smootherstep((r_ij - r_ct) / merge_w); <n_c> over integer partitions with sum n = Q,
//      P ~ exp(-sum (n_c - s_c)^2 / 2 tau^2); shift d_c = w_c (<n_c> - s_c) - w_c sum_e d_e / sum_e w_e (sum 0), spread
//      uniformly over the component: q0_i = s_i + d_c / n_c. One component: q0 = s (sum Q, no rounding).
//    "head": the learned logistic head s alone (one global shift to Q, as DFT gets only Q).
//    "graph" (feat/lr-chargehead): a MACE-POLAR-1-style charge head on backbone features. A frozen charge pass (a GGUF
//      copy of the start backbone whose energy head is the fitted charge head, pet.plato_lr.lrc_qmodel) gives raw
//      per-atom charges q_raw_i (R, Q, spin in); Fukui equilibration to the total charge
//        q0_i = q_raw_i + f_i (Q - sum_j q_raw_j) / sum_j f_j,   f_i = softplus(phi_Z) (pet.plato_lr.lrc_fukui_*)
//      With lrc_qpass_charge = "zero" the charge pass runs at total charge 0 (local chemistry only) and Q enters only
//      through the equilibration. No SQE flows: the long-range energy is the fixed-charge Coulomb sum of the inter-cluster pairs,
//        E_lr = sum_{i<j} KE q0_i q0_j T(r_ij) (1 - R_{a_i b_j}) / r_ij,  T the smootherstep 0 -> 1 on [lr_R - lr_t, lr_R]
//      (pet.plato_lr.sqe_lr_R / _lr_t), forces exact (direct, cluster and charge-pass chain rules).
//    "fermi2" (default): two-level Fermi reference charges (one chemical potential, gapped levels; no structure-wide uniform shift):
//      q0_i = h_Z sigmoid(t - I_i / T) - l_Z sigmoid(A_i / T - t),   sum_i q0_i = Q  (t = -mu / T, by bisection + Newton)
//      I_i = IP_Z - D tanh(f+_i),  A_i = EA_Z + D tanh(f-_i),  f+-_i = a+-_Z + sum_Z' b+-_{Z Z'} CN_{i,Z'}
//    h_Z / l_Z the cation / anion capacities (sqe_q0_hi, -sqe_q0_lo), CN the element-resolved D3 coordination numbers
//    of sqe.cpp. A closed-shell fragment whose levels straddle no mu takes charge only as exp(-gap / 2T).
// 2. Smooth clusters: A_ij = 1 - smootherstep((r - a_on) / (a_off - a_on)); strong components (r <= a_on) are
//    super-nodes; weak pairs between them make super-edges p_ab = 1 - prod (1 - A_ij); R_ab = two-terminal reliability
//    (probability that a and b are connected when each super-edge is present with probability p): a polynomial in the
//    A's, 0 / 1 at binary links, unchanged by contracting p = 1 edges (so C2 in the positions).
// 3. Local charge Q_loc_a = sum_b R_ab Q0_b (Q0_b = sum of q0 over super-node b) conditions every atom of super-node a:
//    rows k = floor(Q_loc), blend S = smootherstep(Q_loc - k), c = c(k) + S (c(k+1) - c(k)) (one structure-wide
//    charge for a structure that is one cluster: the plain model). Long-range energy on the SQE charges (q0 the
//    reference): E*(gamma) - E*(gamma (1 - T(r) (1 - R))), T the sqe lr switch: inter-cluster pairs only, Hellmann-
//    Feynman forces plus the q0 and R chain rules.
// Open structures only.

#include "sqe.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;
namespace mlipcpp {
class GGUFLoader;
}

namespace mlipcpp::runtime {

/// raw per-atom charges of the "graph" q0 mode (the frozen charge pass) and their vector-Jacobian product
struct LrcQSource {
  virtual ~LrcQSource() = default;
  virtual std::vector<double> raw(int n, const int32_t *Z, const double *x, double Q, int spin) = 0;
  /// g [3n] += d(sum_i w_i q_raw_i)/dx
  virtual void raw_vjp(int n, const int32_t *Z, const double *x, double Q, int spin, const double *w, double *g) = 0;
};

struct LrcSpec {
  bool enabled = false;
  double a_on = 5.0, a_off = 6.0; // cluster link window (A)
  double T = 0.25, D = 5.0;       // Fermi temperature (eV), bound of the level shifts (eV)
  int max_super_edges = 22;       // exact reliability enumeration limit per weak component
  bool round = false;             // q0 mode "round": soft-rounded component reference (option; imposes integers)
  bool head_only = false;         // q0 mode "head": the learned logistic head alone (sum Q, no rounding)
  double tau = 0.25, merge_w = 2.0; // round: Gaussian width (e) and the merge switch width beyond r_ct (A)
  // q0 mode "graph": the charge pass (path, relative to the model's GGUF unless absolute; MLIPCPP_LRC_QMODEL
  // overrides), Fukui parameters per element, the source (set by the model that loads it)
  bool graph = false;
  bool qpass_neutral = false; // the charge pass sees total charge 0 (pet.plato_lr.lrc_qpass_charge = "zero"); Q enters
                              // through the Fukui equilibration only
  std::string qmodel;
  std::vector<int32_t> fukui_Z;
  std::vector<double> fukui_phi;
  std::shared_ptr<LrcQSource> qsrc;
  double fukui(int z) const; // softplus(phi_Z)
  // per element in SqeSpec::Z order
  std::vector<double> ip, ea, h, l, ap, am, bp, bm; // bp, bm: E x E (row: the atom's element)
  // graph mode, atomic multipoles (pet.plato_lr.lrc_mp*, feat/plato-lr-v2): a prefitted (frozen) atomic dipole and
  // traceless (Buckingham) quadrupole per atom, linear in element-pair x radial channels of the neighbours within mp_rc,
  //   mu_i    = sum_j sum_k theta[Z_i, Z_j, k] g_k(r_ij) u_ij,                       u_ij = (x_j - x_i) / r_ij
  //   Theta_i = sum_j sum_k tau[Z_i, Z_j, k]   g_k(r_ij) (3/2 u_ij u_ij^T - 1/2 I),
  //   g_k(r) = exp(-(r - c_k)^2 / 2 w^2) * (1 + cos(pi r / mp_rc)) / 2   (r < mp_rc)
  // (equivariant by construction; elements outside mp_Z carry none). They add the charge-dipole, dipole-dipole and
  // charge-quadrupole terms to the inter-cluster pairs of the fixed-charge Coulomb sum (same switch and (1 - R)).
  // These are the octahedral distributed charges' moments (kernel-lr lane) as point multipoles: the six satellites at
  // +-a e_k reproduce (q, mu, Theta) exactly and differ only above the quadrupole, O((a / r)^2) <= 0.2 % at 8 A.
  bool mp = false;
  double mp_rc = 4.5, mp_width = 0.5;
  std::vector<double> mp_centers;      // K radial centres
  std::vector<int32_t> mp_Z;           // E elements
  std::vector<double> mp_theta, mp_tau; // E x E x K (row: the centre atom's element)
  std::vector<int> mp_index;           // atomic number -> index into mp_Z (-1: none), built by from_gguf / mp_build
  void mp_build();
  // graph mode, induced atomic dipoles (pet.plato_lr.lrc_pol*): additive isotropic polarisabilities per element (A^3,
  // prefitted to DFT CPHF monomer polarisabilities), responding to the switched field of the OTHER clusters' charges and
  // to each other's dipoles across clusters (no intra-cluster coupling: that is the short-range model's),
  //   (alpha^-1 - T) mu = F0,  E_pol = -1/2 KE mu . F0,  forces Hellmann-Feynman (+ charge-pass VJP, switch, clusters).
  // Thole damping is not needed: every coupled pair is beyond lr_R - lr_t (8 A).
  // backbone conditioning (pet.plato_lr.lrc_cond): "cluster" (default: every atom conditioned on its smooth cluster's
  // local charge Q_loc) or "global" (every atom on the structure's Q, as the plain model: no seam where clusters split
  // inside the backbone cutoff; the charge pass then feeds only the long-range term)
  bool cond_global = false;
  // blend width of the local-charge conditioning rows (pet.plato_lr.lrc_cond_w, default 1): S = smootherstep((u - 1/2) / w
  // + 1/2), u the fractional part of Q_loc. w < 1 gives flat plateaus around integers (|u| < (1 - w) / 2 maps to the
  // integer row exactly), so a small residual charge leaked onto a neutral cluster does not shift its conditioning at
  // any distance; still C2 in the positions. The long-range charges themselves are not rounded.
  double cond_w = 1.0;
  // graph mode, soft charge projection of the long-range charges (pet.plato_lr.lrc_qproj, width lrc_qproj_w; R1q):
  // per link state of the soft clusters, every group G's charge total Q0_G is moved to the plateau value
  // P(Q0_G) = k + smootherstep((Q0_G - k - 1/2) / w + 1/2) (k = floor Q0_G; |frac| < (1 - w) / 2 maps onto the integer),
  // shared within G by Fukui weight; the super-node correction s_a is the expectation over the link states, and the
  // residual -sum_a s_a is put back with weights F_a (1e-3 + E[P'(Q0_G)]) so sum q = Q exactly:
  //   q'_i = q_i + f_i / F_a (s_a - S g_a / G),  S = sum s, g_a = F_a (1e-3 + E[P']), G = sum g.
  // Reduces to the separated-cluster projection with no active link and to the merged one with all links. Model charges
  // only (no labels); the conditioning is unchanged. Validated offline in rdelta/neutralise (project.py).
  bool qproj = false;
  double qproj_w = 0.6;
  bool pol = false;
  std::vector<int32_t> pol_Z;
  std::vector<double> pol_alpha;
  static LrcSpec from_gguf(const GGUFLoader &L, const SqeSpec &sqe);
  void to_gguf(gguf_context *g) const;
};

struct LrcQ0 {
  std::vector<double> q0;
  std::vector<double> wt, wp, wm; // fermi2: dq0/dt, dq0/df+, dq0/df-; round: wt = the head's dq/dt (= dq/df)
  bool feasible = true;
  // round: components, merge weights, partition moments
  int K = 1;
  std::vector<int32_t> comp;           // per atom
  std::vector<double> sc, w, nbar, cov; // per component; cov K x K (of the partition distribution)
  std::vector<int> ncomp;              // atoms per component
  // graph: the structure's charge and spin (the charge pass is evaluated again for the vector-Jacobian products)
  double Q = 0;
  int spin = 1;
};

struct LrcClusters {
  int s = 1;               // super-nodes
  std::vector<int32_t> sn; // super-node per atom
  std::vector<double> R;   // s x s
  struct Weak {
    int i, j, e; // atoms, super-edge index
    double A, dA; // link value and dA/dr
  };
  std::vector<Weak> weak;
  std::vector<std::pair<int, int>> se; // super-edges (a < b)
  std::vector<double> p;               // super-edge probabilities
  bool fallback = false;               // a weak component above max_super_edges was taken as one cluster
  bool trivial() const; // one super-node, or no weak links (R binary)
};

struct LrcCondition {
  LrcClusters cl;
  LrcQ0 q0;                    // empty when cl.s == 1
  std::vector<double> Q0, Qloc; // per super-node
  std::vector<double> S, dS;    // per super-node: blend and dS/dQloc
  std::vector<int32_t> k;       // per super-node: floor(Qloc)
  // per atom
  std::vector<int32_t> row_charge, row_charge_hi; // charges of the lower / upper rows (k, k + 1)
  std::vector<float> blend;                       // S
  bool needs_cond_forces() const;                 // any dS != 0
};

struct LrcEnergy {
  double energy = 0;
  std::vector<double> forces, q, q0;
  std::vector<double> mu_ind; // lrc_pol: induced dipoles [3n] (e A)
};

int lrc_index(const SqeSpec &sqe, int z); // element index, throws when absent

LrcQ0 lrc_q0(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, double Q, int spin = 1);
/// dE/dx (added into g, [3n]) of sum_i c_i q0_i(x)
void lrc_q0_vjp(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, const LrcQ0 &q,
                const double *c, double *g);

LrcClusters lrc_clusters(const LrcSpec &L, int n, const double *x);
/// dE/dx (added into g) of sum_{a != b} W_ab R_ab(x); W s x s
void lrc_clusters_vjp(const LrcSpec &L, const LrcClusters &c, int n, const double *x, const std::vector<double> &W,
                      double *g);

/// conditioning of one structure (max_charge: rows are clamped into [-max_charge, max_charge])
LrcCondition lrc_condition(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, double Q,
                           int max_charge, int spin = 1);
/// the rows dS_a/dx [3n] of every super-node a with dS_a != 0 (zero rows otherwise), s x 3n
std::vector<double> lrc_blend_rows(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                                   const double *x);
/// forces (-dE/dx, added into F) of the conditioning's position dependence, G_a = dE/dS_a per super-node
void lrc_condition_forces(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                          const double *x, const std::vector<double> &G, double *F);

/// graph mode with lrc_mp: the atomic dipoles mu [3n] and quadrupoles Theta [9n] (row-major 3 x 3) of every atom
void lrc_moments(const LrcSpec &L, int n, const int32_t *Z, const double *x, std::vector<double> &mu,
                 std::vector<double> &theta);
/// g [3n] += d(sum_i G_i . mu_i + H_i : Theta_i)/dx, G [3n], H [9n]
void lrc_moments_vjp(const LrcSpec &L, int n, const int32_t *Z, const double *x, const double *G, const double *H,
                     double *g);

/// lrc_qproj: the projected charges of charges q at positions x (diagnostics / parity with rdelta/neutralise)
std::vector<double> lrc_project_charges(const LrcSpec &L, int n, const int32_t *Z, const double *x, const double *q);

/// inter-cluster long-range energy and forces (zero, exactly, for a one-cluster structure)
LrcEnergy lrc_energy(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                     const double *x, double Q, bool forces);

} // namespace mlipcpp::runtime
