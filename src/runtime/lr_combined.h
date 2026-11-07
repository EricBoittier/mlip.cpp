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
#include <vector>

struct gguf_context;
namespace mlipcpp {
class GGUFLoader;
}

namespace mlipcpp::runtime {

struct LrcSpec {
  bool enabled = false;
  double a_on = 5.0, a_off = 6.0; // cluster link window (A)
  double T = 0.25, D = 5.0;       // Fermi temperature (eV), bound of the level shifts (eV)
  int max_super_edges = 22;       // exact reliability enumeration limit per weak component
  bool round = false;             // q0 mode "round": soft-rounded component reference (option; imposes integers)
  bool head_only = false;         // q0 mode "head": the learned logistic head alone (sum Q, no rounding)
  double tau = 0.25, merge_w = 2.0; // round: Gaussian width (e) and the merge switch width beyond r_ct (A)
  // per element in SqeSpec::Z order
  std::vector<double> ip, ea, h, l, ap, am, bp, bm; // bp, bm: E x E (row: the atom's element)
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
};

int lrc_index(const SqeSpec &sqe, int z); // element index, throws when absent

LrcQ0 lrc_q0(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, double Q);
/// dE/dx (added into g, [3n]) of sum_i c_i q0_i(x)
void lrc_q0_vjp(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, const LrcQ0 &q,
                const double *c, double *g);

LrcClusters lrc_clusters(const LrcSpec &L, int n, const double *x);
/// dE/dx (added into g) of sum_{a != b} W_ab R_ab(x); W s x s
void lrc_clusters_vjp(const LrcSpec &L, const LrcClusters &c, int n, const double *x, const std::vector<double> &W,
                      double *g);

/// conditioning of one structure (max_charge: rows are clamped into [-max_charge, max_charge])
LrcCondition lrc_condition(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, double Q,
                           int max_charge);
/// forces (-dE/dx, added into F) of the conditioning's position dependence, G_a = dE/dS_a per super-node
void lrc_condition_forces(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                          const double *x, const std::vector<double> &G, double *F);

/// inter-cluster long-range energy and forces (zero, exactly, for a one-cluster structure)
LrcEnergy lrc_energy(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                     const double *x, double Q, bool forces);

} // namespace mlipcpp::runtime
