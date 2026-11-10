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
//
// Periodic cells (all three directions; "graph" q0 mode without lrc_mp / lrc_pol, i.e. R1 / R1q): the clusters are
// those of the infinite tiling, links found through the periodic images.
//  - bulk: one cluster connected to every one of its images (a liquid or a solid; every atom has a neighbour within
//    a_on). There is no inter-cluster pair, so no long range, and the conditioning is the structure's charge: the
//    plain model. The charge pass is not run.
//  - separated: every cluster finite in the tiling (molecules in vacuum). Each one meets the other clusters and its
//    own images, so the inter-cluster sum runs over the images: E_lr = Ewald(q) minus the pairs inside lr_R weighted
//    1 - T and the pairs of linked clusters weighted T R (the reliability is that of the unwrapped weak component),
//    neutralising background for a charged cell. The open-structure value is its large-cell limit.
//  - anything else (a slab, a wire, a network next to separate molecules, weak links that wrap the cell) throws.
// Forces and the virial (dE/d strain, for the stress) are exact in both cases.
//
// Intra-cluster long range (pet.plato_lr.lrc_intra = "ewald"; absent key or "none" = everything above, bit for bit):
// the reliability mask leaves the energy (R = 0 there, consistently, open and periodic), so the Coulomb term of the
// charges also acts inside a connected cluster and in a bulk cell,
//   E_lr = KE [ Ewald(q) - sum_{pairs r < lr_R} (1 - T(r)) q_i q_j / r ]       (periodic: every image pair once)
//        = KE sum_{i < j} T(r_ij) q_i q_j / r_ij                                (open: the direct sum, its limit)
// T the smootherstep 0 -> 1 on [lr_R - lr_t, lr_R]: the part inside lr_R - lr_t, which the backbone covers, is removed
// smoothly, so point charges have no short-range singularity (T = 0 there) and need no Gaussian smearing. The charge
// pass runs on every structure, with the cell (also in a bulk cell), and is equilibrated to the structure's Q. Ewald:
// tin-foil boundary, neutralising background for a charged cell (-pi Q^2 / 2 V alpha^2; the open limit differs by the
// Madelung term of the net charge, -alpha_M KE Q^2 / 2 L). The clusters, their reliabilities, the local-charge
// conditioning and lrc_qproj's link states are unchanged (only the energy loses its mask). Graph q0 mode without
// lrc_mp / lrc_pol only.
//
// Slabs (pet.plato_lr.lrc_slab = "yb" | "ewald2d"; absent key or "none" = slabs throw, everything else bit for bit):
// periodic in two lattice directions, open along the third. Detection:
//  - pbc (T, T, F) (any one direction open): the cell given to the lrc functions has that row ZERO (graph_model and
//    mlipcpp-train pass it so; the training data already store an open direction as a zero cell row);
//  - pbc (T, T, T) with pet.plato_lr.lrc_slab_gap = g > 0 (A): a cell whose atoms leave an empty layer at least g wide
//    across exactly one lattice direction is a slab open along it (the classic Yeh-Berkowitz setting: the given cell
//    height is the elongated cell; atoms are unwrapped across the gap). g must exceed a_off and lr_R. Absent / 0: off,
//    so a (T, T, T) cell with vacuum stays a 3D-periodic system (tin-foil Ewald, as periodic DFT without a dipole
//    correction). In MD the rule is evaluated per structure: if molecules (vapour) leave no empty layer >= g, the cell
//    falls back to 3D tin-foil (under lrc_intra; the energy jumps by about the dipole term) or throws, and a molecule
//    crossing the cell boundary along the normal moves to the other side of the unwrapped slab. Pick g below the
//    smallest vacuum the trajectory keeps; (T, T, F) has neither issue.
// Models with the Plato mesh / LES heads (pet.plato_lr.mesh_coupling / les, e.g. the FTq family) refuse (T, T, F) cells
// in the backbone (and mlipcpp-train refuses such slabs for them): give these a (T, T, T) cell with vacuum and the gap
// rule; mesh / LES then see a 3D-periodic cell, the lrc term the slab.
// Electrostatics of the 2D-periodic tiling (n the unit normal, z_i = n . x_i unwrapped, A the in-plane area):
//  - "yb" (default): 3D Ewald (tin-foil, neutralising background) in an elongated cell of height Lz plus the
//    Yeh-Berkowitz dipole term with Ballenegger-Arnold-Cerda's net-charge terms,
//      E_c = 2 pi / V [ M_z^2 - Q sum_i q_i z_i^2 - Q^2 Lz^2 / 12 ],   M_z = sum_i q_i z_i,  V = A Lz,
//    which equals the exact 2D Ewald energy up to the interactions of the elongated cell's z-images beyond the mean
//    field, O(exp(-2 pi gap / L_inplane)). (T, T, F): Lz is internal, the slab extent plus a vacuum of
//    max(lrc_slab_vacuum, 4 L_inplane, lr_R + 1) rounded up to 4 A steps (locally constant, so forces are exact; the
//    energy steps by ~1e-11 relative where the extent crosses a step). (T, T, T) + gap: Lz the given cell's height.
//  - "ewald2d": the exact 2D Ewald sum (Parry / de Leeuw-Perram, O(N^2) per in-plane wave vector; validation).
//  Net charge: the h = 0 divergence of the 2D sum is dropped (the standard convention; with it E depends only on the
//  z_ij, so a charged slab is translation invariant along n). Forces are those of the 2D-periodic system.
// Clusters: one cluster through every in-plane image is the slab analogue of a bulk cell (conditioned on Q; no long
// range without lrc_intra); finite clusters in the 2D tiling are separated clusters (inter-cluster sum over the images,
// as for 3D cells). A slab beside separate molecules (or any cluster wrapping only some periodic directions) needs
// lrc_intra (the energy does not depend on the clusters then); weak links to a wrapping cluster then use their
// image displacements (R on the quotient graph of the cell's super-nodes). With lrc_slab set, (T, T, T) cells with such
// layered clusters also run under lrc_intra (3D Ewald).

#include "sqe.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;
namespace mlipcpp {
class GGUFLoader;
}

namespace mlipcpp::runtime {

/// raw per-atom charges of the "graph" q0 mode (the frozen charge pass) and their vector-Jacobian product;
/// cell: the lattice vectors as rows (9, A), null for an open structure
struct LrcQSource {
  virtual ~LrcQSource() = default;
  virtual std::vector<double> raw(int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell) = 0;
  /// g [3n] += d(sum_i w_i q_raw_i)/dx; vir (3 x 3, periodic only, may be null) += its derivative in the homogeneous
  /// strain (positions and cell deformed together)
  virtual void raw_vjp(int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell, const double *w,
                       double *g, double *vir) = 0;
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
  double fukui(int z) const; // softplus(phi_Z) (1 in fukui_mode "uniform")
  // Fukui weights of the graph-mode equilibration (pet.plato_lr.lrc_fukui_mode; feat/plato-lr-fukui), absent key =
  // "element" (bit-identical to the per-element model above):
  //   "uniform"  f_i = 1 (U: the uniform shift, (Q - sum q_raw) / N)
  //   "element"  f_i = softplus(phi_Z) + 1e-6 (E)
  //   "atom"     f_i = softplus(phi_Z + a_i) + 1e-6, a_i the per-atom output of a second frozen pass (lrc_fmodel: a GGUF
  //              copy of the charge-pass backbone whose energy head is the fitted Fukui head; same R, Q, spin inputs)
  //              (A: MACE-POLAR-1-style environment-dependent Fukui weights). f depends on positions: the equilibration's
  //              VJP carries (Q - sum q_raw) d(f_i / sum f)/dx through the f pass.
  //   "atom_pm"  two such passes, f+ (electron addition, lrc_fmodel, phi) and f- (removal, lrc_fmodel_minus,
  //              lrc_fukui_phi_minus), blended by the residual r = Q - sum q_raw: w = (1 - s) f+ / F+ + s f- / F-,
  //              s = smootherstep(r / d + 1/2), d = lrc_fukui_pm_width (e, default 0.05; 0 = the hard switch, whose
  //              q is continuous but dq/dx jumps by (w- - w+) dr/dx at r = 0) (A+-)
  // lrc_fukui_scope "cluster" (AC; with uniform / element / atom): no global step; the raw charges are projected per
  // soft cluster onto plateaued totals and the residual Q - sum P is routed by the plateau slope (lrc_qproj's
  // machinery with per-atom shares f_i and width lrc_fukui_cluster_w), so sum q = Q exactly and a near-integer neutral
  // cluster keeps its integer whatever the other clusters' residual. These charges are the reference charges: the
  // conditioning sees them too, and lrc_qproj is not applied again on top (AC replaces it).
  enum class FukuiMode { Element, Uniform, Atom, AtomPM };
  FukuiMode fukui_mode = FukuiMode::Element;
  bool fukui_cluster = false;
  double fukui_cluster_w = 0.6, fukui_pm_width = 0.05;
  std::string fmodel, fmodel_minus;
  std::vector<double> fukui_phi_minus;
  std::shared_ptr<LrcQSource> fsrc, fsrc_minus;
  double fukui_phi_of(int z, bool minus = false) const; // phi_Z
  void fukui_check() const;                              // throws on an inconsistent Fukui configuration
  bool fukui_atom() const { return fukui_mode == FukuiMode::Atom || fukui_mode == FukuiMode::AtomPM; }
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
  // pet.plato_lr.lrc_intra = "ewald": the long-range energy also inside clusters and bulk cells (header comment above)
  bool intra = false;
  // slabs (header comment above): lrc_slab mode, the (T, T, F) minimum vacuum (A, 0 = automatic), the (T, T, T) gap rule
  // (A, 0 = off); slab_rc: real-space cutoff override of the slab Ewald sums (A, 0 = automatic; tests, not in the GGUF)
  enum class Slab { None, YB, Ewald2D };
  Slab slab = Slab::None;
  double slab_vacuum = 0, slab_gap = 0, slab_rc = 0;
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
  // graph: the structure's charge, spin and cell (the charge pass is evaluated again for the vector-Jacobian products)
  double Q = 0;
  int spin = 1;
  // graph Fukui equilibration: raw charges, per-atom weights f (+ / -) and df/da, their sums, the residual
  // r = Q - sum q_raw, the +/- blend s and ds/dr
  std::vector<double> qraw, fp, fm, dfp, dfm;
  double Fp = 0, Fm = 0, r = 0, s = 0, ds = 0;
  std::vector<double> cell; // 9 when periodic
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
  // periodic: the cell (rows), whether it is one bulk cluster, and the cluster-frame positions: x with every weak
  // component unwrapped across the cell, so that its links are plain differences of xh (xh = x when open)
  std::vector<double> cell, xh;
  bool bulk = false;
  // slabs: the open lattice direction (-1: none), whether some cluster wraps only part of the periodic directions
  // (layered: xh = x, weak links carry their image displacement wd), and per atom the multiple of the open row added
  // to unwrap it across the vacuum ((T, T, T) gap rule only; empty otherwise)
  int open = -1;
  bool layered = false;
  std::vector<std::array<double, 3>> wd;
  std::vector<int32_t> zshift;
  bool slab() const { return open >= 0; }
  bool periodic() const { return !cell.empty(); }
  bool images() const { return periodic() && !bulk; } // the long range runs over the periodic images
  bool trivial() const; // one super-node, or no weak links (R binary)
};

struct LrcCondition {
  LrcClusters cl;
  LrcQ0 q0;                    // empty when cl.s == 1 (open or bulk) unless lrc_intra
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
  std::vector<double> virial; // periodic with forces: dE/d strain (3 x 3, eV); stress = virial / volume
};

int lrc_index(const SqeSpec &sqe, int z); // element index, throws when absent

/// cell (here and below): the lattice vectors as rows (9, A), all three directions periodic, or one row zero for a slab
/// open along it (needs lrc_slab); null for open structures
LrcQ0 lrc_q0(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, double Q, int spin = 1,
             const double *cell = nullptr);
/// dE/dx (added into g, [3n]) of sum_i c_i q0_i(x); vir (periodic, may be null) += its strain derivative
void lrc_q0_vjp(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, const LrcQ0 &q,
                const double *c, double *g, double *vir = nullptr);

LrcClusters lrc_clusters(const LrcSpec &L, int n, const double *x, const double *cell = nullptr);
/// dE/dx (added into g) of sum_{a != b} W_ab R_ab(x); W s x s
void lrc_clusters_vjp(const LrcSpec &L, const LrcClusters &c, int n, const double *x, const std::vector<double> &W,
                      double *g, double *vir = nullptr);

/// conditioning of one structure (max_charge: rows are clamped into [-max_charge, max_charge])
LrcCondition lrc_condition(const LrcSpec &L, const SqeSpec &sqe, int n, const int32_t *Z, const double *x, double Q,
                           int max_charge, int spin = 1, const double *cell = nullptr);
/// the rows dS_a/dx [3n] of every super-node a with dS_a != 0 (zero rows otherwise), s x 3n
std::vector<double> lrc_blend_rows(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                                   const double *x);
/// forces (-dE/dx, added into F) of the conditioning's position dependence, G_a = dE/dS_a per super-node; vir
/// (periodic, may be null) += dE/d strain
void lrc_condition_forces(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                          const double *x, const std::vector<double> &G, double *F, double *vir = nullptr);

/// graph mode with lrc_mp: the atomic dipoles mu [3n] and quadrupoles Theta [9n] (row-major 3 x 3) of every atom
void lrc_moments(const LrcSpec &L, int n, const int32_t *Z, const double *x, std::vector<double> &mu,
                 std::vector<double> &theta);
/// g [3n] += d(sum_i G_i . mu_i + H_i : Theta_i)/dx, G [3n], H [9n]
void lrc_moments_vjp(const LrcSpec &L, int n, const int32_t *Z, const double *x, const double *G, const double *H,
                     double *g);

/// lrc_qproj: the projected charges of charges q at positions x (diagnostics / parity with rdelta/neutralise)
std::vector<double> lrc_project_charges(const LrcSpec &L, int n, const int32_t *Z, const double *x, const double *q,
                                        const double *cell = nullptr);

/// inter-cluster long-range energy and forces (zero, exactly, for a one-cluster open structure and a bulk cell; with
/// lrc_intra every pair beyond the switch, inside clusters too); the cell is the conditioning's (c.cl.cell)
LrcEnergy lrc_energy(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                     const double *x, double Q, bool forces);

} // namespace mlipcpp::runtime
