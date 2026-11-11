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
// Periodic cells under lrc_links = backbone (and lrc_states = mix; feat/lrc-states-pbc). The tiling's backbone graph:
// adaptive radii from every image within the backbone cutoff, strong links (d <= lo_ij) and weak ones (lo_ij < d < c_ij)
// through the images. Every strong component of the tiling is either finite or percolating (connected to every one of
// its images: liquids, ionic solutions, crystals). Super-nodes are the strong components modulo the lattice (a
// percolating one is ONE super-node; its Q0 is its charge per cell, with no interpolation across its own images), and
// the weak links between them (with their image displacements) give the super-edges, R, Q_loc, lrc_qproj's states and
// the lrc_states copies exactly as for open structures (one state applies to every image: a copy drops the backbone
// edges between the groups it separates at every image shift). The long range is the expectation over a tiling whose
// image copies carry independent link states (so a supercell is consistent):
//   E_lr = KE [ Ewald_W(q) - sum_{a finite} (1 - r_a^2) D_aa - sum_{a < b, same finite component} (R_ab - r_a r_b) D_ab ]
// with r_a = R(a, P) the probability that a is joined to the percolating super-node P of its weak component (r_P = 1;
// r_a = 0 without one), Ewald_W the tin-foil Ewald sum (neutralising background) with every pair of super-nodes a, b
// weighted W_ab = 1 - r_a r_b when both belong to P's weak component (1 otherwise), D_ab the direct sum of q_i q_j / r
// over the pairs of a and b in the unwrapped frame of their finite component. Limits: without a percolating super-node
// this is Ewald(q) minus the linked pairs weighted R (the open model's (1 - R) up to the images' term KE [Ewald(q) -
// direct(q)], exactly); one bulk super-node gives 0 (no charge pass: the plain model); a finite super-node whose link to
// P becomes strong (r -> 1) joins P continuously. A weak component that reaches its own images (a cycle of weak links
// through the cell boundary, a weak link to its own image, or two percolating clusters: an infinite chain of link
// states) is taken as one percolating cluster (fallback, like max_super_edges: not smooth where it first forms; counted
// in LrcClusters::fallback). Slab cells (a zero cell row, lrc_slab) throw.
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
  void links_check() const;                              // throws on an unsupported lrc_links configuration
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
  // cluster links from the backbone's own edges (pet.plato_lr.lrc_links = "backbone"; absent key or "window" = the fixed
  // a_on / a_off window above, bit for bit). The link of a pair is tied to the backbone's (adaptive) pair cutoff
  //   c_ij = (r_i + r_j) / 2   (r_i the adaptive radius of atom i, pet.adaptive_cutoff_method; the plain cutoff without
  //                             an adaptive cutoff), the same pair cutoff at which the backbone drops edge ij,
  //   A_ij = 1 - smootherstep((d_ij - lo_ij) / (c_ij - lo_ij)),   lo_ij = link_a c_ij + link_b   (lrc_link_a / _b),
  // so a link is exactly 0 where the backbone has no edge (d >= c_ij) and exactly 1 (a strong link: one super-node)
  // inside lo_ij. With the defaults (0.5, 1 A) the taper is 3 A wide at c = 8 A (bare ions) and 1 A at c = 4 A (dense
  // structures, where a fixed 3 A taper would cut covalent bonds). Clusters, reliabilities, the local-charge conditioning
  // and lrc_qproj's link states are then those of the backbone's graph: joint conditioning exactly while the backbone
  // connects two fragments, split once it does not. The radii depend on every neighbour within the cutoff, so the link
  // gradients carry the adaptive-cutoff chain rule (d r_i / d d_ik, as the backbone's forces). The inter-cluster
  // Coulomb term has no distance switch under these links (weight 1 - R_ab on every inter-cluster pair; such pairs are
  // at least lo_ij apart): it is fully on exactly where the backbone has disconnected the two clusters.
  // Open structures and periodic cells (see "Periodic cells under lrc_links = backbone" above), graph q0 mode without
// lrc_mp / lrc_pol / lrc_intra / lrc_slab only (others throw).
  bool links_backbone = false;
  double link_a = 0.5, link_b = 1.0;
  // backbone link states (pet.plato_lr.lrc_states = "mix", needs lrc_links = backbone; absent key or "none" = the
  // conditioning-only links above, bit for bit). The link states sigma of the weak super-edges (the states of R_ab and
  // lrc_qproj) also weight the backbone energy:
  //   E = sum_sigma w_sigma E_bb(G_sigma, rows_sigma) + E_lr,
  // G_sigma the backbone graph without the edges between clusters that sigma separates (radii, pair cutoffs and the
  // radius chain stay those of the full structure), rows_sigma every cluster of sigma on its own Q0 total (the usual
  // two-row blend of the fractional part). Per weak component c the edge states e (P_e = prod p^on (1 - p)^off) get the
  // smooth cut w_e = P_e s((P_e / eps - 1/2) * 2) (s the smootherstep; 0 below eps / 2, P_e above eps), summed per
  // partition and renormalised; a mass guard (kept weight W_c below 0.9 -> 0.5) blends in that component's fix-3
  // treatment (one graph, blended rows), so the renormalisation never divides by a small number. Components combine as
  // a product (size consistent); a smooth count of the kept partitions, M = prod_c (N_c + lam_c), N_c = sum over the
  // partitions of 1 - prod_e (1 - s(4 P_e / eps - 1)), lam_c the mass guard's fix-3 weight (each factor -> 1 where a
  // component appears or vanishes), blends in the whole structure's fix-3 evaluation from
  // M = states_cap_lo to states_cap (then no state is evaluated), which bounds the backbone evaluations by
  // about states_cap. Forces: sum_sigma w_sigma F_sigma - sum_sigma E_sigma grad w_sigma - grad E_lr, exact (the second
  // term through the links' gradient with the radius chain).
  bool states = false;
  double states_eps = 1e-3, states_cap = 16, states_cap_lo = 8;
  // neutral receivers (pet.plato_lr.lrc_cond_receiver = "neutral"; absent key or "all" = Q_loc_a = sum_b R_ab Q0_b, bit
  // for bit): a super-node takes its linked partners' charge in proportion to how charged it is itself,
  //   Q_loc_a = Q0_a + h(|Q0_a|) sum_{b != a} R_ab Q0_b,   h = smootherstep((|Q0_a| - receiver_lo) / (receiver_hi - receiver_lo))
  // (defaults 0.3, 0.7 e), so a near-neutral molecule beside a cation keeps its own (~0) row instead of a fraction of
  // the cation's charge, and charged-charged links (ion pairs) are unchanged. A fixed function of the predicted charges,
  // not a learned gate; the forces carry h' through Q0_a. Also in each lrc_states state (R binary there).
  bool receiver_neutral = false;
  double receiver_lo = 0.3, receiver_hi = 0.7;
  // the backbone's cutoff (pet.cutoff, pet.num_neighbors_adaptive, pet.cutoff_width_adaptive, pet.adaptive_cutoff_method;
  // read from the model's GGUF, not written by to_gguf)
  double bb_cutoff = 0, bb_nadapt = 0, bb_width = 0.5;
  bool bb_grid = true;
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
    double dAc = 0; // backbone links: dA/dc_ij (c_ij the pair cutoff, through the adaptive radii)
    bool img = false;                 // periodic backbone links: the link's image displacement d = x_j' - x_i below
    std::array<double, 3> d{0, 0, 0}; // replaces the cluster-frame difference xh_j - xh_i (lrc_links = backbone, PBC)
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
  // backbone links: the adaptive radius of every atom and, per atom i, its neighbours k within the backbone cutoff with
  // d r_i / d d_ik (CSR: nb_start [n + 1], nb_j, nb_jac); empty for the window links
  std::vector<double> rad, nb_jac;
  std::vector<int32_t> nb_start, nb_j;
  std::vector<double> nb_d; // periodic: the displacement x_k' - x_i of every CSR neighbour (3 per entry; empty when open)
  // periodic backbone links (lr_combined.h "Periodic cells under lrc_links = backbone"): per super-node, whether it
  // percolates (its strong component is connected to every image), the percolating super-node of its weak component
  // (-1: none) and its finite component (super-nodes joined by weak links between finite super-nodes; -1 when
  // percolating). Empty otherwise.
  std::vector<int32_t> perc, pnode, fcomp;
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
/// dQ0_b/dx [3n] of every super-node b (rows, s x 3n; one charge-pass VJP each), and the blend rows of c built from
/// them: identical to lrc_blend_rows (the q0 VJP is linear in its weights), so one set of q0 rows serves every
/// lrc_states copy of a structure (their super-nodes and q0 are the parent's)
std::vector<double> lrc_q0_rows(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                                const double *x);
std::vector<double> lrc_blend_rows_q0(const LrcSpec &L, const LrcCondition &c, int n, const double *x,
                                      const std::vector<double> &q0rows);
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

/// lrc_states: the backbone evaluations ("copies") of a structure and their weights (LrcSpec::states)
struct LrcStateCopy {
  std::vector<int32_t> choice; // per weak component: its partition index, or -1 = fix 3 (joint graph, blended rows)
  double w = 0;                // weight (sum over the copies = 1)
  std::vector<double> dw;      // dw / dp_e per super-edge of the clusters
};
struct LrcStates {
  bool active = false;                    // anything but the plain fix-3 evaluation
  std::vector<std::vector<int>> comps;    // weak components: their super-edges
  std::vector<std::vector<int>> comp_nodes; // and super-nodes
  // per component, per partition: the group label of every super-node of the component (the smallest super-node index
  // of its group)
  std::vector<std::vector<std::vector<int32_t>>> parts;
  std::vector<LrcStateCopy> copies;
  double lambda = 0;   // weight of the whole-structure fix-3 copy from the cap
  double M = 0;        // the smooth count of kept partitions
  int fix3_comps = 0;  // components whose mass guard is active (or that exceed max_super_edges)
  std::vector<int32_t> comp_of; // per super-node: its weak component (-1: none)
  /// per atom: the edge-drop group of copy v; edges i-j are dropped iff group_i >= 0, group_j >= 0, group_i != group_j
  /// and both lie in the same weak component (encoded: component * 2^20 + label)
  std::vector<int32_t> drop_groups(const LrcClusters &c, size_t v) const;
};
LrcStates lrc_states(const LrcSpec &L, const LrcClusters &c);
/// the conditioning of copy v: state-mode components on their partition (binary R, no link gradient), fix-3-mode
/// components and everything else as c (returns c itself for the all-fix-3 copy)
LrcCondition lrc_state_condition(const LrcSpec &L, const LrcCondition &c, const LrcStates &st, size_t v, double Q,
                                 int max_charge);
/// g [3n] += d/dx sum_v E_v w_v(x) (the copies' energies E held fixed); vir (periodic, may be null) += its strain
/// derivative
void lrc_states_vjp(const LrcSpec &L, const LrcClusters &c, const LrcStates &st, const std::vector<double> &E,
                    double *g, double *vir = nullptr);
/// grad p_e for every super-edge e (rows, E x 3n)
std::vector<double> lrc_superedge_grads(const LrcSpec &L, const LrcClusters &c, int n);

/// lrc_qproj: the projected charges of charges q at positions x (diagnostics / parity with rdelta/neutralise)
std::vector<double> lrc_project_charges(const LrcSpec &L, int n, const int32_t *Z, const double *x, const double *q,
                                        const double *cell = nullptr);

/// inter-cluster long-range energy and forces (zero, exactly, for a one-cluster open structure and a bulk cell; with
/// lrc_intra every pair beyond the switch, inside clusters too); the cell is the conditioning's (c.cl.cell)
LrcEnergy lrc_energy(const LrcSpec &L, const SqeSpec &sqe, const LrcCondition &c, int n, const int32_t *Z,
                     const double *x, double Q, bool forces);

} // namespace mlipcpp::runtime
