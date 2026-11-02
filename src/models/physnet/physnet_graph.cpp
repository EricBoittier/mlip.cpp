#include <cstdlib>
#include "physnet_graph.h"

#include <cmath>
#include <ggml.h>
#include <stdexcept>

namespace mlipcpp::physnet {

namespace {

// e3x / mmml ZBL constants (models/zbl.py, trainable=False)
constexpr double kBohr = 0.529177249;
constexpr double kCoulomb = 14.3996454784255; // eV A
constexpr double kPhiC[4] = {0.18175, 0.50986, 0.28022, 0.02817};
constexpr double kPhiA[4] = {3.19980, 0.94229, 0.40290, 0.20162};

struct Builder {
  ggml_context *c;
  const PhysNetHypers &h;
  const WeightFn &weight;
  const ConstantFn &constant;
  int64_t n = 0, m = 0, E = 0, F = 0;

  ggml_tensor *w(const std::string &name) const {
    ggml_tensor *t = weight(name);
    if (!t) throw std::runtime_error("PhysNet GGUF: missing tensor " + name);
    return t;
  }
  ggml_tensor *w_opt(const std::string &name) const { return weight(name); }

  // the e3x gated silu: silu on the scalars, sigmoid(scalars) on the vectors
  void silu(ggml_tensor *&s, ggml_tensor *&v) const {
    if (v) v = ggml_mul(c, v, ggml_reshape_3d(c, ggml_sigmoid(c, s), F, 1, n));
    s = ggml_silu(c, s);
  }
  // e3x Dense: degree-wise linear map, bias on the scalars
  void dense(int d, ggml_tensor *&s, ggml_tensor *&v) const {
    const std::string p = "dense" + std::to_string(d) + ".";
    s = ggml_add(c, ggml_mul_mat(c, w(p + "w0"), s), w(p + "b0"));
    if (v)
      v = ggml_reshape_3d(c, ggml_mul_mat(c, w(p + "w1"), ggml_reshape_2d(c, v, F, 3 * n)), F, 3, n);
  }
  ggml_tensor *neighbour = nullptr; // I32 [E]: the atom j of slot (i, s), or nullptr
  // features of the neighbour j at every slot (i, s): rows of j, j the centre of
  // the reverse edge (j, s') = floor(rev / m); for n m >= 2^24 (float32 index
  // arithmetic) node features repeated over the slots, gathered at the reverse edge
  ggml_tensor *gather_neighbours(ggml_tensor *x, int64_t rows, ggml_tensor *rev) const {
    if (neighbour) return ggml_get_rows(c, ggml_reshape_2d(c, x, rows, n), neighbour);
    ggml_tensor *x3 = ggml_reshape_3d(c, x, rows, 1, n);
    ggml_tensor *rep = ggml_repeat(c, x3, ggml_new_tensor_3d(c, GGML_TYPE_F32, rows, m, n));
    return ggml_get_rows(c, ggml_reshape_2d(c, rep, rows, E), rev); // [rows, E]
  }
  ggml_tensor *center = nullptr; // I32 [E]: the atom i of slot (i, s), or nullptr
  // sum over the neighbour slots: [rows, m, n] -> [rows, n]; a scatter-add by
  // centre (its gradient is a gather, not a single-threaded repeat), else
  // a permuted copy and a row sum
  ggml_tensor *sum_slots(ggml_tensor *x, int64_t rows) const {
    if (center)
      return ggml_get_rows_back(c, ggml_reshape_2d(c, x, rows, E), center,
                                ggml_new_tensor_2d(c, GGML_TYPE_F32, rows, n));
    ggml_tensor *x3 = ggml_reshape_3d(c, x, rows, m, n);
    ggml_tensor *t = ggml_cont(c, ggml_permute(c, x3, 1, 0, 2, 3)); // [m, rows, n]
    return ggml_reshape_2d(c, ggml_sum_rows(c, t), rows, n);
  }
  // concatenate [1, E] tensors along dim 0, balanced (O(K E log K) copies)
  ggml_tensor *stack(std::vector<ggml_tensor *> parts) const {
    while (parts.size() > 1) {
      std::vector<ggml_tensor *> next;
      for (size_t i = 0; i + 1 < parts.size(); i += 2) next.push_back(ggml_concat(c, parts[i], parts[i + 1], 0));
      if (parts.size() % 2) next.push_back(parts.back());
      parts.swap(next);
    }
    return parts[0];
  }
};

} // namespace

ggml_tensor *build_physnet(ggml_context *ctx, const PhysNetHypers &h, const PhysNetInputs &in,
                           const WeightFn &weight, const ConstantFn &constant, PhysNetHeads *heads) {
  if (h.max_degree < 0 || h.max_degree > 1) throw std::runtime_error("PhysNet: max_degree must be 0 or 1");
  if (h.n_res < 1) throw std::runtime_error("PhysNet: n_res must be >= 1");
  Builder b{ctx, h, weight, constant};
  ggml_context *c = ctx;
  const int64_t n = in.edge_vectors->ne[2], m = in.edge_vectors->ne[1], E = n * m, F = h.features;
  b.n = n, b.m = m, b.E = E, b.F = F;
  const int L = h.max_degree, K = h.num_basis;

  // --- geometry: [1, m, n] per slot ---
  ggml_tensor *ev = in.edge_vectors;
  ggml_tensor *valid = ggml_scale_bias(c, ggml_reshape_3d(c, in.padding_mask, 1, m, n), -1.0f, 1.0f);
  ggml_tensor *r2 = ggml_sum_rows(c, ggml_sqr(c, ev)); // [1, m, n]
  ggml_tensor *r = ggml_sqrt(c, r2);
  ggml_tensor *ones = constant(c, {1, m, n}, 1.0f);
  // smooth_cutoff: exp(1 - 1 / (1 - (r / rc)^2)) inside the cutoff (padded slots
  // sit at the cutoff: clamped to a finite value, then masked)
  const float rc = h.cutoff;
  ggml_tensor *x2 = ggml_clamp(c, ggml_scale(c, r2, 1.0f / (rc * rc)), 0.0f, 1.0f - 1e-6f);
  ggml_tensor *inv = ggml_div(c, ones, ggml_scale_bias(c, x2, -1.0f, 1.0f));
  ggml_tensor *cut = ggml_mul(c, ggml_exp(c, ggml_scale_bias(c, inv, -1.0f, 1.0f)), valid);
  // exponential Chebyshev radial basis T_k(2 e^{-r} - 1), recurrence
  ggml_tensor *y = ggml_scale_bias(c, ggml_exp(c, ggml_scale(c, r, -1.0f)), 2.0f, -1.0f);
  std::vector<ggml_tensor *> T{ones};
  if (K > 1) T.push_back(y);
  for (int k = 2; k < K; k++)
    T.push_back(ggml_sub(c, ggml_scale(c, ggml_mul(c, y, T[k - 1]), 2.0f), T[k - 2]));
  for (auto &t : T) t = ggml_reshape_2d(c, t, 1, E);
  ggml_tensor *rbf = ggml_mul(c, b.stack(T), ggml_reshape_2d(c, cut, 1, E)); // [K, E]
  // unit vectors (Racah l = 1 harmonics in Cartesian order), broadcast over the
  // features once as an outer product (a multithreaded GEMM and its gradient,
  // not a broadcast multiply whose gradient reduces over the features in a
  // single thread): [F, 3, E]
  ggml_tensor *u = nullptr;
  if (L >= 1) {
    ggml_tensor *u1 = ggml_reshape_2d(c, ggml_div(c, ev, ggml_repeat(c, r, ev)), 1, 3 * E);
    u = ggml_reshape_3d(c, ggml_mul_mat(c, constant(c, {1, F}, 1.0f), u1), F, 3, E);
  }
  // index arithmetic in float32 is exact below 2^24
  ggml_tensor *center = nullptr;
  if (E < (int64_t(1) << 24) && !std::getenv("MLIPCPP_PHYSNET_REPEAT_GATHER")) { // (env: the old path, for tests)
    auto index_div_m = [&](ggml_tensor *flat) { // floor((flat + 0.5) / m), as I32
      return ggml_cast(c, ggml_floor(c, ggml_scale_bias(c, flat, 1.0f / float(m), 0.5f / float(m))), GGML_TYPE_I32);
    };
    b.neighbour = index_div_m(ggml_cast(c, in.reverse_index, GGML_TYPE_F32));
    b.center = center = index_div_m(ggml_arange(c, 0.0f, float(E), 1.0f));
  }

  // --- embedding (+ charge / spin conditioning) ---
  ggml_tensor *s = ggml_get_rows(c, b.w("embed"), in.species); // [F, n]
  ggml_tensor *v = nullptr;                                      // [F, 3, n]
  ggml_tensor *q_atom = nullptr, *mult_atom = nullptr;           // [1, n]: Q and 2S+1 of the atom's structure
  if (h.spooky) {
    if (!in.charge_index || !in.spin_index) throw std::runtime_error("SpookyPhysNet: no charge / spin inputs");
    ggml_tensor *q = ggml_get_rows(c, b.w("cond.charge_values"), in.charge_index); // [1, n]
    ggml_tensor *sp = ggml_get_rows(c, b.w("cond.spin_values"), in.spin_index);
    q_atom = q, mult_atom = sp;
    ggml_tensor *cq = ggml_add(c, ggml_mul_mat(c, b.w("cond.charge.w"), q), b.w("cond.charge.b"));
    ggml_tensor *cs = ggml_add(c, ggml_mul_mat(c, b.w("cond.spin.w"), sp), b.w("cond.spin.b"));
    s = ggml_add(c, s, ggml_add(c, cq, cs));
  }

  // --- message passing + refinement ---
  // every iteration's filters in one product (one GEMM and one gradient GEMM over
  // the edges instead of two per iteration): rows of fall, [sum F, E]
  std::vector<std::pair<int64_t, int64_t>> filters; // first row of filter.0 / filter.1
  ggml_tensor *wall = nullptr;
  int64_t rows_all = 0;
  for (int t = 0; t < h.num_iterations; t++) {
    const std::string p = "mp" + std::to_string(t) + ".";
    std::pair<int64_t, int64_t> f{-1, -1};
    for (int l = 0; l <= std::min(L, 1); l++) {
      ggml_tensor *wl = b.w(p + "filter." + std::to_string(l)); // [K, F]
      wall = wall ? ggml_concat(c, wall, wl, 1) : wl;
      (l == 0 ? f.first : f.second) = rows_all;
      rows_all += F;
    }
    filters.push_back(f);
  }
  ggml_tensor *fall = ggml_mul_mat(c, wall, rbf); // [rows_all, E]
  auto filter_rows = [&](int64_t row0) {
    return ggml_view_2d(c, fall, F, E, fall->nb[1], row0 * sizeof(float)); // (consumers: elementwise)
  };
  const float inv_sqrt3 = static_cast<float>(1.0 / std::sqrt(3.0));
  for (int t = 0; t < h.num_iterations; t++) {
    const std::string p = "mp" + std::to_string(t) + ".";
    const int L1 = L, L2 = v ? 1 : 0, L3 = (t == h.num_iterations - 1) ? 0 : std::max(L1, L2);
    ggml_tensor *f0 = filter_rows(filters[t].first);                             // [F, E]
    ggml_tensor *f1 = L1 >= 1 ? filter_rows(filters[t].second) : nullptr;       // [F, E]
    ggml_tensor *b0 = b.gather_neighbours(s, F, in.reverse_index);              // [F, E]
    ggml_tensor *b1 = v ? ggml_reshape_3d(c, b.gather_neighbours(v, 3 * F, in.reverse_index), F, 3, E) : nullptr;
    // scalar output: 00->0 and 11->0
    ggml_tensor *o0 = ggml_mul(c, ggml_mul(c, f0, b0), b.w(p + "tp.000"));
    if (f1 && b1) {
      ggml_tensor *ub = ggml_mul(c, b1, u); // [F, 3, E], same shapes
      ggml_tensor *dot = ggml_add(c, ggml_add(c, ggml_view_2d(c, ub, F, E, ub->nb[2], 0),
                                              ggml_view_2d(c, ub, F, E, ub->nb[2], ub->nb[1])),
                                  ggml_view_2d(c, ub, F, E, ub->nb[2], 2 * ub->nb[1]));
      o0 = ggml_add(c, o0, ggml_mul(c, ggml_mul(c, f1, dot), ggml_scale(c, b.w(p + "tp.110"), inv_sqrt3)));
    }
    // vector output: 01->1 (filter scalar x neighbour vector), 10->1 (filter vector x neighbour scalar)
    ggml_tensor *o1 = nullptr;
    if (L3 >= 1) {
      if (b1) o1 = ggml_mul(c, b1, ggml_reshape_3d(c, ggml_mul(c, f0, b.w(p + "tp.011")), F, 1, E));
      if (f1) {
        ggml_tensor *g = ggml_reshape_3d(c, ggml_mul(c, ggml_mul(c, f1, b0), b.w(p + "tp.101")), F, 1, E);
        ggml_tensor *t1 = ggml_mul(c, u, g); // g broadcast over the 3 components
        o1 = o1 ? ggml_add(c, o1, t1) : t1;
      }
    }
    s = b.sum_slots(o0, F);
    v = o1 ? ggml_reshape_3d(c, b.sum_slots(o1, 3 * F), F, 3, n) : nullptr;

    // refinement (model._refinement_iteration)
    ggml_tensor *ys = nullptr, *yv = nullptr;
    for (int k = 0; k < h.n_res; k++) {
      ggml_tensor *as = s, *av = v;
      b.silu(as, av);
      ys = ggml_add(c, s, as);
      yv = v ? ggml_add(c, v, av) : nullptr;
      b.dense(t * (h.n_res + 1) + k, ys, yv);
      s = ggml_add(c, s, ys);
      if (v) v = ggml_add(c, v, yv);
    }
    b.dense(t * (h.n_res + 1) + h.n_res, ys, yv);
    b.silu(ys, yv);
    s = ys, v = yv;
  }

  // --- multitask heads (per atom) ---
  if (heads && h.any_head()) {
    auto head = [&](const char *name) { // x w + b[Z]: [1, n]
      const std::string p = std::string("mt.") + name + ".";
      return ggml_add(c, ggml_mul_mat(c, b.w(p + "w"), s), ggml_get_rows(c, b.w(p + "zbias"), in.species));
    };
    // per-structure sums gathered back to the atoms: [1, n]
    auto structure_sum = [&](ggml_tensor *x) {
      if (!in.system_index) return ggml_repeat(c, ggml_sum(c, x), x);
      ggml_tensor *sums = ggml_get_rows_back(c, x, in.system_index, ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, n));
      return ggml_reshape_2d(c, ggml_get_rows(c, sums, in.system_index), 1, n);
    };
    ggml_tensor *count = (h.head_charges || h.head_spins) ? structure_sum(constant(c, {1, n}, 1.0f)) : nullptr;
    // x + (target - sum_s x) / N_s
    auto constrain = [&](ggml_tensor *x, ggml_tensor *target) {
      ggml_tensor *d = ggml_neg(c, structure_sum(x));
      if (target) d = ggml_add(c, d, target);
      return ggml_reshape_1d(c, ggml_add(c, x, ggml_div(c, d, count)), n);
    };
    if (h.head_charges) heads->charges = constrain(head("charge"), q_atom);
    if (h.head_spins) heads->spins = constrain(head("spin"), mult_atom ? ggml_scale_bias(c, mult_atom, 1.0f, -1.0f) : nullptr);
    if (h.head_orbitals) {
      ggml_tensor *hid = ggml_silu(c, ggml_add(c, ggml_mul_mat(c, b.w("mt.orb.hidden.w"), s), b.w("mt.orb.hidden.b")));
      ggml_tensor *o = ggml_add(c, ggml_mul_mat(c, b.w("mt.orb.out.w"), hid), b.w("mt.orb.out.b")); // [2, n]
      heads->homo = ggml_scale_bias(c, ggml_cont(c, ggml_view_1d(c, ggml_cont(c, ggml_transpose(c, o)), n, 0)), 1.0f,
                                    h.homo_offset);
      heads->gap = ggml_scale_bias(c, ggml_cont(c, ggml_view_1d(c, ggml_cont(c, ggml_transpose(c, o)), n,
                                                              n * sizeof(float))), 1.0f, h.gap_offset);
    }
  }

  // --- atomic energies ---
  ggml_tensor *e = ggml_mul_mat(c, b.w("energy.w"), s); // [1, n]
  e = ggml_mul(c, e, b.w("energy.scale"));
  if (h.energy_bias) e = ggml_add(c, e, ggml_get_rows(c, b.w("energy_bias"), in.species));

  // --- ZBL repulsion (models/zbl.py), per centre atom ---
  if (h.zbl) {
    ggml_tensor *ztab = b.w("zbl.z"), *zatab = b.w("zbl.za"); // [1, Zmax + 1]
    ggml_tensor *ns = ggml_reshape_1d(c, in.neighbor_species, E);
    ggml_tensor *zi, *zai;
    if (center) { // the centre's species at every slot
      ggml_tensor *cs = ggml_get_rows(c, ggml_reshape_2d(c, in.species, 1, n), center); // I32 [1, E]
      cs = ggml_reshape_1d(c, cs, E);
      zi = ggml_reshape_3d(c, ggml_get_rows(c, ztab, cs), 1, m, n);
      zai = ggml_reshape_3d(c, ggml_get_rows(c, zatab, cs), 1, m, n);
    } else {
      ggml_tensor *shape1mn = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, m, n);
      zi = ggml_repeat(c, ggml_reshape_3d(c, ggml_get_rows(c, ztab, in.species), 1, 1, n), shape1mn);
      zai = ggml_repeat(c, ggml_reshape_3d(c, ggml_get_rows(c, zatab, in.species), 1, 1, n), shape1mn);
    }
    ggml_tensor *zj = ggml_reshape_3d(c, ggml_get_rows(c, ztab, ns), 1, m, n);
    ggml_tensor *zaj = ggml_reshape_3d(c, ggml_get_rows(c, zatab, ns), 1, m, n);
    // x = r / a_ij, a_ij = 0.8854 a0 / (Zi^0.23 + Zj^0.23)
    ggml_tensor *x = ggml_scale(c, ggml_mul(c, r, ggml_add(c, zai, zaj)), static_cast<float>(1.0 / (0.8854 * kBohr)));
    double norm = 0;
    for (double v4 : kPhiC) norm += v4 * v4;
    norm = std::sqrt(norm);
    ggml_tensor *phi = nullptr;
    for (int k = 0; k < 4; k++) {
      ggml_tensor *term = ggml_scale(c, ggml_exp(c, ggml_scale(c, x, static_cast<float>(-kPhiA[k]))),
                                     static_cast<float>(kPhiC[k] / norm));
      phi = phi ? ggml_add(c, phi, term) : term;
    }
    // quintic switch, 1 below cuton, 0 above cutoff
    const float range = h.zbl_cutoff - h.zbl_cuton;
    ggml_tensor *xs = ggml_clamp(c, ggml_scale_bias(c, r, -1.0f / range, h.zbl_cutoff / range), 0.0f, 1.0f);
    ggml_tensor *poly = ggml_scale_bias(c, ggml_mul(c, ggml_scale_bias(c, xs, 6.0f, -15.0f), xs), 1.0f, 10.0f);
    ggml_tensor *sw = ggml_mul(c, poly, ggml_mul(c, xs, ggml_sqr(c, xs)));
    ggml_tensor *pair = ggml_div(c, ggml_mul(c, ggml_mul(c, zi, zj), ggml_mul(c, phi, ggml_mul(c, sw, valid))), r);
    pair = ggml_scale(c, pair, static_cast<float>(0.5 * kCoulomb));
    e = ggml_add(c, e, ggml_sum_rows(c, ggml_reshape_2d(c, pair, m, n))); // [1, n]
  }
  return ggml_reshape_1d(c, e, n);
}

} // namespace mlipcpp::physnet
