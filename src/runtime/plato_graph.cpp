#include "plato_graph.h"

#include <ggml.h>

#include <cmath>
#include <stdexcept>

namespace mlipcpp::runtime {

namespace {

constexpr double KE = 14.399645; // eV A / e^2, as modules/ewald.py

// a product in full FP32 (and its gradient products) even with TF32 GEMMs: the
// geometry (RoPE angles, frames, fractional coordinates) and the long-range phases
// and structure factors, where TF32's 10-bit mantissa costs meV
ggml_tensor *mm_f32(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b) {
  ggml_tensor *y = ggml_mul_mat(ctx, a, b);
  ggml_mul_mat_set_prec(y, GGML_PREC_F32);
  ggml_mul_mat_set_grad_prec(y, GGML_PREC_F32);
  return y;
}

struct B {
  ggml_context *ctx;
  const PlatoSpec &spec;
  const std::map<std::string, ggml_tensor *> &w;
  const PlatoInputs &in;

  ggml_tensor *weight(const std::string &name) const {
    auto it = w.find(name);
    if (it == w.end()) throw std::runtime_error("Plato: missing weight " + name);
    return it->second;
  }
  // torch Linear (weight [out, in] = ggml [in, out])
  ggml_tensor *linear(ggml_tensor *x, const std::string &p, bool bias = true) const {
    ggml_tensor *y = ggml_mul_mat(ctx, weight(p + ".weight"), x);
    return bias ? ggml_add(ctx, y, weight(p + ".bias")) : y;
  }
  // FrameLinear with one frame: W_eff x + b, W_eff = sum_b k_b W_b, k = phi(R^T R = I) * 1
  ggml_tensor *frame_linear(ggml_tensor *x, const std::string &p) const {
    ggml_tensor *W;
    if (!spec.fold_in_graph) {
      W = weight("fold." + p); // ggml [c_in, c_out]
    } else {
      ggml_tensor *k = linear(ggml_silu(ctx, linear(in.rel_identity, p + ".phi.0")), p + ".phi.2"); // [nb]
      ggml_tensor *Wr = weight(p + ".W");                                                            // [c_in, c_out, nb]
      const int64_t ci = Wr->ne[0], co = Wr->ne[1], nb = Wr->ne[2];
      ggml_tensor *Wt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, Wr, ci * co, nb))); // [nb, ci co]
      W = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, Wt, ggml_reshape_2d(ctx, k, nb, 1)), ci, co);
    }
    return ggml_add(ctx, ggml_mul_mat(ctx, W, x), weight(p + ".b"));
  }
  ggml_tensor *layer_norm(ggml_tensor *x, const std::string &p) const {
    if (!spec.decomposed_layer_norm)
      return ggml_pet_layer_norm(ctx, x, weight(p + ".weight"), weight(p + ".bias"), spec.ln_eps);
    const float inv_d = 1.0f / static_cast<float>(x->ne[0]);
    ggml_tensor *c = ggml_sub(ctx, x, ggml_repeat(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, x), inv_d), x));
    ggml_tensor *var = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, c)), inv_d);
    ggml_tensor *y = ggml_div(ctx, c, ggml_repeat(ctx, ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0f, spec.ln_eps)), x));
    return ggml_add(ctx, ggml_mul(ctx, y, weight(p + ".weight")), weight(p + ".bias"));
  }
  // FrameMLP: l2(SiLU(l1(LayerNorm(x))))
  ggml_tensor *frame_mlp(ggml_tensor *x, const std::string &p) const {
    return frame_linear(ggml_silu(ctx, frame_linear(layer_norm(x, p + ".norm"), p + ".l1")), p + ".l2");
  }
  ggml_tensor *cols(ggml_tensor *x, int64_t first, int64_t count) const {
    return ggml_view_2d(ctx, x, x->ne[0], count, x->nb[1], first * x->nb[1]);
  }
  // k (I + eps)^-T to first order: k_b - sum_a eps_ba k_a, per structure (k [3, m, G])
  ggml_tensor *strained(ggml_tensor *k) const { return ggml_sub(ctx, k, mm_f32(ctx, in.strain, k)); }
  // 1 / V(eps) to first order per structure: (1 - tr eps) / V, [1, G]
  ggml_tensor *inv_volume() const {
    const int64_t G = in.strain->ne[2];
    auto e = [&](int k) { return ggml_cont(ctx, ggml_view_2d(ctx, in.strain, 1, G, in.strain->nb[2], k * sizeof(float))); };
    ggml_tensor *tr = ggml_add(ctx, ggml_add(ctx, e(0), e(4)), e(8)); // [1, G]
    return ggml_mul(ctx, ggml_scale_bias(ctx, tr, -1.0f, 1.0f), in.inv_volume);
  }
};

} // namespace

PlatoOutputs build_plato(ggml_context *ctx, const PlatoSpec &spec, const std::map<std::string, ggml_tensor *> &weights,
                         const PlatoInputs &in, int n, int64_t E, int max_len, bool periodic) {
  const B b{ctx, spec, weights, in};
  PlatoOutputs out;
  const int64_t H = spec.heads, ch = spec.head_dim(), F = spec.frames;
  const float rc = spec.cutoff;

  // edge geometry: distance, cutoff, log-cutoff bias
  ggml_tensor *ev = in.edge_vectors;                                     // [3, E]
  ggml_tensor *dist = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, ev))); // [1, E]
  ggml_tensor *x2 = ggml_sqr(ctx, ggml_scale(ctx, dist, 1.0f / rc));
  // 1 - (d / rc)^2, > 0 for the listed edges; clamped at 0 (+ 1e-30 inside the log, a
  // softmax weight of 1e-90) for geometries moved past the list's cutoff (the training
  // step's displaced evaluations), where f_c and its first two derivatives vanish
  ggml_tensor *one_m = ggml_relu(ctx, ggml_scale_bias(ctx, x2, -1.0f, 1.0f));
  ggml_tensor *fc = ggml_mul(ctx, ggml_sqr(ctx, one_m), one_m);          // [1, E]
  ggml_tensor *log_fc = ggml_reshape_1d(ctx, ggml_scale(ctx, ggml_log(ctx, ggml_scale_bias(ctx, one_m, 1.0f, 1e-30f)), 3.0f), E);

  // RoPE angles r . R_f omega_j: [hd F, T], the centres' (r = 0) zero
  ggml_tensor *ang_e = mm_f32(ctx, b.weight("rope.angles"), ev);   // [F hd, E]
  ggml_tensor *ang = ggml_concat(ctx, in.zero_ang, ang_e, 1);            // [F hd, T]

  // edge embedding: sum_f w_f SiLU(W R_f^T r) + rad(rbf(d)) + sp_nbr[s_j] + sp_center[s_i]
  ggml_tensor *vec = nullptr;
  for (int64_t f = 0; f < F; f++) {
    ggml_tensor *rf = mm_f32(ctx, b.weight("rope.frame_t." + std::to_string(f)), ev); // R_f^T r [3, E]
    ggml_tensor *t = ggml_scale(ctx, ggml_silu(ctx, b.linear(rf, "embed.vec", false)), spec.frame_w[f]);
    vec = vec ? ggml_add(ctx, vec, t) : t;
  }
  const int64_t nr = spec.n_rbf;
  const float width = rc / static_cast<float>(nr);
  // rbf_k = exp(-((d - mu_k) / width)^2), mu_k = k rc / (n - 1): [nr, E]
  ggml_tensor *mu = b.weight("rbf.mu"); // [nr]
  ggml_tensor *dd = ggml_sub(ctx, ggml_repeat(ctx, dist, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nr, E)), mu);
  ggml_tensor *rbf = ggml_exp(ctx, ggml_scale(ctx, ggml_sqr(ctx, dd), -1.0f / (width * width)));
  ggml_tensor *edge_tok = ggml_add(ctx, ggml_add(ctx, vec, b.linear(rbf, "embed.rad")),
                                   ggml_add(ctx, ggml_get_rows(ctx, b.weight("embed.sp_nbr.weight"), in.neighbor_species),
                                            ggml_get_rows(ctx, b.weight("embed.sp_center.weight"), in.edge_center_species)));
  ggml_tensor *center_tok = ggml_add(ctx, ggml_add(ctx, ggml_get_rows(ctx, b.weight("embed.sp_nbr.weight"), in.species),
                                                   ggml_get_rows(ctx, b.weight("embed.sp_center.weight"), in.species)),
                                     b.linear(ggml_reshape_2d(ctx, in.rbf0, nr, 1), "embed.rad"));
  ggml_tensor *tok = ggml_concat(ctx, center_tok, edge_tok, 1); // [d, T]

  ggml_tensor *frac = nullptr;
  if (periodic && (spec.mesh || spec.les)) // [3, n]: each atom with the inverse cell of its structure
    frac = ggml_reshape_2d(ctx, mm_f32(ctx, in.atom_inv_cell_t, ggml_reshape_3d(ctx, in.positions, 3, 1, n)), 3, n);
  const int64_t G = in.n_struct;

  // charge / spin conditioning: project(cat(charge_emb, spin_emb)) per atom, [d, n]
  ggml_tensor *cond = nullptr;
  if (spec.conditioning) {
    if (!in.charge_index || !in.spin_index) throw std::runtime_error("Plato: conditioning needs charge / spin inputs");
    ggml_tensor *c = ggml_get_rows(ctx, b.weight("system_conditioning.charge_embedding.weight"), in.charge_index);
    ggml_tensor *s = ggml_get_rows(ctx, b.weight("system_conditioning.spin_multiplicity_embedding.weight"), in.spin_index);
    cond = b.linear(ggml_silu(ctx, b.linear(ggml_concat(ctx, c, s, 0), "system_conditioning.project.0")),
                    "system_conditioning.project.2");
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(ch));
  for (int l = 0; l < spec.layers; l++) {
    const std::string p = "layers." + std::to_string(l) + ".";
    ggml_tensor *t = b.layer_norm(tok, p + "n0");
    ggml_tensor *t_rev = ggml_get_rows(ctx, t, in.reverse);
    ggml_tensor *x = ggml_add(ctx, tok, b.frame_linear(ggml_concat(ctx, t, t_rev, 0), p + "merge"));
    ggml_tensor *qkv = b.frame_linear(b.layer_norm(x, p + "n1"), p + "qkv"); // [3d, T]
    ggml_tensor *o = ggml_plato_attn(ctx, qkv, ang, log_fc, in.offsets, (int) H, max_len, scale, (int) F,
                                     spec.frame_w.data());
    x = ggml_add(ctx, x, b.frame_linear(o, p + "proj"));
    tok = ggml_add(ctx, x, b.frame_linear(ggml_silu(ctx, b.frame_linear(b.layer_norm(x, p + "n2"), p + "mlp1")), p + "mlp2"));
    if (cond) tok = ggml_concat(ctx, ggml_add(ctx, b.cols(tok, 0, n), cond), b.cols(tok, n, E), 1);

    if (l == spec.couple_at && spec.mesh && periodic) {
      const int M0 = in.mesh_size[0], M1 = in.mesh_size[1], M2 = in.mesh_size[2];
      const int64_t Mh = int64_t(M0) * M1 * (M2 / 2 + 1), C = spec.mesh_channels;
      ggml_tensor *xc = b.cols(tok, 0, n);
      ggml_tensor *v = b.frame_linear(b.layer_norm(xc, "coupling.norm"), "coupling.down"); // [C, n]
      // structure g's channels at [g C, (g + 1) C) of one mesh
      ggml_tensor *rho = G > 1 ? ggml_plato_spread_blocked(ctx, v, frac, in.struct_index, (int) G, M0, M1, M2)
                               : ggml_plato_spread(ctx, v, frac, M0, M1, M2);           // [M2, M1, M0, C G]
      ggml_tensor *rk = ggml_plato_fft3(ctx, rho, 0, M0, M1, M2);                         // [2 Mh2, M1, M0, C G]
      // the filter 4 pi / V exp(-sigma^2 k^2 / 2) / k^2 (1 + filt(|k|)) / deconv, under strain, per structure
      ggml_tensor *k = b.strained(in.mesh_k);                                    // [3, Mh, G]
      ggml_tensor *k2 = ggml_sum_rows(ctx, ggml_sqr(ctx, k));                    // [1, Mh, G]
      ggml_tensor *k2s = ggml_add(ctx, k2, in.mesh_k0_pad);                      // 1 at k = 0
      const float s2 = spec.mesh_sigma * spec.mesh_sigma;
      ggml_tensor *base = ggml_div(ctx, ggml_exp(ctx, ggml_scale(ctx, k2, -0.5f * s2)), k2s);
      base = ggml_mul(ctx, ggml_mul(ctx, base, in.mesh_deconv),
                      ggml_reshape_3d(ctx, ggml_scale(ctx, b.inv_volume(), 4.0f * M_PI), 1, 1, in.strain->ne[2]));
      ggml_tensor *filt = b.linear(ggml_silu(ctx, b.linear(ggml_sqrt(ctx, k2s), "coupling.filt.0")), "coupling.filt.2"); // [C, Mh, G]
      ggml_tensor *g = ggml_mul(ctx, ggml_scale_bias(ctx, filt, 1.0f, 1.0f), base);                                     // [C, Mh, G]
      const int64_t Gf = g->ne[2];                                                                                       // G, or 1 (shared)
      g = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, g, 1, 0, 2, 3)), 1, Mh, C, Gf);                         // [1, Mh, C, Gf]
      rk = ggml_reshape_4d(ctx, ggml_mul(ctx, ggml_reshape_4d(ctx, rk, 2, Mh, C, G), g), rk->ne[0], M1, M0, C * G);
      ggml_tensor *phi = ggml_plato_fft3(ctx, rk, 1, M0, M1, M2);                  // [M2, M1, M0, C G]
      ggml_tensor *m = G > 1 ? ggml_plato_gather_blocked(ctx, phi, frac, in.struct_index, (int) C, false)
                             : ggml_plato_gather(ctx, phi, frac, false);            // [C, n]
      ggml_tensor *delta = b.frame_linear(m, "coupling.up");                        // [d, n]
      tok = ggml_concat(ctx, ggml_add(ctx, xc, delta), b.cols(tok, n, E), 1);
    }
  }

  // readout: h = tok_i + sum_e fc_e tok_e; energies per structure
  ggml_tensor *edges = ggml_mul(ctx, b.cols(tok, n, E), fc);
  ggml_tensor *h = ggml_add(ctx, b.cols(tok, 0, n), ggml_get_rows_back(ctx, edges, in.centers, b.cols(tok, 0, n)));
  ggml_tensor *shape_g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, G);
  auto per_structure = [&](ggml_tensor *atom_values) { // [1, n] -> [1, G]
    return G == 1 ? ggml_reshape_2d(ctx, ggml_sum(ctx, atom_values), 1, 1)
                  : ggml_get_rows_back(ctx, ggml_reshape_2d(ctx, atom_values, 1, n), in.struct_index, shape_g);
  };
  out.short_range = per_structure(b.frame_mlp(h, "energy_head")); // [1, G]
  out.energies = out.short_range;
  if (spec.les) {
    const int64_t Q = spec.n_charges;
    if (G > 1 && Q != 1) throw std::runtime_error("Plato: batched LES needs one charge channel");
    ggml_tensor *q = b.frame_mlp(h, "charge_head");                 // [Q, n]
    q = Q == 1 ? ggml_reshape_2d(ctx, q, n, 1) : ggml_cont(ctx, ggml_transpose(ctx, q)); // [n, Q]
    if (G == 1) {
      q = ggml_sub(ctx, q, ggml_repeat(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, q), 1.0f / n), q));
    } else { // the mean of each structure
      ggml_tensor *mean = ggml_mul(ctx, ggml_reshape_2d(ctx, mm_f32(ctx, in.onehot, q), 1, G), in.inv_count); // [1, G]
      q = ggml_sub(ctx, q, ggml_reshape_2d(ctx, ggml_get_rows(ctx, mean, in.struct_index), n, 1));
    }
    out.charges = q;
    const float sigma = spec.les_sigma;
    ggml_tensor *e = nullptr;
    if (periodic && in.les_mesh[0] > 0) {
      if (Q != 1) throw std::runtime_error("Plato: LES PME needs one charge channel");
      const int M0 = in.les_mesh[0], M1 = in.les_mesh[1], M2 = in.les_mesh[2];
      // structure g on channel g of one mesh
      ggml_tensor *rho = ggml_plato_spread_order(ctx, ggml_reshape_2d(ctx, q, 1, n), frac, G > 1 ? in.struct_index : nullptr,
                                                 (int) G, M0, M1, M2, in.les_order);   // [M2, M1, M0, G]
      ggml_tensor *rk = ggml_plato_fft3(ctx, rho, 0, M0, M1, M2);                       // [2 Mh2, M1, M0, G]
      ggml_tensor *K = b.strained(in.les_K);                                            // [3, 3, G]
      ggml_tensor *A = ggml_reshape_2d(ctx, mm_f32(ctx, K, K), 9, K->ne[2]);            // k^2 = m^T A m
      const float kc = spec.les_kcut / sigma, wd = 1.0f / sigma;
      ggml_tensor *ep = ggml_plato_pme(ctx, rk, A, in.les_tab, M0, M1, M2, sigma, kc, wd); // [M0, G]
      // (2 pi / V) sum over the full spectrum = (4 pi / V) over the half space, as the direct sum
      ggml_tensor *rec = ggml_mul(ctx, ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ep), 1, G), ggml_scale(ctx, b.inv_volume(), 2.0f * M_PI));
      ggml_tensor *qb = G == 1 ? q : ggml_mul(ctx, in.onehot, q);
      ggml_tensor *self = G == 1 ? ggml_reshape_2d(ctx, ggml_sum(ctx, ggml_sqr(ctx, q)), 1, 1)
                                 : ggml_sum_rows(ctx, ggml_sqr(ctx, qb));
      self = ggml_scale(ctx, self, 1.0f / (std::sqrt(2.0f * M_PI) * sigma));
      e = ggml_scale(ctx, ggml_sub(ctx, rec, self), KE);
    } else if (periodic) {
      ggml_tensor *qb = G == 1 ? q : ggml_mul(ctx, in.onehot, q);    // [n, G Q]: each structure's charges
      const int64_t nk = in.les_n->ne[1];
      ggml_tensor *sk;                                               // [nk, G Q]: |S(k)|^2
      if (Q == 1) { // fused structure factors (no [n, nk] phase matrices)
        ggml_tensor *r = ggml_plato_ewald(ctx, frac, ggml_reshape_1d(ctx, q, n), G == 1 ? nullptr : in.struct_index, in.les_n, (int) G);
        sk = ggml_view_2d(ctx, r, nk, G, r->nb[1], 2 * r->nb[2]);
      } else {
        ggml_tensor *ph = mm_f32(ctx, frac, in.les_n);               // [n, nk]
        ggml_tensor *Cq = mm_f32(ctx, ggml_cos(ctx, ph), qb);        // [nk, G Q]
        ggml_tensor *Sq = mm_f32(ctx, ggml_sin(ctx, ph), qb);
        sk = ggml_add(ctx, ggml_sqr(ctx, Cq), ggml_sqr(ctx, Sq));
      }
      ggml_tensor *k = b.strained(in.les_k);                         // [3, nk, G]
      ggml_tensor *k2 = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, k)), nk, in.les_k->ne[2]); // [nk, G or 1]
      const float kc = spec.les_kcut / sigma, wd = 1.0f / sigma;
      // cosine window over |k| in (kc - wd, kc): x = (|k| - (kc - wd)) / wd clamped to [0, 1], so 1 inside
      // kc - wd and 0 from kc on; a batch of different cells shares one integer k set (the union) and
      // each structure's k beyond kc drops out
      ggml_tensor *xw = ggml_clamp(ctx, ggml_scale_bias(ctx, ggml_sqrt(ctx, k2), 1.0f / wd, -(kc - wd) / wd), 0.0f, 1.0f);
      ggml_tensor *win = ggml_scale_bias(ctx, ggml_cos(ctx, ggml_scale(ctx, xw, static_cast<float>(M_PI))), 0.5f, 0.5f);
      ggml_tensor *gk = ggml_div(ctx, ggml_mul(ctx, win, ggml_exp(ctx, ggml_scale(ctx, k2, -0.5f * sigma * sigma))), k2);
      ggml_tensor *rec;
      if (G == 1) rec = ggml_reshape_2d(ctx, ggml_sum(ctx, ggml_mul(ctx, sk, gk)), 1, 1);
      else rec = ggml_sum_rows(ctx, ggml_mul(ctx, sk, gk));                       // [1, G]
      rec = ggml_mul(ctx, rec, ggml_scale(ctx, b.inv_volume(), 4.0f * M_PI));
      ggml_tensor *self = G == 1 ? ggml_reshape_2d(ctx, ggml_sum(ctx, ggml_sqr(ctx, q)), 1, 1)
                                 : ggml_sum_rows(ctx, ggml_sqr(ctx, qb));          // [1, G]
      self = ggml_scale(ctx, self, 1.0f / (std::sqrt(2.0f * M_PI) * sigma));
      e = ggml_scale(ctx, ggml_sub(ctx, rec, self), KE);
    } else {
      if (G != 1) throw std::runtime_error("Plato: open-system LES for one structure at a time");
      ggml_tensor *pos = in.positions;                                          // [3, n]
      ggml_tensor *shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, n, n);
      ggml_tensor *rij = ggml_sub(ctx, ggml_repeat(ctx, ggml_reshape_3d(ctx, pos, 3, n, 1), shape),
                                  ggml_repeat(ctx, ggml_reshape_3d(ctx, pos, 3, 1, n), shape));
      ggml_tensor *d2 = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, rij)), n, n);
      // 1 on the diagonal: d = sqrt(d2 + I)
      ggml_tensor *dij = ggml_sqrt(ctx, ggml_add(ctx, d2, ggml_scale_bias(ctx, in.offdiag, -1.0f, 1.0f)));
      ggml_tensor *pair = ggml_mul(ctx, ggml_div(ctx, ggml_plato_erf(ctx, ggml_scale(ctx, dij, 1.0f / (std::sqrt(2.0f) * sigma))), dij),
                                   in.offdiag);
      e = ggml_reshape_2d(ctx, ggml_scale(ctx, ggml_sum(ctx, ggml_mul(ctx, q, mm_f32(ctx, pair, q))), 0.5f * KE), 1, 1);
    }
    out.les = e;
    out.energies = ggml_add(ctx, out.energies, e);
    if (spec.chi) {
      // in model units: sum_i chi_i q_m,i / charge_unit (q_m: the centred first channel)
      ggml_tensor *dchi = b.frame_mlp(h, "electronegativity.0.head");               // [1, n]
      if (spec.chi_bound > 0) dchi = ggml_scale(ctx, ggml_tanh(ctx, dchi), spec.chi_bound);
      ggml_tensor *chi0 = b.weight("electronegativity.0.chi0");
      ggml_tensor *chi = ggml_add(ctx, ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_2d(ctx, chi0, 1, chi0->ne[0]), in.species), 1, n), dchi);
      out.chi_atoms = chi;
      ggml_tensor *q0 = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, q, n, 1, q->nb[1], 0)), 1, n);
      ggml_tensor *ea = ggml_div(ctx, ggml_mul(ctx, chi, q0), b.weight("electronegativity.0.charge_unit"));
      out.chi = per_structure(ea);
      out.energies = ggml_add(ctx, out.energies, out.chi);
    }
  }
  out.total = in.energy_weights ? ggml_sum(ctx, ggml_mul(ctx, out.energies, in.energy_weights)) : ggml_sum(ctx, out.energies);
  if (in.charge_weights && out.charges) { // a loss on the charges enters the parameter gradient through total
    ggml_tensor *q0 = ggml_view_2d(ctx, out.charges, n, 1, out.charges->nb[1], 0);
    out.total = ggml_add(ctx, out.total, ggml_sum(ctx, ggml_mul(ctx, q0, in.charge_weights)));
  }
  return out;
}

std::map<std::string, std::vector<float>> plato_fold_weights(const std::map<std::string, std::vector<float>> &raw,
                                                             const std::map<std::string, std::vector<int64_t>> &shapes,
                                                             std::vector<std::string> &prefixes) {
  std::map<std::string, std::vector<float>> out;
  prefixes.clear();
  const std::vector<float> I = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (const auto &[name, W] : raw) {
    const std::string suffix = ".W";
    if (name.size() < 2 || name.compare(name.size() - 2, 2, suffix) != 0) continue;
    const std::string p = name.substr(0, name.size() - 2);
    if (!raw.count(p + ".phi.0.weight")) continue;
    const auto &s = shapes.at(name); // [nb, c_out, c_in]
    const int64_t nb = s[0], co = s[1], ci = s[2];
    const auto &w0 = raw.at(p + ".phi.0.weight"), &b0 = raw.at(p + ".phi.0.bias");
    const auto &w2 = raw.at(p + ".phi.2.weight"), &b2 = raw.at(p + ".phi.2.bias");
    const int64_t hid = static_cast<int64_t>(b0.size());
    std::vector<double> hdn(hid), k(nb);
    for (int64_t j = 0; j < hid; j++) {
      double a = b0[j];
      for (int i = 0; i < 9; i++) a += double(w0[j * 9 + i]) * I[i];
      hdn[j] = a / (1.0 + std::exp(-a));
    }
    for (int64_t q = 0; q < nb; q++) {
      double a = b2[q];
      for (int64_t j = 0; j < hid; j++) a += double(w2[q * hid + j]) * hdn[j];
      k[q] = a; // times the frame weight 1
    }
    std::vector<float> f(co * ci);
    for (int64_t o = 0; o < co; o++)
      for (int64_t c = 0; c < ci; c++) {
        double a = 0;
        for (int64_t q = 0; q < nb; q++) a += k[q] * W[(q * co + o) * ci + c];
        f[o * ci + c] = static_cast<float>(a);
      }
    out["fold." + p] = std::move(f);
    prefixes.push_back(p);
  }
  return out;
}

} // namespace mlipcpp::runtime
