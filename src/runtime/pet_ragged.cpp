#include "pet_ragged.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace mlipcpp::runtime {

namespace {

// A/B switches of this builder: set and not "0" (env_flag), or unset or not "0" (env_on, a default)
bool env_flag(const char *name) {
  const char *e = std::getenv(name);
  return e && *e && *e != '0';
}
bool env_on(const char *name) {
  const char *e = std::getenv(name);
  return !e || !*e || *e != '0';
}

struct Builder {
  ggml_context *ctx;
  const std::map<std::string, ggml_tensor *> &w;

  ggml_tensor *weight(const std::string &name) const {
    auto it = w.find(name);
    if (it == w.end()) throw std::runtime_error("PET (ragged): missing weight " + name);
    return it->second;
  }
  // torch Linear: W [out, in] is ggml [in, out]
  ggml_tensor *linear(ggml_tensor *x, const std::string &name) const {
    ggml_tensor *y = ggml_mul_mat(ctx, weight(name + "_weight"), x);
    auto b = w.find(name + "_bias");
    return b == w.end() ? y : ggml_add(ctx, y, b->second);
  }
  ggml_tensor *rms_norm(ggml_tensor *x, const std::string &name, float eps) const {
    // one op, whose gradient also sums the residual's (MLIPCPP_RAGGED_FUSE_RMS=0: RMS_NORM and MUL)
    static const bool fuse = env_on("MLIPCPP_RAGGED_FUSE_RMS");
    if (fuse) return ggml_pet_rms_norm(ctx, x, weight(name + "_weight"), eps);
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight(name + "_weight"));
  }
  ggml_tensor *layer_norm(ggml_tensor *x, const std::string &name, float eps) const {
    // MLIPCPP_RAGGED_LN_DECOMP=1: the LayerNorm of primitive ops (as GraphInterpreter::build_layer_norm)
    static const bool decomp = env_flag("MLIPCPP_RAGGED_LN_DECOMP");
    if (!decomp) return ggml_pet_layer_norm(ctx, x, weight(name + "_weight"), weight(name + "_bias"), eps);
    const float inv_d = 1.0f / static_cast<float>(x->ne[0]);
    ggml_tensor *centred = ggml_sub(ctx, x, ggml_repeat(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, x), inv_d), x));
    ggml_tensor *var = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, centred)), inv_d);
    ggml_tensor *sd = ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0f, eps));
    ggml_tensor *y = ggml_div(ctx, centred, ggml_repeat(ctx, sd, x));
    return ggml_add(ctx, ggml_mul(ctx, y, weight(name + "_weight")), weight(name + "_bias"));
  }
  // FeedForward with SwiGLU: v, g = chunk(w_in(x)); w_out(v * sigmoid(g)). Split form: the two
  // halves of w_in are blocks of its rows: two products instead of views of one
  // (the gradient of a second view of a tensor is a full-size ACC)
  ggml_tensor *gated_mlp(ggml_tensor *x, const std::string &name) const {
    // MLIPCPP_RAGGED_GLU_VIEWS=1 (with MLIPCPP_RAGGED_FUSE_GLU=0): one product and two views of it (the older form)
    static const bool glu_views = env_flag("MLIPCPP_RAGGED_GLU_VIEWS");
    // one product and one op over all of it, whose gradient is one [2f, rows]
    // tensor: one gradient product, no ADD of two (MLIPCPP_RAGGED_FUSE_GLU=0: the split form below)
    static const bool fuse_glu = env_on("MLIPCPP_RAGGED_FUSE_GLU");
    if (fuse_glu) return linear(ggml_pet_glu(ctx, linear(x, name + "_w_in")), name + "_w_out");
    if (glu_views) {
      ggml_tensor *h = linear(x, name + "_w_in");
      const int64_t f = h->ne[0] / 2;
      ggml_tensor *v = ggml_view_2d(ctx, h, f, h->ne[1], h->nb[1], 0);
      ggml_tensor *g = ggml_view_2d(ctx, h, f, h->ne[1], h->nb[1], f * ggml_element_size(h));
      return linear(ggml_siglu_split(ctx, g, v), name + "_w_out");
    }
    ggml_tensor *w_in = weight(name + "_w_in_weight"), *b_in = weight(name + "_w_in_bias"); // [in, 2f], [2f]
    const int64_t f = w_in->ne[1] / 2;
    auto half = [&](int k) {
      ggml_tensor *w = ggml_view_2d(ctx, w_in, w_in->ne[0], f, w_in->nb[1], k * f * w_in->nb[1]);
      return ggml_add(ctx, ggml_mul_mat(ctx, w, x), ggml_view_1d(ctx, b_in, f, k * f * ggml_element_size(b_in)));
    };
    return linear(ggml_siglu_split(ctx, half(1), half(0)), name + "_w_out");
  }
  // Linear, SiLU, Linear (, SiLU)
  ggml_tensor *mlp2(ggml_tensor *x, const std::string &a, const std::string &b, bool silu_out) const {
    ggml_tensor *y = linear(ggml_silu(ctx, linear(x, a)), b);
    return silu_out ? ggml_silu(ctx, y) : y;
  }
  ggml_tensor *columns(ggml_tensor *x, int64_t first, int64_t count) const {
    return ggml_view_2d(ctx, x, x->ne[0], count, x->nb[1], first * x->nb[1]);
  }
};

} // namespace

bool pet_ragged_spec(const std::map<std::string, ggml_tensor *> &w, int64_t d_pet, int heads, float cutoff_width,
                     PetRaggedSpec &spec, std::string &why) {
  auto has = [&](const std::string &name) { return w.count(name) > 0; };
  spec = {};
  while (has("gnn_layers_" + std::to_string(spec.layers) + "_compress_0_weight")) spec.layers++;
  if (spec.layers == 0) return why = "no gnn_layers_*_compress weights", false;
  if (!has("node_embedders_0_weight") || !has("neighbor_embedder_weight"))
    return why = "no node / neighbour embedders", false;
  if (has("node_embedders_1_weight")) return why = "residual featurizer (several readout layers)", false;
  if (!has("combination_mlps_0_0_weight") || !has("combination_norms_0_weight"))
    return why = "no combination MLPs (not the feedforward featurizer)", false;
  for (int l = 0; l < spec.layers; l++) {
    const std::string t = "gnn_layers_" + std::to_string(l) + "_trans_layers_";
    if (has(t + "1_attention_input_linear_weight")) return why = "more than one attention layer per GNN layer", false;
    for (const char *name : {"0_attention_input_linear_weight", "0_attention_output_linear_weight",
                             "0_norm_attention_weight", "0_norm_mlp_weight", "0_mlp_w_in_weight",
                             "0_mlp_w_out_weight", "0_center_contraction_weight", "0_center_expansion_weight",
                             "0_norm_center_features_weight", "0_center_mlp_w_in_weight"})
      if (!has(t + name)) return why = std::string("no ") + t + name, false;
    if (has(t + "0_norm_attention_bias")) return why = "LayerNorm transformer (RMSNorm only)", false;
    if (l > 0 && !has("gnn_layers_" + std::to_string(l) + "_neighbor_embedder_weight"))
      return why = "no neighbour embedder in a later layer", false;
  }
  for (const char *name : {"node_energy_heads_0_0_weight", "node_final_layers_0_weight",
                           "edge_energy_heads_0_0_weight", "edge_final_layers_0_weight"})
    if (!has(name)) return why = std::string("no ") + name, false;
  const ggml_tensor *qkv = w.at("gnn_layers_0_trans_layers_0_attention_input_linear_weight");
  const ggml_tensor *mlp = w.at("gnn_layers_0_trans_layers_0_mlp_w_in_weight");
  const ggml_tensor *out = w.at("gnn_layers_0_trans_layers_0_mlp_w_out_weight");
  if (qkv->ne[0] != d_pet || qkv->ne[1] != 3 * d_pet) return why = "attention width is not d_pet", false;
  if (mlp->ne[1] != 2 * out->ne[0]) return why = "edge MLP is not gated (SwiGLU)", false;
  if (heads <= 0 || d_pet % heads != 0) return why = "unknown number of attention heads", false;
  const int64_t d = d_pet / heads;
  if (d != 8 && d != 16 && d != 32 && d != 64) return why = "head size not 8, 16, 32 or 64", false;
  spec.heads = heads, spec.d_pet = d_pet;
  spec.d_node = w.at("node_embedders_0_weight")->ne[0];
  spec.cutoff_width = cutoff_width;
  return true;
}

PetRaggedOutputs build_pet_ragged(ggml_context *ctx, const PetRaggedSpec &spec,
                                  const std::map<std::string, ggml_tensor *> &weights, const PetRaggedInputs &in,
                                  int n, int64_t e_cap, int max_len) {
  const Builder b{ctx, weights};
  const int64_t d_pet = spec.d_pet;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d_pet / spec.heads));

  // distances: + 1e-15 outside the root for the cutoff (metatrain's
  // compute_batch_tensors), inside for the edge features (its NEF distances)
  ggml_tensor *ev = in.edge_vectors;                              // [3, E]
  ggml_tensor *r2 = ggml_sum_rows(ctx, ggml_sqr(ctx, ev));        // [1, E]
  ggml_tensor *d_cut = ggml_scale_bias(ctx, ggml_sqrt(ctx, r2), 1.0f, 1e-15f);
  ggml_tensor *d_emb = ggml_sqrt(ctx, ggml_scale_bias(ctx, r2, 1.0f, 1e-15f));
  ggml_tensor *bump = ggml_pet_bump(ctx, ggml_reshape_1d(ctx, d_cut, e_cap), in.cutoff_values, spec.cutoff_width, 1e-15f);
  ggml_tensor *fc = ggml_view_2d(ctx, bump, e_cap, 1, bump->nb[1], 0);              // [E, 1]
  ggml_tensor *log_fc = ggml_reshape_1d(ctx, ggml_view_2d(ctx, bump, e_cap, 1, bump->nb[1], bump->nb[1]), e_cap); // [E]
  ggml_tensor *edge_geometry = ggml_concat(ctx, ev, d_emb, 0);                      // [4, E]

  ggml_tensor *node = ggml_get_rows(ctx, b.weight("node_embedders_0_weight"), in.species);              // [d_node, n]
  ggml_tensor *messages = ggml_get_rows(ctx, b.weight("neighbor_embedder_weight"), in.neighbor_species); // [d_pet, E]

  for (int l = 0; l < spec.layers; l++) {
    const std::string g = "gnn_layers_" + std::to_string(l) + "_", t = g + "trans_layers_0_";
    // edge tokens: compress(cat(edge_embedder(v, d), [neighbour embedding,] messages)).
    // compress_0 is linear, so with W = [W_a | W_b | W_c] over the three parts:
    //   W_a (W_ee [v, d] + b_ee) + W_b Emb[s] + W_c messages + b
    //   = (W_a W_ee) [v, d] + (W_b Emb)[s] + W_c messages + (W_a b_ee + b)
    // the products of weights are small (d_pet x 4, d_pet x species) and replace the
    // concatenated [3 d_pet, E] input; in layer 0 the messages are an embedding too
    // MLIPCPP_RAGGED_NO_FOLD=1: the concatenated input and the full compress_0 instead
    static const bool no_fold = env_flag("MLIPCPP_RAGGED_NO_FOLD");
    ggml_tensor *edge_tokens = nullptr;
    if (no_fold) {
      ggml_tensor *tokens_in = b.linear(edge_geometry, g + "edge_embedder");
      if (l > 0)
        tokens_in = ggml_concat(ctx, tokens_in, ggml_get_rows(ctx, b.weight(g + "neighbor_embedder_weight"), in.neighbor_species), 0);
      tokens_in = ggml_concat(ctx, tokens_in, messages, 0);
      edge_tokens = b.mlp2(tokens_in, g + "compress_0", g + "compress_2", false);
    } else {
    ggml_tensor *w0 = b.weight(g + "compress_0_weight"); // [k d_pet, d_pet]
    auto block = [&](int k) { // columns [k d_pet, (k + 1) d_pet) of compress_0 as a [d_pet, d_pet] weight
      return ggml_cont(ctx, ggml_view_2d(ctx, w0, d_pet, d_pet, w0->nb[1], k * d_pet * ggml_element_size(w0)));
    };
    // the products, computed once in double by pet_ragged_fold_weights when the weights
    // hold them (otherwise here, in the graph: in TF32 with MLIPCPP_TF32=1)
    ggml_tensor *geometry_w, *bias, *table;
    if (weights.count(g + "fold_geometry")) {
      geometry_w = b.weight(g + "fold_geometry"), bias = b.weight(g + "fold_bias"), table = b.weight(g + "fold_table");
    } else {
      ggml_tensor *w_a = block(0);
      ggml_tensor *w_ee = b.weight(g + "edge_embedder_weight"); // [4, d_pet]
      geometry_w = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, w_ee)), w_a); // [4, d_pet]
      bias = ggml_add(ctx, ggml_mul_mat(ctx, w_a, ggml_reshape_2d(ctx, b.weight(g + "edge_embedder_bias"), d_pet, 1)),
                      ggml_reshape_2d(ctx, b.weight(g + "compress_0_bias"), d_pet, 1));
      table = l > 0 ? ggml_mul_mat(ctx, block(1), b.weight(g + "neighbor_embedder_weight")) // [d_pet, species]
                    : ggml_mul_mat(ctx, block(1), b.weight("neighbor_embedder_weight"));
    }
    // the K = 4 product of the raw edge vectors and distances in full FP32 even with
    // MLIPCPP_TF32=1: in TF32 its inputs lose all but 10 mantissa bits, which put
    // 0.6 meV/atom into PET-MAD-xs energies and most of the TF32 force error
    // MLIPCPP_RAGGED_GEOM_TF32=1: in TF32 like the other products
    static const bool geom_tf32 = env_flag("MLIPCPP_RAGGED_GEOM_TF32");
    ggml_tensor *geo = ggml_mul_mat(ctx, geometry_w, edge_geometry);
    if (!geom_tf32) ggml_mul_mat_set_prec(geo, GGML_PREC_F32);
    ggml_tensor *pre = ggml_add(ctx, geo, ggml_get_rows(ctx, table, in.neighbor_species));
    if (l > 0) pre = ggml_add(ctx, pre, ggml_mul_mat(ctx, block(2), messages));
    edge_tokens = b.linear(ggml_silu(ctx, ggml_add(ctx, pre, bias)), g + "compress_2"); // [d_pet, E]
    }

    // one PreLN transformer layer over [central token, its edges] per atom
    ggml_tensor *tokens = ggml_concat(ctx, b.linear(node, t + "center_contraction"), edge_tokens, 1); // [d_pet, n + E]
    ggml_tensor *qkv = b.linear(b.rms_norm(tokens, t + "norm_attention", spec.rms_eps), t + "attention_input_linear");
    ggml_tensor *attn = ggml_pet_attn_ragged(ctx, qkv, log_fc, in.offsets, spec.heads, max_len, scale);
    ggml_tensor *o = ggml_view_2d(ctx, attn, d_pet, n + e_cap, attn->nb[1], 0);
    o = b.linear(o, t + "attention_output_linear");

    node = ggml_add(ctx, node, b.linear(b.columns(o, 0, n), t + "center_expansion"));
    node = ggml_add(ctx, node, b.gated_mlp(b.rms_norm(node, t + "norm_center_features", spec.rms_eps), t + "center_mlp"));

    ggml_tensor *edges = ggml_add(ctx, edge_tokens, b.columns(o, n, e_cap));
    edges = ggml_add(ctx, edges, b.gated_mlp(b.rms_norm(edges, t + "norm_mlp", spec.rms_eps), t + "mlp"));

    // messages of the next layer: the reversed edges mixed in
    const std::string c = std::to_string(l);
    // LayerNorm([edge; its reverse]) as one op: no gather and concatenation of [2 d_pet, E]
    // (the reverse of an edge's reverse is the edge, spare rows are their own reverse)
    // MLIPCPP_RAGGED_PAIR_CONCAT=1: gather the reverse rows, concatenate, then LayerNorm
    static const bool pair_concat = env_flag("MLIPCPP_RAGGED_PAIR_CONCAT");
    ggml_tensor *paired = pair_concat
        ? b.layer_norm(ggml_concat(ctx, edges, ggml_get_rows(ctx, edges, in.reverse), 0), "combination_norms_" + c, spec.ln_eps)
        : ggml_pet_pair_norm(ctx, edges, in.reverse, b.weight("combination_norms_" + c + "_weight"),
                             b.weight("combination_norms_" + c + "_bias"), spec.ln_eps);
    ggml_tensor *mixed = b.mlp2(paired, "combination_mlps_" + c + "_0", "combination_mlps_" + c + "_2", false);
    messages = ggml_add(ctx, ggml_add(ctx, messages, edges), mixed);
  }

  // readout: node head, and edge head weighted by the cutoff and summed per atom
  ggml_tensor *node_e = b.linear(b.mlp2(node, "node_energy_heads_0_0", "node_energy_heads_0_2", true),
                                 "node_final_layers_0"); // [1, n]
  ggml_tensor *edge_e = b.linear(b.mlp2(messages, "edge_energy_heads_0_0", "edge_energy_heads_0_2", true),
                                 "edge_final_layers_0"); // [1, E]
  edge_e = ggml_mul(ctx, edge_e, ggml_reshape_2d(ctx, fc, 1, e_cap));

  PetRaggedOutputs out;
  out.total = ggml_add(ctx, ggml_sum(ctx, node_e), ggml_sum(ctx, edge_e));
  // per atom (no gradient needed): edge contributions scattered onto their centres
  ggml_tensor *per_atom_edges = ggml_get_rows_back(ctx, ggml_reshape_2d(ctx, edge_e, 1, e_cap), in.centers, node_e);
  out.atom_energy = ggml_add(ctx, node_e, per_atom_edges);
  return out;
}

bool pet_ragged_fold_weights(ggml_backend_buffer_type *buft, const PetRaggedSpec &spec,
                             std::map<std::string, ggml_tensor *> &w, ggml_context **ctx_out,
                             ggml_backend_buffer **buffer_out) {
  const int64_t d = spec.d_pet;
  auto host = [&](const std::string &name, ggml_tensor *&t) {
    t = w.at(name);
    std::vector<float> v;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return v;
    v.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
  };
  struct Fold {
    std::string name;
    int64_t ne0, ne1;
    std::vector<float> v;
    ggml_tensor *t = nullptr;
  };
  std::vector<Fold> folds;
  for (int l = 0; l < spec.layers; l++) {
    const std::string g = "gnn_layers_" + std::to_string(l) + "_";
    ggml_tensor *t0, *tee, *tbee, *tb0, *temb;
    const std::vector<float> w0 = host(g + "compress_0_weight", t0), wee = host(g + "edge_embedder_weight", tee),
                             bee = host(g + "edge_embedder_bias", tbee), b0 = host(g + "compress_0_bias", tb0),
                             emb = host(l > 0 ? g + "neighbor_embedder_weight" : "neighbor_embedder_weight", temb);
    if (w0.empty() || wee.empty() || bee.empty() || b0.empty() || emb.empty()) return false;
    const int64_t k_in = t0->ne[0], n_geo = tee->ne[0], n_species = temb->ne[1];
    // compress_0 row o (an output), column c of block k: w0[o k_in + k d + c]
    auto wa = [&](int64_t o, int64_t c) { return double(w0[o * k_in + c]); };
    auto wb = [&](int64_t o, int64_t c) { return double(w0[o * k_in + d + c]); };
    Fold geo{g + "fold_geometry", n_geo, d, std::vector<float>(n_geo * d)}; // W_a W_ee
    Fold bias{g + "fold_bias", d, 1, std::vector<float>(d)};                 // W_a b_ee + b
    Fold table{g + "fold_table", d, n_species, std::vector<float>(d * n_species)}; // W_b Emb
    for (int64_t o = 0; o < d; o++) {
      for (int64_t i = 0; i < n_geo; i++) {
        double s = 0;
        for (int64_t k = 0; k < d; k++) s += wa(o, k) * wee[k * n_geo + i];
        geo.v[o * n_geo + i] = static_cast<float>(s);
      }
      double s = b0[o];
      for (int64_t k = 0; k < d; k++) s += wa(o, k) * bee[k];
      bias.v[o] = static_cast<float>(s);
      for (int64_t sp = 0; sp < n_species; sp++) {
        double t = 0;
        for (int64_t k = 0; k < d; k++) t += wb(o, k) * emb[sp * d + k];
        table.v[sp * d + o] = static_cast<float>(t);
      }
    }
    folds.push_back(std::move(geo)), folds.push_back(std::move(bias)), folds.push_back(std::move(table));
  }
  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * folds.size(), nullptr, /*no_alloc=*/true});
  if (!ctx) return false;
  for (Fold &f : folds) f.t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, f.ne0, f.ne1), ggml_set_name(f.t, f.name.c_str());
  ggml_backend_buffer *buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
  if (!buffer) return ggml_free(ctx), false;
  for (Fold &f : folds) ggml_backend_tensor_set(f.t, f.v.data(), 0, ggml_nbytes(f.t)), w[f.name] = f.t;
  *ctx_out = ctx, *buffer_out = buffer;
  return true;
}

} // namespace mlipcpp::runtime

