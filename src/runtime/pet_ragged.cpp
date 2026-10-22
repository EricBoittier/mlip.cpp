#include "pet_ragged.h"

#include <ggml.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace mlipcpp::runtime {

namespace {

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
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight(name + "_weight"));
  }
  ggml_tensor *layer_norm(ggml_tensor *x, const std::string &name, float eps) const {
    return ggml_pet_layer_norm(ctx, x, weight(name + "_weight"), weight(name + "_bias"), eps);
  }
  // FeedForward with SwiGLU: v, g = chunk(w_in(x)); w_out(v * sigmoid(g)). The two
  // halves of w_in are blocks of its rows: two products instead of views of one
  // (the gradient of a second view of a tensor is a full-size ACC)
  ggml_tensor *gated_mlp(ggml_tensor *x, const std::string &name) const {
    // MLIPCPP_RAGGED_GLU_VIEWS=1: one product and two views of it (the older form)
    static const bool glu_views = [] {
      const char *e = std::getenv("MLIPCPP_RAGGED_GLU_VIEWS");
      return e && *e && *e != '0';
    }();
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
    ggml_tensor *w0 = b.weight(g + "compress_0_weight"); // [k d_pet, d_pet]
    auto block = [&](int k) { // columns [k d_pet, (k + 1) d_pet) of compress_0 as a [d_pet, d_pet] weight
      return ggml_cont(ctx, ggml_view_2d(ctx, w0, d_pet, d_pet, w0->nb[1], k * d_pet * ggml_element_size(w0)));
    };
    ggml_tensor *w_a = block(0);
    ggml_tensor *w_ee = b.weight(g + "edge_embedder_weight"); // [4, d_pet]
    ggml_tensor *geometry_w = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, w_ee)), w_a); // [4, d_pet]
    ggml_tensor *bias = ggml_add(ctx, ggml_mul_mat(ctx, w_a, ggml_reshape_2d(ctx, b.weight(g + "edge_embedder_bias"), d_pet, 1)),
                                 ggml_reshape_2d(ctx, b.weight(g + "compress_0_bias"), d_pet, 1));
    ggml_tensor *table = l > 0 ? ggml_mul_mat(ctx, block(1), b.weight(g + "neighbor_embedder_weight")) // [d_pet, species]
                               : ggml_mul_mat(ctx, block(1), b.weight("neighbor_embedder_weight"));
    ggml_tensor *pre = ggml_add(ctx, ggml_mul_mat(ctx, geometry_w, edge_geometry), ggml_get_rows(ctx, table, in.neighbor_species));
    if (l > 0) pre = ggml_add(ctx, pre, ggml_mul_mat(ctx, block(2), messages));
    ggml_tensor *edge_tokens = b.linear(ggml_silu(ctx, ggml_add(ctx, pre, bias)), g + "compress_2"); // [d_pet, E]

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
    ggml_tensor *paired = ggml_pet_pair_norm(ctx, edges, in.reverse, b.weight("combination_norms_" + c + "_weight"),
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

} // namespace mlipcpp::runtime
