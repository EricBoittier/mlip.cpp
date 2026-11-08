#include "energy_heads.h"

#include <ggml.h>

#include <stdexcept>

namespace mlipcpp::runtime {

int EnergyHeads::index(const std::string &s) const {
  for (int h = 0; h < n; h++)
    if (h < static_cast<int>(names.size()) && names[h] == s) return h;
  try {
    size_t pos = 0;
    const int h = std::stoi(s, &pos);
    if (pos == s.size() && h >= 0 && h < n) return h;
  } catch (const std::exception &) {
  }
  std::string known;
  for (int h = 0; h < n; h++) known += (h ? ", " : "") + std::to_string(h) + (h < int(names.size()) ? "=" + names[h] : "");
  throw std::runtime_error("energy head \"" + s + "\" not in the model (" + known + ")");
}

int energy_node(const GIRGraph &g) {
  if (g.outputs.empty()) throw std::runtime_error("energy heads: the graph has no output");
  auto id_of = [](const std::string &ref) {
    if (ref.rfind("node:", 0) != 0) throw std::runtime_error("energy heads: output " + ref + " is not a node");
    return std::stoi(ref.substr(5));
  };
  const GIRNode *node = g.get_node(id_of(g.outputs[0].node_ref));
  if (node && node->op == "CONCAT") node = g.get_node(id_of(node->inputs.at(0)));
  while (node && (node->op == "RESHAPE" || node->op == "CONT" || node->op == "VIEW"))
    node = g.get_node(id_of(node->inputs.at(0)));
  if (!node) throw std::runtime_error("energy heads: cannot find the per-atom energy node");
  return node->id;
}

void install_head_select(GraphInterpreter &interp, const EnergyHeads &heads, ggml_tensor *mask) {
  const int target = energy_node(interp.graph());
  std::vector<std::map<std::string, std::string>> subs(heads.n);
  for (int h = 1; h < heads.n; h++)
    for (const auto &w : heads.weights) {
      if (!interp.find_weight(EnergyHeads::prefix(h) + w))
        throw std::runtime_error("energy heads: the GGUF has no tensor " + EnergyHeads::prefix(h) + w);
      subs[h]["weight:" + w] = "weight:" + EnergyHeads::prefix(h) + w;
    }
  auto chained = interp.post_node;
  interp.post_node = [&interp, target, subs, mask, chained, H = heads.n](ggml_context *ctx, const GIRNode &node,
                                                                         ggml_tensor *out) -> ggml_tensor * {
    if (chained) out = chained(ctx, node, out);
    if (node.id != target) return out;
    const int64_t n = ggml_nelements(out);
    if (mask->ne[0] != n || mask->ne[1] != H) throw std::runtime_error("energy heads: head mask shape");
    ggml_tensor *flat = ggml_reshape_1d(ctx, ggml_cont(ctx, out), n);
    ggml_tensor *sum = ggml_mul(ctx, flat, ggml_view_1d(ctx, mask, n, 0));
    for (int h = 1; h < H; h++) {
      ggml_tensor *e = interp.rebuild_substituted(ctx, target, subs[h]);
      e = ggml_reshape_1d(ctx, ggml_cont(ctx, e), n);
      sum = ggml_add(ctx, sum, ggml_mul(ctx, e, ggml_view_1d(ctx, mask, n, size_t(h) * n * sizeof(float))));
    }
    return ggml_reshape(ctx, sum, out);
  };
}

} // namespace mlipcpp::runtime
