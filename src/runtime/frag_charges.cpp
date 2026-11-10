#include "frag_charges.h"
#include "graph_interpreter.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mlipcpp::runtime {

namespace {

constexpr double KE = 14.399645; // eV A / e^2, as metatrain

// ase.data.covalent_radii (Cordero et al. 2008), index = Z
constexpr double kCovalent[119] = {
    0.2,  0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03,
    1.76, 1.7,  1.6,  1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.2,  1.19, 1.2,  1.2,  1.16, 2.2,  1.95, 1.9,
    1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.4,  2.44, 2.15, 2.07, 2.04, 2.03,
    2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.9,  1.87, 1.87, 1.75, 1.7,  1.62, 1.51, 1.44, 1.41, 1.36, 1.36,
    1.32, 1.45, 1.46, 1.48, 1.4,  1.5,  1.5,  2.6,  2.21, 2.15, 2.06, 2.0,  1.96, 1.9,  1.87, 1.8,  1.69, 2.0,  2.0,  2.0,
    2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0};

bool metal(int z) { // transition (21-30, 39-48, 72-80), lanthanides (57-71), post-transition (Al Ga In Sn Tl Pb Bi)
  return (z >= 21 && z <= 30) || (z >= 39 && z <= 48) || (z >= 57 && z <= 80) || z == 13 || z == 31 || z == 49 ||
         z == 50 || z == 81 || z == 82 || z == 83;
}
bool alkali(int z) { return z == 3 || z == 11 || z == 19 || z == 37 || z == 55 || z == 87; }

int find(std::vector<int> &p, int i) {
  while (p[i] != i) i = p[i] = p[p[i]];
  return i;
}

} // namespace

std::vector<int32_t> frag_bonded_units(int n, const int32_t *Z, const float *x, int *n_units) {
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  for (int i = 0; i < n; i++) {
    if (Z[i] < 0 || Z[i] > 118) throw std::runtime_error("frag_charges: atomic number out of range");
    if (alkali(Z[i])) continue;
    const double ri = kCovalent[Z[i]], fi = metal(Z[i]) ? 1.3 : 1.2;
    for (int j = i + 1; j < n; j++) {
      if (alkali(Z[j])) continue;
      const double f = std::max(fi, metal(Z[j]) ? 1.3 : 1.2), thr = f * (ri + kCovalent[Z[j]]);
      double d2 = 0;
      for (int c = 0; c < 3; c++) {
        const double d = double(x[3 * i + c]) - double(x[3 * j + c]);
        d2 += d * d;
      }
      if (std::sqrt(d2) < thr) {
        const int a = find(parent, i), b = find(parent, j);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
      }
    }
  }
  std::vector<int32_t> lab(n), root_label(n, -1);
  int U = 0;
  for (int i = 0; i < n; i++) { // numbered by first atom
    const int r = find(parent, i);
    if (root_label[r] < 0) root_label[r] = U++;
    lab[i] = root_label[r];
  }
  if (n_units) *n_units = U;
  return lab;
}

void frag_unit_charges(int n, const std::vector<int32_t> &units, const float *formal, std::vector<int32_t> &unit_charge,
                       std::vector<double> &q) {
  const int U = n ? *std::max_element(units.begin(), units.end()) + 1 : 0;
  std::vector<double> Qs(U, 0.0);
  std::vector<int> nu(U, 0);
  for (int i = 0; i < n; i++) Qs[units[i]] += formal[i], nu[units[i]]++;
  std::vector<int32_t> Qu(U);
  for (int u = 0; u < U; u++) {
    const double r = std::round(Qs[u]);
    if (std::abs(Qs[u] - r) > 1e-3)
      throw std::runtime_error("frag_charges: the formal charges of a bonded unit sum to " + std::to_string(Qs[u]) +
                               ", not an integer");
    Qu[u] = static_cast<int32_t>(r);
  }
  unit_charge.resize(n), q.resize(n);
  for (int i = 0; i < n; i++) unit_charge[i] = Qu[units[i]], q[i] = double(Qu[units[i]]) / nu[units[i]];
}

std::vector<double> frag_atom_charges(int n, const int32_t *units, const int32_t *unit_charge) {
  int U = 0;
  for (int i = 0; i < n; i++) U = std::max(U, units[i] + 1);
  std::vector<int> nu(U, 0);
  for (int i = 0; i < n; i++) nu[units[i]]++;
  std::vector<double> q(n);
  for (int i = 0; i < n; i++) q[i] = double(unit_charge[i]) / nu[units[i]];
  return q;
}

double frag_coulomb(const FragChargesSpec &s, int n, const float *x, const double *q, double *F) {
  if (F) std::fill(F, F + 3 * size_t(n), 0.0);
  const double r0 = s.R - s.t, a = 1.0 / (std::sqrt(2.0) * s.sigma), c = 2.0 * a / std::sqrt(M_PI);
  const double pre = KE * s.scale; // 1/2 sum_{i != j} = sum_{i < j}
  double E = 0;
  for (int i = 0; i < n; i++) {
    if (q[i] == 0) continue;
    for (int j = i + 1; j < n; j++) {
      if (q[j] == 0) continue;
      double d[3], r2 = 0;
      for (int k = 0; k < 3; k++) d[k] = double(x[3 * i + k]) - double(x[3 * j + k]), r2 += d[k] * d[k];
      const double r = std::sqrt(r2);
      if (!(r > r0)) continue; // T = 0 and T' = 0 below R - t
      const double u = (r - r0) / s.t;
      double T = 1, dT = 0;
      if (u < 1) T = u * u * u * (u * (6 * u - 15) + 10), dT = 30 * u * u * (u - 1) * (u - 1) / s.t;
      const double er = std::erf(a * r), k = er / r;
      const double dk = (c * std::exp(-a * a * r2) - k) / r; // d/dr erf(a r) / r
      const double qq = pre * q[i] * q[j];
      E += qq * k * T;
      if (F) {
        const double g = qq * (dk * T + k * dT) / r; // dE/dr / r
        for (int m = 0; m < 3; m++) F[3 * i + m] -= g * d[m], F[3 * j + m] += g * d[m];
      }
    }
  }
  return E;
}

void install_frag_cond_gate(GraphInterpreter &interp) {
  interp.post_node = [&interp](ggml_context *ctx, const GIRNode &node, ggml_tensor *out) -> ggml_tensor * {
    if (node.op != "LINEAR") return out;
    bool proj = false;
    for (const auto &r : node.inputs) proj |= r == "weight:system_conditioning_project_2_weight";
    if (!proj) return out;
    ggml_tensor *g = interp.find_weight("plato_frag_cond_gate");
    if (!g) throw std::runtime_error("frag_cond_gate: the GGUF has no plato_frag_cond_gate tensor");
    if (ggml_nelements(g) != out->ne[0]) throw std::runtime_error("frag_cond_gate: gate size != conditioning width");
    ggml_tensor *cu = interp.rebuild_substituted(ctx, node.id, "input:charge_index", "input:charge_index_unit");
    return ggml_add(ctx, out, ggml_mul(ctx, ggml_sub(ctx, cu, out), ggml_reshape_1d(ctx, g, out->ne[0])));
  };
}

void install_lrc_blend(GraphInterpreter &interp, ggml_tensor *blend) {
  interp.post_node = [&interp, blend](ggml_context *ctx, const GIRNode &node, ggml_tensor *out) -> ggml_tensor * {
    if (node.op != "LINEAR") return out;
    bool proj = false;
    for (const auto &r : node.inputs) proj |= r == "weight:system_conditioning_project_2_weight";
    if (!proj) return out;
    if (ggml_nelements(blend) != out->ne[1]) throw std::runtime_error("lrc blend: atom count != conditioning rows");
    ggml_tensor *cu = interp.rebuild_substituted(ctx, node.id, "input:charge_index", "input:charge_index_unit");
    return ggml_add(ctx, out, ggml_mul(ctx, ggml_sub(ctx, cu, out), ggml_reshape_2d(ctx, blend, 1, out->ne[1])));
  };
}

void install_lrc_readout(GraphInterpreter &interp, ggml_tensor *q) {
  const GIRGraph &G = interp.graph();
  auto node_id = [](const std::string &ref) { return ref.rfind("node:", 0) == 0 ? std::stoi(ref.substr(5)) : -1; };
  // the per-atom energy: the output node, or its first input when the output concatenates [E, non-conservative F]
  if (G.outputs.empty()) throw std::runtime_error("lrc readout: the graph has no output");
  int e_id = node_id(G.outputs[0].node_ref);
  if (const GIRNode *o = G.get_node(e_id); o && o->op == "CONCAT") e_id = node_id(o->inputs[0]);
  // h: the input of the last node-energy head's first LINEAR (node_energy_heads_K_0_weight, largest K)
  int h_id = -1, best = -1;
  for (const auto &nd : G.nodes) {
    if (nd.op != "LINEAR") continue;
    for (const auto &r : nd.inputs) {
      const std::string pre = "weight:node_energy_heads_";
      if (r.rfind(pre, 0) != 0 || r.size() < pre.size() + 9 || r.substr(r.size() - 9) != "_0_weight") continue;
      const int k = std::stoi(r.substr(pre.size()));
      if (k > best) best = k, h_id = node_id(nd.inputs[0]);
    }
  }
  if (e_id < 0 || h_id < 0) throw std::runtime_error("lrc readout: cannot find the per-atom energy / node features");
  auto prev = interp.post_node;
  interp.post_node = [&interp, q, e_id, h_id, prev](ggml_context *ctx, const GIRNode &node, ggml_tensor *out) {
    if (prev) out = prev(ctx, node, out);
    if (node.id != e_id) return out;
    ggml_tensor *W = interp.find_weight("lrc_readout_weight"), *beta = interp.find_weight("lrc_readout_bias");
    if (!W || !beta) throw std::runtime_error("lrc readout: the GGUF has no lrc_readout_weight / lrc_readout_bias");
    ggml_tensor *h = interp.node_outputs().at(h_id);
    const int64_t n = ggml_nelements(q);
    if (!ggml_is_contiguous(h)) h = ggml_cont(ctx, h);
    h = ggml_reshape_2d(ctx, h, h->ne[0], ggml_nelements(h) / h->ne[0]);
    if (h->ne[1] != n || W->ne[0] != h->ne[0] || W->ne[1] != 2 || ggml_nelements(beta) != 2)
      throw std::runtime_error("lrc readout: shapes (W [2, d], beta [2], h [n, d], q [n]) do not match");
    ggml_tensor *ab = ggml_mul_mat(ctx, W, h); // [2, n]
    ab = ggml_add(ctx, ab, ggml_repeat(ctx, ggml_reshape_2d(ctx, beta, 2, 1), ab));
    ggml_tensor *a = ggml_cont(ctx, ggml_view_2d(ctx, ab, 1, n, ab->nb[1], 0));
    ggml_tensor *b = ggml_cont(ctx, ggml_view_2d(ctx, ab, 1, n, ab->nb[1], sizeof(float)));
    // softplus(b) = relu(b) + log(1 + exp(-|b|)): finite for any b (-log sigmoid(-b) overflows for b > ~88)
    ggml_tensor *sp = ggml_add(ctx, ggml_relu(ctx, b),
                               ggml_log(ctx, ggml_scale_bias(ctx, ggml_exp(ctx, ggml_neg(ctx, ggml_abs(ctx, b))), 1.0f, 1.0f)));
    ggml_tensor *qq = ggml_reshape_2d(ctx, q, 1, n);
    ggml_tensor *e = ggml_add(ctx, ggml_mul(ctx, a, qq), ggml_scale(ctx, ggml_mul(ctx, sp, ggml_sqr(ctx, qq)), 0.5f));
    if (ggml_nelements(out) != n) throw std::runtime_error("lrc readout: per-atom energy of the wrong size");
    return ggml_add(ctx, out, ggml_reshape(ctx, e, out));
  };
}

} // namespace mlipcpp::runtime
