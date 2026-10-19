// Forward-mode check (runtime/graph_jvp.h) on the real PET graph: one fixed,
// unrotated batch (structures 0..n-1 of <data>/train), E = sum of the atomic
// energies, random tangents dv of the edge vectors and dc of the cutoff values
// (real edges only), a random weight direction p.
//
//   1. JVP(E; dv, dc) against <dE/dv, dv> + <dE/dc, dc> from the backward pass
//      (an identity: only rounding separates them);
//   2. the weight gradient of JVP(E), i.e. reverse mode through the tangent
//      graph (the mixed second derivative conservative training needs), along
//      p, against central differences of JVP(E) at w +- h p.
//
// Also prints the forward graph's ops, so a missing tangent rule shows up as
// the op that throws.
//
//   mlipcpp-jvp-check --data DIR --init INIT.gguf [--n 4] [--gpus 1] [--cache-dir DIR] [--seed 1]
//
// --attn-test 1 (no model): the fused tangent of the PET attention
// (ggml_pet_attn_jvp) against the unfused expression, on random inputs with
// masked (-inf) biases: tangent values and the gradients of a random linear
// function of them w.r.t. qkv, bias and both tangents, on CPU and the GPU;
// then wall time and compute buffer of both at a training-sized shape.
#include "core/backend.h"
#include "runtime/graph_jvp.h"
#include "train/crash.h"
#include "train/trainer.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <fmt/core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using namespace mlipcpp;
using namespace mlipcpp::train;

namespace {

constexpr size_t GRAPH_SIZE = 1 << 18;

std::vector<float> get(const ggml_tensor *t) {
  std::vector<float> v(ggml_nelements(t));
  ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
  return v;
}

double dot(const std::vector<float> &a, const std::vector<float> &b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); i++) s += double(a[i]) * b[i];
  return s;
}

/// One graph over the batch: E, its tangent J along (dv, dc), and either the
/// gradients of E w.r.t. the edge vectors and cutoff values (mixed = false) or
/// the weight gradients of J (mixed = true).
struct Probe {
  ggml_context *ctx = nullptr;
  ggml_cgraph *gf = nullptr;
  ggml_gallocr_t galloc = nullptr;
  std::map<std::string, ggml_tensor *> in;
  ggml_tensor *E = nullptr, *J = nullptr, *gv = nullptr, *gc = nullptr;
  std::vector<std::pair<ggml_tensor *, float>> constants;
  std::map<std::string, int> ops; // forward ops by name
  int n_forward = 0;
  size_t buffer = 0;

  std::map<std::string, int> missing; // ops without a tangent rule

  // mixed: weight gradients of J, or with energy_fd of E (a control: the
  // first derivative of the plain forward pass)
  Probe(Worker &w, const Batch &b, bool mixed, bool energy_fd = false) {
    auto &model = w.model();
    auto &interp = model.interpreter();
    const int64_t n = b.n_atoms, m = b.max_neighbors;
    interp.set_dimension("n_atoms", n);
    interp.set_dimension("max_neighbors", m);
    interp.set_dimension("n_edges", n * m);
    interp.set_dimension("max_neighbors_plus_one", m + 1);
    ctx = ggml_init({ggml_tensor_overhead() * GRAPH_SIZE * 2 + ggml_graph_overhead_custom(GRAPH_SIZE, true),
                     nullptr, true});
    auto input = [&](ggml_type type, std::initializer_list<int64_t> ne, const char *name) {
      ggml_tensor *t = ggml_new_tensor(ctx, type, int(ne.size()), std::data(ne));
      ggml_set_name(t, name);
      ggml_set_input(t);
      return in[name] = t;
    };
    interp.set_input("species", input(GGML_TYPE_I32, {n}, "species"));
    interp.set_input("neighbor_species", input(GGML_TYPE_I32, {m, n}, "neighbor_species"));
    ggml_tensor *edges = input(GGML_TYPE_F32, {3, m, n}, "edge_vectors");
    interp.set_input("edge_vectors", edges);
    interp.set_input("padding_mask", input(GGML_TYPE_F32, {m, n}, "padding_mask"));
    interp.set_input("reverse_neighbor_index", input(GGML_TYPE_I32, {n * m}, "reverse_neighbor_index"));
    ggml_tensor *cv = input(GGML_TYPE_F32, {m, n}, "cutoff_values");
    interp.set_input("cutoff_values", cv);
    ggml_tensor *dv = input(GGML_TYPE_F32, {3, m, n}, "dv");
    ggml_tensor *dc = input(GGML_TYPE_F32, {m, n}, "dc");

    ggml_tensor *out = interp.build(ctx); // (energy | force) per atom: ne = (4, n)
    E = ggml_sum(ctx, ggml_cont(ctx, ggml_view_2d(ctx, out, 1, n, out->nb[1], 0)));
    runtime::GraphJVP jvp(ctx);
    jvp.missing = &missing;
    jvp.seed(edges, dv);
    jvp.seed(cv, dc);
    J = jvp(E);
    if (!missing.empty()) return;
    if (!J) throw std::runtime_error("the energy has no tangent");
    ggml_set_output(E);
    ggml_set_output(J);

    // parameter flags select what the backward pass differentiates
    for (ggml_tensor *p : w.params()) {
      if (mixed) ggml_set_param(p);
      else p->flags &= ~GGML_TENSOR_FLAG_PARAM;
    }
    if (!mixed) ggml_set_param(edges), ggml_set_param(cv);
    gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, true);
    ggml_build_forward_expand(gf, E);
    ggml_build_forward_expand(gf, J);
    n_forward = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_forward; i++) ops[ggml_op_desc(ggml_graph_node(gf, i))]++;
    ggml_tensor *loss = mixed && !energy_fd ? J : E;
    ggml_set_loss(loss);
    ggml_build_backward_expand(ctx, gf, nullptr);
    for (ggml_tensor *p : w.params()) ggml_set_param(p); // the Worker's state
    ggml_tensor *seed = ggml_graph_get_grad_acc(gf, loss);
    ggml_set_input(seed);
    if (mixed) {
      for (ggml_tensor *p : w.params())
        if (ggml_tensor *g = ggml_graph_get_grad(gf, p)) ggml_set_output(g);
    } else {
      gv = ggml_graph_get_grad(gf, edges), gc = ggml_graph_get_grad(gf, cv);
      if (!gv) throw std::runtime_error("no edge-vector gradient");
      ggml_set_output(gv);
      if (gc) ggml_set_output(gc);
    }
    constants = interp.take_constants();
    for (auto &c : constants) ggml_set_input(c.first);
    constants.emplace_back(seed, 1.0f);

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend()));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) throw std::runtime_error("cannot allocate the graph");
    buffer = ggml_gallocr_get_buffer_size(galloc, 0);
  }
  ~Probe() {
    if (galloc) ggml_gallocr_free(galloc);
    if (ctx) ggml_free(ctx);
  }

  void upload(const Batch &b, const std::vector<float> &dv, const std::vector<float> &dc) {
    auto set = [&](const char *name, const void *data, size_t bytes) {
      ggml_tensor *t = in.at(name);
      if (ggml_nbytes(t) != bytes) throw std::logic_error(std::string("size of ") + name);
      ggml_backend_tensor_set(t, data, 0, bytes);
    };
    set("species", b.species.data(), b.species.size() * 4);
    set("neighbor_species", b.neighbor_species.data(), b.neighbor_species.size() * 4);
    set("edge_vectors", b.edge_vectors.data(), b.edge_vectors.size() * 4);
    set("padding_mask", b.padding_mask.data(), b.padding_mask.size() * 4);
    set("reverse_neighbor_index", b.reverse_neighbor_index.data(), b.reverse_neighbor_index.size() * 4);
    set("cutoff_values", b.cutoff_values.data(), b.cutoff_values.size() * 4);
    set("dv", dv.data(), dv.size() * 4);
    set("dc", dc.data(), dc.size() * 4);
  }

  // inputs every time: the allocator may compute a single-use input's
  // consumer in place
  void compute(ggml_backend_t backend, const Batch &b, const std::vector<float> &dv, const std::vector<float> &dc) {
    upload(b, dv, dc);
    std::vector<float> fill;
    for (auto &[t, v] : constants) {
      if (!t->buffer) continue;
      fill.assign(ggml_nelements(t), v);
      ggml_backend_tensor_set(t, fill.data(), 0, ggml_nbytes(t));
    }
    // no accumulators: without grad_accs only the loss seed is one
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
      throw std::runtime_error("graph compute failed");
  }
};

/// The tangent of ggml_pet_attn along (dqkv, dbias) and the gradients of
/// <R, tangent> w.r.t. qkv, bias, dqkv, dbias, fused or not.
struct AttnProbe {
  std::vector<float> t, g[4];
  double ms = 0;
  size_t buffer = 0;
};

AttnProbe attn_probe(ggml_backend_t backend, bool fused, int H, int d, int L, int N, const std::vector<float> (&in)[4],
                     const std::vector<float> &R, int reps) {
  const int64_t D = int64_t(H) * d;
  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, true), nullptr, true});
  const int64_t ne[4][4] = {{D, 3, L, N}, {L, L, 1, N}, {D, 3, L, N}, {L, L, 1, N}};
  ggml_tensor *x[4];
  for (int k = 0; k < 4; k++) {
    x[k] = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne[k]);
    ggml_set_input(x[k]);
    ggml_set_param(x[k]);
  }
  ggml_tensor *r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D + H, L, N);
  ggml_set_input(r);
  ggml_tensor *o = ggml_pet_attn(ctx, x[0], x[1], H, 1.0f / std::sqrt(float(d)));
  runtime::GraphJVP jvp(ctx);
  jvp.fused_attention = fused;
  jvp.seed(x[0], x[2]);
  jvp.seed(x[1], x[3]);
  ggml_tensor *t = jvp(o);
  ggml_set_output(t);
  ggml_tensor *loss = ggml_sum(ctx, ggml_mul(ctx, t, r));
  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 4096, true);
  ggml_build_forward_expand(gf, loss);
  ggml_set_loss(loss);
  ggml_build_backward_expand(ctx, gf, nullptr);
  ggml_tensor *seed = ggml_graph_get_grad_acc(gf, loss), *g[4];
  ggml_set_input(seed);
  for (int k = 0; k < 4; k++) ggml_set_output(g[k] = ggml_graph_get_grad(gf, x[k]));
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  if (!ggml_gallocr_alloc_graph(ga, gf)) throw std::runtime_error("attn test: cannot allocate");
  AttnProbe p;
  p.buffer = ggml_gallocr_get_buffer_size(ga, 0);
  const float one = 1;
  for (int rep = 0; rep <= reps; rep++) { // rep 0 warms up
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < 4; k++) ggml_backend_tensor_set(x[k], in[k].data(), 0, ggml_nbytes(x[k]));
    ggml_backend_tensor_set(r, R.data(), 0, ggml_nbytes(r));
    ggml_backend_tensor_set(seed, &one, 0, sizeof one);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("attn test: compute");
    ggml_backend_synchronize(backend);
    if (rep) p.ms += 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
  }
  p.t = get(t);
  for (int k = 0; k < 4; k++) p.g[k] = get(g[k]);
  ggml_gallocr_free(ga);
  ggml_free(ctx);
  return p;
}

int attn_test(int gpus) {
  std::mt19937_64 rng(3);
  std::normal_distribution<float> N01(0, 1);
  auto inputs = [&](int H, int d, int L, int N, std::vector<float> (&in)[4], std::vector<float> &R) {
    const size_t nq = size_t(H) * d * 3 * L * N, nb = size_t(L) * L * N;
    for (int k = 0; k < 4; k++) in[k].resize(k % 2 ? nb : nq);
    for (auto &v : in[0]) v = 0.7f * N01(rng);
    for (auto &v : in[2]) v = N01(rng);
    for (size_t e = 0; e < nb; e++) {
      const bool masked = e % L != 0 && N01(rng) > 1.2f; // never the first key (the centre token)
      in[1][e] = masked ? -INFINITY : 0.5f * N01(rng);
      in[3][e] = masked ? 0.0f : N01(rng);
    }
    R.resize(size_t(H) * (d + 1) * L * N);
    for (auto &v : R) v = N01(rng);
  };
  auto rel = [](const std::vector<float> &a, const std::vector<float> &b) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); i++) e += std::pow(double(a[i]) - b[i], 2), n += double(b[i]) * b[i];
    return std::sqrt(e / std::max(n, 1e-300));
  };
  bool ok = true;
  std::vector<BackendPreference> prefs = {BackendPreference::CPU};
  if (gpus > 0) prefs.push_back(BackendPreference::Auto);
  for (auto pref : prefs) {
    auto bp = BackendProvider::create(pref);
    for (auto [H, d, L, N] : {std::array<int, 4>{4, 16, 13, 5}, {2, 8, 40, 3}, {16, 16, 34, 7}}) {
      std::vector<float> in[4], R;
      inputs(H, d, L, N, in, R);
      const AttnProbe u = attn_probe(bp->primary(), false, H, d, L, N, in, R, 0),
                      f = attn_probe(bp->primary(), true, H, d, L, N, in, R, 0);
      const double e[5] = {rel(f.t, u.t), rel(f.g[0], u.g[0]), rel(f.g[1], u.g[1]), rel(f.g[2], u.g[2]),
                           rel(f.g[3], u.g[3])};
      const bool good = *std::max_element(e, e + 5) < 1e-4;
      ok = ok && good;
      fmt::print("{:<6} H {} d {} L {} N {}: fused vs unfused rel: tangent {:.1e}, d/dqkv {:.1e}, d/dbias {:.1e}, "
                 "d/d(dqkv) {:.1e}, d/d(dbias) {:.1e}  {}\n",
                 bp->name(), H, d, L, N, e[0], e[1], e[2], e[3], e[4], good ? "ok" : "WRONG");
    }
    if (pref != BackendPreference::CPU) { // training-sized: the big model, adaptive cutoff
      const int H = 16, d = 16, L = 34, N = 4096;
      std::vector<float> in[4], R;
      inputs(H, d, L, N, in, R);
      for (bool fused : {false, true}) {
        const AttnProbe p = attn_probe(bp->primary(), fused, H, d, L, N, in, R, 5);
        fmt::print("{:<6} H {} d {} L {} N {}: {} tangent + backward {:.2f} ms, compute buffer {:.0f} MiB\n", bp->name(),
                   H, d, L, N, fused ? "fused  " : "unfused", p.ms, p.buffer / 1048576.0);
      }
    }
  }
  fmt::print("{}\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

} // namespace

int run(int argc, char **argv) {
  std::string data, init, cache_dir;
  int n = 4, gpus = 1;
  uint64_t seed = 1;
  double tol_identity = 1e-3, tol_fd = 2e-2;
  bool attn = false, per_tensor = false, energy_fd = false;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--data") data = v;
    else if (a == "--init") init = v;
    else if (a == "--cache-dir") cache_dir = v;
    else if (a == "--n") n = std::stoi(v);
    else if (a == "--gpus") gpus = std::stoi(v);
    else if (a == "--seed") seed = std::stoull(v);
    else if (a == "--tol-identity") tol_identity = std::stod(v);
    else if (a == "--tol-fd") tol_fd = std::stod(v);
    else if (a == "--attn-test") attn = v == "1";
    else if (a == "--per-tensor") per_tensor = v == "1"; // central differences tensor by tensor
    else if (a == "--fd-of") energy_fd = v == "energy";    // test 2 on E instead of its tangent J
  }
  if (attn) return attn_test(gpus);
  if (data.empty() || init.empty()) {
    fmt::print(stderr, "usage: {} --data DIR --init INIT.gguf [--n 4] [--gpus 1] [--cache-dir DIR] [--seed 1]\n"
                       "       {} --attn-test 1 [--gpus 1]\n",
               argv[0], argv[0]);
    return 2;
  }
  MemmapSplit split(data + "/train");
  Worker worker(init, gpus > 0 ? 0 : -1);
  auto &model = worker.model();
  const TargetScaling sc = TargetScaling::fit(split);
  const AdaptiveCutoff adaptive = adaptive_cutoff(model);
  NeighborCache nl = NeighborCache::load_or_build(split, model.cutoff(),
                                                  cache_path(cache_dir, split, model.cutoff(), adaptive), 8, adaptive);
  std::vector<int64_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  Batch b = make_batch(split, nl, ids, {}, model.species_to_index(), sc, model.cutoff());
  fmt::print("{} structures, {} atoms, {} neighbours max, adaptive cutoff {}, device {}\n", n, b.n_atoms,
             b.max_neighbors, adaptive.enabled() ? "on" : "off", gpus > 0 ? "GPU" : "CPU");

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> N01(0, 1);
  std::vector<float> dv(b.edge_vectors.size(), 0.0f), dc(b.cutoff_values.size(), 0.0f);
  for (size_t f = 0; f < dc.size(); f++) {
    if (b.padding_mask[f] > 0.5f) continue;
    for (int k = 0; k < 3; k++) dv[3 * f + k] = N01(rng);
    if (adaptive.enabled()) dc[f] = 0.1f * N01(rng);
  }

  bool ok = true;
  // 1. forward vs reverse mode
  {
    auto t0 = std::chrono::steady_clock::now();
    Probe p(worker, b, false);
    if (!p.missing.empty()) {
      for (auto &[what, count] : p.missing) fmt::print("no tangent rule: {} (x{})\n", what, count);
      fmt::print("FAIL\n");
      return 1;
    }
    p.compute(model.backend(), b, dv, dc);
    const double ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string ops;
    for (auto &[op, count] : p.ops) ops += fmt::format(" {}:{}", op, count);
    fmt::print("forward + tangent graph: {} nodes;{}\n", p.n_forward, ops);
    const double J = get(p.J)[0];
    double R = dot(get(p.gv), dv);
    if (p.gc) R += dot(get(p.gc), dc);
    const double rel = std::abs(J - R) / std::max(std::abs(R), 1e-30);
    fmt::print("1. E {:.8g}; JVP {:.8g} vs reverse {:.8g}: rel {:.2e}  ({:.0f} ms, buffer {:.0f} MiB)\n",
               get(p.E)[0], J, R, rel, ms, p.buffer / 1048576.0);
    if (!(rel < tol_identity)) ok = false;
  }

  // 2. reverse through the tangent graph vs central differences over the weights
  {
    auto t0 = std::chrono::steady_clock::now();
    Probe p(worker, b, true, energy_fd);
    if (energy_fd) p.J = p.E; // the differenced quantity
    p.compute(model.backend(), b, dv, dc);
    const double ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // direction: per tensor, N(0, 1) scaled by the tensor's RMS
    std::vector<std::vector<float>> w0, dir;
    double analytic = 0;
    for (ggml_tensor *t : worker.params()) {
      auto w = get(t);
      double ms2 = 0;
      for (float x : w) ms2 += double(x) * x;
      const float s = float(std::sqrt(ms2 / w.size()) + 1e-3);
      std::vector<float> d(w.size());
      for (auto &x : d) x = s * N01(rng);
      if (ggml_tensor *g = ggml_graph_get_grad(p.gf, t)) analytic += dot(get(g), d);
      w0.push_back(std::move(w));
      dir.push_back(std::move(d));
    }
    const double J0 = get(p.J)[0];
    fmt::print("2. mixed graph: {} nodes, buffer {:.0f} MiB, {:.0f} ms; dJ/dw . p = {:.8g}\n",
               ggml_graph_n_nodes(p.gf), p.buffer / 1048576.0, ms, analytic);
    double best = INFINITY;
    for (double h : {1e-2, 3e-3, 1e-3}) {
      double Jpm[2];
      for (int s = 0; s < 2; s++) {
        size_t k = 0;
        for (ggml_tensor *t : worker.params()) {
          std::vector<float> w = w0[k];
          for (size_t i = 0; i < w.size(); i++) w[i] += float((s ? -h : h)) * dir[k][i];
          ggml_backend_tensor_set(t, w.data(), 0, ggml_nbytes(t));
          k++;
        }
        p.compute(model.backend(), b, dv, dc);
        Jpm[s] = get(p.J)[0];
      }
      const double fd = (Jpm[0] - Jpm[1]) / (2 * h);
      const double rel = std::abs(fd - analytic) / std::max(std::abs(analytic), 1e-30);
      best = std::min(best, rel);
      fmt::print("   h {:.0e}: central difference {:.8g} (J0 {:.8g}), rel {:.2e}\n", h, fd, J0, rel);
    }
    size_t k = 0;
    for (ggml_tensor *t : worker.params()) ggml_backend_tensor_set(t, w0[k++].data(), 0, ggml_nbytes(t));
    if (!(best < tol_fd)) ok = false;
    if (per_tensor) {
      // one tensor along its direction at a time: which ones disagree
      std::vector<std::tuple<double, std::string, double, double>> rows;
      k = 0;
      for (ggml_tensor *t : worker.params()) {
        const ggml_tensor *g = ggml_graph_get_grad(p.gf, t);
        const double an = g ? dot(get(g), dir[k]) : 0.0;
        double fd[2];
        for (int hi = 0; hi < 2; hi++) {
          const double h = hi ? 1e-3 : 1e-2;
          double Jpm[2];
          for (int sg = 0; sg < 2; sg++) {
            std::vector<float> w = w0[k];
            for (size_t i = 0; i < w.size(); i++) w[i] += float(sg ? -h : h) * dir[k][i];
            ggml_backend_tensor_set(t, w.data(), 0, ggml_nbytes(t));
            p.compute(model.backend(), b, dv, dc);
            Jpm[sg] = get(p.J)[0];
          }
          fd[hi] = (Jpm[0] - Jpm[1]) / (2 * h);
        }
        ggml_backend_tensor_set(t, w0[k].data(), 0, ggml_nbytes(t));
        const double rel = std::min(std::abs(fd[0] - an), std::abs(fd[1] - an)) / std::max(std::abs(an), 1e-6);
        rows.emplace_back(rel, ggml_get_name(t), an, fd[1]);
        k++;
      }
      std::sort(rows.rbegin(), rows.rend());
      fmt::print("   per tensor (worst first): rel, name, analytic, central difference h 1e-3\n");
      for (size_t r = 0; r < std::min<size_t>(rows.size(), 15); r++)
        fmt::print("   {:.2e}  {:<48} {:>14.6g} {:>14.6g}\n", std::get<0>(rows[r]), std::get<1>(rows[r]),
                   std::get<2>(rows[r]), std::get<3>(rows[r]));
    }
  }
  fmt::print("{}\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  install_crash_backtrace();
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    fmt::print("error: {}\nFAIL\n", e.what());
    return 1;
  }
}
