// Weight gradient of a linear layer, dW = x g^T (x [in, R], g [out, R], dW
// [in, out] in ggml order), on PET shapes: ggml's MUL_MAT backward
// (mul_mat(cont(x^T), cont(g^T)): two transposed copies of R-row tensors)
// against out_prod(x, g) (one GEMM with a transposed operand, no copies) and
// the bare GEMM on pre-transposed inputs (the copies' share). Times per call.
#include "core/backend.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <tuple>
#include <vector>

using namespace mlipcpp;

namespace {

enum Variant { MULMAT_CONT, OUT_PROD, GEMM_ONLY, NV };
const char *NAMES[NV] = {"mul_mat(cont,cont)", "out_prod", "gemm only"};

struct Result {
  std::vector<float> dw;
  double ms = 0;
};

Result run(BackendProvider &bp, Variant v, int64_t in, int64_t out, int64_t R, const std::vector<float> &x,
           const std::vector<float> &g, int reps) {
  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true});
  ggml_tensor *X, *G, *dw;
  if (v == GEMM_ONLY) { // inputs already transposed: [R, in], [R, out]
    X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R, in), G = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R, out);
    dw = ggml_mul_mat(ctx, X, G);
  } else {
    X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, R), G = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out, R);
    dw = v == OUT_PROD ? ggml_out_prod(ctx, X, G)
                       : ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, X)), ggml_cont(ctx, ggml_transpose(ctx, G)));
  }
  ggml_set_input(X), ggml_set_input(G), ggml_set_output(dw);
  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 16, false);
  ggml_build_forward_expand(gf, dw);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bp.primary()));
  ggml_gallocr_alloc_graph(ga, gf);
  if (v == GEMM_ONLY) { // transpose on the host
    std::vector<float> xt(x.size()), gt(g.size());
    for (int64_t r = 0; r < R; r++) {
      for (int64_t i = 0; i < in; i++) xt[i * R + r] = x[r * in + i];
      for (int64_t o = 0; o < out; o++) gt[o * R + r] = g[r * out + o];
    }
    ggml_backend_tensor_set(X, xt.data(), 0, ggml_nbytes(X)), ggml_backend_tensor_set(G, gt.data(), 0, ggml_nbytes(G));
  } else {
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X)), ggml_backend_tensor_set(G, g.data(), 0, ggml_nbytes(G));
  }
  Result res;
  ggml_backend_graph_compute(bp.primary(), gf); // warm-up
  ggml_backend_synchronize(bp.primary());
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; i++) ggml_backend_graph_compute(bp.primary(), gf);
  ggml_backend_synchronize(bp.primary());
  res.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
  res.dw.resize(ggml_nelements(dw));
  ggml_backend_tensor_get(dw, res.dw.data(), 0, ggml_nbytes(dw));
  ggml_gallocr_free(ga);
  ggml_free(ctx);
  return res;
}

using NsShape = std::tuple<int64_t, int64_t, int64_t>; // ne0, ne1, k

// ggml_newton_schulz_multi on stacks of k [ne0, ne1] matrices (one op for
// all of them): per stack the result, and ms per call
std::pair<std::vector<std::vector<float>>, double> newton_schulz(ggml_backend_t backend, const std::vector<NsShape> &shapes,
                                                                 const std::vector<std::vector<float>> &xs, int reps) {
  const size_t n = shapes.size();
  const size_t gs = n + 4; // graph: n leafs + the op
  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * gs + ggml_graph_overhead_custom(gs, false), nullptr, true});
  std::vector<ggml_tensor *> X;
  for (auto [ne0, ne1, k] : shapes) {
    X.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, k));
    ggml_set_input(X.back());
  }
  ggml_tensor *Y = ggml_newton_schulz_multi(ctx, X.data(), static_cast<int>(n), 5, 3.4445f, -4.7750f, 2.0315f);
  ggml_set_output(Y);
  ggml_cgraph *gf = ggml_new_graph_custom(ctx, gs, false);
  ggml_build_forward_expand(gf, Y);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  ggml_gallocr_alloc_graph(ga, gf);
  for (size_t i = 0; i < n; i++) ggml_backend_tensor_set(X[i], xs[i].data(), 0, ggml_nbytes(X[i]));
  ggml_backend_graph_compute(backend, gf);
  ggml_backend_synchronize(backend);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; i++) ggml_backend_graph_compute(backend, gf);
  ggml_backend_synchronize(backend);
  const double ms = reps ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps : 0;
  std::vector<std::vector<float>> ys;
  size_t off = 0;
  for (ggml_tensor *x : X) {
    ys.emplace_back(ggml_nelements(x));
    ggml_backend_tensor_get(Y, ys.back().data(), off, ggml_nbytes(x));
    off += ggml_nbytes(x);
  }
  ggml_gallocr_free(ga);
  ggml_free(ctx);
  return {ys, ms};
}

} // namespace

int main(int argc, char **argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
  setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1); // each run is a new graph, often at the last one's addresses
  auto bp = BackendProvider::create(BackendPreference::Auto);
  fmt::print("backend {}; ms per call ({} reps), (speed-up vs mul_mat(cont,cont), max relative error of 16 dW entries vs double)\n", bp->name(),
             reps);
  // d_pet 256 big model: 256 -> 256 / 512 / 768 / 1024 and back; R = edges
  // of the profiled micro-batches (4x682, 30x176, 15x461) and larger packs
  for (auto [in, out] : {std::pair<int64_t, int64_t>{256, 256}, {256, 768}, {256, 1024}, {1024, 256}, {512, 256}})
    for (int64_t R : {2728, 5280, 6915, 30000, 100000}) {
      std::mt19937 rng(in + out + R);
      std::normal_distribution<float> U(0, 1);
      std::vector<float> x(in * R), g(out * R);
      for (auto &e : x) e = U(rng);
      for (auto &e : g) e = U(rng);
      fmt::print("in {:4} out {:4} R {:6}", in, out, R);
      // 16 entries of dW in double: relative error (NaN propagates)
      std::vector<std::pair<int64_t, double>> ref;
      for (int k = 0; k < 16; k++) {
        const int64_t i = rng() % in, o = rng() % out;
        double s = 0;
        for (int64_t r = 0; r < R; r++) s += double(x[r * in + i]) * g[r * out + o];
        ref.push_back({o * in + i, s});
      }
      auto rel = [&](const Result &r) {
        double e = 0, m = 0;
        for (auto [k, s] : ref) {
          if (std::isnan(r.dw[k])) return double(NAN);
          e = std::max(e, std::abs(r.dw[k] - s)), m = std::max(m, std::abs(s));
        }
        return e / m;
      };
      double base = 0;
      for (int v = 0; v < NV; v++) {
        const Result r = run(*bp, Variant(v), in, out, R, x, g, reps);
        if (v == 0) base = r.ms;
        fmt::print("  {} {:.3f} ({}{:.1e})", NAMES[v], r.ms, v ? fmt::format("x{:.2f}, ", base / r.ms) : "", rel(r));
      }
      fmt::print("\n");
    }

  // Newton-Schulz op: the backend's kernel (bf16 storage, or fp32/TF32 with
  // GGML_NS_F32=1; grouped GEMMs, or per stack with GGML_NS_LOOP=1) against the
  // CPU reference, per Muon group of the big model, then all groups in one op
  // (the optimizer's call) against the per-group calls
  ggml_backend_t cpu = ggml_backend_cpu_init();
  fmt::print("newton-schulz ({}, {}): ms per call, max |gpu - cpu| / max |cpu|\n",
             std::getenv("GGML_NS_F32") ? "fp32 storage" : "bf16 storage", std::getenv("GGML_NS_LOOP") ? "loop" : "grouped");
  const std::vector<NsShape> groups = {{256, 256, 11}, {256, 512, 4}, {256, 768, 4}, {256, 1024, 4},
                                       {512, 256, 12}, {512, 512, 1}, {512, 2048, 4}, {1024, 512, 4}};
  std::vector<std::vector<float>> xs, ys;
  double sum_ms = 0;
  for (auto [ne0, ne1, k] : groups) {
    std::mt19937 rng(ne0 * ne1 + k);
    std::normal_distribution<float> U(0, 1);
    xs.emplace_back(ne0 * ne1 * k);
    for (auto &e : xs.back()) e = U(rng);
    const auto [y, ms] = newton_schulz(bp->primary(), {{ne0, ne1, k}}, {xs.back()}, reps);
    const std::vector<float> x2(xs.back().begin(), xs.back().begin() + ne0 * ne1 * std::min<int64_t>(k, 2));
    const auto [yc, msc] = newton_schulz(cpu, {{ne0, ne1, std::min<int64_t>(k, 2)}}, {x2}, 0); // first two matrices
    double e = 0, m = 0;
    for (size_t i = 0; i < yc[0].size(); i++)
      e = std::max(e, double(std::abs(y[0][i] - yc[0][i]))), m = std::max(m, double(std::abs(yc[0][i])));
    fmt::print("ns {:4} x {:4} x {:2}  {:.3f} ms  {:.1e}\n", ne0, ne1, k, ms, e / m);
    ys.push_back(y[0]);
    sum_ms += ms;
  }
  const auto [ya, msa] = newton_schulz(bp->primary(), groups, xs, reps);
  double e = 0, m = 0;
  for (size_t g = 0; g < groups.size(); g++)
    for (size_t i = 0; i < ys[g].size(); i++)
      e = std::max(e, double(std::abs(ya[g][i] - ys[g][i]))), m = std::max(m, double(std::abs(ys[g][i])));
  fmt::print("ns all {} groups in one op  {:.3f} ms (sum of per-group calls {:.3f} ms)  vs per-group {:.1e}\n",
             groups.size(), msa, sum_ms, e / m);
  ggml_backend_free(cpu);
  return 0;
}
