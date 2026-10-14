// Op-level backward check: L = sum(r * (rms_norm(x) * w)) + sum(s * x), the
// shape of a PET pre-norm residual block, on every available backend,
// against the analytic gradients in double precision.
#include "core/backend.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <fmt/core.h>

#include <cmath>
#include <random>
#include <vector>

using namespace mlipcpp;

static int check(BackendPreference pref, int64_t d, int64_t m, int64_t n) {
  auto bp = BackendProvider::create(pref);
  const float eps = 1e-8f;
  const int64_t N = d * m * n;
  std::mt19937 rng(1);
  std::normal_distribution<float> U(0, 1);
  std::vector<float> x(N), r(N), s(N), w(d);
  for (auto &v : x) v = U(rng);
  for (auto &v : r) v = U(rng);
  for (auto &v : s) v = U(rng);
  for (auto &v : w) v = 1 + 0.1f * U(rng);

  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, true), nullptr, true});
  auto mk = [&](int64_t a, int64_t b, int64_t c) { auto *t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c); ggml_set_input(t); return t; };
  ggml_tensor *tx = mk(d, m, n), *tr = mk(d, m, n), *ts = mk(d, m, n);
  ggml_tensor *tw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d);
  ggml_set_input(tw);
  ggml_set_param(tx), ggml_set_param(tw);
  ggml_tensor *y = ggml_mul(ctx, ggml_rms_norm(ctx, tx, eps), tw);
  ggml_tensor *loss = ggml_add(ctx, ggml_sum(ctx, ggml_mul(ctx, y, tr)), ggml_sum(ctx, ggml_mul(ctx, tx, ts)));
  ggml_set_loss(loss);
  ggml_cgraph *gf = ggml_new_graph_custom(ctx, 256, true);
  ggml_build_forward_expand(gf, loss);
  ggml_build_backward_expand(ctx, gf, nullptr);
  ggml_tensor *gx = ggml_graph_get_grad(gf, tx), *gw = ggml_graph_get_grad(gf, tw);
  for (ggml_tensor *g : {gx, gw})
    for (ggml_tensor *v = g; v; v = v->view_src) ggml_set_output(v);
  ggml_tensor *lg = ggml_graph_get_grad_acc(gf, loss);
  ggml_set_input(lg);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bp->primary()));
  ggml_gallocr_alloc_graph(ga, gf);
  ggml_backend_tensor_set(tx, x.data(), 0, N * 4), ggml_backend_tensor_set(tr, r.data(), 0, N * 4);
  ggml_backend_tensor_set(ts, s.data(), 0, N * 4), ggml_backend_tensor_set(tw, w.data(), 0, d * 4);
  const float one = 1;
  ggml_backend_tensor_set(lg, &one, 0, 4);
  ggml_backend_graph_compute(bp->primary(), gf);
  std::vector<float> cx(N), cw(d);
  ggml_backend_tensor_get(gx, cx.data(), 0, N * 4), ggml_backend_tensor_get(gw, cw.data(), 0, d * 4);

  // analytic, double
  std::vector<double> ex(N), ew(d, 0.0);
  for (int64_t row = 0; row < m * n; row++) {
    const float *xr = &x[row * d], *rr = &r[row * d];
    double ss = 0, sxg = 0;
    for (int64_t k = 0; k < d; k++) ss += double(xr[k]) * xr[k];
    const double mean_eps = ss / d + eps, rrms = 1 / std::sqrt(mean_eps);
    for (int64_t k = 0; k < d; k++) sxg += double(xr[k]) * rr[k] * w[k];
    for (int64_t k = 0; k < d; k++) {
      ex[row * d + k] = rrms * (rr[k] * w[k] - xr[k] * sxg / (d * mean_eps)) + s[row * d + k];
      ew[k] += rr[k] * xr[k] * rrms;
    }
  }
  auto rel = [](auto &c, auto &e) {
    double a = 0, b = 0;
    for (size_t i = 0; i < e.size(); i++) a += std::pow(c[i] - e[i], 2), b += e[i] * e[i];
    return std::sqrt(a / b);
  };
  const double rx = rel(cx, ex), rw = rel(cw, ew);
  fmt::print("{:<8} x[{},{},{}]: rel err dL/dx {:.2e}, dL/dw {:.2e}  {}\n", bp->name(), n, m, d, rx, rw,
             rx < 1e-4 && rw < 1e-4 ? "ok" : "WRONG");
  ggml_gallocr_free(ga);
  ggml_free(ctx);
  return rx < 1e-4 && rw < 1e-4 ? 0 : 1;
}

int main() {
  int bad = 0;
  for (auto pref : {BackendPreference::CPU, BackendPreference::Auto})
    for (auto [d, m, n] : {std::tuple<int64_t, int64_t, int64_t>{128, 178, 47}, {256, 1, 47}, {128, 7, 3}})
      bad += check(pref, d, m, n);
  return bad ? 1 : 0;
}
