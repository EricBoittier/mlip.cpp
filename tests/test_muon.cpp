// Muon in mlipcpp-train (src/train/trainer.cpp orthogonalize, src/train/lr_schedule.cpp GroupOptimizer):
// - Newton-Schulz output: for g = U diag(s) V^T it is U diag(phi(s / |s|)) V^T with the scalar quintic
//   phi (Muon's 3.4445, -4.7750, 2.0315, 5 iterations), i.e. orthogonal up to singular values in about
//   [0.7, 1.2]; scale invariant; shape preserved (wide, tall, square, stacked);
// - the hidden-matrix rule (2-D, both dims >= 16, no embeddings);
// - the grouped optimizer's Muon step: Nesterov momentum, RMS-matched step lr scale 0.2 sqrt(max dim),
//   decoupled weight decay x (1 - lr wd), AdamW for a Muon group's non-matrix tensors; bitwise determinism;
// - a small ill-conditioned matrix regression where Muon matches or beats AdamW (best lr of a grid each).
#include "train/lr_schedule.h"
#include "train/trainer.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

using mlipcpp::train::GroupOptimizer;
using mlipcpp::train::is_muon_matrix;
using mlipcpp::train::LrSchedule;
using mlipcpp::train::muon_orthogonalize;

namespace {
using Mat = std::vector<double>; // row-major rows x cols

// columns of a random rows x r matrix, Gram-Schmidt orthonormalised (r <= rows)
Mat orthonormal(int rows, int r, std::mt19937 &rng) {
  std::normal_distribution<double> n(0, 1);
  Mat q(size_t(rows) * r);
  for (auto &x : q) x = n(rng);
  for (int j = 0; j < r; j++) {
    for (int pass = 0; pass < 2; pass++)
      for (int i = 0; i < j; i++) {
        double d = 0;
        for (int k = 0; k < rows; k++) d += q[k * r + i] * q[k * r + j];
        for (int k = 0; k < rows; k++) q[k * r + j] -= d * q[k * r + i];
      }
    double nn = 0;
    for (int k = 0; k < rows; k++) nn += q[k * r + j] * q[k * r + j];
    for (int k = 0; k < rows; k++) q[k * r + j] /= std::sqrt(nn);
  }
  return q;
}

// phi: the scalar Newton-Schulz map of a singular value after Frobenius normalisation
double phi(double x, int steps) {
  for (int i = 0; i < steps; i++) x = 3.4445 * x - 4.7750 * x * x * x + 2.0315 * x * x * x * x * x;
  return x;
}

// muon_orthogonalize on the CPU of a stack of k matrices rows x cols (ggml ne0 = cols, ne1 = rows)
std::vector<float> ns_cpu(const std::vector<float> &g, int cols, int rows, int k, int steps, std::vector<int64_t> *shape = nullptr) {
  ggml_context *ctx = ggml_init({size_t(256) << 20, nullptr, false});
  ggml_tensor *t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cols, rows, k);
  std::copy(g.begin(), g.end(), static_cast<float *>(t->data));
  ggml_tensor *o = muon_orthogonalize(ctx, t, steps);
  ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, o);
  ggml_graph_compute_with_ctx(ctx, gf, 4);
  if (shape) *shape = {o->ne[0], o->ne[1], o->ne[2], o->ne[3]};
  std::vector<float> out(static_cast<float *>(o->data), static_cast<float *>(o->data) + ggml_nelements(o));
  ggml_free(ctx);
  return out;
}
} // namespace

TEST_CASE("muon: Newton-Schulz orthogonalises (U diag(phi(s)) V^T), any shape, scale invariant", "[muon]") {
  std::mt19937 rng(7);
  struct Case { int rows, cols, k; };
  for (Case c : {Case{32, 64, 1}, Case{64, 32, 1}, Case{48, 48, 1}, Case{24, 80, 3}}) {
    INFO("rows " << c.rows << " cols " << c.cols << " k " << c.k);
    const int r = std::min(c.rows, c.cols);
    std::vector<float> g(size_t(c.rows) * c.cols * c.k);
    std::vector<Mat> Us, Vs;
    std::vector<std::vector<double>> ss;
    for (int m = 0; m < c.k; m++) {
      Mat U = orthonormal(c.rows, r, rng), V = orthonormal(c.cols, r, rng);
      std::vector<double> s(r);
      for (int i = 0; i < r; i++) s[i] = 7.3 * std::pow(10.0, -2.0 * i / (r - 1)); // 7.3 .. 0.073, any scale
      for (int a = 0; a < c.rows; a++)
        for (int b = 0; b < c.cols; b++) {
          double v = 0;
          for (int i = 0; i < r; i++) v += U[a * r + i] * s[i] * V[b * r + i];
          g[size_t(m) * c.rows * c.cols + size_t(a) * c.cols + b] = float(v);
        }
      Us.push_back(U), Vs.push_back(V), ss.push_back(s);
    }
    std::vector<int64_t> shape;
    const std::vector<float> o = ns_cpu(g, c.cols, c.rows, c.k, 5, &shape);
    CHECK(shape == std::vector<int64_t>{c.cols, c.rows, c.k, 1});
    // scale invariance
    std::vector<float> g2(g);
    for (auto &x : g2) x *= 1e-3f;
    const std::vector<float> o2 = ns_cpu(g2, c.cols, c.rows, c.k, 5);
    double dmax = 0;
    for (size_t i = 0; i < o.size(); i++) dmax = std::max(dmax, double(std::fabs(o[i] - o2[i])));
    CHECK(dmax < 1e-4);
    for (int m = 0; m < c.k; m++) {
      const Mat &U = Us[m], &V = Vs[m];
      double fro = 0;
      for (double x : ss[m]) fro += x * x;
      fro = std::sqrt(fro);
      // M = U^T O V: diagonal phi(s_i / |s|), off-diagonal ~ 0
      double off = 0, diag_err = 0, smin = 1e9, smax = 0, polar = 0;
      for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++) {
          double v = 0;
          for (int a = 0; a < c.rows; a++)
            for (int b = 0; b < c.cols; b++)
              v += U[a * r + i] * o[size_t(m) * c.rows * c.cols + size_t(a) * c.cols + b] * V[b * r + j];
          if (i != j) off = std::max(off, std::fabs(v));
          else {
            diag_err = std::max(diag_err, std::fabs(v - phi(ss[m][i] / fro, 5)));
            smin = std::min(smin, v), smax = std::max(smax, v);
            polar += (v - 1) * (v - 1);
          }
        }
      CHECK(off < 2e-3);
      CHECK(diag_err < 2e-3);
      // Muon's quintic: every singular value pushed into about [0.7, 1.2] (none left near 0)
      CHECK(smin > 0.6);
      CHECK(smax < 1.25);
      // distance to the exact polar factor U V^T, relative
      CHECK(std::sqrt(polar / r) < 0.3);
    }
  }
}

TEST_CASE("muon: hidden-matrix rule", "[muon]") {
  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
  auto named = [](ggml_tensor *t, const char *n) { ggml_set_name(t, n); return t; };
  CHECK(is_muon_matrix(named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32), "gnn_layers.0.mlp.w")));
  CHECK(is_muon_matrix(named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 512), "gnn_layers.0.attn.w")));
  CHECK_FALSE(is_muon_matrix(named(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64), "gnn_layers.0.mlp.b")));     // bias
  CHECK_FALSE(is_muon_matrix(named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8), "heads.0.w")));            // 8 outputs
  CHECK_FALSE(is_muon_matrix(named(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64), "edge_embedder.w")));     // embedding
  CHECK_FALSE(is_muon_matrix(named(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 64, 2), "gnn_layers.0.w3")));  // 3-D
  ggml_free(ctx);
}

namespace {
struct Toy { // W: rows x cols (Muon matrix), b: cols (AdamW) in one Muon group
  ggml_backend_t be = nullptr;
  ggml_context *ctx = nullptr;
  ggml_backend_buffer_t buf = nullptr;
  ggml_tensor *W = nullptr, *gW = nullptr, *b = nullptr, *gb = nullptr;
  std::unique_ptr<GroupOptimizer> opt;
  Toy(int rows, int cols, const std::string &spec, double mu, double scale) {
    be = ggml_backend_cpu_init();
    ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows), gW = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cols), gb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cols);
    ggml_set_name(W, "layer.w"), ggml_set_name(b, "layer.b");
    ggml_set_param(W), ggml_set_param(b);
    buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    opt = std::make_unique<GroupOptimizer>(
        be, std::vector<GroupOptimizer::Param>{{W, gW, "layer.w", is_muon_matrix(W)}, {b, gb, "layer.b", is_muon_matrix(b)}},
        LrSchedule::parse(spec), /*clip*/ 1e9, mu, scale,
        [](ggml_context *c, ggml_tensor *g) { return muon_orthogonalize(c, g, 5); });
  }
  ~Toy() {
    opt.reset();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
  }
  static std::vector<float> get(ggml_tensor *t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
  }
  static void set(ggml_tensor *t, const std::vector<float> &v) { ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t)); }
};
} // namespace

TEST_CASE("muon: grouped optimizer step (Nesterov momentum, RMS-matched lr, decoupled decay, AdamW rest)", "[muon]") {
  const int rows = 32, cols = 48;
  const double lr = 0.01, wd = 0.1, mu = 0.95, scale = 2.0;
  const std::string spec = R"({"base_lr": 0.01, "shape": "constant", "groups": [
      {"name": "m", "match": ["layer."], "optimizer": "muon", "weight_decay": 0.1}], "default_group": "m"})";
  std::mt19937 rng(3);
  std::normal_distribution<float> n(0, 1);
  std::vector<float> W0(rows * cols), b0(cols), g1(rows * cols), g2(rows * cols), gb(cols);
  for (auto *v : {&W0, &b0, &g1, &g2, &gb})
    for (auto &x : *v) x = n(rng);
  Toy t(rows, cols, spec, mu, scale);
  REQUIRE(t.opt->muon_param() == std::vector<bool>{true, false});
  Toy::set(t.W, W0), Toy::set(t.b, b0), Toy::set(t.gW, g1), Toy::set(t.gb, gb);
  t.opt->step(0, 10, 10);
  const double step_lr = lr * scale * 0.2 * std::sqrt(double(std::max(rows, cols)));
  // step 1: momentum b1 = (1 - mu) g1, Nesterov u1 = (1 - mu) g1 + mu b1 ∝ g1
  const std::vector<float> o1 = ns_cpu(g1, cols, rows, 1, 5);
  std::vector<float> W1 = Toy::get(t.W);
  double err = 0, rms = 0;
  std::vector<double> ref1(W0.size());
  for (size_t i = 0; i < W0.size(); i++) {
    ref1[i] = W0[i] * (1 - lr * wd) - step_lr * o1[i];
    err = std::max(err, std::fabs(W1[i] - ref1[i]));
    rms += std::pow(W1[i] - W0[i] * (1 - lr * wd), 2);
  }
  CHECK(err < 1e-5);
  // RMS matching: the update's RMS is about 0.2 lr scale (as a typical AdamW step at that lr)
  rms = std::sqrt(rms / W0.size());
  CHECK(rms > 0.6 * 0.2 * lr * scale);
  CHECK(rms < 1.25 * 0.2 * lr * scale);
  // the bias takes AdamW: first step -lr sign(g) after the decay
  const std::vector<float> b1 = Toy::get(t.b);
  for (int i = 0; i < cols; i++) CHECK(b1[i] == Catch::Approx(b0[i] * (1 - lr * wd) - lr * (gb[i] > 0 ? 1 : -1)).margin(1e-5));
  // step 2: b2 = mu b1 + (1 - mu) g2, u2 = (1 - mu) g2 + mu b2
  Toy::set(t.gW, g2);
  t.opt->step(1, 10, 10);
  std::vector<float> u2(g1.size());
  for (size_t i = 0; i < u2.size(); i++) {
    const double m2 = mu * (1 - mu) * g1[i] + (1 - mu) * g2[i];
    u2[i] = float((1 - mu) * g2[i] + mu * m2);
  }
  const std::vector<float> o2 = ns_cpu(u2, cols, rows, 1, 5), W2 = Toy::get(t.W);
  err = 0;
  for (size_t i = 0; i < W0.size(); i++) err = std::max(err, std::fabs(W2[i] - (W1[i] * (1 - lr * wd) - step_lr * o2[i])));
  CHECK(err < 1e-5);
}

TEST_CASE("muon: bitwise deterministic", "[muon]") {
  const std::string spec = R"({"base_lr": 0.003, "shape": "cosine", "warmup_epochs": 0.2, "groups": [
      {"name": "m", "match": ["layer."], "optimizer": "muon"}], "default_group": "m"})";
  std::mt19937 rng(11);
  std::normal_distribution<float> n(0, 1);
  std::vector<float> W0(40 * 24), b0(24);
  for (auto *v : {&W0, &b0})
    for (auto &x : *v) x = n(rng);
  std::vector<std::vector<float>> gs;
  for (int s = 0; s < 6; s++) {
    gs.emplace_back(W0.size() + b0.size());
    for (auto &x : gs.back()) x = n(rng);
  }
  auto run = [&] {
    Toy t(40, 24, spec, 0.95, 1.0);
    Toy::set(t.W, W0), Toy::set(t.b, b0);
    for (int s = 0; s < 6; s++) {
      Toy::set(t.gW, std::vector<float>(gs[s].begin(), gs[s].begin() + W0.size()));
      Toy::set(t.gb, std::vector<float>(gs[s].begin() + W0.size(), gs[s].end()));
      t.opt->step(s, 5, 6);
    }
    std::vector<float> w = Toy::get(t.W), bb = Toy::get(t.b);
    w.insert(w.end(), bb.begin(), bb.end());
    return w;
  };
  const auto a = run(), b = run();
  REQUIRE(a.size() == b.size());
  CHECK(std::equal(a.begin(), a.end(), b.begin())); // bitwise
}

TEST_CASE("muon: ill-conditioned matrix regression, Muon matches or beats AdamW", "[muon]") {
  // W* (32 x 64), inputs x = R diag(s) z with a random rotation R and scales s over two decades
  // (non-axis-aligned conditioning), loss L = |X W^T - Y|^2 / (2 N), W from 0, 300 steps, cosine to 0;
  // the lr grid is wide enough that neither optimiser's best sits at its edge.
  const int rows = 32, cols = 64, N = 512, steps = 300;
  std::mt19937 rng(5);
  std::normal_distribution<double> n(0, 1);
  const Mat R = orthonormal(cols, cols, rng);
  Mat X(size_t(N) * cols), Wt(size_t(rows) * cols), Y(size_t(N) * rows);
  for (auto &w : Wt) w = n(rng);
  for (int i = 0; i < N; i++) {
    std::vector<double> z(cols);
    for (int j = 0; j < cols; j++) z[j] = n(rng) * std::pow(10.0, -2.0 * j / (cols - 1));
    for (int a = 0; a < cols; a++) {
      double v = 0;
      for (int j = 0; j < cols; j++) v += R[a * cols + j] * z[j];
      X[size_t(i) * cols + a] = v;
    }
    for (int r = 0; r < rows; r++) {
      double v = 0;
      for (int a = 0; a < cols; a++) v += X[size_t(i) * cols + a] * Wt[size_t(r) * cols + a];
      Y[size_t(i) * rows + r] = v;
    }
  }
  auto loss_grad = [&](const std::vector<float> &W, std::vector<float> *G) {
    double L = 0;
    if (G) std::fill(G->begin(), G->end(), 0.0f);
    std::vector<double> Gd(W.size(), 0.0);
    for (int i = 0; i < N; i++)
      for (int r = 0; r < rows; r++) {
        double e = -Y[size_t(i) * rows + r];
        for (int a = 0; a < cols; a++) e += X[size_t(i) * cols + a] * W[size_t(r) * cols + a];
        L += 0.5 * e * e / N;
        for (int a = 0; a < cols; a++) Gd[size_t(r) * cols + a] += e * X[size_t(i) * cols + a] / N;
      }
    if (G)
      for (size_t k = 0; k < Gd.size(); k++) (*G)[k] = float(Gd[k]);
    return L;
  };
  const double L0 = loss_grad(std::vector<float>(size_t(rows) * cols, 0.0f), nullptr);
  auto train = [&](const char *opt, double lr) {
    const std::string spec = std::string(R"({"base_lr": )") + std::to_string(lr) +
                             R"(, "min_lr": 0, "shape": "cosine", "groups": [{"name": "m", "match": ["layer."], "optimizer": ")" +
                             opt + R"("}], "default_group": "m"})";
    Toy t(rows, cols, spec, 0.95, 1.0);
    std::vector<float> W(size_t(rows) * cols, 0.0f), G(W.size());
    Toy::set(t.W, W), Toy::set(t.b, std::vector<float>(cols, 0.0f)), Toy::set(t.gb, std::vector<float>(cols, 0.0f));
    for (int s = 0; s < steps; s++) {
      W = Toy::get(t.W);
      loss_grad(W, &G);
      Toy::set(t.gW, G);
      t.opt->step(s, steps, steps);
    }
    return loss_grad(Toy::get(t.W), nullptr) / L0;
  };
  double best_adam = 1e9, best_muon = 1e9;
  for (double lr : {0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0}) {
    const double a = train("adam", lr), m = train("muon", lr);
    UNSCOPED_INFO("lr " << lr << ": AdamW " << a << ", Muon " << m << " (final / initial loss)");
    best_adam = std::min(best_adam, a), best_muon = std::min(best_muon, m);
  }
  INFO("best AdamW " << best_adam << ", best Muon " << best_muon);
  CHECK(best_muon <= best_adam * 1.05);
  CHECK(best_muon < 1e-2);
}
