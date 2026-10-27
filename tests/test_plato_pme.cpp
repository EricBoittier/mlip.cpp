// Plato LES by smooth PME (cmake/patches/ggml-plato-pme.patch): order-p B-spline spread,
// rfftn and ggml_plato_pme against the direct windowed Ewald sum in double, energy and its
// gradients w.r.t. the fractional coordinates, the charges and the cell metric (forces,
// virial), on every backend of the build.
#include "models/plato/plato.h"

#include <catch2/catch_test_macros.hpp>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr double SIGMA = 1.0, KC = 6.0, WD = 1.0;

void phi(double k2, double &f, double &df) { // as ggml_plato_pme
  f = df = 0;
  if (k2 <= 0) return;
  const double k = std::sqrt(k2), x = (k - (KC - WD)) / WD;
  if (x >= 1) return;
  double w = 1, dw = 0;
  if (x > 0) w = 0.5 * (1 + std::cos(M_PI * x)), dw = -0.5 * M_PI * std::sin(M_PI * x) / WD;
  const double e = std::exp(-0.5 * SIGMA * SIGMA * k2) / k2;
  f = w * e, df = dw / (2 * k) * e - w * e * (0.5 * SIGMA * SIGMA + 1 / k2);
}

struct Ref {
  double E = 0;
  std::vector<double> dfrac, dq, dA;
};

// (2 pi / V) sum_{n != 0} phi(n^T A n) |S(n)|^2, S(n) = sum_j q_j exp(2 pi i n . f_j)
Ref direct(const std::vector<double> &f, const std::vector<double> &q, const double A[9], double V, int nmax) {
  const int n = static_cast<int>(q.size());
  Ref r;
  r.dfrac.assign(3 * n, 0), r.dq.assign(n, 0), r.dA.assign(9, 0);
  const double pre = 2 * M_PI / V;
  for (int a = -nmax; a <= nmax; a++)
    for (int b = -nmax; b <= nmax; b++)
      for (int c = -nmax; c <= nmax; c++) {
        const double m[3] = {double(a), double(b), double(c)};
        double k2 = 0;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) k2 += A[3 * i + j] * m[i] * m[j];
        double p, dp;
        phi(k2, p, dp);
        if (p == 0 && dp == 0) continue;
        double C = 0, S = 0;
        std::vector<double> cs(n), sn(n);
        for (int j = 0; j < n; j++) {
          const double t = 2 * M_PI * (m[0] * f[3 * j] + m[1] * f[3 * j + 1] + m[2] * f[3 * j + 2]);
          cs[j] = std::cos(t), sn[j] = std::sin(t), C += q[j] * cs[j], S += q[j] * sn[j];
        }
        r.E += pre * p * (C * C + S * S);
        for (int j = 0; j < n; j++) {
          r.dq[j] += pre * p * 2 * (C * cs[j] + S * sn[j]);
          for (int x = 0; x < 3; x++) r.dfrac[3 * j + x] += pre * p * 2 * 2 * M_PI * m[x] * q[j] * (S * cs[j] - C * sn[j]);
        }
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) r.dA[3 * i + j] += pre * dp * (C * C + S * S) * m[i] * m[j];
      }
  return r;
}

double rel(const std::vector<double> &a, const std::vector<float> &b) {
  double d = 0, s = 0;
  for (size_t i = 0; i < a.size(); i++) d += (a[i] - b[i]) * (a[i] - b[i]), s += a[i] * a[i];
  return std::sqrt(d / s);
}

std::vector<ggml_backend_t> backends() {
  std::vector<ggml_backend_t> out = {ggml_backend_cpu_init()};
  if (ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))
    if (ggml_backend_t b = ggml_backend_dev_init(gpu, nullptr)) out.push_back(b);
  return out;
}

} // namespace

TEST_CASE("Plato LES PME matches the direct Ewald sum and its gradients", "[plato][pme]") {
  // a triclinic 8 A cell, 12 neutral charges; order 8, h = 0.5 sigma
  const double cell[3][3] = {{8.0, 0, 0}, {0.4, 8.3, 0}, {-0.3, 0.5, 7.7}};
  double inv[3][3], V = cell[0][0] * cell[1][1] * cell[2][2]; // lower triangular
  {
    const double a = cell[0][0], b = cell[1][0], c = cell[1][1], d = cell[2][0], e = cell[2][1], g = cell[2][2];
    const double I[3][3] = {{1 / a, 0, 0}, {-b / (a * c), 1 / c, 0}, {(b * e - c * d) / (a * c * g), -e / (c * g), 1 / g}};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) inv[i][j] = I[i][j];
  }
  // A = 4 pi^2 inv^T inv (k = 2 pi m inv^T, as plato.cpp's K)
  double A[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      double s = 0;
      for (int q = 0; q < 3; q++) s += inv[q][i] * inv[q][j];
      A[3 * i + j] = 4 * M_PI * M_PI * s;
    }
  const int n = 12, order = 8;
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> U(0, 1);
  std::normal_distribution<double> N(0, 1);
  std::vector<double> f(3 * n), q(n);
  double mean = 0;
  for (int j = 0; j < n; j++) {
    for (int x = 0; x < 3; x++) f[3 * j + x] = U(rng);
    q[j] = N(rng), mean += q[j] / n;
  }
  for (double &x : q) x -= mean;
  const Ref ref = direct(f, q, A, V, 12);

  int mesh[3];
  for (int a = 0; a < 3; a++) {
    const float row[3] = {float(cell[a][0]), float(cell[a][1]), float(cell[a][2])};
    mesh[a] = mlipcpp::plato::pme_mesh_points(row, 0.5 * SIGMA, order);
  }
  const std::vector<float> tab = mlipcpp::plato::pme_axis_table(mesh, order);

  for (ggml_backend_t be : backends()) {
    INFO("backend " << ggml_backend_name(be));
    ggml_context *ctx = ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(512, true), nullptr, true});
    ggml_tensor *tf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, n), *tq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n);
    ggml_tensor *tA = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 9), *tt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tab.size());
    ggml_set_param(tf), ggml_set_param(tq), ggml_set_param(tA);
    ggml_tensor *rho = ggml_plato_spread_order(ctx, tq, tf, nullptr, 1, mesh[0], mesh[1], mesh[2], order);
    ggml_tensor *rk = ggml_plato_fft3(ctx, rho, 0, mesh[0], mesh[1], mesh[2]);
    ggml_tensor *ep = ggml_plato_pme(ctx, rk, tA, tt, mesh[0], mesh[1], mesh[2], SIGMA, KC, WD);
    ggml_tensor *E = ggml_scale(ctx, ggml_sum(ctx, ep), float(2 * M_PI / V));
    ggml_set_loss(E);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 512, true);
    ggml_build_forward_expand(gf, E);
    ggml_build_backward_expand(ctx, gf, nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    REQUIRE(buf != nullptr);
    std::vector<float> ff(f.begin(), f.end()), qf(q.begin(), q.end()), Af(A, A + 9);
    ggml_backend_tensor_set(tf, ff.data(), 0, ggml_nbytes(tf));
    ggml_backend_tensor_set(tq, qf.data(), 0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tA, Af.data(), 0, ggml_nbytes(tA));
    ggml_backend_tensor_set(tt, tab.data(), 0, ggml_nbytes(tt));
    ggml_graph_reset(gf);
    REQUIRE(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
    float e = 0;
    ggml_backend_tensor_get(E, &e, 0, sizeof(float));
    std::vector<float> gfr(3 * n), gq(n), gA(9);
    ggml_backend_tensor_get(ggml_graph_get_grad(gf, tf), gfr.data(), 0, gfr.size() * 4);
    ggml_backend_tensor_get(ggml_graph_get_grad(gf, tq), gq.data(), 0, gq.size() * 4);
    ggml_backend_tensor_get(ggml_graph_get_grad(gf, tA), gA.data(), 0, gA.size() * 4);
    CHECK(std::abs(e - ref.E) < 2e-6 * std::abs(ref.E));
    CHECK(rel(ref.dfrac, gfr) < 5e-5);
    CHECK(rel(ref.dq, gq) < 5e-5);
    CHECK(rel(ref.dA, gA) < 5e-5);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
  }
}

TEST_CASE("Plato order-p spread and gather are adjoint partitions of unity", "[plato][pme]") {
  const int n = 7, M[3] = {12, 15, 16};
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> U(-0.5f, 1.5f);
  std::vector<float> f(3 * n), v(n), m(M[0] * M[1] * M[2]);
  for (float &x : f) x = U(rng);
  for (float &x : v) x = U(rng);
  for (float &x : m) x = U(rng);
  for (ggml_backend_t be : backends())
    for (int order : {4, 6, 8, 10, 12}) {
      INFO("backend " << ggml_backend_name(be) << " order " << order);
      ggml_context *ctx = ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
      ggml_tensor *tf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, n), *tv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n);
      ggml_tensor *tm = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, M[2], M[1], M[0], 1);
      ggml_tensor *rho = ggml_plato_spread_order(ctx, tv, tf, nullptr, 1, M[0], M[1], M[2], order);
      ggml_tensor *g = ggml_plato_gather_order(ctx, tm, tf, nullptr, 1, false, order);
      ggml_tensor *a = ggml_sum(ctx, ggml_mul(ctx, rho, tm)), *b = ggml_sum(ctx, ggml_mul(ctx, g, tv));
      ggml_tensor *mass = ggml_sum(ctx, rho);
      ggml_cgraph *gr = ggml_new_graph(ctx);
      ggml_build_forward_expand(gr, a), ggml_build_forward_expand(gr, b), ggml_build_forward_expand(gr, mass);
      ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
      ggml_backend_tensor_set(tf, f.data(), 0, ggml_nbytes(tf));
      ggml_backend_tensor_set(tv, v.data(), 0, ggml_nbytes(tv));
      ggml_backend_tensor_set(tm, m.data(), 0, ggml_nbytes(tm));
      REQUIRE(ggml_backend_graph_compute(be, gr) == GGML_STATUS_SUCCESS);
      float x = 0, y = 0, s = 0, vs = 0;
      ggml_backend_tensor_get(a, &x, 0, 4), ggml_backend_tensor_get(b, &y, 0, 4), ggml_backend_tensor_get(mass, &s, 0, 4);
      for (float t : v) vs += t;
      CHECK(std::abs(x - y) < 1e-5f * std::max(1.0f, std::abs(x))); // <spread v, m> = <v, gather m>
      CHECK(std::abs(s - vs) < 1e-5f * std::max(1.0f, std::abs(vs)));
      ggml_backend_buffer_free(buf);
      ggml_free(ctx);
    }
}
