// Combined long-range model (src/runtime/lr_combined.h), host side: integer charge per separated fragment with no charge
// on a neutral partner, the local charge's smoothness through the cluster link window, forces of the long-range term and
// of the conditioning against central differences, the reduction to the structure's charge for one cluster, and a zero
// long-range energy when no inter-cluster pair reaches lr_R - lr_t.
#include "runtime/lr_combined.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <vector>

using namespace mlipcpp::runtime;
using Catch::Approx;

namespace {
constexpr double KE = 14.399645;
// H Li C N O F Na P Cl
const int ZS[] = {1, 3, 6, 7, 8, 9, 11, 15, 17};
const double IP[] = {13.598, 5.392, 11.260, 14.534, 13.618, 17.423, 5.139, 10.487, 12.968};
const double EA[] = {0.754, 0.618, 1.262, 0.0, 1.461, 3.401, 0.548, 0.746, 3.613};
const double RC[] = {0.31, 1.28, 0.76, 0.71, 0.66, 0.57, 1.66, 1.07, 1.02};
const double LO[] = {-1, 0, -1, -1, -1, -1, 0, -1, -1};
const double HI[] = {1, 1, 1, 1, 1, 0, 1, 1, 0};

SqeSpec sqe_spec() {
  SqeSpec s;
  s.enabled = true, s.lr_R = 8.0, s.lr_t = 2.0;
  const double lJ = (std::log(s.J_hi) + std::log(s.J_lo)) / 2, hJ = (std::log(s.J_hi) - std::log(s.J_lo)) / 2;
  for (int k = 0; k < 9; k++) {
    const double chi0 = 0.5 * (IP[k] + EA[k]), J0 = IP[k] - EA[k];
    s.Z.push_back(ZS[k]), s.chi0.push_back(chi0), s.J0.push_back(J0), s.rcov.push_back(RC[k]);
    s.sigma.push_back(KE * std::sqrt(2.0 / M_PI) / (s.J_lo * J0));
    s.chi_a.push_back(0.1 * k - 0.3), s.chi_b.push_back(0.2), s.j_z.push_back(std::atanh(-lJ / hJ));
    s.kappa_z.push_back(0.0), s.q0_lo.push_back(LO[k]), s.q0_hi.push_back(HI[k]), s.q0_a.push_back(0.0);
  }
  s.q0_b.assign(81, 0.0);
  s.w_raw = -0.821;
  return s;
}
LrcSpec lrc_spec() {
  LrcSpec L;
  L.enabled = true;
  for (int k = 0; k < 9; k++) L.ip.push_back(IP[k]), L.ea.push_back(EA[k]), L.h.push_back(HI[k]), L.l.push_back(-LO[k]);
  L.ap.assign(9, 0.0), L.am.assign(9, 0.0), L.bp.assign(81, 0.0), L.bm.assign(81, 0.0);
  // molecular levels as a fit would make them: Na donor 5.14 - 5 tanh(1) = 1.33 eV, Cl acceptor 3.61 + 5 tanh(0.5) =
  // 5.92 eV (so NaCl is ionic), the water's O acceptor pushed down by its H neighbours (bm[O][H] = -0.5 per H) and the
  // H donor level deepened by O (bp[H][O]): mu sits >= 2 eV from every level it must not cross
  L.ap[6] = 1.0, L.am[8] = 0.5;
  L.bm[4 * 9 + 0] = -0.5, L.bp[0 * 9 + 4] = -0.3;
  return L;
}
// Na+ at the origin, a water (O H H) around (wx, 0, 0), Cl- at (cx, cy, 0)
std::vector<double> geom(double wx, double cx, double cy) {
  return {0, 0, 0, wx, 0, 0, wx + 0.96, 0, 0, wx - 0.24, 0.93, 0, cx, cy, 0};
}
const int32_t ZG[] = {11, 8, 1, 1, 17};

double fd(const std::function<double(const std::vector<double> &)> &f, std::vector<double> x, size_t k, double h) {
  x[k] += h;
  const double a = f(x);
  x[k] -= 2 * h;
  return (a - f(x)) / (2 * h);
}
} // namespace

TEST_CASE("lrc: integer charge per separated fragment, none on the neutral partner", "[lrc]") {
  const SqeSpec S = sqe_spec();
  const LrcSpec L = lrc_spec();
  const auto x = geom(12.0, -12.0, 0.0);
  const LrcQ0 q = lrc_q0(L, S, 5, ZG, x.data(), 0.0);
  REQUIRE(q.feasible);
  CHECK(q.q0[0] == Approx(1.0).margin(1e-3));
  CHECK(q.q0[4] == Approx(-1.0).margin(1e-3));
  CHECK(std::abs(q.q0[1] + q.q0[2] + q.q0[3]) < 1e-3); // sum q_N on the water
  // Na+ and water, Q = +1: the charge sits on Na; the water keeps < 1e-4 e
  const LrcQ0 q1 = lrc_q0(L, S, 4, ZG, x.data(), 1.0);
  CHECK(q1.q0[0] == Approx(1.0).margin(1e-3));
  CHECK(std::abs(q1.q0[1] + q1.q0[2] + q1.q0[3]) < 1e-3);
  // the conditioning: three clusters, each on its own integer charge, blend 0
  const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), 0.0, 10);
  REQUIRE(c.cl.s == 3);
  // effective conditioning charge k + S (rows k, k + 1 blended): each cluster's own integer
  auto eff = [&](int i) { return c.row_charge[i] + double(c.blend[i]); };
  CHECK(eff(0) == Approx(1.0).margin(1e-3));
  CHECK(eff(4) == Approx(-1.0).margin(1e-3));
  CHECK(eff(1) == Approx(0.0).margin(1e-3));
}

TEST_CASE("lrc: one cluster is the structure's charge and no long-range term", "[lrc]") {
  const SqeSpec S = sqe_spec();
  const LrcSpec L = lrc_spec();
  const auto x = geom(2.5, -2.4, 0.5); // all within 5 A of each other
  const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), 0.0, 10);
  CHECK(c.cl.s == 1);
  for (int i = 0; i < 5; i++) {
    CHECK(c.row_charge[i] == 0);
    CHECK(c.blend[i] == 0.0f);
  }
  const LrcEnergy e = lrc_energy(L, S, c, 5, ZG, x.data(), 0.0, true);
  CHECK(e.energy == 0.0);
  for (double f : e.forces) CHECK(f == 0.0);
}

TEST_CASE("lrc: zero long-range energy when no inter-cluster pair passes lr_R - lr_t", "[lrc]") {
  const SqeSpec S = sqe_spec(); // switch 6 -> 8 A
  const LrcSpec L = lrc_spec();
  // Na+ / Cl- 5.6 A apart: two clusters joined by a weak link, no pair beyond 6 A
  const std::vector<double> x = {0, 0, 0, -5.6, 0.1, 0};
  const int32_t Z2[] = {11, 17};
  const LrcCondition c = lrc_condition(L, S, 2, Z2, x.data(), 0.0, 10);
  REQUIRE(c.cl.s == 2);
  const LrcEnergy e = lrc_energy(L, S, c, 2, Z2, x.data(), 0.0, true);
  CHECK(std::abs(e.energy) < 1e-12);
}

TEST_CASE("lrc: local charge is smooth through the link window", "[lrc]") {
  const SqeSpec S = sqe_spec();
  const LrcSpec L = lrc_spec();
  // Cl- leaves the Na+ water cluster: nearest distance 4.6 -> 6.4 A (window 5 - 6 A)
  double prev = 0, prev_d = 0, max_step = 0, max_d2 = 0;
  const double h = 1e-3;
  std::vector<double> Ql;
  for (double r = 4.6; r <= 6.4 + 1e-9; r += h) {
    const auto x = geom(2.4, -r, 0.0);
    const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), 0.0, 10);
    Ql.push_back(c.Qloc[c.cl.sn[0]]);
  }
  for (size_t k = 1; k < Ql.size(); k++) max_step = std::max(max_step, std::abs(Ql[k] - Ql[k - 1]));
  for (size_t k = 1; k + 1 < Ql.size(); k++) max_d2 = std::max(max_d2, std::abs(Ql[k + 1] - 2 * Ql[k] + Ql[k - 1]) / (h * h));
  (void)prev, (void)prev_d;
  CHECK(Ql.front() == Approx(0.0).margin(1e-6)); // connected: Na cluster sees the total (0)
  CHECK(Ql.back() == Approx(1.0).margin(1e-3));  // separated: its own +1
  CHECK(max_step < 3e-3);                        // continuous (slope <= 1.875 / A over a 1 A window)
  CHECK(max_d2 < 10.0);                          // bounded curvature (C2 switch)
}

TEST_CASE("lrc: long-range and conditioning forces match central differences", "[lrc]") {
  const SqeSpec S = sqe_spec();
  const LrcSpec L = lrc_spec();
  // Cl- at 5.5 A from the water (a weak link) and 7-9 A from Na+ (in the long-range switch); everything off-axis
  const auto x0 = std::vector<double>{0, 0, 0, 2.4, 0.1, 0.2, 3.3, 0.3, 0.1, 2.1, 1.0, -0.2, 8.6, 2.5, 0.4};
  const double Q = 0.0;
  auto energy = [&](const std::vector<double> &x) {
    const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10);
    return lrc_energy(L, S, c, 5, ZG, x.data(), Q, false).energy;
  };
  const LrcCondition c0 = lrc_condition(L, S, 5, ZG, x0.data(), Q, 10);
  REQUIRE(c0.cl.s == 2);
  REQUIRE(!c0.cl.trivial());
  const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, x0.data(), Q, true);
  CHECK(std::abs(e0.energy) > 1e-4);
  double fmax = 0, err = 0;
  for (size_t k = 0; k < x0.size(); k++) {
    const double num = -fd(energy, x0, k, 1e-5);
    fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
  }
  INFO("long-range force: max |F| " << fmax << ", max error " << err);
  CHECK(err < 1e-6 * std::max(1.0, fmax));
  // conditioning: f(x) = sum_i g_i S(atom i), forces -df/dx from lrc_condition_forces with G_a = sum_{i in a} g_i
  const double g[5] = {0.7, -0.3, 0.2, 0.5, -1.1};
  auto fS = [&](const std::vector<double> &x) {
    const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10);
    double v = 0;
    for (int i = 0; i < 5; i++) v += g[i] * (c.k[c.cl.sn[i]] + c.S[c.cl.sn[i]]);
    return v;
  };
  std::vector<double> G(c0.cl.s, 0.0), F(15, 0.0);
  for (int i = 0; i < 5; i++) G[c0.cl.sn[i]] += g[i];
  REQUIRE(c0.needs_cond_forces());
  lrc_condition_forces(L, S, c0, 5, ZG, x0.data(), G, F.data());
  fmax = 0, err = 0;
  for (size_t k = 0; k < x0.size(); k++) {
    const double num = -fd(fS, x0, k, 1e-5);
    fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - F[k]));
  }
  INFO("conditioning force: max |F| " << fmax << ", max error " << err);
  CHECK(fmax > 1e-3);
  CHECK(err < 1e-6 * std::max(1.0, fmax));
}

TEST_CASE("lrc: head and soft-rounded q0 modes, forces match central differences", "[lrc]") {
  SqeSpec S = sqe_spec();
  for (int k = 0; k < 9; k++) S.q0_a[k] = 0.3 * (k % 3) - 0.3;
  S.q0_b[6 * 9 + 4] = 0.8, S.q0_b[8 * 9 + 0] = -0.4, S.q0_b[4 * 9 + 0] = 0.2; // Na-O, Cl-H, O-H
  // Cl- 6.6 A from the water H (a cross pair inside the 6-8 A merge switch), 9.9 A from Na+
  const auto x0 = std::vector<double>{0, 0, 0, 2.4, 0.1, 0.2, 3.3, 0.3, 0.1, 2.1, 1.0, -0.2, 9.5, 2.5, 0.4};
  for (int mode = 0; mode < 2; mode++) {
    LrcSpec L = lrc_spec();
    (mode == 0 ? L.head_only : L.round) = true;
    const LrcQ0 q = lrc_q0(L, S, 5, ZG, x0.data(), 0.0);
    double sum = 0;
    for (double v : q.q0) sum += v;
    CHECK(sum == Approx(0.0).margin(1e-9));
    if (mode == 1) {
      REQUIRE(q.K == 2);
      CHECK(q.w[0] > 0.0);
      CHECK(q.w[0] < 1.0);
    }
    auto energy = [&](const std::vector<double> &x) {
      const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), 0.0, 10);
      return lrc_energy(L, S, c, 5, ZG, x.data(), 0.0, false).energy;
    };
    const LrcCondition c0 = lrc_condition(L, S, 5, ZG, x0.data(), 0.0, 10);
    REQUIRE(c0.cl.s == 2);
    const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, x0.data(), 0.0, true);
    double fmax = 0, err = 0;
    for (size_t k = 0; k < x0.size(); k++) {
      const double num = -fd(energy, x0, k, 1e-5);
      fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
    }
    INFO("mode " << mode << " long-range force: max |F| " << fmax << ", max error " << err);
    CHECK(fmax > 1e-4);
    CHECK(err < 1e-6 * std::max(1.0, fmax));
    const double g[5] = {0.7, -0.3, 0.2, 0.5, -1.1};
    auto fS = [&](const std::vector<double> &x) {
      const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), 0.0, 10);
      double v = 0;
      for (int i = 0; i < 5; i++) v += g[i] * (c.k[c.cl.sn[i]] + c.S[c.cl.sn[i]]);
      return v;
    };
    std::vector<double> G(c0.cl.s, 0.0), F(15, 0.0);
    for (int i = 0; i < 5; i++) G[c0.cl.sn[i]] += g[i];
    lrc_condition_forces(L, S, c0, 5, ZG, x0.data(), G, F.data());
    fmax = 0, err = 0;
    for (size_t k = 0; k < x0.size(); k++) {
      const double num = -fd(fS, x0, k, 1e-5);
      fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - F[k]));
    }
    INFO("mode " << mode << " conditioning force: max |F| " << fmax << ", max error " << err);
    CHECK(err < 1e-6 * std::max(1.0, fmax));
  }
}

namespace {
// analytic stand-in for the frozen charge pass: q_raw_i = a_Z + sum_j b_ZZ' exp(-r_ij / 2) + 0.1 Q (position dependent,
// smooth, like a backbone head); its VJP analytic
struct ToyQ final : LrcQSource {
  static double a(int z) { return z == 11 ? 0.8 : z == 17 ? -0.7 : z == 8 ? -0.5 : 0.25; }
  static double b(int zi, int zj) { return 0.05 * ((zi * 7 + zj * 3) % 5) - 0.1; }
  std::vector<double> raw(int n, const int32_t *Z, const double *x, double Q, int) override {
    std::vector<double> q(n);
    for (int i = 0; i < n; i++) {
      q[i] = a(Z[i]) + 0.1 * Q;
      for (int j = 0; j < n; j++)
        if (j != i) q[i] += b(Z[i], Z[j]) * std::exp(-r(x, i, j) / 2);
    }
    return q;
  }
  void raw_vjp(int n, const int32_t *Z, const double *x, double, int, const double *w, double *g) override {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        if (j == i) continue;
        const double d = r(x, i, j), dq = -0.5 * w[i] * b(Z[i], Z[j]) * std::exp(-d / 2);
        for (int k = 0; k < 3; k++) {
          const double u = (x[3 * i + k] - x[3 * j + k]) / d;
          g[3 * i + k] += dq * u, g[3 * j + k] -= dq * u;
        }
      }
  }
  static double r(const double *x, int i, int j) {
    double s = 0;
    for (int k = 0; k < 3; k++) s += (x[3 * i + k] - x[3 * j + k]) * (x[3 * i + k] - x[3 * j + k]);
    return std::sqrt(s);
  }
};
LrcSpec graph_spec() {
  LrcSpec L;
  L.enabled = true, L.graph = true;
  L.fukui_Z = {1, 8, 11, 17}, L.fukui_phi = {0.3, -0.2, 1.1, 0.5};
  L.qsrc = std::make_shared<ToyQ>();
  return L;
}
} // namespace

TEST_CASE("lrc graph: Fukui equilibration to Q, fixed-charge inter-cluster Coulomb, exact forces", "[lrc]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0; // the 8-10 A switch of feat/lr-chargehead
  const LrcSpec L = graph_spec();
  // Cl- 5.5 A from the water (a weak link: transition) and 8-10 A from Na+ (inside the long-range switch)
  const auto x0 = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
  for (double Q : {0.0, 1.0}) {
    const LrcQ0 q = lrc_q0(L, S, 5, ZG, x0.data(), Q, 1);
    double sum = 0;
    for (double v : q.q0) sum += v;
    CHECK(sum == Approx(Q).margin(1e-12));
  }
  const double Q = 0.0;
  auto energy = [&](const std::vector<double> &x) {
    const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, 1);
    return lrc_energy(L, S, c, 5, ZG, x.data(), Q, false).energy;
  };
  const LrcCondition c0 = lrc_condition(L, S, 5, ZG, x0.data(), Q, 10, 1);
  REQUIRE(c0.cl.s == 2);
  REQUIRE(!c0.cl.trivial());
  REQUIRE(c0.needs_cond_forces());
  const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, x0.data(), Q, true);
  CHECK(std::abs(e0.energy) > 1e-4);
  double fmax = 0, err = 0;
  for (size_t k = 0; k < x0.size(); k++) {
    const double num = -fd(energy, x0, k, 1e-5);
    fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
  }
  INFO("graph long-range force: max |F| " << fmax << ", max error " << err);
  CHECK(fmax > 1e-4);
  CHECK(err < 1e-6 * std::max(1.0, fmax));
  // conditioning forces and the blend rows dS_a/dx
  const double g[5] = {0.7, -0.3, 0.2, 0.5, -1.1};
  auto fS = [&](const std::vector<double> &x) {
    const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, 1);
    double v = 0;
    for (int i = 0; i < 5; i++) v += g[i] * (c.k[c.cl.sn[i]] + c.S[c.cl.sn[i]]);
    return v;
  };
  std::vector<double> G(c0.cl.s, 0.0), F(15, 0.0);
  for (int i = 0; i < 5; i++) G[c0.cl.sn[i]] += g[i];
  lrc_condition_forces(L, S, c0, 5, ZG, x0.data(), G, F.data());
  const std::vector<double> rows = lrc_blend_rows(L, S, c0, 5, ZG, x0.data());
  fmax = 0, err = 0;
  double err_rows = 0;
  for (size_t k = 0; k < x0.size(); k++) {
    const double num = -fd(fS, x0, k, 1e-5);
    double fr = 0;
    for (int a = 0; a < c0.cl.s; a++) fr -= G[a] * rows[size_t(a) * 15 + k];
    fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - F[k]));
    err_rows = std::max(err_rows, std::abs(num - fr));
  }
  INFO("graph conditioning force: max |F| " << fmax << ", max error " << err << ", via rows " << err_rows);
  CHECK(fmax > 1e-3);
  CHECK(err < 1e-6 * std::max(1.0, fmax));
  CHECK(err_rows < 1e-6 * std::max(1.0, fmax));
  // nothing inside lr_R - lr_t: Cl- pulled to 7 A of Na+ keeps a zero long-range energy
  const auto xn = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 7.0, 0.0, 0.0};
  const LrcCondition cn = lrc_condition(L, S, 5, ZG, xn.data(), Q, 10, 1);
  if (cn.cl.s > 1) CHECK(lrc_energy(L, S, cn, 5, ZG, xn.data(), Q, false).energy == 0.0);
}

namespace {
// graph spec with atomic multipoles: three radial channels inside 4.5 A, element-pair parameters from a fixed pattern
LrcSpec mp_spec(double scale) {
  LrcSpec L = graph_spec();
  L.mp = true, L.mp_rc = 4.5, L.mp_width = 0.6, L.mp_centers = {1.0, 2.0, 3.2};
  L.mp_Z = {1, 8, 11, 17};
  const size_t E = 4, K = 3;
  for (size_t a = 0; a < E; a++)
    for (size_t b = 0; b < E; b++)
      for (size_t k = 0; k < K; k++) {
        L.mp_theta.push_back(scale * 0.1 * (double((a * 5 + b * 3 + k * 7) % 7) - 3.0));
        L.mp_tau.push_back(scale * 0.08 * (double((a * 3 + b * 7 + k * 5) % 5) - 2.0));
      }
  L.mp_build();
  return L;
}
std::vector<double> rot(const std::vector<double> &x, double t) { // rotation about (1, 1, 1) / sqrt 3
  const double c = std::cos(t), s = std::sin(t), u = 1 / std::sqrt(3.0);
  const double R[3][3] = {{c + u * u * (1 - c), u * u * (1 - c) - u * s, u * u * (1 - c) + u * s},
                          {u * u * (1 - c) + u * s, c + u * u * (1 - c), u * u * (1 - c) - u * s},
                          {u * u * (1 - c) - u * s, u * u * (1 - c) + u * s, c + u * u * (1 - c)}};
  std::vector<double> y(x.size());
  for (size_t i = 0; i < x.size() / 3; i++)
    for (int a = 0; a < 3; a++) y[3 * i + a] = R[a][0] * x[3 * i] + R[a][1] * x[3 * i + 1] + R[a][2] * x[3 * i + 2];
  return y;
}
} // namespace

TEST_CASE("lrc mp: atomic multipoles are equivariant and traceless, zero parameters change nothing", "[lrc]") {
  const LrcSpec L = mp_spec(1.0);
  const auto x0 = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
  std::vector<double> mu, th, mu2, th2;
  lrc_moments(L, 5, ZG, x0.data(), mu, th);
  const double t = 0.7;
  lrc_moments(L, 5, ZG, rot(x0, t).data(), mu2, th2);
  const std::vector<double> mur = rot(mu, t);
  double nz = 0;
  for (int i = 0; i < 5; i++) {
    CHECK(th[9 * i] + th[9 * i + 4] + th[9 * i + 8] == Approx(0).margin(1e-12));
    for (int a = 0; a < 3; a++) {
      CHECK(mu2[3 * i + a] == Approx(mur[3 * i + a]).margin(1e-12));
      nz += std::abs(mu[3 * i + a]);
    }
    // Theta' = R Theta R^T: compare the quadratic forms on the rotated probe direction
    const std::vector<double> v = {0.3, -0.5, 0.8}, vr = rot(v, t);
    double q1 = 0, q2 = 0;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) q1 += v[a] * th[9 * i + 3 * a + b] * v[b], q2 += vr[a] * th2[9 * i + 3 * a + b] * vr[b];
    CHECK(q2 == Approx(q1).margin(1e-12));
  }
  CHECK(nz > 1e-3);
  // zero parameters: identical to the charge-only path
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  const LrcSpec L0 = mp_spec(0.0), Lq = graph_spec();
  const LrcCondition c = lrc_condition(Lq, S, 5, ZG, x0.data(), 0.0, 10, 1);
  const LrcEnergy a = lrc_energy(Lq, S, c, 5, ZG, x0.data(), 0.0, true), b = lrc_energy(L0, S, c, 5, ZG, x0.data(), 0.0, true);
  CHECK(b.energy == Approx(a.energy).epsilon(1e-12));
  for (size_t k = 0; k < a.forces.size(); k++) CHECK(b.forces[k] == Approx(a.forces[k]).margin(1e-12));
}

TEST_CASE("lrc mp: charge, dipole and quadrupole long range, forces match central differences", "[lrc]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  const LrcSpec L = mp_spec(1.0);
  const auto x0 = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
  for (double Q : {0.0, 1.0}) {
    auto energy = [&](const std::vector<double> &x) {
      const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, 1);
      return lrc_energy(L, S, c, 5, ZG, x.data(), Q, false).energy;
    };
    const LrcCondition c0 = lrc_condition(L, S, 5, ZG, x0.data(), Q, 10, 1);
    REQUIRE(c0.cl.s == 2);
    const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, x0.data(), Q, true);
    const LrcEnergy eq = lrc_energy(graph_spec(), S, c0, 5, ZG, x0.data(), Q, false);
    CHECK(std::abs(e0.energy - eq.energy) > 1e-5); // the multipoles contribute
    double fmax = 0, err = 0;
    for (size_t k = 0; k < x0.size(); k++) {
      const double num = -fd(energy, x0, k, 1e-5);
      fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
    }
    INFO("Q " << Q << " mp long-range force: max |F| " << fmax << ", max error " << err);
    CHECK(fmax > 1e-4);
    CHECK(err < 1e-6 * std::max(1.0, fmax));
  }
}

TEST_CASE("lrc mp: point multipoles equal octahedral satellites far apart", "[lrc]") {
  // two separated molecules 30 A apart (switch 1, no links): the pair sum over the satellites (centre q, six charges at
  // +-a e_k of each atom, a = 0.3 A) agrees with the q / mu / Theta interaction to the omitted mu-Theta, Theta-Theta and
  // higher terms
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  const LrcSpec L = mp_spec(0.5);
  const std::vector<double> x0 = {0, 0, 0, 0.96, 0.1, 0, -0.3, 0.9, 0.1, 30.0, 1.0, 0.5, 31.6, 1.2, 0.3};
  const int32_t Zs[] = {8, 1, 1, 11, 17};
  const LrcCondition c = lrc_condition(L, S, 5, Zs, x0.data(), 0.0, 10, 1);
  REQUIRE(c.cl.s == 2);
  const double e_mp = lrc_energy(L, S, c, 5, Zs, x0.data(), 0.0, false).energy;
  std::vector<double> mu, th;
  lrc_moments(L, 5, Zs, x0.data(), mu, th);
  const double a = 0.3;
  std::vector<std::vector<std::pair<std::array<double, 3>, double>>> sites(5);
  for (int i = 0; i < 5; i++) {
    sites[i].push_back({{x0[3 * i], x0[3 * i + 1], x0[3 * i + 2]}, c.q0.q0[i]});
    // eigenvectors of Theta_i by Jacobi rotations
    double A[3][3], V[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int p = 0; p < 3; p++)
      for (int q = 0; q < 3; q++) A[p][q] = th[9 * i + 3 * p + q];
    for (int sweep = 0; sweep < 50; sweep++)
      for (int p = 0; p < 2; p++)
        for (int q = p + 1; q < 3; q++) {
          if (std::abs(A[p][q]) < 1e-15) continue;
          const double phi = 0.5 * std::atan2(2 * A[p][q], A[q][q] - A[p][p]), cs = std::cos(phi), sn = std::sin(phi);
          for (int k = 0; k < 3; k++) {
            const double akp = A[k][p], akq = A[k][q];
            A[k][p] = cs * akp - sn * akq, A[k][q] = sn * akp + cs * akq;
          }
          for (int k = 0; k < 3; k++) {
            const double apk = A[p][k], aqk = A[q][k];
            A[p][k] = cs * apk - sn * aqk, A[q][k] = sn * apk + cs * aqk;
          }
          for (int k = 0; k < 3; k++) {
            const double vkp = V[k][p], vkq = V[k][q];
            V[k][p] = cs * vkp - sn * vkq, V[k][q] = sn * vkp + cs * vkq;
          }
        }
    for (int k = 0; k < 3; k++) {
      const double sk = 2 * A[k][k] / (3 * a * a);
      double me = 0;
      for (int d = 0; d < 3; d++) me += mu[3 * i + d] * V[d][k];
      for (int sg : {1, -1}) {
        std::array<double, 3> p;
        for (int d = 0; d < 3; d++) p[d] = x0[3 * i + d] + sg * a * V[d][k];
        sites[i].push_back({p, sk / 2 + sg * me / (2 * a)});
      }
    }
  }
  double e_sat = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 3; j < 5; j++)
      for (const auto &[pi, qi] : sites[i])
        for (const auto &[pj, qj] : sites[j]) {
          double r2 = 0;
          for (int d = 0; d < 3; d++) r2 += (pi[d] - pj[d]) * (pi[d] - pj[d]);
          e_sat += KE * qi * qj / std::sqrt(r2);
        }
  INFO("point multipoles " << e_mp << " eV, satellites " << e_sat << " eV");
  CHECK(std::abs(e_mp) > 1e-4);
  CHECK(e_sat == Approx(e_mp).epsilon(2e-3));
}

TEST_CASE("lrc pol: induced dipoles, -1/2 alpha F^2 limit and forces against central differences", "[lrc]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  for (bool with_mp : {false, true}) {
    LrcSpec L = with_mp ? mp_spec(1.0) : graph_spec();
    L.pol = true, L.pol_Z = {1, 8, 11, 17}, L.pol_alpha = {0.4, 0.8, 0.2, 2.5};
    const auto x0 = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
    for (double Q : {0.0, 1.0}) {
      auto energy = [&](const std::vector<double> &x) {
        const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, 1);
        return lrc_energy(L, S, c, 5, ZG, x.data(), Q, false).energy;
      };
      const LrcCondition c0 = lrc_condition(L, S, 5, ZG, x0.data(), Q, 10, 1);
      REQUIRE(c0.cl.s == 2);
      const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, x0.data(), Q, true);
      LrcSpec Ln = L;
      Ln.pol = false;
      const double enp = lrc_energy(Ln, S, c0, 5, ZG, x0.data(), Q, false).energy;
      CHECK(e0.energy - enp < -1e-6); // induction lowers the energy
      double fmax = 0, err = 0;
      for (size_t k = 0; k < x0.size(); k++) {
        const double num = -fd(energy, x0, k, 1e-5);
        fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
      }
      INFO("mp " << with_mp << " Q " << Q << " pol long-range force: max |F| " << fmax << ", max error " << err);
      CHECK(err < 1e-6 * std::max(1.0, fmax));
    }
  }
  // isolated Cl (alpha 2.5) 20 A from a Na (alpha 0.2): -1/2 KE (alpha_Cl q_Na^2 + alpha_Na q_Cl^2) / R^4 to the mutual term
  LrcSpec L = graph_spec();
  L.pol = true, L.pol_Z = {11, 17}, L.pol_alpha = {0.2, 2.5};
  const int32_t Zp[] = {11, 17};
  const std::vector<double> x = {0, 0, 0, 20.0, 0, 0};
  const LrcCondition c = lrc_condition(L, S, 2, Zp, x.data(), 0.0, 10, 1);
  REQUIRE(c.cl.s == 2);
  LrcSpec Ln = L;
  Ln.pol = false;
  const double ep = lrc_energy(L, S, c, 2, Zp, x.data(), 0.0, false).energy - lrc_energy(Ln, S, c, 2, Zp, x.data(), 0.0, false).energy;
  const double qa = c.q0.q0[0], qb = c.q0.q0[1];
  const double ref = -0.5 * KE * (2.5 * qa * qa + 0.2 * qb * qb) / std::pow(20.0, 4);
  INFO("induction " << ep << " eV, -1/2 alpha F^2 " << ref << " eV");
  CHECK(ep == Approx(ref).epsilon(2e-3));
}

TEST_CASE("lrc cond global: every atom on the structure's charge, no conditioning forces, same long range", "[lrc]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  LrcSpec Lc = graph_spec(), Lg = graph_spec();
  Lg.cond_global = true;
  // Cl- 5.5 A from the water (cluster transition) and 8-10 A from Na+: per-cluster conditioning has dS != 0 here
  const auto x0 = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
  for (double Q : {0.0, 1.0, -1.0}) {
    const LrcCondition cc = lrc_condition(Lc, S, 5, ZG, x0.data(), Q, 10, 1);
    const LrcCondition cg = lrc_condition(Lg, S, 5, ZG, x0.data(), Q, 10, 1);
    REQUIRE(cg.cl.s == 2);
    CHECK(!cg.needs_cond_forces());
    for (int i = 0; i < 5; i++) {
      CHECK(cg.row_charge[i] == int(Q));
      CHECK(cg.blend[i] == 0.0f);
    }
    const LrcEnergy ec = lrc_energy(Lc, S, cc, 5, ZG, x0.data(), Q, true), eg = lrc_energy(Lg, S, cg, 5, ZG, x0.data(), Q, true);
    CHECK(eg.energy == Approx(ec.energy).epsilon(1e-12));
    for (size_t k = 0; k < ec.forces.size(); k++) CHECK(eg.forces[k] == Approx(ec.forces[k]).margin(1e-12));
    std::vector<double> G(cg.cl.s, 1.0), F(15, 0.0);
    lrc_condition_forces(Lg, S, cg, 5, ZG, x0.data(), G, F.data());
    for (double f : F) CHECK(f == 0.0);
  }
}
