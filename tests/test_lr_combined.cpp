// Combined long-range model (src/runtime/lr_combined.h), host side: integer charge per separated fragment with no charge
// on a neutral partner, the local charge's smoothness through the cluster link window, forces of the long-range term and
// of the conditioning against central differences, the reduction to the structure's charge for one cluster, and a zero
// long-range energy when no inter-cluster pair reaches lr_R - lr_t.
#include "runtime/lr_combined.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
