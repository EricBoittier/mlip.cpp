// Combined long-range model (src/runtime/lr_combined.h), host side: integer charge per separated fragment with no charge
// on a neutral partner, the local charge's smoothness through the cluster link window, forces of the long-range term and
// of the conditioning against central differences, the reduction to the structure's charge for one cluster, and a zero
// long-range energy when no inter-cluster pair reaches lr_R - lr_t.
#include "runtime/graph_model.h"
#include "runtime/lr_combined.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>

#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
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
// rcut > 0: the pair terms are cut smoothly at rcut, so that a periodic cell sums them over the images (the same
// charges as the open structure while no image is within rcut)
struct ToyQ final : LrcQSource {
  double rcut = 0;
  static double a(int z) { return z == 11 ? 0.8 : z == 17 ? -0.7 : z == 8 ? -0.5 : 0.25; }
  static double b(int zi, int zj) { return 0.05 * ((zi * 7 + zj * 3) % 5) - 0.1; }
  double t(double r, double &dt) const { // exp(-r / 2) times the cutoff
    double c = 1, dc = 0;
    if (rcut > 0) {
      if (r >= rcut) return dt = 0, 0.0;
      c = 0.5 * (1 + std::cos(M_PI * r / rcut)), dc = -0.5 * M_PI / rcut * std::sin(M_PI * r / rcut);
    }
    const double e = std::exp(-r / 2);
    dt = e * (dc - 0.5 * c);
    return e * c;
  }
  // f(i, j, d, r) over every partner j of i (periodic: its images up to two cells away), d = x_j' - x_i
  template <class F> static void pairs(int n, const double *x, const double *cell, F &&f) {
    const int m = cell ? 2 : 0;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        for (int s0 = -m; s0 <= m; s0++)
          for (int s1 = -m; s1 <= m; s1++)
            for (int s2 = -m; s2 <= m; s2++) {
              if (i == j && !s0 && !s1 && !s2) continue;
              double d[3], r2 = 0;
              for (int k = 0; k < 3; k++) {
                d[k] = x[3 * j + k] - x[3 * i + k];
                if (cell) d[k] += s0 * cell[k] + s1 * cell[3 + k] + s2 * cell[6 + k];
                r2 += d[k] * d[k];
              }
              f(i, j, d, std::sqrt(r2));
            }
  }
  static double r(const double *x, int i, int j) {
    double s = 0;
    for (int k = 0; k < 3; k++) s += (x[3 * i + k] - x[3 * j + k]) * (x[3 * i + k] - x[3 * j + k]);
    return std::sqrt(s);
  }
  std::vector<double> raw(int n, const int32_t *Z, const double *x, double Q, int, const double *cell) override {
    std::vector<double> q(n);
    for (int i = 0; i < n; i++) q[i] = a(Z[i]) + 0.1 * Q;
    pairs(n, x, cell, [&](int i, int j, const double *, double r) {
      double dt;
      q[i] += b(Z[i], Z[j]) * t(r, dt);
    });
    return q;
  }
  void raw_vjp(int n, const int32_t *Z, const double *x, double, int, const double *cell, const double *w, double *g,
               double *vir) override {
    pairs(n, x, cell, [&](int i, int j, const double *d, double r) {
      double dt;
      t(r, dt);
      const double f = w[i] * b(Z[i], Z[j]) * dt / r; // d(w_i q_i)/dd = f d
      for (int k = 0; k < 3; k++) {
        g[3 * j + k] += f * d[k], g[3 * i + k] -= f * d[k];
        if (vir && cell)
          for (int l = 0; l < 3; l++) vir[3 * k + l] += f * d[k] * d[l];
      }
    });
  }
};
LrcSpec graph_spec(double rcut = 0) {
  LrcSpec L;
  L.enabled = true, L.graph = true;
  L.fukui_Z = {1, 8, 11, 17}, L.fukui_phi = {0.3, -0.2, 1.1, 0.5};
  auto q = std::make_shared<ToyQ>();
  q->rcut = rcut;
  L.qsrc = q;
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

TEST_CASE("lrc cond_w: plateaus around integer local charge, conditioning forces still match central differences", "[lrc]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  LrcSpec L = graph_spec();
  L.cond_w = 0.6;
  const auto x0 = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
  const double Q = 0.0;
  const LrcCondition c0 = lrc_condition(L, S, 5, ZG, x0.data(), Q, 10, 1);
  for (int a = 0; a < c0.cl.s; a++) {
    const double u = c0.Qloc[a] - std::floor(c0.Qloc[a]);
    if (u < 0.2) CHECK(c0.S[a] == 0.0);
    if (u > 0.8) CHECK(c0.S[a] == 1.0);
  }
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
  double err = 0, fmax = 0;
  for (size_t k = 0; k < x0.size(); k++) {
    const double num = -fd(fS, x0, k, 1e-5);
    fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - F[k]));
  }
  INFO("cond_w conditioning force: max |F| " << fmax << ", max error " << err);
  CHECK(err < 1e-6 * std::max(1.0, fmax));
}

TEST_CASE("lrc qproj: soft charge projection keeps sum q = Q, separated / merged limits, exact forces", "[lrc]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  for (bool mp : {false, true})
    for (double w : {0.2, 0.6, 1.0}) {
      LrcSpec L = mp ? mp_spec(1.0) : graph_spec();
      L.qproj = true, L.qproj_w = w, L.a_on = 5.0, L.a_off = 6.0;
      // Cl- in the link window of the water (partly linked) and 8-10 A from Na+
      const auto x0 = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
      for (double Q : {0.0, 1.0, -1.0}) {
        auto energy = [&](const std::vector<double> &x) {
          const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, 1);
          return lrc_energy(L, S, c, 5, ZG, x.data(), Q, false).energy;
        };
        const LrcCondition c0 = lrc_condition(L, S, 5, ZG, x0.data(), Q, 10, 1);
        REQUIRE(c0.cl.s >= 2);
        const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, x0.data(), Q, true);
        double sum = 0;
        for (double v : e0.q) sum += v;
        CHECK(sum == Approx(Q).margin(1e-12));
        double fmax = 0, err = 0;
        for (size_t k = 0; k < x0.size(); k++) {
          const double num = -fd(energy, x0, k, 1e-5);
          fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
        }
        INFO("mp " << mp << " w " << w << " Q " << Q << ": max |F| " << fmax << ", max error " << err);
        CHECK(err < 1e-6 * std::max(1.0, fmax));
      }
    }
  // separated limit (no active link): every cluster total on its plateau value, the residual spread by Fukui weight;
  // a cluster whose charge-pass total is within (1 - w) / 2 of an integer ends on that integer when the others are too
  {
    LrcSpec L = graph_spec();
    L.qproj = true, L.qproj_w = 0.6;
    const int32_t Zp[] = {11, 8, 1, 1};
    const std::vector<double> x = {0, 0, 0, 30.0, 0, 0, 30.96, 0, 0, 29.76, 0.93, 0};
    const LrcCondition c = lrc_condition(L, S, 4, Zp, x.data(), 1.0, 10, 1);
    REQUIRE(c.cl.s == 2);
    REQUIRE(c.cl.trivial());
    const LrcEnergy e = lrc_energy(L, S, c, 4, Zp, x.data(), 1.0, false);
    double qa = e.q[0], qb = e.q[1] + e.q[2] + e.q[3], qa0 = c.q0.q0[0];
    INFO("charge pass Na " << qa0 << " -> projected " << qa << ", water " << qb);
    CHECK(qa + qb == Approx(1.0).margin(1e-12));
    if (std::abs(qa0 - std::round(qa0)) < 0.2) {
      CHECK(qa == Approx(std::round(qa0)).margin(1e-9));
      CHECK(qb == Approx(1.0 - std::round(qa0)).margin(1e-9));
    }
  }
  // permutation invariance: reversing the atom order permutes the projected charges
  {
    LrcSpec L = graph_spec();
    L.qproj = true, L.qproj_w = 0.6, L.a_on = 5.0, L.a_off = 6.0;
    const auto x0 = std::vector<double>{0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
    std::vector<double> xr(15);
    int32_t Zr[5];
    for (int i = 0; i < 5; i++) {
      Zr[i] = ZG[4 - i];
      for (int k = 0; k < 3; k++) xr[3 * i + k] = x0[3 * (4 - i) + k];
    }
    const LrcCondition ca = lrc_condition(L, S, 5, ZG, x0.data(), 0.0, 10, 1), cb = lrc_condition(L, S, 5, Zr, xr.data(), 0.0, 10, 1);
    const LrcEnergy ea = lrc_energy(L, S, ca, 5, ZG, x0.data(), 0.0, false), eb = lrc_energy(L, S, cb, 5, Zr, xr.data(), 0.0, false);
    CHECK(eb.energy == Approx(ea.energy).epsilon(1e-12));
    for (int i = 0; i < 5; i++) CHECK(eb.q[i] == Approx(ea.q[4 - i]).margin(1e-12));
  }
}

// ---- Fukui equilibration modes (feat/plato-lr-fukui): U / E / A / A+- / AC --------------------------------------------
namespace {
// raw charges with a spin dependence (ToyQ + 0.03 (spin - 1))
struct ToyQS final : LrcQSource {
  ToyQ q;
  std::vector<double> raw(int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell) override {
    auto v = q.raw(n, Z, x, Q, spin, cell);
    for (double &e : v) e += 0.03 * (spin - 1);
    return v;
  }
  void raw_vjp(int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell, const double *w, double *g,
               double *vir) override {
    q.raw_vjp(n, Z, x, Q, spin, cell, w, g, vir);
  }
};
// stand-in for a per-atom Fukui pass: a_i = c_Z + sum_j d_ZZ' exp(-r_ij / 3) + 0.05 Q + 0.02 (spin - 1) (+ k for f-)
struct ToyA final : LrcQSource {
  double k = 0;
  explicit ToyA(double k_ = 0) : k(k_) {}
  static double c(int z) { return z == 11 ? 0.9 : z == 17 ? -0.4 : z == 8 ? 0.3 : -0.6; }
  double d(int zi, int zj) const { return 0.3 * ((zi * 5 + zj * 11) % 7) / 7.0 - 0.1 + k; }
  std::vector<double> raw(int n, const int32_t *Z, const double *x, double Q, int spin, const double *) override {
    std::vector<double> a(n);
    for (int i = 0; i < n; i++) {
      a[i] = c(Z[i]) + 0.05 * Q + 0.02 * (spin - 1) + k;
      for (int j = 0; j < n; j++)
        if (j != i) a[i] += d(Z[i], Z[j]) * std::exp(-ToyQ::r(x, i, j) / 3);
    }
    return a;
  }
  void raw_vjp(int n, const int32_t *Z, const double *x, double, int, const double *, const double *w, double *g,
               double *) override {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        if (j == i) continue;
        const double r = ToyQ::r(x, i, j), da = -w[i] * d(Z[i], Z[j]) * std::exp(-r / 3) / 3;
        for (int q = 0; q < 3; q++) {
          const double u = (x[3 * i + q] - x[3 * j + q]) / r;
          g[3 * i + q] += da * u, g[3 * j + q] -= da * u;
        }
      }
  }
};
struct FukuiArm {
  const char *name;
  LrcSpec::FukuiMode mode;
  bool cluster;
  double pm_width;
};
const FukuiArm kArms[] = {{"U", LrcSpec::FukuiMode::Uniform, false, 0.05},   {"E", LrcSpec::FukuiMode::Element, false, 0.05},
                          {"A", LrcSpec::FukuiMode::Atom, false, 0.05},      {"A+-", LrcSpec::FukuiMode::AtomPM, false, 0.05},
                          {"A+-hard", LrcSpec::FukuiMode::AtomPM, false, 0.0}, {"UC", LrcSpec::FukuiMode::Uniform, true, 0.05},
                          {"EC", LrcSpec::FukuiMode::Element, true, 0.05},   {"AC", LrcSpec::FukuiMode::Atom, true, 0.05}};
// FTq-like settings: per-cluster conditioning (link window a_on / a_off), plateau projection 0.6, Coulomb 8 -> 10 A
LrcSpec fukui_spec(const FukuiArm &arm, double a_on = 8.0, double a_off = 10.0) {
  LrcSpec L = graph_spec();
  L.qsrc = std::make_shared<ToyQS>();
  L.qpass_neutral = true;
  L.a_on = a_on, L.a_off = a_off;
  L.qproj = true, L.qproj_w = 0.6, L.cond_w = 0.6;
  L.fukui_mode = arm.mode, L.fukui_cluster = arm.cluster, L.fukui_pm_width = arm.pm_width;
  if (L.fukui_atom()) L.fsrc = std::make_shared<ToyA>(0.0);
  if (arm.mode == LrcSpec::FukuiMode::AtomPM) L.fsrc_minus = std::make_shared<ToyA>(0.35), L.fukui_phi_minus = {0.1, 0.4, 0.2, -0.3};
  L.fukui_check();
  return L;
}
// Na+ (0), water (3.4 A) in one cluster; Cl- 8.7 A from a water H (in the 8-10 A link window) and 13 A from Na
const std::vector<double> kX8 = {0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 13.0, 1.4, 0.6};
} // namespace

TEST_CASE("lrc fukui: E mode is the per-element equilibration of bin4 bit for bit", "[lrc][fukui]") {
  const LrcSpec L = graph_spec();
  const SqeSpec S = sqe_spec();
  for (double Q : {0.0, 1.0, -1.0}) {
    const LrcQ0 o = lrc_q0(L, S, 5, ZG, kX8.data(), Q, 1);
    // the bin4 formula
    std::vector<double> q = L.qsrc->raw(5, ZG, kX8.data(), Q, 1, nullptr), wt(5);
    double sq = 0, sf = 0;
    for (int i = 0; i < 5; i++) sq += q[i], wt[i] = L.fukui(ZG[i]), sf += wt[i];
    for (int i = 0; i < 5; i++) wt[i] /= sf, q[i] += wt[i] * (Q - sq);
    for (int i = 0; i < 5; i++) CHECK(o.q0[i] == q[i]);
  }
}

TEST_CASE("lrc fukui: sum q = Q for every mode (neutral, cation, anion, spin), per cluster for AC", "[lrc][fukui]") {
  const SqeSpec S = sqe_spec();
  for (const auto &arm : kArms) {
    const LrcSpec L = fukui_spec(arm);
    for (auto [Q, spin] : std::vector<std::pair<double, int>>{{0, 1}, {1, 2}, {-1, 2}, {2, 1}, {0, 3}}) {
      const LrcQ0 o = lrc_q0(L, S, 5, ZG, kX8.data(), Q, spin);
      double sum = 0;
      for (double v : o.q0) sum += v;
      INFO(arm.name << " Q " << Q << " spin " << spin);
      CHECK(sum == Approx(Q).margin(1e-12));
    }
  }
  // AC, separated clusters (no link): every cluster ends on its plateaued total when those sum to Q, so a neutral
  // partner keeps exactly zero; global A leaks the residual onto it
  const int32_t Zs[] = {11, 8, 1, 1, 17};
  const std::vector<double> xs = {0, 0, 0, 30.0, 0, 0, 30.96, 0, 0, 29.76, 0.93, 0, 60.0, 0, 0};
  for (const auto &arm : kArms) {
    if (!arm.cluster) continue;
    LrcSpec L = fukui_spec(arm);
    const LrcQ0 raw = lrc_q0(fukui_spec(kArms[2]), S, 5, Zs, xs.data(), 0.0, 1);
    const double rn = raw.qraw[0], rw = raw.qraw[1] + raw.qraw[2] + raw.qraw[3], rc = raw.qraw[4];
    for (double Q : {0.0, 1.0, -1.0}) {
      const LrcQ0 o = lrc_q0(L, S, 5, Zs, xs.data(), Q, 1);
      const double qn = o.q0[0], qw = o.q0[1] + o.q0[2] + o.q0[3], qc = o.q0[4];
      INFO(arm.name << " Q " << Q << ": raw cluster sums " << rn << " " << rw << " " << rc << " -> " << qn << " " << qw << " "
                    << qc);
      CHECK(qn + qw + qc == Approx(Q).margin(1e-12));
      const bool plateau_sum = std::round(rn) + std::round(rw) + std::round(rc) == Q &&
                               std::abs(rn - std::round(rn)) < 0.2 && std::abs(rw - std::round(rw)) < 0.2 &&
                               std::abs(rc - std::round(rc)) < 0.2;
      if (plateau_sum) {
        CHECK(qn == Approx(std::round(rn)).margin(1e-9));
        CHECK(qw == Approx(std::round(rw)).margin(1e-9));
        CHECK(qc == Approx(std::round(rc)).margin(1e-9));
      }
    }
  }
}

TEST_CASE("lrc fukui: q0 VJP, long-range and conditioning forces match central differences in every mode", "[lrc][fukui]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  const double cw[5] = {0.7, -0.3, 0.2, 0.5, -1.1};
  for (const auto &arm : kArms) {
    const LrcSpec L = fukui_spec(arm);
    for (auto [Q, spin] : std::vector<std::pair<double, int>>{{0, 1}, {1, 2}, {-1, 2}}) {
      // q0 VJP alone: d(sum c_i q0_i)/dx
      auto cq = [&](const std::vector<double> &x) {
        const LrcQ0 o = lrc_q0(L, S, 5, ZG, x.data(), Q, spin);
        double v = 0;
        for (int i = 0; i < 5; i++) v += cw[i] * o.q0[i];
        return v;
      };
      const LrcQ0 o0 = lrc_q0(L, S, 5, ZG, kX8.data(), Q, spin);
      std::vector<double> g(15, 0.0);
      lrc_q0_vjp(L, S, 5, ZG, kX8.data(), o0, cw, g.data());
      double gmax = 0, gerr = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = fd(cq, kX8, k, 1e-5);
        gmax = std::max(gmax, std::abs(num)), gerr = std::max(gerr, std::abs(num - g[k]));
      }
      // long-range energy
      auto energy = [&](const std::vector<double> &x) {
        const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, spin);
        return lrc_energy(L, S, c, 5, ZG, x.data(), Q, false).energy;
      };
      const LrcCondition c0 = lrc_condition(L, S, 5, ZG, kX8.data(), Q, 10, spin);
      REQUIRE(c0.cl.s == 2);
      REQUIRE(!c0.cl.trivial());
      const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, kX8.data(), Q, true);
      double fmax = 0, err = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = -fd(energy, kX8, k, 1e-5);
        fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
      }
      // conditioning (dS/dx through the reference charges), direct and via the blend rows
      auto fS = [&](const std::vector<double> &x) {
        const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, spin);
        double v = 0;
        for (int i = 0; i < 5; i++) v += cw[i] * (c.k[c.cl.sn[i]] + c.S[c.cl.sn[i]]);
        return v;
      };
      std::vector<double> G(c0.cl.s, 0.0), F(15, 0.0);
      for (int i = 0; i < 5; i++) G[c0.cl.sn[i]] += cw[i];
      lrc_condition_forces(L, S, c0, 5, ZG, kX8.data(), G, F.data());
      const std::vector<double> rows = lrc_blend_rows(L, S, c0, 5, ZG, kX8.data());
      double cmax = 0, cerr = 0, rerr = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = -fd(fS, kX8, k, 1e-5);
        double fr = 0;
        for (int a = 0; a < c0.cl.s; a++) fr -= G[a] * rows[size_t(a) * 15 + k];
        cmax = std::max(cmax, std::abs(num)), cerr = std::max(cerr, std::abs(num - F[k])), rerr = std::max(rerr, std::abs(num - fr));
      }
      INFO(arm.name << " Q " << Q << " spin " << spin << " residual r " << o0.r << ": q0 VJP max " << gmax << " err " << gerr
                    << "; long range max |F| " << fmax << " err " << err << "; conditioning max " << cmax << " err " << cerr
                    << " rows " << rerr);
      std::printf("FD %-8s Q %+.0f spin %d r %+.3f | q0-VJP max %.3e err %.2e | E_lr %.4e F max %.3e err %.2e | cond max %.3e err %.2e rows %.2e\n",
                  arm.name, Q, spin, o0.r, gmax, gerr, e0.energy, fmax, err, cmax, cerr, rerr);
      CHECK(gmax > 1e-3);
      CHECK(gerr < 1e-7 * std::max(1.0, gmax));
      CHECK(err < 1e-7 * std::max(1.0, fmax));
      CHECK(cerr < 1e-6 * std::max(1.0, cmax));
      CHECK(rerr < 1e-6 * std::max(1.0, cmax));
    }
  }
}

TEST_CASE("lrc fukui: the per-atom term matters (A differs from E), rotation and permutation invariance", "[lrc][fukui]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  {
    const LrcSpec La = fukui_spec(kArms[2]), Le = fukui_spec(kArms[1]);
    std::vector<double> ga(15, 0.0), ge(15, 0.0);
    const double cw[5] = {0.7, -0.3, 0.2, 0.5, -1.1};
    const LrcQ0 oa = lrc_q0(La, S, 5, ZG, kX8.data(), 1.0, 1), oe = lrc_q0(Le, S, 5, ZG, kX8.data(), 1.0, 1);
    lrc_q0_vjp(La, S, 5, ZG, kX8.data(), oa, cw, ga.data());
    lrc_q0_vjp(Le, S, 5, ZG, kX8.data(), oe, cw, ge.data());
    double d = 0;
    for (int k = 0; k < 15; k++) d = std::max(d, std::abs(ga[k] - ge[k]));
    CHECK(d > 1e-3);
  }
  for (const auto &arm : kArms) {
    const LrcSpec L = fukui_spec(arm);
    for (double Q : {1.0, -1.0}) {
      const LrcCondition c = lrc_condition(L, S, 5, ZG, kX8.data(), Q, 10, 2);
      const LrcEnergy e = lrc_energy(L, S, c, 5, ZG, kX8.data(), Q, true);
      // rotation: energy invariant, forces rotate
      const double t = 0.9;
      const auto xr = rot(kX8, t);
      const LrcCondition cr = lrc_condition(L, S, 5, ZG, xr.data(), Q, 10, 2);
      const LrcEnergy er = lrc_energy(L, S, cr, 5, ZG, xr.data(), Q, true);
      CHECK(er.energy == Approx(e.energy).epsilon(1e-12).margin(1e-14));
      const auto fr = rot(e.forces, t);
      for (int k = 0; k < 15; k++) CHECK(er.forces[k] == Approx(fr[k]).margin(1e-11));
      for (int i = 0; i < 5; i++) CHECK(cr.q0.q0[i] == Approx(c.q0.q0[i]).margin(1e-12));
      // permutation: reversed atom order
      std::vector<double> xp(15);
      int32_t Zp[5];
      for (int i = 0; i < 5; i++) {
        Zp[i] = ZG[4 - i];
        for (int k = 0; k < 3; k++) xp[3 * i + k] = kX8[3 * (4 - i) + k];
      }
      const LrcCondition cp = lrc_condition(L, S, 5, Zp, xp.data(), Q, 10, 2);
      const LrcEnergy ep = lrc_energy(L, S, cp, 5, Zp, xp.data(), Q, true);
      INFO(arm.name << " Q " << Q);
      CHECK(ep.energy == Approx(e.energy).epsilon(1e-12).margin(1e-14));
      for (int i = 0; i < 5; i++) {
        CHECK(cp.q0.q0[i] == Approx(c.q0.q0[4 - i]).margin(1e-12));
        for (int k = 0; k < 3; k++) CHECK(ep.forces[3 * i + k] == Approx(e.forces[3 * (4 - i) + k]).margin(1e-11));
      }
    }
  }
}

TEST_CASE("lrc fukui: A+- kink at zero residual (hard switch) and its smoothing", "[lrc][fukui]") {
  const SqeSpec S = sqe_spec();
  const double cw[5] = {0.7, -0.3, 0.2, 0.5, -1.1};
  for (const FukuiArm &arm : {kArms[3], kArms[4]}) {
    const LrcSpec L = fukui_spec(arm);
    const LrcQ0 r0 = lrc_q0(L, S, 5, ZG, kX8.data(), 0.0, 1);
    const double sq = std::accumulate(r0.qraw.begin(), r0.qraw.end(), 0.0); // q pass at 0: q_raw independent of Q
    std::vector<double> gp(15, 0.0), gm(15, 0.0);
    const LrcQ0 op = lrc_q0(L, S, 5, ZG, kX8.data(), sq + 1e-7, 1), om = lrc_q0(L, S, 5, ZG, kX8.data(), sq - 1e-7, 1);
    lrc_q0_vjp(L, S, 5, ZG, kX8.data(), op, cw, gp.data());
    lrc_q0_vjp(L, S, 5, ZG, kX8.data(), om, cw, gm.data());
    double jump = 0, gmax = 0;
    for (int k = 0; k < 15; k++) jump = std::max(jump, std::abs(gp[k] - gm[k])), gmax = std::max(gmax, std::abs(gp[k]));
    std::printf("KINK %-8s |d(sum c q0)/dx| max %.3e, jump across r = 0: %.3e\n", arm.name, gmax, jump);
    if (arm.pm_width == 0) CHECK(jump > 1e-3);
    else CHECK(jump < 1e-5);
  }
}

TEST_CASE("lrc fukui: dump for the torch parity check (MLIPCPP_FUKUI_PARITY=out.json)", "[lrc][fukui]") {
  const char *out = std::getenv("MLIPCPP_FUKUI_PARITY");
  if (!out) return;
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  // no weak link (window 5 / 6 A), Cl- 8-9 A from the water (inside the Coulomb switch), 12.4 A from Na+
  const std::vector<double> x = {0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 12.3, 1.4, 0.6};
  FILE *f = std::fopen(out, "w");
  std::fprintf(f, "{\"x\": [");
  for (size_t k = 0; k < x.size(); k++) std::fprintf(f, "%s%.17g", k ? ", " : "", x[k]);
  std::fprintf(f, "], \"Z\": [11, 8, 1, 1, 17], \"cases\": [");
  bool first = true;
  for (const auto &arm : kArms) {
    const LrcSpec L = fukui_spec(arm, 5.0, 6.0);
    for (auto [Q, spin] : std::vector<std::pair<double, int>>{{0, 1}, {1, 2}, {-1, 2}, {2, 1}}) {
      const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, spin);
      REQUIRE(c.cl.s == 2);
      REQUIRE(c.cl.trivial());
      const LrcEnergy e = lrc_energy(L, S, c, 5, ZG, x.data(), Q, true);
      auto arr = [&](const std::vector<double> &v) {
        std::string s = "[";
        for (size_t k = 0; k < v.size(); k++) s += (k ? ", " : "") + std::to_string(v[k]).substr(0, 0) + [&] {
          char b[32];
          std::snprintf(b, 32, "%.17g", v[k]);
          return std::string(b);
        }();
        return s + "]";
      };
      std::fprintf(f, "%s{\"arm\": \"%s\", \"Q\": %g, \"spin\": %d, \"q0\": %s, \"q\": %s, \"E\": %.17g, \"F\": %s}", first ? "" : ", ",
                   arm.name, Q, spin, arr(e.q0).c_str(), arr(e.q).c_str(), e.energy, arr(e.forces).c_str());
      first = false;
    }
  }
  std::fprintf(f, "]}\n");
  std::fclose(f);
}
// E04: bare ions beyond the backbone cutoff (no graph edge) still move their conditioning through the cluster link.
// Host side (double): dS_a/dx of two bare ions inside the 8/10 A link window against central differences.
TEST_CASE("lrc edge-free: bare-ion conditioning dS/dx matches central differences", "[lrc]") {
  const SqeSpec S = sqe_spec();
  LrcSpec L = lrc_spec();
  L.a_on = 8.0, L.a_off = 10.0; // the R1 window
  const int32_t Z[2] = {3, 11};  // Li+ ... Na+, Q = +2
  const double u[3] = {0.92031, 0.34051, -0.19327}; // unit vector
  const double Q = 2.0;
  for (double d : {8.5, 9.185, 9.217, 9.6}) {
    const std::vector<double> x0{0, 0, 0, d * u[0], d * u[1], d * u[2]};
    const LrcCondition c0 = lrc_condition(L, S, 2, Z, x0.data(), Q, 10);
    REQUIRE(c0.cl.s == 2);
    REQUIRE(c0.needs_cond_forces());
    const std::vector<double> rows = lrc_blend_rows(L, S, c0, 2, Z, x0.data());
    double fmax = 0, err = 0;
    for (int a = 0; a < 2; a++) {
      auto Sa = [&](const std::vector<double> &x) {
        const LrcCondition c = lrc_condition(L, S, 2, Z, x.data(), Q, 10);
        return c.k[a] + c.S[a];
      };
      for (size_t k = 0; k < 6; k++) {
        const double num = fd(Sa, x0, k, 1e-5);
        fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - rows[size_t(a) * 6 + k]));
      }
    }
    INFO("d " << d << ": max |dS/dx| " << fmax << " /A, max error " << err);
    CHECK(fmax > 1e-2);
    CHECK(err < 1e-8);
  }
}

// Graph side: the whole model's force on bare Li+ ... Na+ (no edge: before the fix dE/dS came from central differences
// in S with h = 0.05, ~3 eV/A off). The network is fp32, so the tolerance is fp32-limited (0.05 eV/A at |F| ~ 13-25
// eV/A, the level the molecular E04 centres pass at), not the 1e-8 of the double host-side checks.
// MLIPCPP_TEST_LRC_GGUF: a cluster-conditioned combined model (R1 type); its charge pass from the GGUF's directory or
// MLIPCPP_LRC_QMODEL.
TEST_CASE("lrc edge-free: bare-ion forces of a conditioned model match central differences", "[lrc][gguf]") {
  const char *path = std::getenv("MLIPCPP_TEST_LRC_GGUF");
  if (!path || !*path || !std::filesystem::exists(path)) SKIP("MLIPCPP_TEST_LRC_GGUF not set");
  mlipcpp::runtime::GraphModel m;
  m.set_backend_preference(mlipcpp::BackendPreference::CPU);
  REQUIRE(m.load_from_gguf(path));
  if (!m.plato_lr().lrc.enabled || m.plato_lr().lrc.cond_global) SKIP("not a cluster-conditioned combined model");
  const int32_t Z[2] = {3, 11};
  const double u[3] = {0.92031, 0.34051, -0.19327}; // unit vector
  auto run = [&](double d, bool forces) {
    const float x[6] = {0, 0, 0, float(d * u[0]), float(d * u[1]), float(d * u[2])};
    mlipcpp::AtomicSystem sys(2, x, Z, nullptr);
    sys.set_charge(2), sys.set_spin_multiplicity(1);
    return m.predict(sys, forces);
  };
  for (double d0 : {9.185, 9.217}) {
    const mlipcpp::ModelResult r = run(d0, true);
    REQUIRE(r.forces.size() == 6);
    double Fu = 0; // force on Na+ along the separation
    for (int k = 0; k < 3; k++) Fu += double(r.forces[3 + k]) * u[k];
    double best = 1e30;
    for (double h : {6.25e-4, 3.125e-4}) {
      const double Fc = -(double(run(d0 + h, false).energy) - double(run(d0 - h, false).energy)) / (2 * h);
      best = std::min(best, std::abs(Fc - Fu));
    }
    INFO("d " << d0 << ": F.u " << Fu << " eV/A, |F_c - F.u| " << best);
    CHECK(std::abs(Fu) > 1.0);
    CHECK(best < 0.05);
  }
}

// ---- periodic cells ----------------------------------------------------------------------------------------------------
namespace {
std::vector<double> cubic(double a) { return {a, 0, 0, 0, a, 0, 0, 0, a}; }
// Na+, a water 3.4 A away, Cl- 5.5 A from the water (weak link) and 8-10 A from Na+ (inside the long-range switch)
const std::vector<double> X5 = {0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 9.6, 2.5, 0.4};
constexpr double MADELUNG_SC = 2.837297479480620; // simple cubic, tin-foil, neutralising background
struct NoQ final : LrcQSource { // a charge pass that must not run
  std::vector<double> raw(int, const int32_t *, const double *, double, int, const double *) override {
    throw std::logic_error("charge pass called");
  }
  void raw_vjp(int, const int32_t *, const double *, double, int, const double *, const double *, double *,
               double *) override {
    throw std::logic_error("charge pass called");
  }
};
} // namespace

TEST_CASE("lrc periodic: a bulk cell is one cluster, no long range, no charge pass", "[lrc][periodic]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  LrcSpec L = graph_spec();
  L.qsrc = std::make_shared<NoQ>();
  // rock salt, 2 x 2 x 2 sites 3 A apart in a 6 A cell: every atom linked to its images in all three directions
  std::vector<double> x;
  std::vector<int32_t> Z;
  for (int a = 0; a < 2; a++)
    for (int b = 0; b < 2; b++)
      for (int c = 0; c < 2; c++) x.insert(x.end(), {3.0 * a + 0.1, 3.0 * b, 3.0 * c - 0.2}), Z.push_back((a + b + c) % 2 ? 17 : 11);
  const auto h = cubic(6.0);
  for (double Q : {0.0, 1.0}) {
    const LrcCondition c = lrc_condition(L, S, 8, Z.data(), x.data(), Q, 10, 1, h.data());
    CHECK(c.cl.bulk);
    CHECK(c.cl.s == 1);
    CHECK(!c.cl.images());
    for (int i = 0; i < 8; i++) CHECK(c.row_charge[i] + double(c.blend[i]) == Approx(Q));
    const LrcEnergy e = lrc_energy(L, S, c, 8, Z.data(), x.data(), Q, true);
    CHECK(e.energy == 0.0);
    for (double f : e.forces) CHECK(f == 0.0);
    std::vector<double> F(24, 0.0), vir(9, 0.0);
    lrc_condition_forces(L, S, c, 8, Z.data(), x.data(), {1.0}, F.data(), vir.data());
    for (double f : F) CHECK(f == 0.0);
  }
}

TEST_CASE("lrc periodic: one ion in a cubic cell has the Madelung energy", "[lrc][periodic]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  const LrcSpec L = graph_spec(6.0);
  const int32_t Z1[] = {11};
  const std::vector<double> x = {1.3, -0.4, 2.2};
  for (double a : {12.0, 31.0})
    for (double Q : {1.0, -1.0}) {
      const auto h = cubic(a);
      const LrcCondition c = lrc_condition(L, S, 1, Z1, x.data(), Q, 10, 1, h.data());
      REQUIRE(c.cl.images());
      const LrcEnergy e = lrc_energy(L, S, c, 1, Z1, x.data(), Q, true);
      CHECK(e.q[0] == Approx(Q).margin(1e-14));
      CHECK(e.energy == Approx(-KE * Q * Q * MADELUNG_SC / (2 * a)).epsilon(1e-10));
      CHECK(std::abs(e.forces[0]) + std::abs(e.forces[1]) + std::abs(e.forces[2]) < 1e-10);
      // E ~ 1 / a: dE/d strain = -E on the diagonal (three directions share it), zero off it
      for (int k = 0; k < 9; k++) CHECK(e.virial[k] == Approx(k % 4 == 0 ? -e.energy / 3 : 0.0).margin(1e-10));
    }
}

TEST_CASE("lrc periodic: separated clusters tend to the open structure as the cell grows", "[lrc][periodic]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  for (bool qp : {false, true})
    for (double Q : {0.0, 1.0}) {
      LrcSpec L = graph_spec(6.0);
      L.qproj = qp;
      const LrcCondition co = lrc_condition(L, S, 5, ZG, X5.data(), Q, 10, 1);
      const double Eo = lrc_energy(L, S, co, 5, ZG, X5.data(), Q, false).energy;
      REQUIRE(co.cl.s == 2);
      std::vector<double> dev;
      for (double a : {60.0, 120.0, 240.0}) {
        const auto h = cubic(a);
        const LrcCondition c = lrc_condition(L, S, 5, ZG, X5.data(), Q, 10, 1, h.data());
        REQUIRE(c.cl.images());
        REQUIRE(c.cl.s == 2);
        for (int b = 0; b < 2; b++) CHECK(c.Qloc[b] == Approx(co.Qloc[b]).margin(1e-12));
        // less the Madelung energy of the net charge, what is left (dipole and quadrupole terms) falls as 1 / a^3
        dev.push_back(lrc_energy(L, S, c, 5, ZG, X5.data(), Q, false).energy - Eo + KE * Q * Q * MADELUNG_SC / (2 * a));
      }
      INFO("qproj " << qp << " Q " << Q << ": open " << Eo << ", deviations " << dev[0] << " " << dev[1] << " " << dev[2]);
      CHECK(std::abs(dev[0]) < 0.05);
      CHECK(std::abs(dev[1]) < std::abs(dev[0]) / 6 + 1e-9);
      CHECK(std::abs(dev[2]) < std::abs(dev[1]) / 6 + 1e-9);
    }
}

TEST_CASE("lrc periodic: separated clusters, forces and virial match central differences", "[lrc][periodic]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  const std::vector<double> h0 = {22, 0, 0, 3, 21, 0, -2, 4, 23}; // triclinic
  // strained copies: x_a -> x_a + e x_b and the same for every lattice vector
  auto strain = [](std::vector<double> v, int a, int b, double e) {
    for (size_t i = 0; i < v.size() / 3; i++) v[3 * i + a] += e * v[3 * i + b];
    return v;
  };
  for (bool qp : {false, true})
    for (double Q : {0.0, 1.0}) {
      LrcSpec L = graph_spec(6.0);
      L.qproj = qp, L.qproj_w = 0.6;
      auto energy = [&](const std::vector<double> &x, const std::vector<double> &h) {
        const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, 1, h.data());
        return lrc_energy(L, S, c, 5, ZG, x.data(), Q, false).energy;
      };
      const LrcCondition c0 = lrc_condition(L, S, 5, ZG, X5.data(), Q, 10, 1, h0.data());
      REQUIRE(c0.cl.images());
      REQUIRE(c0.cl.s == 2);
      REQUIRE(!c0.cl.trivial());
      REQUIRE(c0.needs_cond_forces());
      const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, X5.data(), Q, true);
      double sum = 0;
      for (double v : e0.q) sum += v;
      CHECK(sum == Approx(Q).margin(1e-12));
      double fmax = 0, err = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = -fd([&](const std::vector<double> &x) { return energy(x, h0); }, X5, k, 1e-5);
        fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
      }
      double vmax = 0, verr = 0;
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
          const double ep = 1e-5;
          const double num = (energy(strain(X5, a, b, ep), strain(h0, a, b, ep)) -
                              energy(strain(X5, a, b, -ep), strain(h0, a, b, -ep))) / (2 * ep);
          vmax = std::max(vmax, std::abs(num)), verr = std::max(verr, std::abs(num - e0.virial[3 * a + b]));
        }
      INFO("qproj " << qp << " Q " << Q << ": E " << e0.energy << ", max |F| " << fmax << " error " << err
                    << ", max |virial| " << vmax << " error " << verr);
      CHECK(fmax > 1e-4);
      CHECK(err < 1e-6 * std::max(1.0, fmax));
      CHECK(vmax > 1e-4);
      CHECK(verr < 1e-6 * std::max(1.0, vmax));
      // conditioning forces and virial
      const double g[5] = {0.7, -0.3, 0.2, 0.5, -1.1};
      auto fS = [&](const std::vector<double> &x, const std::vector<double> &h) {
        const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, 1, h.data());
        double v = 0;
        for (int i = 0; i < 5; i++) v += g[i] * (c.k[c.cl.sn[i]] + c.S[c.cl.sn[i]]);
        return v;
      };
      std::vector<double> G(c0.cl.s, 0.0), F(15, 0.0), vir(9, 0.0);
      for (int i = 0; i < 5; i++) G[c0.cl.sn[i]] += g[i];
      lrc_condition_forces(L, S, c0, 5, ZG, X5.data(), G, F.data(), vir.data());
      fmax = 0, err = 0, vmax = 0, verr = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = -fd([&](const std::vector<double> &x) { return fS(x, h0); }, X5, k, 1e-5);
        fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - F[k]));
      }
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
          const double ep = 1e-5;
          const double num =
              (fS(strain(X5, a, b, ep), strain(h0, a, b, ep)) - fS(strain(X5, a, b, -ep), strain(h0, a, b, -ep))) / (2 * ep);
          vmax = std::max(vmax, std::abs(num)), verr = std::max(verr, std::abs(num - vir[3 * a + b]));
        }
      INFO("conditioning: max |F| " << fmax << " error " << err << ", max |virial| " << vmax << " error " << verr);
      CHECK(fmax > 1e-3);
      CHECK(err < 1e-6 * std::max(1.0, fmax));
      CHECK(verr < 1e-6 * std::max(1.0, vmax));
    }
}

TEST_CASE("lrc periodic: atoms moved by lattice vectors change nothing", "[lrc][periodic]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  LrcSpec L = graph_spec(6.0);
  L.qproj = true;
  const std::vector<double> h = {22, 0, 0, 3, 21, 0, -2, 4, 23};
  std::vector<double> y = X5;
  for (int a = 0; a < 3; a++) y[12 + a] -= h[a], y[3 + a] += h[3 + a] + h[6 + a], y[6 + a] += h[3 + a] + h[6 + a] - h[a];
  const LrcCondition ca = lrc_condition(L, S, 5, ZG, X5.data(), 0.0, 10, 1, h.data());
  const LrcCondition cb = lrc_condition(L, S, 5, ZG, y.data(), 0.0, 10, 1, h.data());
  REQUIRE(cb.cl.s == ca.cl.s);
  const LrcEnergy ea = lrc_energy(L, S, ca, 5, ZG, X5.data(), 0.0, true), eb = lrc_energy(L, S, cb, 5, ZG, y.data(), 0.0, true);
  CHECK(eb.energy == Approx(ea.energy).epsilon(1e-10));
  for (int k = 0; k < 15; k++) CHECK(eb.forces[k] == Approx(ea.forces[k]).margin(1e-10));
  for (int k = 0; k < 9; k++) CHECK(eb.virial[k] == Approx(ea.virial[k]).margin(1e-10));
  for (int i = 0; i < 5; i++) CHECK(eb.q[i] == Approx(ea.q[i]).margin(1e-12));
}

TEST_CASE("lrc periodic: slabs, wires and clusters linked to their own images are refused", "[lrc][periodic]") {
  SqeSpec S = sqe_spec();
  S.lr_R = 10.0, S.lr_t = 2.0;
  const LrcSpec L = graph_spec(6.0);
  const int32_t Z1[] = {11};
  const std::vector<double> x1 = {0, 0, 0};
  const std::vector<double> wire = {4, 0, 0, 0, 30, 0, 0, 0, 30}, slab = {4, 0, 0, 0, 4.5, 0, 0, 0, 30};
  CHECK_THROWS(lrc_condition(L, S, 1, Z1, x1.data(), 1.0, 10, 1, wire.data()));
  CHECK_THROWS(lrc_condition(L, S, 1, Z1, x1.data(), 1.0, 10, 1, slab.data()));
  // a water whose H is 5.4 A from its own image's O: a weak link through the cell
  const int32_t Zw[] = {8, 1, 1};
  const std::vector<double> xw = {0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0}, hw = {6.5, 0, 0, 0, 30, 0, 0, 0, 30};
  CHECK_THROWS(lrc_condition(L, S, 3, Zw, xw.data(), 0.0, 10, 1, hw.data()));
  // a bulk cell is fine in any q0 mode; separated clusters need the graph mode
  const LrcSpec Lf = lrc_spec();
  CHECK_THROWS(lrc_condition(Lf, S, 5, ZG, X5.data(), 0.0, 10, 1, cubic(40.0).data()));
}

// ---- lrc_intra = ewald: the long range inside clusters and bulk cells ------------------------------------------------
namespace {
inline double sstep(double x) { return x <= 0 ? 0.0 : x >= 1 ? 1.0 : x * x * x * (x * (x * 6.0 - 15.0) + 10.0); }
double dist(const double *x, int i, int j) {
  double s = 0;
  for (int k = 0; k < 3; k++) s += (x[3 * i + k] - x[3 * j + k]) * (x[3 * i + k] - x[3 * j + k]);
  return std::sqrt(s);
}
// formal +-1 on Na / Cl (Madelung checks): no position dependence
struct FixedQ final : LrcQSource {
  std::vector<double> raw(int n, const int32_t *Z, const double *, double, int, const double *) override {
    std::vector<double> q(n, 0.0);
    for (int i = 0; i < n; i++) q[i] = Z[i] == 11 ? 1.0 : Z[i] == 17 ? -1.0 : 0.0;
    return q;
  }
  void raw_vjp(int, const int32_t *, const double *, double, int, const double *, const double *, double *,
               double *) override {}
};
// pairwise charge transfer q_i = sum_j c(Z_i, Z_j) t(r_ij), c antisymmetric: every connected group (within rcut) sums to
// zero exactly, so separated fragments carry no residual (size-consistent charges); periodic: over the images (two cells)
struct ToyT final : LrcQSource {
  double rcut = 6.0;
  static double c(int a, int b) { return 0.08 * (std::sin(0.7 * a) - std::sin(0.7 * b)); }
  double t(double r, double &dt) const {
    if (r >= rcut) return dt = 0, 0.0;
    const double cc = 0.5 * (1 + std::cos(M_PI * r / rcut)), dc = -0.5 * M_PI / rcut * std::sin(M_PI * r / rcut);
    const double e = std::exp(-r / 2);
    dt = e * (dc - 0.5 * cc);
    return e * cc;
  }
  std::vector<double> raw(int n, const int32_t *Z, const double *x, double, int, const double *cell) override {
    std::vector<double> q(n, 0.0);
    ToyQ::pairs(n, x, cell, [&](int i, int j, const double *, double r) {
      double dt;
      q[i] += c(Z[i], Z[j]) * t(r, dt);
    });
    return q;
  }
  void raw_vjp(int n, const int32_t *Z, const double *x, double, int, const double *cell, const double *w, double *g,
               double *vir) override {
    ToyQ::pairs(n, x, cell, [&](int i, int j, const double *d, double r) {
      double dt;
      t(r, dt);
      const double f = w[i] * c(Z[i], Z[j]) * dt / r;
      for (int k = 0; k < 3; k++) {
        g[3 * j + k] += f * d[k], g[3 * i + k] -= f * d[k];
        if (vir && cell)
          for (int l = 0; l < 3; l++) vir[3 * k + l] += f * d[k] * d[l];
      }
    });
  }
};
LrcSpec intra_spec(std::shared_ptr<LrcQSource> src = nullptr, double rcut = 6.0) {
  LrcSpec L = graph_spec(rcut);
  if (src) L.qsrc = std::move(src);
  L.intra = true;
  return L;
}
SqeSpec intra_sqe(double R = 10.0, double t = 2.0) {
  SqeSpec S = sqe_spec();
  S.lr_R = R, S.lr_t = t;
  return S;
}
// fourth-order central difference
double fd4(const std::function<double(const std::vector<double> &)> &f, std::vector<double> x, size_t k, double h) {
  const double x0 = x[k];
  auto at = [&](double s) {
    x[k] = x0 + s * h;
    return f(x);
  };
  return (8 * (at(1) - at(-1)) - (at(2) - at(-2))) / (12 * h);
}
std::vector<double> strained(std::vector<double> v, int a, int b, double e) {
  for (size_t i = 0; i < v.size() / 3; i++) v[3 * i + a] += e * v[3 * i + b];
  return v;
}
// rock salt: conventional cell a (8 ions), Na at even sites; reps^3 cells; optional rattle (deterministic)
void rocksalt(double a, int reps, double rattle, std::vector<double> &x, std::vector<int32_t> &Z, std::vector<double> &h) {
  x.clear(), Z.clear();
  int k = 0;
  for (int i = 0; i < 2 * reps; i++)
    for (int j = 0; j < 2 * reps; j++)
      for (int l = 0; l < 2 * reps; l++, k++) {
        x.insert(x.end(), {0.5 * a * i + rattle * std::sin(1.3 * k + 0.2), 0.5 * a * j + rattle * std::sin(2.1 * k + 0.5),
                           0.5 * a * l + rattle * std::sin(0.7 * k + 1.1)});
        Z.push_back((i + j + l) % 2 ? 17 : 11);
      }
  h = {a * reps, 0, 0, 0, a * reps, 0, 0, 0, a * reps};
}
// a water chain (one cluster at a_on = 5 A) 14 A long with an Na+ on one end and a Cl- on the other, so that pairs inside
// the one cluster lie beyond lr_R - lr_t = 8 A
std::vector<double> chain(std::vector<int32_t> &Z) {
  std::vector<double> x = {-0.3, 0.2, 0.1};
  Z = {11};
  for (int w = 0; w < 4; w++) {
    const double ox = 2.6 + 3.1 * w, oy = 0.4 * std::sin(1.7 * w), oz = 0.3 * std::cos(1.1 * w);
    x.insert(x.end(), {ox, oy, oz, ox + 0.76, oy + 0.59, oz, ox - 0.2, oy - 0.62, oz + 0.55});
    Z.insert(Z.end(), {8, 1, 1});
  }
  x.insert(x.end(), {14.9, -0.5, 0.4});
  Z.push_back(17);
  return x;
}
double madelung_brute_sub(const std::vector<double> &x, const std::vector<int32_t> &Z, const std::vector<double> &h,
                          const SqeSpec &S) { // KE sum over pairs (each image pair once) r < lr_R of (1 - T) q q / r
  const int n = int(Z.size());
  const double r0 = S.lr_R - S.lr_t;
  double e = 0;
  const int m = int(std::ceil(S.lr_R / h[0])) + 1;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int a = -m; a <= m; a++)
        for (int b = -m; b <= m; b++)
          for (int c = -m; c <= m; c++) {
            if (i == j && !a && !b && !c) continue;
            double r2 = 0;
            for (int k = 0; k < 3; k++) {
              const double d = x[3 * j + k] - x[3 * i + k] + a * h[k] + b * h[3 + k] + c * h[6 + k];
              r2 += d * d;
            }
            const double r = std::sqrt(r2);
            if (r >= S.lr_R) continue;
            const double T = r <= r0 ? 0.0 : sstep((r - r0) / S.lr_t);
            const double qi = Z[i] == 11 ? 1 : -1, qj = Z[j] == 11 ? 1 : -1;
            e += 0.5 * KE * (1 - T) * qi * qj / r; // ordered pairs: half
          }
  return e;
}
constexpr double MADELUNG_NACL = 1.747564594633182; // per ion pair, nearest-neighbour distance
} // namespace

TEST_CASE("lrc intra: the conditioning is the same with the intra Coulomb on or off", "[lrc][intra]") {
  const SqeSpec S = intra_sqe();
  std::vector<int32_t> Zc;
  const std::vector<double> xc = chain(Zc);
  for (bool qp : {false, true})
    for (double Q : {0.0, 1.0, -1.0}) {
      LrcSpec Lo = graph_spec(6.0), Li = intra_spec(nullptr, 6.0);
      Lo.qproj = Li.qproj = qp;
      for (int sys = 0; sys < 3; sys++) {
        const int n = sys == 2 ? int(Zc.size()) : 5;
        const int32_t *Z = sys == 2 ? Zc.data() : ZG;
        const double *x = sys == 0 ? kX8.data() : sys == 1 ? X5.data() : xc.data();
        const std::vector<double> h = cubic(37.0);
        for (bool per : {false, true}) {
          const LrcCondition a = lrc_condition(Lo, S, n, Z, x, Q, 10, 1, per ? h.data() : nullptr);
          const LrcCondition b = lrc_condition(Li, S, n, Z, x, Q, 10, 1, per ? h.data() : nullptr);
          REQUIRE(a.cl.s == b.cl.s);
          for (int i = 0; i < n; i++) {
            CHECK(a.row_charge[i] == b.row_charge[i]);
            CHECK(a.row_charge_hi[i] == b.row_charge_hi[i]);
            CHECK(a.blend[i] == b.blend[i]);
          }
          for (int k = 0; k < a.cl.s; k++) CHECK(a.dS[k] == b.dS[k]);
          CHECK(b.q0.q0.size() == size_t(n)); // charges always present with intra
        }
      }
    }
}

TEST_CASE("lrc intra: open structures, energy is the switched direct sum over every pair, forces exact", "[lrc][intra]") {
  const SqeSpec S = intra_sqe();
  std::vector<int32_t> Zc;
  const std::vector<double> xc = chain(Zc);
  const int nc = int(Zc.size());
  for (bool qp : {false, true})
    for (double Q : {0.0, 1.0}) {
      LrcSpec L = intra_spec();
      L.qproj = qp;
      const LrcCondition c0 = lrc_condition(L, S, nc, Zc.data(), xc.data(), Q, 10, 1);
      REQUIRE(c0.cl.s == 1); // one cluster: zero without intra
      const LrcEnergy e0 = lrc_energy(L, S, c0, nc, Zc.data(), xc.data(), Q, true);
      // the reference: KE sum_{i<j} T q q / r on the (projected) charges
      double ref = 0, sum = 0;
      for (int i = 0; i < nc; i++) {
        sum += e0.q[i];
        for (int j = i + 1; j < nc; j++) {
          const double r = dist(xc.data(), i, j);
          if (r > S.lr_R - S.lr_t) ref += KE * e0.q[i] * e0.q[j] * sstep((r - S.lr_R + S.lr_t) / S.lr_t) / r;
        }
      }
      CHECK(sum == Approx(Q).margin(1e-12));
      CHECK(std::abs(e0.energy) > 1e-3);
      CHECK(e0.energy == Approx(ref).epsilon(1e-13));
      // without intra: exactly zero
      LrcSpec Lo = L;
      Lo.intra = false;
      CHECK(lrc_energy(Lo, S, lrc_condition(Lo, S, nc, Zc.data(), xc.data(), Q, 10, 1), nc, Zc.data(), xc.data(), Q, true)
                .energy == 0.0);
      auto energy = [&](const std::vector<double> &x) {
        const LrcCondition c = lrc_condition(L, S, nc, Zc.data(), x.data(), Q, 10, 1);
        return lrc_energy(L, S, c, nc, Zc.data(), x.data(), Q, false).energy;
      };
      double fmax = 0, err = 0;
      for (size_t k = 0; k < xc.size(); k++) {
        const double num = -fd4(energy, xc, k, 1e-3);
        fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
      }
      std::printf("INTRA-FD open chain qproj %d Q %+.0f: E_lr %.6e eV, max |F| %.3e, max FD error %.2e eV/A\n", qp, Q,
                  e0.energy, fmax, err);
      CHECK(fmax > 1e-3);
      CHECK(err < 1e-8);
      // two clusters with a weak link (kX8, 8-10 A window): the reliability no longer enters the energy
      LrcSpec L8 = intra_spec();
      L8.qproj = qp, L8.a_on = 8.0, L8.a_off = 10.0;
      const LrcCondition c8 = lrc_condition(L8, S, 5, ZG, kX8.data(), Q, 10, 1);
      REQUIRE(c8.cl.s == 2);
      REQUIRE(!c8.cl.trivial());
      const LrcEnergy e8 = lrc_energy(L8, S, c8, 5, ZG, kX8.data(), Q, true);
      auto energy8 = [&](const std::vector<double> &x) {
        const LrcCondition c = lrc_condition(L8, S, 5, ZG, x.data(), Q, 10, 1);
        return lrc_energy(L8, S, c, 5, ZG, x.data(), Q, false).energy;
      };
      double fmax8 = 0, err8 = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = -fd4(energy8, kX8, k, 1e-3);
        fmax8 = std::max(fmax8, std::abs(num)), err8 = std::max(err8, std::abs(num - e8.forces[k]));
      }
      std::printf("INTRA-FD open weak link qproj %d Q %+.0f: E_lr %.6e eV, max |F| %.3e, max FD error %.2e eV/A\n", qp, Q,
                  e8.energy, fmax8, err8);
      CHECK(fmax8 > 1e-3);
      CHECK(err8 < 1e-8);
    }
}

TEST_CASE("lrc intra: rock salt with formal charges has the Madelung energy", "[lrc][intra][periodic]") {
  const LrcSpec L = intra_spec(std::make_shared<FixedQ>());
  const double a = 5.64, r0 = a / 2;
  for (int reps : {1, 2}) {
    std::vector<double> x, h;
    std::vector<int32_t> Z;
    rocksalt(a, reps, 0.0, x, Z, h);
    const int n = int(Z.size()), npair = n / 2;
    // switch below the nearest-neighbour distance: the plain Ewald (Madelung) energy
    {
      const SqeSpec S = intra_sqe(2.0, 1.0);
      const LrcCondition c = lrc_condition(L, S, n, Z.data(), x.data(), 0.0, 10, 1, h.data());
      REQUIRE(c.cl.bulk);
      const LrcEnergy e = lrc_energy(L, S, c, n, Z.data(), x.data(), 0.0, true);
      const double ref = -npair * MADELUNG_NACL * KE / r0;
      std::printf("INTRA-MADELUNG reps %d: E/pair %.12f eV, -M KE / r0 %.12f eV, rel %.2e\n", reps, e.energy / npair,
                  ref / npair, std::abs(e.energy / ref - 1));
      CHECK(e.energy == Approx(ref).epsilon(1e-10));
      for (double f : e.forces) CHECK(std::abs(f) < 1e-9);
      // E ~ 1 / a: the virial is -E / 3 on the diagonal
      for (int k = 0; k < 9; k++) CHECK(e.virial[k] == Approx(k % 4 == 0 ? -e.energy / 3 : 0.0).margin(1e-9));
    }
    // lr_R = 10 A (cell 5.64 A < lr_R: the switch subtraction includes atoms' own images): Madelung less a brute-force sum
    {
      const SqeSpec S = intra_sqe(10.0, 2.0);
      const LrcCondition c = lrc_condition(L, S, n, Z.data(), x.data(), 0.0, 10, 1, h.data());
      const LrcEnergy e = lrc_energy(L, S, c, n, Z.data(), x.data(), 0.0, true);
      const double ref = -npair * MADELUNG_NACL * KE / r0 - madelung_brute_sub(x, Z, h, S);
      std::printf("INTRA-MADELUNG-SUB reps %d: E/pair %.12f eV, Madelung - brute subtraction %.12f eV, diff %.2e eV\n",
                  reps, e.energy / npair, ref / npair, std::abs(e.energy - ref));
      CHECK(e.energy == Approx(ref).epsilon(1e-10));
    }
  }
}

TEST_CASE("lrc intra: a supercell is extensive (cells smaller than lr_R, own images inside the switch)", "[lrc][intra][periodic]") {
  const SqeSpec S = intra_sqe();
  const LrcSpec L = intra_spec(std::make_shared<ToyT>(), 6.0);
  std::vector<double> x1, h1, x2, h2;
  std::vector<int32_t> Z1, Z2;
  rocksalt(5.64, 1, 0.12, x1, Z1, h1);
  // the 2 x 2 x 2 supercell of the rattled cell
  x2.clear(), Z2.clear();
  for (int a = 0; a < 2; a++)
    for (int b = 0; b < 2; b++)
      for (int c = 0; c < 2; c++)
        for (size_t i = 0; i < Z1.size(); i++) {
          x2.insert(x2.end(), {x1[3 * i] + 5.64 * a, x1[3 * i + 1] + 5.64 * b, x1[3 * i + 2] + 5.64 * c});
          Z2.push_back(Z1[i]);
        }
  h2 = {11.28, 0, 0, 0, 11.28, 0, 0, 0, 11.28};
  for (double Q : {0.0, 1.0}) {
    const LrcCondition c1 = lrc_condition(L, S, 8, Z1.data(), x1.data(), Q, 10, 1, h1.data());
    const LrcCondition c2 = lrc_condition(L, S, 64, Z2.data(), x2.data(), 8 * Q, 10, 1, h2.data());
    REQUIRE(c1.cl.bulk);
    REQUIRE(c2.cl.bulk);
    const LrcEnergy e1 = lrc_energy(L, S, c1, 8, Z1.data(), x1.data(), Q, true);
    const LrcEnergy e2 = lrc_energy(L, S, c2, 64, Z2.data(), x2.data(), 8 * Q, true);
    std::printf("INTRA-SUPERCELL Q %+.0f: E(1x1x1) %.10f, E(2x2x2) / 8 %.10f eV, diff %.2e\n", Q, e1.energy, e2.energy / 8,
                std::abs(e2.energy - 8 * e1.energy));
    CHECK(std::abs(e1.energy) > 1e-3);
    CHECK(e2.energy == Approx(8 * e1.energy).epsilon(1e-10));
    double ferr = 0, verr = 0;
    for (int k = 0; k < 64 * 3; k++) ferr = std::max(ferr, std::abs(e2.forces[k] - e1.forces[k % 24]));
    for (int k = 0; k < 9; k++) verr = std::max(verr, std::abs(e2.virial[k] - 8 * e1.virial[k]));
    CHECK(ferr < 1e-9);
    CHECK(verr < 1e-8);
    for (int i = 0; i < 64; i++) CHECK(e2.q[i] == Approx(e1.q[i % 8]).margin(1e-12));
  }
}

TEST_CASE("lrc intra: periodic forces and virial match differences (bulk and separated clusters)", "[lrc][intra][periodic]") {
  const SqeSpec S = intra_sqe();
  std::vector<double> xb, hb;
  std::vector<int32_t> Zb;
  rocksalt(5.64, 1, 0.12, xb, Zb, hb);
  hb[3] = 0.4, hb[7] = -0.3; // triclinic
  struct Case {
    const char *name;
    std::vector<double> x, h;
    std::vector<int32_t> Z;
    std::shared_ptr<LrcQSource> src;
  };
  std::vector<Case> cases = {
      {"bulk NaCl ToyT", xb, hb, Zb, std::make_shared<ToyT>()},
      {"bulk NaCl ToyQ", xb, hb, Zb, nullptr},
      {"separated X5 ToyQ", X5, {22, 0, 0, 3, 21, 0, -2, 4, 23}, std::vector<int32_t>(ZG, ZG + 5), nullptr}};
  for (const Case &cs : cases)
    for (bool qp : {false, true})
      for (double Q : {0.0, 1.0}) {
        LrcSpec L = intra_spec(cs.src, 5.0);
        L.qproj = qp;
        const int n = int(cs.Z.size());
        auto energy = [&](const std::vector<double> &x, const std::vector<double> &h) {
          const LrcCondition c = lrc_condition(L, S, n, cs.Z.data(), x.data(), Q, 10, 1, h.data());
          return lrc_energy(L, S, c, n, cs.Z.data(), x.data(), Q, false).energy;
        };
        const LrcCondition c0 = lrc_condition(L, S, n, cs.Z.data(), cs.x.data(), Q, 10, 1, cs.h.data());
        const LrcEnergy e0 = lrc_energy(L, S, c0, n, cs.Z.data(), cs.x.data(), Q, true);
        double sum = 0;
        for (double v : e0.q) sum += v;
        CHECK(sum == Approx(Q).margin(1e-12));
        double fmax = 0, err = 0, vmax = 0, verr = 0;
        for (size_t k = 0; k < cs.x.size(); k++) {
          const double num = -fd4([&](const std::vector<double> &x) { return energy(x, cs.h); }, cs.x, k, 1e-3);
          fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
        }
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) {
            const double ep = 1e-4;
            auto at = [&](double s) { return energy(strained(cs.x, a, b, s * ep), strained(cs.h, a, b, s * ep)); };
            const double num = (8 * (at(1) - at(-1)) - (at(2) - at(-2))) / (12 * ep);
            vmax = std::max(vmax, std::abs(num)), verr = std::max(verr, std::abs(num - e0.virial[3 * a + b]));
          }
        std::printf("INTRA-FD %-18s qproj %d Q %+.0f: E_lr %+.6e eV, max |F| %.3e err %.2e eV/A, max |vir| %.3e err %.2e eV\n",
                    cs.name, qp, Q, e0.energy, fmax, err, vmax, verr);
        CHECK(fmax > 1e-3);
        CHECK(err < 1e-8);
        CHECK(vmax > 1e-3);
        CHECK(verr < 1e-8);
      }
}

TEST_CASE("lrc intra: a cluster in a growing box tends to the open direct sum (Ewald vs direct)", "[lrc][intra][periodic]") {
  const SqeSpec S = intra_sqe();
  std::vector<int32_t> Zc;
  const std::vector<double> xc = chain(Zc);
  const int n = int(Zc.size());
  for (bool qp : {false, true})
    for (double Q : {0.0, 1.0}) {
      LrcSpec L = intra_spec(nullptr, 6.0);
      L.qproj = qp;
      const LrcCondition co = lrc_condition(L, S, n, Zc.data(), xc.data(), Q, 10, 1);
      const double Eo = lrc_energy(L, S, co, n, Zc.data(), xc.data(), Q, false).energy;
      std::vector<double> dev;
      for (double a : {40.0, 80.0, 160.0, 320.0}) {
        const auto h = cubic(a);
        const LrcCondition c = lrc_condition(L, S, n, Zc.data(), xc.data(), Q, 10, 1, h.data());
        REQUIRE(c.cl.images());
        // the neutralising background's Madelung term of the net charge is removed; the rest falls as 1 / a^3
        dev.push_back(lrc_energy(L, S, c, n, Zc.data(), xc.data(), Q, false).energy - Eo + KE * Q * Q * MADELUNG_SC / (2 * a));
      }
      std::printf("INTRA-CONV qproj %d Q %+.0f: E_open %.8f eV; E_box - E_open + Madelung(Q) at L = 40 / 80 / 160 / 320 A: "
                  "%.3e %.3e %.3e %.3e (ratios %.2f %.2f %.2f)\n",
                  qp, Q, Eo, dev[0], dev[1], dev[2], dev[3], dev[0] / dev[1], dev[1] / dev[2], dev[2] / dev[3]);
      for (int k = 0; k < 3; k++) {
        CHECK(std::abs(dev[k + 1]) < std::abs(dev[k]) / 6 + 1e-10);
      }
      CHECK(std::abs(dev[2] / dev[3]) == Approx(8.0).margin(0.25)); // 1 / L^3
    }
}

TEST_CASE("lrc intra: rotation, translation, permutation and lattice-vector invariance", "[lrc][intra]") {
  const SqeSpec S = intra_sqe();
  std::vector<int32_t> Zc;
  const std::vector<double> xc = chain(Zc);
  const int n = int(Zc.size());
  for (bool qp : {false, true}) {
    LrcSpec L = intra_spec(nullptr, 6.0);
    L.qproj = qp;
    const double Q = 1.0;
    const LrcEnergy e = lrc_energy(L, S, lrc_condition(L, S, n, Zc.data(), xc.data(), Q, 10, 1), n, Zc.data(), xc.data(), Q, true);
    // rotation
    const auto xr = rot(xc, 0.9);
    const LrcEnergy er = lrc_energy(L, S, lrc_condition(L, S, n, Zc.data(), xr.data(), Q, 10, 1), n, Zc.data(), xr.data(), Q, true);
    CHECK(er.energy == Approx(e.energy).epsilon(1e-12));
    const auto fr = rot(e.forces, 0.9);
    for (int k = 0; k < 3 * n; k++) CHECK(er.forces[k] == Approx(fr[k]).margin(1e-11));
    // translation
    std::vector<double> xt = xc;
    for (int i = 0; i < n; i++) xt[3 * i] += 3.7, xt[3 * i + 1] -= 11.2, xt[3 * i + 2] += 0.4;
    const LrcEnergy et = lrc_energy(L, S, lrc_condition(L, S, n, Zc.data(), xt.data(), Q, 10, 1), n, Zc.data(), xt.data(), Q, true);
    CHECK(et.energy == Approx(e.energy).epsilon(1e-12));
    for (int k = 0; k < 3 * n; k++) CHECK(et.forces[k] == Approx(e.forces[k]).margin(1e-11));
    // permutation (reversed order)
    std::vector<double> xp(3 * n);
    std::vector<int32_t> Zp(n);
    for (int i = 0; i < n; i++) {
      Zp[i] = Zc[n - 1 - i];
      for (int k = 0; k < 3; k++) xp[3 * i + k] = xc[3 * (n - 1 - i) + k];
    }
    const LrcEnergy ep = lrc_energy(L, S, lrc_condition(L, S, n, Zp.data(), xp.data(), Q, 10, 1), n, Zp.data(), xp.data(), Q, true);
    CHECK(ep.energy == Approx(e.energy).epsilon(1e-12));
    for (int i = 0; i < n; i++)
      for (int k = 0; k < 3; k++) CHECK(ep.forces[3 * i + k] == Approx(e.forces[3 * (n - 1 - i) + k]).margin(1e-11));
    // periodic bulk: atoms moved by lattice vectors, the whole cell rotated with its contents, permuted atoms
    std::vector<double> xb, hb;
    std::vector<int32_t> Zb;
    rocksalt(5.64, 1, 0.12, xb, Zb, hb);
    hb[3] = 0.4;
    LrcSpec Lb = intra_spec(std::make_shared<ToyT>(), 5.0);
    Lb.qproj = qp;
    auto run = [&](const std::vector<double> &x, const std::vector<int32_t> &Z, const std::vector<double> &h) {
      return lrc_energy(Lb, S, lrc_condition(Lb, S, 8, Z.data(), x.data(), 0.0, 10, 1, h.data()), 8, Z.data(), x.data(), 0.0, true);
    };
    const LrcEnergy b0 = run(xb, Zb, hb);
    std::vector<double> xs = xb;
    for (int a = 0; a < 3; a++) xs[a] += hb[a] - hb[6 + a], xs[9 + a] -= hb[3 + a];
    const LrcEnergy b1 = run(xs, Zb, hb);
    CHECK(b1.energy == Approx(b0.energy).epsilon(1e-11));
    for (int k = 0; k < 24; k++) CHECK(b1.forces[k] == Approx(b0.forces[k]).margin(1e-10));
    const LrcEnergy b2 = run(rot(xb, 0.7), Zb, rot(hb, 0.7));
    CHECK(b2.energy == Approx(b0.energy).epsilon(1e-11));
    const auto fb = rot(b0.forces, 0.7);
    for (int k = 0; k < 24; k++) CHECK(b2.forces[k] == Approx(fb[k]).margin(1e-10));
    std::vector<double> xq(24);
    std::vector<int32_t> Zq(8);
    for (int i = 0; i < 8; i++) {
      Zq[i] = Zb[7 - i];
      for (int k = 0; k < 3; k++) xq[3 * i + k] = xb[3 * (7 - i) + k];
    }
    const LrcEnergy b3 = run(xq, Zq, hb);
    CHECK(b3.energy == Approx(b0.energy).epsilon(1e-11));
    for (int k = 0; k < 9; k++) CHECK(b3.virial[k] == Approx(b0.virial[k]).margin(1e-10));
    std::printf("INTRA-INV qproj %d: open E %.10f (rot %.1e, trans %.1e, perm %.1e); bulk E %.10f (lattice %.1e, rot %.1e, perm %.1e)\n",
                qp, e.energy, std::abs(er.energy - e.energy), std::abs(et.energy - e.energy), std::abs(ep.energy - e.energy),
                b0.energy, std::abs(b1.energy - b0.energy), std::abs(b2.energy - b0.energy), std::abs(b3.energy - b0.energy));
  }
}

TEST_CASE("lrc intra: size consistency to 100 A (separable charges)", "[lrc][intra]") {
  const SqeSpec S = intra_sqe();
  // fragment A: the water chain without its ions (neutral by ToyT); fragment B: an Na+ ... Cl- pair 9 A apart
  std::vector<int32_t> Zc;
  const std::vector<double> xc = chain(Zc);
  std::vector<double> xA(xc.begin() + 3, xc.end() - 3);
  std::vector<int32_t> ZA(Zc.begin() + 1, Zc.end() - 1);
  for (int src = 0; src < 2; src++) {
    // src 0: ToyT, neutral fragments; src 1: formal charges, Na+ and Cl- as two charged fragments
    const LrcSpec L = intra_spec(src == 0 ? std::shared_ptr<LrcQSource>(std::make_shared<ToyT>())
                                          : std::shared_ptr<LrcQSource>(std::make_shared<FixedQ>()));
    std::vector<double> xB = src == 0 ? std::vector<double>{0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0, 9.0, 0.5, 0.2, 9.9, 0.4, 0.3}
                                      : std::vector<double>{0, 0, 0};
    std::vector<int32_t> ZB = src == 0 ? std::vector<int32_t>{8, 1, 1, 11, 17} : std::vector<int32_t>{11};
    if (src == 1) xA = {0, 0, 0}, ZA = {17};
    const double QA = 0, QB = src == 0 ? 0.0 : 1.0, QAx = src == 0 ? 0.0 : -1.0;
    auto E = [&](const std::vector<double> &x, const std::vector<int32_t> &Z, double Q) {
      const int n = int(Z.size());
      return lrc_energy(L, S, lrc_condition(L, S, n, Z.data(), x.data(), Q, 10, 1), n, Z.data(), x.data(), Q, false);
    };
    const LrcEnergy eA = E(xA, ZA, QA + QAx), eB = E(xB, ZB, QB);
    for (double d : {20.0, 50.0, 100.0}) {
      std::vector<double> x = xA;
      std::vector<int32_t> Z = ZA;
      for (size_t i = 0; i < ZB.size(); i++) x.insert(x.end(), {xB[3 * i] + d, xB[3 * i + 1] + 0.3 * d, xB[3 * i + 2] - 0.2 * d}), Z.push_back(ZB[i]);
      const LrcEnergy eAB = E(x, Z, QA + QAx + QB);
      // the exact cross term on the fragments' own charges (every cross pair beyond lr_R: T = 1)
      double cross = 0;
      const int nA = int(ZA.size());
      for (int i = 0; i < nA; i++)
        for (size_t j = 0; j < ZB.size(); j++) cross += KE * eA.q[i] * eB.q[j] / dist(x.data(), i, nA + int(j));
      const double gap = eAB.energy - eA.energy - eB.energy;
      double dq = 0;
      for (int i = 0; i < nA; i++) dq = std::max(dq, std::abs(eAB.q[i] - eA.q[i]));
      for (size_t j = 0; j < ZB.size(); j++) dq = std::max(dq, std::abs(eAB.q[nA + j] - eB.q[j]));
      std::printf("INTRA-SIZE %s d %5.1f A: E_AB - E_A - E_B %+.6e eV, exact cross term %+.6e eV, diff %.1e; max |dq| %.1e\n",
                  src == 0 ? "neutral (ToyT)" : "ions (formal)", d, gap, cross, std::abs(gap - cross), dq);
      CHECK(dq < 1e-12);
      CHECK(gap == Approx(cross).epsilon(1e-10).margin(1e-12));
    }
  }
}

TEST_CASE("lrc intra: continuity through the switch (8 and 10 A) inside one cluster", "[lrc][intra]") {
  const SqeSpec S = intra_sqe();
  const LrcSpec L = intra_spec(std::make_shared<FixedQ>());
  // Na+ ... O ... O ... Cl- on a line: one cluster at a_on = 5 A while the ions sweep through the switch
  const int32_t Z[] = {11, 8, 8, 17};
  auto E = [&](double d, bool forces, double *F = nullptr) {
    const std::vector<double> x = {0, 0, 0, d / 3, 0.3, 0, 2 * d / 3, -0.3, 0, d, 0, 0};
    const LrcCondition c = lrc_condition(L, S, 4, Z, x.data(), 0.0, 10, 1);
    REQUIRE(c.cl.s == 1);
    const LrcEnergy e = lrc_energy(L, S, c, 4, Z, x.data(), 0.0, forces);
    if (F) *F = e.forces[9];
    return e.energy;
  };
  double worst = 0;
  for (double d0 : {8.0, 10.0}) {
    for (double eps : {1e-4, 1e-6}) {
      double Fm, Fp;
      const double Em = E(d0 - eps, true, &Fm), Ep = E(d0 + eps, true, &Fp);
      worst = std::max(worst, std::abs(Ep - Em) / eps);
      CHECK(std::abs(Ep - Em) < 2.5 * eps); // |dE/dd| <= KE / 8^2 = 0.23 eV/A
      CHECK(std::abs(Fp - Fm) < 1e-3);     // C1: the force is continuous (smootherstep: C2)
    }
    // the force against differences right at the switch ends
    const double h = 1e-4;
    double F0;
    E(d0, true, &F0);
    const double num = -(E(d0 + h, false) - E(d0 - h, false)) / (2 * h);
    std::printf("INTRA-SWITCH d %.1f A: E %.10f eV, F_Cl %.8f eV/A, central difference %.8f, |diff| %.1e\n", d0, E(d0, false),
                F0, num, std::abs(num - F0));
    CHECK(std::abs(num - F0) < 1e-7);
  }
  CHECK(E(7.99, false) == 0.0);
  CHECK(E(10.0, false) == Approx(-KE / 10.0).epsilon(1e-12));
}

TEST_CASE("lrc intra: unsupported combinations throw", "[lrc][intra]") {
  const SqeSpec S = intra_sqe();
  LrcSpec Lf = lrc_spec();
  Lf.intra = true;
  CHECK_THROWS(lrc_condition(Lf, S, 5, ZG, kX8.data(), 0.0, 10, 1));
  LrcSpec Lm = intra_spec();
  Lm.pol = true, Lm.pol_Z = {8}, Lm.pol_alpha = {1.0};
  CHECK_THROWS(lrc_condition(Lm, S, 5, ZG, kX8.data(), 0.0, 10, 1));
}

// ---- lrc_slab: slabs (2D-periodic, open along one lattice direction) --------------------------------------------------
namespace {
// partner images for the toy charge passes: two cells in every periodic (nonzero) row, none along a zero (open) row
template <class F> void pairs_pbc(int n, const double *x, const double *cell, F &&f) {
  int m[3] = {0, 0, 0};
  for (int k = 0; k < 3 && cell; k++) m[k] = (cell[3 * k] || cell[3 * k + 1] || cell[3 * k + 2]) ? 2 : 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int s0 = -m[0]; s0 <= m[0]; s0++)
        for (int s1 = -m[1]; s1 <= m[1]; s1++)
          for (int s2 = -m[2]; s2 <= m[2]; s2++) {
            if (i == j && !s0 && !s1 && !s2) continue;
            double d[3], r2 = 0;
            for (int k = 0; k < 3; k++) {
              d[k] = x[3 * j + k] - x[3 * i + k];
              if (cell) d[k] += s0 * cell[k] + s1 * cell[3 + k] + s2 * cell[6 + k];
              r2 += d[k] * d[k];
            }
            f(i, j, d, std::sqrt(r2));
          }
}
// ToyQ (position-dependent, smooth, cut at rcut) over the slab's images
struct ToyQ2 final : LrcQSource {
  ToyQ t;
  explicit ToyQ2(double rcut) { t.rcut = rcut; }
  std::vector<double> raw(int n, const int32_t *Z, const double *x, double Q, int, const double *cell) override {
    std::vector<double> q(n);
    for (int i = 0; i < n; i++) q[i] = ToyQ::a(Z[i]) + 0.1 * Q;
    pairs_pbc(n, x, cell, [&](int i, int j, const double *, double r) {
      double dt;
      q[i] += ToyQ::b(Z[i], Z[j]) * t.t(r, dt);
    });
    return q;
  }
  void raw_vjp(int n, const int32_t *Z, const double *x, double, int, const double *cell, const double *w, double *g,
               double *vir) override {
    pairs_pbc(n, x, cell, [&](int i, int j, const double *d, double r) {
      double dt;
      t.t(r, dt);
      const double f = w[i] * ToyQ::b(Z[i], Z[j]) * dt / r;
      for (int k = 0; k < 3; k++) {
        g[3 * j + k] += f * d[k], g[3 * i + k] -= f * d[k];
        if (vir && cell)
          for (int l = 0; l < 3; l++) vir[3 * k + l] += f * d[k] * d[l];
      }
    });
  }
};
LrcSpec slab_spec(LrcSpec::Slab mode, std::shared_ptr<LrcQSource> src, bool intra = true) {
  LrcSpec L = graph_spec(5.0);
  L.qsrc = src ? std::move(src) : std::make_shared<ToyQ2>(5.0);
  L.intra = intra, L.slab = mode;
  return L;
}
// rock-salt (100) slab: in-plane conventional a x a (4 ions per layer, layers a / 2 apart), the open row zero; rattle
// deterministic; z0 the first layer; skew shears the in-plane cell (triclinic)
void nacl_slab(double a, int layers, double rattle, double z0, std::vector<double> &x, std::vector<int32_t> &Z,
               std::vector<double> &h, double skew = 0) {
  x.clear(), Z.clear();
  int k = 0;
  for (int l = 0; l < layers; l++)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++, k++) {
        x.insert(x.end(), {0.5 * a * i + skew * 0.5 * a * j + rattle * std::sin(1.3 * k + 0.2),
                           0.5 * a * j + rattle * std::sin(2.1 * k + 0.5), z0 + 0.5 * a * l + rattle * std::sin(0.7 * k + 1.1)});
        Z.push_back((i + j + l) % 2 ? 17 : 11);
      }
  h = {a, 0, 0, skew * a, a, 0, 0, 0, 0};
}
struct SlabRun {
  LrcCondition c;
  LrcEnergy e;
};
SlabRun slab_run(const LrcSpec &L, const SqeSpec &S, const std::vector<double> &x, const std::vector<int32_t> &Z,
                 const std::vector<double> &h, double Q, bool forces) {
  const int n = int(Z.size());
  SlabRun r{lrc_condition(L, S, n, Z.data(), x.data(), Q, 10, 1, h.data()), {}};
  r.e = lrc_energy(L, S, r.c, n, Z.data(), x.data(), Q, forces);
  return r;
}
double slab_E(const LrcSpec &L, const SqeSpec &S, const std::vector<double> &x, const std::vector<int32_t> &Z,
              const std::vector<double> &h, double Q) {
  return slab_run(L, S, x, Z, h, Q, false).e.energy;
}
} // namespace

TEST_CASE("lrc slab: without lrc_slab a slab still throws; wires always throw", "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  std::vector<double> x, h;
  std::vector<int32_t> Z;
  nacl_slab(5.64, 4, 0.0, 0.0, x, Z, h);
  for (bool intra : {false, true}) {
    LrcSpec L = slab_spec(LrcSpec::Slab::None, std::make_shared<FixedQ>(), intra);
    CHECK_THROWS(lrc_condition(L, S, int(Z.size()), Z.data(), x.data(), 0.0, 10, 1, h.data()));
    L.slab = LrcSpec::Slab::YB;
    const std::vector<double> wire = {5.64, 0, 0, 0, 0, 0, 0, 0, 0};
    CHECK_THROWS(lrc_condition(L, S, int(Z.size()), Z.data(), x.data(), 0.0, 10, 1, wire.data()));
    // (T, T, T) with vacuum and no gap rule: layered, so only with lrc_intra
    const std::vector<double> h3 = {5.64, 0, 0, 0, 5.64, 0, 0, 0, 60};
    if (intra) CHECK_NOTHROW(slab_E(L, S, x, Z, h3, 0.0));
    else CHECK_THROWS(lrc_condition(L, S, int(Z.size()), Z.data(), x.data(), 0.0, 10, 1, h3.data()));
  }
}

TEST_CASE("lrc slab: the exact 2D sum and Yeh-Berkowitz do not depend on the Ewald splitting (neutral, charged)",
          "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  std::vector<double> x, h;
  std::vector<int32_t> Z;
  nacl_slab(5.64, 4, 0.15, 1.0, x, Z, h, 0.2);
  for (auto mode : {LrcSpec::Slab::Ewald2D, LrcSpec::Slab::YB})
    for (double Q : {0.0, 1.0, -2.0}) {
      std::vector<double> Es;
      for (double rc : {10.0, 12.5, 16.0}) {
        LrcSpec L = slab_spec(mode, std::make_shared<FixedQ>());
        L.slab_rc = rc;
        Es.push_back(slab_E(L, S, x, Z, h, Q));
      }
      std::printf("SLAB-ALPHA %s Q %+.0f: E %.12f, spread over rc 10 / 12.5 / 16 A: %.2e %.2e eV\n",
                  mode == LrcSpec::Slab::YB ? "yb     " : "ewald2d", Q, Es[0], Es[1] - Es[0], Es[2] - Es[0]);
      CHECK(std::abs(Es[1] - Es[0]) < 1e-10);
      CHECK(std::abs(Es[2] - Es[0]) < 1e-10);
    }
}

TEST_CASE("lrc slab: Yeh-Berkowitz (automatic vacuum) equals the exact 2D sum; classic YB converges with the vacuum",
          "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  std::vector<double> x, h;
  std::vector<int32_t> Z;
  nacl_slab(5.64, 5, 0.15, 1.0, x, Z, h, 0.2); // 5 layers: a net dipole along z, and position-dependent charges
  const int n = int(Z.size());
  for (double Q : {0.0, 1.0}) {
    const double E2 = slab_E(slab_spec(LrcSpec::Slab::Ewald2D, nullptr), S, x, Z, h, Q);
    const double Ey = slab_E(slab_spec(LrcSpec::Slab::YB, nullptr), S, x, Z, h, Q);
    // classic Yeh-Berkowitz: a (T, T, T) cell of height thickness + vacuum, the slab found by the gap rule
    double lo = 1e9, hi = -1e9;
    for (int i = 0; i < n; i++) lo = std::min(lo, x[3 * i + 2]), hi = std::max(hi, x[3 * i + 2]);
    std::vector<double> dev;
    for (double vac : {11.0, 15.0, 20.0, 30.0, 45.0}) {
      LrcSpec L = slab_spec(LrcSpec::Slab::YB, nullptr);
      L.slab_gap = 10.5;
      std::vector<double> h3 = h;
      h3[8] = hi - lo + vac;
      std::vector<double> xs = x;
      for (int i = 0; i < n; i++) xs[3 * i + 2] += i % 3 == 0 ? h3[8] : 0.0; // some atoms wrapped across the cell
      const SlabRun r = slab_run(L, S, xs, Z, h3, Q, false);
      REQUIRE(r.c.cl.open == 2);
      dev.push_back(r.e.energy - E2);
    }
    std::printf("SLAB-YB Q %+.0f: E_2D %.10f eV, YB(auto) - 2D %.2e; classic YB - 2D at vacuum 11 / 15 / 20 / 30 / 45 A: "
                "%.2e %.2e %.2e %.2e %.2e\n",
                Q, E2, Ey - E2, dev[0], dev[1], dev[2], dev[3], dev[4]);
    CHECK(std::abs(Ey - E2) < 1e-9);
    CHECK(std::abs(dev[4]) < 1e-9);
    for (int k = 0; k + 1 < 4; k++) CHECK(std::abs(dev[k + 1]) < std::abs(dev[k]) + 1e-12);
  }
}

TEST_CASE("lrc slab: a finite cluster in a growing in-plane cell tends to the open direct sum", "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  std::vector<int32_t> Zc;
  const std::vector<double> xc = chain(Zc);
  const int n = int(Zc.size());
  for (auto mode : {LrcSpec::Slab::YB, LrcSpec::Slab::Ewald2D})
    for (double Q : {0.0, 1.0}) {
      const LrcSpec L = slab_spec(mode, std::make_shared<ToyQ2>(6.0));
      const LrcSpec Li = slab_spec(mode, std::make_shared<FixedQ>());
      const LrcCondition co = lrc_condition(L, S, n, Zc.data(), xc.data(), Q, 10, 1);
      const double Eo = lrc_energy(L, S, co, n, Zc.data(), xc.data(), Q, false).energy;
      std::vector<double> dev;
      const int32_t Z1[] = {11};
      for (double a : {40.0, 80.0, 160.0, 320.0}) {
        const std::vector<double> h = {a, 0, 0, 0, a, 0, 0, 0, 0};
        const SlabRun r = slab_run(L, S, xc, Zc, h, Q, false);
        REQUIRE(r.c.cl.images());
        // the 2D lattice energy of the net charge (one point charge per cell, the h = 0 term dropped: ~ Q^2 / L)
        const double Eq = Q == 0 ? 0.0 : slab_E(Li, S, {0.3, 0.1, 0.2}, std::vector<int32_t>(Z1, Z1 + 1), h, 1.0);
        dev.push_back(r.e.energy - Eo - Q * Q * Eq);
      }
      std::printf("SLAB-OPEN %s Q %+.0f: E_open %.8f; E_slab - E_open - Q^2 E_point at L = 40 / 80 / 160 / 320 A: "
                  "%.3e %.3e %.3e %.3e (ratios %.2f %.2f %.2f)\n",
                  mode == LrcSpec::Slab::YB ? "yb     " : "ewald2d", Q, Eo, dev[0], dev[1], dev[2], dev[3],
                  dev[0] / dev[1], dev[1] / dev[2], dev[2] / dev[3]);
      // neutral: dipole-dipole lattice sums, 1 / L^3; charged, less the point-charge lattice: 1 / L^2 or faster
      for (int k = 0; k < 3; k++) CHECK(std::abs(dev[k + 1]) < std::abs(dev[k]) / 3.5 + 1e-10);
    }
}

TEST_CASE("lrc slab: forces and virial match differences (slab, separated clusters, slab beside a molecule)",
          "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  std::vector<double> xs, hs;
  std::vector<int32_t> Zs;
  nacl_slab(5.64, 3, 0.12, 0.5, xs, Zs, hs, 0.15);
  // slab + Na+ 7.5 A above the top layer (outside a_off = 6: its own cluster; within the 8-10 A switch of some ions) +
  // a water 5.5 A from the Na+ (weak link) -> layered
  std::vector<double> xm = xs;
  std::vector<int32_t> Zm = Zs;
  const double top = 0.5 + 5.64 + 7.5;
  xm.insert(xm.end(), {1.1, 0.7, top, 1.3, 0.9, top + 5.5, 2.2, 1.0, top + 5.6, 1.0, 1.6, top + 5.9});
  Zm.insert(Zm.end(), {11, 8, 1, 1});
  struct Case {
    const char *name;
    std::vector<double> x, h;
    std::vector<int32_t> Z;
    bool intra;
  };
  const std::vector<double> hsep = {21, 0, 0, 3, 22, 0, 0, 0, 0};
  std::vector<Case> cases = {{"slab (bulk-like)", xs, hs, Zs, true},
                             {"separated X5", X5, hsep, std::vector<int32_t>(ZG, ZG + 5), true},
                             {"separated X5 inter", X5, hsep, std::vector<int32_t>(ZG, ZG + 5), false},
                             {"slab + molecules", xm, hs, Zm, true}};
  for (auto mode : {LrcSpec::Slab::YB, LrcSpec::Slab::Ewald2D})
    for (const Case &cs : cases)
      for (bool qp : {false, true})
        for (double Q : {0.0, 1.0}) {
          LrcSpec L = slab_spec(mode, nullptr, cs.intra);
          L.qproj = qp;
          const int n = int(cs.Z.size());
          auto energy = [&](const std::vector<double> &x, const std::vector<double> &h) { return slab_E(L, S, x, cs.Z, h, Q); };
          const SlabRun r0 = slab_run(L, S, cs.x, cs.Z, cs.h, Q, true);
          REQUIRE(r0.c.cl.slab());
          double sum = 0;
          for (double v : r0.e.q) sum += v;
          CHECK(sum == Approx(Q).margin(1e-12));
          double fmax = 0, err = 0, vmax = 0, verr = 0;
          for (size_t k = 0; k < cs.x.size(); k++) {
            // steps: the toy charge pass makes the 5th derivatives large here; the differences converge as h^4
            // (virial: 4e-4 / 1e-4 / 2.5e-5 -> 2e-5 / 4e-8 / 1e-9 eV), so the steps are smaller than in the 3D tests
            const double hfd = std::getenv("SLAB_H") ? std::atof(std::getenv("SLAB_H")) : 5e-4;
            const double num = -fd4([&](const std::vector<double> &x) { return energy(x, cs.h); }, cs.x, k, hfd);
            fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - r0.e.forces[k]));
          }
          for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) {
              const double ep = std::getenv("SLAB_EP") ? std::atof(std::getenv("SLAB_EP")) : 2.5e-5;
              auto at = [&](double s) { return energy(strained(cs.x, a, b, s * ep), strained(cs.h, a, b, s * ep)); };
              const double num = (8 * (at(1) - at(-1)) - (at(2) - at(-2))) / (12 * ep);
              vmax = std::max(vmax, std::abs(num)), verr = std::max(verr, std::abs(num - r0.e.virial[3 * a + b]));
            }
          std::printf("SLAB-FD %s %-19s qproj %d Q %+.0f: E %+.6e eV, layered %d, max |F| %.3e err %.2e eV/A, "
                      "max |vir| %.3e err %.2e eV\n",
                      mode == LrcSpec::Slab::YB ? "yb     " : "ewald2d", cs.name, qp, Q, r0.e.energy, int(r0.c.cl.layered),
                      fmax, err, vmax, verr);
          CHECK(fmax > 1e-3);
          CHECK(err < 1e-8);
          CHECK(vmax > 1e-3);
          CHECK(verr < 1e-8);
          (void)n;
        }
}

TEST_CASE("lrc slab: layered conditioning (a molecule weakly linked to a slab) has exact conditioning forces",
          "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  std::vector<double> x, h;
  std::vector<int32_t> Z;
  nacl_slab(5.64, 2, 0.1, 0.0, x, Z, h);
  // a Cl- 5.4 A above the top layer: inside the a_on / a_off = 5 / 6 A link window of several slab ions (and images)
  x.insert(x.end(), {1.0, 0.6, 2.82 + 5.4});
  Z.push_back(17);
  const int n = int(Z.size());
  for (bool qp : {false, true}) {
    LrcSpec L = slab_spec(LrcSpec::Slab::YB, nullptr);
    L.qproj = qp;
    const LrcCondition c = lrc_condition(L, S, n, Z.data(), x.data(), -1.0, 10, 1, h.data());
    REQUIRE(c.cl.layered);
    REQUIRE(c.cl.s == 2);
    CHECK(c.cl.R[1] > 0.01);
    CHECK(c.cl.R[1] < 0.99);
    const std::vector<double> G = {0.7, -1.3};
    std::vector<double> F(3 * size_t(n), 0.0);
    lrc_condition_forces(L, S, c, n, Z.data(), x.data(), G, F.data());
    double err = 0, fmax = 0;
    for (int k = 0; k < 3 * n; k++) {
      const double num = -fd4(
          [&](const std::vector<double> &y) {
            const LrcCondition cc = lrc_condition(L, S, n, Z.data(), y.data(), -1.0, 10, 1, h.data());
            REQUIRE(cc.cl.s == 2);
            return G[0] * cc.S[cc.cl.sn[0]] + G[1] * cc.S[cc.cl.sn[n - 1]];
          },
          x, k, 1e-4);
      fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - F[k]));
    }
    std::printf("SLAB-COND qproj %d: R_slab,ion %.4f, max |F_cond| %.3e, err %.2e\n", qp, c.cl.R[1], fmax, err);
    CHECK(fmax > 1e-3);
    CHECK(err < 1e-8);
  }
}

TEST_CASE("lrc slab: parallel-plate capacitor and like-charged planes: energy slope and field are analytic",
          "[lrc][slab]") {
  // two square planes of formal ions (spacing 3 A, 4 per 6 x 6 A cell) at z = 0 and d; the switch (0.5 -> 1 A) lies below
  // every distance, so the sum is the bare Coulomb energy. Uniform sheets: dE/dd = -2 pi KE Q1 Q2 / A and every ion of
  // plane 2 feels the field 2 pi KE Q1 / A of plane 1 (corrugation ~ exp(-2 pi d / 3 A))
  const SqeSpec S = intra_sqe(1.0, 0.5);
  const std::vector<double> h = {6, 0, 0, 0, 6, 0, 0, 0, 0};
  const double A = 36;
  for (auto mode : {LrcSpec::Slab::YB, LrcSpec::Slab::Ewald2D})
    for (int like : {0, 1}) {
      const LrcSpec L = slab_spec(mode, std::make_shared<FixedQ>());
      auto build = [&](double d, std::vector<int32_t> &Z) {
        std::vector<double> x;
        Z.clear();
        for (int p = 0; p < 2; p++)
          for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
              x.insert(x.end(), {3.0 * i + 1.5 * p, 3.0 * j, p * d}), Z.push_back(p == 0 || like ? 11 : 17);
        return x;
      };
      const double Q1 = 4, Q2 = like ? 4 : -4, Q = Q1 + Q2;
      std::vector<int32_t> Z;
      std::vector<double> E;
      double Fz = 0;
      for (double d : {12.0, 15.0, 18.0}) {
        const std::vector<double> x = build(d, Z);
        const SlabRun r = slab_run(L, S, x, Z, h, Q, true);
        REQUIRE(r.c.cl.s == 2);
        E.push_back(r.e.energy);
        if (d == 15.0) Fz = r.e.forces[3 * 4 + 2];
        for (int i = 0; i < 8; i++)
          CHECK(r.e.forces[3 * i + 2] == Approx((i < 4 ? -1 : 1) * 2 * M_PI * KE * Q1 * Q2 / A / 4).epsilon(1e-9));
      }
      const double slope = -2 * M_PI * KE * Q1 * Q2 / A;
      std::printf("SLAB-PLATES %s %s: dE/dd %.10f %.10f eV/A (analytic %.10f), F_z(plane 2 ion) %.10f (analytic %.10f)\n",
                  mode == LrcSpec::Slab::YB ? "yb     " : "ewald2d", like ? "like-charged" : "capacitor   ",
                  (E[1] - E[0]) / 3, (E[2] - E[1]) / 3, slope, Fz, -slope / 4);
      CHECK((E[1] - E[0]) / 3 == Approx(slope).epsilon(1e-9));
      CHECK((E[2] - E[1]) / 3 == Approx(slope).epsilon(1e-9));
    }
}

TEST_CASE("lrc slab: a dipolar slab in a 3D cell differs by the dipole term 2 pi M^2 / V (tin-foil vs slab)",
          "[lrc][slab]") {
  // the capacitor in a (T, T, T) cell of height Lz (3D tin-foil Ewald, layered clusters under lrc_intra) against the
  // slab: E_3D + 2 pi KE M_z^2 / V -> E_slab exponentially in the vacuum, while E_3D alone misses it as 1 / Lz
  const SqeSpec S = intra_sqe(1.0, 0.5);
  const LrcSpec L = slab_spec(LrcSpec::Slab::YB, std::make_shared<FixedQ>());
  std::vector<double> x;
  std::vector<int32_t> Z;
  for (int p = 0; p < 2; p++)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) x.insert(x.end(), {3.0 * i, 3.0 * j + 1.5 * p, 2.0 + 6.0 * p}), Z.push_back(p ? 17 : 11);
  const double Es = slab_E(L, S, x, Z, {6, 0, 0, 0, 6, 0, 0, 0, 0}, 0.0);
  const double Mz = 4 * 2.0 - 4 * 8.0;
  for (double Lz : {20.0, 40.0, 80.0}) {
    const std::vector<double> h3 = {6, 0, 0, 0, 6, 0, 0, 0, Lz};
    const double E3 = slab_E(L, S, x, Z, h3, 0.0), Ec = 2 * M_PI * KE * Mz * Mz / (36 * Lz);
    std::printf("SLAB-DIPOLE Lz %3.0f: E_3D tin-foil %.10f, + 2 pi KE M^2 / V = %.10f, E_slab %.10f (diff %.1e)\n", Lz, E3,
                E3 + Ec, Es, E3 + Ec - Es);
    CHECK(E3 + Ec == Approx(Es).margin(1e-8));
    CHECK(std::abs(E3 - Es) > 0.1);
  }
}

TEST_CASE("lrc slab: a thick slab's interior layers have the 3D bulk energy (formal rock salt)", "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  for (auto mode : {LrcSpec::Slab::YB, LrcSpec::Slab::Ewald2D}) {
    const LrcSpec L = slab_spec(mode, std::make_shared<FixedQ>());
    std::vector<double> xb, hb, x, h;
    std::vector<int32_t> Zb, Z;
    rocksalt(5.64, 1, 0.0, xb, Zb, hb);
    const double per_layer = slab_E(L, S, xb, Zb, hb, 0.0) / 2; // 3D Ewald, 2 layers per conventional cell
    std::vector<double> E;
    for (int layers : {8, 10, 12}) {
      nacl_slab(5.64, layers, 0.0, 0.0, x, Z, h);
      E.push_back(slab_E(L, S, x, Z, h, 0.0));
    }
    std::printf("SLAB-THICK %s: 3D per layer %.12f eV; slab (E10 - E8) / 2 %.12f, (E12 - E10) / 2 %.12f\n",
                mode == LrcSpec::Slab::YB ? "yb     " : "ewald2d", per_layer, (E[1] - E[0]) / 2, (E[2] - E[1]) / 2);
    CHECK((E[2] - E[1]) / 2 == Approx(per_layer).epsilon(1e-9));
  }
}

TEST_CASE("lrc slab: invariance (translations incl. along the normal when charged, permutation, lattice shifts, rotation)",
          "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  std::vector<double> x, h;
  std::vector<int32_t> Z;
  nacl_slab(5.64, 3, 0.12, 0.5, x, Z, h, 0.15);
  x.insert(x.end(), {1.1, 0.7, 13.6});
  Z.push_back(11);
  const int n = int(Z.size());
  for (auto mode : {LrcSpec::Slab::YB, LrcSpec::Slab::Ewald2D})
    for (double Q : {0.0, 1.0}) {
      const LrcSpec L = slab_spec(mode, nullptr);
      const double E0 = slab_E(L, S, x, Z, h, Q);
      double worst = 0;
      // translation (in plane and along z)
      std::vector<double> y = x;
      for (int i = 0; i < n; i++) y[3 * i] += 0.37, y[3 * i + 1] -= 1.21, y[3 * i + 2] += 3.3;
      worst = std::max(worst, std::abs(slab_E(L, S, y, Z, h, Q) - E0));
      // single atoms moved by in-plane lattice vectors
      y = x;
      for (int a = 0; a < 3; a++) y[3 * 2 + a] += h[a], y[3 * 7 + a] -= h[3 + a]; // (toy charge pass: images to 2 cells)
      worst = std::max(worst, std::abs(slab_E(L, S, y, Z, h, Q) - E0));
      // permutation
      std::vector<int> perm(n);
      std::iota(perm.begin(), perm.end(), 0);
      std::reverse(perm.begin(), perm.end());
      std::vector<double> yp(3 * n);
      std::vector<int32_t> Zp(n);
      for (int i = 0; i < n; i++) {
        Zp[i] = Z[perm[i]];
        for (int a = 0; a < 3; a++) yp[3 * i + a] = x[3 * perm[i] + a];
      }
      worst = std::max(worst, std::abs(slab_E(L, S, yp, Zp, h, Q) - E0));
      // rotation of positions and cell (the open row stays zero)
      const double c1 = std::cos(0.7), s1 = std::sin(0.7), c2 = std::cos(-0.4), s2 = std::sin(-0.4);
      const double R[9] = {c1, -s1 * c2, s1 * s2, s1, c1 * c2, -c1 * s2, 0, s2, c2};
      auto rot = [&](const std::vector<double> &v) {
        std::vector<double> o(v.size());
        for (size_t i = 0; i < v.size() / 3; i++)
          for (int a = 0; a < 3; a++) o[3 * i + a] = R[3 * a] * v[3 * i] + R[3 * a + 1] * v[3 * i + 1] + R[3 * a + 2] * v[3 * i + 2];
        return o;
      };
      worst = std::max(worst, std::abs(slab_E(L, S, rot(x), Z, rot(h), Q) - E0));
      std::printf("SLAB-INV %s Q %+.0f: E %.10f, worst |dE| %.1e eV\n", mode == LrcSpec::Slab::YB ? "yb     " : "ewald2d", Q,
                  E0, worst);
      CHECK(worst < 1e-10);
    }
}

TEST_CASE("lrc slab: continuity of an ion above a slab through the 8 and 10 A switch", "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  const LrcSpec L = slab_spec(LrcSpec::Slab::YB, std::make_shared<FixedQ>());
  std::vector<double> xs, h;
  std::vector<int32_t> Zs;
  nacl_slab(5.64, 2, 0.0, 0.0, xs, Zs, h);
  Zs.push_back(11);
  const int n = int(Zs.size());
  // Na+ directly above a top-layer Cl- (at (2.82, 0, 2.82)): the pair distance d sweeps through 8 and 10 A
  auto E = [&](double d, bool forces, double *F = nullptr) {
    std::vector<double> x = xs;
    x.insert(x.end(), {2.82, 0.0, 2.82 + d});
    const SlabRun r = slab_run(L, S, x, Zs, h, 1.0, forces);
    if (F) *F = r.e.forces[3 * (n - 1) + 2];
    return r.e.energy;
  };
  for (double d0 : {8.0, 10.0}) {
    for (double eps : {1e-4, 1e-6}) {
      double Fm, Fp;
      const double Em = E(d0 - eps, true, &Fm), Ep = E(d0 + eps, true, &Fp);
      CHECK(std::abs(Ep - Em) < 10 * eps);
      CHECK(std::abs(Fp - Fm) < 50 * eps); // continuous: the jump falls with the gap (smootherstep: C2)
    }
    const double hh = 1e-4;
    double F0;
    E(d0, true, &F0);
    const double num = -(E(d0 + hh, false) - E(d0 - hh, false)) / (2 * hh);
    std::printf("SLAB-SWITCH d %.1f A: F_z %.8f eV/A, central difference %.8f, |diff| %.1e\n", d0, F0, num, std::abs(num - F0));
    CHECK(std::abs(num - F0) < 1e-6);
  }
}

TEST_CASE("lrc slab: one slab cluster without lrc_intra is the plain model (no long range, no charge pass)",
          "[lrc][slab]") {
  const SqeSpec S = intra_sqe();
  LrcSpec L = slab_spec(LrcSpec::Slab::YB, std::make_shared<NoQ>(), false);
  std::vector<double> x, h;
  std::vector<int32_t> Z;
  nacl_slab(5.64, 3, 0.1, 0.0, x, Z, h);
  for (double Q : {0.0, 1.0}) {
    const SlabRun r = slab_run(L, S, x, Z, h, Q, true);
    CHECK(r.c.cl.bulk);
    CHECK(r.c.cl.slab());
    CHECK(r.e.energy == 0.0);
    for (int i = 0; i < int(Z.size()); i++) CHECK(r.c.row_charge[i] + double(r.c.blend[i]) == Approx(Q));
  }
  // a slab beside a separate molecule needs lrc_intra
  x.insert(x.end(), {1.0, 1.0, 20.0});
  Z.push_back(11);
  L.qsrc = std::make_shared<FixedQ>();
  CHECK_THROWS(lrc_condition(L, S, int(Z.size()), Z.data(), x.data(), 1.0, 10, 1, h.data()));
}
