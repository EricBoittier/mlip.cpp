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
