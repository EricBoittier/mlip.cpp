// Combo charge-model components of the combined long-range model (lr_combined.h, feat/plato-lr-combo; after M. Langer,
// ipsum/PET-LR; ~/data/plato-lr-scratch/combo/SPEC.md): the Fermi / threshold / species-fill / QEq references, the
// potential-driven flows, the neutral conditioning and the diagnostics (metallicity, exact Born charges). Analytic toy
// charge / score / Fukui passes; every derivative against central differences, sum q = Q, rotation / permutation
// invariance, size consistency to 100 A, and a dump for the independent torch reference (scripts/plato_lr_combo).
#include "runtime/graph_model.h"
#include "runtime/lr_combined.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace mlipcpp::runtime;
using Catch::Approx;

namespace {
double rr(const double *x, int i, int j) {
  double s = 0;
  for (int k = 0; k < 3; k++) s += (x[3 * i + k] - x[3 * j + k]) * (x[3 * i + k] - x[3 * j + k]);
  return std::sqrt(s);
}
// toy pass: v_i = c_Z + sum_j d_ZZ' exp(-r_ij / L) + 0.03 (spin - 1) (+ k); analytic VJP
struct Toy final : LrcQSource {
  double k, L;
  int variant;
  Toy(int v, double k_, double L_) : k(k_), L(L_), variant(v) {}
  double c(int z) const {
    if (variant == 0) return z == 11 ? 0.8 : z == 17 ? -0.7 : z == 8 ? -0.5 : 0.25;
    return z == 11 ? 0.9 : z == 17 ? -0.4 : z == 8 ? 0.3 : -0.6;
  }
  double d(int zi, int zj) const {
    if (variant == 0) return 0.05 * ((zi * 7 + zj * 3) % 5) - 0.1;
    return 0.3 * ((zi * 5 + zj * 11) % 7) / 7.0 - 0.1 + k;
  }
  std::vector<double> raw(int n, const int32_t *Z, const double *x, double, int spin) override {
    std::vector<double> v(n);
    for (int i = 0; i < n; i++) {
      v[i] = c(Z[i]) + 0.03 * (spin - 1) + k;
      for (int j = 0; j < n; j++)
        if (j != i) v[i] += d(Z[i], Z[j]) * std::exp(-rr(x, i, j) / L);
    }
    return v;
  }
  void raw_vjp(int n, const int32_t *Z, const double *x, double, int, const double *w, double *g) override {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        if (j == i) continue;
        const double r = rr(x, i, j), dv = -w[i] * d(Z[i], Z[j]) * std::exp(-r / L) / L;
        for (int q = 0; q < 3; q++) {
          const double u = (x[3 * i + q] - x[3 * j + q]) / r;
          g[3 * i + q] += dv * u, g[3 * j + q] -= dv * u;
        }
      }
  }
};

SqeSpec sqe_spec() { // only lr_R / lr_t are read in graph mode
  SqeSpec s;
  s.enabled = true, s.lr_R = 10.0, s.lr_t = 2.0;
  return s;
}

struct Arm {
  const char *name;
  LrcSpec::Q0Ref ref;
  LrcSpec::FukuiMode fm;
  int flows;
  bool coulomb;
};
const Arm kArms[] = {
    {"fermi", LrcSpec::Q0Ref::Fermi, LrcSpec::FukuiMode::Element, 0, false},
    {"thresh", LrcSpec::Q0Ref::Threshold, LrcSpec::FukuiMode::Element, 0, false},
    {"threshA", LrcSpec::Q0Ref::Threshold, LrcSpec::FukuiMode::Atom, 0, false},
    {"sfill", LrcSpec::Q0Ref::SpeciesFill, LrcSpec::FukuiMode::Element, 0, false},
    {"qeq", LrcSpec::Q0Ref::Qeq, LrcSpec::FukuiMode::Element, 0, false},
    {"E+fl", LrcSpec::Q0Ref::Fukui, LrcSpec::FukuiMode::Element, 1, false},
    {"A+fl2", LrcSpec::Q0Ref::Fukui, LrcSpec::FukuiMode::Atom, 2, false},
    {"fermi+fl", LrcSpec::Q0Ref::Fermi, LrcSpec::FukuiMode::Element, 1, false},
    {"E+flC", LrcSpec::Q0Ref::Fukui, LrcSpec::FukuiMode::Element, 1, true},
    {"qeq+fl", LrcSpec::Q0Ref::Qeq, LrcSpec::FukuiMode::Element, 1, false},
};

// FTq-like settings (per-cluster conditioning 8 / 10 A, plateau projection 0.6, cond_w 0.6, Coulomb 8 -> 10 A)
LrcSpec combo_spec(const Arm &a, double a_on = 8.0, double a_off = 10.0) {
  LrcSpec L;
  L.enabled = true, L.graph = true, L.qpass_neutral = true;
  L.fukui_Z = {1, 8, 11, 17}, L.fukui_phi = {0.3, -0.2, 1.1, 0.5};
  L.qsrc = std::make_shared<Toy>(0, 0.0, 2.0);
  L.a_on = a_on, L.a_off = a_off, L.qproj = true, L.qproj_w = 0.6, L.cond_w = 0.6;
  L.fukui_mode = a.fm;
  if (L.fukui_atom()) L.fsrc = std::make_shared<Toy>(1, 0.0, 3.0);
  L.q0_ref = a.ref;
  L.ssrc = std::make_shared<Toy>(1, 0.15, 2.5); // score pass
  L.thr_theta = {0.2, -0.3, -0.9, 0.4};
  L.sf_score = {0.1, -0.6, 1.2, -1.0}, L.sf_scale = {0.2, 0.5, -0.3, 0.1};
  L.qeq_eta = {1.0, 1.5, 0.5, 1.2};
  L.flow_passes = a.flows, L.flow_coulomb = a.coulomb, L.flow_rc = 5.0;
  L.flow_kappa = {-3.0, -2.5, -3.5, -2.8, -2.2, -3.1, -2.9, -2.6, -3.3, -2.7, -3.0, -3.4, -2.4, -2.9, -3.2, -3.0};
  L.fukui_check(), L.combo_check();
  return L;
}
// Na+ (0), water (3.4 A) in one cluster; Cl- 8.7 A from a water H (in the 8-10 A link window), 13 A from Na
const std::vector<double> kX8 = {0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 13.0, 1.4, 0.6};
const int32_t ZG[] = {11, 8, 1, 1, 17};
const double CW[5] = {0.7, -0.3, 0.2, 0.5, -1.1};

double fd(const std::function<double(const std::vector<double> &)> &f, std::vector<double> x, size_t k, double h) {
  x[k] += h;
  const double a = f(x);
  x[k] -= 2 * h;
  return (a - f(x)) / (2 * h);
}
std::vector<double> rot(const std::vector<double> &v, double t) { // about (1, 1, 1) / sqrt 3
  const double c = std::cos(t), s = std::sin(t), u = 1 / std::sqrt(3.0);
  const double R[3][3] = {{c + u * u * (1 - c), u * u * (1 - c) - u * s, u * u * (1 - c) + u * s},
                          {u * u * (1 - c) + u * s, c + u * u * (1 - c), u * u * (1 - c) - u * s},
                          {u * u * (1 - c) - u * s, u * u * (1 - c) + u * s, c + u * u * (1 - c)}};
  std::vector<double> o(v.size());
  for (size_t i = 0; i < v.size() / 3; i++)
    for (int a = 0; a < 3; a++) o[3 * i + a] = R[a][0] * v[3 * i] + R[a][1] * v[3 * i + 1] + R[a][2] * v[3 * i + 2];
  return o;
}
} // namespace

TEST_CASE("lrc combo: absent keys keep the Fukui E model bit for bit", "[lrc][combo]") {
  Arm a{"E", LrcSpec::Q0Ref::Fukui, LrcSpec::FukuiMode::Element, 0, false};
  const LrcSpec L = combo_spec(a);
  REQUIRE(!L.combo());
  const SqeSpec S = sqe_spec();
  for (double Q : {0.0, 1.0, -1.0}) {
    const LrcQ0 o = lrc_q0(L, S, 5, ZG, kX8.data(), Q, 1);
    std::vector<double> q = L.qsrc->raw(5, ZG, kX8.data(), 0.0, 1), wt(5);
    double sq = 0, sf = 0;
    for (int i = 0; i < 5; i++) sq += q[i], wt[i] = L.fukui(ZG[i]), sf += wt[i];
    for (int i = 0; i < 5; i++) CHECK(o.q0[i] == q[i] + wt[i] / sf * (Q - sq));
  }
}

TEST_CASE("lrc combo: sum q = Q in every arm; flows conserve; metallicity", "[lrc][combo]") {
  const SqeSpec S = sqe_spec();
  for (const auto &a : kArms) {
    const LrcSpec L = combo_spec(a);
    for (auto [Q, spin] : std::vector<std::pair<double, int>>{{0, 1}, {1, 2}, {-1, 2}, {2, 1}, {-2, 1}, {0, 3}}) {
      const LrcQ0 o = lrc_q0(L, S, 5, ZG, kX8.data(), Q, spin);
      double sum = 0, sref = 0;
      for (int i = 0; i < 5; i++) sum += o.q0[i], sref += o.qref[i];
      INFO(a.name << " Q " << Q << " spin " << spin);
      CHECK(sum == Approx(Q).margin(1e-12));
      CHECK(sref == Approx(Q).margin(1e-12));
      if (a.ref == LrcSpec::Q0Ref::Fermi || a.ref == LrcSpec::Q0Ref::Threshold || a.ref == LrcSpec::Q0Ref::SpeciesFill)
        CHECK(o.M > 0);
    }
  }
  // local flows conserve each set closed under the 5 A kernel: Cl- far from the rest keeps its reference charge
  Arm a{"E+fl", LrcSpec::Q0Ref::Fukui, LrcSpec::FukuiMode::Element, 2, false};
  const LrcSpec L = combo_spec(a);
  const LrcQ0 o = lrc_q0(L, S, 5, ZG, kX8.data(), -1.0, 2);
  double moved = 0;
  for (int i = 0; i < 4; i++) moved += std::abs(o.q0[i] - o.qref[i]);
  CHECK(moved > 1e-3);                                // the water polarises
  CHECK(o.q0[4] == Approx(o.qref[4]).margin(1e-14)); // Cl- (> 5 A from all) keeps its charge exactly
  CHECK(o.q0[0] + o.q0[1] + o.q0[2] + o.q0[3] == Approx(o.qref[0] + o.qref[1] + o.qref[2] + o.qref[3]).margin(1e-13));
}

TEST_CASE("lrc combo: q0 VJP, long-range and conditioning forces match central differences in every arm", "[lrc][combo]") {
  const SqeSpec S = sqe_spec();
  for (const auto &a : kArms) {
    const LrcSpec L = combo_spec(a);
    for (auto [Q, spin] : std::vector<std::pair<double, int>>{{0, 1}, {1, 2}, {-1, 2}}) {
      auto cq = [&](const std::vector<double> &x) {
        const LrcQ0 o = lrc_q0(L, S, 5, ZG, x.data(), Q, spin);
        double v = 0;
        for (int i = 0; i < 5; i++) v += CW[i] * o.q0[i];
        return v;
      };
      const LrcQ0 o0 = lrc_q0(L, S, 5, ZG, kX8.data(), Q, spin);
      std::vector<double> g(15, 0.0);
      lrc_q0_vjp(L, S, 5, ZG, kX8.data(), o0, CW, g.data());
      double gmax = 0, gerr = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = fd(cq, kX8, k, 1e-5);
        gmax = std::max(gmax, std::abs(num)), gerr = std::max(gerr, std::abs(num - g[k]));
      }
      auto energy = [&](const std::vector<double> &x) {
        const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, spin);
        return lrc_energy(L, S, c, 5, ZG, x.data(), Q, false).energy;
      };
      const LrcCondition c0 = lrc_condition(L, S, 5, ZG, kX8.data(), Q, 10, spin);
      REQUIRE(c0.cl.s == 2);
      const LrcEnergy e0 = lrc_energy(L, S, c0, 5, ZG, kX8.data(), Q, true);
      double fmax = 0, err = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = -fd(energy, kX8, k, 1e-5);
        fmax = std::max(fmax, std::abs(num)), err = std::max(err, std::abs(num - e0.forces[k]));
      }
      auto fS = [&](const std::vector<double> &x) {
        const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, spin);
        double v = 0;
        for (int i = 0; i < 5; i++) v += CW[i] * (c.k[c.cl.sn[i]] + c.S[c.cl.sn[i]]);
        return v;
      };
      std::vector<double> G(c0.cl.s, 0.0), F(15, 0.0);
      for (int i = 0; i < 5; i++) G[c0.cl.sn[i]] += CW[i];
      lrc_condition_forces(L, S, c0, 5, ZG, kX8.data(), G, F.data());
      double cmax = 0, cerr = 0;
      for (size_t k = 0; k < 15; k++) {
        const double num = -fd(fS, kX8, k, 1e-5);
        cmax = std::max(cmax, std::abs(num)), cerr = std::max(cerr, std::abs(num - F[k]));
      }
      std::printf("FD %-9s Q %+.0f spin %d r %+.3f M %.3e | q0-VJP max %.3e err %.2e | E_lr %+.4e F max %.3e err %.2e | cond max %.3e err %.2e\n",
                  a.name, Q, spin, o0.r, o0.M, gmax, gerr, e0.energy, fmax, err, cmax, cerr);
      INFO(a.name << " Q " << Q);
      CHECK(gmax > 1e-3);
      CHECK(gerr < 1e-8 * std::max(1.0, gmax));
      CHECK(err < 1e-8 * std::max(1.0, fmax));
      CHECK(cerr < 1e-6 * std::max(1.0, cmax)); // the plateau blend's FD noise, as the Fukui test
    }
  }
}

TEST_CASE("lrc combo: rotation and permutation invariance", "[lrc][combo]") {
  const SqeSpec S = sqe_spec();
  for (const auto &a : kArms) {
    const LrcSpec L = combo_spec(a);
    for (double Q : {1.0, -1.0}) {
      const LrcCondition c = lrc_condition(L, S, 5, ZG, kX8.data(), Q, 10, 2);
      const LrcEnergy e = lrc_energy(L, S, c, 5, ZG, kX8.data(), Q, true);
      const auto xr = rot(kX8, 0.9);
      const LrcCondition cr = lrc_condition(L, S, 5, ZG, xr.data(), Q, 10, 2);
      const LrcEnergy er = lrc_energy(L, S, cr, 5, ZG, xr.data(), Q, true);
      INFO(a.name << " Q " << Q);
      CHECK(er.energy == Approx(e.energy).epsilon(1e-12).margin(1e-14));
      const auto fr = rot(e.forces, 0.9);
      for (int k = 0; k < 15; k++) CHECK(er.forces[k] == Approx(fr[k]).margin(1e-11));
      for (int i = 0; i < 5; i++) CHECK(cr.q0.q0[i] == Approx(c.q0.q0[i]).margin(1e-12));
      std::vector<double> xp(15);
      int32_t Zp[5];
      for (int i = 0; i < 5; i++) {
        Zp[i] = ZG[4 - i];
        for (int k = 0; k < 3; k++) xp[3 * i + k] = kX8[3 * (4 - i) + k];
      }
      const LrcCondition cp = lrc_condition(L, S, 5, Zp, xp.data(), Q, 10, 2);
      const LrcEnergy ep = lrc_energy(L, S, cp, 5, Zp, xp.data(), Q, true);
      CHECK(ep.energy == Approx(e.energy).epsilon(1e-12).margin(1e-14));
      for (int i = 0; i < 5; i++) {
        CHECK(cp.q0.q0[i] == Approx(c.q0.q0[4 - i]).margin(1e-12));
        for (int k = 0; k < 3; k++) CHECK(ep.forces[3 * i + k] == Approx(e.forces[3 * (4 - i) + k]).margin(1e-11));
      }
    }
  }
}

TEST_CASE("lrc combo: size consistency, the partner moved to 100 / 200 / 400 A", "[lrc][combo]") {
  // Local references keep every charge once the toy passes have decayed (exp(-r / 3) < 1e-14 at 100 A). Flows and QEq
  // respond to the far ion through its potential (physical induction; the difference must decay with distance). The
  // long-range energy equals the point-charge sum over the inter-cluster pairs (fully switched, no link).
  const SqeSpec S = sqe_spec();
  for (const auto &a : kArms) {
    const LrcSpec L = combo_spec(a);
    for (double Q : {0.0, 1.0, -1.0}) {
      std::vector<std::vector<double>> qs;
      for (double d : {100.0, 200.0, 400.0}) {
        std::vector<double> x = kX8;
        x[12] = d;
        qs.push_back(lrc_q0(L, S, 5, ZG, x.data(), Q, 1).q0);
      }
      double d1 = 0, d2 = 0;
      for (int i = 0; i < 5; i++) d1 = std::max(d1, std::abs(qs[0][i] - qs[1][i])), d2 = std::max(d2, std::abs(qs[1][i] - qs[2][i]));
      std::vector<double> x = kX8;
      x[12] = 100.0;
      const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, 1);
      const LrcEnergy e = lrc_energy(L, S, c, 5, ZG, x.data(), Q, false);
      double ref = 0;
      for (int i = 0; i < 4; i++) {
        double r2 = 0;
        for (int k = 0; k < 3; k++) r2 += (x[3 * i + k] - x[12 + k]) * (x[3 * i + k] - x[12 + k]);
        ref += 14.399645 * e.q[i] * e.q[4] / std::sqrt(r2);
      }
      std::printf("SIZE %-9s Q %+.0f max|dq| 100->200 %.2e 200->400 %.2e | E_lr(100) %+.9e point sum %+.9e | partner q %+.4f\n",
                  a.name, Q, d1, d2, e.energy, ref, e.q[4]);
      INFO(a.name << " Q " << Q);
      if (!(a.ref == LrcSpec::Q0Ref::Qeq || a.flows > 0)) CHECK(d1 < 1e-12);
      else CHECK(d2 < 0.55 * d1 + 1e-13); // decays at least as 1 / r
      CHECK(e.energy == Approx(ref).epsilon(1e-12).margin(1e-15));
    }
  }
}

TEST_CASE("lrc combo: exact Born charges against central differences of P, acoustic sum rule", "[lrc][combo]") {
  const SqeSpec S = sqe_spec();
  for (const auto &a : kArms) {
    const LrcSpec L = combo_spec(a);
    for (double Q : {0.0, 1.0}) {
      const LrcQ0 o = lrc_q0(L, S, 5, ZG, kX8.data(), Q, 1);
      const std::vector<double> Zb = lrc_bec(L, S, o, 5, ZG, kX8.data());
      double err = 0, zmax = 0, asr = 0;
      for (int ax = 0; ax < 3; ax++) {
        auto P = [&](const std::vector<double> &x) {
          const LrcQ0 q = lrc_q0(L, S, 5, ZG, x.data(), Q, 1);
          double p = 0;
          for (int k = 0; k < 5; k++) p += q.q0[k] * x[3 * k + ax];
          return p;
        };
        for (int i = 0; i < 5; i++)
          for (int b = 0; b < 3; b++) {
            const double num = fd(P, kX8, 3 * i + b, 1e-5), an = Zb[9 * i + 3 * ax + b];
            err = std::max(err, std::abs(num - an)), zmax = std::max(zmax, std::abs(an));
          }
        for (int b = 0; b < 3; b++) {
          double s = 0;
          for (int i = 0; i < 5; i++) s += Zb[9 * i + 3 * ax + b];
          asr = std::max(asr, std::abs(s - (ax == b ? Q : 0.0)));
        }
      }
      std::printf("BEC %-9s Q %+.0f max |Z| %.3e  |Z - FD| %.2e  ASR %.2e\n", a.name, Q, zmax, err, asr);
      INFO(a.name << " Q " << Q);
      CHECK(err < 1e-8 * std::max(1.0, zmax));
      CHECK(asr < 1e-10);
    }
  }
}

TEST_CASE("lrc combo: neutral conditioning puts every atom on the Q = 0 row, no conditioning forces", "[lrc][combo]") {
  const SqeSpec S = sqe_spec();
  LrcSpec L = combo_spec(kArms[0]);
  L.cond_neutral = true, L.readout = true;
  for (double Q : {1.0, -1.0, 2.0}) {
    const LrcCondition c = lrc_condition(L, S, 5, ZG, kX8.data(), Q, 10, 2);
    for (int i = 0; i < 5; i++) {
      CHECK(c.row_charge[i] == 0);
      CHECK(c.blend[i] == 0.0f);
    }
    CHECK(!c.needs_cond_forces());
    double s = 0;
    for (double v : c.q0.q0) s += v;
    CHECK(s == Approx(Q).margin(1e-12)); // charges still sum to Q
    // one cluster: the readout still gets its charges
    std::vector<double> x1 = kX8;
    x1[12] = 6.0, x1[13] = 0.0, x1[14] = 0.0;
    const LrcCondition c1 = lrc_condition(L, S, 5, ZG, x1.data(), Q, 10, 2);
    REQUIRE(c1.cl.s == 1);
    REQUIRE(c1.q0.q0.size() == 5);
    CHECK(c1.row_charge[0] == 0);
  }
}

TEST_CASE("lrc combo: dump for the torch reference (MLIPCPP_COMBO_PARITY=out.json)", "[lrc][combo]") {
  const char *out = std::getenv("MLIPCPP_COMBO_PARITY");
  if (!out) return;
  const SqeSpec S = sqe_spec();
  // no weak link (window 5 / 6 A), Cl- 8-9 A from the water (inside the Coulomb switch), 12.4 A from Na+
  const std::vector<double> x = {0, 0, 0, 3.4, 0.1, 0.2, 4.3, 0.3, 0.1, 3.1, 1.0, -0.2, 12.3, 1.4, 0.6};
  FILE *f = std::fopen(out, "w");
  auto arr = [](const std::vector<double> &v) {
    std::string s = "[";
    for (size_t k = 0; k < v.size(); k++) {
      char b[40];
      std::snprintf(b, 40, "%s%.17g", k ? ", " : "", v[k]);
      s += b;
    }
    return s + "]";
  };
  std::fprintf(f, "{\"x\": %s, \"Z\": [11, 8, 1, 1, 17], \"cases\": [", arr(x).c_str());
  bool first = true;
  for (const auto &a : kArms) {
    const LrcSpec L = combo_spec(a, 5.0, 6.0);
    for (auto [Q, spin] : std::vector<std::pair<double, int>>{{0, 1}, {1, 2}, {-1, 2}, {2, 1}}) {
      const LrcCondition c = lrc_condition(L, S, 5, ZG, x.data(), Q, 10, spin);
      REQUIRE(c.cl.s == 2);
      REQUIRE(c.cl.trivial());
      const LrcEnergy e = lrc_energy(L, S, c, 5, ZG, x.data(), Q, true);
      const std::vector<double> Zb = lrc_bec(L, S, c.q0, 5, ZG, x.data());
      std::fprintf(f, "%s{\"arm\": \"%s\", \"Q\": %g, \"spin\": %d, \"q0\": %s, \"q\": %s, \"E\": %.17g, \"F\": %s, \"M\": %.17g, \"bec\": %s}",
                   first ? "" : ", ", a.name, Q, spin, arr(e.q0).c_str(), arr(e.q).c_str(), e.energy, arr(e.forces).c_str(),
                   c.q0.M, arr(Zb).c_str());
      first = false;
    }
  }
  std::fprintf(f, "]}\n");
  std::fclose(f);
}
