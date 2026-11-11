// Physics-normalised tail loss (src/train/tail_loss.h): zero at the DFT differences, the normalisation by
// max(|dE_ref|, floor), the Huber branches, the gradient against central differences, and the pair-file loader.
#include "train/tail_loss.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

using namespace mlipcpp::train;
using Catch::Approx;

namespace {
// one scan: d = 8, 10, 12, 16 against d_far = 25 (local index 4); dE in eV
std::vector<TailPair> scan() {
  const double dft[4] = {-0.120, -0.080, -0.050, -0.004}, ref[4] = {-0.110, -0.075, -0.049, -0.0012};
  std::vector<TailPair> p;
  for (int k = 0; k < 4; k++) p.push_back({k, 4, dft[k], ref[k], 8.0 + 2 * k});
  return p;
}
std::vector<double> energies_at_dft(const std::vector<TailPair> &p, double offset) {
  std::vector<double> E(5, offset); // a common offset (absolute-energy mismatch) must not matter
  for (const auto &x : p) E[x.k] = offset + x.de_dft;
  return E;
}
} // namespace

TEST_CASE("tail loss: zero and zero gradient at the DFT differences", "[tail]") {
  TailLoss tl{1.0};
  const auto p = scan();
  for (double off : {0.0, -8309.712972663105, 1.6e-3 * 17}) {
    std::vector<double> g;
    const auto E = energies_at_dft(p, off);
    // round-off only: |E| ulp ~ 1e-12 eV at 8 keV, over s^2 ~ 1e-5 eV^2
    REQUIRE(tl.eval(p, E, &g) == Approx(0.0).margin(1e-15));
    for (double x : g) REQUIRE(std::abs(x) < 1e-6);
  }
}

TEST_CASE("tail loss: normalised residual, floor and Huber branches", "[tail]") {
  TailLoss tl{2.0, 2.585e-3, 1.0};
  auto p = scan();
  auto E = energies_at_dft(p, -100.0);
  // a model with no tail at all (dE_model = 0): r = -dE_DFT / s; ion pairs with dE_DFT = dE_ref give r = -1
  std::vector<double> E0(5, -100.0);
  REQUIRE(tl.residual(p[0], E0) == Approx(0.120 / 0.110));
  REQUIRE(tl.scale(p[3]) == Approx(2.585e-3)); // |dE_ref| = 1.2 meV < floor
  REQUIRE(tl.residual(p[3], E0) == Approx(0.004 / 2.585e-3));
  // a model with exactly the reference tail on one pair: r = (ref - dft) / s
  E[p[1].k] = -100.0 + p[1].de_ref;
  REQUIRE(tl.residual(p[1], E) == Approx((-0.075 + 0.080) / 0.075));
  // the loss: weight x mean over pairs; quadratic inside delta, linear outside
  std::vector<TailPair> one = {p[0]};
  std::vector<double> E1(5, 0.0);
  E1[0] = p[0].de_dft + 0.5 * tl.scale(p[0]); // r = 0.5
  REQUIRE(tl.eval(one, E1, nullptr) == Approx(2.0 * 0.5 * 0.25));
  E1[0] = p[0].de_dft + 3.0 * tl.scale(p[0]); // r = 3
  REQUIRE(tl.eval(one, E1, nullptr) == Approx(2.0 * (3.0 - 0.5)));
  std::vector<TailPair> two = {p[0], p[0]};
  REQUIRE(tl.eval(two, E1, nullptr) == Approx(2.0 * (3.0 - 0.5))); // a mean, not a sum
  // unit invariance of the normalised loss: residuals and scales x 1000 (meV) with the floor in meV
  TailLoss tm{2.0, 2.585, 1.0};
  auto pm = p;
  for (auto &x : pm) x.de_dft *= 1e3, x.de_ref *= 1e3;
  std::vector<double> Em(5);
  for (int k = 0; k < 5; k++) Em[k] = 1e3 * (E[k] + 100.0);
  REQUIRE(tm.eval(pm, Em, nullptr) == Approx(tl.eval(p, E, nullptr)));
}

TEST_CASE("tail loss: gradient matches central differences and stays finite", "[tail]") {
  TailLoss tl{0.3, 2.585e-3, 1.0};
  const auto p = scan();
  std::mt19937_64 rng(7);
  std::normal_distribution<double> nd(0.0, 0.03);
  for (int trial = 0; trial < 20; trial++) {
    auto E = energies_at_dft(p, -50.0);
    for (double &x : E) x += nd(rng);
    std::vector<double> g;
    tl.eval(p, E, &g);
    for (size_t k = 0; k < E.size(); k++) {
      REQUIRE(std::isfinite(g[k]));
      const double h = 1e-7;
      auto Ep = E, Em = E;
      Ep[k] += h, Em[k] -= h;
      const double fd = (tl.eval(p, Ep, nullptr) - tl.eval(p, Em, nullptr)) / (2 * h);
      REQUIRE(g[k] == Approx(fd).epsilon(1e-5).margin(1e-6));
    }
    double sum = 0; // translation invariance: the gradient sums to zero
    for (double x : g) sum += x;
    REQUIRE(std::abs(sum) < 1e-12);
  }
  // huge residuals: bounded gradient |dL/dE| <= weight x delta / floor
  std::vector<double> E(5, 0.0);
  E[0] = 1e6;
  std::vector<double> g;
  REQUIRE(std::isfinite(tl.eval(p, E, &g)));
  for (double x : g) REQUIRE(std::abs(x) <= tl.weight * tl.delta / tl.floor + 1e-12);
}

TEST_CASE("tail loss: pair file", "[tail]") {
  const std::string path = (std::filesystem::temp_directory_path() / "mlipcpp_test_tail_pairs.tsv").string();
  {
    std::ofstream f(path);
    f << "group\tclass\tk\tl\tdE_dft_eV\tdE_ref_eV\td_A\n"
      << "A\tion_pair\t10\t14\t-0.5\t-0.49\t8\n"
      << "A\tion_pair\t11\t14\t-0.3\t-0.29\t10\n"
      << "B\tion_neutral\t20\t21\t-0.01\t-0.012\t8\n"
      << "C\tion_pair\t30\t31\t-0.4\t-0.41\t8\n";
  }
  const TailSet t = TailSet::load(path);
  std::remove(path.c_str());
  REQUIRE(t.classes == std::vector<std::string>{"ion_pair", "ion_neutral"});
  REQUIRE(t.groups.size() == 3);
  REQUIRE(t.n_pairs() == 4);
  REQUIRE(t.by_class[0] == std::vector<int>{0, 2});
  REQUIRE(t.groups[0].ids == std::vector<int64_t>{10, 14, 11});
  REQUIRE(t.groups[0].pairs[1].k == 2);
  REQUIRE(t.groups[0].pairs[1].l == 1);
  REQUIRE(t.groups[0].pairs[1].de_dft == Approx(-0.3));
}

// Same-geometry charge / spin-state pair loss (mlipcpp-train --pair-data): TailLoss with dE_ref = 0 and the scale
// fixed to --pair-scale (eV), so r = (dE_model - dE_DFT) / scale and L = W mean_p Huber_delta(r)
namespace {
// one geometry, three states: N (local 0), N-1 (1, the cation), N+1 (2, the anion); IP = E(N-1) - E(N),
// EA = E(N) - E(N+1), both as k - l
std::vector<TailPair> ipea_group() { return {{1, 0, 9.31, 0.0, 0.0}, {0, 2, 1.12, 0.0, 0.0}}; }
} // namespace

TEST_CASE("pair loss: residual in eV over the fixed scale, quadratic and linear branches", "[pair]") {
  const auto p = ipea_group();
  TailLoss pl{0.5, 1.0, 1.0}; // weight 0.5, scale 1 eV, delta 1
  std::vector<double> E = {-2000.0, -2000.0 + 9.31, -2000.0 - 1.12};
  std::vector<double> g;
  REQUIRE(pl.eval(p, E, &g) == Approx(0.0).margin(1e-12)); // at the DFT differences, any common offset
  for (double x : g) REQUIRE(std::abs(x) < 1e-12);
  REQUIRE(pl.scale(p[0]) == 1.0);
  E[1] += 0.4; // IP 0.4 eV too high: r = 0.4, L = 0.5 x mean(0.5 0.16, 0) = 0.02
  REQUIRE(pl.residual(p[0], E) == Approx(0.4));
  REQUIRE(pl.eval(p, E, nullptr) == Approx(0.5 * 0.5 * (0.5 * 0.16)));
  E[1] += 57.6; // a -58 eV-class outlier (here +58): linear, |dL/dE| bounded by W delta / (scale n_pairs)
  REQUIRE(pl.eval(p, E, &g) == Approx(0.5 * 0.5 * (58.0 - 0.5)));
  REQUIRE(g[1] == Approx(0.5 * 1.0 / 2.0));
  REQUIRE(g[0] == Approx(-0.5 * 1.0 / 2.0));
  // the scale: the same loss in meV with scale 1000 meV
  TailLoss pm{0.5, 1000.0, 1.0};
  auto q = p;
  for (auto &x : q) x.de_dft *= 1e3;
  std::vector<double> Em = {1e3 * E[0], 1e3 * E[1], 1e3 * E[2]};
  REQUIRE(pm.eval(q, Em, nullptr) == Approx(pl.eval(p, E, nullptr)));
}

TEST_CASE("pair loss: gradient matches central differences, sums to zero", "[pair]") {
  const auto p = ipea_group();
  std::mt19937_64 rng(11);
  std::normal_distribution<double> nd(0.0, 0.8);
  for (double delta : {0.25, 1.0, 1e6}) { // delta 1e6: the pure square W/2 (dE_model - dE_DFT)^2 / scale^2
    TailLoss pl{1.7, 0.5, delta};
    for (int trial = 0; trial < 20; trial++) {
      std::vector<double> E = {-713.0, -713.0 + 9.31, -713.0 - 1.12};
      for (double &x : E) x += nd(rng);
      std::vector<double> g;
      const double L = pl.eval(p, E, &g);
      if (delta == 1e6) {
        double ref = 0;
        for (const auto &x : p) ref += 0.5 * std::pow((E[x.k] - E[x.l] - x.de_dft) / 0.5, 2);
        REQUIRE(L == Approx(1.7 * ref / p.size()));
      }
      double sum = 0;
      for (size_t k = 0; k < E.size(); k++) {
        const double h = 1e-6;
        auto Ep = E, Em = E;
        Ep[k] += h, Em[k] -= h;
        const double fd = (pl.eval(p, Ep, nullptr) - pl.eval(p, Em, nullptr)) / (2 * h);
        REQUIRE(g[k] == Approx(fd).epsilon(1e-5).margin(1e-7));
        sum += g[k];
      }
      REQUIRE(std::abs(sum) < 1e-12); // a common shift of the group (composition, offsets) has no gradient
    }
  }
}
