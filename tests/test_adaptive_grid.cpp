// PET's "grid" adaptive cutoff (core/adaptive_cutoff.h) against metatrain's
// get_adaptive_cutoffs_grid in float64 (scripts/adaptive_grid/grid_golden.py):
// radii and d r_i / d d_ik (autograd), plus finite differences of the radius.
// With MLIPCPP_TEST_GRID_GGUF (a "grid" PET, e.g. PET-OMol-S): the cached path
// against the general one and against finite differences of its energy.
#include "core/adaptive_cutoff.h"
#include "mlipcpp/system.h"
#include "runtime/graph_model.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace mlipcpp;

TEST_CASE("adaptive grid: metatrain golden values", "[adaptive]") {
  std::ifstream f(MLIPCPP_TEST_RESOURCES "/adaptive_grid_golden.json");
  REQUIRE(f);
  const auto cases = nlohmann::json::parse(f);
  for (const auto &c : cases) {
    const double cutoff = c["cutoff"], width = c["width"], target = c["target"];
    const auto centers = c["centers"].get<std::vector<int>>();
    const auto dist = c["distances"].get<std::vector<double>>();
    const auto radii = c["radii"].get<std::vector<double>>();
    const auto jac_ref = c["jacobian"].get<std::vector<double>>();
    for (size_t i = 0; i < radii.size(); i++) {
      std::vector<double> d;
      std::vector<size_t> index;
      for (size_t e = 0; e < dist.size(); e++)
        if (centers[e] == int(i)) d.push_back(dist[e]), index.push_back(e);
      std::vector<double> jac(d.size());
      const double r = adaptive::radius(true, int(d.size()), [&](int k) { return d[k]; }, cutoff, target, width,
                                        jac.data());
      CHECK(r == Catch::Approx(radii[i]).margin(1e-12));
      double norm = 0;
      for (double v : jac_ref) norm = std::max(norm, std::abs(v));
      for (size_t k = 0; k < d.size(); k++) CHECK(jac[k] == Catch::Approx(jac_ref[index[k]]).margin(1e-10 * norm));
      // finite differences of the radius in each distance
      const double h = 1e-6;
      for (size_t k = 0; k < d.size(); k += 3) {
        auto at = [&](double dd) {
          std::vector<double> e = d;
          e[k] += dd;
          return adaptive::radius(true, int(e.size()), [&](int q) { return e[q]; }, cutoff, target, width);
        };
        CHECK((at(h) - at(-h)) / (2 * h) == Catch::Approx(jac[k]).margin(1e-6 + 1e-5 * std::abs(jac[k])));
      }
    }
  }
}

TEST_CASE("adaptive grid: probes as torch.arange", "[adaptive]") {
  CHECK(adaptive::grid_probes(8.0, 0.5) == 60);
  CHECK(adaptive::grid_probes(4.5, 1.0) == 16);
  CHECK(adaptive::grid_probes(4.5, 0.2) == 80);
  CHECK(adaptive::grid_probes(0.5, 0.5) == 0);
}

TEST_CASE("adaptive grid: model, cached path", "[adaptive][model]") {
  const char *path = std::getenv("MLIPCPP_TEST_GRID_GGUF");
  if (!path) SKIP("MLIPCPP_TEST_GRID_GGUF not set");
  runtime::GraphModel model;
  model.load_from_gguf(path);
  REQUIRE(model.adaptive_cutoff_method() == "grid");
  // ethanol-like, and a periodic box of it
  const std::vector<float> x0 = {0.00f, 0.00f, 0.00f, 1.52f, 0.02f, 0.01f, 2.05f, 1.33f, -0.04f, -0.39f, 1.02f, 0.03f,
                                 -0.37f, -0.52f, 0.89f, -0.38f, -0.50f, -0.90f, 1.90f, -0.51f, 0.90f, 1.88f, -0.53f,
                                 -0.88f, 3.00f, 1.25f, 0.02f};
  const std::vector<int32_t> z = {6, 6, 8, 1, 1, 1, 1, 1, 1};
  const float lattice[3][3] = {{6.5f, 0, 0}, {0.4f, 6.2f, 0}, {-0.3f, 0.5f, 6.0f}};
  const Cell cell(lattice);
  for (const Cell *c : {static_cast<const Cell *>(nullptr), &cell}) {
    auto sys = [&](const std::vector<float> &x) {
      AtomicSystem s(int(z.size()), x.data(), z.data(), c);
      s.set_charge(-1), s.set_spin_multiplicity(2);
      return s;
    };
    const ModelResult cached = model.predict(sys(x0), true);
    const ModelResult general = model.predict_with_capture(sys(x0), nullptr); // the general path (energy)
    CHECK(cached.energy == Catch::Approx(general.energy).margin(2e-4));
    REQUIRE(cached.has_forces);
    const double h = 4e-3;
    for (size_t k = 0; k < x0.size(); k += 2) {
      std::vector<float> xp = x0, xm = x0;
      xp[k] += float(h), xm[k] -= float(h);
      const double fd = -(double(model.predict(sys(xp), false).energy) - model.predict(sys(xm), false).energy) /
                        (double(xp[k]) - double(xm[k]));
      CHECK(cached.forces[k] == Catch::Approx(fd).margin(1e-2));
    }
  }
}

// "solver" method on the surface Pd atom of metatrain#1324 (PET-MAD-XS settings: 16 neighbours, 7.5 A,
// width 1 A; distances rounded to 4 decimals): the old Newton-bisection hybrid two-cycled inside its
// bracket and returned 4.7076 A; the exact root of n_total(r) = 16 is 4.158976470 A (fp64 bisection)
TEST_CASE("adaptive solver: converges where Newton two-cycled (metatrain#1324)", "[adaptive]") {
  const std::vector<double> d = {
      2.6438, 2.6805, 2.6813, 2.6914, 2.7026, 2.7072, 2.7680, 2.7724, 2.8071, 2.8113, 2.8302, 2.8473, 3.6458, 3.7493,
      3.8703, 3.9268, 3.9275, 3.9958, 4.5983, 4.6906, 4.6999, 4.7060, 4.7174, 4.7250, 4.7268, 4.7493, 4.7513, 4.7698,
      4.7740, 4.7795, 4.7833, 4.8046, 4.8267, 4.8323, 4.8403, 4.8424, 4.8506, 4.8631, 5.4531, 5.4779, 5.4867, 5.5006,
      5.5111, 5.5167, 5.5214, 5.5869, 5.6298, 5.6467, 6.0969, 6.1321, 6.1425, 6.1688, 6.1729, 6.1828, 6.1892, 6.1896,
      6.2005, 6.2010, 6.2025, 6.2128, 6.2295, 6.2378, 6.2682, 6.2796, 6.3135, 6.6912, 6.6931, 6.7052, 6.7293, 6.7398,
      6.8774, 7.1155, 7.1646, 7.1779, 7.1800, 7.1838, 7.1925, 7.2289, 7.2329, 7.2347, 7.2461, 7.2464, 7.2583, 7.2606,
      7.2610, 7.2674, 7.2751, 7.3072, 7.3094, 7.3135, 7.3152, 7.3207, 7.3223, 7.3279, 7.3287, 7.3317, 7.3336, 7.3371,
      7.3397, 7.3595, 7.3698, 7.3814, 7.3851, 7.4075, 7.4351};
  auto at = [&](const std::vector<double> &e) {
    return adaptive::solve(int(e.size()), [&](int k) { return e[k]; }, 7.5, 16.0, 1.0);
  };
  const auto a = at(d);
  CHECK(a.r == Catch::Approx(4.158976470).margin(1e-8));
  CHECK_FALSE(a.clamped);
  // the cutoff's gradient against finite differences of the radius (the IFT slope is taken
  // within cutoff / 2^16 of the root, so agreement to ~1e-3 relative)
  for (size_t k = 0; k < d.size(); k++) {
    const double g = adaptive::ddist(d[k], a, 1.0);
    if (std::abs(g) < 1e-3) continue;
    const double h = 1e-6;
    std::vector<double> p = d, m = d;
    p[k] += h, m[k] -= h;
    CHECK((at(p).r - at(m).r) / (2 * h) == Catch::Approx(g).epsilon(2e-3));
  }
}
