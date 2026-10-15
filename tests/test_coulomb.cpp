#include "core/coulomb.h"
#include "core/p3m.h"
#include "runtime/graph_model.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace mlipcpp;
using Catch::Matchers::WithinAbs;

namespace {
AtomicSystem make_system(bool periodic) {
  float r[] = {0.3f, 0.5f, 0.2f, 1.5f, 0.8f, 0.4f, 5.1f, 1.7f, 2.1f};
  int32_t z[] = {1, 1, 1};
  AtomicSystem s(3, r, z);
  if (periodic) {
    float c[3][3] = {{7.3f, 0, 0}, {0.6f, 6.1f, 0}, {0.2f, 0.7f, 5.7f}};
    s.set_cell(Cell(c));
  }
  return s;
}
CoulombResult evaluate(const AtomicSystem &s,
                       const std::vector<float> *a = nullptr) {
  return coulomb_matrix(s, NeighborListBuilder({4.5f, true, false}).build(s),
                        {}, a);
}
double energy(const AtomicSystem &s, const std::vector<float> &a) {
  auto k = evaluate(s).matrix;
  double e = 0;
  for (size_t i = 0; i < k.size(); ++i)
    e += k[i] * a[i];
  return e;
}
} // namespace

TEST_CASE("Coulomb analytic adjoints match finite differences", "[coulomb]") {
  for (bool periodic : {false, true}) {
    auto s = make_system(periodic);
    std::vector<float> adj = {0.7f,  -0.4f, 0.9f, 0.3f, 0.1f,
                              -0.5f, -0.8f, 0.6f, 0.2f};
    auto analytic = evaluate(s, &adj);
    const double h = 0.001;
    for (int i = 0; i < 9; ++i) {
      auto plus = s, minus = s;
      plus.positions_mut()[i] += h;
      minus.positions_mut()[i] -= h;
      REQUIRE_THAT(
          analytic.forces[i],
          WithinAbs(-(energy(plus, adj) - energy(minus, adj)) / (2 * h), 2e-5));
    }
    for (int d = 0; d < 3; ++d) {
      double total = 0;
      for (int i = 0; i < 3; ++i)
        total += analytic.forces[3 * i + d];
      REQUIRE_THAT(total, WithinAbs(0, 1e-12));
    }
    if (!periodic)
      continue;
    const auto &c = s.cell()->matrix;
    double volume = c[0][0] * c[1][1] * c[2][2];
    int aa[] = {0, 1, 2, 1, 0, 0}, bb[] = {0, 1, 2, 2, 2, 1};
    for (int v = 0; v < 6; ++v) {
      auto strained = [&](double h) {
        auto t = s;
        int a = aa[v], b = bb[v];
        for (int i = 0; i < 3; ++i) {
          t.positions_mut()[3 * i + b] +=
              h * s.positions()[3 * i + a] * (a == b ? 1 : 0.5);
          if (a != b)
            t.positions_mut()[3 * i + a] += h * s.positions()[3 * i + b] * 0.5;
        }
        float cc[3][3];
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            cc[i][j] = c[i][j];
        for (int i = 0; i < 3; ++i) {
          cc[i][b] += h * c[i][a] * (a == b ? 1 : 0.5);
          if (a != b)
            cc[i][a] += h * c[i][b] * 0.5;
        }
        t.set_cell(Cell(cc));
        return t;
      };
      REQUIRE_THAT(analytic.stress[v], WithinAbs((energy(strained(h), adj) -
                                                  energy(strained(-h), adj)) /
                                                     (2 * h * volume),
                                                 2e-7));
    }
  }
}

TEST_CASE("Coulomb rejects unsupported geometry", "[coulomb]") {
  auto s = make_system(true);
  auto cell = *s.cell();
  cell.periodic[2] = false;
  s.set_cell(cell);
  REQUIRE_THROWS(evaluate(s));
  s = make_system(false);
  for (int i = 0; i < 3; ++i)
    s.positions_mut()[3 + i] = s.positions()[i];
  REQUIRE_THROWS(evaluate(s));
}

TEST_CASE("PET-LR matches metatrain reference fixtures", "[pet-lr]") {
  const char *path = std::getenv("MLIPCPP_PET_LR_REFERENCE");
  if (!path) {
    SKIP("Run tests/generate_pet_lr_reference.py and set "
         "MLIPCPP_PET_LR_REFERENCE");
  }
  std::ifstream f(path);
  REQUIRE(f.good());
  nlohmann::json cases;
  f >> cases;
  for (const auto &test : cases) {
    INFO(test.at("name").get<std::string>());
    auto pos = test.at("positions").get<std::vector<float>>();
    auto z = test.at("numbers").get<std::vector<int32_t>>();
    AtomicSystem s(z.size(), pos.data(), z.data());
    if (test.contains("cell")) {
      auto c = test.at("cell").get<std::vector<std::vector<float>>>();
      float cc[3][3];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          cc[i][j] = c[i][j];
      s.set_cell(Cell(cc));
    }
    if (test.contains("matrix")) {
      auto reference_k = test.at("matrix").get<std::vector<float>>();
      std::vector<float> actual_k;
      if (test.value("solver", "direct-ewald") == "p3m") {
        std::vector<float> identity(z.size() * z.size(), 0);
        for (size_t i = 0; i < z.size(); ++i)
          identity[i * z.size() + i] = 1;
        actual_k = p3m(s, NeighborListBuilder({4.5f, true, false}).build(s), {},
                       z.size(), identity)
                       .potentials;
      } else
        actual_k = evaluate(s).matrix;
      REQUIRE(actual_k.size() == reference_k.size());
      for (size_t i = 0; i < actual_k.size(); ++i)
        REQUIRE_THAT(actual_k[i], WithinAbs(reference_k[i], 2e-7));
    }
    runtime::GraphModel model;
    model.set_backend_preference(BackendPreference::CPU);
    REQUIRE(model.load_from_gguf(test.at("gguf")));
    const bool compute_forces = !test.value("energy_only", false);
    auto result = model.predict(s, compute_forces);
    REQUIRE(result.has_forces == compute_forces);
    REQUIRE_THAT(result.energy, WithinAbs(test.at("energy").get<double>(),
                                          std::max(3e-5, z.size() * 2e-6)));
    if (compute_forces) {
      auto force = test.at("forces").get<std::vector<double>>();
      for (size_t i = 0; i < force.size(); ++i)
        REQUIRE_THAT(result.forces[i], WithinAbs(force[i], 5e-5));
      if (test.contains("stress")) {
        REQUIRE(result.has_stress);
        auto stress = test.at("stress").get<std::vector<double>>();
        for (int i = 0; i < 6; ++i)
          REQUIRE_THAT(
              result.stress[i],
              WithinAbs(stress[i], std::max(2e-6, std::abs(stress[i]) * 5e-5)));
      }
    } else {
      REQUIRE_THROWS(model.predict(s, true));
    }
    auto energy_only = model.predict(s, false);
    REQUIRE_THAT(energy_only.energy, WithinAbs(result.energy, 3e-5));
  }
}

TEST_CASE("P3M charge and geometry adjoints match finite differences",
          "[p3m]") {
  for (bool periodic : {false, true}) {
    auto s = make_system(periodic);
    std::vector<float> q = {.7f, -.3f, .2f, .4f, -.5f, .9f},
                       g = {-.2f, .6f, .8f, -.7f, .3f, .1f};
    auto run = [&](const AtomicSystem &system,
                   const std::vector<float> &charges,
                   const std::vector<float> *adj = nullptr) {
      return p3m(system, NeighborListBuilder({4.5f, true, false}).build(system),
                 {}, 2, charges, adj);
    };
    auto energy = [&](const AtomicSystem &system,
                      const std::vector<float> &charges) {
      auto v = run(system, charges).potentials;
      double e = 0;
      for (size_t i = 0; i < v.size(); ++i)
        e += double(v[i]) * g[i];
      return e;
    };
    auto analytic = run(s, q, &g);
    double h = .001;
    for (int i = 0; i < 9; ++i) {
      auto a = s, b = s;
      a.positions_mut()[i] += h;
      b.positions_mut()[i] -= h;
      REQUIRE_THAT(analytic.forces[i],
                   WithinAbs(-(energy(a, q) - energy(b, q)) / (2 * h), 2e-5));
    }
    for (size_t i = 0; i < q.size(); ++i) {
      auto a = q, b = q;
      a[i] += h;
      b[i] -= h;
      REQUIRE_THAT(analytic.charge_gradient[i],
                   WithinAbs((energy(s, a) - energy(s, b)) / (2 * h), 2e-5));
    }
    if (!periodic)
      continue;
    int aa[] = {0, 1, 2, 1, 0, 0}, bb[] = {0, 1, 2, 2, 2, 1};
    const auto &c = s.cell()->matrix;
    double volume = c[0][0] * c[1][1] * c[2][2];
    for (int t = 0; t < 6; ++t) {
      auto strain = [&](double eps) {
        auto sys = s;
        int a = aa[t], b = bb[t];
        float cc[3][3];
        for (int i = 0; i < 3; ++i) {
          sys.positions_mut()[3 * i + b] +=
              eps * s.positions()[3 * i + a] * (a == b ? 1 : .5);
          if (a != b)
            sys.positions_mut()[3 * i + a] +=
                eps * s.positions()[3 * i + b] * .5;
          for (int j = 0; j < 3; ++j)
            cc[i][j] = c[i][j];
          cc[i][b] += eps * c[i][a] * (a == b ? 1 : .5);
          if (a != b)
            cc[i][a] += eps * c[i][b] * .5;
        }
        sys.set_cell(Cell(cc));
        return sys;
      };
      REQUIRE_THAT(analytic.stress[t],
                   WithinAbs((energy(strain(h), q) - energy(strain(-h), q)) /
                                 (2 * h * volume),
                             2e-7));
    }
  }
}

TEST_CASE("P3M runs beyond the dense solver atom limit", "[p3m]") {
  const int n = 729;
  std::vector<float> r(3 * n), q(n), g(n);
  std::vector<int32_t> z(n, 1);
  for (int x = 0, i = 0; x < 9; ++x)
    for (int y = 0; y < 9; ++y)
      for (int zz = 0; zz < 9; ++zz, ++i) {
        r[3 * i] = 2.1f * x + .13f;
        r[3 * i + 1] = 2.1f * y + .23f;
        r[3 * i + 2] = 2.1f * zz + .31f;
        q[i] = std::sin(i * .37);
        g[i] = std::cos(i * .19);
      }
  AtomicSystem s(n, r.data(), z.data());
  float c[3][3] = {{18.9f, 0, 0}, {0, 18.9f, 0}, {0, 0, 18.9f}};
  s.set_cell(Cell(c));
  auto out =
      p3m(s, NeighborListBuilder({4.5f, true, false}).build(s), {}, 1, q, &g);
  REQUIRE(out.potentials.size() == n);
  REQUIRE(out.forces.size() == 3 * n);
  REQUIRE(out.stress.size() == 6);
  for (auto v : out.potentials)
    REQUIRE(std::isfinite(v));
  for (auto v : out.forces)
    REQUIRE(std::isfinite(v));
  for (auto v : out.stress)
    REQUIRE(std::isfinite(v));
  auto cell = *s.cell();
  cell.periodic[2] = false;
  s.set_cell(cell);
  REQUIRE_THROWS(p3m(s, {}, {}, 1, q));
}

TEST_CASE("P3M reproduces torch-pme 0.5 potential influence convention",
          "[p3m]") {
  // Float64 torch-pme P3MCalculator, order=5, spacing=1.33, sigma=1.4,
  // exclusion=4.5; two non-neutral channels and independent potential adjoints.
  auto s = make_system(true);
  std::vector<float> q = {.7f, -.3f, .2f, .4f, -.5f, .9f},
                     g = {-.2f, .6f, .8f, -.7f, .3f, .1f};
  auto out =
      p3m(s, NeighborListBuilder({4.5f, true, false}).build(s), {}, 2, q, &g);
  std::vector<double> v = {-.15614583937141835, -.06165513391137139,
                           -.13649166358617199, -.08729517836399708,
                           .05320593339320703,  -.20150011168320478};
  std::vector<double> dq = {-.11319609042214523, -.018564122782702447,
                            -.15906735785088252, .04566335370956735,
                            -.09809670377797264, -.02207893886553671};
  std::vector<double> force = {
      .04995873749566026,   .021778671848683456,  .024566861400305513,
      -.059329428517012506, -.017140416240608542, -.014793699178040898,
      .009365679620352136,  -.004638523044469749, -.009773636368480571};
  std::vector<double> stress = {.0005164769542141779,   .00012183085470412183,
                                .00013132841020993297,  6.031066220313969e-5,
                                -.00011125281127198958, -8.566074182492515e-6};
  for (size_t i = 0; i < v.size(); ++i) {
    REQUIRE_THAT(out.potentials[i], WithinAbs(v[i], 3e-8));
    REQUIRE_THAT(out.charge_gradient[i], WithinAbs(dq[i], 3e-8));
    REQUIRE_THAT(out.stress[i], WithinAbs(stress[i], 1e-10));
  }
  for (size_t i = 0; i < force.size(); ++i)
    REQUIRE_THAT(out.forces[i], WithinAbs(force[i], 1e-8));
}
