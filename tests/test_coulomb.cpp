#include "core/coulomb.h"
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
    auto reference_k = test.at("matrix").get<std::vector<float>>();
    auto actual_k = evaluate(s).matrix;
    REQUIRE(actual_k.size() == reference_k.size());
    for (size_t i = 0; i < actual_k.size(); ++i)
      REQUIRE_THAT(actual_k[i], WithinAbs(reference_k[i], 2e-7));
    runtime::GraphModel model;
    model.set_backend_preference(BackendPreference::CPU);
    REQUIRE(model.load_from_gguf(test.at("gguf")));
    const bool compute_forces = !test.value("energy_only", false);
    auto result = model.predict(s, compute_forces);
    REQUIRE(result.has_forces == compute_forces);
    REQUIRE_THAT(result.energy,
                 WithinAbs(test.at("energy").get<double>(), 3e-5));
    if (compute_forces) {
      auto force = test.at("forces").get<std::vector<double>>();
      for (size_t i = 0; i < force.size(); ++i)
        REQUIRE_THAT(result.forces[i], WithinAbs(force[i], 5e-5));
      if (test.contains("stress")) {
        REQUIRE(result.has_stress);
        auto stress = test.at("stress").get<std::vector<double>>();
        for (int i = 0; i < 6; ++i)
          REQUIRE_THAT(result.stress[i], WithinAbs(stress[i], 2e-6));
      }
    } else {
      REQUIRE_THROWS(model.predict(s, true));
    }
    auto energy_only = model.predict(s, false);
    REQUIRE_THAT(energy_only.energy, WithinAbs(result.energy, 3e-5));
  }
}
