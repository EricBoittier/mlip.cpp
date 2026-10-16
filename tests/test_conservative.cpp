#include "train/conservative.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
using namespace mlipcpp;
using Catch::Matchers::WithinAbs;
TEST_CASE(
    "Analytic force training batches and replays independent periodic systems",
    "[pet-lr][training]") {
  auto *path = std::getenv("MLIPCPP_PET_LR_REFERENCE");
  if (!path)
    SKIP("Set MLIPCPP_PET_LR_REFERENCE to generated fixtures");
  nlohmann::json cases;
  std::ifstream(path) >> cases;
  for (auto &test : cases) {
    std::string name = test.at("name");
    if (name != "feedforward-triclinic" && name != "residual-triclinic")
      continue;
    auto pos = test.at("positions").get<std::vector<float>>();
    auto z = test.at("numbers").get<std::vector<int32_t>>();
    auto c = test.at("cell").get<std::vector<std::vector<float>>>();
    float cell[3][3];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        cell[i][j] = c[i][j];
    AtomicSystem a(z.size(), pos.data(), z.data());
    a.set_cell(Cell(cell));
    auto b = a;
    b.positions_mut()[0] += .07;
    std::vector<float> fa(3 * z.size()), fb(fa.size()), fab;
    for (size_t i = 0; i < fa.size(); ++i) {
      fa[i] = .3 * std::sin(i + .5);
      fb[i] = .2 * std::cos(i + .2);
    }
    fab = fa;
    fab.insert(fab.end(), fb.begin(), fb.end());
    std::vector<float> cpu;
    for (auto backend : {BackendPreference::CPU, BackendPreference::Metal}) {
      INFO(name);
      INFO(static_cast<int>(backend));
      runtime::GraphModel model;
      model.set_backend_preference(backend);
      bool loaded = false;
      try {
        loaded = model.load_from_gguf(test.at("gguf"));
      } catch (const BackendUnavailableError &) {
        continue;
      }
      REQUIRE(loaded);
      auto prediction = model.predict(a, true);
      train::ConservativeTrainer trainer(model);
      auto one = trainer.evaluate({a}, {prediction.energy + .1}, fa, 1, 1);
      REQUIRE_THAT(one.energies[0], WithinAbs(prediction.energy, 1e-5));
      for (size_t i = 0; i < fa.size(); ++i)
        REQUIRE_THAT(one.forces[i], WithinAbs(prediction.forces[i], 2e-5));
      auto two = trainer.evaluate({b}, {prediction.energy - .2}, fb, 1, 1);
      auto batch = trainer.evaluate(
          {a, b}, {prediction.energy + .1, prediction.energy - .2}, fab, 1, 1);
      REQUIRE_THAT(batch.loss, WithinAbs((one.loss + two.loss) / 2, 2e-5));
      for (size_t i = 0; i < batch.gradient.size(); ++i)
        REQUIRE_THAT(batch.gradient[i],
                     WithinAbs((one.gradient[i] + two.gradient[i]) / 2,
                               2e-5 + std::abs(batch.gradient[i]) * 2e-4));
      auto repeat = trainer.evaluate(
          {b, a}, {prediction.energy - .2, prediction.energy + .1},
          [&] {
            auto f = fb;
            f.insert(f.end(), fa.begin(), fa.end());
            return f;
          }(),
          1, 1);
      for (size_t i = 0; i < batch.gradient.size(); ++i)
        REQUIRE_THAT(repeat.gradient[i], WithinAbs(batch.gradient[i], 2e-5));
      // Independent numerical parameter-direction check, including the LR
      // featurizer and projection, with force-only loss.
      auto fg = trainer.evaluate({a}, {0}, fa, 0, 1);
      double gn = 0;
      for (auto x : fg.gradient)
        gn += x * x;
      gn = std::sqrt(gn);
      REQUIRE(gn > 1e-7);
      std::vector<ggml_tensor *> tensors;
      std::vector<std::vector<float>> values;
      auto all = model.weights();
      for (auto &p : model.param_names()) {
        auto it = std::find_if(all.begin(), all.end(),
                               [&](auto &w) { return w.first == p; });
        tensors.push_back(it->second);
        values.emplace_back(ggml_nelements(it->second));
        ggml_backend_tensor_get(it->second, values.back().data(), 0,
                                ggml_nbytes(it->second));
      }
      auto perturb = [&](double h) {
        size_t off = 0;
        for (size_t t = 0; t < tensors.size(); ++t) {
          auto v = values[t];
          for (auto &x : v)
            x += h * fg.gradient[off++] / gn;
          ggml_backend_tensor_set(tensors[t], v.data(), 0,
                                  v.size() * sizeof(float));
        }
      };
      perturb(.001);
      auto lp = trainer.evaluate({a}, {0}, fa, 0, 1, false).loss;
      perturb(-.001);
      auto lm = trainer.evaluate({a}, {0}, fa, 0, 1, false).loss;
      perturb(0);
      REQUIRE_THAT((lp - lm) / .002, WithinAbs(gn, .003 * gn + 2e-6));
      if (backend == BackendPreference::CPU)
        cpu = fg.gradient;
      else
        for (size_t i = 0; i < cpu.size(); ++i)
          REQUIRE_THAT(fg.gradient[i],
                       WithinAbs(cpu[i], 1e-5 + std::abs(cpu[i]) * 5e-4));
    }
  }
}
