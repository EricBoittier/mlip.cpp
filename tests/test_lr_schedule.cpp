// Per-group learning-rate schedules (src/train/lr_schedule.h) against metatrain's reference
// (src/metatrain/pet/modules/lr_schedule.py, EricBoittier/metatrain feat/pet-plato-lr): the
// golden tables tests/resources/lr_schedule_golden.json, the name -> group matching, and the
// grouped optimizer's freezing (no update, no state) and per-group Adam step counts.
#include "train/lr_schedule.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <vector>

using mlipcpp::train::GroupOptimizer;
using mlipcpp::train::LrSchedule;

namespace {
nlohmann::json golden() {
  std::ifstream f(MLIPCPP_TEST_RESOURCES "/lr_schedule_golden.json");
  REQUIRE(f.good());
  return nlohmann::json::parse(f);
}
} // namespace

TEST_CASE("lr schedule: golden tables", "[lr_schedule]") {
  const auto g = golden();
  for (const auto &c : g["cases"]) {
    const LrSchedule s = LrSchedule::parse(c["spec"].dump());
    const int64_t S = c["steps_per_epoch"], E = c["num_epochs"], T = S * E;
    INFO("shape " << c["spec"]["shape"]);
    REQUIRE(s.groups.size() == c["lr"].size());
    double worst = 0;
    for (auto &[name, lrs] : c["lr"].items()) {
      const int gi = s.group_index(name);
      REQUIRE(gi >= 0);
      for (int64_t n = 0; n < T; n++) {
        const auto [lr, frozen] = s.group_lr(gi, n, S, T);
        const double want = lrs[n];
        CHECK(frozen == c["frozen"][name][n].get<bool>());
        worst = std::max(worst, std::abs(lr - want) / std::max(1e-300, std::abs(want)));
        CHECK(lr == Catch::Approx(want).epsilon(1e-15).margin(0));
      }
    }
    WARN("shape " << c["spec"]["shape"].get<std::string>() << ": largest relative difference " << worst);
  }
}

TEST_CASE("lr schedule: name matching", "[lr_schedule]") {
  const auto g = golden();
  const LrSchedule s = LrSchedule::parse(g["matching"]["spec"].dump());
  for (auto &[name, group] : g["matching"]["groups"].items()) {
    INFO(name);
    CHECK(s.groups.at(s.group_of(name)).name == group.get<std::string>());
  }
  CHECK(LrSchedule::matches("backend.mesh.mesh_gate", "*_gate"));
  CHECK(!LrSchedule::matches("backend.mesh.mesh_gate.weight", "*_gate"));
  CHECK(LrSchedule::matches("layers.0.qkv.W", "layers.?.qkv.*"));
  CHECK(LrSchedule::matches("backend.gnn_layers.1.x", "gnn_layers."));
}

TEST_CASE("lr schedule: defaults and errors", "[lr_schedule]") {
  const LrSchedule s = LrSchedule::parse(R"({"base_lr": 1e-3})");
  REQUIRE(s.groups.size() == 1);
  CHECK(s.groups[0].name == "default");
  CHECK(s.base_lr(0, 10, 100) == 1e-3); // no warmup, decay from step 0: cosine at p = 0
  CHECK(s.base_lr(99, 10, 100) == Catch::Approx(0.0).margin(1e-18));
  CHECK_THROWS(LrSchedule::parse(R"({"base_lr": 1e-3, "shape": "step"})"));
  CHECK_THROWS(LrSchedule::parse(R"({"base_lr": 1e-3, "groups": [{"name": "a", "optimizer": "sgd"}]})"));
  CHECK_THROWS(LrSchedule::parse(R"({"min_lr": 1e-3})"));
}

TEST_CASE("grouped optimizer: frozen groups and per-group Adam", "[lr_schedule]") {
  // two parameters: "a.w" (group fast, never frozen), "b.w" (group slow, frozen for 1 epoch of 2 steps)
  const char *spec = R"({"base_lr": 0.1, "shape": "constant", "groups": [
      {"name": "fast", "match": ["a."]},
      {"name": "slow", "match": ["b."], "freeze_epochs": 1, "lr_mult": 0.5, "weight_decay": 0.1}],
      "default_group": "fast"})";
  ggml_backend_t be = ggml_backend_cpu_init();
  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
  ggml_tensor *pa = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3), *pb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
  ggml_tensor *ga = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3), *gb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
  ggml_set_param(pa), ggml_set_param(pb);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  const float a0[3] = {1, 2, 3}, b0[3] = {-1, 0.5f, 2}, g_a[3] = {0.3f, -0.2f, 0.1f}, g_b[3] = {0.2f, 0.2f, -0.4f};
  ggml_backend_tensor_set(pa, a0, 0, 12), ggml_backend_tensor_set(pb, b0, 0, 12);
  ggml_backend_tensor_set(ga, g_a, 0, 12), ggml_backend_tensor_set(gb, g_b, 0, 12);
  GroupOptimizer opt(be, {{pa, ga, "a.w", false}, {pb, gb, "b.w", false}}, LrSchedule::parse(spec), /*clip*/ 1e9);
  const int64_t S = 2, T = 8;
  auto get = [](ggml_tensor *t) {
    std::vector<float> v(3);
    ggml_backend_tensor_get(t, v.data(), 0, 12);
    return v;
  };
  // torch.optim.AdamW, one parameter, constant gradient g, from step t0 + 1 to t0 + k
  auto adamw = [](std::vector<double> x, const float *g, double lr, double wd, int k) {
    std::vector<double> m(3, 0), v(3, 0);
    for (int t = 1; t <= k; t++)
      for (int i = 0; i < 3; i++) {
        x[i] *= 1 - lr * wd;
        m[i] = 0.9 * m[i] + 0.1 * g[i], v[i] = 0.999 * v[i] + 0.001 * double(g[i]) * g[i];
        const double mh = m[i] / (1 - std::pow(0.9, t)), vh = v[i] / (1 - std::pow(0.999, t));
        x[i] -= lr * mh / (std::sqrt(vh) + 1e-8);
      }
    return x;
  };
  opt.step(0, S, T), opt.step(1, S, T); // b frozen
  CHECK(get(pb) == std::vector<float>(b0, b0 + 3));
  opt.step(2, S, T), opt.step(3, S, T); // b's first two Adam steps (t = 1, 2)
  const auto wa = adamw({1, 2, 3}, g_a, 0.1, 0.0, 4), wb = adamw({-1, 0.5, 2}, g_b, 0.05, 0.1, 2);
  const auto xa = get(pa), xb = get(pb);
  // ggml's AdamW takes the betas as float: 1 - 0.999f = 0.00100005 moves each step by ~2e-5
  // relative against the double reference (as in both trainers without --lr-schedule)
  for (int i = 0; i < 3; i++) {
    CHECK(xa[i] == Catch::Approx(wa[i]).epsilon(1e-4));
    CHECK(xb[i] == Catch::Approx(wb[i]).epsilon(1e-4));
  }
  // state round trip
  const auto st = opt.get_state();
  CHECK_NOTHROW(opt.set_state(st));
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(be);
}
