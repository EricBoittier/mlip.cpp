// Energy heads (runtime/energy_heads.h) in training: head 1 of a two-head model must train exactly as the single-head
// model made of head 1's tensors and composition (loss, conservative forces, parameter gradients), the other head
// must get no gradient from head-1 structures, energy_weight 0 must remove a structure's energy term only, and the
// gradient of a head-1 readout weight must match central differences of the loss.
// Fixture (skipped without it): MLIPCPP_TEST_MULTIHEAD_DIR from scripts/multihead/make_test_fixture.py.
#include "train/batch.h"
#include "train/trainer.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <map>
#include <numeric>
#include <string>

using namespace mlipcpp::train;
using Catch::Approx;

namespace {
struct Result {
  StepStats st;
  std::map<std::string, std::vector<float>> grad; // by GGUF tensor name
};

TargetScaling scaling_of(const mlipcpp::runtime::GraphModel &m) {
  TargetScaling sc;
  for (auto [Z, c] : m.composition_energies()) sc.composition[Z] = c;
  for (auto [Z, f] : m.nc_force_scales()) sc.force_scale[Z] = f;
  sc.energy_scale = m.energy_scale();
  if (m.energy_heads().enabled()) {
    sc.head_composition.resize(m.energy_heads().n);
    for (int h = 0; h < m.energy_heads().n; h++)
      for (auto [Z, c] : m.energy_heads().composition[h]) sc.head_composition[h][Z] = c;
  }
  return sc;
}

Result run(const std::string &gguf, const std::string &data, const std::vector<float> *params = nullptr,
           bool backward = true) {
  Worker w(gguf, -1);
  w.conservative = {true, 1.0, 1.0, 0.0};
  w.energy_mask = true;
  if (params) w.set_params(*params);
  const auto &m = w.model();
  MemmapSplit split(data + "/train");
  const TargetScaling sc = scaling_of(m);
  const AdaptiveCutoff ad = adaptive_cutoff(m);
  NeighborCache nl = NeighborCache::load_or_build(split, m.cutoff(), "", 2, ad);
  nl.taper = ad.enabled();
  std::vector<int64_t> ids(split.size());
  std::iota(ids.begin(), ids.end(), 0);
  Batch b = make_batch(split, nl, ids, {}, m.species_to_index(), sc, m.cutoff());
  w.zero_grads();
  Result r;
  r.st = w.run(b, sc.energy_scale, backward);
  if (backward) {
    const std::vector<float> g = w.get_grads(); // parameter order
    size_t off = 0;
    for (const ggml_tensor *p : w.params()) {
      r.grad[ggml_get_name(p)] = std::vector<float>(g.begin() + off, g.begin() + off + ggml_nelements(p));
      off += ggml_nelements(p);
    }
  }
  return r;
}

double rel(const std::vector<float> &a, const std::vector<float> &b) {
  double d = 0, n = 0;
  for (size_t i = 0; i < a.size(); i++) d += std::pow(double(a[i]) - b[i], 2), n += double(b[i]) * b[i];
  return n > 0 ? std::sqrt(d / n) : std::sqrt(d);
}
} // namespace

TEST_CASE("energy heads: head 1 trains as the single-head model of its tensors", "[train][heads]") {
  const char *dir = std::getenv("MLIPCPP_TEST_MULTIHEAD_DIR");
  if (!dir) SKIP("MLIPCPP_TEST_MULTIHEAD_DIR not set (scripts/multihead/make_test_fixture.py)");
  const std::string D = dir;
  const Result two = run(D + "/mh.gguf", D + "/d1"), one = run(D + "/ref1.gguf", D + "/d0");
  // the same structures, losses and errors (head-1 tensors in the two-head graph = head-0 tensors of ref1)
  CHECK(two.st.loss_energy == Approx(one.st.loss_energy).epsilon(1e-5));
  CHECK(two.st.loss_force == Approx(one.st.loss_force).epsilon(1e-5));
  CHECK(two.st.se_force == Approx(one.st.se_force).epsilon(1e-5));
  CHECK(two.st.se_energy_per_atom == Approx(one.st.se_energy_per_atom).epsilon(1e-5));
  CHECK(two.st.n_energy_masked == 1);
  CHECK(two.st.h_ne[1] == 5);
  CHECK(two.st.h_ne[0] == 0);
  // gradients: backbone and head 1 match ref1's backbone and head 0; head 0 of the two-head model gets none
  int n_head = 0;
  for (const auto &[name, g] : one.grad) {
    const bool readout = two.grad.count("head1." + name) > 0;
    const auto &g2 = two.grad.at(readout ? "head1." + name : name);
    INFO(name);
    CHECK(rel(g2, g) < 1e-4);
    if (readout) {
      n_head++;
      double mx = 0;
      for (float x : two.grad.at(name)) mx = std::max(mx, double(std::abs(x)));
      CHECK(mx == 0.0);
    }
  }
  CHECK(n_head == 12);
}

TEST_CASE("energy heads: head-1 readout gradient against central differences", "[train][heads]") {
  const char *dir = std::getenv("MLIPCPP_TEST_MULTIHEAD_DIR");
  if (!dir) SKIP("MLIPCPP_TEST_MULTIHEAD_DIR not set (scripts/multihead/make_test_fixture.py)");
  const std::string D = dir;
  Worker w(D + "/mh.gguf", -1);
  std::vector<float> p0 = w.get_params();
  const Result base = run(D + "/mh.gguf", D + "/d1", &p0);
  // flat offset of head1.node_final_layers_0_weight in parameter order
  size_t off = 0;
  bool found = false;
  for (ggml_tensor *t : w.params()) {
    if (std::string(ggml_get_name(t)) == "head1.node_final_layers_0_weight") {
      found = true;
      break;
    }
    off += ggml_nelements(t);
  }
  REQUIRE(found);
  const auto &g = base.grad.at("head1.node_final_layers_0_weight");
  for (int k : {0, 7, 100}) {
    const float h = 1e-2f;
    std::vector<float> pp = p0, pm = p0;
    pp[off + k] += h, pm[off + k] -= h;
    const double lp = run(D + "/mh.gguf", D + "/d1", &pp, false).st.loss;
    const double lm = run(D + "/mh.gguf", D + "/d1", &pm, false).st.loss;
    const double fd = (lp - lm) / (2.0 * h);
    INFO("component " << k << ": analytic " << g[k] << ", FD " << fd);
    CHECK(g[k] == Approx(fd).epsilon(2e-2).margin(1e-6));
  }
}
