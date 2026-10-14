// Gradient-parity probe: loss and parameter gradients of one fixed,
// unrotated batch (structures 0..n-1 of <data>/train), written for
// scripts/train/check_gradients.py to compare against PyTorch autograd.
//
//   mlipcpp-train-check --data DIR --init INIT.gguf --out DIR [--n 64] [--gpus 1]
#include "train/crash.h"
#include "train/trainer.h"

#include <fmt/core.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <fstream>
#include <numeric>
#include <string>

using namespace mlipcpp::train;

int main(int argc, char **argv) {
  mlipcpp::train::install_crash_backtrace();
  std::string data, init, out, cache_dir;
  int n = 64, gpus = 1;
  bool node_grads_on = false, mb_failed = false, profile = false;
  double tile_ratio = 1.25;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--data") data = v;
    else if (a == "--init") init = v;
    else if (a == "--out") out = v;
    else if (a == "--cache-dir") cache_dir = v;
    else if (a == "--n") n = std::stoi(v);
    else if (a == "--gpus") gpus = std::stoi(v);
    else if (a == "--node-grads") node_grads_on = v == "1";
    else if (a == "--profile") profile = v == "1";
    else if (a == "--tile-nb-ratio") tile_ratio = std::stod(v);
  }
  if (data.empty() || init.empty() || out.empty()) {
    fmt::print(stderr, "usage: {} --data DIR --init INIT.gguf --out DIR [--n 64] [--gpus 1] [--cache-dir DIR] [--node-grads 0|1]\n", argv[0]);
    return 2;
  }
  MemmapSplit split(data + "/train");
  Worker worker(init, gpus > 0 ? 0 : -1);
  const TargetScaling sc = TargetScaling::fit(split);
  NeighborCache nl = NeighborCache::load_or_build(
      split, worker.model().cutoff(), cache_path(cache_dir, split, worker.model().cutoff()), 8);
  std::vector<int64_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  Batch b = make_batch(split, nl, ids, {}, worker.model().species_to_index(), sc, worker.model().cutoff());

  // gradient of every IR node output, for scripts/train/compare_node_grads.py
  // grad points into the compute context, which run() frees: copy ne and data in after_compute
  struct NodeGrad { int id; std::string name, op; ggml_tensor *grad; std::vector<float> data; std::vector<int64_t> ne; };
  std::vector<NodeGrad> node_grads;
  auto &interp = worker.model().interpreter();
  // keeping every node gradient alive defeats memory reuse: small batches only
  if (node_grads_on) worker.after_build = [&](ggml_cgraph *gf) {
    std::map<int, const mlipcpp::runtime::GIRNode *> by_id;
    for (const auto &node : interp.graph().nodes) by_id[node.id] = &node;
    for (auto [id, t] : interp.node_outputs()) {
      if (!t) continue; // pass-through nodes (SPLIT, CHUNK) have no tensor
      ggml_tensor *g = ggml_graph_get_grad(gf, t);
      if (!g || !by_id.count(id) || g->type != GGML_TYPE_F32) continue;
      for (ggml_tensor *v = g; v; v = v->view_src) ggml_set_output(v); // keep view sources
      node_grads.push_back({id, by_id[id]->name, by_id[id]->op, g, {}, {}});
    }
  };
  if (node_grads_on) worker.after_compute = [&](ggml_cgraph *gf) {
    // only gradients computed as nodes of this graph hold valid data; views
    // of tensors that were never allocated carry dangling pointers
    std::set<ggml_tensor *> computed;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) computed.insert(ggml_graph_node(gf, i));
    for (auto &ng : node_grads) {
      if (!computed.count(ng.grad) || !ng.grad->buffer || !ng.grad->data ||
          !ggml_is_contiguous(ng.grad)) continue;
      ng.ne.assign(ng.grad->ne, ng.grad->ne + 4);
      ng.data.resize(ggml_nelements(ng.grad));
      ggml_backend_tensor_get(ng.grad, ng.data.data(), 0, ggml_nbytes(ng.grad));
    }
  };

  worker.zero_grads();
  StepStats st = worker.run(b, sc.energy_scale, true);
  const std::vector<float> grads = worker.get_grads();
  fmt::print("loss {:.8g} (E {:.8g}, F {:.8g}); {} atoms, {} neighbours max; "
             "build {:.1f} ms, alloc {:.1f} ms, compute {:.1f} ms\n",
             st.loss, st.loss_energy, st.loss_force, b.n_atoms, b.max_neighbors,
             1e3 * worker.t_build, 1e3 * worker.t_alloc, 1e3 * worker.t_compute);

  // micro-batching must reproduce the full-batch loss and gradients
  if (!node_grads_on) {
    auto mbs = make_microbatches(split, nl, ids, {}, worker.model().species_to_index(), sc,
                                 worker.model().cutoff(), 5e7, 2.0);
    double loss = 0;
    worker.zero_grads(); // micro-batch gradients accumulate on the device
    for (const Batch &mb : mbs) loss += worker.run(mb, sc.energy_scale, true).loss;
    const std::vector<float> sum = worker.get_grads();
    double worst = 0;
    size_t off = 0;
    for (auto *p : worker.params()) {
      double d2 = 0, r2 = 0;
      for (int64_t i = 0; i < ggml_nelements(p); i++, off++)
        d2 += std::pow(double(sum[off]) - grads[off], 2), r2 += std::pow(double(grads[off]), 2);
      if (r2 > 0) worst = std::max(worst, std::sqrt(d2 / r2));
    }
    fmt::print("micro-batches: {} (growth 2), loss {:.8g} vs {:.8g}, worst relative gradient "
               "difference {:.2e}\n", mbs.size(), loss, st.loss, worst);
    if (!(worst < 1e-3)) mb_failed = true;
  }

  // step packing for 2 workers (cost-model cuts, longest-first dealing): all
  // micro-batches of both "workers" run here; their summed gradient / 2 is
  // the full-batch gradient, and the metrics add up exactly
  if (!node_grads_on) {
    auto per = make_step_batches(split, nl, ids, {}, worker.model().species_to_index(), sc,
                                 worker.model().cutoff(), 2, CostModel{}, 5.0, 5e7);
    double loss = 0;
    StepStats sum_st;
    std::string plan;
    worker.zero_grads();
    for (size_t w = 0; w < per.size(); w++) {
      plan += fmt::format(" w{}:", w);
      for (const Batch &mb : per[w]) {
        StepStats s1 = worker.run(mb, sc.energy_scale, true);
        loss += s1.loss, sum_st += s1;
        plan += fmt::format(" {}x{}", mb.n_atoms, mb.max_neighbors);
      }
    }
    const std::vector<float> sum = worker.get_grads();
    double worst = 0;
    size_t off = 0;
    for (auto *p : worker.params()) {
      double d2 = 0, r2 = 0;
      for (int64_t i = 0; i < ggml_nelements(p); i++, off++)
        d2 += std::pow(0.5 * sum[off] - grads[off], 2), r2 += std::pow(double(grads[off]), 2);
      if (r2 > 0) worst = std::max(worst, std::sqrt(d2 / r2));
    }
    auto close = [](double a, double b) { return std::abs(a - b) <= 1e-4 * std::abs(b); };
    fmt::print("step packing (2 workers):{}; loss/2 {:.8g} vs {:.8g}, worst relative gradient "
               "difference {:.2e}; se E {:.6g} vs {:.6g}, se F {:.6g} vs {:.6g}\n", plan, loss / 2,
               st.loss, worst, sum_st.se_energy_per_atom, st.se_energy_per_atom, sum_st.se_force,
               st.se_force);
    if (!(worst < 1e-3) || !close(loss / 2, st.loss) || !close(sum_st.se_force, st.se_force) ||
        !close(sum_st.se_energy_per_atom, st.se_energy_per_atom) || sum_st.n_systems != st.n_systems ||
        sum_st.n_force_components != st.n_force_components)
      mb_failed = true;
  }

  // tiles: padded fixed-shape micro-batches through a graph-caching worker
  // (CUDA graphs replay from the second call on) must give the same loss and
  // gradients on every repetition
  if (!node_grads_on && tile_ratio > 1) {
    Worker tiled(init, gpus > 0 ? 0 : -1, 256);
    tiled.set_params(worker.get_params());
    const Tiling tiling{tile_ratio, 1.25, n};
    const double growths[] = {2.0, 1e9};
    for (double growth : growths) {
      auto mbs = make_microbatches(split, nl, ids, {}, worker.model().species_to_index(), sc,
                                   worker.model().cutoff(), 5e7, growth, tiling);
      for (int rep = 0; rep < 4; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        double loss = 0;
        StepStats sum_st;
        tiled.zero_grads();
        for (const Batch &mb : mbs) {
          StepStats s1 = tiled.run(mb, sc.energy_scale, true);
          loss += s1.loss, sum_st += s1;
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const std::vector<float> sum = tiled.get_grads();
        double worst = 0;
        size_t off = 0;
        for (auto *p : worker.params()) {
          double d2 = 0, r2 = 0;
          for (int64_t i = 0; i < ggml_nelements(p); i++, off++)
            d2 += std::pow(double(sum[off]) - grads[off], 2), r2 += std::pow(double(grads[off]), 2);
          if (r2 > 0) worst = std::max(worst, std::sqrt(d2 / r2));
        }
        std::string shapes;
        for (const Batch &mb : mbs) shapes += fmt::format(" {}x{}", mb.n_atoms, mb.max_neighbors);
        fmt::print("tiles (growth {:g}, rep {}):{}; loss {:.8g} vs {:.8g}, worst relative gradient "
                   "difference {:.2e}; se E {:.6g} vs {:.6g}, se F {:.6g} vs {:.6g}, {} systems, {} force "
                   "components; wall {:.1f} ms\n", growth, rep, shapes, loss, st.loss, worst,
                   sum_st.se_energy_per_atom, st.se_energy_per_atom, sum_st.se_force, st.se_force,
                   sum_st.n_systems, sum_st.n_force_components, 1e3 * wall);
        auto close = [](double a, double b) { return std::abs(a - b) <= 1e-4 * std::abs(b); };
        if (!(worst < 1e-3) || !close(loss, st.loss) || !close(sum_st.se_force, st.se_force) ||
            !close(sum_st.se_energy_per_atom, st.se_energy_per_atom) ||
            sum_st.n_systems != st.n_systems || sum_st.n_force_components != st.n_force_components)
          mb_failed = true;
      }
    }
  }

  if (profile) {
    // one training step of this batch as micro-batches: wall time, then per op
    auto mbs = make_microbatches(split, nl, ids, {}, worker.model().species_to_index(), sc,
                                 worker.model().cutoff(), 5e7, 2.0);
    for (int rep = 0; rep < 3; rep++) { // warm-up, then timed
      auto t0 = std::chrono::steady_clock::now();
      double compute = 0;
      worker.zero_grads();
      for (const Batch &mb : mbs) worker.run(mb, sc.energy_scale, true), compute += worker.t_compute;
      fmt::print("step ({} micro-batches): wall {:.1f} ms, compute {:.1f} ms\n", mbs.size(),
                 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 1e3 * compute);
    }
    worker.profile = true;
    for (const Batch &mb : mbs) worker.run(mb, sc.energy_scale, true);
    std::vector<std::pair<double, std::string>> rows;
    double total = 0;
    for (auto &[op, v] : worker.profile_ms) rows.push_back({v.first, op}), total += v.first;
    std::sort(rows.rbegin(), rows.rend());
    fmt::print("profile (node by node, synchronised; total {:.1f} ms):\n", total);
    for (size_t k = 0; k < std::min<size_t>(rows.size(), 30); k++)
      fmt::print("  {:>8.2f} ms {:5.1f}%  {:<24} x{}\n", rows[k].first, 100 * rows[k].first / total,
                 rows[k].second, worker.profile_ms[rows[k].second].second);
    worker.profile = false;
  }

  std::filesystem::create_directories(out);
  nlohmann::json meta;
  meta["ids"] = ids;
  meta["loss"] = st.loss, meta["loss_energy"] = st.loss_energy, meta["loss_force"] = st.loss_force;
  meta["energy_scale"] = sc.energy_scale;
  for (auto [Z, c] : sc.composition) meta["composition"][std::to_string(Z)] = c;
  for (auto [Z, s] : sc.force_scale) meta["force_scale"][std::to_string(Z)] = s;
  size_t off = 0;
  for (auto *p : worker.params()) {
    meta["params"].push_back({{"name", ggml_get_name(p)}, {"offset", off}, {"size", ggml_nelements(p)}});
    off += ggml_nelements(p);
  }
  auto dump = [&](const std::string &name, const auto &v) {
    std::ofstream(out + "/" + name + ".bin", std::ios::binary)
        .write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(v[0]));
  };
  meta["batch"] = {{"n_atoms", b.n_atoms}, {"n_systems", b.n_systems}, {"max_neighbors", b.max_neighbors}};
  dump("species", b.species), dump("neighbor_species", b.neighbor_species);
  dump("edge_vectors", b.edge_vectors), dump("padding_mask", b.padding_mask);
  dump("reverse_neighbor_index", b.reverse_neighbor_index), dump("cutoff_values", b.cutoff_values);
  dump("atom_weight", b.atom_weight), dump("energy_target", b.energy_target);
  dump("force_target", b.force_target);
  {
    std::ofstream f(out + "/node_grads.bin", std::ios::binary);
    size_t off = 0;
    for (auto &ng : node_grads) {
      if (ng.data.empty()) continue;
      meta["node_grads"].push_back({{"id", ng.id}, {"name", ng.name}, {"op", ng.op},
                                    {"ne", ng.ne}, {"offset", off}, {"size", ng.data.size()}});
      f.write(reinterpret_cast<const char *>(ng.data.data()), ng.data.size() * sizeof(float));
      off += ng.data.size();
    }
  }
  std::ofstream(out + "/check.json") << meta.dump(1);
  std::ofstream(out + "/grads.bin", std::ios::binary)
      .write(reinterpret_cast<const char *>(grads.data()), grads.size() * sizeof(float));
  fmt::print("wrote {}/check.json and grads.bin ({} parameters)\n", out, grads.size());
  return mb_failed ? 3 : 0;
}
