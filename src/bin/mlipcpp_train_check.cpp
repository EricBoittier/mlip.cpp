// Gradient-parity probe: loss and parameter gradients of one fixed,
// unrotated batch (structures 0..n-1 of <data>/train), written for
// scripts/train/check_gradients.py to compare against PyTorch autograd.
//
//   mlipcpp-train-check --data DIR --init INIT.gguf --out DIR [--n 64] [--gpus 1]
//                       [--conservative 1 --energy-weight W --force-weight W --nc-force-weight W]
#include "train/crash.h"
#include "train/trainer.h"

#include <cstdlib>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <fstream>
#include <numeric>
#include <random>
#include <string>

using namespace mlipcpp::train;

int main(int argc, char **argv) {
  mlipcpp::train::install_crash_backtrace();
  std::string data, init, out, cache_dir;
  int n = 64, gpus = 1;
  bool node_grads_on = false, mb_failed = false, profile = false, sweep = false;
  Worker::Conservative cons;
  double tile_ratio = 1.25, tol = 1e-3;
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
    else if (a == "--sweep") sweep = v == "1";
    else if (a == "--tile-nb-ratio") tile_ratio = std::stod(v);
    else if (a == "--tol") tol = std::stod(v); // max relative gradient difference
    else if (a == "--conservative") cons.on = v == "1";
    else if (a == "--energy-weight") cons.energy = std::stod(v);
    else if (a == "--force-weight") cons.force = std::stod(v);
    else if (a == "--nc-force-weight") cons.nc_force = std::stod(v);
  }
  // the node-by-node profile computes one-node graph views; ggml's CUDA graph
  // cache (keyed by the first node's address) could replay a stale capture
  // into a later graph's view -> no CUDA graphs when profiling (read once)
  if (profile) setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
  if (data.empty() || init.empty() || out.empty()) {
    fmt::print(stderr, "usage: {} --data DIR --init INIT.gguf --out DIR [--n 64] [--gpus 1] [--cache-dir DIR] [--node-grads 0|1] [--tol 1e-3] [--profile 0|1] [--sweep 0|1]\n", argv[0]);
    return 2;
  }
  MemmapSplit split(data + "/train");
  Worker worker(init, gpus > 0 ? 0 : -1);
  worker.conservative = cons;
  const TargetScaling sc = TargetScaling::fit(split);
  const AdaptiveCutoff adaptive = adaptive_cutoff(worker.model());
  NeighborCache nl = NeighborCache::load_or_build(
      split, worker.model().cutoff(), cache_path(cache_dir, split, worker.model().cutoff(), adaptive), 8, adaptive);
  nl.taper = cons.on && adaptive.enabled();
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
  fmt::print("loss {:.8g} (E {:.8g}, F {:.8g}{}); {} atoms, {} neighbours max; "
             "build {:.1f} ms, alloc {:.1f} ms, compute {:.1f} ms\n",
             st.loss, st.loss_energy, st.loss_force,
             cons.on ? fmt::format(", F_nc {:.8g}, {} taper edges", st.loss_nc, b.taper_center.size()) : "",
             b.n_atoms, b.max_neighbors,
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
    if (!(worst < tol)) mb_failed = true;
  }

  // step packing for 2 workers (cost-model cuts, longest-first dealing): all
  // micro-batches of both "workers" run here; their summed gradient / 2 is
  // the full-batch gradient, and the metrics add up exactly
  if (!node_grads_on) {
    auto per = make_step_batches(split, nl, ids, {}, worker.model().species_to_index(), sc,
                                 worker.model().cutoff(), 2, CostModel{}, 5e7);
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
    if (!(worst < tol) || !close(loss / 2, st.loss) || !close(sum_st.se_force, st.se_force) ||
        !close(sum_st.se_energy_per_atom, st.se_energy_per_atom) || sum_st.n_systems != st.n_systems ||
        sum_st.n_force_components != st.n_force_components)
      mb_failed = true;
  }

  // tiles: padded fixed-shape micro-batches through a graph-caching worker
  // (CUDA graphs replay from the second call on) must give the same loss and
  // gradients on every repetition
  if (!node_grads_on && tile_ratio > 1) {
    Worker tiled(init, gpus > 0 ? 0 : -1, 256);
    tiled.conservative = cons;
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
        if (!(worst < tol) || !close(loss, st.loss) || !close(sum_st.se_force, st.se_force) ||
            !close(sum_st.se_energy_per_atom, st.se_energy_per_atom) ||
            sum_st.n_systems != st.n_systems || sum_st.n_force_components != st.n_force_components)
          mb_failed = true;
      }
    }
  }

  if (sweep) {
    // FLOPs, ideal traffic and compute buffer of one training micro-batch, by
    // shape (graphs are only built and measured)
    fmt::print("sweep: n_atoms max_neighbors n_systems GFLOP GB_ideal buffer_GB\n");
    for (int n : {32, 128, 512, 2048})
      for (int m : {8, 16, 32, 64, 128, 256, 512, 1024, 2048}) {
        if (int64_t(n) * m > 1000000) continue; // the interpreter's sanity limit on n_edges
        const int S = std::max(1, n / 8);
        const auto c = worker.graph_cost(n, m, S);
        fmt::print("sweep {} {} {} {:.4f} {:.4f} {:.4f}\n", n, m, S, c.flops / 1e9, c.bytes / 1e9, c.buffer / 1e9);
      }
  }

  if (profile) {
    // one training step of this batch as micro-batches: wall time, then per op
    auto mbs = make_microbatches(split, nl, ids, {}, worker.model().species_to_index(), sc,
                                 worker.model().cutoff(), 5e7, 2.0);
    for (int rep = 0; rep < 3; rep++) { // warm-up, then timed
      auto t0 = std::chrono::steady_clock::now();
      double build = 0, alloc = 0, upload = 0, compute = 0, download = 0, cpu = 0;
      worker.zero_grads();
      for (const Batch &mb : mbs) {
        worker.run(mb, sc.energy_scale, true);
        build += worker.t_build, alloc += worker.t_alloc, upload += worker.t_upload;
        compute += worker.t_compute, download += worker.t_download, cpu += worker.t_cpu;
      }
      fmt::print("step ({} micro-batches): wall {:.1f} ms = build {:.1f} + alloc {:.1f} + upload {:.1f} + compute {:.1f} "
                 "+ download {:.1f} + host {:.1f} ms (+ the rest)\n", mbs.size(),
                 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 1e3 * build,
                 1e3 * alloc, 1e3 * upload, 1e3 * compute, 1e3 * download, 1e3 * cpu);
    }
    worker.profile = true;
    for (const Batch &mb : mbs) worker.run(mb, sc.energy_scale, true);
    std::vector<std::pair<double, std::string>> rows;
    Worker::OpStats tot;
    for (auto &[op, v] : worker.profile_ops)
      rows.push_back({v.ms, op}), tot.ms += v.ms, tot.flops += v.flops, tot.bytes += v.bytes;
    std::sort(rows.rbegin(), rows.rend());
    std::map<std::string, Worker::OpStats> phases; // by the label before ':'
    for (auto &[op, v] : worker.profile_ops) {
      auto &p = phases[op.substr(0, op.find(':'))];
      p.ms += v.ms, p.count += v.count, p.flops += v.flops, p.bytes += v.bytes;
    }
    fmt::print("profile by phase:\n");
    for (auto &[ph, v] : phases)
      fmt::print("  {:<6} {:>8.2f} ms {:5.1f}%  x{:<6} {:>9.2f} GFLOP {:>7.2f} TFLOP/s {:>8.2f} GB {:>6.0f} GB/s\n", ph,
                 v.ms, 100 * v.ms / tot.ms, v.count, v.flops / 1e9, v.flops / v.ms / 1e9, v.bytes / 1e9,
                 v.bytes / v.ms / 1e6);
    fmt::print("profile (node by node, synchronised; total {:.1f} ms, {:.1f} GFLOP, {:.1f} GB ideal traffic: "
               "{:.2f} TFLOP/s, {:.0f} GB/s):\n", tot.ms, tot.flops / 1e9, tot.bytes / 1e9,
               tot.flops / tot.ms / 1e9, tot.bytes / tot.ms / 1e6);
    for (size_t k = 0; k < std::min<size_t>(rows.size(), 40); k++) {
      const auto &v = worker.profile_ops[rows[k].second];
      fmt::print("  {:>8.2f} ms {:5.1f}%  {:<24} x{:<5} {:>9.2f} GFLOP {:>7.2f} TFLOP/s {:>8.2f} GB {:>6.0f} GB/s\n",
                 v.ms, 100 * v.ms / tot.ms, rows[k].second, v.count, v.flops / 1e9, v.flops / v.ms / 1e9,
                 v.bytes / 1e9, v.bytes / v.ms / 1e6);
    }
    std::vector<std::pair<double, std::string>> copies;
    for (auto &[k, v] : worker.profile_copies) copies.push_back({v.ms, k});
    std::sort(copies.rbegin(), copies.rend());
    fmt::print("copies by source:\n");
    for (size_t k = 0; k < std::min<size_t>(copies.size(), 20); k++) {
      const auto &v = worker.profile_copies[copies[k].second];
      fmt::print("  {:>8.2f} ms {:5.1f}%  x{:<5} {:>8.2f} GB  {}\n", v.ms, 100 * v.ms / tot.ms, v.count, v.bytes / 1e9,
                 copies[k].second);
    }
    auto &nodes = worker.profile_nodes;
    std::sort(nodes.rbegin(), nodes.rend());
    fmt::print("slowest nodes:\n");
    for (size_t k = 0; k < std::min<size_t>(nodes.size(), 25); k++)
      fmt::print("  {:>8.2f} ms {:5.1f}%  {}\n", nodes[k].first, 100 * nodes[k].first / tot.ms, nodes[k].second);
    for (const Batch &mb : mbs) // the profiled shapes
      fmt::print("  micro-batch {} atoms x {} neighbours, {} systems\n", mb.n_atoms, mb.max_neighbors, mb.n_systems);
    worker.profile = false;

    // the optimizer step (Adam, then Muon with 5 NS steps) on one more worker
    // of the same model: wall time per step, then node by node by op
    for (bool muon : {false, true}) {
      std::vector<std::unique_ptr<Worker>> one;
      one.push_back(std::make_unique<Worker>(init, gpus > 0 ? 0 : -1));
      Optimizer opt(one, 1.0, muon, 0.95, 10.0, 5);
      std::map<std::string, Worker::OpStats> ops;
      const double ms = opt.profile(20, ops);
      double total = 0;
      int64_t nodes = 0;
      std::vector<std::pair<double, std::string>> r;
      for (auto &[op, v] : ops) r.push_back({v.ms, op}), total += v.ms, nodes += v.count;
      std::sort(r.rbegin(), r.rend());
      fmt::print("optimizer step ({}): {:.2f} ms per step; {} nodes, {:.2f} ms node by node{}\n", muon ? "muon" : "adam", ms,
                 nodes, total, muon ? fmt::format("; groups {}", fmt::join(opt.muon_groups, " ")) : std::string());
      for (size_t k = 0; k < std::min<size_t>(r.size(), 12); k++) {
        const auto &v = ops[r[k].second];
        fmt::print("  {:>8.2f} ms  {:<16} x{:<5} {:>8.2f} GFLOP {:>8.3f} GB\n", v.ms, r[k].second, v.count, v.flops / 1e9,
                   v.bytes / 1e9);
      }
    }
  }

  // the Muon step split over two GPUs == the step on device 0
  // (MLIPCPP_MUON_ONE_DEVICE): the same random gradients through both for a
  // few steps, then the parameters compared; then each one's wall time per step
  if (gpus >= 2) {
    const char *env = std::getenv("MLIPCPP_MUON_ONE_DEVICE");
    const std::string saved = env ? env : "";
    auto run = [&](bool split) {
      if (split) unsetenv("MLIPCPP_MUON_ONE_DEVICE");
      else setenv("MLIPCPP_MUON_ONE_DEVICE", "1", 1);
      std::vector<std::unique_ptr<Worker>> ws;
      for (int d = 0; d < 2; d++) ws.push_back(std::make_unique<Worker>(init, d));
      Optimizer opt(ws, 1.0, true, 0.95, 10.0, 5);
      std::mt19937 rng(7);
      std::normal_distribution<float> normal(0.0f, 1.0f);
      std::vector<float> g(ggml_nelements(ws[0]->grad_flat()));
      std::vector<double> norms;
      for (int s = 0; s < 3; s++) {
        for (auto &w : ws) {
          for (float &x : g) x = normal(rng);
          ggml_backend_tensor_set(w->grad_flat(), g.data(), 0, g.size() * sizeof(float));
        }
        norms.push_back(opt.step(1e-3));
      }
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < 50; r++) opt.step(1e-6);
      const double ms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 50;
      return std::make_tuple(ws[0]->get_params(), ws[1]->get_params(), norms, ms, opt.muon_split);
    };
    const auto [a0, a1, an, ams, as] = run(false);
    const auto [b0, b1, bn, bms, bs] = run(true);
    if (saved.empty()) unsetenv("MLIPCPP_MUON_ONE_DEVICE");
    else setenv("MLIPCPP_MUON_ONE_DEVICE", saved.c_str(), 1);
    auto rel = [](const std::vector<float> &x, const std::vector<float> &y) {
      double d = 0, s = 0;
      for (size_t i = 0; i < x.size(); i++) d = std::max(d, double(std::abs(x[i] - y[i]))), s = std::max(s, double(std::abs(x[i])));
      return d / s;
    };
    const double diff = std::max(rel(a0, b0), rel(b0, b1));
    fmt::print("muon split ({}): max relative parameter difference to one device {:.2e} (workers {:.2e}), "
               "norms {} vs {}; {:.2f} ms per step vs {:.2f} on one device\n",
               bs, rel(a0, b0), rel(b0, b1), fmt::join(bn, " "), fmt::join(an, " "), bms, ams);
    if (!(diff < 1e-5)) {
      fmt::print("!! the Muon split changes the step\n");
      return 1;
    }
  }

  // the NCCL all-reduce with replicated optimizers == the step on device 0:
  // the same random gradients for a few steps, then the parameters; then
  // each one's wall time per step (gradients included: the all-reduce is part of it)
  if (gpus >= 2) {
    for (bool muon : {false, true}) {
      auto run = [&](bool allreduce) {
        std::vector<std::unique_ptr<Worker>> ws;
        for (int d = 0; d < gpus; d++) ws.push_back(std::make_unique<Worker>(init, d));
        Optimizer opt(ws, 1.0, muon, 0.95, 10.0, 5, allreduce);
        std::mt19937 rng(11);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> g(ggml_nelements(ws[0]->grad_flat()));
        auto fill = [&] {
          for (auto &w : ws) {
            for (float &x : g) x = normal(rng);
            ggml_backend_tensor_set(w->grad_flat(), g.data(), 0, g.size() * sizeof(float));
          }
        };
        for (int st = 0; st < 3; st++) fill(), opt.step(1e-3);
        const auto params = std::make_pair(ws[0]->get_params(), ws.back()->get_params());
        double ms = 0;
        for (int r = 0; r < 20; r++) {
          fill();
          for (auto &w : ws) ggml_backend_synchronize(w->model().backend());
          const auto t0 = std::chrono::steady_clock::now();
          for (size_t w = 0; w < ws.size(); w++) opt.stage(w);
          opt.step(1e-6);
          ms += 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
        }
        return std::make_tuple(params.first, params.second, ms, opt.allreduce_info);
      };
      const auto [a0, a1, ams, ainfo] = run(true);
      const auto [b0, b1, bms, binfo] = run(false);
      auto rel = [](const std::vector<float> &x, const std::vector<float> &y) {
        double d = 0, s = 0;
        for (size_t i = 0; i < x.size(); i++) d = std::max(d, double(std::abs(x[i] - y[i]))), s = std::max(s, double(std::abs(x[i])));
        return d / s;
      };
      fmt::print("allreduce ({}, {} GPUs; {}): max relative parameter difference to the device-0 step {:.2e}, "
                 "replicas {:.2e}; {:.2f} ms per step vs {:.2f}\n",
                 muon ? "muon" : "adam", gpus, ainfo, rel(a0, b0), rel(a0, a1), ams, bms);
      // Muon's default bf16 Newton-Schulz turns the summation-order difference
      // (~1e-7) into ~4e-5 after 3 steps (7.8e-9 with GGML_NS_F32=1)
      const double tol = muon && !std::getenv("GGML_NS_F32") ? 1e-4 : 1e-5;
      if (!(rel(a0, b0) < tol) || rel(a0, a1) != 0) {
        fmt::print("!! the all-reduce changes the step\n");
        return 1;
      }
    }
  }

  std::filesystem::create_directories(out);
  nlohmann::json meta;
  meta["ids"] = ids;
  meta["loss"] = st.loss, meta["loss_energy"] = st.loss_energy, meta["loss_force"] = st.loss_force;
  if (cons.on)
    meta["conservative"] = {{"energy_weight", cons.energy}, {"force_weight", cons.force},
                            {"nc_force_weight", cons.nc_force}, {"loss_nc", st.loss_nc}};
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
