// Conservative PET-LR training with analytic mixed derivatives or diagnostic
// directional finite differences. Network forward/backward runs on the selected
// backend; P3M and Adam run on CPU.
#include "runtime/graph_model.h"
#include "train/conservative.h"
#include "train/dataset.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <ggml-cpu.h>
#include <gguf.h>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <stdexcept>

using mlipcpp::AtomicSystem;
using mlipcpp::runtime::GraphModel;

namespace {
struct Parameters {
  std::vector<ggml_tensor *> tensors;
  std::vector<float> values;
  explicit Parameters(GraphModel &model) {
    const auto all = model.weights();
    for (const auto &name : model.param_names()) {
      auto it = std::find_if(all.begin(), all.end(),
                             [&](auto &w) { return w.first == name; });
      if (it == all.end() || it->second->type != GGML_TYPE_F32)
        throw std::runtime_error("Missing or non-F32 trainable tensor: " +
                                 name);
      tensors.push_back(it->second);
      size_t offset = values.size();
      values.resize(offset + ggml_nelements(it->second));
      ggml_backend_tensor_get(it->second, values.data() + offset, 0,
                              ggml_nbytes(it->second));
    }
    if (values.empty())
      throw std::runtime_error("No trainable parameters");
  }
  void upload(const std::vector<float> &v) {
    size_t offset = 0;
    for (auto *t : tensors) {
      ggml_backend_tensor_set(t, v.data() + offset, 0, ggml_nbytes(t));
      offset += ggml_nelements(t);
    }
  }
  void save(const std::string &source, const std::string &path,
            double displacement, bool analytic = false) {
    ggml_context *data = nullptr;
    auto *in = gguf_init_from_file(source.c_str(), {false, &data});
    if (!in)
      throw std::runtime_error("Cannot read source checkpoint");
    size_t offset = 0;
    for (auto *p : tensors) {
      auto *t = ggml_get_tensor(data, ggml_get_name(p));
      if (!t || ggml_nbytes(t) != ggml_nbytes(p))
        throw std::runtime_error("Checkpoint tensor mismatch");
      std::memcpy(t->data, values.data() + offset, ggml_nbytes(t));
      offset += ggml_nelements(t);
    }
    auto *out = gguf_init_empty();
    gguf_set_kv(out, in);
    for (int64_t i = 0; i < gguf_get_n_tensors(in); ++i)
      gguf_add_tensor(out, ggml_get_tensor(data, gguf_get_tensor_name(in, i)));
    gguf_set_val_str(out, "train.method",
                     analytic ? "conservative-analytic-jvp"
                              : "conservative-directional-finite-difference");
    if (analytic)
      gguf_remove_key(out, "train.fd_displacement");
    else
      gguf_set_val_f64(out, "train.fd_displacement", displacement);
    const auto tmp = path + ".tmp";
    bool ok = gguf_write_to_file(out, tmp.c_str(), false);
    gguf_free(out);
    gguf_free(in);
    ggml_free(data);
    if (!ok)
      throw std::runtime_error("Cannot write checkpoint");
    std::filesystem::rename(tmp, path);
  }
};

struct Objective {
  double loss = 0, energy_error = 0, force_rmse = 0;
  std::vector<float> gradient;
};

Objective objective(GraphModel &model, const AtomicSystem &system,
                    double energy, const float *forces, double ew, double fw,
                    double displacement, bool backward) {
  Objective out;
  auto prediction =
      backward ? model.predict_with_parameter_gradients(system, out.gradient)
               : model.predict(system, true);
  const int n = system.num_atoms();
  if (!prediction.has_forces)
    throw std::runtime_error("Conservative forces unavailable");
  out.energy_error = (prediction.energy - energy) / n;
  out.loss = ew * out.energy_error * out.energy_error;
  std::vector<double> direction(3 * n);
  double max_direction = 0;
  for (int i = 0; i < 3 * n; ++i) {
    const double r = prediction.forces[i] - forces[i];
    out.force_rmse += r * r / (3 * n);
    direction[i] = 2 * fw * r / (3 * n);
    max_direction = std::max(max_direction, std::abs(direction[i]));
  }
  out.loss += fw * out.force_rmse;
  out.force_rmse = std::sqrt(out.force_rmse);
  if (!std::isfinite(out.loss))
    throw std::runtime_error("Non-finite loss");
  if (!backward)
    return out;
  for (auto &g : out.gradient)
    g *= 2 * ew * out.energy_error / n;
  if (max_direction > 0) {
    // F=-dE/dR, so dL_F/dtheta = -D_R(dE/dtheta)[dL_F/dF].
    // Freeze the residual direction, and bound each displacement by epsilon.
    auto plus = system, minus = system;
    for (int i = 0; i < 3 * n; ++i) {
      const double delta = displacement * direction[i] / max_direction;
      plus.positions_mut()[i] += delta;
      minus.positions_mut()[i] -= delta;
    }
    std::vector<float> gp, gm;
    model.predict_with_parameter_gradients(plus, gp);
    model.predict_with_parameter_gradients(minus, gm);
    for (size_t i = 0; i < out.gradient.size(); ++i)
      out.gradient[i] -=
          max_direction * (double(gp[i]) - gm[i]) / (2 * displacement);
  }
  for (float g : out.gradient)
    if (!std::isfinite(g))
      throw std::runtime_error("Non-finite parameter gradient");
  if (!std::isfinite(out.loss))
    throw std::runtime_error("Non-finite loss");
  return out;
}

double norm(const std::vector<float> &v) {
  double result = 0;
  for (double x : v)
    result += x * x;
  return std::sqrt(result);
}
} // namespace

int main(int argc, char **argv) try {
  std::map<std::string, std::string> args;
  bool check = false, evaluate = false;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--evaluate") {
      evaluate = true;
      continue;
    }
    if (key == "--check-gradient") {
      check = true;
      continue;
    }
    if (key == "--help") {
      fmt::print(
          "Experimental native conservative PET-LR trainer\n"
          "--init MODEL --data MEMMAP_SPLIT --output GGUF [--backend "
          "metal|cpu]\n"
          "[--steps 20] [--lr 0.0001] [--fd-step 0.01] [--energy-weight 1]\n"
          "[--force-weight 1] [--seed 42] [--check-gradient | --evaluate]\n"
          "[--validation SPLIT --eval-every 255] [--index 0 --dump-gradient "
          "JSON]\n"
          "--evaluate reports metrics on the entire split without updates.\n"
          "[--method analytic|fd] [--batch-size 1] [--threads 2] (analytic "
          "default)\n"
          "Starts from checkpoint scaling. P3M and "
          "Adam on CPU.\n"
          "FD is a diagnostic alternative. No optimizer "
          "resume.\n");
      return 0;
    }
    if (i + 1 == argc)
      throw std::runtime_error("Missing value for " + key);
    if (key != "--init" && key != "--data" && key != "--output" &&
        key != "--backend" && key != "--steps" && key != "--lr" &&
        key != "--fd-step" && key != "--energy-weight" &&
        key != "--force-weight" && key != "--seed" && key != "--validation" &&
        key != "--eval-every" && key != "--index" && key != "--dump-gradient" &&
        key != "--method" && key != "--batch-size" && key != "--threads")
      throw std::runtime_error("Unknown option: " + key);
    args[key] = argv[++i];
  }
  auto get = [&](const std::string &key, const std::string &fallback) {
    auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
  };
  if (!args.count("--init") || !args.count("--data") ||
      (!check && !evaluate && !args.count("--output")))
    throw std::runtime_error("Required: --init MODEL --data MEMMAP_SPLIT "
                             "--output GGUF (see --help)");
  const bool analytic = get("--method", "analytic") == "analytic";
  if (get("--method", "analytic") != "analytic" &&
      get("--method", "analytic") != "fd")
    throw std::runtime_error("method must be analytic or fd");
  const int batch_size = std::stoi(get("--batch-size", "1"));
  if (batch_size < 1 || (!analytic && batch_size != 1))
    throw std::runtime_error("Invalid batch size (FD supports one)");
  const auto backend = get("--backend", "metal");
  if (check && evaluate)
    throw std::runtime_error("Choose check-gradient or evaluate");
  if (backend != "cpu" && backend != "metal")
    throw std::runtime_error("Backend must be cpu or metal");
  const int steps = std::stoi(get("--steps", "20"));
  const double lr = std::stod(get("--lr", "0.0001"));
  const double eps = std::stod(get("--fd-step", "0.01"));
  const double ew = std::stod(get("--energy-weight", "1")),
               fw = std::stod(get("--force-weight", "1"));
  if (steps < 1 || !std::isfinite(lr) || lr <= 0 || !std::isfinite(eps) ||
      eps <= 0 || !std::isfinite(ew) || ew < 0 || !std::isfinite(fw) ||
      fw < 0 || ew + fw == 0)
    throw std::runtime_error("Invalid training options");
  if (args.count("--output") &&
      std::filesystem::absolute(args["--output"]).lexically_normal() ==
          std::filesystem::absolute(args["--init"]).lexically_normal())
    throw std::runtime_error("Output must differ from input checkpoint");
  GraphModel model;
  model.set_backend_preference(backend == "metal"
                                   ? mlipcpp::BackendPreference::Metal
                                   : mlipcpp::BackendPreference::CPU);
  if (!model.load_from_gguf(args["--init"]))
    throw std::runtime_error("Failed to load model");
  if (!model.long_range() || model.nc_forces())
    throw std::runtime_error("Requires conservative PET-LR export");
  const int threads = std::stoi(get("--threads", "2"));
  if (threads < 1)
    throw std::runtime_error("threads must be positive");
  if (backend == "cpu")
    ggml_backend_cpu_set_n_threads(model.backend(), threads);
  Parameters parameters(model);
  mlipcpp::train::MemmapSplit data(args["--data"]);
  if (data.size() == 0)
    throw std::runtime_error("Empty dataset");
  fmt::print("backend={} parameters={} structures={} fd_step={} A\n",
             ggml_backend_name(model.backend()), parameters.values.size(),
             data.size(), eps);
  std::unique_ptr<mlipcpp::train::ConservativeTrainer> trainer;
  if (analytic)
    trainer = std::make_unique<mlipcpp::train::ConservativeTrainer>(model);
  auto analytic_objective = [&](const std::vector<int64_t> &ids,
                                bool backward) {
    std::vector<AtomicSystem> systems;
    std::vector<double> energies;
    std::vector<float> forces;
    for (auto id : ids) {
      systems.push_back(data.system(id));
      energies.push_back(data.energy(id));
      forces.insert(forces.end(), data.forces(id),
                    data.forces(id) + 3 * data.n_atoms(id));
    }
    auto r = trainer->evaluate(systems, energies, forces, ew, fw, backward);
    Objective o;
    o.loss = r.loss;
    o.energy_error = r.energy_error;
    o.force_rmse = r.force_rmse;
    o.gradient = std::move(r.gradient);
    return o;
  };
  fmt::print("method={} batch_size={} cpu_threads={} p3m_threads={}\n",
             analytic ? "analytic" : "fd", batch_size, threads,
             analytic ? 2 : 1);
  if (evaluate) {
    double energy_se = 0, force_se = 0;
    int64_t components = 0;
    for (int64_t i = 0; i < data.size(); ++i) {
      auto result = objective(model, data.system(i), data.energy(i),
                              data.forces(i), ew, fw, eps, false);
      energy_se += result.energy_error * result.energy_error;
      force_se += result.force_rmse * result.force_rmse * 3 * data.n_atoms(i);
      components += 3 * data.n_atoms(i);
    }
    fmt::print(
        "evaluation structures={} energy_rmse={} meV/atom force_rmse={} eV/A\n",
        data.size(), 1000 * std::sqrt(energy_se / data.size()),
        std::sqrt(force_se / components));
    return 0;
  }
  if (check) {
    const int64_t index = std::stoll(get("--index", "0"));
    if (index < 0 || index >= data.size())
      throw std::runtime_error("Invalid check index");
    const auto system = data.system(index);
    auto g = analytic ? analytic_objective({index}, true)
                      : objective(model, system, data.energy(index),
                                  data.forces(index), ew, fw, eps, true);
    auto fine = objective(model, system, data.energy(index), data.forces(index),
                          ew, fw, analytic ? eps : eps / 2, true);
    std::vector<float> difference(g.gradient.size());
    for (size_t i = 0; i < difference.size(); ++i)
      difference[i] = g.gradient[i] - fine.gradient[i];
    if (args.count("--dump-gradient")) {
      nlohmann::json dump;
      dump["loss"] = g.loss;
      dump["energy_error"] = g.energy_error;
      dump["force_rmse"] = g.force_rmse;
      dump["index"] = index;
      dump["fd_step"] = eps;
      dump["method"] = analytic ? "analytic" : "fd";
      size_t offset = 0;
      for (auto *t : parameters.tensors) {
        size_t count = ggml_nelements(t);
        dump["gradients"][ggml_get_name(t)] = std::vector<float>(
            g.gradient.begin() + offset, g.gradient.begin() + offset + count);
        offset += count;
      }
      std::ofstream stream(args["--dump-gradient"]);
      stream << dump.dump();
      if (!stream)
        throw std::runtime_error("Could not write gradient dump");
    }
    const double convergence =
        norm(difference) / std::max(norm(fine.gradient), 1e-12);
    const double gn = norm(g.gradient), h = 0.002;
    if (gn < 1e-9)
      throw std::runtime_error("Gradient check has no signal");
    auto probe = parameters.values;
    for (size_t i = 0; i < probe.size(); ++i)
      probe[i] += h * g.gradient[i] / gn;
    parameters.upload(probe);
    double lp = objective(model, system, data.energy(index), data.forces(index),
                          ew, fw, eps, false)
                    .loss;
    for (size_t i = 0; i < probe.size(); ++i)
      probe[i] = parameters.values[i] - h * g.gradient[i] / gn;
    parameters.upload(probe);
    double lm = objective(model, system, data.energy(index), data.forces(index),
                          ew, fw, eps, false)
                    .loss;
    parameters.upload(parameters.values);
    const double numerical = (lp - lm) / (2 * h),
                 relative = std::abs(numerical - gn) / gn;
    fmt::print("loss={} gradient_norm={} numerical_directional={} "
               "relative_error={} fd_comparison_relative={}\n",
               g.loss, gn, numerical, relative, convergence);
    if (!std::isfinite(relative) || !std::isfinite(convergence) ||
        relative > 0.05 || convergence > 0.05)
      throw std::runtime_error("Gradient verification failed (5% tolerance)");
    return 0;
  }
  std::vector<int64_t> order(data.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(std::stoul(get("--seed", "42")));
  std::shuffle(order.begin(), order.end(), rng);
  const int eval_every = std::stoi(get("--eval-every", "255"));
  if (eval_every < 1)
    throw std::runtime_error("eval-every must be positive");
  std::unique_ptr<mlipcpp::train::MemmapSplit> validation;
  if (args.count("--validation"))
    validation =
        std::make_unique<mlipcpp::train::MemmapSplit>(args["--validation"]);
  if (validation && !validation->size())
    throw std::runtime_error("Empty validation set");
  double best_loss = std::numeric_limits<double>::infinity();
  const auto started = std::chrono::steady_clock::now();
  auto validate = [&](int step) {
    if (!validation)
      return;
    double energy_se = 0, force_se = 0;
    int64_t components = 0;
    for (int64_t i = 0; i < validation->size(); ++i) {
      auto result =
          objective(model, validation->system(i), validation->energy(i),
                    validation->forces(i), ew, fw, eps, false);
      energy_se += result.energy_error * result.energy_error;
      force_se +=
          result.force_rmse * result.force_rmse * 3 * validation->n_atoms(i);
      components += 3 * validation->n_atoms(i);
    }
    double score =
        ew * energy_se / validation->size() + fw * force_se / components;
    if (score < best_loss) {
      best_loss = score;
      parameters.save(args["--init"], args["--output"] + ".best.gguf", eps,
                      analytic);
    }
    fmt::print("validation step={} energy_rmse={} meV/atom force_rmse={} eV/A "
               "loss={} elapsed_seconds={}\n",
               step, 1000 * std::sqrt(energy_se / validation->size()),
               std::sqrt(force_se / components), score,
               std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             started)
                   .count());
    std::fflush(stdout);
  };
  validate(0);
  std::vector<double> m(parameters.values.size()), v(m.size());
  size_t cursor = 0;
  for (int step = 0; step < steps; ++step) {
    auto step_start = std::chrono::steady_clock::now();
    std::vector<int64_t> ids;
    for (int b = 0; b < batch_size; ++b) {
      if (cursor == order.size()) {
        cursor = 0;
        std::shuffle(order.begin(), order.end(), rng);
      }
      ids.push_back(order[cursor++]);
    }
    const auto id = ids.front();
    auto result = analytic ? analytic_objective(ids, true)
                           : objective(model, data.system(id), data.energy(id),
                                       data.forces(id), ew, fw, eps, true);
    const double gn = norm(result.gradient), clip = 1 / std::max(gn, 1.0);
    for (size_t i = 0; i < m.size(); ++i) {
      const double g = result.gradient[i] * clip;
      m[i] = 0.9 * m[i] + 0.1 * g;
      v[i] = 0.999 * v[i] + 0.001 * g * g;
      parameters.values[i] -=
          lr * (m[i] / (1 - std::pow(0.9, step + 1))) /
          (std::sqrt(v[i] / (1 - std::pow(0.999, step + 1))) + 1e-8);
      if (!std::isfinite(parameters.values[i]))
        throw std::runtime_error("Non-finite updated parameter");
    }
    parameters.upload(parameters.values);
    fmt::print("step={} system={} loss={:.8g} energy_error={:.6g} eV/atom "
               "force_rmse={:.6g} eV/A gradient_norm={:.6g} seconds={:.6f}\n",
               step + 1, id, result.loss, result.energy_error,
               result.force_rmse, gn,
               std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             step_start)
                   .count());
    std::fflush(stdout);
    if ((step + 1) % eval_every == 0 || step + 1 == steps) {
      parameters.save(args["--init"],
                      args["--output"] + ".step" + std::to_string(step + 1) +
                          ".gguf",
                      eps, analytic);
      validate(step + 1);
    }
  }
  parameters.save(args["--init"], args["--output"], eps, analytic);
  fmt::print("Saved {}\n", args["--output"]);
  return 0;
} catch (const std::exception &e) {
  fmt::print(stderr, "error: {}\n", e.what());
  return 1;
}
