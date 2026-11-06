/**
 * Per-model CUDA matmul precision in one process (MLIPCPP_PRECISION_*):
 * loads one GGUF once per listed precision through the C API, all before the
 * first predict, then predicts every instance on one structure and writes the
 * energy and forces of each, with the median time of a predict.
 *
 *   mlipcpp-precision-check <model.gguf> <structure.xyz> <out prefix> <prec>[,<prec>...]
 *       [--backend cuda|cpu|auto] [--warmup W] [--reps R] [--charge Q] [--spin S] [--late-tf32]
 *
 * prec: default | fp32 | tf32. Writes <out>.<i>.<prec>.txt: n_atoms, the energy
 * (eV), then n_atoms lines of forces (eV/A); prints one JSON line per instance.
 * --late-tf32: after the predicts, loads one more instance in TF32 and reports
 * whether that failed (it must in a process committed to full FP32 on CUDA).
 * tests/precision_check.py runs it in several processes and compares them.
 */

#include "mlipcpp/io.h"
#include "mlipcpp/mlipcpp.h"
#include "mlipcpp/system.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

mlipcpp_precision_t parse_precision(const std::string &s) {
  if (s == "default") return MLIPCPP_PRECISION_F32;
  if (s == "fp32") return MLIPCPP_PRECISION_F32_EXACT;
  if (s == "tf32") return MLIPCPP_PRECISION_TF32;
  throw std::runtime_error("precision: default, fp32 or tf32, not " + s);
}

mlipcpp_backend_t parse_backend(const std::string &s) {
  if (s == "cuda") return MLIPCPP_BACKEND_CUDA;
  if (s == "cpu") return MLIPCPP_BACKEND_CPU;
  return MLIPCPP_BACKEND_AUTO;
}

mlipcpp_model_t load(const char *path, mlipcpp_backend_t backend, mlipcpp_precision_t precision) {
  mlipcpp_model_options_t opts;
  mlipcpp_model_options_default(&opts);
  opts.backend = backend;
  opts.precision = precision;
  mlipcpp_model_t m = mlipcpp_model_create(&opts);
  if (!m) throw std::runtime_error(std::string("create: ") + mlipcpp_get_last_error());
  if (mlipcpp_model_load(m, path) != MLIPCPP_OK) {
    const std::string err = mlipcpp_get_last_error();
    mlipcpp_model_free(m);
    throw std::runtime_error(err);
  }
  return m;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <model.gguf> <structure.xyz> <out prefix> <prec>[,<prec>...] [--backend B] "
                 "[--warmup W] [--reps R] [--charge Q] [--spin S] [--late-tf32]\n",
                 argv[0]);
    return 2;
  }
  std::string backend_name = "auto";
  int warmup = 2, reps = 5, charge = 0, spin = 1;
  bool late_tf32 = false;
  for (int i = 5; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--backend") backend_name = next();
    else if (a == "--warmup") warmup = std::stoi(next());
    else if (a == "--reps") reps = std::stoi(next());
    else if (a == "--charge") charge = std::stoi(next());
    else if (a == "--spin") spin = std::stoi(next());
    else if (a == "--late-tf32") late_tf32 = true;
    else throw std::runtime_error("unknown option " + a);
  }
  try {
    std::vector<std::string> names;
    std::stringstream ss(argv[4]);
    for (std::string t; std::getline(ss, t, ',');) names.push_back(t);
    const mlipcpp_backend_t backend = parse_backend(backend_name);

    const mlipcpp::AtomicSystem sys = mlipcpp::io::read_xyz(argv[2]);
    const int n = sys.num_atoms();
    float cell[9];
    bool pbc[3] = {false, false, false};
    const mlipcpp::Cell *c = sys.cell();
    if (c)
      for (int i = 0; i < 3; i++) {
        pbc[i] = c->periodic[i];
        for (int j = 0; j < 3; j++) cell[3 * i + j] = c->matrix[i][j];
      }

    // every instance before the first predict (the process decides TF32 then)
    std::vector<mlipcpp_model_t> models;
    for (const auto &p : names) models.push_back(load(argv[1], backend, parse_precision(p)));

    mlipcpp_predict_options_t po;
    mlipcpp_predict_options_default(&po);
    po.charge = charge, po.spin_multiplicity = spin;
    std::vector<float> forces(3 * size_t(n));
    for (size_t k = 0; k < models.size(); k++) {
      std::vector<double> ms;
      float energy = 0;
      for (int r = 0; r < warmup + reps; r++) {
        mlipcpp_result_t res = nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        if (mlipcpp_predict_ptr_with_options(models[k], n, sys.positions(), sys.atomic_numbers(), c ? cell : nullptr,
                                             c ? pbc : nullptr, &po, &res) != MLIPCPP_OK)
          throw std::runtime_error(std::string("predict: ") + mlipcpp_get_last_error());
        mlipcpp_result_get_energy(res, &energy);
        mlipcpp_result_get_forces(res, forces.data(), n);
        const auto t1 = std::chrono::steady_clock::now();
        mlipcpp_result_free(res);
        if (r >= warmup) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      std::sort(ms.begin(), ms.end());
      const double med = ms.empty() ? 0.0 : ms[ms.size() / 2];
      const std::string out = std::string(argv[3]) + "." + std::to_string(k) + "." + names[k] + ".txt";
      FILE *f = std::fopen(out.c_str(), "w");
      if (!f) throw std::runtime_error("cannot write " + out);
      std::fprintf(f, "%d\n%.10g\n", n, double(energy));
      for (int i = 0; i < n; i++)
        std::fprintf(f, "%.9g %.9g %.9g\n", forces[3 * i], forces[3 * i + 1], forces[3 * i + 2]);
      std::fclose(f);
      std::printf("{\"instance\": %zu, \"precision\": \"%s\", \"n_atoms\": %d, \"energy\": %.10g, \"ms_median\": %.3f, "
                  "\"ms_min\": %.3f, \"reps\": %zu, \"backend\": \"%s\"}\n",
                  k, names[k].c_str(), n, double(energy), med, ms.empty() ? 0.0 : ms.front(), ms.size(),
                  mlipcpp_get_backend_name());
      std::fflush(stdout);
    }
    if (late_tf32) {
      bool failed = false;
      std::string why;
      try {
        mlipcpp_model_free(load(argv[1], backend, MLIPCPP_PRECISION_TF32));
      } catch (const std::exception &e) {
        failed = true, why = e.what();
      }
      for (auto &ch : why)
        if (ch == '"' || ch == '\n') ch = '\'';
      std::printf("{\"late_tf32_failed\": %s, \"error\": \"%s\"}\n", failed ? "true" : "false", why.c_str());
    }
    for (auto m : models) mlipcpp_model_free(m);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
