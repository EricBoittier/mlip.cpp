/**
 * Predictions of a pet-graph GGUF on structures of a training-format split,
 * through the inference engine (GraphModel), at full precision: for checking
 * mlip.cpp against metatrain / metatomic (scripts/train/predict_metatomic.py writes the
 * same files, compare_predictions.py compares them) and against mlipcpp-train.
 *
 *   mlipcpp-predict <model.gguf> <split dir> <ids.npy> <out prefix> [--backend cpu|cuda|...]
 *
 * Writes <out>.energy.bin (f64 [n], eV) and, for nc-force models,
 * <out>.nc_forces.bin (f32 [sum n_atoms, 3], eV/A), in the order of ids.npy.
 */

#include "core/backend.h"
#include "runtime/graph_model.h"
#include "train/dataset.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlipcpp;

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <model.gguf> <split dir> <ids.npy> <out prefix> [--backend <name>]\n";
    return 1;
  }
  std::string backend = "auto";
  for (int i = 5; i + 1 < argc; i++)
    if (std::string(argv[i]) == "--backend") backend = argv[++i];
  try {
    runtime::GraphModel model;
    model.set_backend_preference(parse_backend_preference(backend));
    if (!model.load_from_gguf(argv[1])) throw std::runtime_error("cannot load " + std::string(argv[1]));
    const train::MemmapSplit split(argv[2]);
    const std::vector<int64_t> ids = train::read_npy_i64(argv[3]);
    const std::string out = argv[4];

    std::vector<double> energy;
    std::vector<float> forces;
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t s : ids) {
      if (s < 0 || s >= split.size()) throw std::runtime_error("id out of range: " + std::to_string(s));
      const ModelResult r = model.predict(split.system(s), false);
      energy.push_back(r.energy);
      if (model.nc_forces()) {
        if (!r.has_nc_forces || r.nc_forces.size() != 3 * size_t(split.n_atoms(s)))
          throw std::runtime_error("no direct forces for structure " + std::to_string(s));
        forces.insert(forces.end(), r.nc_forces.begin(), r.nc_forces.end());
      }
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    auto write = [](const std::string &path, const void *data, size_t bytes) {
      std::unique_ptr<std::FILE, int (*)(std::FILE *)> f(std::fopen(path.c_str(), "wb"), std::fclose);
      if (!f || std::fwrite(data, 1, bytes, f.get()) != bytes) throw std::runtime_error("cannot write " + path);
    };
    write(out + ".energy.bin", energy.data(), energy.size() * sizeof(double));
    if (model.nc_forces()) write(out + ".nc_forces.bin", forces.data(), forces.size() * sizeof(float));
    std::cout << ids.size() << " structures in " << sec << " s on " << backend << " -> " << out << ".*\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
