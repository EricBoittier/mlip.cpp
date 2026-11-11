/**
 * mlipcpp-capture: every GIR node's forward output for one frame of an extended-XYZ file (Predictor::
 * predict_with_activations), written as a flat binary for backend-against-backend bisection (parity work):
 *   per node: int32 id, int32 name length, name, int32 ndims, int64 shape[ndims], int32 dtype (0 f32, 1 i32,
 *   2 other), int64 nbytes, data.
 *
 *   mlipcpp-capture MODEL.gguf IN.xyz OUT.bin [--backend cpu|cuda] [--frame I]
 */
#include "mlipcpp/io.h"
#include "mlipcpp/mlipcpp.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace mlipcpp;

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " MODEL.gguf IN.xyz OUT.bin [--backend cpu|cuda] [--frame I]\n";
    return 1;
  }
  std::string backend = "cpu";
  int frame = 0;
  for (int i = 4; i + 1 < argc; i++) {
    const std::string a = argv[i];
    if (a == "--backend") backend = argv[++i];
    else if (a == "--frame") frame = std::stoi(argv[++i]);
  }
  try {
    ModelOptions mo;
    mo.backend = backend == "cuda" ? Backend::CUDA : Backend::CPU;
    Predictor model(argv[1], mo);
    std::ifstream in(argv[2]);
    AtomicSystem sys = io::read_xyz(in);
    for (int f = 0; f < frame; f++) sys = io::read_xyz(in);
    const auto r = model.predict_with_activations(int(sys.num_atoms()), sys.positions(), sys.atomic_numbers());
    FILE *fp = std::fopen(argv[3], "wb");
    if (!fp) throw std::runtime_error("cannot write output");
    for (const auto &a : r.activations) {
      const int32_t id = a.node_id, nl = int32_t(a.name.size()), nd = int32_t(a.shape.size());
      const int32_t dt = a.dtype == "f32" ? 0 : a.dtype == "i32" ? 1 : 2;
      const int64_t nb = int64_t(a.data.size());
      std::fwrite(&id, 4, 1, fp), std::fwrite(&nl, 4, 1, fp), std::fwrite(a.name.data(), 1, nl, fp);
      std::fwrite(&nd, 4, 1, fp), std::fwrite(a.shape.data(), 8, nd, fp), std::fwrite(&dt, 4, 1, fp);
      std::fwrite(&nb, 8, 1, fp), std::fwrite(a.data.data(), 1, nb, fp);
    }
    std::fclose(fp);
    std::cout << r.activations.size() << " nodes, E = " << r.prediction.energy << "\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
