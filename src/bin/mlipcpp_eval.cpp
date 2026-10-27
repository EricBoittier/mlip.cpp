/**
 * Energies, forces and stresses of any GGUF (pet-graph, plato-pet-lr, ...) on
 * every frame of an extended-XYZ file, through the public C++ API (Predictor),
 * as JSON: for parity checks against metatrain / metatomic
 * (scripts/conditioning/parity_*.py).
 *
 *   mlipcpp-eval MODEL.gguf IN.xyz OUT.json [--backend cpu|cuda|...] [--charge Q] [--spin S] [--repeat N]
 *
 * --repeat N: each frame is evaluated N + 1 times (the first a warm-up) and its
 * median wall time per evaluation is written as "seconds".
 *
 * Charge and spin multiplicity per frame come from the frame's charge= and
 * spin_multiplicity= (or spin=) keys, else 0 and 1; --charge / --spin
 * override them for every frame. Stress (Voigt xx yy zz yz xz xy, eV/A^3) for
 * frames with a Lattice. Plato-PET-LR models with LES also write "latent_charges" (e per
 * atom, first channel, after the neutralisation / total-charge shift). A frame whose charge or spin the model rejects gets
 * {"error": "..."} instead of results.
 */

#include "mlipcpp/io.h"
#include "mlipcpp/mlipcpp.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlipcpp;

namespace {

Backend parse(const std::string &name) {
  if (name == "auto") return Backend::Auto;
  if (name == "cpu") return Backend::CPU;
  if (name == "cuda") return Backend::CUDA;
  if (name == "hip") return Backend::HIP;
  if (name == "metal") return Backend::Metal;
  if (name == "vulkan") return Backend::Vulkan;
  throw std::runtime_error("unknown backend " + name);
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " MODEL.gguf IN.xyz OUT.json [--backend cpu|cuda|...] [--charge Q] [--spin S]\n";
    return 1;
  }
  std::optional<int> charge, spin;
  int repeat = 0;
  std::string backend = "cpu";
  for (int i = 4; i + 1 < argc; i++) {
    const std::string a = argv[i];
    if (a == "--backend") backend = argv[++i];
    else if (a == "--charge") charge = std::stoi(argv[++i]);
    else if (a == "--spin") spin = std::stoi(argv[++i]);
    else if (a == "--repeat") repeat = std::stoi(argv[++i]);
    else {
      std::cerr << "unknown option " << a << "\n";
      return 1;
    }
  }
  try {
    ModelOptions mo;
    mo.backend = parse(backend);
    Predictor model(argv[1], mo);
    std::ifstream in(argv[2]);
    if (!in) throw std::runtime_error(std::string("cannot read ") + argv[2]);
    nlohmann::json out = nlohmann::json::object();
    out["model"] = argv[1];
    out["model_type"] = std::string(model.model_type());
    out["conditioned"] = model.conditioned();
    out["max_charge"] = model.max_charge();
    out["max_spin_multiplicity"] = model.max_spin_multiplicity();
    out["backend"] = backend;
    nlohmann::json frames = nlohmann::json::array();
    while (in.peek() != EOF) {
      std::streampos at = in.tellg();
      std::string line;
      if (!std::getline(in, line)) break;
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      in.seekg(at);
      AtomicSystem sys = io::read_xyz(in);
      if (charge) sys.set_charge(*charge);
      if (spin) sys.set_spin_multiplicity(*spin);
      PredictOptions po;
      po.compute_forces = true;
      po.compute_stress = sys.is_periodic();
      po.charge = sys.charge();
      po.spin_multiplicity = sys.spin_multiplicity();
      nlohmann::json f;
      f["charge"] = sys.charge();
      f["spin_multiplicity"] = sys.spin_multiplicity();
      f["n_atoms"] = sys.num_atoms();
      try {
        float cell[9];
        bool pbc[3];
        if (const Cell *c = sys.cell()) {
          for (int i = 0; i < 3; i++) {
            pbc[i] = c->periodic[i];
            for (int j = 0; j < 3; j++) cell[3 * i + j] = c->matrix[i][j];
          }
        }
        Result r = model.predict(sys.num_atoms(), sys.positions(), sys.atomic_numbers(),
                                 sys.cell() ? cell : nullptr, sys.cell() ? pbc : nullptr, po);
        if (repeat > 0) {
          std::vector<double> t;
          for (int k = 0; k < repeat; k++) {
            const auto t0 = std::chrono::steady_clock::now();
            model.predict(sys.num_atoms(), sys.positions(), sys.atomic_numbers(), sys.cell() ? cell : nullptr,
                          sys.cell() ? pbc : nullptr, po);
            t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
          }
          std::sort(t.begin(), t.end());
          f["seconds"] = t[t.size() / 2];
        }
        f["energy"] = r.energy;
        f["forces"] = r.forces;
        if (r.has_stress()) f["stress"] = r.stress;
        if (!r.latent_charges.empty()) f["latent_charges"] = r.latent_charges; // Plato-PET-LR LES (e)
      } catch (const std::exception &e) {
        f["error"] = e.what();
      }
      frames.push_back(f);
    }
    out["frames"] = frames;
    std::ofstream(argv[3]) << out.dump(1) << "\n";
    std::cout << frames.size() << " frames -> " << argv[3] << "\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
