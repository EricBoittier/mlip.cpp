/**
 * Energies, forces and stresses of any GGUF (pet-graph, plato-pet-lr, ...) on
 * every frame of an extended-XYZ file, through the public C++ API (Predictor),
 * as JSON: for parity checks against metatrain / metatomic
 * (scripts/conditioning/parity_*.py).
 *
 *   mlipcpp-eval MODEL.gguf IN.xyz OUT.json [--backend cpu|cuda|...] [--charge Q] [--spin S] [--repeat N]
 *                [--iris-freeze-ewald 0|1] [--iris-max-neighbors N] [--no-forces]
 * iris PET-LR on CUDA: the F32 MUL_MATs reducing over K = 1036 (the 512-wide checkpoints' transfer-MLP first LINEAR)
 * accumulate in double by default; MLIPCPP_PRECISE_MATMUL_K=0 turns that off, =<K> moves it to another K,
 * MLIPCPP_PRECISE_MATMUL=1 (with _K unset) flags every F32 MUL_MAT; MLIPCPP_PRECISE_SUM / _SIGMOID / _SQRT=1 are
 * opt-in (see runtime/graph_model.cpp mark_precise_math). CPU results are unaffected by all of these.
 * (iris models: --iris-freeze-ewald 1 freezes the Ewald parameters over the frames of the file as his calculator does
 * over a run; --iris-max-neighbors overrides the GGUF's overflow limit, 0 = no check)
 *
 * --fp64: iris GGUFs on the fp64 executor (runtime/iris_fp64.h; PARITY.md "fp64 executor"): positions, cell and every
 * host value in double (positions and Lattice re-read from the extxyz text, not through the float AtomicSystem), the
 * graph in double, results written with 17 significant digits. MLIPCPP_EXEC=fp64 is the same executor behind the float
 * API (float positions promoted exactly).
 *
 * --repeat N: each frame is evaluated N + 1 times (the first a warm-up) and its
 * median wall time per evaluation is written as "seconds".
 *
 * Charge and spin multiplicity per frame come from the frame's charge= and
 * spin_multiplicity= (or spin=) keys, else 0 and 1; --charge / --spin
 * override them for every frame. A per-atom array formal_charges (Properties) is passed to
 * models with fragment-constrained charges (pet.plato_lr.frag_charges). Stress (Voigt xx yy zz yz xz xy, eV/A^3) for
 * frames with a Lattice. Plato-PET-LR models with LES also write "latent_charges" (e per
 * atom, first channel, after the neutralisation / total-charge shift). A frame whose charge or spin the model rejects gets
 * {"error": "..."} instead of results.
 */

#include "mlipcpp/io.h"
#include "mlipcpp/mlipcpp.hpp"
#include "runtime/graph_model.h"
#include "runtime/iris_fp64.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
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

// one extxyz frame's positions and Lattice in double (the float AtomicSystem rounds them): the pos columns of
// Properties (default species x y z), Lattice="..." of the comment line
void read_double_frame(const std::string &text, int n, std::vector<double> &pos, bool &has_lattice, double cell[3][3]) {
  std::istringstream is(text);
  std::string line;
  std::getline(is, line);
  std::getline(is, line);
  int col = 1;
  const auto pp = line.find("Properties=");
  if (pp != std::string::npos) {
    std::string props = line.substr(pp + 11);
    props = props.substr(0, props.find_first_of(" \t"));
    std::stringstream ss(props);
    std::string name, type, count;
    int at = 0;
    col = -1;
    while (std::getline(ss, name, ':') && std::getline(ss, type, ':') && std::getline(ss, count, ':')) {
      if (name == "pos") col = at;
      at += std::stoi(count);
    }
    if (col < 0) throw std::runtime_error("--fp64: no pos in Properties");
  }
  has_lattice = false;
  const auto lp = line.find("Lattice=\"");
  if (lp != std::string::npos) {
    std::istringstream ls(line.substr(lp + 9, line.find('"', lp + 9) - lp - 9));
    for (int a = 0; a < 9; a++) ls >> cell[a / 3][a % 3];
    if (!ls) throw std::runtime_error("--fp64: Lattice");
    has_lattice = true;
  }
  pos.assign(3 * size_t(n), 0.0);
  for (int i = 0; i < n; i++) {
    std::getline(is, line);
    std::istringstream ls(line);
    std::string tok;
    for (int c = 0; c < col; c++) ls >> tok;
    ls >> pos[3 * i] >> pos[3 * i + 1] >> pos[3 * i + 2];
    if (!ls) throw std::runtime_error("--fp64: positions of atom " + std::to_string(i));
  }
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " MODEL.gguf IN.xyz OUT.json [--backend cpu|cuda|...] [--charge Q] [--spin S]\n";
    return 1;
  }
  std::optional<int> charge, spin;
  bool no_forces = false;
  int repeat = 0, max_nb = -1, qeq_maxiter = -1;
  double qeq_tol = -1.0;
  bool freeze = false;
  std::string backend = "cpu";
  bool fp64 = false;
  for (int i = 4; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--no-forces") { no_forces = true; continue; } // energies (and charges / dipoles) only
    if (a == "--fp64") { fp64 = true; continue; }
    if (i + 1 >= argc) {
      std::cerr << "option " << a << " needs a value\n";
      return 1;
    }
    if (a == "--backend") backend = argv[++i];
    else if (a == "--charge") charge = std::stoi(argv[++i]);
    else if (a == "--spin") spin = std::stoi(argv[++i]);
    else if (a == "--repeat") repeat = std::stoi(argv[++i]);
    else if (a == "--iris-freeze-ewald") freeze = std::stoi(argv[++i]) != 0;
    else if (a == "--iris-max-neighbors") max_nb = std::stoi(argv[++i]);
    else if (a == "--iris-qeq-maxiter") qeq_maxiter = std::stoi(argv[++i]); // iris QEq: override his cg settings
    else if (a == "--iris-qeq-tol") qeq_tol = std::stod(argv[++i]);
    else {
      std::cerr << "unknown option " << a << "\n";
      return 1;
    }
  }
  try {
    ModelOptions mo;
    mo.backend = parse(backend);
    mo.iris_freeze_ewald = freeze, mo.iris_max_neighbors = max_nb;
    mo.iris_qeq_maxiter = qeq_maxiter, mo.iris_qeq_tol = qeq_tol;
    Predictor model(argv[1], mo);
    std::unique_ptr<runtime::GraphModel> gm;
    std::unique_ptr<runtime::fp64::IrisFp64> x64;
    if (fp64) {
      gm = std::make_unique<runtime::GraphModel>();
      if (!gm->load_from_gguf(argv[1])) throw std::runtime_error("cannot load " + std::string(argv[1]));
      if (max_nb >= 0) gm->set_iris_max_neighbors(max_nb);
      gm->set_iris_freeze_ewald(freeze);
      gm->set_iris_qeq_solver(qeq_maxiter, qeq_tol);
      x64 = std::make_unique<runtime::fp64::IrisFp64>(*gm);
    }
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
      std::string frame_text;
      if (fp64) { // the frame's raw text (count line, comment, atoms), for the double positions
        const int na = std::stoi(line);
        std::string l;
        for (int k = 0; k < na + 2 && std::getline(in, l); k++) frame_text += l + "\n";
        in.clear();
        in.seekg(at);
      }
      AtomicSystem sys = io::read_xyz(in);
      if (charge) sys.set_charge(*charge);
      if (spin) sys.set_spin_multiplicity(*spin);
      PredictOptions po;
      po.compute_forces = true;
      po.compute_stress = sys.is_periodic();
      po.charge = sys.charge();
      po.spin_multiplicity = sys.spin_multiplicity();
      if (sys.has_dipole_origin()) po.dipole_origin.assign(sys.dipole_origin().begin(), sys.dipole_origin().end());
      if (sys.has_field()) po.field.assign(sys.field().begin(), sys.field().end());
      if (no_forces) po.compute_forces = false, po.compute_stress = false;
      po.formal_charges = sys.formal_charges(); // per-atom formal_charges array (fragment-constrained charges)
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
        if (fp64) {
          runtime::fp64::System64 s64 = runtime::fp64::to_system64(sys);
          bool lat = false;
          double h[3][3];
          read_double_frame(frame_text, s64.n, s64.positions, lat, h);
          if (lat && s64.has_cell)
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++) s64.cell[a][b] = h[a][b];
          const runtime::fp64::Result64 r = x64->predict(s64, !no_forces);
          f["exec"] = "fp64";
          f["energy"] = r.energy;
          if (r.has_forces) f["forces"] = r.forces;
          if (r.has_stress) f["stress"] = r.stress, f["virial"] = r.virial;
          if (r.has_charges) {
            f["charges"] = r.charges;
            f["dipole"] = r.dipole;
            f["fermi_level"] = r.fermi_level;
            f["metallicity"] = r.metallicity;
            if (r.has_charge_residual) f["charge_residual"] = r.charge_residual;
            if (!r.dipole_flow.empty()) f["dipole_flow"] = r.dipole_flow;
            if (!r.components.empty()) f["components"] = r.components, f["component_charges"] = r.component_charges;
          }
          f["atomic_energies"] = r.atomic_energies;
          frames.push_back(f);
          continue;
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
        if (!r.virial.empty()) f["virial"] = r.virial; // full dE/d strain, row-major (eV), before symmetrisation
        if (!r.latent_charges.empty()) f["latent_charges"] = r.latent_charges; // Plato-PET-LR LES (e)
        if (!r.sqe_charges.empty()) f["sqe_charges"] = r.sqe_charges;           // pet.plato_lr.sqe (e)
        if (!r.charges.empty()) { // iris PET-LR (pet.iris_lr)
          f["charges"] = r.charges;
          f["dipole"] = r.dipole;
          f["fermi_level"] = r.fermi_level;
          f["metallicity"] = r.metallicity;
          if (r.has_charge_residual) f["charge_residual"] = r.charge_residual;
          if (!r.dipole_flow.empty()) f["dipole_flow"] = r.dipole_flow;
          if (!r.components.empty()) f["components"] = r.components, f["component_charges"] = r.component_charges;
        }
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
