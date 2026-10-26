// Plato-PET-LR in mlip.cpp: parity against the PyTorch references of
// scripts/plato/plato_reference.py, and timing.
//
//   mlipcpp-plato check MODEL.gguf REFDIR [--backend cpu|cuda] [--json OUT]
//   mlipcpp-plato bench MODEL.gguf XYZ [--tile K] [--repeat N] [--warmup W] [--backend B] [--energy-only]
//
// check: every structure of REFDIR/structures.xyz, every variant (full, no_les,
// no_mesh, sr), against the float64 and float32 PyTorch results: energy, forces,
// stress; and the D2 group invariance of the cc0_d2_* rotations.
// bench: ms per energy + forces call of the first frame tiled K x K x K (median of N).
#include "mlipcpp/io.h"
#include "models/plato/plato.h"
#include "core/gguf_loader.h"
#include "runtime/graph_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace mlipcpp;
using json = nlohmann::ordered_json;

namespace {

std::string arg(int argc, char **argv, const char *name, const std::string &fallback) {
  for (int i = 1; i + 1 < argc; i++)
    if (!std::strcmp(argv[i], name)) return argv[i + 1];
  return fallback;
}
bool flag(int argc, char **argv, const char *name) {
  for (int i = 1; i < argc; i++)
    if (!std::strcmp(argv[i], name)) return true;
  return false;
}

std::vector<AtomicSystem> read_all(const std::string &path) {
  std::ifstream f(path);
  std::vector<AtomicSystem> out;
  while (f.peek() != EOF) {
    std::streampos p = f.tellg();
    std::string line;
    if (!std::getline(f, line)) break;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    f.seekg(p);
    out.push_back(io::read_xyz(f));
  }
  return out;
}

struct Err {
  double de = 0, f_max = 0, f_rms = 0, f_rel = 0, s_max = 0;
};

Err compare(const ModelResult &r, const json &ref) {
  Err e;
  e.de = std::abs(double(r.energy) - ref["energy"].get<double>());
  const auto &F = ref["forces"];
  double s2 = 0, n2 = 0;
  for (size_t i = 0; i < F.size(); i++)
    for (int c = 0; c < 3; c++) {
      const double d = double(r.forces[3 * i + c]) - F[i][c].get<double>();
      e.f_max = std::max(e.f_max, std::abs(d));
      s2 += d * d, n2 += std::pow(F[i][c].get<double>(), 2);
    }
  e.f_rms = std::sqrt(s2 / (3.0 * F.size()));
  e.f_rel = std::sqrt(s2 / std::max(n2, 1e-300));
  if (ref.contains("stress") && r.has_stress)
    for (int k = 0; k < 6; k++) e.s_max = std::max(e.s_max, std::abs(double(r.stress[k]) - ref["stress"][k].get<double>()));
  return e;
}

BackendPreference backend_of(const std::string &b) { return parse_backend_preference(b); }

int check(int argc, char **argv) {
  const std::string model = argv[2], dir = argv[3];
  const BackendPreference be = backend_of(arg(argc, argv, "--backend", "auto"));
  std::ifstream rf(dir + "/reference.json");
  const json ref = json::parse(rf);
  const auto systems = read_all(dir + "/structures.xyz");
  const std::vector<std::string> variants = {"full", "no_les", "no_mesh", "sr"};
  std::vector<std::unique_ptr<plato::PlatoModel>> models;
  for (const auto &v : variants) {
    plato::PlatoOptions o;
    o.backend = be;
    o.mesh = v == "full" || v == "no_les";
    o.les = v == "full" || v == "no_mesh";
    models.push_back(std::make_unique<plato::PlatoModel>(o));
    models.back()->load_from_gguf(model);
  }
  std::printf("backend %s\n", models[0]->backend_name());
  json out = json::object();
  size_t k = 0;
  std::map<std::string, double> d2_energy;
  std::printf("%-10s %-8s %4s %12s | %10s %10s %10s %10s | %10s %10s %10s | mesh\n", "structure", "variant", "n",
              "E (eV)", "dE64 meV", "dF64 max", "dF64 rel", "dS64 max", "dE32 meV", "dF32 max", "dS32 max");
  for (const auto &[name, r] : ref["structures"].items()) {
    const AtomicSystem &sys = systems.at(k++);
    for (size_t v = 0; v < variants.size(); v++) {
      const ModelResult res = models[v]->predict(sys, true);
      const Err e64 = compare(res, r["float64"][variants[v]]), e32 = compare(res, r["float32"][variants[v]]);
      const auto &P = models[v]->last_parts();
      std::printf("%-10s %-8s %4d %12.6f | %10.4f %10.2e %10.2e %10.2e | %10.4f %10.2e %10.2e | %d %d %d\n", name.c_str(),
                  variants[v].c_str(), sys.num_atoms(), res.energy, 1e3 * e64.de, e64.f_max, e64.f_rel, e64.s_max,
                  1e3 * e32.de, e32.f_max, e32.s_max, P.mesh[0], P.mesh[1], P.mesh[2]);
      out[name][variants[v]] = {{"energy", res.energy},
                                {"e_sr", P.short_range},
                                {"e_les", P.les},
                                {"vs_float64", {{"dE", e64.de}, {"dF_max", e64.f_max}, {"dF_rms", e64.f_rms}, {"dF_rel", e64.f_rel}, {"dS_max", e64.s_max}}},
                                {"vs_float32", {{"dE", e32.de}, {"dF_max", e32.f_max}, {"dF_rms", e32.f_rms}, {"dF_rel", e32.f_rel}, {"dS_max", e32.s_max}}},
                                {"mesh", {P.mesh[0], P.mesh[1], P.mesh[2]}},
                                {"n_k", P.n_k}};
      if (variants[v] == "full" && name.rfind("cc0", 0) == 0) d2_energy[name] = res.energy;
    }
    std::fflush(stdout);
  }
  if (d2_energy.count("cc0")) {
    std::printf("D2 invariance (full model): E(cc0) = %.7f eV\n", d2_energy["cc0"]);
    for (const auto &[name, e] : d2_energy)
      if (name.rfind("cc0_d2", 0) == 0 || name == "cc0_rot")
        std::printf("  %-10s E - E(cc0) = %+.3e eV\n", name.c_str(), e - d2_energy["cc0"]);
  }
  const std::string js = arg(argc, argv, "--json", "");
  if (!js.empty()) std::ofstream(js) << out.dump(1);
  return 0;
}

AtomicSystem tile(const AtomicSystem &s, int K) {
  if (K <= 1) return s;
  const Cell *c = s.cell();
  if (!c) throw std::runtime_error("--tile needs a cell");
  const int n = s.num_atoms();
  std::vector<float> x;
  std::vector<int32_t> z;
  for (int a = 0; a < K; a++)
    for (int b = 0; b < K; b++)
      for (int d = 0; d < K; d++)
        for (int i = 0; i < n; i++) {
          for (int q = 0; q < 3; q++)
            x.push_back(s.positions()[3 * i + q] + a * c->matrix[0][q] + b * c->matrix[1][q] + d * c->matrix[2][q]);
          z.push_back(s.atomic_numbers()[i]);
        }
  float L[3][3];
  for (int a = 0; a < 3; a++)
    for (int q = 0; q < 3; q++) L[a][q] = K * c->matrix[a][q];
  Cell cell(L, c->periodic[0], c->periodic[1], c->periodic[2]);
  return AtomicSystem(n * K * K * K, x.data(), z.data(), &cell);
}

int bench(int argc, char **argv) {
  const std::string model = argv[2], xyz = argv[3];
  const int K = std::stoi(arg(argc, argv, "--tile", "1")), N = std::stoi(arg(argc, argv, "--repeat", "20")),
            W = std::stoi(arg(argc, argv, "--warmup", "3"));
  const bool forces = !flag(argc, argv, "--energy-only");
  plato::PlatoOptions o;
  o.backend = backend_of(arg(argc, argv, "--backend", "auto"));
  plato::PlatoModel m(o);
  m.load_from_gguf(model);
  const AtomicSystem s = tile(read_all(xyz).at(std::stoi(arg(argc, argv, "--frame", "0"))), K);
  for (int i = 0; i < W; i++) m.predict(s, forces);
  std::vector<double> tot, nl, comp;
  double E = 0;
  for (int i = 0; i < N; i++) {
    const auto t0 = std::chrono::steady_clock::now();
    const ModelResult r = m.predict(s, forces);
    tot.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    nl.push_back(m.last_parts().nl_ms), comp.push_back(m.last_parts().compute_ms);
    E = r.energy;
  }
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const auto &P = m.last_parts();
  std::printf("{\"backend\": \"%s\", \"atoms\": %d, \"edges\": %ld, \"max_len\": %d, \"mesh\": [%d, %d, %d], \"n_k\": %ld, "
              "\"forces\": %s, \"energy\": %.6f, \"ms_total\": %.3f, \"ms_neighbors\": %.3f, \"ms_compute\": %.3f, "
              "\"ms_total_minus_nl\": %.3f, \"ms_min\": %.3f, \"repeat\": %d}\n",
              m.backend_name(), s.num_atoms(), (long)P.edges, P.max_len, P.mesh[0], P.mesh[1], P.mesh[2], (long)P.n_k,
              forces ? "true" : "false", E, med(tot), med(nl), med(comp), med(tot) - med(nl),
              *std::min_element(tot.begin(), tot.end()), N);
  return 0;
}

std::unique_ptr<Model> any_model(int argc, char **argv) {
  const BackendPreference be = backend_of(arg(argc, argv, "--backend", "auto"));
  GGUFLoader probe(argv[2]);
  const std::string a = probe.get_string("general.architecture", "");
  if (a == "pet-graph") {
    auto gm = std::make_unique<runtime::GraphModel>();
    gm->set_backend_preference(be);
    if (!gm->load_from_gguf(argv[2])) throw std::runtime_error("failed to load graph model");
    return gm;
  }
  if (a.rfind("plato", 0) == 0) {
    plato::PlatoOptions o;
    o.backend = be;
    auto m = std::make_unique<plato::PlatoModel>(o);
    m->load_from_gguf(argv[2]);
    return m;
  }
  return load_model(argv[2]);
}

// energy and forces of every frame as JSON lines
int dump(int argc, char **argv) {
  auto m = any_model(argc, argv);
  for (const auto &s : read_all(argv[3])) {
    const ModelResult r = m->predict(s, true);
    json j = {{"n", s.num_atoms()}, {"energy", r.energy}, {"forces", r.forces}};
    std::printf("%s\n", j.dump().c_str());
  }
  return 0;
}

// any GGUF (PET GraphModel, Plato, ...): ms per energy + forces call
int bench_any(int argc, char **argv) {
  const int K = std::stoi(arg(argc, argv, "--tile", "1")), N = std::stoi(arg(argc, argv, "--repeat", "20")),
            W = std::stoi(arg(argc, argv, "--warmup", "3"));
  auto m = any_model(argc, argv);
  if (flag(argc, argv, "--all")) { // one pass over every frame, one call each (neighbour lists included)
    const auto all = read_all(argv[3]);
    auto pass = [&] {
      double E = 0;
      for (const auto &s : all) E += m->predict(s, true).energy;
      return E;
    };
    for (int i = 0; i < W; i++) pass();
    std::vector<double> t;
    double E = 0;
    int atoms = 0;
    for (const auto &s : all) atoms += s.num_atoms();
    for (int i = 0; i < N; i++) {
      const auto t0 = std::chrono::steady_clock::now();
      E = pass();
      t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    std::printf("{\"model\": \"%s\", \"frames\": %zu, \"atoms\": %d, \"energy_sum\": %.6f, \"ms_total\": %.3f, \"ms_min\": %.3f, \"repeat\": %d}\n",
                m->model_type().c_str(), all.size(), atoms, E, t[t.size() / 2], t[0], N);
    return 0;
  }
  const AtomicSystem s = tile(read_all(argv[3]).at(std::stoi(arg(argc, argv, "--frame", "0"))), K);
  for (int i = 0; i < W; i++) m->predict(s, true);
  std::vector<double> tot;
  double E = 0;
  for (int i = 0; i < N; i++) {
    const auto t0 = std::chrono::steady_clock::now();
    E = m->predict(s, true).energy;
    tot.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(tot.begin(), tot.end());
  std::printf("{\"model\": \"%s\", \"atoms\": %d, \"energy\": %.6f, \"ms_total\": %.3f, \"ms_min\": %.3f, \"repeat\": %d}\n",
              m->model_type().c_str(), s.num_atoms(), E, tot[tot.size() / 2], tot[0], N);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: mlipcpp-plato check MODEL.gguf REFDIR [--backend B] [--json OUT]\n"
                         "       mlipcpp-plato bench MODEL.gguf XYZ [--tile K] [--repeat N] [--warmup W] [--backend B]\n");
    return 2;
  }
  try {
    if (!std::strcmp(argv[1], "check")) return check(argc, argv);
    if (!std::strcmp(argv[1], "bench")) return bench(argc, argv);
    if (!std::strcmp(argv[1], "bench-any")) return bench_any(argc, argv);
    if (!std::strcmp(argv[1], "dump")) return dump(argc, argv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "unknown command %s\n", argv[1]);
  return 2;
}
