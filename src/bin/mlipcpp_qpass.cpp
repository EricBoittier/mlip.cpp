// The lrc graph-mode charges of every frame of an extended XYZ, with the same code the trained models use:
// raw charges of a frozen charge pass (a plain GGUF whose per-atom outputs are charges, evaluated at total charge 0 or Q)
// and the element Fukui equilibration to the frame's total charge, q_i = q_raw_i + f_i (Q - sum q_raw) / sum f.
//   mlipcpp-qpass CHARGE.gguf FUKUI.json IN.xyz OUT.json [--qpass-charge zero|total] [--backend cpu|cuda]
// FUKUI.json: {"Z": [...], "phi": [...]} (f = softplus(phi)). Writes {"frames": [{"q": [...], "q_raw": [...]}]}.
#include "mlipcpp/io.h"
#include "runtime/graph_model.h"
#include "runtime/lr_combined.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace mlipcpp;

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: mlipcpp-qpass CHARGE.gguf FUKUI.json IN.xyz OUT.json [--qpass-charge zero|total] [--backend cpu|cuda]\n";
    return 1;
  }
  bool zero = true;
  BackendPreference be = BackendPreference::Auto;
  for (int i = 5; i + 1 < argc; i++) {
    const std::string a = argv[i];
    if (a == "--qpass-charge") zero = std::string(argv[++i]) == "zero";
    else if (a == "--backend") be = std::string(argv[++i]) == "cpu" ? BackendPreference::CPU : BackendPreference::CUDA;
  }
  try {
    runtime::LrcSpec L;
    L.enabled = L.graph = true, L.qpass_neutral = zero;
    const nlohmann::json fk = nlohmann::json::parse(std::ifstream(argv[2]));
    for (double z : fk["Z"]) L.fukui_Z.push_back(static_cast<int32_t>(z));
    for (double p : fk["phi"]) L.fukui_phi.push_back(p);
    L.qsrc = runtime::make_lrc_graph_charges(argv[1], be, 0);
    runtime::SqeSpec S;
    std::ifstream in(argv[3]);
    nlohmann::json frames = nlohmann::json::array();
    while (in.peek() != EOF) {
      std::streampos at = in.tellg();
      std::string line;
      if (!std::getline(in, line)) break;
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      in.seekg(at);
      AtomicSystem sys = io::read_xyz(in);
      const int n = sys.num_atoms();
      std::vector<double> x(sys.positions(), sys.positions() + 3 * n);
      const runtime::LrcQ0 q = runtime::lrc_q0(L, S, n, sys.atomic_numbers(), x.data(), sys.charge(), sys.spin_multiplicity());
      frames.push_back({{"q", q.q0}, {"charge", sys.charge()}, {"n_atoms", n}});
    }
    nlohmann::json out = {{"model", argv[1]}, {"fukui", argv[2]}, {"qpass_charge", zero ? "zero" : "total"}, {"frames", frames}};
    std::ofstream(argv[4]) << out.dump(1);
  } catch (const std::exception &e) {
    std::cerr << "mlipcpp-qpass: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
