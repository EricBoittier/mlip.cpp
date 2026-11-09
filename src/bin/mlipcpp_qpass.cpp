// The lrc graph-mode charges of every frame of an extended XYZ, with the same code the trained models use:
// raw charges of a frozen charge pass (a plain GGUF whose per-atom outputs are charges, evaluated at total charge 0 or Q)
// and the element Fukui equilibration to the frame's total charge, q_i = q_raw_i + f_i (Q - sum q_raw) / sum f.
//   mlipcpp-qpass CHARGE.gguf FUKUI.json IN.xyz OUT.json [--qpass-charge zero|total] [--backend cpu|cuda]
//       [--proj W A_ON A_OFF]   also write "q_proj": the lrc_qproj soft projection of the charges (width W, clusters A_ON / A_OFF)
//       [--fd H]   with --proj: the inter-cluster long-range energy (switch 8 -> 10 A, projected charges) and its forces
//                  against central differences of step H (A) in double precision, per frame ("e_lr", "fd_max_err", "f_max")
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
  double pw = 0, pon = 8, poff = 10, fdh = 0;
  BackendPreference be = BackendPreference::Auto;
  for (int i = 5; i + 1 < argc; i++) {
    const std::string a = argv[i];
    if (a == "--qpass-charge") zero = std::string(argv[++i]) == "zero";
    else if (a == "--proj" && i + 3 < argc) pw = std::stod(argv[++i]), pon = std::stod(argv[++i]), poff = std::stod(argv[++i]);
    else if (a == "--fd") fdh = std::stod(argv[++i]);
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
      nlohmann::json fr = {{"q", q.q0}, {"charge", sys.charge()}, {"n_atoms", n}};
      if (pw > 0) {
        runtime::LrcSpec Lp = L;
        Lp.qproj = true, Lp.qproj_w = pw, Lp.a_on = pon, Lp.a_off = poff;
        fr["q_proj"] = runtime::lrc_project_charges(Lp, n, sys.atomic_numbers(), x.data(), q.q0.data());
        if (fdh > 0) {
          runtime::SqeSpec Sl;
          Sl.enabled = true, Sl.lr_R = 10.0, Sl.lr_t = 2.0;
          const int32_t *Zs = sys.atomic_numbers();
          const double Qt = sys.charge();
          const int sp = sys.spin_multiplicity();
          auto energy = [&](const std::vector<double> &xx) {
            const runtime::LrcCondition c = runtime::lrc_condition(Lp, Sl, n, Zs, xx.data(), Qt, 10, sp);
            return runtime::lrc_energy(Lp, Sl, c, n, Zs, xx.data(), Qt, false).energy;
          };
          const runtime::LrcCondition c0 = runtime::lrc_condition(Lp, Sl, n, Zs, x.data(), Qt, 10, sp);
          const runtime::LrcEnergy e0 = runtime::lrc_energy(Lp, Sl, c0, n, Zs, x.data(), Qt, true);
          double err = 0, fmax = 0;
          for (size_t k = 0; k < x.size(); k++) {
            std::vector<double> xp = x, xm = x;
            xp[k] += fdh, xm[k] -= fdh;
            const double num = -(energy(xp) - energy(xm)) / (2 * fdh);
            err = std::max(err, std::abs(num - e0.forces[k])), fmax = std::max(fmax, std::abs(e0.forces[k]));
          }
          fr["e_lr"] = e0.energy, fr["fd_max_err"] = err, fr["f_max"] = fmax, fr["n_clusters"] = c0.cl.s;
        }
      }
      frames.push_back(fr);
    }
    nlohmann::json out = {{"model", argv[1]}, {"fukui", argv[2]}, {"qpass_charge", zero ? "zero" : "total"}, {"frames", frames}};
    std::ofstream(argv[4]) << out.dump(1);
  } catch (const std::exception &e) {
    std::cerr << "mlipcpp-qpass: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
