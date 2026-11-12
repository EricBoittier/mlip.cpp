// The lrc graph-mode charges of every frame of an extended XYZ, with the same code the trained models use:
// raw charges of a frozen charge pass (a plain GGUF whose per-atom outputs are charges, evaluated at total charge 0 or Q)
// and the element Fukui equilibration to the frame's total charge, q_i = q_raw_i + f_i (Q - sum q_raw) / sum f.
//   mlipcpp-qpass CHARGE.gguf FUKUI.json IN.xyz OUT.json [--qpass-charge zero|total] [--backend cpu|cuda]
//       [--proj W A_ON A_OFF]   also write "q_proj": the lrc_qproj soft projection of the charges (width W, clusters A_ON / A_OFF)
//       [--fd H]   with --proj: the inter-cluster long-range energy (switch 8 -> 10 A, projected charges) and its forces
//                  against central differences of step H (A) in double precision, per frame ("e_lr", "fd_max_err", "f_max")
//       [--fukui-mode element|uniform|atom|atom_pm] [--fmodel F.gguf] [--fmodel-minus F2.gguf] [--pm-width D]
//       [--fukui-scope global|cluster] [--cluster-w W] [--clusters A_ON A_OFF]   the Fukui arms of feat/plato-lr-fukui
//                  (per-atom passes: GGUF copies of the charge-pass backbone with the fitted Fukui head; AC clusters use
//                  the lrc link window A_ON / A_OFF, default 8 / 10)
//       [--combo MODEL.gguf]   take the whole charge model (keys and passes) from a combined-model GGUF instead, e.g.
//                  the combo references / flows of feat/plato-lr-combo (CHARGE.gguf and FUKUI.json are then ignored);
//                  writes "M" (metallicity) per frame
//       [--bec [H]]   the exact Born effective charges "bec" [n][3][3] (dP_a / dx_b, P = sum q x, open frames), the
//                  acoustic sum-rule error "asr" (max |sum_i Z_i - Q I|), and with H > 0 their central-difference check
//                  "bec_fd_err" (step H A)
// FUKUI.json: {"Z": [...], "phi": [...], optional "phi_minus": [...]} (f = softplus(phi [+ a])). Writes
// {"frames": [{"q": [...], "q_raw": [...], "f": [...]}]}.
#include "mlipcpp/io.h"
#include "runtime/graph_model.h"
#include "runtime/lr_combined.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <memory>
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
  std::string fmode = "element", fscope = "global", fmodel, fmodel_m;
  double cw = 0.6, pmw = 0.05, con = 8, coff = 10, bech = -1;
  std::string combo;
  for (int i = 5; i < argc; i++) {
    const std::string a = argv[i];
    if (a == "--combo") { combo = argv[++i]; continue; }
    if (a == "--bec") {
      bech = 0;
      if (i + 1 < argc && argv[i + 1][0] != '-') bech = std::stod(argv[++i]);
      continue;
    }
    if (a == "--fukui-mode") { fmode = argv[++i]; continue; }
    if (a == "--fukui-scope") { fscope = argv[++i]; continue; }
    if (a == "--fmodel") { fmodel = argv[++i]; continue; }
    if (a == "--fmodel-minus") { fmodel_m = argv[++i]; continue; }
    if (a == "--cluster-w") { cw = std::stod(argv[++i]); continue; }
    if (a == "--pm-width") { pmw = std::stod(argv[++i]); continue; }
    if (a == "--clusters" && i + 2 < argc) { con = std::stod(argv[++i]), coff = std::stod(argv[++i]); continue; }
    if (a == "--qpass-charge") zero = std::string(argv[++i]) == "zero";
    else if (a == "--proj" && i + 3 < argc) pw = std::stod(argv[++i]), pon = std::stod(argv[++i]), poff = std::stod(argv[++i]);
    else if (a == "--fd") fdh = std::stod(argv[++i]);
    else if (a == "--backend") be = std::string(argv[++i]) == "cpu" ? BackendPreference::CPU : BackendPreference::CUDA;
  }
  try {
    runtime::LrcSpec L;
    std::unique_ptr<runtime::GraphModel> cm;
    if (!combo.empty()) {
      cm = std::make_unique<runtime::GraphModel>();
      cm->set_backend_preference(be);
      if (!cm->load_from_gguf(combo)) throw std::runtime_error("cannot load " + combo);
      L = cm->plato_lr().lrc;
      if (!L.enabled || !L.graph) throw std::runtime_error(combo + ": not an lrc graph-mode model");
    } else {
    L.enabled = L.graph = true, L.qpass_neutral = zero;
    const nlohmann::json fk = nlohmann::json::parse(std::ifstream(argv[2]));
    for (double z : fk["Z"]) L.fukui_Z.push_back(static_cast<int32_t>(z));
    for (double p : fk["phi"]) L.fukui_phi.push_back(p);
    if (fk.contains("phi_minus"))
      for (double p : fk["phi_minus"]) L.fukui_phi_minus.push_back(p);
    L.qsrc = runtime::make_lrc_graph_charges(argv[1], be, 0);
    using FM = runtime::LrcSpec::FukuiMode;
    L.fukui_mode = fmode == "uniform" ? FM::Uniform : fmode == "atom" ? FM::Atom : fmode == "atom_pm" ? FM::AtomPM : FM::Element;
    if (fmode != "element" && fmode != "uniform" && fmode != "atom" && fmode != "atom_pm")
      throw std::runtime_error("unknown --fukui-mode " + fmode);
    L.fukui_cluster = fscope == "cluster", L.fukui_cluster_w = cw, L.fukui_pm_width = pmw, L.a_on = con, L.a_off = coff;
    L.fmodel = fmodel, L.fmodel_minus = fmodel_m;
    if (!fmodel.empty()) L.fsrc = runtime::make_lrc_graph_charges(fmodel, be, 0);
    if (!fmodel_m.empty()) L.fsrc_minus = runtime::make_lrc_graph_charges(fmodel_m, be, 0);
    L.fukui_check();
    }
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
      nlohmann::json fr = {{"q", q.q0}, {"q_raw", q.qraw}, {"f", q.fp}, {"charge", sys.charge()}, {"n_atoms", n}};
      if (!q.fm.empty()) fr["f_minus"] = q.fm;
      if (!combo.empty()) fr["M"] = q.M, fr["q_ref"] = q.qref;
      if (bech >= 0) {
        const int32_t *Zs = sys.atomic_numbers();
        const std::vector<double> Zb = runtime::lrc_bec(L, S, q, n, Zs, x.data());
        double asr = 0;
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) {
            double t = 0;
            for (int i = 0; i < n; i++) t += Zb[9 * size_t(i) + 3 * a + b];
            asr = std::max(asr, std::abs(t - (a == b ? double(sys.charge()) : 0.0)));
          }
        fr["bec"] = Zb, fr["asr"] = asr;
        if (bech > 0) {
          double err = 0;
          for (int k = 0; k < 3 * n; k++) {
            std::vector<double> xp = x, xm = x;
            xp[k] += bech, xm[k] -= bech;
            const auto qp = runtime::lrc_q0(L, S, n, Zs, xp.data(), sys.charge(), sys.spin_multiplicity()).q0;
            const auto qm = runtime::lrc_q0(L, S, n, Zs, xm.data(), sys.charge(), sys.spin_multiplicity()).q0;
            for (int a = 0; a < 3; a++) {
              double Pp = 0, Pm = 0;
              for (int j = 0; j < n; j++) Pp += qp[j] * xp[3 * j + a], Pm += qm[j] * xm[3 * j + a];
              err = std::max(err, std::abs((Pp - Pm) / (2 * bech) - Zb[9 * size_t(k / 3) + 3 * a + k % 3]));
            }
          }
          fr["bec_fd_err"] = err;
        }
      }
      if (pw > 0) {
        runtime::LrcSpec Lp = L;
        Lp.qproj = true, Lp.qproj_w = pw, Lp.a_on = pon, Lp.a_off = poff;
        Lp.fukui_cluster = false; // --proj: lrc_qproj on these charges (diagnostic), whatever the Fukui scope
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
    nlohmann::json out = {{"model", argv[1]}, {"fukui", argv[2]}, {"qpass_charge", zero ? "zero" : "total"},
                        {"fukui_mode", fmode}, {"fukui_scope", fscope}, {"frames", frames}};
    std::ofstream(argv[4]) << out.dump(1);
  } catch (const std::exception &e) {
    std::cerr << "mlipcpp-qpass: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
