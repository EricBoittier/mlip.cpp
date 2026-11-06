// mlipcpp-sqe-check CASES.json: the host SQE (runtime/sqe.h) against metatrain float64 references
// (scripts/plato_lr/sqe_cases.py): energy, charges, forces and the training gradient dL/dtheta for
// L = a E + sum w . F (implicit / adjoint term included). Exit 1 on any mismatch above tolerance.
#include "runtime/sqe.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>

using json = nlohmann::json;
using namespace mlipcpp::runtime;

int main(int argc, char **argv) {
  if (argc < 2) return std::fprintf(stderr, "usage: mlipcpp-sqe-check CASES.json\n"), 2;
  std::ifstream f(argv[1]);
  const json J = json::parse(f);
  const json &sp = J["spec"];
  SqeSpec S;
  S.enabled = true;
  S.Z = sp["Z"].get<std::vector<int32_t>>();
  S.chi0 = sp["chi0"].get<std::vector<double>>(), S.J0 = sp["J0"].get<std::vector<double>>();
  S.rcov = sp["rcov"].get<std::vector<double>>(), S.sigma = sp["sigma"].get<std::vector<double>>();
  S.chi_a = sp["chi_a"].get<std::vector<double>>(), S.chi_b = sp["chi_b"].get<std::vector<double>>();
  S.j_z = sp["j_z"].get<std::vector<double>>(), S.kappa_z = sp["kappa_z"].get<std::vector<double>>();
  S.w_raw = sp["w_raw"].get<double>();
  if (sp.contains("q0_lo")) {
    S.q0_lo = sp["q0_lo"].get<std::vector<double>>(), S.q0_hi = sp["q0_hi"].get<std::vector<double>>();
    S.q0_a = sp["q0_a"].get<std::vector<double>>(), S.q0_b = sp["q0_b"].get<std::vector<double>>();
  } else {
    const size_t E = S.Z.size();
    S.q0_lo.assign(E, 0.0), S.q0_hi.assign(E, 0.0), S.q0_a.assign(E, 0.0), S.q0_b.assign(E * E, 0.0);
  }
  S.q0_learned = sp.value("learned", false);
  const json &c = sp["cfg"];
  S.r_ct = c["r_ct"], S.r_ct_width = c["r_ct_width"], S.dchi_max = c["dchi_max"], S.J_lo = c["J_lo"], S.J_hi = c["J_hi"];
  S.kappa_lo = c["kappa_lo"], S.kappa_hi = c["kappa_hi"], S.w_lo = c["w_lo"], S.w_hi = c["w_hi"];
  S.cn_on = c["cn_on"], S.cn_off = c["cn_off"];
  S.lr_R = c.value("lr_R", 0.0), S.lr_t = c.value("lr_t", 1.0);
  double mE = 0, mq = 0, mF = 0, mG = 0, mGrel = 0;
  int maxit = 0;
  for (const json &k : J["cases"]) {
    const auto Z = k["Z"].get<std::vector<int32_t>>();
    std::vector<double> x;
    for (const auto &p : k["pos"]) for (double v : p) x.push_back(v);
    const int n = static_cast<int>(Z.size());
    std::vector<double> q0;
    if (!k["q0"].is_null()) q0 = k["q0"].get<std::vector<double>>();
    const auto w = k["w"].get<std::vector<double>>();
    std::vector<double> grad(S.n_params(), 0.0);
    std::vector<int32_t> unit;
    std::vector<double> Qu;
    SqeQ0Supervision sup;
    if (k.value("ws", 0.0) > 0) {
      unit = k["unit"].get<std::vector<int32_t>>(), Qu = k["Qu"].get<std::vector<double>>();
      sup.unit = unit.data(), sup.Qu = Qu.data(), sup.n_units = static_cast<int>(Qu.size()), sup.weight = k["ws"];
    }
    const SqeOutput o = sqe_train_grad(S, n, Z.data(), x.data(), k["Q"].get<double>(), q0.empty() ? nullptr : q0.data(),
                                       k["a"].get<double>(), w.data(), grad, sup.unit ? &sup : nullptr);
    if (k.contains("q0ref")) {
      const auto q0r = k["q0ref"].get<std::vector<double>>();
      for (int i = 0; i < n; i++) mq = std::max(mq, std::abs(o.q0[i] - q0r[i]));
    }
    const auto qr = k["q"].get<std::vector<double>>(), Fr = k["F"].get<std::vector<double>>(), Gr = k["grad"].get<std::vector<double>>();
    double dE = std::abs(o.energy - k["E"].get<double>()), dq = 0, dF = 0, dG = 0, gs = 0;
    for (int i = 0; i < n; i++) dq = std::max(dq, std::abs(o.q[i] - qr[i]));
    for (int i = 0; i < 3 * n; i++) dF = std::max(dF, std::abs(o.forces[i] - Fr[i]));
    for (size_t i = 0; i < Gr.size(); i++) dG = std::max(dG, std::abs(grad[i] - Gr[i])), gs = std::max(gs, std::abs(Gr[i]));
    std::printf("%-28s n %3d  cg %3d  |dE| %.2e  |dq| %.2e  |dF| %.2e  |dgrad| %.2e (max |grad| %.2e)\n",
                k["name"].get<std::string>().c_str(), n, o.cg_iters, dE, dq, dF, dG, gs);
    mE = std::max(mE, dE), mq = std::max(mq, dq), mF = std::max(mF, dF), mG = std::max(mG, dG);
    mGrel = std::max(mGrel, dG / std::max(gs, 1e-3)), maxit = std::max(maxit, o.cg_iters); // relative above 1e-3
  }
  std::printf("max: |dE| %.2e eV  |dq| %.2e e  |dF| %.2e eV/A  |dgrad| %.2e (rel %.2e)  cg iters %d\n", mE, mq, mF, mG, mGrel, maxit);
  const bool ok = mE < 1e-9 && mq < 1e-9 && mF < 1e-8 && mGrel < 1e-7;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
