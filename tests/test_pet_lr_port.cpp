// PET-LR port (after M. Langer, ipsum/PET-LR): model-level checks of converted iris GGUFs
// (scripts/pet_lr_port/iris_to_gguf.py). By default the tiny seeded models committed in tests/resources
// (make_seeded.py tiny_* presets, his JAX init; petlr_tiny_trunk.gguf and one LR GGUF per reference type);
// MLIPCPP_PETLR_GGUF / MLIPCPP_PETLR_LR_GGUF point the trunk / LR checks at another GGUF instead:
//  - the full virial W_ab = sum r_a dE/dr_b equals dE/d eps_ba under a general (non-symmetric) homogeneous
//    deformation x -> (I + eps) x of positions and cell, by central differences; PET is not rotation
//    invariant, so W is not symmetric and its antisymmetric part is tested too;
//  - the Voigt stress is sym(W) / V;
//  - pet.conditioning_keys = spin_multiplicity: the energy does not depend on the charge.
#include "mlipcpp/mlipcpp.hpp"
#include "../src/runtime/graph_model.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace mlipcpp;

namespace {
std::string resource(const char *name) { return std::string(MLIPCPP_TEST_RESOURCES) + "/" + name; }
std::string trunk_gguf() {
  const char *e = std::getenv("MLIPCPP_PETLR_GGUF");
  return e && *e ? e : resource("petlr_tiny_trunk.gguf");
}
// the LR GGUFs: (label, path); every IrisLR reference / pass type of the tiny presets
std::vector<std::pair<std::string, std::string>> lr_ggufs() {
  if (const char *e = std::getenv("MLIPCPP_PETLR_LR_GGUF"); e && *e) return {{"env", e}};
  return {{"fermi (T 0.1, 1 pass, readout)", resource("petlr_tiny_fermi.gguf")},
          {"fermi gapped (T 0.02)", resource("petlr_tiny_cold.gguf")},
          {"species (softplus, 3 unshared passes, hardness, 2 transfers)", resource("petlr_tiny_species.gguf")},
          {"both (Gaussian charges, 2 shared passes)", resource("petlr_tiny_both.gguf")},
          // his branch feat/long-range-transfer (3ed96b0): make_seeded.py lrt_weighted / lrt_fill
          {"species_weighted (2 shared passes)", resource("petlr_tiny_weighted.gguf")},
          {"species_fill (weighted Fermi level)", resource("petlr_tiny_fill.gguf")},
          {"species_fill + long-range transfer", resource("petlr_tiny_lrt.gguf")},
          // his branch feat/conserving-charges (a98c2f6): make_seeded.py cc_fermi / cc_residual (one component per
          // structure: more raise, as his sample prep without fragment labels)
          {"conserving fermi (2 shared passes, hardness)", resource("petlr_tiny_ccfermi.gguf")},
          {"conserving residual (Q_c basis, q0 transfer, 3 unshared softplus passes)",
           resource("petlr_tiny_ccresidual.gguf")},
          // his QEq (a2658e3 qeq_reference: cg + custom_linear_solve; make_seeded.py tiny_qeq, 2 shared passes):
          // mlip.cpp's three-stage evaluation (host cg, cotangent cg, second backward)
          {"qeq (cg + custom_linear_solve, Gaussian charges, 2 shared passes)", resource("petlr_tiny_qeq.gguf")}};
}

// 24 water molecules in a 9.0 A cube, a fixed jittered lattice (deterministic)
void water_box(std::vector<float> &x, std::vector<int32_t> &Z, float cell[9]) {
  const double L = 9.0, a = L / 3.0;
  uint32_t s = 12345;
  auto rnd = [&s] { s = s * 1664525u + 1013904223u; return (s >> 8) / double(1u << 24) - 0.5; };
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) {
        if (x.size() / 3 >= 72) break;
        const double o[3] = {(i + 0.5) * a + 0.3 * rnd(), (j + 0.5) * a + 0.3 * rnd(), (k + 0.5) * a + 0.3 * rnd()};
        const double h1[3] = {0.757, 0.586, 0.1 * rnd()}, h2[3] = {-0.757, 0.586, 0.1 * rnd()};
        for (int c = 0; c < 3; c++) x.push_back(float(o[c]));
        for (int c = 0; c < 3; c++) x.push_back(float(o[c] + h1[c]));
        for (int c = 0; c < 3; c++) x.push_back(float(o[c] + h2[c]));
        Z.insert(Z.end(), {8, 1, 1});
      }
  for (int c = 0; c < 9; c++) cell[c] = (c % 4 == 0) ? float(L) : 0.0f;
}
} // namespace

TEST_CASE("pet_lr_port: full virial = dE/d eps (non-symmetric deformation), Voigt = sym / V", "[pet_lr_port]") {
  const std::string path = trunk_gguf();
  ModelOptions mo;
  Predictor model(path, mo);
  std::vector<float> x;
  std::vector<int32_t> Z;
  float cell[9];
  water_box(x, Z, cell);
  const int n = int(Z.size());
  const bool pbc[3] = {true, true, true};
  PredictOptions po;
  po.compute_forces = true, po.compute_stress = true;
  const Result r = model.predict(n, x.data(), Z.data(), cell, pbc, po);
  REQUIRE(r.virial.size() == 9);
  // dE/d eps_ij by central differences in double positions rounded to float (h large enough for fp32)
  const double h = 2e-3;
  auto energy = [&](int i, int j, double e) {
    std::vector<float> y(x.size());
    float c2[9];
    for (int k = 0; k < n; k++)
      for (int a = 0; a < 3; a++) y[3 * k + a] = float(x[3 * k + a] + (a == i ? e * x[3 * k + j] : 0.0));
    for (int v = 0; v < 3; v++)
      for (int a = 0; a < 3; a++) c2[3 * v + a] = float(cell[3 * v + a] + (a == i ? e * cell[3 * v + j] : 0.0));
    return double(model.predict(n, y.data(), Z.data(), c2, pbc, po).energy);
  };
  double scale = 0, worst = 0, anti = 0;
  for (double w : r.virial) scale = std::max(scale, std::abs(w));
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      const double fd = (energy(i, j, h) - energy(i, j, -h)) / (2 * h);
      worst = std::max(worst, std::abs(fd - r.virial[3 * j + i]));
      anti = std::max(anti, std::abs(r.virial[3 * i + j] - r.virial[3 * j + i]));
    }
  INFO("max |FD - W| = " << worst << " eV, max |W| = " << scale << " eV, max |W - W^T| = " << anti);
  CHECK(worst < 2e-3 * scale + 2e-3); // fp32 energies; h = 2e-3
  // Voigt stress = sym(W) / V
  const double V = 9.0 * 9.0 * 9.0;
  const int va[6] = {0, 1, 2, 1, 0, 0}, vb[6] = {0, 1, 2, 2, 2, 1};
  for (int k = 0; k < 6; k++) {
    const double sym = 0.5 * (r.virial[3 * va[k] + vb[k]] + r.virial[3 * vb[k] + va[k]]) / V;
    CHECK(std::abs(sym - r.stress[k]) < 1e-5 * scale / V + 1e-7);
  }
}

TEST_CASE("pet_lr_port: spin-only conditioning ignores the charge", "[pet_lr_port]") {
  const std::string path = trunk_gguf();
  ModelOptions mo;
  Predictor model(path, mo);
  const std::vector<float> x = {0, 0, 0, 0.757f, 0.586f, 0, -0.757f, 0.586f, 0};
  const std::vector<int32_t> Z = {8, 1, 1};
  PredictOptions po;
  po.compute_forces = true;
  po.spin_multiplicity = 2;
  po.charge = 0;
  const double e0 = model.predict(3, x.data(), Z.data(), nullptr, nullptr, po).energy;
  po.charge = 1;
  CHECK(model.predict(3, x.data(), Z.data(), nullptr, nullptr, po).energy == e0);
  po.spin_multiplicity = 1;
  CHECK(model.predict(3, x.data(), Z.data(), nullptr, nullptr, po).energy != e0);
}

// ---- iris PET-LR long-range branch (pet.iris_lr GGUFs: iris_to_gguf.py --lr) ----
#include "../src/runtime/iris_lr.h"

TEST_CASE("pet_lr_port: iris fermi_level host side (bisection, gapped decision, tie split)", "[pet_lr_port]") {
  using namespace mlipcpp::runtime::iris;
  FermiConfig c;
  std::vector<double> s = {0.31, -1.2, 2.05, 0.77, -0.4, 1.6, -2.3, 0.05, 0.9, -0.66, 1.1, 0.2};
  for (double Q : {-2.0, 0.0, 1.0, 3.0}) {
    const FermiHost h = fermi_host(s, Q, c);
    double q = 0;
    for (double x : s) q += occupation(x - h.mu, c);
    CHECK(std::fabs(q - Q) < 1e-9); // 40 halvings of a ~12-wide bracket
    CHECK_FALSE(h.gapped);          // T = 0.1: the density at any integer filling is >> eps
    double wt = 0, wb = 0;
    for (size_t i = 0; i < s.size(); i++) wt += h.top[i], wb += h.bottom[i];
    CHECK(std::fabs(wt - 1) < 1e-6);
    CHECK(std::fabs(wb - 1) < 1e-6);
  }
  // T = 0.02, scores 0.1-0.2 above integers (m = 1, 1, 1, -1, 0, -1; sum 1 = Q): gapped; s - m = 0.2 three times
  // (top, 1/3 each) and 0.1 - 2e-17 twice (-0.9 + 1; bottom, 1/2 each; 0.1 - 0 is larger in floating point)
  c.temperature = 0.02;
  const std::vector<double> g = {1.2, 1.2, 1.2, -0.9, 0.1, -0.9};
  const FermiHost h = fermi_host(g, 1.0, c);
  CHECK(h.gapped);
  for (int i = 0; i < 3; i++) CHECK(std::fabs(h.top[i] - 1.0 / 3.0) < 1e-7);
  CHECK(h.top[4] == 0.f);
  CHECK(std::fabs(h.bottom[3] - 0.5) < 1e-7);
  CHECK(std::fabs(h.bottom[5] - 0.5) < 1e-7);
}

TEST_CASE("pet_lr_port: iris fermi_level with weights (species_fill): sum_i w_i f(s_i - mu) = Q", "[pet_lr_port]") {
  using namespace mlipcpp::runtime::iris;
  FermiConfig c;
  const std::vector<double> s = {0.31, -1.2, 2.05, 0.77, -0.4, 1.6, -2.3, 0.05, 0.9, -0.66, 1.1, 0.2};
  const std::vector<double> w = {0.7, 1.3, 0.45, 2.1, 1.0, 0.9, 1.6, 0.33, 1.2, 0.8, 1.05, 0.6};
  const std::vector<double> ones(s.size(), 1.0);
  for (double Q : {-2.0, 0.0, 1.0, 3.0}) {
    const FermiHost h = fermi_host(s, Q, c, &w);
    double q = 0;
    for (size_t i = 0; i < s.size(); i++) q += w[i] * occupation(s[i] - h.mu, c);
    CHECK(std::fabs(q - Q) < 1e-9);
    CHECK(fermi_host(s, Q, c, &ones).mu == fermi_host(s, Q, c).mu); // unit weights: the unweighted bisection
  }
  // gapped with weights: |sum_i w_i m_i - Q| < eps (not 0.5): two atoms on m = 1 with w = 0.5, 1.5 at Q = 2
  c.temperature = 0.02;
  const std::vector<double> g = {1.2, 1.15}, wg = {0.5, 1.5};
  CHECK(fermi_host(g, 2.0, c, &wg).gapped);
  CHECK_FALSE(fermi_host(g, 2.0 + 1e-3, c, &wg).gapped);
}

TEST_CASE("pet_lr_port: iris QEq host cg = jax.scipy cg (recurrences, stopping rule, truncation)", "[pet_lr_port]") {
  using namespace mlipcpp::runtime::iris;
  // a fixed SPD matrix (diagonally dominant, n = 6) with a Jacobi preconditioner
  const int n = 6;
  double A[n][n];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) A[i][j] = i == j ? 4.0 + i : 1.0 / (1.0 + std::abs(i - j));
  auto mv = [&](const std::vector<double> &x) {
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) y[i] += A[i][j] * x[j];
    return y;
  };
  auto jac = [&](const std::vector<double> &r) {
    std::vector<double> z(n);
    for (int i = 0; i < n; i++) z[i] = r[i] / A[i][i];
    return z;
  };
  const std::vector<double> b = {1.0, -2.0, 0.5, 3.0, -1.0, 0.25};
  const CGResult full = cg_host(mv, b, jac, 100, 1e-14);
  const std::vector<double> Ax = mv(full.x);
  double err = 0;
  for (int i = 0; i < n; i++) err = std::max(err, std::fabs(Ax[i] - b[i]));
  CHECK(err < 1e-12);
  CHECK(full.iterations <= n + 1); // finite termination (+ roundoff)
  CHECK(full.final_ratio < 1.0);
  // truncation: maxiter binds, the iterate is the k-step Krylov one (not converged)
  const CGResult two = cg_host(mv, b, jac, 2, 1e-14);
  CHECK(two.iterations == 2);
  CHECK(two.final_ratio > 1.0);
  // the relative tolerance stops early (his tol 1e-5): |r| <= 1e-5 |b|
  const CGResult tol = cg_host(mv, b, jac, 100, 1e-5);
  CHECK(tol.iterations <= full.iterations);
  CHECK(tol.final_ratio <= 1.0);
  // zero right-hand side (a single atom: P b = 0): no iteration, x = 0, no NaN
  const CGResult zero = cg_host(mv, std::vector<double>(n, 0.0), jac, 20, 1e-5);
  CHECK(zero.iterations == 0);
  for (double x : zero.x) CHECK(x == 0.0);
}

namespace {
double iris_energy(Predictor &m, const std::vector<float> &x, const std::vector<int32_t> &Z, int Q,
                   const PredictOptions &base) {
  PredictOptions po = base;
  po.compute_forces = false;
  po.charge = Q;
  return m.predict(int(Z.size()), x.data(), Z.data(), nullptr, nullptr, po).energy;
}

// forces = -dE/dx by central differences, Richardson-extrapolated from h = 8e-3 and 4e-3 A (the error falls with h: fp32 noise, not truncation) (O(h^4): the tiny
// seeded models are rough, forces up to 1e5 eV/A on close contacts); tolerance ftol + 1e-3 max|F|; charges sum to Q;
// the dipole is sum_i q_i (r_i - r0); max|F| > fmin (not trivially zero)
double iris_check(Predictor &m, const std::vector<float> &x, const std::vector<int32_t> &Z, int Q, double ftol,
                  double fmin = 0.05) {
  PredictOptions po;
  po.compute_forces = true;
  po.charge = Q;
  po.spin_multiplicity = 1;
  po.dipole_origin = {0.1, -0.2, 0.3};
  const auto r = m.predict(int(Z.size()), x.data(), Z.data(), nullptr, nullptr, po);
  REQUIRE(r.charges.size() == Z.size());
  REQUIRE(r.forces.size() == 3 * Z.size());
  double qs = 0, D[3] = {0, 0, 0};
  for (size_t i = 0; i < Z.size(); i++) {
    qs += r.charges[i];
    for (int a = 0; a < 3; a++) D[a] += r.charges[i] * (x[3 * i + a] - po.dipole_origin[a]);
  }
  CHECK(std::fabs(qs - Q) < 2e-4);
  for (int a = 0; a < 3; a++) CHECK(std::fabs(D[a] - r.dipole[a]) < 2e-4 * std::max(1.0, std::fabs(D[a])));
  auto fd = [&](size_t k, double h) {
    std::vector<float> xp = x, xm = x;
    xp[k] += float(h), xm[k] -= float(h);
    return -(iris_energy(m, xp, Z, Q, po) - iris_energy(m, xm, Z, Q, po)) / (double(xp[k]) - double(xm[k]));
  };
  double worst = 0, fmax = 0;
  for (size_t k = 0; k < x.size(); k++) {
    const double f = (4.0 * fd(k, 4e-3) - fd(k, 8e-3)) / 3.0;
    worst = std::max(worst, std::fabs(f - r.forces[k]));
    fmax = std::max(fmax, double(std::fabs(r.forces[k])));
  }
  // + the a-priori fp32 noise of the Richardson FD: 4 ulps of |E| per energy, (4 / (2 h1) + 1 / (2 h2)) / 3 x 2 = 375
  // per unit energy noise (h1 = 4e-3, h2 = 8e-3); binding on the GPU (unordered reductions) for nearly force-free
  // structures (species_weighted Na / Cl 8 A apart: max|F| 1e-4, FD 2.4e-5 off on the H100, 2026-11-11)
  const double noise = 375.0 * 4.0 * 5.96e-8 * std::fabs(double(r.energy));
  INFO("max |F_FD - F| = " << worst << " eV/A, max |F| = " << fmax << ", fp32 FD noise bound " << noise);
  CHECK(worst < ftol + 1e-3 * fmax + noise);
  CHECK(fmax > fmin);
  return r.metallicity;
}
} // namespace

TEST_CASE("pet_lr_port: iris PET-LR forces = -dE/dx (FD), charge sum, dipole; every reference type; several (n, k)",
          "[pet_lr_port]") {
  // H3O+ ... H2O (7 atoms, Q = +1)
  const std::vector<float> a = {0, 0, 0, 0.98f, 0.1f, 0, -0.35f, 0.92f, 0.05f, -0.33f, -0.5f, 0.8f,
                                2.55f, 0.2f, 0.1f, 3.1f, 0.95f, 0.2f, 3.05f, -0.55f, -0.1f};
  // water trimer (9 atoms, Q = 0)
  const std::vector<float> b = {0, 0, 0, 0.96f, 0, 0, -0.24f, 0.93f, 0, 2.8f, 0.3f, 0.2f, 3.3f, 1.1f, 0.3f,
                                3.2f, -0.4f, 0.7f, 1.2f, 2.6f, -0.3f, 0.5f, 3.2f, -0.5f, 1.9f, 3.1f, 0.2f};
  // Na+ and Cl- 7 A apart with a water each: two pieces, LR between them (Q = 0)
  const std::vector<float> c = {0, 0, 0, 2.3f, 0, 0, 2.9f, 0.75f, 0.1f, 2.85f, -0.7f, -0.1f,
                                7.0f, 0, 0, 10.1f, 0.2f, 0, 10.6f, 0.9f, 0.3f, 10.55f, -0.55f, -0.2f};
  for (const auto &[label, path] : lr_ggufs()) {
    INFO("model: " << label << " (" << path << ")");
    ModelOptions mo;
    Predictor m(path, mo);
    const double ga = iris_check(m, a, {8, 1, 1, 1, 8, 1, 1}, 1, 5e-3);
    const double gb = iris_check(m, b, {8, 1, 1, 8, 1, 1, 8, 1, 1}, 0, 5e-3);
    const double gc = iris_check(m, c, {11, 8, 1, 1, 17, 8, 1, 1}, 0, 5e-3);
    if (label.rfind("fermi gapped", 0) == 0) { // the gapped branch (metallicity ~ 0) must be what the FD test saw
      INFO("metallicity: " << ga << " " << gb << " " << gc);
      CHECK(std::min({ga, gb, gc}) < 1e-6);
    }
  }
}

TEST_CASE("pet_lr_port: iris PET-LR on atoms without neighbours keeps the Coulomb forces", "[pet_lr_port]") {
  for (const auto &[label, path] : lr_ggufs()) {
    INFO("model: " << label);
    ModelOptions mo;
    Predictor m(path, mo);
    // Na+ and Cl- 8 A apart: no edge within the 5 A cutoff. The tiny models put |q| 0.01-0.17 e on them (F ~ 1e-5 -
    // 7e-3 eV/A), the gapped one rounds both to 0 (no force: nothing to require)
    const bool gapped = label.rfind("fermi gapped", 0) == 0;
    if (label.rfind("conserving", 0) == 0) { // two components: raises (his sample prep without fragment labels)
      const std::vector<float> x2 = {0, 0, 0, 8.0f, 0.3f, -0.2f};
      const std::vector<int32_t> z2 = {11, 17};
      PredictOptions p0;
      p0.charge = 0, p0.spin_multiplicity = 1;
      CHECK_THROWS(m.predict(2, x2.data(), z2.data(), nullptr, nullptr, p0));
      continue;
    }
    // (FD floor of fp32 energies here ~1e-5 eV/A: the Gaussian-charge model's 1.4e-5 eV/A sits at it)
    iris_check(m, {0, 0, 0, 8.0f, 0.3f, -0.2f}, {11, 17}, 0, 2e-5, gapped ? -1.0 : 5e-6);
  }
}

TEST_CASE("pet_lr_port: iris QEq three-stage evaluation: both cg solves run; his tol 1e-5 = a converged solve here",
          "[pet_lr_port]") {
  // water trimer (9 atoms): with his settings (maxiter 20, tol 1e-5) the cg stops below the tolerance; a converged
  // solve (tol 1e-7, near the fp32 floor of the operator; tol 0 would iterate on roundoff and diverge) gives the same
  // energy, charges and forces to fp32
  // noise, so the FD test above (which differentiates the energy of the solved charges) checks the implicit gradient
  const std::vector<float> x = {0, 0, 0, 0.96f, 0, 0, -0.24f, 0.93f, 0, 2.8f, 0.3f, 0.2f, 3.3f, 1.1f, 0.3f,
                                3.2f, -0.4f, 0.7f, 1.2f, 2.6f, -0.3f, 0.5f, 3.2f, -0.5f, 1.9f, 3.1f, 0.2f};
  const std::vector<int32_t> Z = {8, 1, 1, 8, 1, 1, 8, 1, 1};
  PredictOptions po;
  po.compute_forces = true, po.charge = 0, po.spin_multiplicity = 1;
  ModelOptions his, conv;
  conv.iris_qeq_maxiter = 100, conv.iris_qeq_tol = 1e-7;
  Predictor a(resource("petlr_tiny_qeq.gguf"), his), b(resource("petlr_tiny_qeq.gguf"), conv);
  const auto ra = a.predict(9, x.data(), Z.data(), nullptr, nullptr, po);
  const auto rb = b.predict(9, x.data(), Z.data(), nullptr, nullptr, po);
  double fmax = 0, df = 0, dq = 0;
  for (size_t k = 0; k < ra.forces.size(); k++)
    fmax = std::max(fmax, double(std::fabs(ra.forces[k]))), df = std::max(df, double(std::fabs(ra.forces[k] - rb.forces[k])));
  for (size_t i = 0; i < ra.charges.size(); i++) dq = std::max(dq, double(std::fabs(ra.charges[i] - rb.charges[i])));
  INFO("E " << ra.energy << " vs " << rb.energy << ", max|dF| " << df << " (max|F| " << fmax << "), max|dq| " << dq
            << ", |P (A q + chi)| " << ra.metallicity);
  CHECK(std::fabs(ra.energy - rb.energy) < 1e-5 * std::fabs(ra.energy) + 1e-5);
  CHECK(df < 1e-4 * fmax + 1e-4);
  CHECK(dq < 1e-5);
  CHECK(ra.metallicity < 1e-4); // the stationarity residual (his qeq_residual, eV) of the solved charges
  // truncated (maxiter 2 < the 8 iterations n - 1 needs here): the charges move, and the solve is reported unconverged
  ModelOptions trunc;
  trunc.iris_qeq_maxiter = 2;
  Predictor c(resource("petlr_tiny_qeq.gguf"), trunc);
  const auto rc = c.predict(9, x.data(), Z.data(), nullptr, nullptr, po);
  double dqt = 0;
  for (size_t i = 0; i < ra.charges.size(); i++) dqt = std::max(dqt, double(std::fabs(ra.charges[i] - rc.charges[i])));
  CHECK(dqt > 1e-3);
  CHECK(rc.metallicity > 1e-3);
}

// ---- periodic cells (his jax-pme tiled Ewald at the calculator's frozen parameters; graph_pbc.json) ----
TEST_CASE("pet_lr_port: iris PET-LR in a 3-D cell and a slab: forces = -dE/dx and the virial = dE/d eps (FD)",
          "[pet_lr_port]") {
  // 3 waters of the jittered lattice in its 9 A cube: sparse enough that every atom keeps all its pairs within the
  // 5 A trunk cutoff (the adaptive cutoff saturates), so the trunk list, and with it his Gaussian-charge SR
  // correction (erfc / r on the SR edges: discontinuous where the selection changes), is fixed under the FD steps;
  // ~50 eV total energies keep the fp32 FD noise ~1e-3 eV/A
  std::vector<float> x;
  std::vector<int32_t> Z;
  float cell[9];
  water_box(x, Z, cell);
  x.resize(27), Z.resize(9);
  const int n = int(Z.size());
  for (const auto &[label, path] : lr_ggufs()) {
    // the long-range transfer's periodic kernel is the k-space sum at the batch's Ewald smearing (2 wavelengths, a
    // function of the cell; jax-pme TiledEwald lr_k2(sr_batch.smearing, k2)), so unlike the Ewald energy its energy
    // depends on the smearing: his stress is dE/d eps at FIXED Ewald parameters (the batch smearing is a host
    // constant), not the derivative of per-strain re-derived energies (those differ by (sigma / 3) dE/d sigma on the
    // diagonal: 9 eV of 179 here). The strain FD of that model runs at frozen parameters (the unstrained frame's)
    const bool lrt = label.find("long-range transfer") != std::string::npos;
    ModelOptions mo;
    mo.iris_freeze_ewald = lrt;
    Predictor m(path, mo);
    for (int slab = 0; slab < 2; slab++) {
      INFO("model: " << label << (slab ? ", slab (pbc T T F, 22 A vacuum vector)" : ", 3-D cell"));
      float c[9];
      std::copy(cell, cell + 9, c);
      if (slab) c[8] = 22.0f;
      const bool pbc[3] = {true, true, !slab};
      if (lrt && slab) { // his ToSample raises on a slab
        PredictOptions p0;
        p0.charge = 0, p0.spin_multiplicity = 1;
        CHECK_THROWS(m.predict(n, x.data(), Z.data(), c, pbc, p0));
        continue;
      }
      PredictOptions po;
      po.compute_forces = true, po.compute_stress = true;
      po.charge = 0, po.spin_multiplicity = 1;
      const Result r = m.predict(n, x.data(), Z.data(), c, pbc, po);
      REQUIRE(r.forces.size() == 3 * size_t(n));
      REQUIRE(r.virial.size() == 9);
      double qs = 0;
      for (float q : r.charges) qs += q;
      CHECK(std::fabs(qs) < 2e-4);
      auto energy = [&](const std::vector<float> &y, const float *cc) {
        PredictOptions p = po;
        p.compute_forces = false, p.compute_stress = false;
        return double(m.predict(n, y.data(), Z.data(), cc, pbc, p).energy);
      };
      // forces: the three coordinates of up to 5 atoms, Richardson from h = 2e-3 and 1e-3 A. His real-space Ewald sum
      // drops pairs at r >= rc (a jump of ~erfc(2.83) q_i q_j / rc in the energy), and his forces are the
      // derivative at a fixed pair set: FD only atoms with no pair image within 0.02 A of rc
      double cd[3][3];
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) cd[a][b] = c[3 * a + b];
      std::vector<double> xd(x.begin(), x.end());
      const auto es = mlipcpp::runtime::iris::ewald_setup(cd, pbc, xd.data(), n, 1024, 0.0, 2.0);
      std::vector<int> atoms_fd;
      for (int i = 0; i < n && atoms_fd.size() < 5; i++) {
        bool near = false;
        for (int j = 0; j < n && !near; j++)
          for (int s0 = -2; s0 <= 2 && !near; s0++)
            for (int s1 = -2; s1 <= 2 && !near; s1++)
              for (int s2 = (slab ? 0 : -2); s2 <= (slab ? 0 : 2) && !near; s2++) {
                double d2 = 0;
                for (int a = 0; a < 3; a++) {
                  const double v = x[3 * j + a] - x[3 * i + a] + s0 * cd[0][a] + s1 * cd[1][a] + s2 * cd[2][a];
                  d2 += v * v;
                }
                near = std::fabs(std::sqrt(d2) - es.cutoff) < 0.02;
              }
        if (!near) atoms_fd.push_back(i);
      }
      REQUIRE(atoms_fd.size() >= 3);
      double worst = 0, fmax = 0;
      for (double f : r.forces) fmax = std::max(fmax, std::fabs(f));
      for (int idx = 0; idx < 3 * int(atoms_fd.size()); idx++) {
        const int k = 3 * atoms_fd[idx / 3] + idx % 3;
        auto fd = [&](double h) {
          std::vector<float> xp = x, xm = x;
          xp[k] += float(h), xm[k] -= float(h);
          return -(energy(xp, c) - energy(xm, c)) / (double(xp[k]) - double(xm[k]));
        };
        // (steps 2e-3 / 1e-3 A: at 8e-3 the adaptive selection of the Gaussian-charge model's trunk changes,
        // whose SR erfc correction then jumps; torch fp64 FD at 1e-5 = autograd = these forces to 6e-7). The fp32
        // energy noise (his Gaussian-charge model: ~1e-4 eV on -174 eV, i.e. ~0.1 eV/A at these steps) is
        // measured as |fd(h) - fd(2h)| (the truncation part of that is ~1e-6 here) and allowed for; a wrong
        // derivative is systematic, the same in both differences, and is not absorbed
        const double f1 = fd(1e-3), f2 = fd(2e-3);
        worst = std::max(worst, std::fabs((4.0 * f1 - f2) / 3.0 - r.forces[k]) - 2.0 * std::fabs(f1 - f2));
      }
      INFO("max |F_FD - F| = " << worst << " eV/A, max |F| = " << fmax);
      CHECK(worst < 5e-3 + 1e-3 * fmax);
      CHECK(fmax > 0.05);
      if (slab) continue; // (his calculator reports no stress for a slab; the virial is his predict's, not a strain derivative)
      const double h = 2e-3;
      auto strained = [&](int i, int j, double e) {
        std::vector<float> y(x.size());
        float c2[9];
        for (int k = 0; k < n; k++)
          for (int a = 0; a < 3; a++) y[3 * k + a] = float(x[3 * k + a] + (a == i ? e * x[3 * k + j] : 0.0));
        for (int v = 0; v < 3; v++)
          for (int a = 0; a < 3; a++) c2[3 * v + a] = float(c[3 * v + a] + (a == i ? e * c[3 * v + j] : 0.0));
        return energy(y, c2);
      };
      // Richardson from h = 2e-3 / 1e-3 (the strain energy is curved: a plain central difference at 2e-3 is off by
      // ~0.3 eV of 179 on the long-range-transfer model) with the a-priori fp32 energy noise (2 ulp of E per energy)
      // allowed for, as in the slab test
      const double E0 = std::abs(double(r.energy));
      const double noise = 3.0 * (2.0 * 1.2e-7 * E0) / (h / 2);
      double scale = 0, wv = 0;
      for (double w : r.virial) scale = std::max(scale, std::abs(w));
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          auto fd = [&](double hh) { return (strained(i, j, hh) - strained(i, j, -hh)) / (2 * hh); };
          const double f1 = fd(h / 2), f2 = fd(h);
          wv = std::max(wv, std::abs((4.0 * f1 - f2) / 3.0 - r.virial[3 * j + i]));
        }
      INFO("max |FD - W| = " << wv << " eV, max |W| = " << scale << " eV, fp32 noise bound " << noise << " eV");
      CHECK(wv < 2e-3 * scale + 2e-3 + noise);
    }
  }
}

TEST_CASE("pet_lr_port: slab virial chain rule (normal / basis): W = dE/d eps at frozen Ewald parameters (FD)",
          "[pet_lr_port]") {
  // His predict's slab virial (his calculator reports no slab stress) is the strain derivative with the Ewald
  // parameters of the frame held (his compose_cell: the effective non-periodic row a constant, wavelength /
  // smearing / k grid fixed): mlip.cpp's host chain rule through the slab normal n = (a1 x a2) / |a1 x a2| and the
  // basis |h_k . n| is checked here against that derivative, with the Ewald parameters frozen at the unstrained
  // frame (ModelOptions::iris_freeze_ewald). All nine components, positions and every cell row strained.
  std::vector<float> x;
  std::vector<int32_t> Z;
  float cell[9];
  water_box(x, Z, cell);
  x.resize(27), Z.resize(9);
  // plus Na below and Cl above the three waters: a large dipole along the normal (|q| ~0.1 e on 11 A), so the slab
  // (Yeh-Berkowitz) term and with it the normal / basis chain rule carry a visible share of the virial
  x.insert(x.end(), {4.0f, 4.6f, -1.0f, 4.9f, 4.4f, 10.0f});
  Z.insert(Z.end(), {11, 17});
  const int n = int(Z.size());
  float c[9];
  std::copy(cell, cell + 9, c);
  c[8] = 22.0f;
  const bool pbc[3] = {true, true, false};
  for (const auto &[label, path] : lr_ggufs()) {
    INFO("model: " << label << ", slab (pbc T T F, 22 A vacuum vector)");
    if (label.find("long-range transfer") != std::string::npos) continue; // refused on a slab (his ToSample)
    ModelOptions mo;
    mo.iris_freeze_ewald = true;
    Predictor m(path, mo);
    PredictOptions po;
    po.compute_forces = true, po.compute_stress = true;
    po.charge = 0, po.spin_multiplicity = 1;
    const Result r = m.predict(n, x.data(), Z.data(), c, pbc, po); // freezes the parameters on this frame
    REQUIRE(r.virial.size() == 9);
    PredictOptions pe = po;
    pe.compute_forces = false, pe.compute_stress = false;
    auto strained = [&](int i, int j, double e) {
      std::vector<float> y(x.size());
      float c2[9];
      for (int k = 0; k < n; k++)
        for (int a = 0; a < 3; a++) y[3 * k + a] = float(x[3 * k + a] + (a == i ? e * x[3 * k + j] : 0.0));
      for (int v = 0; v < 3; v++)
        for (int a = 0; a < 3; a++) c2[3 * v + a] = float(c[3 * v + a] + (a == i ? e * c[3 * v + j] : 0.0));
      return double(m.predict(n, y.data(), Z.data(), c2, pbc, pe).energy);
    };
    // Richardson from h = 2e-3 / 1e-3 (the strain energy is strongly curved here: fd(h) moves by ~1 eV between
    // h = 5e-4 and 4e-3, so |fd(h) - fd(2h)| is truncation, not noise, and is NOT allowed for); the fp32 energy
    // noise is bounded a priori: 2 ulp of E per energy, i.e. <= (4/3 + 1/6) * 2 * 2 ulp(E) / (2 h) in the extrapolate
    const double E0 = std::abs(double(r.energy));
    const double noise = 3.0 * (2.0 * 1.2e-7 * E0) / 1e-3;
    double scale = 0, wv = 0, anti = 0;
    for (double w : r.virial) scale = std::max(scale, std::abs(w));
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++) {
        auto fd = [&](double h) { return (strained(i, j, h) - strained(i, j, -h)) / (2 * h); };
        const double f1 = fd(1e-3), f2 = fd(2e-3);
        wv = std::max(wv, std::abs((4.0 * f1 - f2) / 3.0 - r.virial[3 * j + i]));
        anti = std::max(anti, std::abs(r.virial[3 * i + j] - r.virial[3 * j + i]));
      }
    INFO("max |FD - W| = " << wv << " eV, max |W| = " << scale << " eV, max antisym " << anti << ", fp32 noise bound "
                           << noise << " eV (|E| " << E0 << ")");
    CHECK(wv < 2e-3 * scale + 2e-3 + noise);
    CHECK(scale > 0.05);
    // the normal-row (vacuum) components are not trivially zero through the positions
    CHECK(std::abs(r.virial[8]) > 1e-3);
  }
}

TEST_CASE("pet_lr_port: iris Ewald parameters frozen per run (his calculator) vs per call (his oracle)",
          "[pet_lr_port]") {
  using mlipcpp::runtime::GraphModel;
  namespace iris = mlipcpp::runtime::iris;
  std::vector<float> x;
  std::vector<int32_t> Z;
  float cell[9];
  water_box(x, Z, cell);
  x.resize(27), Z.resize(9);
  const int n = int(Z.size());
  const auto path = lr_ggufs().front().second;
  GraphModel a, b;
  REQUIRE(a.load_from_gguf(path));
  REQUIRE(b.load_from_gguf(path));
  b.set_iris_freeze_ewald(true);
  const int num_k = a.iris_lr().num_k;
  auto sys = [&](const std::vector<float> &y, int m, const float *c, bool slab) {
    float L[3][3];
    for (int r = 0; r < 3; r++)
      for (int q = 0; q < 3; q++) L[r][q] = c[3 * r + q];
    AtomicSystem s(m, y.data(), Z.data());
    s.set_cell(Cell(L, true, true, !slab));
    return s;
  };
  auto fresh = [&](const AtomicSystem &s) { // his _freeze on this frame
    double h[3][3];
    bool p[3];
    for (int r = 0; r < 3; r++) {
      p[r] = s.cell()->periodic[r];
      for (int q = 0; q < 3; q++) h[r][q] = s.cell()->matrix[r][q];
    }
    std::vector<double> pos(s.positions(), s.positions() + 3 * s.num_atoms());
    return iris::ewald_freeze(h, p, pos.data(), s.num_atoms(), num_k, a.iris_lr().smearing, a.iris_lr().slab_margin)
        .to_vec();
  };
  // the first call of a system: the frozen parameters are this frame's, and the result is bit for bit per call
  const AtomicSystem s0 = sys(x, n, cell, false);
  const auto r0a = a.predict(s0, true), r0b = b.predict(s0, true);
  CHECK(r0a.energy == r0b.energy);
  // (forces: bit for bit on CPU; CUDA's backward scatter is not run-to-run deterministic, ~1e-6 eV/A)
  double df = 0;
  for (size_t k = 0; k < r0a.forces.size(); k++) df = std::max(df, double(std::fabs(r0a.forces[k] - r0b.forces[k])));
  CHECK(df < 1e-5);
  const std::vector<double> p0 = b.iris_frozen_ewald();
  CHECK(p0 == fresh(s0));
  // a 6 % larger cell (an NPT-like move): frozen keeps the first frame's parameters; per call re-derives them
  float c2[9];
  std::vector<float> y(x);
  for (int k = 0; k < 9; k++) c2[k] = cell[k] * 1.06f;
  for (float &v : y) v *= 1.06f;
  const AtomicSystem s1 = sys(y, n, c2, false);
  const auto r1a = a.predict(s1, true), r1b = b.predict(s1, true);
  CHECK(b.iris_frozen_ewald() == p0);
  CHECK(fresh(s1)[9] != p0[9]); // (the wavelength would have moved)
  // Ewald converged at both parameter sets: the energies agree to the fp32 level
  CHECK(std::abs(double(r1a.energy) - double(r1b.energy)) < 1e-5 * std::max(1.0, std::abs(double(r1a.energy))));
  // a new system (three atoms fewer) refreezes
  const AtomicSystem s2 = sys(y, n - 3, c2, false);
  (void)b.predict(s2, false);
  CHECK(b.iris_frozen_ewald() == fresh(s2));
  // slab: a 0.5 A excursion of the lowest water stays within the 2 A margin (kept); 3.5 A outgrows it (refrozen)
  float cs[9];
  std::copy(cell, cell + 9, cs);
  cs[8] = 30.0f;
  const AtomicSystem t0 = sys(x, n, cs, true);
  (void)b.predict(t0, false);
  const std::vector<double> q0 = b.iris_frozen_ewald();
  CHECK(q0 == fresh(t0));
  std::vector<float> z(x);
  for (int k = 0; k < 3; k++) z[3 * k + 2] -= 0.5f; // (water 0 is the lowest: away from the slab)
  (void)b.predict(sys(z, n, cs, true), false);
  CHECK(b.iris_frozen_ewald() == q0);
  for (int k = 0; k < 3; k++) z[3 * k + 2] -= 3.0f;
  const AtomicSystem t2 = sys(z, n, cs, true);
  const auto r3a = a.predict(t2, true), r3b = b.predict(t2, true);
  CHECK(b.iris_frozen_ewald() == fresh(t2));
  CHECK(b.iris_frozen_ewald() != q0);
  CHECK(r3a.energy == r3b.energy);
}

TEST_CASE("pet_lr_port: iris models raise when an atom has more than max_neighbors neighbours (his calculator)",
          "[pet_lr_port]") {
  std::vector<float> x;
  std::vector<int32_t> Z;
  float cell[9];
  water_box(x, Z, cell); // 72 atoms in a 9 A cube: ~50 neighbours within 5 A, more than 4 selected
  const int n = int(Z.size());
  const bool pbc[3] = {true, true, true};
  PredictOptions po;
  po.compute_forces = true, po.charge = 0, po.spin_multiplicity = 1;
  for (const char *g : {"petlr_tiny_trunk.gguf", "petlr_tiny_fermi.gguf"}) {
    INFO(g);
    ModelOptions lim, off, wide;
    lim.iris_max_neighbors = 4, off.iris_max_neighbors = 0, wide.iris_max_neighbors = 500;
    Predictor ml(resource(g), lim), mo(resource(g), off), mw(resource(g), wide);
    CHECK_THROWS_WITH(ml.predict(n, x.data(), Z.data(), cell, pbc, po),
                      Catch::Matchers::ContainsSubstring("more than max_neighbors=4"));
    // (the count is of the SELECTED neighbours, as his forward's overflow: every pair within the cutoff for hard
    // cutoffs, the adaptive cutoff's pairs otherwise)
    const Result a = mo.predict(n, x.data(), Z.data(), cell, pbc, po), b = mw.predict(n, x.data(), Z.data(), cell, pbc, po);
    CHECK(a.energy == b.energy); // below the limit: the check changes nothing
  }
}
