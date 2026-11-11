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

#include <catch2/catch_test_macros.hpp>

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
          {"both (Gaussian charges, 2 shared passes)", resource("petlr_tiny_both.gguf")}};
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
  INFO("max |F_FD - F| = " << worst << " eV/A, max |F| = " << fmax);
  CHECK(worst < ftol + 1e-3 * fmax);
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
    // (FD floor of fp32 energies here ~1e-5 eV/A: the Gaussian-charge model's 1.4e-5 eV/A sits at it)
    iris_check(m, {0, 0, 0, 8.0f, 0.3f, -0.2f}, {11, 17}, 0, 2e-5, gapped ? -1.0 : 5e-6);
  }
}
