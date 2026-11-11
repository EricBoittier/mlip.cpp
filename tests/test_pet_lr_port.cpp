// PET-LR port (after M. Langer, ipsum/PET-LR): model-level checks of a converted iris trunk GGUF
// (scripts/pet_lr_port/iris_to_gguf.py). The GGUF comes from MLIPCPP_PETLR_GGUF (skipped when unset):
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
  const char *path = std::getenv("MLIPCPP_PETLR_GGUF");
  if (!path) SKIP("MLIPCPP_PETLR_GGUF not set");
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
  const char *path = std::getenv("MLIPCPP_PETLR_GGUF");
  if (!path) SKIP("MLIPCPP_PETLR_GGUF not set");
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
  // T = 0.02, scores near integers with a three-way tie at the top: gapped, weights 1/3 on the tie
  c.temperature = 0.02;
  const std::vector<double> g = {1.0, 1.0, 1.0, -1.0, 0.0, -1.0};
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

// forces = -dE/dx by central differences (h = 2e-3 A); charges sum to Q; the dipole is sum_i q_i (r_i - r0)
void iris_check(Predictor &m, const std::vector<float> &x, const std::vector<int32_t> &Z, int Q, double ftol) {
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
  for (int a = 0; a < 3; a++) CHECK(std::fabs(D[a] - r.dipole[a]) < 2e-4);
  const double h = 2e-3;
  double worst = 0, fmax = 0;
  for (size_t k = 0; k < x.size(); k++) {
    std::vector<float> xp = x, xm = x;
    xp[k] += float(h), xm[k] -= float(h);
    const double hh = double(xp[k]) - double(xm[k]);
    const double fd = -(iris_energy(m, xp, Z, Q, po) - iris_energy(m, xm, Z, Q, po)) / hh;
    worst = std::max(worst, std::fabs(fd - r.forces[k]));
    fmax = std::max(fmax, double(std::fabs(r.forces[k])));
  }
  INFO("max |F_FD - F| = " << worst << " eV/A, max |F| = " << fmax);
  CHECK(worst < ftol);
  CHECK(fmax > 0.05); // not trivially zero
}
} // namespace

TEST_CASE("pet_lr_port: iris PET-LR forces = -dE/dx (FD), charge sum, dipole; several (n, k)", "[pet_lr_port]") {
  const char *path = std::getenv("MLIPCPP_PETLR_LR_GGUF");
  if (!path) SKIP("MLIPCPP_PETLR_LR_GGUF not set");
  ModelOptions mo;
  Predictor m(path, mo);
  // H3O+ ... H2O (7 atoms, Q = +1)
  const std::vector<float> a = {0, 0, 0, 0.98f, 0.1f, 0, -0.35f, 0.92f, 0.05f, -0.33f, -0.5f, 0.8f,
                                2.55f, 0.2f, 0.1f, 3.1f, 0.95f, 0.2f, 3.05f, -0.55f, -0.1f};
  iris_check(m, a, {8, 1, 1, 1, 8, 1, 1}, 1, 5e-3);
  // water trimer (9 atoms, Q = 0)
  const std::vector<float> b = {0, 0, 0, 0.96f, 0, 0, -0.24f, 0.93f, 0, 2.8f, 0.3f, 0.2f, 3.3f, 1.1f, 0.3f,
                                3.2f, -0.4f, 0.7f, 1.2f, 2.6f, -0.3f, 0.5f, 3.2f, -0.5f, 1.9f, 3.1f, 0.2f};
  iris_check(m, b, {8, 1, 1, 8, 1, 1, 8, 1, 1}, 0, 5e-3);
  // Na+ and Cl- 7 A apart with a water each: two pieces, LR between them (Q = 0)
  const std::vector<float> c = {0, 0, 0, 2.3f, 0, 0, 2.9f, 0.75f, 0.1f, 2.85f, -0.7f, -0.1f,
                                7.0f, 0, 0, 10.1f, 0.2f, 0, 10.6f, 0.9f, 0.3f, 10.55f, -0.55f, -0.2f};
  iris_check(m, c, {11, 8, 1, 1, 17, 8, 1, 1}, 0, 5e-3);
}

TEST_CASE("pet_lr_port: iris PET-LR on atoms without neighbours keeps the Coulomb forces", "[pet_lr_port]") {
  const char *path = std::getenv("MLIPCPP_PETLR_LR_GGUF");
  if (!path) SKIP("MLIPCPP_PETLR_LR_GGUF not set");
  ModelOptions mo;
  Predictor m(path, mo);
  // Na+ and Cl- 8 A apart: no edge within the 5 A cutoff
  iris_check(m, {0, 0, 0, 8.0f, 0.3f, -0.2f}, {11, 17}, 0, 5e-3);
}
