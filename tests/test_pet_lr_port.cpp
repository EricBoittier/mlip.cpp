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
