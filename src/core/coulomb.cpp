#include "coulomb.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace mlipcpp {
namespace {
constexpr double pi = std::numbers::pi;
using Vec = std::array<double, 3>;
constexpr int va[6] = {0, 1, 2, 1, 0, 0};
constexpr int vb[6] = {0, 1, 2, 2, 2, 1};
double dot(const Vec &a, const Vec &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
} // namespace

CoulombResult coulomb_matrix(const AtomicSystem &system,
                             const NeighborList &neighbors,
                             const CoulombParameters &p,
                             const std::vector<float> *adjoint) {
  const int n = system.num_atoms();
  if (n < 1 || n > 512)
    throw std::runtime_error(
        "PET-LR dense reference solver supports 1..512 atoms");
  if (!std::isfinite(p.cutoff) || p.cutoff <= 0 || !std::isfinite(p.smearing) ||
      p.smearing <= 0 || !std::isfinite(p.wavelength) || p.wavelength <= 0)
    throw std::runtime_error("Invalid PET-LR Coulomb parameters");
  if (adjoint && adjoint->size() != size_t(n) * n)
    throw std::runtime_error("Invalid Coulomb matrix adjoint shape");
  for (int i = 0; i < 3 * n; ++i)
    if (!std::isfinite(system.positions()[i]))
      throw std::runtime_error("PET-LR requires finite positions");
  const Cell *cell = system.cell();
  int npbc = 0;
  if (cell)
    for (bool b : cell->periodic)
      npbc += b;
  if (npbc != 0 && npbc != 3)
    throw std::runtime_error(
        "PET-LR supports nonperiodic or fully periodic cells only");
  const bool periodic = npbc == 3;
  double volume = 1;
  double inv[3][3] = {};
  if (periodic) {
    // Compute in double rather than using the Cell's float inverse.
    const auto &c = cell->matrix;
    double det = double(c[0][0]) *
                     (double(c[1][1]) * c[2][2] - double(c[1][2]) * c[2][1]) -
                 double(c[0][1]) *
                     (double(c[1][0]) * c[2][2] - double(c[1][2]) * c[2][0]) +
                 double(c[0][2]) *
                     (double(c[1][0]) * c[2][1] - double(c[1][1]) * c[2][0]);
    volume = std::abs(det);
    if (!std::isfinite(volume) || volume < 1e-12)
      throw std::runtime_error("PET-LR requires a nonsingular periodic cell");
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        inv[a][b] =
            (double(c[(b + 1) % 3][(a + 1) % 3]) * c[(b + 2) % 3][(a + 2) % 3] -
             double(c[(b + 1) % 3][(a + 2) % 3]) *
                 c[(b + 2) % 3][(a + 1) % 3]) /
            det;
  }
  CoulombResult out;
  std::vector<double> matrix(size_t(n) * n, 0.0);
  if (adjoint) {
    out.forces.assign(3 * n, 0.0);
    if (periodic)
      out.stress.assign(6, 0.0);
  }
  auto displacement = [&](int i, int j) {
    Vec r{};
    for (int a = 0; a < 3; ++a)
      r[a] =
          double(system.positions()[3 * j + a]) - system.positions()[3 * i + a];
    return r;
  };
  // Every contribution includes its own derivative w.r.t. r_j-r_i and
  // homogeneous strain (positions and lattice deformed together).
  auto add = [&](int i, int j, double value, const Vec &dr,
                 const std::array<double, 6> &strain) {
    const size_t ij = size_t(i) * n + j;
    if (!adjoint)
      matrix[ij] += value;
    else {
      double g = (*adjoint)[ij];
      for (int a = 0; a < 3; ++a) {
        out.forces[3 * i + a] += g * dr[a];
        out.forces[3 * j + a] -= g * dr[a];
      }
      if (periodic)
        for (int a = 0; a < 6; ++a)
          out.stress[a] += g * strain[a] / volume;
    }
  };
  auto radial = [&](int i, int j, const Vec &r) {
    const double d = std::sqrt(dot(r, r));
    if (d < 1e-12)
      throw std::runtime_error("Coincident atoms in PET-LR Coulomb solver");
    double f = 0, df = 0;
    if (d < p.cutoff) {
      f = 0.5 * (1 + std::cos(pi * d / p.cutoff));
      df = -0.5 * pi / p.cutoff * std::sin(pi * d / p.cutoff);
    }
    double v, dv;
    if (periodic) {
      const double a = 1 / (std::sqrt(2.0) * p.smearing);
      const double lr = std::erf(a * d) / d;
      const double dlr =
          (2 * a / std::sqrt(pi) * std::exp(-a * a * d * d) - lr) / d;
      v = -0.5 * lr * f;
      dv = -0.5 * (dlr * f + lr * df);
    } else {
      v = 0.5 * (1 - f) / d;
      dv = -0.5 * (df / d + (1 - f) / (d * d));
    }
    Vec dr{};
    std::array<double, 6> strain{};
    for (int a = 0; a < 3; ++a)
      dr[a] = dv * r[a] / d;
    for (int a = 0; a < 6; ++a)
      strain[a] = dv * r[va[a]] * r[vb[a]] / d;
    add(i, j, v, dr, strain);
  };
  if (!periodic) {
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (i != j)
          radial(i, j, displacement(i, j));
  } else {
    for (int e = 0; e < neighbors.num_pairs(); ++e) {
      const int i = neighbors.centers[e], j = neighbors.neighbors[e];
      Vec r = displacement(i, j);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          r[a] += double(neighbors.cell_shifts[e][b]) * cell->matrix[b][a];
      radial(i, j, r);
    }
    int ns[3];
    int64_t nk = 1;
    for (int a = 0; a < 3; ++a) {
      double norm = 0;
      for (int b = 0; b < 3; ++b)
        norm += double(cell->matrix[a][b]) * cell->matrix[a][b];
      double count = std::ceil(std::sqrt(norm) / p.wavelength);
      if (count > 100000)
        throw std::runtime_error("PET-LR reciprocal grid too large");
      ns[a] = std::max(1, int(count));
      nk *= ns[a];
    }
    if (double(nk) * n * n > 1e8)
      throw std::runtime_error(
          "PET-LR dense Ewald work limit exceeded; use a smaller cell/system");
    for (int x = 0; x < ns[0]; ++x)
      for (int y = 0; y < ns[1]; ++y)
        for (int z = 0; z < ns[2]; ++z) {
          const int idx[3] = {x, y, z};
          Vec k{};
          for (int a = 0; a < 3; ++a) {
            int m = idx[a] < (ns[a] + 1) / 2 ? idx[a] : idx[a] - ns[a];
            for (int b = 0; b < 3; ++b)
              k[b] += 2 * pi * m * inv[b][a];
          }
          double k2 = dot(k, k);
          if (k2 == 0)
            continue;
          double weight = 2 * pi *
                          std::exp(-0.5 * p.smearing * p.smearing * k2) /
                          (volume * k2);
          for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
              double phase = dot(k, displacement(i, j));
              double v = weight * std::cos(phase);
              Vec dr{};
              std::array<double, 6> strain{};
              for (int a = 0; a < 3; ++a)
                dr[a] = -weight * std::sin(phase) * k[a];
              // k.r is invariant under homogeneous strain; only weight changes.
              for (int a = 0; a < 6; ++a)
                strain[a] = v * ((p.smearing * p.smearing + 2 / k2) * k[va[a]] *
                                     k[vb[a]] -
                                 (a < 3 ? 1 : 0));
              add(i, j, v, dr, strain);
            }
        }
    const double background = -pi * p.smearing * p.smearing / volume;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double v = background;
        if (i == j)
          v -= 0.5 * std::sqrt(2 / pi) / p.smearing;
        add(i, j, v, {}, {-background, -background, -background, 0, 0, 0});
      }
  }
  if (!adjoint)
    out.matrix.assign(matrix.begin(), matrix.end());
  return out;
}
} // namespace mlipcpp
