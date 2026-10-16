#include "p3m.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <future>
#include <numbers>
#include <stdexcept>
#ifdef MLIPCPP_ACCELERATE_FFT
#include <Accelerate/Accelerate.h>
#include <map>
#endif

namespace mlipcpp {
namespace {
constexpr double pi = std::numbers::pi;
using Complex = std::complex<double>;
using Vec = std::array<double, 3>;
constexpr int va[] = {0, 1, 2, 1, 0, 0}, vb[] = {0, 1, 2, 2, 2, 1};
// Both transforms are unnormalised, as in torch-pme's backward/forward FFTs.
#ifndef MLIPCPP_ACCELERATE_FFT
void fft(std::vector<Complex> &v, bool inverse) {
  const size_t n = v.size();
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t b = n >> 1;
    for (; j & b; b >>= 1)
      j ^= b;
    j ^= b;
    if (i < j)
      std::swap(v[i], v[j]);
  }
  for (size_t len = 2; len <= n; len *= 2) {
    Complex root = std::polar(1.0, (inverse ? 2 : -2) * pi / len);
    for (size_t i = 0; i < n; i += len) {
      Complex w = 1;
      for (size_t j = 0; j < len / 2; ++j, w *= root) {
        auto a = v[i + j], b = v[i + j + len / 2] * w;
        v[i + j] = a + b;
        v[i + j + len / 2] = a - b;
      }
    }
  }
}
#endif
void fft3(std::vector<Complex> &v, const std::array<int, 3> &ns, bool inverse) {
#ifdef MLIPCPP_ACCELERATE_FFT
  struct Plans {
    std::map<int, FFTSetupD> plans;
    ~Plans() {
      for (auto [_, plan] : plans)
        vDSP_destroy_fftsetupD(plan);
    }
    FFTSetupD get(int logn) {
      auto &plan = plans[logn];
      if (!plan)
        plan = vDSP_create_fftsetupD(logn, kFFTRadix2);
      if (!plan)
        throw std::runtime_error("Could not allocate Accelerate FFT plan");
      return plan;
    }
  };
  thread_local Plans plans;
  size_t stride = 1;
  for (int axis = 2; axis >= 0; --axis) {
    const size_t n = ns[axis], block = n * stride;
    int logn = 0;
    for (size_t k = n; k > 1; k >>= 1)
      ++logn;
    if (n > 1) {
      auto plan = plans.get(logn);
      std::vector<double> real(n), imag(n);
      DSPDoubleSplitComplex split{real.data(), imag.data()};
      for (size_t base = 0; base < v.size(); base += block)
        for (size_t offset = 0; offset < stride; ++offset) {
          for (size_t i = 0; i < n; ++i) {
            auto x = v[base + offset + i * stride];
            real[i] = x.real();
            imag[i] = x.imag();
          }
          vDSP_fft_zipD(plan, &split, 1, logn,
                        inverse ? FFT_INVERSE : FFT_FORWARD);
          for (size_t i = 0; i < n; ++i)
            v[base + offset + i * stride] = {real[i], imag[i]};
        }
    }
    stride = block;
  }
#else
  size_t stride = 1;
  for (int axis = 2; axis >= 0; --axis) {
    size_t n = ns[axis], block = n * stride;
    std::vector<Complex> line(n);
    for (size_t base = 0; base < v.size(); base += block)
      for (size_t offset = 0; offset < stride; ++offset) {
        for (size_t i = 0; i < n; ++i)
          line[i] = v[base + offset + i * stride];
        fft(line, inverse);
        for (size_t i = 0; i < n; ++i)
          v[base + offset + i * stride] = line[i];
      }
    stride = block;
  }
#endif
}
// Centered cardinal B-spline, order five, and its analytic derivative.
std::array<double, 2> spline(double x) {
  constexpr double binomial[] = {1, 5, 10, 10, 5, 1};
  double w = 0, d = 0;
  for (int k = 0; k <= 5; ++k) {
    double t = std::max(0.0, x + 2.5 - k), b = (k % 2 ? -1 : 1) * binomial[k];
    w += b * t * t * t * t / 24;
    d += b * t * t * t / 6;
  }
  return {w, d};
}
struct Assignment {
  std::array<std::array<int, 5>, 3> index;
  std::array<std::array<double, 5>, 3> w, d;
};
} // namespace

P3MResult p3m(const AtomicSystem &s, const NeighborList &nl,
              const CoulombParameters &p, int channels,
              const std::vector<float> &q, const std::vector<float> *g,
              const std::vector<double> *direction, int threads) {
  const size_t n = s.num_atoms(), nc = size_t(channels);
  if (!n || channels <= 0 || q.size() != n * nc || (g && g->size() != q.size()))
    throw std::runtime_error("Invalid P3M charge/adjoint shape");
  if (direction && direction->size() != 3 * n)
    throw std::runtime_error("Invalid P3M position direction shape");
  if (direction)
    for (auto v : *direction)
      if (!std::isfinite(v))
        throw std::runtime_error("P3M requires finite direction");
  if (!(p.cutoff > 0 && p.smearing > 0 && p.wavelength > 0) ||
      !std::isfinite(p.cutoff + p.smearing + p.wavelength))
    throw std::runtime_error("Invalid P3M parameters");
  for (size_t i = 0; i < 3 * n; ++i)
    if (!std::isfinite(s.positions()[i]))
      throw std::runtime_error("P3M requires finite positions");
  int npbc = 0;
  if (s.cell())
    for (bool b : s.cell()->periodic)
      npbc += b;
  if (npbc != 0 && npbc != 3)
    throw std::runtime_error("P3M requires zero or three periodic axes");
  bool periodic = npbc == 3;
  double volume = 1, inv[3][3] = {};
  Vec lengths{};
  std::array<int, 3> ns{1, 1, 1};
  size_t mesh = 1;
  if (periodic) {
    const auto &c = s.cell()->matrix;
    double det = double(c[0][0]) *
                     (double(c[1][1]) * c[2][2] - double(c[1][2]) * c[2][1]) -
                 double(c[0][1]) *
                     (double(c[1][0]) * c[2][2] - double(c[1][2]) * c[2][0]) +
                 double(c[0][2]) *
                     (double(c[1][0]) * c[2][1] - double(c[1][1]) * c[2][0]);
    volume = std::abs(det);
    if (!std::isfinite(volume) || volume < 1e-12)
      throw std::runtime_error("Invalid P3M cell");
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        inv[a][b] =
            (double(c[(b + 1) % 3][(a + 1) % 3]) * c[(b + 2) % 3][(a + 2) % 3] -
             double(c[(b + 1) % 3][(a + 2) % 3]) *
                 c[(b + 2) % 3][(a + 1) % 3]) /
            det;
        lengths[a] += double(c[a][b]) * c[a][b];
      }
      lengths[a] = std::sqrt(lengths[a]);
      double target = 2 * lengths[a] / p.wavelength + 1;
      if (!std::isfinite(target) || target > 1024)
        throw std::runtime_error("P3M mesh axis exceeds 1024 points");
      while (ns[a] < target)
        ns[a] *= 2;
      mesh *= ns[a];
    }
    if (mesh > (size_t(1) << 23))
      throw std::runtime_error(
          "P3M mesh exceeds 8 million points; increase mesh spacing");
  }
  P3MResult out;
  std::vector<double> pot(q.size(), 0), dq(g ? q.size() : 0, 0);
  std::vector<double> tangent(direction ? q.size() : 0, 0);
  if (g) {
    out.forces.assign(3 * n, 0);
    if (periodic)
      out.stress.assign(6, 0);
  }
  auto pair = [&](size_t i, size_t j, const Vec &r) {
    double d = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (d < 1e-12)
      throw std::runtime_error("Coincident atoms in P3M");
    double f = 0, df = 0;
    if (d < p.cutoff) {
      f = .5 * (1 + std::cos(pi * d / p.cutoff));
      df = -.5 * pi / p.cutoff * std::sin(pi * d / p.cutoff);
    }
    double v, dv;
    if (periodic) {
      double a = 1 / (std::sqrt(2.0) * p.smearing), lr = std::erf(a * d) / d;
      double dlr = (2 * a / std::sqrt(pi) * std::exp(-a * a * d * d) - lr) / d;
      v = -.5 * lr * f;
      dv = -.5 * (dlr * f + lr * df);
    } else {
      v = .5 * (1 - f) / d;
      dv = -.5 * (df / d + (1 - f) / (d * d));
    }
    double dv_direction = 0;
    if (direction)
      for (int a = 0; a < 3; ++a)
        dv_direction +=
            dv * r[a] / d * ((*direction)[3 * j + a] - (*direction)[3 * i + a]);
    double adj = 0;
    for (size_t ch = 0; ch < nc; ++ch) {
      pot[i * nc + ch] += v * q[j * nc + ch];
      if (direction)
        tangent[i * nc + ch] += dv_direction * q[j * nc + ch];
      if (g) {
        dq[j * nc + ch] += v * (*g)[i * nc + ch];
        adj += (*g)[i * nc + ch] * q[j * nc + ch];
      }
    }
    if (g) {
      for (int a = 0; a < 3; ++a) {
        double f = adj * dv * r[a] / d;
        out.forces[3 * i + a] += f;
        out.forces[3 * j + a] -= f;
      }
      if (periodic)
        for (int a = 0; a < 6; ++a)
          out.stress[a] += adj * dv * r[va[a]] * r[vb[a]] / (d * volume);
    }
  };
  if (!periodic) {
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < n; ++j)
        if (i != j) {
          Vec r{};
          for (int a = 0; a < 3; ++a)
            r[a] = double(s.positions()[3 * j + a]) - s.positions()[3 * i + a];
          pair(i, j, r);
        }
  } else {
    const auto &cell = s.cell()->matrix;
    for (int e = 0; e < nl.num_pairs(); ++e) {
      size_t i = nl.centers[e], j = nl.neighbors[e];
      Vec r{};
      for (int a = 0; a < 3; ++a) {
        r[a] = double(s.positions()[3 * j + a]) - s.positions()[3 * i + a];
        for (int b = 0; b < 3; ++b)
          r[a] += double(nl.cell_shifts[e][b]) * cell[b][a];
      }
      pair(i, j, r);
    }
    std::vector<Assignment> assignments(n);
    for (size_t i = 0; i < n; ++i)
      for (int a = 0; a < 3; ++a) {
        double frac = 0;
        for (int b = 0; b < 3; ++b)
          frac += s.positions()[3 * i + b] * inv[b][a];
        double u = (frac - std::floor(frac)) * ns[a];
        int center = int(std::nearbyint(u));
        for (int t = 0; t < 5; ++t) {
          auto wd = spline(u - center - (t - 2));
          assignments[i].index[a][t] = (center + t - 2 + ns[a] * 2) % ns[a];
          assignments[i].w[a][t] = wd[0];
          assignments[i].d[a][t] = wd[1];
        }
      }
    auto visit = [&](size_t i, auto fn) {
      const auto &as = assignments[i];
      for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y)
          for (int z = 0; z < 5; ++z) {
            size_t idx =
                (size_t(as.index[0][x]) * ns[1] + as.index[1][y]) * ns[2] +
                as.index[2][z];
            double w = as.w[0][x] * as.w[1][y] * as.w[2][z];
            Vec dw{as.d[0][x] * as.w[1][y] * as.w[2][z],
                   as.w[0][x] * as.d[1][y] * as.w[2][z],
                   as.w[0][x] * as.w[1][y] * as.d[2][z]};
            fn(idx, w, dw);
          }
    };
    std::vector<double> kernel(mesh);
    std::vector<std::array<double, 6>> dk(g ? mesh : 0);
    for (int x = 0; x < ns[0]; ++x)
      for (int y = 0; y < ns[1]; ++y)
        for (int z = 0; z < ns[2]; ++z) {
          size_t idx = (size_t(x) * ns[1] + y) * ns[2] + z;
          // Reproduce rfftn/irfftn, including its Nyquist convention in skew
          // cells.
          int ix = x, iy = y, iz = z;
          if (z > ns[2] / 2) {
            ix = (ns[0] - x) % ns[0];
            iy = (ns[1] - y) % ns[1];
            iz = ns[2] - z;
          }
          int m[] = {ix < (ns[0] + 1) / 2 ? ix : ix - ns[0],
                     iy < (ns[1] + 1) / 2 ? iy : iy - ns[1], iz};
          Vec k{};
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              k[b] += 2 * pi * m[a] * inv[b][a];
          double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
          if (k2 == 0)
            continue;
          double influence = 1;
          Vec logder{};
          for (int a = 0; a < 3; ++a) {
            double u = k[a] * lengths[a] / ns[a] / 2;
            double sinc = std::abs(u) < 1e-8 ? 1 : std::sin(u) / u;
            influence /= std::pow(sinc, 10);
            logder[a] = std::abs(u) < 1e-5 ? 10 * (u / 3 + u * u * u / 45) / 2
                                           : -5 * (1 / std::tan(u) - 1 / u);
          }
          double v = 2 * pi * std::exp(-.5 * p.smearing * p.smearing * k2) *
                     influence / (volume * k2);
          if (!std::isfinite(v))
            throw std::runtime_error("Ill-conditioned P3M influence function");
          kernel[idx] = v;
          if (g)
            for (int t = 0; t < 6; ++t) {
              int a = va[t], b = vb[t];
              double deriv = (p.smearing * p.smearing + 2 / k2) * k[a] * k[b] -
                             (a == b ? 1 : 0);
              for (int d = 0; d < 3; ++d) {
                double dh = lengths[d] / ns[d] * cell[d][a] * cell[d][b] /
                            (lengths[d] * lengths[d]);
                // Symmetric shear strain gives the usual six-component stress.
                double dkh =
                    k[d] * dh -
                    (d == a ? k[b] * lengths[d] / ns[d] * (a == b ? 1 : .5)
                            : 0) -
                    (a != b && d == b ? k[a] * lengths[d] / ns[d] * .5 : 0);
                deriv += logder[d] * dkh;
              }
              dk[idx][t] = v * deriv;
            }
        }
    std::vector<Vec> velocity(n);
    if (direction)
      for (size_t i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            velocity[i][b] += (*direction)[3 * i + a] * ns[b] * inv[a][b];
    auto weight_tangent = [&](size_t i, const Vec &dw) {
      return dw[0] * velocity[i][0] + dw[1] * velocity[i][1] +
             dw[2] * velocity[i][2];
    };
    std::vector<std::vector<double>> channel_forces(g ? nc : 0),
        channel_stress(g ? nc : 0);
    if (g)
      for (size_t ch = 0; ch < nc; ++ch) {
        channel_forces[ch].resize(3 * n);
        channel_stress[ch].resize(6);
      }
    auto channel = [&](size_t ch) {
      std::vector<Complex> drho(direction ? mesh : 0);
      std::vector<Complex> rho(mesh), adj(g ? mesh : 0);
      double totalq = 0, totalg = 0;
      for (size_t i = 0; i < n; ++i) {
        totalq += q[i * nc + ch];
        if (g)
          totalg += (*g)[i * nc + ch];
        visit(i, [&](size_t idx, double w, const Vec &dw) {
          if (direction)
            drho[idx] += weight_tangent(i, dw) * q[i * nc + ch];
          rho[idx] += w * q[i * nc + ch];
          if (g)
            adj[idx] += w * (*g)[i * nc + ch];
        });
      }
      fft3(rho, ns, false);
      if (direction)
        fft3(drho, ns, false);
      if (g)
        fft3(adj, ns, false);
      for (size_t j = 0; j < mesh; ++j) {
        if (g) {
          double product = (std::conj(adj[j]) * rho[j]).real();
          for (int t = 0; t < 6; ++t)
            channel_stress[ch][t] += product * dk[j][t] / volume;
          adj[j] *= kernel[j];
        }
        rho[j] *= kernel[j];
        if (direction)
          drho[j] *= kernel[j];
      }
      fft3(rho, ns, true);
      if (direction)
        fft3(drho, ns, true);
      if (g)
        fft3(adj, ns, true);
      const double self = -.5 * std::sqrt(2 / pi) / p.smearing,
                   bg = -pi * p.smearing * p.smearing / volume;
      for (size_t i = 0; i < n; ++i) {
        pot[i * nc + ch] += self * q[i * nc + ch] + bg * totalq;
        if (g)
          dq[i * nc + ch] += self * (*g)[i * nc + ch] + bg * totalg;
        visit(i, [&](size_t idx, double w, const Vec &dw) {
          pot[i * nc + ch] += w * rho[idx].real();
          if (direction)
            tangent[i * nc + ch] +=
                weight_tangent(i, dw) * rho[idx].real() + w * drho[idx].real();
          if (g) {
            dq[i * nc + ch] += w * adj[idx].real();
            double scale = (*g)[i * nc + ch] * rho[idx].real() +
                           q[i * nc + ch] * adj[idx].real();
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b < 3; ++b)
                channel_forces[ch][3 * i + a] -=
                    scale * dw[b] * ns[b] * inv[a][b];
          }
        });
      }
      if (g)
        for (int t = 0; t < 3; ++t)
          channel_stress[ch][t] -= bg * totalq * totalg / volume;
    };
    const int workers = std::max(1, std::min(threads, channels));
    std::vector<std::future<void>> jobs;
    for (int worker = 1; worker < workers; ++worker)
      jobs.push_back(std::async(std::launch::async, [&, worker] {
        for (size_t ch = worker; ch < nc; ch += workers)
          channel(ch);
      }));
    for (size_t ch = 0; ch < nc; ch += workers)
      channel(ch);
    for (auto &job : jobs)
      job.get();
    if (g)
      for (size_t ch = 0; ch < nc; ++ch) {
        for (size_t i = 0; i < 3 * n; ++i)
          out.forces[i] += channel_forces[ch][i];
        for (int i = 0; i < 6; ++i)
          out.stress[i] += channel_stress[ch][i];
      }
  }
  out.potential_tangent.assign(tangent.begin(), tangent.end());
  out.potentials.assign(pot.begin(), pot.end());
  out.charge_gradient.assign(dq.begin(), dq.end());
  return out;
}
} // namespace mlipcpp
