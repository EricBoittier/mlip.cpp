#include "sqe.h"

#include "core/gguf_loader.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mlipcpp::runtime {

namespace {

constexpr double KE = 14.399645; // eV A / e^2, as metatrain
constexpr double SQRT_PI = 1.7724538509055159;

// forward-mode dual number: value and one directional derivative
struct Dual {
  double v = 0, d = 0;
  Dual() = default;
  Dual(double a, double b = 0) : v(a), d(b) {}
};
inline Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
inline Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
inline Dual operator-(Dual a) { return {-a.v, -a.d}; }
inline Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
inline Dual operator/(Dual a, Dual b) { return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)}; }
inline Dual &operator+=(Dual &a, Dual b) { return a = a + b; }
inline Dual &operator-=(Dual &a, Dual b) { return a = a - b; }
inline Dual exp(Dual a) { const double e = std::exp(a.v); return {e, e * a.d}; }
inline Dual sqrt(Dual a) { const double s = std::sqrt(a.v); return {s, s > 0 ? 0.5 * a.d / s : 0.0}; }
inline Dual tanh(Dual a) { const double t = std::tanh(a.v); return {t, (1 - t * t) * a.d}; }
inline Dual erf(Dual a) { return {std::erf(a.v), 2.0 / SQRT_PI * std::exp(-a.v * a.v) * a.d}; }
inline double val(double a) { return a; }
inline double val(Dual a) { return a.v; }
using std::erf;
using std::exp;
using std::sqrt;
using std::tanh;

template <class T> T smootherstep(T x) {
  if (val(x) <= 0) return T(0.0);
  if (val(x) >= 1) return T(1.0);
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}
template <class T> T switch_off(T r, double on, double off) { return T(1.0) - smootherstep((r - on) / (off - on)); }
template <class T> T sigmoid(T z) { return T(1.0) / (T(1.0) + exp(-z)); }

struct Pair {
  int i, j;
};

// everything that depends on the positions (type T) for fixed parameters
template <class T> struct Geo {
  int n = 0;
  std::vector<int> e;   // element index per atom
  std::vector<Pair> pr; // split-charge pairs (r < r_ct, s > 0 in double)
  std::vector<T> r;     // dense n x n distances
  std::vector<T> cn, c, th, chi, J, gam, s, sq, lns_w; // gam dense n x n; per pair s, sqrt s, dln s / dw_raw
  // reference charges q0 (given, or learned: lo + d sigmoid(f + t)); learned: per atom w = d sigmoid', the
  // element-resolved coordination numbers cnp (n x E) and whether the total charge was feasible within the bounds
  std::vector<T> q0, qw, cnp;
  bool learned = false, feasible = true;
};

template <class T>
Geo<T> geometry(const SqeSpec &S, int n, const std::vector<int> &e, const std::vector<T> &x, const std::vector<Pair> &pr,
                const std::vector<double> &q0_given, double Q, bool local = false,
                const SqeClusterMask *mask = nullptr) {
  Geo<T> g;
  const size_t E = S.Z.size();
  g.learned = S.q0_learned;
  if (g.learned) g.cnp.assign(size_t(n) * E, T(0.0));
  g.n = n, g.e = e, g.pr = pr;
  g.r.assign(size_t(n) * n, T(0.0));
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      T d2 = T(0.0);
      for (int k = 0; k < 3; k++) {
        const T d = x[3 * i + k] - x[3 * j + k];
        d2 += d * d;
      }
      g.r[size_t(i) * n + j] = g.r[size_t(j) * n + i] = sqrt(d2);
    }
  g.cn.assign(n, T(0.0));
  g.gam.assign(size_t(n) * n, T(0.0));
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const T r = g.r[size_t(i) * n + j];
      const double R = S.rcov[e[i]] + S.rcov[e[j]];
      if (val(r) < S.cn_off) {
        const T w = sigmoid(T(16.0) * (T(4.0 / 3.0 * R) / r - 1.0)) * switch_off(r, S.cn_on, S.cn_off);
        g.cn[i] += w, g.cn[j] += w;
        if (g.learned) g.cnp[size_t(i) * E + e[j]] += w, g.cnp[size_t(j) * E + e[i]] += w;
      }
      const double sij = std::sqrt(S.sigma[e[i]] * S.sigma[e[i]] + S.sigma[e[j]] * S.sigma[e[j]]);
      T gm = T(KE) * erf(r / sij) / r;
      if (local) { // gamma (1 - T(r)), or with a cluster mask gamma (1 - T(r) (1 - R))
        const double mr = mask ? 1.0 - mask->R[size_t(mask->sn[i]) * mask->s + mask->sn[j]] : 1.0;
        gm = gm * (T(1.0) - smootherstep((r - (S.lr_R - S.lr_t)) / S.lr_t) * mr);
      }
      g.gam[size_t(i) * n + j] = g.gam[size_t(j) * n + i] = gm;
    }
  g.c.resize(n), g.th.resize(n), g.chi.resize(n), g.J.resize(n);
  for (int i = 0; i < n; i++) {
    g.c[i] = sqrt(g.cn[i] + 1.0) - 1.0;
    g.th[i] = tanh(T(S.chi_a[e[i]]) + T(S.chi_b[e[i]]) * g.c[i]);
    g.chi[i] = T(S.chi0[e[i]]) + T(S.dchi_max) * g.th[i];
    g.J[i] = T(S.J(e[i]));
  }
  // reference charges
  g.q0.resize(n);
  if (!g.learned) {
    for (int i = 0; i < n; i++) g.q0[i] = T(q0_given[i]);
  } else {
    std::vector<T> f(n);
    double slo = 0, shi = 0;
    for (int i = 0; i < n; i++) {
      T a = T(S.q0_a[e[i]]);
      for (size_t k = 0; k < E; k++) a += T(S.q0_b[size_t(e[i]) * E + k]) * g.cnp[size_t(i) * E + k];
      f[i] = a, slo += S.q0_lo[e[i]], shi += S.q0_hi[e[i]];
    }
    auto sum_at = [&](double t) {
      double v = 0;
      for (int i = 0; i < n; i++)
        v += S.q0_lo[e[i]] + (S.q0_hi[e[i]] - S.q0_lo[e[i]]) / (1.0 + std::exp(-(val(f[i]) + t)));
      return v;
    };
    g.feasible = slo < Q && Q < shi;
    T t;
    if (g.feasible) { // bisection to the root, then Newton steps in T (exact derivatives of t(f))
      double a = -60, b = 60;
      for (int it = 0; it < 200; it++) {
        const double m = 0.5 * (a + b);
        (sum_at(m) < Q ? a : b) = m;
      }
      t = T(0.5 * (a + b));
      for (int it = 0; it < 3; it++) {
        T S0 = T(0.0), S1 = T(0.0);
        for (int i = 0; i < n; i++) {
          const double d = S.q0_hi[e[i]] - S.q0_lo[e[i]];
          const T sg = sigmoid(f[i] + t);
          S0 += T(S.q0_lo[e[i]]) + T(d) * sg, S1 += T(d) * sg * (T(1.0) - sg);
        }
        if (val(S1) > 0) t = t - (S0 - Q) / S1;
      }
    } else {
      t = T(Q >= shi ? 60.0 : -60.0);
    }
    g.qw.resize(n);
    T tot = T(0.0);
    for (int i = 0; i < n; i++) {
      const double d = S.q0_hi[e[i]] - S.q0_lo[e[i]];
      const T sg = sigmoid(f[i] + t);
      g.q0[i] = T(S.q0_lo[e[i]]) + T(d) * sg, g.qw[i] = T(d) * sg * (T(1.0) - sg), tot += g.q0[i];
    }
    if (!g.feasible) // the remainder spread uniformly (bounds broken)
      for (int i = 0; i < n; i++) g.q0[i] += (T(Q) - tot) / double(n);
  }
  const double w = S.w();
  const double sw = 1.0 / (1.0 + std::exp(-S.w_raw)), dw = (S.w_hi - S.w_lo) * sw * (1 - sw);
  g.s.resize(pr.size()), g.sq.resize(pr.size()), g.lns_w.resize(pr.size());
  for (size_t k = 0; k < pr.size(); k++) {
    const int i = pr[k].i, j = pr[k].j;
    const T r = g.r[size_t(i) * n + j];
    const double r0 = S.rcov[e[i]] + S.rcov[e[j]], k0 = std::sqrt(S.kappa0(e[i]) * S.kappa0(e[j]));
    g.s[k] = exp(-(r - r0) / w) * switch_off(r, S.r_ct - S.r_ct_width, S.r_ct) / k0;
    g.sq[k] = sqrt(g.s[k]);
    g.lns_w[k] = (r - r0) / (w * w) * dw;
  }
  return g;
}

// A v (A = diag(J) + gamma)
template <class T> std::vector<T> Amul(const Geo<T> &g, const std::vector<T> &v) {
  const int n = g.n;
  std::vector<T> out(n);
  for (int i = 0; i < n; i++) {
    T a = g.J[i] * v[i];
    for (int j = 0; j < n; j++)
      if (j != i) a += g.gam[size_t(i) * n + j] * v[j];
    out[i] = a;
  }
  return out;
}

template <class T> std::vector<T> charges(const Geo<T> &g, const std::vector<T> &u) {
  std::vector<T> q = g.q0;
  for (size_t k = 0; k < g.pr.size(); k++) {
    const T p = g.sq[k] * u[k];
    q[g.pr[k].i] += p, q[g.pr[k].j] -= p;
  }
  return q;
}

// dE/du = B^T mu + u, mu = chi + A q
template <class T> std::vector<T> grad_u(const Geo<T> &g, const std::vector<T> &u) {
  const std::vector<T> q = charges(g, u), Aq = Amul(g, q);
  std::vector<T> out(u.size());
  for (size_t k = 0; k < g.pr.size(); k++) {
    const int i = g.pr[k].i, j = g.pr[k].j;
    out[k] = g.sq[k] * ((g.chi[i] + Aq[i]) - (g.chi[j] + Aq[j])) + u[k];
  }
  return out;
}

// H v = B^T A B v + v (double)
std::vector<double> Hmul(const Geo<double> &g, const std::vector<double> &v) {
  std::vector<double> Bv(g.n, 0.0);
  for (size_t k = 0; k < g.pr.size(); k++) Bv[g.pr[k].i] += g.sq[k] * v[k], Bv[g.pr[k].j] -= g.sq[k] * v[k];
  const std::vector<double> ABv = Amul(g, Bv);
  std::vector<double> out(v.size());
  for (size_t k = 0; k < g.pr.size(); k++) out[k] = g.sq[k] * (ABv[g.pr[k].i] - ABv[g.pr[k].j]) + v[k];
  return out;
}

// Jacobi-preconditioned CG for H u = b
std::vector<double> cg(const Geo<double> &g, const std::vector<double> &b, double tol, int &iters) {
  const size_t P = b.size();
  std::vector<double> u(P, 0.0), r = b, z(P), p(P), dg(P);
  for (size_t k = 0; k < P; k++) {
    const int i = g.pr[k].i, j = g.pr[k].j;
    dg[k] = 1.0 + g.s[k] * (g.J[i] + g.J[j] - 2.0 * g.gam[size_t(i) * g.n + j]);
    if (!(dg[k] > 0)) dg[k] = 1.0;
  }
  double bn = 0;
  for (double v : b) bn += v * v;
  bn = std::sqrt(bn);
  iters = 0;
  if (bn == 0) return u;
  for (size_t k = 0; k < P; k++) z[k] = r[k] / dg[k], p[k] = z[k];
  double rz = 0;
  for (size_t k = 0; k < P; k++) rz += r[k] * z[k];
  for (iters = 1; iters <= 10000; iters++) {
    const std::vector<double> Hp = Hmul(g, p);
    double pHp = 0;
    for (size_t k = 0; k < P; k++) pHp += p[k] * Hp[k];
    const double a = rz / pHp;
    double rn = 0;
    for (size_t k = 0; k < P; k++) u[k] += a * p[k], r[k] -= a * Hp[k], rn += r[k] * r[k];
    if (std::sqrt(rn) <= tol * bn) break;
    double rz2 = 0;
    for (size_t k = 0; k < P; k++) z[k] = r[k] / dg[k], rz2 += r[k] * z[k];
    const double beta = rz2 / rz;
    rz = rz2;
    for (size_t k = 0; k < P; k++) p[k] = z[k] + beta * p[k];
  }
  return u;
}

// dE/dtheta at fixed u (flat order of SqeSpec::flat)
// learned q0: sum_i c_i dq0_i/dtheta into G (q0_a at 5E + 1, q0_b after it), for per-atom weights c (dE/dq0 = mu,
// or a supervision loss's dL/dq0): dq0_i = w_i (df_i + dt), dt = -sum_j w_j df_j / sum w (feasible), so the weight on
// df_i is w_i (c_i - m), m = sum w c / sum w (infeasible: m = mean c)
template <class T> void q0_backprop(const SqeSpec &S, const Geo<T> &g, const std::vector<T> &c, std::vector<T> &G) {
  if (!g.learned) return;
  const size_t E = S.Z.size(), o = 4 * E + 1;
  T sw = T(0.0), swc = T(0.0), sc = T(0.0);
  for (int i = 0; i < g.n; i++) sw += g.qw[i], swc += g.qw[i] * c[i], sc += c[i];
  const T m = g.feasible ? swc / sw : sc / double(g.n);
  for (int i = 0; i < g.n; i++) {
    const T k = g.qw[i] * (c[i] - m);
    const size_t ei = g.e[i];
    G[o + ei] += k;
    for (size_t z = 0; z < E; z++) G[o + E + ei * E + z] += k * g.cnp[size_t(i) * E + z];
  }
}

template <class T> std::vector<T> grad_theta(const SqeSpec &S, const Geo<T> &g, const std::vector<T> &u) {
  const size_t E = S.Z.size();
  std::vector<T> G(S.n_params(), T(0.0));
  const std::vector<T> q = charges(g, u), Aq = Amul(g, q);
  for (int i = 0; i < g.n; i++) {
    const int e = g.e[i];
    const T dchi = T(S.dchi_max) * (T(1.0) - g.th[i] * g.th[i]);
    G[e] += q[i] * dchi;
    G[E + e] += q[i] * dchi * g.c[i];
    const double tz = std::tanh(S.j_z[e]);
    const double lJ = (std::log(S.J_hi) + std::log(S.J_lo)) / 2, hJ = (std::log(S.J_hi) - std::log(S.J_lo)) / 2;
    (void)lJ;
    G[2 * E + e] += T(0.5) * q[i] * q[i] * g.J[i] * (hJ * (1 - tz * tz));
  }
  const double lo = std::log(S.kappa_lo), hi = std::log(S.kappa_hi);
  for (size_t k = 0; k < g.pr.size(); k++) {
    const int i = g.pr[k].i, j = g.pr[k].j;
    // u_k dmu_k dsqrt(s)/dtheta = 1/2 sqrt(s) u_k dmu_k dln s/dtheta
    const T t = T(0.5) * g.sq[k] * u[k] * ((g.chi[i] + Aq[i]) - (g.chi[j] + Aq[j]));
    for (int a : {g.e[i], g.e[j]}) {
      const double sg = 1.0 / (1.0 + std::exp(-S.kappa_z[a]));
      G[3 * E + a] += t * (-0.5 * (hi - lo) * sg * (1 - sg));
    }
    G[4 * E] += t * g.lns_w[k];
  }
  if (g.learned) {
    std::vector<T> mu(g.n);
    for (int i = 0; i < g.n; i++) mu[i] = g.chi[i] + Aq[i];
    q0_backprop(S, g, mu, G);
  }
  return G;
}

struct Prepared {
  std::vector<int> e;
  std::vector<Pair> pr;
  std::vector<double> q0;
};

Prepared prepare(const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q, const double *q0) {
  Prepared p;
  p.e.resize(n);
  for (int i = 0; i < n; i++) {
    p.e[i] = S.index_of(Z[i]);
    if (p.e[i] < 0) throw std::runtime_error("sqe: no parameters for atomic number " + std::to_string(Z[i]));
  }
  p.q0.assign(n, n ? Q / n : 0.0);
  if (q0) p.q0.assign(q0, q0 + n);
  const double w = S.w();
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      double d2 = 0;
      for (int k = 0; k < 3; k++) d2 += (x[3 * i + k] - x[3 * j + k]) * (x[3 * i + k] - x[3 * j + k]);
      const double r = std::sqrt(d2);
      if (r >= S.r_ct) continue;
      // negligible softness (deep in the switch): s < 1e-30 eV^-1 transfers nothing
      const double r0 = S.rcov[p.e[i]] + S.rcov[p.e[j]], k0 = std::sqrt(S.kappa0(p.e[i]) * S.kappa0(p.e[j]));
      if (std::exp(-(r - r0) / w) * switch_off(r, S.r_ct - S.r_ct_width, S.r_ct) / k0 < 1e-30) continue;
      p.pr.push_back({i, j});
    }
  return p;
}

} // namespace

int SqeSpec::index_of(int z) const {
  for (size_t k = 0; k < Z.size(); k++)
    if (Z[k] == z) return static_cast<int>(k);
  return -1;
}
double SqeSpec::J(int e) const {
  const double lJ = (std::log(J_hi) + std::log(J_lo)) / 2, hJ = (std::log(J_hi) - std::log(J_lo)) / 2;
  return J0[e] * std::exp(lJ + hJ * std::tanh(j_z[e]));
}
double SqeSpec::kappa0(int e) const {
  const double lo = std::log(kappa_lo), hi = std::log(kappa_hi);
  return std::exp(lo + (hi - lo) / (1.0 + std::exp(-kappa_z[e])));
}
double SqeSpec::w() const { return w_lo + (w_hi - w_lo) / (1.0 + std::exp(-w_raw)); }

std::string SqeSpec::summary() const {
  double ta[2] = {1e9, -1e9}, bb[2] = {1e9, -1e9}, jf[2] = {1e9, -1e9}, k0[2] = {1e9, -1e9};
  for (size_t e = 0; e < Z.size(); e++) {
    auto mm = [](double *r, double v) { r[0] = std::min(r[0], v), r[1] = std::max(r[1], v); };
    mm(ta, std::tanh(chi_a[e])), mm(bb, chi_b[e]), mm(jf, J(int(e)) / J0[e]), mm(k0, kappa0(int(e)));
  }
  char buf[320];
  std::snprintf(buf, sizeof buf, "dchi/dchi_max(CN0) [%.3f, %.3f] (bound +-1), b [%.3f, %.3f], J/J0 [%.3f, %.3f] (%.2f, %.2f), "
                "kappa0 [%.2f, %.2f] eV (%.2g, %.2g), w %.3f A (%.2g, %.2g)", ta[0], ta[1], bb[0], bb[1], jf[0], jf[1], J_lo,
                J_hi, k0[0], k0[1], kappa_lo, kappa_hi, w(), w_lo, w_hi);
  return buf;
}

std::vector<double> SqeSpec::flat() const {
  std::vector<double> v;
  for (auto *a : {&chi_a, &chi_b, &j_z, &kappa_z}) v.insert(v.end(), a->begin(), a->end());
  v.push_back(w_raw);
  v.insert(v.end(), q0_a.begin(), q0_a.end());
  v.insert(v.end(), q0_b.begin(), q0_b.end());
  return v;
}
void SqeSpec::set_flat(const std::vector<double> &v) {
  if (v.size() != n_params()) throw std::runtime_error("sqe: parameter count");
  const size_t E = Z.size();
  size_t o = 0;
  for (auto *a : {&chi_a, &chi_b, &j_z, &kappa_z}) a->assign(v.begin() + o, v.begin() + o + E), o += E;
  w_raw = v[o++];
  q0_a.assign(v.begin() + o, v.begin() + o + E), o += E;
  q0_b.assign(v.begin() + o, v.begin() + o + E * E);
}
std::vector<std::string> SqeSpec::flat_names() const {
  std::vector<std::string> out;
  for (const char *nm : {"chi_a", "chi_b", "j_z", "kappa_z"})
    for (int z : Z) out.push_back(std::string("sqe.") + nm + "." + std::to_string(z));
  out.push_back("sqe.w_raw");
  for (int z : Z) out.push_back("sqe.q0_a." + std::to_string(z));
  for (int z : Z)
    for (int z2 : Z) out.push_back("sqe.q0_b." + std::to_string(z) + "." + std::to_string(z2));
  return out;
}

namespace {
std::vector<double> get_f64(const GGUFLoader &L, const std::string &key) {
  gguf_context *g = L.gguf_ctx();
  const int64_t id = gguf_find_key(g, key.c_str());
  if (id < 0) return {};
  if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, id) == GGUF_TYPE_FLOAT64) {
    const double *d = static_cast<const double *>(gguf_get_arr_data(g, id));
    return std::vector<double>(d, d + gguf_get_arr_n(g, id));
  }
  if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT64) return {gguf_get_val_f64(g, id)};
  throw std::runtime_error("sqe: " + key + " must be float64");
}
} // namespace

SqeSpec SqeSpec::from_gguf(const GGUFLoader &L) {
  SqeSpec s;
  s.enabled = L.get_int32("pet.plato_lr.sqe", 0) != 0;
  if (!s.enabled) return s;
  s.q0_formal = L.get_string("pet.plato_lr.sqe_q0", "uniform") == "formal";
  s.q0_learned = L.get_string("pet.plato_lr.sqe_q0", "uniform") == "learned";
  s.backbone_neutral = L.get_string("pet.plato_lr.sqe_backbone_charge", "input") == "neutral";
  auto scalar = [&](const char *k, double &v) {
    const auto a = get_f64(L, std::string("pet.plato_lr.sqe_") + k);
    if (!a.empty()) v = a[0];
  };
  scalar("r_ct", s.r_ct), scalar("r_ct_width", s.r_ct_width), scalar("dchi_max", s.dchi_max);
  scalar("J_lo", s.J_lo), scalar("J_hi", s.J_hi), scalar("kappa_lo", s.kappa_lo), scalar("kappa_hi", s.kappa_hi);
  scalar("w_lo", s.w_lo), scalar("w_hi", s.w_hi), scalar("cn_on", s.cn_on), scalar("cn_off", s.cn_off);
  scalar("ewald_alpha", s.ewald_alpha), scalar("w_raw", s.w_raw), scalar("lr_R", s.lr_R), scalar("lr_t", s.lr_t);
  s.Z = L.get_array_int32("pet.plato_lr.sqe_Z");
  s.chi0 = get_f64(L, "pet.plato_lr.sqe_chi0"), s.J0 = get_f64(L, "pet.plato_lr.sqe_J0");
  s.rcov = get_f64(L, "pet.plato_lr.sqe_rcov"), s.sigma = get_f64(L, "pet.plato_lr.sqe_sigma");
  s.chi_a = get_f64(L, "pet.plato_lr.sqe_chi_a"), s.chi_b = get_f64(L, "pet.plato_lr.sqe_chi_b");
  s.j_z = get_f64(L, "pet.plato_lr.sqe_j_z"), s.kappa_z = get_f64(L, "pet.plato_lr.sqe_kappa_z");
  const size_t E = s.Z.size();
  for (auto *a : {&s.chi0, &s.J0, &s.rcov, &s.sigma, &s.chi_a, &s.chi_b, &s.j_z, &s.kappa_z})
    if (a->size() != E || E == 0) throw std::runtime_error("sqe: per-element tables missing or of different lengths");
  // learned reference charges (optional in older GGUFs: zeros, bounds [0, 0])
  s.q0_lo = get_f64(L, "pet.plato_lr.sqe_q0_lo"), s.q0_hi = get_f64(L, "pet.plato_lr.sqe_q0_hi");
  s.q0_a = get_f64(L, "pet.plato_lr.sqe_q0_a"), s.q0_b = get_f64(L, "pet.plato_lr.sqe_q0_b");
  if (s.q0_lo.empty()) s.q0_lo.assign(E, 0.0), s.q0_hi.assign(E, 0.0), s.q0_a.assign(E, 0.0), s.q0_b.assign(E * E, 0.0);
  if (s.q0_lo.size() != E || s.q0_hi.size() != E || s.q0_a.size() != E || s.q0_b.size() != E * E)
    throw std::runtime_error("sqe: q0 tables of the wrong length");
  if (s.q0_learned && s.q0_formal) throw std::logic_error("sqe: q0 mode");
  return s;
}

void SqeSpec::to_gguf(gguf_context *g) const {
  if (!enabled) return;
  gguf_set_val_i32(g, "pet.plato_lr.sqe", 1);
  gguf_set_val_str(g, "pet.plato_lr.sqe_q0", q0_learned ? "learned" : q0_formal ? "formal" : "uniform");
  gguf_set_val_str(g, "pet.plato_lr.sqe_backbone_charge", backbone_neutral ? "neutral" : "input");
  auto scalar = [&](const char *k, double v) { gguf_set_val_f64(g, (std::string("pet.plato_lr.sqe_") + k).c_str(), v); };
  scalar("r_ct", r_ct), scalar("r_ct_width", r_ct_width), scalar("dchi_max", dchi_max), scalar("J_lo", J_lo);
  scalar("J_hi", J_hi), scalar("kappa_lo", kappa_lo), scalar("kappa_hi", kappa_hi), scalar("w_lo", w_lo);
  scalar("w_hi", w_hi), scalar("cn_on", cn_on), scalar("cn_off", cn_off), scalar("ewald_alpha", ewald_alpha);
  scalar("w_raw", w_raw), scalar("lr_R", lr_R), scalar("lr_t", lr_t);
  gguf_set_arr_data(g, "pet.plato_lr.sqe_Z", GGUF_TYPE_INT32, Z.data(), Z.size());
  auto arr = [&](const char *k, const std::vector<double> &v) {
    gguf_set_arr_data(g, (std::string("pet.plato_lr.sqe_") + k).c_str(), GGUF_TYPE_FLOAT64, v.data(), v.size());
  };
  arr("chi0", chi0), arr("J0", J0), arr("rcov", rcov), arr("sigma", sigma);
  arr("chi_a", chi_a), arr("chi_b", chi_b), arr("j_z", j_z), arr("kappa_z", kappa_z);
  arr("q0_lo", q0_lo), arr("q0_hi", q0_hi), arr("q0_a", q0_a), arr("q0_b", q0_b);
}

SqeOutput sqe_compute(const SqeSpec &S, int n, const int32_t *Z, const float *x, double Q, const double *q0,
                      bool forces, const double *cell) {
  std::vector<double> xd(x, x + 3 * size_t(n));
  return sqe_compute(S, n, Z, xd.data(), Q, q0, forces, cell);
}

namespace {
SqeOutput compute_one(const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                      bool forces, bool local, const SqeClusterMask *mask = nullptr) {
  SqeOutput out;
  if (n == 0) return out;
  const Prepared p = prepare(S, n, Z, x, Q, q0);
  const std::vector<double> xv(x, x + 3 * size_t(n));
  const Geo<double> g = geometry(S, n, p.e, xv, p.pr, p.q0, Q, local, local ? mask : nullptr);
  std::vector<double> b = grad_u(g, std::vector<double>(p.pr.size(), 0.0));
  for (double &v : b) v = -v;
  const std::vector<double> u = cg(g, b, S.cg_tol, out.cg_iters);
  out.q = charges(g, u);
  out.q0 = g.q0;
  const std::vector<double> Aq = Amul(g, out.q);
  std::vector<double> mu(n);
  for (int i = 0; i < n; i++) mu[i] = g.chi[i] + Aq[i], out.energy += g.chi[i] * out.q[i] + 0.5 * out.q[i] * Aq[i];
  for (double v : u) out.energy += 0.5 * v * v;
  out.chi = g.chi, out.J = g.J;
  out.dE_dq0 = mu;
  if (mask && local) out.dE_dR.assign(size_t(mask->s) * mask->s, 0.0);
  if (!forces) return out;
  // learned q0: dE/dq0 = mu, so the weight on df_i (f the q0 logits) is w_i (mu_i - m), as in q0_backprop
  std::vector<double> gq(n, 0.0);
  if (g.learned) {
    double sw = 0, swm = 0, sm = 0;
    for (int i = 0; i < n; i++) sw += g.qw[i], swm += g.qw[i] * mu[i], sm += mu[i];
    const double m = g.feasible ? swm / sw : sm / n;
    for (int i = 0; i < n; i++) gq[i] = g.qw[i] * (mu[i] - m);
  }
  const size_t E = S.Z.size();
  // Hellmann-Feynman: dE/dx at fixed u. dE/dr_ij from gamma (all pairs), CN (chi), and the softness on split pairs
  out.forces.assign(3 * size_t(n), 0.0);
  std::vector<double> dEdcn(n);
  for (int i = 0; i < n; i++) {
    const int e = p.e[i];
    dEdcn[i] = out.q[i] * S.dchi_max * (1 - g.th[i] * g.th[i]) * S.chi_b[e] * 0.5 / std::sqrt(g.cn[i] + 1.0);
  }
  auto add = [&](int i, int j, double dEdr) {
    const double r = g.r[size_t(i) * n + j];
    for (int k = 0; k < 3; k++) {
      const double f = dEdr * (x[3 * i + k] - x[3 * j + k]) / r;
      out.forces[3 * i + k] -= f, out.forces[3 * j + k] += f;
    }
  };
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const double r = g.r[size_t(i) * n + j];
      const int ei = p.e[i], ej = p.e[j];
      const double sij = std::sqrt(S.sigma[ei] * S.sigma[ei] + S.sigma[ej] * S.sigma[ej]);
      double dEdr = out.q[i] * out.q[j] * KE *
                    (2.0 / SQRT_PI / sij * std::exp(-r * r / (sij * sij)) / r - std::erf(r / sij) / (r * r));
      if (local) { // gamma (1 - T): gamma' (1 - T) - gamma T'
        const double t = (r - (S.lr_R - S.lr_t)) / S.lr_t;
        double Tv = 0, dT = 0;
        if (t >= 1) Tv = 1;
        else if (t > 0) Tv = t * t * t * (t * (6 * t - 15) + 10), dT = 30 * t * t * (t - 1) * (t - 1) / S.lr_t;
        const double mr = mask ? 1.0 - mask->R[size_t(mask->sn[i]) * mask->s + mask->sn[j]] : 1.0;
        const double gq = out.q[i] * out.q[j] * KE * std::erf(r / sij) / r;
        dEdr = dEdr * (1 - Tv * mr) - gq * dT * mr;
        if (mask && mask->sn[i] != mask->sn[j] && Tv > 0) { // d/dR of gamma (1 - T (1 - R)) = gamma T
          const size_t a = mask->sn[i], b2 = mask->sn[j];
          out.dE_dR[a * mask->s + b2] += 0.5 * gq * Tv, out.dE_dR[b2 * mask->s + a] += 0.5 * gq * Tv;
        }
      }
      if (r < S.cn_off) { // d(cnt * sw)/dr
        const double R = S.rcov[ei] + S.rcov[ej];
        const double z = 16.0 * (4.0 / 3.0 * R / r - 1.0), sg = 1.0 / (1.0 + std::exp(-z));
        const double dsg = sg * (1 - sg) * 16.0 * (-4.0 / 3.0 * R / (r * r));
        const double t = (r - S.cn_on) / (S.cn_off - S.cn_on);
        double sw = 1.0, dsw = 0.0;
        if (t > 0 && t < 1) {
          sw = 1.0 - t * t * t * (t * (6 * t - 15) + 10);
          dsw = -30.0 * t * t * (t - 1) * (t - 1) / (S.cn_off - S.cn_on);
        } else if (t >= 1) {
          sw = 0.0;
        }
        double dw = dEdcn[i] + dEdcn[j];
        if (g.learned) dw += gq[i] * S.q0_b[size_t(ei) * E + ej] + gq[j] * S.q0_b[size_t(ej) * E + ei];
        dEdr += dw * (dsg * sw + sg * dsw);
      }
      add(i, j, dEdr);
    }
  const double w = S.w();
  for (size_t k = 0; k < p.pr.size(); k++) {
    const int i = p.pr[k].i, j = p.pr[k].j;
    const double r = g.r[size_t(i) * n + j], dmu = mu[i] - mu[j];
    // dE/ds = -dmu^2 / 2; s = exp(-(r - r0)/w) fc / k0, ds/dr = exp(..)/k0 (fc' - fc / w)
    const double r0 = S.rcov[p.e[i]] + S.rcov[p.e[j]], k0 = std::sqrt(S.kappa0(p.e[i]) * S.kappa0(p.e[j]));
    const double on = S.r_ct - S.r_ct_width, t = (r - on) / S.r_ct_width;
    double fc = 1.0, dfc = 0.0;
    if (t > 0 && t < 1) {
      fc = 1.0 - t * t * t * (t * (6 * t - 15) + 10);
      dfc = -30.0 * t * t * (t - 1) * (t - 1) / S.r_ct_width;
    } else if (t >= 1) {
      fc = 0.0;
    }
    const double ds = std::exp(-(r - r0) / w) / k0 * (dfc - fc / w);
    add(i, j, -0.5 * dmu * dmu * ds);
  }
  return out;
}

} // namespace

// lr_R > 0: the long-range part E*(gamma) - E*(gamma (1 - T)) (two minima; charges from the full one)
SqeOutput sqe_compute(const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                      bool forces, const double *cell) {
  if (cell) throw std::runtime_error("sqe: periodic structures are not implemented in mlip.cpp (metatrain only)");
  SqeOutput out = compute_one(S, n, Z, x, Q, q0, forces, false);
  if (S.lr_R > 0 && n > 0) {
    const SqeOutput loc = compute_one(S, n, Z, x, Q, q0, forces, true);
    out.energy -= loc.energy;
    for (size_t k = 0; k < out.forces.size(); k++) out.forces[k] -= loc.forces[k];
  }
  return out;
}

SqeOutput sqe_compute_masked(const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                             bool forces, const SqeClusterMask &mask) {
  if (!(S.lr_R > 0)) throw std::runtime_error("sqe_compute_masked: needs lr_R > 0");
  if (S.q0_learned) throw std::runtime_error("sqe_compute_masked: q0 must be given");
  SqeOutput out = compute_one(S, n, Z, x, Q, q0, forces, false);
  if (n == 0) return out;
  const SqeOutput loc = compute_one(S, n, Z, x, Q, q0, forces, true, &mask);
  out.energy -= loc.energy;
  for (size_t k = 0; k < out.forces.size(); k++) out.forces[k] -= loc.forces[k];
  for (int i = 0; i < n; i++) out.dE_dq0[i] -= loc.dE_dq0[i];
  out.dE_dR.assign(loc.dE_dR.size(), 0.0);
  for (size_t k = 0; k < loc.dE_dR.size(); k++) out.dE_dR[k] = -loc.dE_dR[k];
  return out;
}

namespace {
SqeOutput train_one(const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                    double dL_dE, const double *dL_dF, std::vector<double> &grad, const SqeQ0Supervision *sup, bool local,
                    double sign) {
  SqeOutput out = compute_one(S, n, Z, x, Q, q0, true, local);
  if (n == 0) return out;
  if (grad.size() != S.n_params()) grad.assign(S.n_params(), 0.0);
  const Prepared p = prepare(S, n, Z, x, Q, q0);
  const std::vector<double> xv(x, x + 3 * size_t(n));
  const Geo<double> g = geometry(S, n, p.e, xv, p.pr, p.q0, Q, local);
  if (sup && sup->unit && g.learned && sup->weight > 0) { // optional supervision of the learned q0 by unit charges
    std::vector<double> su(sup->n_units, 0.0), c(n, 0.0);
    for (int i = 0; i < n; i++) su[sup->unit[i]] += g.q0[i];
    double L = 0;
    for (int u = 0; u < sup->n_units; u++) L += (su[u] - sup->Qu[u]) * (su[u] - sup->Qu[u]);
    out.q0_loss = sup->weight * L / sup->n_units;
    for (int i = 0; i < n; i++) c[i] = sup->scale * sup->weight * 2.0 * (su[sup->unit[i]] - sup->Qu[sup->unit[i]]) / sup->n_units;
    std::vector<double> Gs(S.n_params(), 0.0);
    q0_backprop(S, g, c, Gs);
    for (size_t k = 0; k < Gs.size(); k++) grad[k] += Gs[k];
  }
  // u* again (cheap; the forward's u is not kept)
  std::vector<double> b = grad_u(g, std::vector<double>(p.pr.size(), 0.0));
  for (double &v : b) v = -v;
  int it = 0;
  const std::vector<double> u = cg(g, b, S.cg_tol, it);
  const std::vector<double> G = grad_theta(S, g, u);
  for (size_t k = 0; k < G.size(); k++) grad[k] += sign * dL_dE * G[k];
  if (!dL_dF) return out;
  // force loss: d(sum_a w_a . F_a)/dtheta = -D_x G[w], with the implicit du = -H^-1 dg/dx[w]
  std::vector<Dual> xd(3 * size_t(n));
  double wn = 0;
  for (size_t k = 0; k < xd.size(); k++) wn += dL_dF[k] * dL_dF[k];
  if (wn == 0) return out;
  for (size_t k = 0; k < xd.size(); k++) xd[k] = Dual(x[k], dL_dF[k]);
  const Geo<Dual> gd = geometry(S, n, p.e, xd, p.pr, p.q0, Q, local);
  std::vector<Dual> ud(u.size());
  for (size_t k = 0; k < u.size(); k++) ud[k] = Dual(u[k], 0.0);
  const std::vector<Dual> gdot = grad_u(gd, ud);
  std::vector<double> rhs(u.size());
  for (size_t k = 0; k < u.size(); k++) rhs[k] = -gdot[k].d;
  const std::vector<double> udot = cg(g, rhs, S.cg_tol, it);
  for (size_t k = 0; k < u.size(); k++) ud[k] = Dual(u[k], udot[k]);
  const std::vector<Dual> Gd = grad_theta(S, gd, ud);
  for (size_t k = 0; k < Gd.size(); k++) grad[k] -= sign * Gd[k].d;
  return out;
}
} // namespace

SqeOutput sqe_train_grad(const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q, const double *q0,
                         double dL_dE, const double *dL_dF, std::vector<double> &grad, const SqeQ0Supervision *sup) {
  if (grad.size() != S.n_params()) grad.assign(S.n_params(), 0.0);
  SqeOutput out = train_one(S, n, Z, x, Q, q0, dL_dE, dL_dF, grad, sup, false, 1.0);
  if (S.lr_R > 0 && n > 0) {
    const SqeOutput loc = train_one(S, n, Z, x, Q, q0, dL_dE, dL_dF, grad, nullptr, true, -1.0);
    out.energy -= loc.energy;
    for (size_t k = 0; k < out.forces.size(); k++) out.forces[k] -= loc.forces[k];
  }
  return out;
}

} // namespace mlipcpp::runtime
