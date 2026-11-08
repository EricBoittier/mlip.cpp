#include "lr_combined.h"

#include "core/gguf_loader.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mlipcpp::runtime {

namespace {

inline double sstep(double x) { return x <= 0 ? 0.0 : x >= 1 ? 1.0 : x * x * x * (x * (x * 6.0 - 15.0) + 10.0); }
inline double dsstep(double x) { return x <= 0 || x >= 1 ? 0.0 : 30.0 * x * x * (x - 1) * (x - 1); }
int uf_find(std::vector<int> &p, int i) {
  while (p[i] != i) i = p[i] = p[p[i]];
  return i;
}
inline double sig(double z) { return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z)); }

double dist(const double *x, int i, int j) {
  double d2 = 0;
  for (int k = 0; k < 3; k++) d2 += (x[3 * i + k] - x[3 * j + k]) * (x[3 * i + k] - x[3 * j + k]);
  return std::sqrt(d2);
}
void add_pair(double *g, const double *x, int i, int j, double r, double dEdr) {
  for (int k = 0; k < 3; k++) {
    const double f = dEdr * (x[3 * i + k] - x[3 * j + k]) / r;
    g[3 * i + k] += f, g[3 * j + k] -= f;
  }
}

// D3 counting function cnt(r) of sqe.cpp (sigmoid times the cn switch) and its derivative
void cn_count(const SqeSpec &S, double R, double r, double &w, double &dw) {
  const double z = 16.0 * (4.0 / 3.0 * R / r - 1.0), sg = sig(z);
  const double dsg = sg * (1 - sg) * 16.0 * (-4.0 / 3.0 * R / (r * r));
  const double t = (r - S.cn_on) / (S.cn_off - S.cn_on);
  const double sw = 1.0 - sstep(t), dsw = -dsstep(t) / (S.cn_off - S.cn_on);
  w = sg * sw, dw = dsg * sw + sg * dsw;
}

std::vector<double> get_f64(const GGUFLoader &L, const std::string &key) {
  gguf_context *g = L.gguf_ctx();
  const int64_t id = gguf_find_key(g, key.c_str());
  if (id < 0) return {};
  if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, id) == GGUF_TYPE_FLOAT64) {
    const double *d = static_cast<const double *>(gguf_get_arr_data(g, id));
    return std::vector<double>(d, d + gguf_get_arr_n(g, id));
  }
  if (gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT64) return {gguf_get_val_f64(g, id)};
  throw std::runtime_error("lrc: " + key + " must be float64");
}

struct Levels {
  std::vector<double> I, A, thp, thm; // levels and tanh(f+-)
  std::vector<double> cnp;            // n x E
};

Levels levels(const LrcSpec &L, const SqeSpec &S, int n, const std::vector<int> &e, const double *x) {
  const size_t E = S.Z.size();
  Levels v;
  v.cnp.assign(size_t(n) * E, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const double r = dist(x, i, j);
      if (!(r < S.cn_off)) continue;
      double w, dw;
      cn_count(S, S.rcov[e[i]] + S.rcov[e[j]], r, w, dw);
      v.cnp[size_t(i) * E + e[j]] += w, v.cnp[size_t(j) * E + e[i]] += w;
    }
  v.I.resize(n), v.A.resize(n), v.thp.resize(n), v.thm.resize(n);
  for (int i = 0; i < n; i++) {
    double fp = L.ap[e[i]], fm = L.am[e[i]];
    for (size_t k = 0; k < E; k++) {
      const double c = v.cnp[size_t(i) * E + k];
      if (c != 0) fp += L.bp[size_t(e[i]) * E + k] * c, fm += L.bm[size_t(e[i]) * E + k] * c;
    }
    v.thp[i] = std::tanh(fp), v.thm[i] = std::tanh(fm);
    v.I[i] = L.ip[e[i]] - L.D * v.thp[i], v.A[i] = L.ea[e[i]] + L.D * v.thm[i];
  }
  return v;
}

std::vector<int> elements(const SqeSpec &S, int n, const int32_t *Z) {
  std::vector<int> e(n);
  for (int i = 0; i < n; i++) e[i] = lrc_index(S, Z[i]);
  return e;
}

} // namespace

int lrc_index(const SqeSpec &S, int z) {
  const int k = S.index_of(z);
  if (k < 0) throw std::runtime_error("lrc: no parameters for atomic number " + std::to_string(z));
  return k;
}

LrcSpec LrcSpec::from_gguf(const GGUFLoader &G, const SqeSpec &S) {
  LrcSpec L;
  L.enabled = G.get_int32("pet.plato_lr.lrc", 0) != 0;
  if (!L.enabled) return L;
  if (!S.enabled) throw std::runtime_error("lrc: needs the sqe keys (pet.plato_lr.sqe)");
  if (S.q0_learned || S.q0_formal) throw std::runtime_error("lrc: sqe_q0 must be uniform (q0 comes from lrc)");
  if (!(S.lr_R > 0)) throw std::runtime_error("lrc: needs sqe_lr_R > 0");
  auto scalar = [&](const char *k, double &v) {
    const auto a = get_f64(G, std::string("pet.plato_lr.lrc_") + k);
    if (!a.empty()) v = a[0];
  };
  scalar("a_on", L.a_on), scalar("a_off", L.a_off), scalar("T", L.T), scalar("D", L.D);
  scalar("tau", L.tau), scalar("merge_w", L.merge_w);
  {
    const std::string mode = G.get_string("pet.plato_lr.lrc_q0_mode", "fermi2");
    if (mode != "fermi2" && mode != "round" && mode != "head" && mode != "graph")
      throw std::runtime_error("lrc: unknown q0 mode " + mode);
    L.round = mode == "round", L.head_only = mode == "head", L.graph = mode == "graph";
  }
  if (L.graph) {
    L.qmodel = G.get_string("pet.plato_lr.lrc_qmodel", "");
    {
      const std::string qc = G.get_string("pet.plato_lr.lrc_qpass_charge", "total");
      if (qc != "total" && qc != "zero") throw std::runtime_error("lrc graph: lrc_qpass_charge must be total or zero");
      L.qpass_neutral = qc == "zero";
    }
    const auto z = get_f64(G, "pet.plato_lr.lrc_fukui_Z");
    L.fukui_phi = get_f64(G, "pet.plato_lr.lrc_fukui_phi");
    for (double v : z) L.fukui_Z.push_back(static_cast<int32_t>(std::lround(v)));
    if (L.qmodel.empty() || L.fukui_Z.empty() || L.fukui_Z.size() != L.fukui_phi.size())
      throw std::runtime_error("lrc graph: needs lrc_qmodel and lrc_fukui_Z / _phi of equal length");
    if (!(L.a_off > L.a_on)) throw std::runtime_error("lrc: bad a_on / a_off");
    L.mp = G.get_int32("pet.plato_lr.lrc_mp", 0) != 0;
    if (L.mp) {
      scalar("mp_rc", L.mp_rc), scalar("mp_width", L.mp_width);
      L.mp_centers = get_f64(G, "pet.plato_lr.lrc_mp_centers");
      for (double v : get_f64(G, "pet.plato_lr.lrc_mp_Z")) L.mp_Z.push_back(static_cast<int32_t>(std::lround(v)));
      L.mp_theta = get_f64(G, "pet.plato_lr.lrc_mp_theta"), L.mp_tau = get_f64(G, "pet.plato_lr.lrc_mp_tau");
      L.mp_build();
    }
    return L;
  }
  const size_t E = S.Z.size();
  if (!(L.a_off > L.a_on)) throw std::runtime_error("lrc: bad a_on / a_off");
  if (L.round || L.head_only) {
    double hs = 0;
    for (size_t k = 0; k < E; k++) hs += S.q0_hi[k] - S.q0_lo[k];
    if (!(hs > 0) || !(L.tau > 0) || !(L.merge_w > 0))
      throw std::runtime_error("lrc round: needs the learned q0 head tables (sqe_q0_*) and tau, merge_w > 0");
    return L;
  }
  L.ip = get_f64(G, "pet.plato_lr.lrc_ip"), L.ea = get_f64(G, "pet.plato_lr.lrc_ea");
  L.h = get_f64(G, "pet.plato_lr.lrc_h"), L.l = get_f64(G, "pet.plato_lr.lrc_l");
  L.ap = get_f64(G, "pet.plato_lr.lrc_ap"), L.am = get_f64(G, "pet.plato_lr.lrc_am");
  L.bp = get_f64(G, "pet.plato_lr.lrc_bp"), L.bm = get_f64(G, "pet.plato_lr.lrc_bm");
  for (auto *a : {&L.ip, &L.ea, &L.h, &L.l, &L.ap, &L.am})
    if (a->size() != E) throw std::runtime_error("lrc: per-element tables of the wrong length");
  if (L.bp.size() != E * E || L.bm.size() != E * E) throw std::runtime_error("lrc: b tables of the wrong length");
  if (!(L.a_off > L.a_on) || !(L.T > 0)) throw std::runtime_error("lrc: bad a_on / a_off / T");
  return L;
}

void LrcSpec::to_gguf(gguf_context *g) const {
  if (!enabled) return;
  gguf_set_val_i32(g, "pet.plato_lr.lrc", 1);
  auto scalar = [&](const char *k, double v) { gguf_set_val_f64(g, (std::string("pet.plato_lr.lrc_") + k).c_str(), v); };
  scalar("a_on", a_on), scalar("a_off", a_off), scalar("T", T), scalar("D", D), scalar("tau", tau);
  scalar("merge_w", merge_w);
  gguf_set_val_str(g, "pet.plato_lr.lrc_q0_mode", graph ? "graph" : round ? "round" : head_only ? "head" : "fermi2");
  auto arr = [&](const char *k, const std::vector<double> &v) {
    gguf_set_arr_data(g, (std::string("pet.plato_lr.lrc_") + k).c_str(), GGUF_TYPE_FLOAT64, v.data(), v.size());
  };
  if (graph) {
    gguf_set_val_str(g, "pet.plato_lr.lrc_qmodel", qmodel.c_str());
    gguf_set_val_str(g, "pet.plato_lr.lrc_qpass_charge", qpass_neutral ? "zero" : "total");
    std::vector<double> z(fukui_Z.begin(), fukui_Z.end());
    arr("fukui_Z", z), arr("fukui_phi", fukui_phi);
    if (mp) {
      gguf_set_val_i32(g, "pet.plato_lr.lrc_mp", 1);
      scalar("mp_rc", mp_rc), scalar("mp_width", mp_width);
      std::vector<double> mz(mp_Z.begin(), mp_Z.end());
      arr("mp_centers", mp_centers), arr("mp_Z", mz), arr("mp_theta", mp_theta), arr("mp_tau", mp_tau);
    }
    return;
  }
  if (round || head_only) return;
  arr("ip", ip), arr("ea", ea), arr("h", h), arr("l", l), arr("ap", ap), arr("am", am), arr("bp", bp), arr("bm", bm);
}

// ---- reference charges -------------------------------------------------------------------------------------------------
LrcQ0 q0_fermi2(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q) {
  LrcQ0 out;
  if (n == 0) return out;
  const std::vector<int> e = elements(S, n, Z);
  const Levels v = levels(L, S, n, e, x);
  const double T = L.T;
  double sh = 0, sl = 0, lo = 1e300, hi = -1e300;
  for (int i = 0; i < n; i++) {
    sh += L.h[e[i]], sl += L.l[e[i]];
    lo = std::min({lo, v.I[i] / T, v.A[i] / T}), hi = std::max({hi, v.I[i] / T, v.A[i] / T});
  }
  auto total = [&](double t, double *d) {
    double s = 0, ds = 0;
    for (int i = 0; i < n; i++) {
      const double sp = sig(t - v.I[i] / T), sm = sig(v.A[i] / T - t);
      s += L.h[e[i]] * sp - L.l[e[i]] * sm, ds += L.h[e[i]] * sp * (1 - sp) + L.l[e[i]] * sm * (1 - sm);
    }
    if (d) *d = ds;
    return s;
  };
  double t;
  out.feasible = -sl < Q && Q < sh;
  if (out.feasible) {
    double a = lo - 60, b = hi + 60;
    for (int it = 0; it < 200; it++) {
      const double m = 0.5 * (a + b);
      (total(m, nullptr) < Q ? a : b) = m;
    }
    t = 0.5 * (a + b);
    for (int it = 0; it < 8; it++) {
      double d;
      const double s = total(t, &d);
      if (d > 0) t -= (s - Q) / d;
    }
  } else {
    t = Q >= sh ? hi + 1e3 : lo - 1e3;
  }
  out.q0.resize(n), out.wt.resize(n), out.wp.resize(n), out.wm.resize(n);
  double tot = 0;
  for (int i = 0; i < n; i++) {
    const double sp = sig(t - v.I[i] / T), sm = sig(v.A[i] / T - t), h = L.h[e[i]], l = L.l[e[i]];
    out.q0[i] = h * sp - l * sm, tot += out.q0[i];
    out.wt[i] = h * sp * (1 - sp) + l * sm * (1 - sm);
    out.wp[i] = h * sp * (1 - sp) * L.D * (1 - v.thp[i] * v.thp[i]) / T;  // dI/df+ = -D (1 - th^2)
    out.wm[i] = -l * sm * (1 - sm) * L.D * (1 - v.thm[i] * v.thm[i]) / T; // dA/df- = +D (1 - th^2)
  }
  if (!out.feasible)
    for (int i = 0; i < n; i++) out.q0[i] += (Q - tot) / n;
  return out;
}

void vjp_fermi2(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, const LrcQ0 &q,
                const double *c, double *g) {
  if (n == 0) return;
  const std::vector<int> e = elements(S, n, Z);
  const size_t E = S.Z.size();
  // dq0_i = wp_i df+_i + wm_i df-_i + wt_i dt, dt = -sum_j (wp_j df+_j + wm_j df-_j) / sum wt (infeasible: the uniform
  // spread makes the projection the plain mean)
  double sw = 0, swc = 0, sc = 0;
  for (int i = 0; i < n; i++) sw += q.wt[i], swc += q.wt[i] * c[i], sc += c[i];
  const double m = q.feasible && sw > 0 ? swc / sw : sc / n;
  std::vector<double> kp(n), km(n);
  for (int i = 0; i < n; i++) kp[i] = (c[i] - m) * q.wp[i], km[i] = (c[i] - m) * q.wm[i];
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const double r = dist(x, i, j);
      if (!(r < S.cn_off)) continue;
      double w, dw;
      cn_count(S, S.rcov[e[i]] + S.rcov[e[j]], r, w, dw);
      const double k = kp[i] * L.bp[size_t(e[i]) * E + e[j]] + km[i] * L.bm[size_t(e[i]) * E + e[j]] +
                       kp[j] * L.bp[size_t(e[j]) * E + e[i]] + km[j] * L.bm[size_t(e[j]) * E + e[i]];
      if (k != 0) add_pair(g, x, i, j, r, k * dw);
    }
}

namespace {
// learned logistic head of sqe.cpp (q0 = lo + d sigmoid(f + t), sum = Q): values, dq/dt per atom, CN pairs for the VJP
void head(const SqeSpec &S, int n, const std::vector<int> &e, const double *x, double Q, LrcQ0 &o) {
  const size_t E = S.Z.size();
  std::vector<double> f(n), cnp(size_t(n) * E, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const double r = dist(x, i, j);
      if (!(r < S.cn_off)) continue;
      double w, dw;
      cn_count(S, S.rcov[e[i]] + S.rcov[e[j]], r, w, dw);
      cnp[size_t(i) * E + e[j]] += w, cnp[size_t(j) * E + e[i]] += w;
    }
  double slo = 0, shi = 0;
  for (int i = 0; i < n; i++) {
    f[i] = S.q0_a[e[i]];
    for (size_t k = 0; k < E; k++) f[i] += S.q0_b[size_t(e[i]) * E + k] * cnp[size_t(i) * E + k];
    slo += S.q0_lo[e[i]], shi += S.q0_hi[e[i]];
  }
  auto tot = [&](double t, double *d) {
    double v = 0, dv = 0;
    for (int i = 0; i < n; i++) {
      const double dd = S.q0_hi[e[i]] - S.q0_lo[e[i]], sg = sig(f[i] + t);
      v += S.q0_lo[e[i]] + dd * sg, dv += dd * sg * (1 - sg);
    }
    if (d) *d = dv;
    return v;
  };
  o.feasible = slo < Q && Q < shi;
  double t;
  if (o.feasible) {
    double a = -60, b = 60;
    for (int it = 0; it < 200; it++) {
      const double m = 0.5 * (a + b);
      (tot(m, nullptr) < Q ? a : b) = m;
    }
    t = 0.5 * (a + b);
    for (int it = 0; it < 6; it++) {
      double d;
      const double v = tot(t, &d);
      if (d > 0) t -= (v - Q) / d;
    }
  } else {
    t = Q >= shi ? 60.0 : -60.0;
  }
  o.q0.resize(n), o.wt.resize(n);
  double sum = 0;
  for (int i = 0; i < n; i++) {
    const double dd = S.q0_hi[e[i]] - S.q0_lo[e[i]], sg = sig(f[i] + t);
    o.q0[i] = S.q0_lo[e[i]] + dd * sg, o.wt[i] = dd * sg * (1 - sg), sum += o.q0[i];
  }
  if (!o.feasible)
    for (int i = 0; i < n; i++) o.q0[i] += (Q - sum) / n;
}

// dE/dx of sum_i h_i s_i(x) for the logistic head (weight on df_i: wt_i (h_i - m))
void head_vjp(const SqeSpec &S, int n, const std::vector<int> &e, const double *x, const LrcQ0 &o, const double *h,
              double *g) {
  const size_t E = S.Z.size();
  double sw = 0, swh = 0, sh = 0;
  for (int i = 0; i < n; i++) sw += o.wt[i], swh += o.wt[i] * h[i], sh += h[i];
  const double m = o.feasible && sw > 0 ? swh / sw : sh / n;
  std::vector<double> k(n);
  for (int i = 0; i < n; i++) k[i] = o.wt[i] * (h[i] - m);
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const double r = dist(x, i, j);
      if (!(r < S.cn_off)) continue;
      double w, dw;
      cn_count(S, S.rcov[e[i]] + S.rcov[e[j]], r, w, dw);
      const double c = k[i] * S.q0_b[size_t(e[i]) * E + e[j]] + k[j] * S.q0_b[size_t(e[j]) * E + e[i]];
      if (c != 0) add_pair(g, x, i, j, r, c * dw);
    }
}

// merge switch S(r) = smootherstep((r - r_ct) / merge_w) of a cross-component pair, and dS/dr
inline void merge_s(const LrcSpec &L, const SqeSpec &S, double r, double &v, double &d) {
  const double u = (r - S.r_ct) / L.merge_w;
  v = sstep(u), d = dsstep(u) / L.merge_w;
}

LrcQ0 q0_round(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q) {
  LrcQ0 o;
  if (n == 0) return o;
  const std::vector<int> e = elements(S, n, Z);
  head(S, n, e, x, Q, o);
  o.comp.assign(n, 0), o.K = 1;
  if (L.head_only) return o;
  // components at r < r_ct
  std::vector<int> par(n);
  std::iota(par.begin(), par.end(), 0);
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      if (dist(x, i, j) < S.r_ct) {
        const int a = uf_find(par, i), b = uf_find(par, j);
        if (a != b) par[std::max(a, b)] = std::min(a, b);
      }
  std::vector<int> lab(n, -1);
  o.comp.resize(n), o.K = 0;
  for (int i = 0; i < n; i++) {
    const int r = uf_find(par, i);
    if (lab[r] < 0) lab[r] = o.K++;
    o.comp[i] = lab[r];
  }
  const int K = o.K;
  if (K == 1) return o;
  o.sc.assign(K, 0.0), o.w.assign(K, 1.0), o.ncomp.assign(K, 0);
  for (int i = 0; i < n; i++) o.sc[o.comp[i]] += o.q0[i], o.ncomp[o.comp[i]]++;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      if (o.comp[i] == o.comp[j]) continue;
      const double r = dist(x, i, j);
      if (r >= S.r_ct + L.merge_w) continue;
      double v, d;
      merge_s(L, S, r, v, d);
      o.w[o.comp[i]] *= v, o.w[o.comp[j]] *= v;
    }
  // partitions sum n = Q around round(s_c)
  const int Qi = static_cast<int>(std::lround(Q));
  const int W = K <= 5 ? 2 : K <= 10 ? 1 : 0;
  if (K > 16) throw std::runtime_error("lrc round: more than 16 components");
  std::vector<int> base(K), lo(K), span(K);
  for (int c = 0; c < K; c++) {
    base[c] = static_cast<int>(std::lround(o.sc[c]));
    lo[c] = W > 0 ? base[c] - W : static_cast<int>(std::floor(o.sc[c]));
    span[c] = W > 0 ? 2 * W + 1 : 2;
  }
  std::vector<std::vector<int>> combos;
  std::vector<double> lw;
  std::vector<int> cur(K);
  double lmax = -1e300;
  std::function<void(int, int)> rec = [&](int c, int s) {
    if (c == K) {
      if (s != Qi) return;
      double l = 0;
      for (int k = 0; k < K; k++) l -= (cur[k] - o.sc[k]) * (cur[k] - o.sc[k]);
      l /= 2 * L.tau * L.tau;
      combos.push_back(cur), lw.push_back(l), lmax = std::max(lmax, l);
      return;
    }
    for (int v = lo[c]; v < lo[c] + span[c]; v++) cur[c] = v, rec(c + 1, s + v);
  };
  rec(0, 0);
  o.nbar.assign(K, 0.0), o.cov.assign(size_t(K) * K, 0.0);
  if (combos.empty()) { // no partition in the window: no rounding
    for (int c = 0; c < K; c++) o.nbar[c] = o.sc[c], o.w[c] = 0.0;
  } else {
    double Zs = 0;
    for (double &l : lw) l = std::exp(l - lmax), Zs += l;
    for (size_t p = 0; p < combos.size(); p++)
      for (int a = 0; a < K; a++) o.nbar[a] += lw[p] / Zs * combos[p][a];
    for (size_t p = 0; p < combos.size(); p++)
      for (int a = 0; a < K; a++)
        for (int b = 0; b < K; b++)
          o.cov[size_t(a) * K + b] += lw[p] / Zs * (combos[p][a] - o.nbar[a]) * (combos[p][b] - o.nbar[b]);
  }
  double U = 0, Wsum = 0;
  std::vector<double> d(K);
  for (int c = 0; c < K; c++) d[c] = o.w[c] * (o.nbar[c] - o.sc[c]), U += d[c], Wsum += o.w[c];
  for (int c = 0; c < K; c++)
    if (Wsum > 0) d[c] -= o.w[c] * U / Wsum;
  for (int i = 0; i < n; i++) o.q0[i] += d[o.comp[i]] / o.ncomp[o.comp[i]];
  return o;
}

void vjp_round(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, const LrcQ0 &o,
               const double *c, double *g) {
  if (n == 0) return;
  const std::vector<int> e = elements(S, n, Z);
  const int K = o.K;
  std::vector<double> h(c, c + n);
  if (K > 1) {
    std::vector<double> gc(K, 0.0), gam(K), u(K);
    double U = 0, Wsum = 0, gw = 0;
    for (int i = 0; i < n; i++) gc[o.comp[i]] += c[i] / o.ncomp[o.comp[i]];
    for (int k = 0; k < K; k++) u[k] = o.w[k] * (o.nbar[k] - o.sc[k]), U += u[k], Wsum += o.w[k], gw += gc[k] * o.w[k];
    for (int k = 0; k < K; k++) gam[k] = Wsum > 0 ? gc[k] - gw / Wsum : gc[k];
    // dE/dw (explicit through the zero-sum correction, and through u = w (nbar - s))
    std::vector<double> dw(K), ds(K, 0.0);
    for (int k = 0; k < K; k++)
      dw[k] = (Wsum > 0 ? -gc[k] * U / Wsum + gw * U / (Wsum * Wsum) : 0.0) + gam[k] * (o.nbar[k] - o.sc[k]);
    for (int b = 0; b < K; b++) {
      ds[b] = -gam[b] * o.w[b];
      for (int a = 0; a < K; a++) ds[b] += gam[a] * o.w[a] * o.cov[size_t(a) * K + b] / (L.tau * L.tau);
    }
    for (int i = 0; i < n; i++) h[i] += ds[o.comp[i]];
    // w_c = prod S(r_ij) over cross pairs: dw_c/dr_ij = (prod of the others) S'(r_ij)
    std::vector<std::vector<std::pair<int, int>>> pairs(K);
    std::vector<std::vector<double>> sv(K), sd(K);
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++) {
        if (o.comp[i] == o.comp[j]) continue;
        const double r = dist(x, i, j);
        if (r >= S.r_ct + L.merge_w) continue;
        double v, d;
        merge_s(L, S, r, v, d);
        for (int k : {o.comp[i], o.comp[j]}) pairs[k].push_back({i, j}), sv[k].push_back(v), sd[k].push_back(d);
      }
    for (int k = 0; k < K; k++) {
      if (dw[k] == 0) continue;
      const size_t m = pairs[k].size();
      std::vector<double> pre(m + 1, 1.0), suf(m + 1, 1.0);
      for (size_t q = 0; q < m; q++) pre[q + 1] = pre[q] * sv[k][q];
      for (size_t q = m; q-- > 0;) suf[q] = suf[q + 1] * sv[k][q];
      for (size_t q = 0; q < m; q++) {
        const double dd = pre[q] * suf[q + 1] * sd[k][q];
        if (dd != 0) add_pair(g, x, pairs[k][q].first, pairs[k][q].second, dist(x, pairs[k][q].first, pairs[k][q].second), dw[k] * dd);
      }
    }
  }
  head_vjp(S, n, e, x, o, h.data(), g);
}
} // namespace

double LrcSpec::fukui(int z) const {
  for (size_t k = 0; k < fukui_Z.size(); k++)
    if (fukui_Z[k] == z) {
      const double p = fukui_phi[k];
      return (p > 30 ? p : std::log1p(std::exp(p))) + 1e-6;
    }
  throw std::runtime_error("lrc graph: no Fukui parameter for atomic number " + std::to_string(z));
}

// ---- atomic multipoles (graph mode, lrc_mp) ---------------------------------------------------------------------------
void LrcSpec::mp_build() {
  const size_t E = mp_Z.size(), K = mp_centers.size();
  if (!mp) return;
  if (E == 0 || K == 0 || mp_theta.size() != E * E * K || mp_tau.size() != E * E * K || !(mp_rc > 0) || !(mp_width > 0))
    throw std::runtime_error("lrc mp: needs mp_Z, mp_centers and E x E x K mp_theta / mp_tau, mp_rc, mp_width > 0");
  mp_index.assign(128, -1);
  for (size_t e = 0; e < E; e++)
    if (mp_Z[e] >= 0 && mp_Z[e] < 128) mp_index[mp_Z[e]] = static_cast<int>(e);
}

namespace {
// coefficient c(r) = sum_k p[k] g_k(r) and dc/dr for one element pair (p: K values)
inline void mp_radial(const LrcSpec &L, const double *p, double r, double &c, double &dc) {
  c = dc = 0;
  if (r >= L.mp_rc) return;
  const double fc = 0.5 * (1 + std::cos(M_PI * r / L.mp_rc)), dfc = -0.5 * M_PI / L.mp_rc * std::sin(M_PI * r / L.mp_rc);
  const double w2 = L.mp_width * L.mp_width;
  for (size_t k = 0; k < L.mp_centers.size(); k++) {
    const double d = r - L.mp_centers[k], e = std::exp(-0.5 * d * d / w2);
    c += p[k] * e * fc, dc += p[k] * e * (dfc - fc * d / w2);
  }
}
// neighbour pairs (i, j) within mp_rc with the element-pair parameter offsets; visit(i, j, r, u[3], off)
template <class F> void mp_pairs(const LrcSpec &L, int n, const int32_t *Z, const double *x, F &&visit) {
  const size_t E = L.mp_Z.size(), K = L.mp_centers.size();
  for (int i = 0; i < n; i++) {
    const int ei = Z[i] >= 0 && Z[i] < 128 ? L.mp_index[Z[i]] : -1;
    if (ei < 0) continue;
    for (int j = 0; j < n; j++) {
      if (j == i) continue;
      const int ej = Z[j] >= 0 && Z[j] < 128 ? L.mp_index[Z[j]] : -1;
      if (ej < 0) continue;
      const double r = dist(x, i, j);
      if (r >= L.mp_rc || r == 0) continue;
      double u[3];
      for (int k = 0; k < 3; k++) u[k] = (x[3 * j + k] - x[3 * i + k]) / r;
      visit(i, j, r, u, (size_t(ei) * E + size_t(ej)) * K);
    }
  }
}
} // namespace

void lrc_moments(const LrcSpec &L, int n, const int32_t *Z, const double *x, std::vector<double> &mu,
                 std::vector<double> &theta) {
  mu.assign(3 * size_t(n), 0.0), theta.assign(9 * size_t(n), 0.0);
  if (!L.mp) return;
  mp_pairs(L, n, Z, x, [&](int i, int, double r, const double *u, size_t off) {
    double c, dc, t, dt;
    mp_radial(L, L.mp_theta.data() + off, r, c, dc);
    mp_radial(L, L.mp_tau.data() + off, r, t, dt);
    for (int a = 0; a < 3; a++) {
      mu[3 * size_t(i) + a] += c * u[a];
      for (int b = 0; b < 3; b++) theta[9 * size_t(i) + 3 * a + b] += t * (1.5 * u[a] * u[b] - (a == b ? 0.5 : 0.0));
    }
  });
}

void lrc_moments_vjp(const LrcSpec &L, int n, const int32_t *Z, const double *x, const double *G, const double *H,
                     double *g) {
  if (!L.mp) return;
  mp_pairs(L, n, Z, x, [&](int i, int j, double r, const double *u, size_t off) {
    double c, dc, t, dt;
    mp_radial(L, L.mp_theta.data() + off, r, c, dc);
    mp_radial(L, L.mp_tau.data() + off, r, t, dt);
    const double *Gi = G + 3 * size_t(i), *Hi = H + 9 * size_t(i);
    double Gu = 0, Hu[3] = {0, 0, 0}, uHu = 0, tr = Hi[0] + Hi[4] + Hi[8];
    for (int a = 0; a < 3; a++) {
      Gu += Gi[a] * u[a];
      for (int b = 0; b < 3; b++) Hu[a] += 0.5 * (Hi[3 * a + b] + Hi[3 * b + a]) * u[b];
    }
    for (int a = 0; a < 3; a++) uHu += u[a] * Hu[a];
    // s = x_j - x_i: d(c(r) G.u)/ds = c' (G.u) u + c / r (G - (G.u) u);
    // d(t(r) (3/2 u.Hu - 1/2 tr H))/ds = t' (3/2 uHu - 1/2 tr H) u + 3 t / r (Hu - (uHu) u)
    for (int a = 0; a < 3; a++) {
      const double d = dc * Gu * u[a] + c / r * (Gi[a] - Gu * u[a]) + dt * (1.5 * uHu - 0.5 * tr) * u[a] +
                       3.0 * t / r * (Hu[a] - uHu * u[a]);
      g[3 * size_t(j) + a] += d, g[3 * size_t(i) + a] -= d;
    }
  });
}

namespace {
// graph mode: q0 = q_raw + f (Q - sum q_raw) / sum f; wt = f / sum f (f per element: no position dependence)
LrcQ0 q0_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, double Q, int spin) {
  if (!L.qsrc) throw std::runtime_error("lrc graph: the charge pass (lrc_qmodel) is not loaded");
  LrcQ0 o;
  o.Q = Q, o.spin = spin;
  if (n == 0) return o;
  o.q0 = L.qsrc->raw(n, Z, x, L.qpass_neutral ? 0.0 : Q, spin);
  o.wt.resize(n);
  double sq = 0, sf = 0;
  for (int i = 0; i < n; i++) sq += o.q0[i], o.wt[i] = L.fukui(Z[i]), sf += o.wt[i];
  for (int i = 0; i < n; i++) o.wt[i] /= sf, o.q0[i] += o.wt[i] * (Q - sq);
  return o;
}
void vjp_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, const LrcQ0 &o, const double *c, double *g) {
  double cb = 0;
  for (int i = 0; i < n; i++) cb += c[i] * o.wt[i];
  std::vector<double> w(n);
  bool any = false;
  for (int i = 0; i < n; i++) w[i] = c[i] - cb, any |= w[i] != 0;
  if (any) L.qsrc->raw_vjp(n, Z, x, L.qpass_neutral ? 0.0 : o.Q, o.spin, w.data(), g);
}
} // namespace

LrcQ0 lrc_q0(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q, int spin) {
  if (L.graph) return q0_graph(L, n, Z, x, Q, spin);
  return L.round || L.head_only ? q0_round(L, S, n, Z, x, Q) : q0_fermi2(L, S, n, Z, x, Q);
}
void lrc_q0_vjp(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, const LrcQ0 &q,
                const double *c, double *g) {
  if (L.graph) vjp_graph(L, n, Z, x, q, c, g);
  else if (L.round || L.head_only) vjp_round(L, S, n, Z, x, q, c, g);
  else vjp_fermi2(L, S, n, Z, x, q, c, g);
}

// ---- smooth clusters -------------------------------------------------------------------------------------------------
bool LrcClusters::trivial() const { return s == 1 || se.empty(); }

namespace {
// weak components over super-nodes: the super-edges of each
std::vector<std::vector<int>> weak_components(const LrcClusters &c) {
  std::vector<int> p(c.s);
  std::iota(p.begin(), p.end(), 0);
  for (auto [a, b] : c.se) {
    const int ra = uf_find(p, a), rb = uf_find(p, b);
    if (ra != rb) p[std::max(ra, rb)] = std::min(ra, rb);
  }
  std::vector<std::vector<int>> comp(c.s);
  for (size_t k = 0; k < c.se.size(); k++) comp[uf_find(p, c.se[k].first)].push_back(int(k));
  std::vector<std::vector<int>> out;
  for (auto &v : comp)
    if (!v.empty()) out.push_back(v);
  return out;
}
// enumerate the 2^m states of a component's super-edges: f(state, connected(a, b)) callback with the state probability
// and, for the gradient, the probability without edge e's factor (pre/suffix products)
template <class F> void enumerate(const LrcClusters &c, const std::vector<int> &edges, F &&visit) {
  const int m = static_cast<int>(edges.size());
  std::vector<int> nodes;
  for (int k : edges) nodes.push_back(c.se[k].first), nodes.push_back(c.se[k].second);
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  std::vector<int> p(c.s), pre(m + 1), dummy;
  std::vector<double> fac(m), pr(m + 1), su(m + 1);
  for (uint64_t st = 0; st < (uint64_t(1) << m); st++) {
    for (int a : nodes) p[a] = a;
    for (int k = 0; k < m; k++) {
      const bool on = (st >> k) & 1;
      fac[k] = on ? c.p[edges[k]] : 1.0 - c.p[edges[k]];
      if (on) {
        const int ra = uf_find(p, c.se[edges[k]].first), rb = uf_find(p, c.se[edges[k]].second);
        if (ra != rb) p[std::max(ra, rb)] = std::min(ra, rb);
      }
    }
    pr[0] = 1, su[m] = 1;
    for (int k = 0; k < m; k++) pr[k + 1] = pr[k] * fac[k];
    for (int k = m - 1; k >= 0; k--) su[k] = su[k + 1] * fac[k];
    for (int a : nodes) p[a] = uf_find(p, a);
    visit(st, pr[m], p, nodes, pr, su);
  }
}
} // namespace

LrcClusters lrc_clusters(const LrcSpec &L, int n, const double *x) {
  LrcClusters c;
  std::vector<int> par(n);
  std::iota(par.begin(), par.end(), 0);
  std::vector<std::pair<int, int>> weak;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const double r = dist(x, i, j);
      if (r <= L.a_on) {
        const int a = uf_find(par, i), b = uf_find(par, j);
        if (a != b) par[std::max(a, b)] = std::min(a, b);
      } else if (r < L.a_off) {
        weak.push_back({i, j});
      }
    }
  c.sn.resize(n);
  std::vector<int> lab(n, -1);
  c.s = 0;
  for (int i = 0; i < n; i++) { // numbered by first atom
    const int r = uf_find(par, i);
    if (lab[r] < 0) lab[r] = c.s++;
    c.sn[i] = lab[r];
  }
  std::vector<int> seidx; // map (a, b) -> index
  for (auto [i, j] : weak) {
    int a = c.sn[i], b = c.sn[j];
    if (a == b) continue;
    if (a > b) std::swap(a, b);
    int k = -1;
    for (size_t q = 0; q < c.se.size(); q++)
      if (c.se[q].first == a && c.se[q].second == b) k = int(q);
    if (k < 0) k = int(c.se.size()), c.se.push_back({a, b}), c.p.push_back(0.0);
    const double r = dist(x, i, j), w = L.a_off - L.a_on, u = (r - L.a_on) / w;
    c.weak.push_back({i, j, k, 1.0 - sstep(u), -dsstep(u) / w});
  }
  std::vector<double> none(c.se.size(), 1.0); // prod (1 - A)
  for (const auto &wk : c.weak) none[wk.e] *= 1.0 - wk.A;
  for (size_t k = 0; k < c.se.size(); k++) c.p[k] = 1.0 - none[k];
  c.R.assign(size_t(c.s) * c.s, 0.0);
  for (int a = 0; a < c.s; a++) c.R[size_t(a) * c.s + a] = 1.0;
  for (const auto &edges : weak_components(c)) {
    if (int(edges.size()) > L.max_super_edges) {
      // fallback (rare: many molecules all 5-6 A apart): the weak component counts as one cluster (R = 1 within it,
      // no gradient). Not smooth where the super-edge count crosses the limit; counted in c.fallback.
      std::vector<int> nodes;
      for (int k : edges) nodes.push_back(c.se[k].first), nodes.push_back(c.se[k].second);
      for (int a : nodes)
        for (int b : nodes) c.R[size_t(a) * c.s + b] = 1.0;
      c.fallback = true;
      continue;
    }
    enumerate(c, edges, [&](uint64_t, double P, const std::vector<int> &p, const std::vector<int> &nodes,
                            const std::vector<double> &, const std::vector<double> &) {
      for (int a : nodes)
        for (int b : nodes)
          if (a != b && p[a] == p[b]) c.R[size_t(a) * c.s + b] += P;
    });
  }
  return c;
}

void lrc_clusters_vjp(const LrcSpec &L, const LrcClusters &c, int n, const double *x, const std::vector<double> &W,
                      double *g) {
  (void)n;
  if (c.trivial()) return;
  std::vector<double> dp(c.se.size(), 0.0);
  for (const auto &edges : weak_components(c)) {
    const int m = static_cast<int>(edges.size());
    if (m > L.max_super_edges) continue; // fallback component: R = 1, no gradient
    enumerate(c, edges, [&](uint64_t st, double, const std::vector<int> &p, const std::vector<int> &nodes,
                            const std::vector<double> &pr, const std::vector<double> &su) {
      double f = 0;
      for (int a : nodes)
        for (int b : nodes)
          if (a != b && p[a] == p[b]) f += W[size_t(a) * c.s + b];
      if (f == 0) return;
      for (int k = 0; k < m; k++) dp[edges[k]] += f * pr[k] * su[k + 1] * (((st >> k) & 1) ? 1.0 : -1.0);
    });
  }
  // p = 1 - prod (1 - A): dp/dA_w = prod over the other links (1 - A)
  for (size_t w = 0; w < c.weak.size(); w++) {
    const auto &wk = c.weak[w];
    if (dp[wk.e] == 0 || wk.dA == 0) continue;
    double others = 1.0;
    for (size_t v = 0; v < c.weak.size(); v++)
      if (v != w && c.weak[v].e == wk.e) others *= 1.0 - c.weak[v].A;
    add_pair(g, x, wk.i, wk.j, dist(x, wk.i, wk.j), dp[wk.e] * others * wk.dA);
  }
}

// ---- conditioning --------------------------------------------------------------------------------------------------
bool LrcCondition::needs_cond_forces() const {
  for (double v : dS)
    if (v != 0) return true;
  return false;
}

LrcCondition lrc_condition(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q,
                           int max_charge, int spin) {
  LrcCondition c;
  c.cl = lrc_clusters(L, n, x);
  const int s = c.cl.s;
  c.Q0.assign(s, 0.0), c.Qloc.assign(s, 0.0), c.S.assign(s, 0.0), c.dS.assign(s, 0.0), c.k.assign(s, 0);
  if (s == 1) {
    c.Q0[0] = c.Qloc[0] = Q;
  } else {
    c.q0 = lrc_q0(L, S, n, Z, x, Q, spin);
    for (int i = 0; i < n; i++) c.Q0[c.cl.sn[i]] += c.q0.q0[i];
    for (int a = 0; a < s; a++)
      for (int b = 0; b < s; b++) c.Qloc[a] += c.cl.R[size_t(a) * s + b] * c.Q0[b];
  }
  for (int a = 0; a < s; a++) {
    double v = c.Qloc[a];
    const double rv = std::round(v);
    if (std::abs(v - rv) < 1e-9) v = rv; // solver noise on an integer
    const double k = std::floor(v), u = v - k;
    c.k[a] = static_cast<int32_t>(k), c.S[a] = sstep(u), c.dS[a] = dsstep(u);
  }
  c.row_charge.resize(n), c.row_charge_hi.resize(n), c.blend.resize(n);
  for (int i = 0; i < n; i++) {
    const int a = c.cl.sn[i];
    const int k = std::clamp<int>(c.k[a], -max_charge, max_charge);
    c.row_charge[i] = k, c.row_charge_hi[i] = std::min(k + 1, max_charge), c.blend[i] = float(c.S[a]);
    if (c.k[a] + 1 > max_charge || c.k[a] < -max_charge) c.blend[i] = c.k[a] >= max_charge ? 0.0f : 1.0f;
  }
  return c;
}

void lrc_condition_forces(const LrcSpec &L, const SqeSpec &S, const LrcCondition &c, int n, const int32_t *Z,
                          const double *x, const std::vector<double> &G, double *F) {
  const int s = c.cl.s;
  if (s == 1) return;
  std::vector<double> g(3 * size_t(n), 0.0), W(size_t(s) * s, 0.0), cq(n, 0.0);
  std::vector<double> gs(s);
  for (int a = 0; a < s; a++) gs[a] = G[a] * c.dS[a]; // dE/dQloc_a
  for (int a = 0; a < s; a++)
    for (int b = 0; b < s; b++)
      if (a != b) W[size_t(a) * s + b] = gs[a] * c.Q0[b];
  lrc_clusters_vjp(L, c.cl, n, x, W, g.data());
  for (int i = 0; i < n; i++) {
    const int b = c.cl.sn[i];
    for (int a = 0; a < s; a++) cq[i] += gs[a] * c.cl.R[size_t(a) * s + b];
  }
  lrc_q0_vjp(L, S, n, Z, x, c.q0, cq.data(), g.data());
  for (size_t k = 0; k < g.size(); k++) F[k] -= g[k];
}

std::vector<double> lrc_blend_rows(const LrcSpec &L, const SqeSpec &S, const LrcCondition &c, int n, const int32_t *Z,
                                   const double *x) {
  const int s = c.cl.s;
  std::vector<double> rows(size_t(s) * 3 * n, 0.0);
  if (s == 1) return rows;
  for (int a = 0; a < s; a++) {
    if (c.dS[a] == 0) continue;
    std::vector<double> G(s, 0.0);
    G[a] = 1.0;
    double *r = rows.data() + size_t(a) * 3 * n; // lrc_condition_forces subtracts dS_a/dx
    lrc_condition_forces(L, S, c, n, Z, x, G, r);
    for (int k = 0; k < 3 * n; k++) r[k] = -r[k];
  }
  return rows;
}

namespace {
// graph mode: fixed-charge Coulomb of the inter-cluster pairs beyond lr_R - lr_t
LrcEnergy energy_fixed(const LrcSpec &L, const SqeSpec &S, const LrcCondition &c, int n, const int32_t *Z,
                       const double *x, bool forces) {
  constexpr double KE = 14.399645;
  LrcEnergy out;
  out.q0 = c.q0.q0, out.q = c.q0.q0;
  if (forces) out.forces.assign(3 * size_t(n), 0.0);
  const int s = c.cl.s;
  const double r0 = S.lr_R - S.lr_t;
  std::vector<double> g(3 * size_t(n), 0.0), dq(n, 0.0), W(size_t(s) * s, 0.0);
  const double *q = c.q0.q0.data();
  if (L.mp) { // charges + atomic dipoles and quadrupoles, same pairs, switch and (1 - R)
    std::vector<double> mu, th, Gm(3 * size_t(n), 0.0), Hm(9 * size_t(n), 0.0);
    lrc_moments(L, n, Z, x, mu, th);
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++) {
        const int a = c.cl.sn[i], b = c.cl.sn[j];
        if (a == b) continue;
        const double R = dist(x, i, j);
        if (R <= r0) continue;
        const double m = 1.0 - c.cl.R[size_t(a) * s + b];
        if (m == 0) continue;
        const double t = (R - r0) / S.lr_t, T = sstep(t), dT = dsstep(t) / S.lr_t;
        double r[3];
        for (int k = 0; k < 3; k++) r[k] = x[3 * j + k] - x[3 * i + k];
        const double *mi = &mu[3 * size_t(i)], *mj = &mu[3 * size_t(j)];
        const double *Ti = &th[9 * size_t(i)], *Tj = &th[9 * size_t(j)];
        const double R2 = R * R, R3 = R2 * R, R5 = R3 * R2, R7 = R5 * R2;
        double mir = 0, mjr = 0, mimj = 0, rTi = 0, rTj = 0, Tir[3] = {0, 0, 0}, Tjr[3] = {0, 0, 0};
        for (int k = 0; k < 3; k++) {
          mir += mi[k] * r[k], mjr += mj[k] * r[k], mimj += mi[k] * mj[k];
          for (int l = 0; l < 3; l++) Tir[k] += Ti[3 * k + l] * r[l], Tjr[k] += Tj[3 * k + l] * r[l];
        }
        for (int k = 0; k < 3; k++) rTi += r[k] * Tir[k], rTj += r[k] * Tjr[k];
        const double pr = q[j] * mir - q[i] * mjr;
        const double e = q[i] * q[j] / R + pr / R3 + mimj / R3 - 3 * mir * mjr / R5 + (q[i] * rTj + q[j] * rTi) / R5;
        const double w = KE * T * m;
        out.energy += w * e;
        if (!forces) continue;
        // de/dr (r = x_j - x_i)
        const double Mr = q[i] * rTj + q[j] * rTi;
        for (int k = 0; k < 3; k++) {
          const double p = q[j] * mi[k] - q[i] * mj[k];
          const double de = -q[i] * q[j] * r[k] / R3 + p / R3 - 3 * pr * r[k] / R5 - 3 * mimj * r[k] / R5 -
                            3 * (mjr * mi[k] + mir * mj[k]) / R5 + 15 * mir * mjr * r[k] / R7 +
                            2 * (q[i] * Tjr[k] + q[j] * Tir[k]) / R5 - 5 * Mr * r[k] / R7;
          const double v = w * de + KE * m * dT * e * r[k] / R;
          g[3 * j + k] += v, g[3 * i + k] -= v;
          Gm[3 * size_t(i) + k] += w * (q[j] * r[k] / R3 + mj[k] / R3 - 3 * mjr * r[k] / R5);
          Gm[3 * size_t(j) + k] += w * (-q[i] * r[k] / R3 + mi[k] / R3 - 3 * mir * r[k] / R5);
          for (int l = 0; l < 3; l++) {
            Hm[9 * size_t(i) + 3 * k + l] += w * q[j] * r[k] * r[l] / R5;
            Hm[9 * size_t(j) + 3 * k + l] += w * q[i] * r[k] * r[l] / R5;
          }
        }
        dq[i] += w * (q[j] / R - mjr / R3 + rTj / R5), dq[j] += w * (q[i] / R + mir / R3 + rTi / R5);
        W[size_t(a) * s + b] -= 0.5 * KE * e * T, W[size_t(b) * s + a] -= 0.5 * KE * e * T;
      }
    if (!forces) return out;
    lrc_moments_vjp(L, n, Z, x, Gm.data(), Hm.data(), g.data());
    lrc_clusters_vjp(L, c.cl, n, x, W, g.data());
    lrc_q0_vjp(L, S, n, Z, x, c.q0, dq.data(), g.data());
    for (size_t k = 0; k < g.size(); k++) out.forces[k] = -g[k];
    return out;
  }
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const int a = c.cl.sn[i], b = c.cl.sn[j];
      if (a == b) continue;
      const double r = dist(x, i, j);
      if (r <= r0) continue;
      const double t = (r - r0) / S.lr_t;
      const double T = sstep(t), dT = dsstep(t) / S.lr_t;
      const double m = 1.0 - c.cl.R[size_t(a) * s + b];
      if (m == 0) continue;
      const double k = KE / r, e = q[i] * q[j] * k;
      out.energy += e * T * m;
      if (!forces) continue;
      add_pair(g.data(), x, i, j, r, q[i] * q[j] * m * (dT * k - T * k / r));
      dq[i] += q[j] * k * T * m, dq[j] += q[i] * k * T * m;
      W[size_t(a) * s + b] -= 0.5 * e * T, W[size_t(b) * s + a] -= 0.5 * e * T;
    }
  if (!forces) return out;
  lrc_clusters_vjp(L, c.cl, n, x, W, g.data());
  lrc_q0_vjp(L, S, n, Z, x, c.q0, dq.data(), g.data());
  for (size_t k = 0; k < g.size(); k++) out.forces[k] = -g[k];
  return out;
}
} // namespace

LrcEnergy lrc_energy(const LrcSpec &L, const SqeSpec &S, const LrcCondition &c, int n, const int32_t *Z,
                     const double *x, double Q, bool forces) {
  LrcEnergy out;
  if (forces) out.forces.assign(3 * size_t(n), 0.0);
  if (c.cl.s == 1 || n == 0) return out; // one cluster: E*(gamma) - E*(gamma) = 0 exactly
  if (L.graph) return energy_fixed(L, S, c, n, Z, x, forces);
  SqeClusterMask m{c.cl.sn.data(), c.cl.s, c.cl.R.data()};
  const SqeOutput so = sqe_compute_masked(S, n, Z, x, Q, c.q0.q0.data(), forces, m);
  out.energy = so.energy, out.q = so.q, out.q0 = c.q0.q0;
  if (!forces) return out;
  std::vector<double> g(3 * size_t(n), 0.0);
  lrc_q0_vjp(L, S, n, Z, x, c.q0, so.dE_dq0.data(), g.data());
  lrc_clusters_vjp(L, c.cl, n, x, so.dE_dR, g.data());
  for (size_t k = 0; k < g.size(); k++) out.forces[k] = so.forces[k] - g[k];
  return out;
}

} // namespace mlipcpp::runtime
