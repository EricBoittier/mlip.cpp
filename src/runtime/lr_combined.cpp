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
  {
    const std::string cm = G.get_string("pet.plato_lr.lrc_cond", "cluster");
    if (cm != "cluster" && cm != "global" && cm != "neutral")
      throw std::runtime_error("lrc: lrc_cond must be cluster, global or neutral");
    L.cond_global = cm == "global", L.cond_neutral = cm == "neutral";
    scalar("cond_w", L.cond_w);
    if (G.get_int32("pet.plato_lr.lrc_qplateau", 0) != 0)
      throw std::runtime_error("lrc_qplateau was removed (it lost sum q = Q and double-counted partly linked partners); "
                               "use lrc_qproj");
    L.qproj = G.get_int32("pet.plato_lr.lrc_qproj", 0) != 0;
    scalar("qproj_w", L.qproj_w);
    if (L.qproj && !(L.qproj_w > 0 && L.qproj_w <= 1)) throw std::runtime_error("lrc: lrc_qproj_w must be in (0, 1]");
    if (!(L.cond_w > 0 && L.cond_w <= 1)) throw std::runtime_error("lrc: lrc_cond_w must be in (0, 1]");
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
    {
      const std::string fm = G.get_string("pet.plato_lr.lrc_fukui_mode", "element");
      if (fm == "element") L.fukui_mode = LrcSpec::FukuiMode::Element;
      else if (fm == "uniform") L.fukui_mode = LrcSpec::FukuiMode::Uniform;
      else if (fm == "atom") L.fukui_mode = LrcSpec::FukuiMode::Atom;
      else if (fm == "atom_pm") L.fukui_mode = LrcSpec::FukuiMode::AtomPM;
      else throw std::runtime_error("lrc graph: lrc_fukui_mode must be element, uniform, atom or atom_pm");
      const std::string sc = G.get_string("pet.plato_lr.lrc_fukui_scope", "global");
      if (sc != "global" && sc != "cluster") throw std::runtime_error("lrc graph: lrc_fukui_scope must be global or cluster");
      L.fukui_cluster = sc == "cluster";
      scalar("fukui_cluster_w", L.fukui_cluster_w), scalar("fukui_pm_width", L.fukui_pm_width);
      L.fmodel = G.get_string("pet.plato_lr.lrc_fmodel", "");
      L.fmodel_minus = G.get_string("pet.plato_lr.lrc_fmodel_minus", "");
      L.fukui_phi_minus = get_f64(G, "pet.plato_lr.lrc_fukui_phi_minus");
      L.fukui_check();
    }
    if (L.qmodel.empty()) throw std::runtime_error("lrc graph: needs lrc_qmodel");
    if (!(L.a_off > L.a_on)) throw std::runtime_error("lrc: bad a_on / a_off");
    L.mp = G.get_int32("pet.plato_lr.lrc_mp", 0) != 0;
    if (L.mp) {
      scalar("mp_rc", L.mp_rc), scalar("mp_width", L.mp_width);
      L.mp_centers = get_f64(G, "pet.plato_lr.lrc_mp_centers");
      for (double v : get_f64(G, "pet.plato_lr.lrc_mp_Z")) L.mp_Z.push_back(static_cast<int32_t>(std::lround(v)));
      L.mp_theta = get_f64(G, "pet.plato_lr.lrc_mp_theta"), L.mp_tau = get_f64(G, "pet.plato_lr.lrc_mp_tau");
      L.mp_build();
    }
    L.pol = G.get_int32("pet.plato_lr.lrc_pol", 0) != 0;
    if (L.pol) {
      for (double v : get_f64(G, "pet.plato_lr.lrc_pol_Z")) L.pol_Z.push_back(static_cast<int32_t>(std::lround(v)));
      L.pol_alpha = get_f64(G, "pet.plato_lr.lrc_pol_alpha");
      if (L.pol_Z.empty() || L.pol_Z.size() != L.pol_alpha.size())
        throw std::runtime_error("lrc pol: needs lrc_pol_Z and lrc_pol_alpha of equal length");
    }
    { // combo components (SPEC.md)
      const std::string ref = G.get_string("pet.plato_lr.lrc_q0_ref", "fukui");
      if (ref == "fukui") L.q0_ref = LrcSpec::Q0Ref::Fukui;
      else if (ref == "fermi") L.q0_ref = LrcSpec::Q0Ref::Fermi;
      else if (ref == "threshold") L.q0_ref = LrcSpec::Q0Ref::Threshold;
      else if (ref == "species_fill") L.q0_ref = LrcSpec::Q0Ref::SpeciesFill;
      else if (ref == "qeq") L.q0_ref = LrcSpec::Q0Ref::Qeq;
      else throw std::runtime_error("lrc graph: lrc_q0_ref must be fukui, fermi, threshold, species_fill or qeq");
      L.fermi_K = G.get_int32("pet.plato_lr.lrc_fermi_K", 3);
      scalar("fermi_T", L.fermi_T), scalar("smear", L.smear), scalar("flow_rc", L.flow_rc);
      L.smodel = G.get_string("pet.plato_lr.lrc_smodel", "");
      L.thr_theta = get_f64(G, "pet.plato_lr.lrc_thr_theta");
      L.sf_score = get_f64(G, "pet.plato_lr.lrc_sf_score"), L.sf_scale = get_f64(G, "pet.plato_lr.lrc_sf_scale");
      L.qeq_eta = get_f64(G, "pet.plato_lr.lrc_qeq_eta");
      L.flow_passes = G.get_int32("pet.plato_lr.lrc_flow_passes", 0);
      const std::string fk = G.get_string("pet.plato_lr.lrc_flow_kernel", "local");
      if (fk != "local" && fk != "coulomb") throw std::runtime_error("lrc graph: lrc_flow_kernel must be local or coulomb");
      L.flow_coulomb = fk == "coulomb";
      L.flow_kappa = get_f64(G, "pet.plato_lr.lrc_flow_kappa");
      L.readout = G.get_int32("pet.plato_lr.lrc_readout", 0) != 0;
      L.combo_check();
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
  gguf_set_val_str(g, "pet.plato_lr.lrc_cond", cond_global ? "global" : cond_neutral ? "neutral" : "cluster");
  scalar("cond_w", cond_w);
  gguf_set_val_i32(g, "pet.plato_lr.lrc_qproj", qproj ? 1 : 0);
  scalar("qproj_w", qproj_w);
  auto arr = [&](const char *k, const std::vector<double> &v) {
    gguf_set_arr_data(g, (std::string("pet.plato_lr.lrc_") + k).c_str(), GGUF_TYPE_FLOAT64, v.data(), v.size());
  };
  if (graph) {
    gguf_set_val_str(g, "pet.plato_lr.lrc_qmodel", qmodel.c_str());
    gguf_set_val_str(g, "pet.plato_lr.lrc_qpass_charge", qpass_neutral ? "zero" : "total");
    std::vector<double> z(fukui_Z.begin(), fukui_Z.end());
    if (!z.empty()) arr("fukui_Z", z), arr("fukui_phi", fukui_phi);
    if (fukui_mode != FukuiMode::Element || fukui_cluster) { // absent keys = element / global (bin4 GGUFs unchanged)
      static const char *names[] = {"element", "uniform", "atom", "atom_pm"};
      gguf_set_val_str(g, "pet.plato_lr.lrc_fukui_mode", names[static_cast<int>(fukui_mode)]);
      gguf_set_val_str(g, "pet.plato_lr.lrc_fukui_scope", fukui_cluster ? "cluster" : "global");
      scalar("fukui_cluster_w", fukui_cluster_w), scalar("fukui_pm_width", fukui_pm_width);
      if (!fmodel.empty()) gguf_set_val_str(g, "pet.plato_lr.lrc_fmodel", fmodel.c_str());
      if (!fmodel_minus.empty()) gguf_set_val_str(g, "pet.plato_lr.lrc_fmodel_minus", fmodel_minus.c_str());
      if (!fukui_phi_minus.empty()) arr("fukui_phi_minus", fukui_phi_minus);
    }
    if (mp) {
      gguf_set_val_i32(g, "pet.plato_lr.lrc_mp", 1);
      scalar("mp_rc", mp_rc), scalar("mp_width", mp_width);
      std::vector<double> mz(mp_Z.begin(), mp_Z.end());
      arr("mp_centers", mp_centers), arr("mp_Z", mz), arr("mp_theta", mp_theta), arr("mp_tau", mp_tau);
    }
    if (pol) {
      gguf_set_val_i32(g, "pet.plato_lr.lrc_pol", 1);
      std::vector<double> pz(pol_Z.begin(), pol_Z.end());
      arr("pol_Z", pz), arr("pol_alpha", pol_alpha);
    }
    if (q0_ref != Q0Ref::Fukui) { // absent = fukui (earlier GGUFs unchanged)
      static const char *refs[] = {"fukui", "fermi", "threshold", "species_fill", "qeq"};
      gguf_set_val_str(g, "pet.plato_lr.lrc_q0_ref", refs[static_cast<int>(q0_ref)]);
      gguf_set_val_i32(g, "pet.plato_lr.lrc_fermi_K", fermi_K);
      scalar("fermi_T", fermi_T), scalar("smear", smear);
      if (!smodel.empty()) gguf_set_val_str(g, "pet.plato_lr.lrc_smodel", smodel.c_str());
      if (!thr_theta.empty()) arr("thr_theta", thr_theta);
      if (!sf_score.empty()) arr("sf_score", sf_score), arr("sf_scale", sf_scale);
      if (!qeq_eta.empty()) arr("qeq_eta", qeq_eta);
    }
    if (flow_passes > 0) {
      gguf_set_val_i32(g, "pet.plato_lr.lrc_flow_passes", flow_passes);
      gguf_set_val_str(g, "pet.plato_lr.lrc_flow_kernel", flow_coulomb ? "coulomb" : "local");
      scalar("flow_rc", flow_rc), scalar("smear", smear), arr("flow_kappa", flow_kappa);
    }
    if (readout) gguf_set_val_i32(g, "pet.plato_lr.lrc_readout", 1);
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
  if (fukui_mode == FukuiMode::Uniform) return 1.0;
  for (size_t k = 0; k < fukui_Z.size(); k++)
    if (fukui_Z[k] == z) {
      const double p = fukui_phi[k];
      return (p > 30 ? p : std::log1p(std::exp(p))) + 1e-6;
    }
  throw std::runtime_error("lrc graph: no Fukui parameter for atomic number " + std::to_string(z));
}

double LrcSpec::fukui_phi_of(int z, bool minus) const {
  const std::vector<double> &phi = minus && !fukui_phi_minus.empty() ? fukui_phi_minus : fukui_phi;
  for (size_t k = 0; k < fukui_Z.size(); k++)
    if (fukui_Z[k] == z) return phi[k];
  throw std::runtime_error("lrc graph: no Fukui parameter for atomic number " + std::to_string(z));
}

void LrcSpec::fukui_check() const {
  if (fukui_mode != FukuiMode::Uniform && (fukui_Z.empty() || fukui_Z.size() != fukui_phi.size()))
    throw std::runtime_error("lrc graph: needs lrc_fukui_Z / _phi of equal length");
  if (!fukui_phi_minus.empty() && fukui_phi_minus.size() != fukui_Z.size())
    throw std::runtime_error("lrc graph: lrc_fukui_phi_minus must have one value per lrc_fukui_Z");
  if (fukui_atom() && fmodel.empty() && !fsrc) throw std::runtime_error("lrc graph: fukui_mode atom / atom_pm needs lrc_fmodel");
  if (fukui_mode == FukuiMode::AtomPM && fmodel_minus.empty() && !fsrc_minus)
    throw std::runtime_error("lrc graph: fukui_mode atom_pm needs lrc_fmodel_minus");
  if (fukui_mode == FukuiMode::AtomPM && fukui_cluster)
    throw std::runtime_error("lrc graph: fukui_scope cluster is implemented for uniform / element / atom weights only");
  if (!(fukui_pm_width >= 0)) throw std::runtime_error("lrc graph: lrc_fukui_pm_width must be >= 0");
  if (fukui_cluster && !(fukui_cluster_w > 0 && fukui_cluster_w <= 1))
    throw std::runtime_error("lrc graph: lrc_fukui_cluster_w must be in (0, 1]");
}

int LrcSpec::elem(int z) const {
  for (size_t k = 0; k < fukui_Z.size(); k++)
    if (fukui_Z[k] == z) return static_cast<int>(k);
  throw std::runtime_error("lrc graph: no combo parameters for atomic number " + std::to_string(z));
}

void LrcSpec::combo_check() const {
  const size_t E = fukui_Z.size();
  if (!combo() && !readout && !cond_neutral) return;
  if (combo() && fukui_cluster) throw std::runtime_error("lrc combo: the Fukui scope cluster (AC) cannot be combined with lrc_q0_ref / flows");
  if (!(fermi_K >= 1) || !(fermi_T > 0) || !(smear > 0)) throw std::runtime_error("lrc combo: needs lrc_fermi_K >= 1, lrc_fermi_T > 0, lrc_smear > 0");
  if ((q0_ref == Q0Ref::Fermi || q0_ref == Q0Ref::Threshold) && smodel.empty() && !ssrc)
    throw std::runtime_error("lrc combo: lrc_q0_ref fermi / threshold needs the score pass lrc_smodel");
  if (q0_ref == Q0Ref::Threshold && (thr_theta.size() != E || fukui_mode == FukuiMode::AtomPM))
    throw std::runtime_error("lrc combo: threshold needs lrc_thr_theta per lrc_fukui_Z and uniform / element / atom weights");
  if (q0_ref == Q0Ref::SpeciesFill && (sf_score.size() != E || sf_scale.size() != E))
    throw std::runtime_error("lrc combo: species_fill needs lrc_sf_score / lrc_sf_scale per lrc_fukui_Z");
  if (q0_ref == Q0Ref::Qeq && qeq_eta.size() != E) throw std::runtime_error("lrc combo: qeq needs lrc_qeq_eta per lrc_fukui_Z");
  if (flow_passes < 0 || (flow_passes > 0 && (flow_kappa.size() != E * E || !(flow_rc > 1))))
    throw std::runtime_error("lrc combo: flows need lrc_flow_kappa (E x E) and lrc_flow_rc > 1");
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
// graph mode (defined after the soft-cluster projection, which the Fukui scope "cluster" uses)
LrcQ0 q0_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, double Q, int spin);
void vjp_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, const LrcQ0 &o, const double *c, double *g);
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
    if (L.graph && L.needs_q_always()) c.q0 = lrc_q0(L, S, n, Z, x, Q, spin); // the readout's charges
    if (L.cond_neutral) c.Qloc[0] = 0.0;
  } else {
    c.q0 = lrc_q0(L, S, n, Z, x, Q, spin);
    for (int i = 0; i < n; i++) c.Q0[c.cl.sn[i]] += c.q0.q0[i];
    for (int a = 0; a < s; a++)
      for (int b = 0; b < s; b++) c.Qloc[a] += c.cl.R[size_t(a) * s + b] * c.Q0[b];
    if (L.cond_global) c.Qloc.assign(s, Q); // global conditioning: no position dependence (dS = 0)
    if (L.cond_neutral) c.Qloc.assign(s, 0.0); // neutral: every atom on the Q = 0 row (dS = 0)
  }
  for (int a = 0; a < s; a++) {
    double v = c.Qloc[a];
    const double rv = std::round(v);
    if (std::abs(v - rv) < 1e-9) v = rv; // solver noise on an integer
    const double k = std::floor(v), u = (v - k - 0.5) / L.cond_w + 0.5;
    c.k[a] = static_cast<int32_t>(k), c.S[a] = sstep(u), c.dS[a] = dsstep(u) / L.cond_w;
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
  if (s == 1 || L.cond_global || L.cond_neutral) return;
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
// one pair's electrostatics (units e, A; times KE outside), r = x_j - x_i:
//   e = [q_i q_j / R] + (q_j mu_i - q_i mu_j).r / R^3 + mu_i.mu_j / R^3 - 3 (mu_i.r)(mu_j.r) / R^5
//       + (q_i r.Theta_j.r + q_j r.Theta_i.r) / R^5          (Theta: Buckingham, traceless; mu / Theta may be null)
// and its derivatives in r, q_i, q_j, mu_i, mu_j, Theta_i, Theta_j
struct PairK {
  double e = 0, dr[3] = {0, 0, 0}, dqi = 0, dqj = 0, dmi[3] = {0, 0, 0}, dmj[3] = {0, 0, 0};
  double dTi[9] = {}, dTj[9] = {};
};
PairK pair_kernel(const double *r, double qi, double qj, const double *mi, const double *mj, const double *Ti,
                  const double *Tj, bool qq) {
  static const double z3[3] = {0, 0, 0}, z9[9] = {};
  if (!mi) mi = z3;
  if (!mj) mj = z3;
  if (!Ti) Ti = z9;
  if (!Tj) Tj = z9;
  PairK k;
  const double R2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2], R = std::sqrt(R2);
  const double R3 = R2 * R, R5 = R3 * R2, R7 = R5 * R2;
  double mir = 0, mjr = 0, mimj = 0, rTi = 0, rTj = 0, Tir[3] = {0, 0, 0}, Tjr[3] = {0, 0, 0};
  for (int a = 0; a < 3; a++) {
    mir += mi[a] * r[a], mjr += mj[a] * r[a], mimj += mi[a] * mj[a];
    for (int b = 0; b < 3; b++) Tir[a] += Ti[3 * a + b] * r[b], Tjr[a] += Tj[3 * a + b] * r[b];
  }
  for (int a = 0; a < 3; a++) rTi += r[a] * Tir[a], rTj += r[a] * Tjr[a];
  const double qf = qq ? 1.0 : 0.0, pr = qj * mir - qi * mjr, Mr = qi * rTj + qj * rTi;
  k.e = qf * qi * qj / R + pr / R3 + mimj / R3 - 3 * mir * mjr / R5 + Mr / R5;
  for (int a = 0; a < 3; a++) {
    const double p = qj * mi[a] - qi * mj[a];
    k.dr[a] = -qf * qi * qj * r[a] / R3 + p / R3 - 3 * pr * r[a] / R5 - 3 * mimj * r[a] / R5 -
              3 * (mjr * mi[a] + mir * mj[a]) / R5 + 15 * mir * mjr * r[a] / R7 + 2 * (qi * Tjr[a] + qj * Tir[a]) / R5 -
              5 * Mr * r[a] / R7;
    k.dmi[a] = qj * r[a] / R3 + mj[a] / R3 - 3 * mjr * r[a] / R5;
    k.dmj[a] = -qi * r[a] / R3 + mi[a] / R3 - 3 * mir * r[a] / R5;
    for (int b = 0; b < 3; b++) k.dTi[3 * a + b] = qj * r[a] * r[b] / R5, k.dTj[3 * a + b] = qi * r[a] * r[b] / R5;
  }
  k.dqi = qf * qj / R - mjr / R3 + rTj / R5, k.dqj = qf * qi / R + mir / R3 + rTi / R5;
  return k;
}

// atomic polarisability (A^3) of atomic number z (0 outside pol_Z)
double pol_alpha(const LrcSpec &L, int z) {
  for (size_t k = 0; k < L.pol_Z.size(); k++)
    if (L.pol_Z[k] == z) return L.pol_alpha[k];
  return 0.0;
}

// smootherstep and its first two derivatives
inline double sstep2(double x) { return x <= 0 || x >= 1 ? 0.0 : 60.0 * x * (2.0 * x * x - 3.0 * x + 1.0); }
// plateau value P(L), P'(L), P''(L) of width w
inline void plateau(double Lq, double w, double &P, double &dP, double &ddP) {
  const double k = std::floor(Lq), u = (Lq - k - 0.5) / w + 0.5;
  P = k + sstep(u), dP = dsstep(u) / w, ddP = sstep2(u) / (w * w);
}

struct LrcProjection {
  std::vector<double> q, s, dpb, g, F, Q0, f;
  double S = 0, G = 0, T = 0, w = 0.6;
  bool resid = false;
};
constexpr double kProjEps = 1e-3;

// visit every group of every link state: cb(prob, group nodes, edges of the component, state, pr, su) (fallback and
// uncovered super-nodes: one state of probability 1, pr / su empty)
template <class CB> void proj_groups(const LrcSpec &L, const LrcClusters &c, CB &&cb) {
  std::vector<char> covered(c.s, 0);
  static const std::vector<double> none;
  for (const auto &edges : weak_components(c)) {
    std::vector<int> nodes;
    for (int k : edges) nodes.push_back(c.se[k].first), nodes.push_back(c.se[k].second);
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    for (int a : nodes) covered[a] = 1;
    if (int(edges.size()) > L.max_super_edges) { // fallback: the component is one cluster
      cb(1.0, nodes, edges, uint64_t(0), none, none, true);
      continue;
    }
    enumerate(c, edges, [&](uint64_t st, double P, const std::vector<int> &p, const std::vector<int> &nd,
                            const std::vector<double> &pr, const std::vector<double> &su) {
      std::vector<int> roots;
      for (int a : nd) roots.push_back(p[a]);
      std::sort(roots.begin(), roots.end());
      roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
      for (int r : roots) {
        std::vector<int> grp;
        for (int a : nd)
          if (p[a] == r) grp.push_back(a);
        cb(P, grp, edges, st, pr, su, false);
      }
    });
  }
  for (int a = 0; a < c.s; a++)
    if (!covered[a]) cb(1.0, std::vector<int>{a}, std::vector<int>{}, uint64_t(0), none, none, true);
}

// per-atom shares f (> 0), plateau width w; resid: also route the residual Q - sum q (Fukui scope "cluster"), else
// (lrc_qproj) sum q' = sum q:
//   q'_i = q_i + f_i / F_a (s_a + T g_a / G),  T = (resid ? Q - sum q : 0) - S
LrcProjection lrc_project(const LrcSpec &L, const LrcClusters &c, int n, const double *f, const double *q, double w,
                          bool resid, double Q) {
  LrcProjection o;
  const int s = c.s;
  o.f.assign(f, f + n), o.w = w, o.resid = resid;
  o.Q0.assign(s, 0.0), o.F.assign(s, 0.0), o.s.assign(s, 0.0), o.dpb.assign(s, 0.0), o.g.assign(s, 0.0);
  double sq = 0;
  for (int i = 0; i < n; i++) o.Q0[c.sn[i]] += q[i], o.F[c.sn[i]] += f[i], sq += q[i];
  proj_groups(L, c, [&](double pr, const std::vector<int> &grp, const std::vector<int> &, uint64_t,
                        const std::vector<double> &, const std::vector<double> &, bool) {
    double QG = 0, FG = 0, P, dP, ddP;
    for (int a : grp) QG += o.Q0[a], FG += o.F[a];
    plateau(QG, w, P, dP, ddP);
    for (int a : grp) o.s[a] += pr * (P - QG) * o.F[a] / FG, o.dpb[a] += pr * dP;
  });
  for (int a = 0; a < s; a++) o.g[a] = o.F[a] * (kProjEps + o.dpb[a]), o.S += o.s[a], o.G += o.g[a];
  o.T = (resid ? Q - sq : 0.0) - o.S;
  o.q.assign(q, q + n);
  for (int i = 0; i < n; i++) {
    const int a = c.sn[i];
    o.q[i] += f[i] / o.F[a] * (o.s[a] + o.T * o.g[a] / o.G);
  }
  return o;
}
// element shares (lrc_qproj on the long-range charges)
LrcProjection lrc_project(const LrcSpec &L, const LrcClusters &c, int n, const int32_t *Z, const double *q) {
  std::vector<double> f(n);
  for (int i = 0; i < n; i++) f[i] = L.fukui(Z[i]);
  return lrc_project(L, c, n, f.data(), q, L.qproj_w, false, 0.0);
}

// dq: dE/dq' on input, dE/dq on output (q the charge-pass charges); g [3n] += the link-probability part; df (optional,
// [n]) += dE/df_i
void lrc_project_vjp(const LrcSpec &L, const LrcClusters &c, const LrcProjection &o, int n, const double *x,
                     std::vector<double> &dq, double *g, double *df = nullptr) {
  const int s = c.s;
  const double *f = o.f.data();
  std::vector<double> h(s, 0.0), Ds(s), Dg(s), dQ0(s, 0.0), dp(c.se.size(), 0.0), dF(s, 0.0), t(s);
  for (int i = 0; i < n; i++) h[c.sn[i]] += dq[i] * f[i] / o.F[c.sn[i]];
  double Hb = 0;
  for (int a = 0; a < s; a++) Hb += h[a] * o.g[a];
  Hb /= o.G;
  for (int a = 0; a < s; a++) Ds[a] = h[a] - Hb, Dg[a] = (o.T / o.G) * (h[a] - Hb), t[a] = o.s[a] + o.T * o.g[a] / o.G;
  proj_groups(L, c, [&](double pr, const std::vector<int> &grp, const std::vector<int> &edges, uint64_t st,
                        const std::vector<double> &prk, const std::vector<double> &suk, bool single) {
    double QG = 0, FG = 0, P, dP, ddP, A = 0, B = 0;
    for (int a : grp) QG += o.Q0[a], FG += o.F[a];
    plateau(QG, o.w, P, dP, ddP);
    for (int a : grp) A += Ds[a] * o.F[a] / FG, B += Dg[a] * o.F[a];
    for (int a : grp) dQ0[a] += pr * (A * (dP - 1.0) + B * ddP);
    if (df)
      for (int a : grp) dF[a] += pr * (P - QG) * (Ds[a] - A) / FG;
    if (single) return;
    const double V = A * (P - QG) + B * dP; // this group's term per unit state probability
    const int m = static_cast<int>(edges.size());
    for (int k = 0; k < m; k++) dp[edges[k]] += V * prk[k] * suk[k + 1] * (((st >> k) & 1) ? 1.0 : -1.0);
  });
  if (df) {
    for (int a = 0; a < s; a++) dF[a] += Dg[a] * (kProjEps + o.dpb[a]) - h[a] * t[a] / o.F[a];
    for (int i = 0; i < n; i++) {
      const int a = c.sn[i];
      df[i] += dq[i] * t[a] / o.F[a] + dF[a];
    }
  }
  for (int i = 0; i < n; i++) dq[i] += dQ0[c.sn[i]] - (o.resid ? Hb : 0.0);
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

// graph mode Fukui weights of one channel: f_i = softplus(phi_Z + a_i) + 1e-6 (a = 0 without a pass), df = df/da
void fukui_channel(const LrcSpec &L, int n, const int32_t *Z, const double *x, double Qp, int spin, bool minus,
                   std::vector<double> &f, std::vector<double> &df) {
  f.assign(n, 1.0), df.assign(n, 0.0);
  if (L.fukui_mode == LrcSpec::FukuiMode::Uniform) return;
  std::vector<double> a(n, 0.0);
  LrcQSource *src = minus ? L.fsrc_minus.get() : L.fsrc.get();
  if (L.fukui_atom()) {
    if (!src) throw std::runtime_error("lrc graph: the Fukui pass (lrc_fmodel" + std::string(minus ? "_minus" : "") +
                                       ") is not loaded");
    a = src->raw(n, Z, x, Qp, spin);
  }
  for (int i = 0; i < n; i++) {
    const double u = L.fukui_phi_of(Z[i], minus) + a[i];
    f[i] = (u > 30 ? u : std::log1p(std::exp(u))) + 1e-6, df[i] = sig(u);
  }
}

// graph mode: q0 = q_raw + w (Q - sum q_raw), w = f / sum f (A+-: blend of f+ / F+ and f- / F- by the residual's sign);
// Fukui scope "cluster": the residual-routing soft-cluster projection of q_raw instead
LrcQ0 q0_graph_fukui(const LrcSpec &L, int n, const int32_t *Z, const double *x, double Q, int spin) {
  if (!L.qsrc) throw std::runtime_error("lrc graph: the charge pass (lrc_qmodel) is not loaded");
  LrcQ0 o;
  o.Q = Q, o.spin = spin;
  if (n == 0) return o;
  const double Qp = L.qpass_neutral ? 0.0 : Q;
  o.qraw = L.qsrc->raw(n, Z, x, Qp, spin);
  o.q0 = o.qraw;
  fukui_channel(L, n, Z, x, Qp, spin, false, o.fp, o.dfp);
  double sq = 0;
  for (int i = 0; i < n; i++) sq += o.qraw[i], o.Fp += o.fp[i];
  o.r = Q - sq;
  if (L.fukui_cluster) {
    const LrcClusters c = lrc_clusters(L, n, x);
    o.q0 = lrc_project(L, c, n, o.fp.data(), o.qraw.data(), L.fukui_cluster_w, true, Q).q;
    return o;
  }
  o.wt.resize(n);
  if (L.fukui_mode == LrcSpec::FukuiMode::AtomPM) {
    fukui_channel(L, n, Z, x, Qp, spin, true, o.fm, o.dfm);
    for (int i = 0; i < n; i++) o.Fm += o.fm[i];
    const double d = L.fukui_pm_width;
    if (d > 0) o.s = sstep(o.r / d + 0.5), o.ds = dsstep(o.r / d + 0.5) / d;
    else o.s = o.r > 0 ? 1.0 : 0.0, o.ds = 0.0;
    for (int i = 0; i < n; i++) o.wt[i] = (1.0 - o.s) * o.fp[i] / o.Fp + o.s * o.fm[i] / o.Fm;
  } else {
    for (int i = 0; i < n; i++) o.wt[i] = o.fp[i] / o.Fp;
  }
  for (int i = 0; i < n; i++) o.q0[i] += o.wt[i] * o.r;
  return o;
}

// dE/dx of sum_i c_i q0_i: the raw-charge pass with weights c_j - dE/dr, the Fukui pass(es) with r d(f / F)/da
void vjp_graph_fukui(const LrcSpec &L, int n, const int32_t *Z, const double *x, const LrcQ0 &o, const double *c,
                     double *g) {
  const double Qp = L.qpass_neutral ? 0.0 : o.Q;
  std::vector<double> w(n), u(n, 0.0);
  bool any = false;
  auto fpass = [&](LrcQSource *src, const std::vector<double> &df) { // u: dE/df_i
    bool nz = false;
    for (int i = 0; i < n; i++) u[i] *= df[i], nz |= u[i] != 0;
    if (nz && src) src->raw_vjp(n, Z, x, Qp, o.spin, u.data(), g);
  };
  if (L.fukui_cluster) {
    const LrcClusters cl = lrc_clusters(L, n, x);
    const LrcProjection pr = lrc_project(L, cl, n, o.fp.data(), o.qraw.data(), L.fukui_cluster_w, true, o.Q);
    std::vector<double> dq(c, c + n);
    lrc_project_vjp(L, cl, pr, n, x, dq, g, L.fukui_atom() ? u.data() : nullptr);
    for (int i = 0; i < n; i++) any |= dq[i] != 0;
    if (any) L.qsrc->raw_vjp(n, Z, x, Qp, o.spin, dq.data(), g);
    if (L.fukui_atom()) fpass(L.fsrc.get(), o.dfp);
    return;
  }
  const bool pm = L.fukui_mode == LrcSpec::FukuiMode::AtomPM;
  // dq0_i/dr = w_i + r dw_i/dr, dw_i/dr = (f-_i / F- - f+_i / F+) ds/dr
  double cr = 0, cp = 0, cm = 0;
  for (int i = 0; i < n; i++) {
    cr += c[i] * o.wt[i];
    if (L.fukui_atom()) cp += c[i] * o.fp[i] / o.Fp;
    if (pm) cm += c[i] * o.fm[i] / o.Fm, cr += c[i] * o.r * (o.fm[i] / o.Fm - o.fp[i] / o.Fp) * o.ds;
  }
  for (int i = 0; i < n; i++) w[i] = c[i] - cr, any |= w[i] != 0;
  if (any) L.qsrc->raw_vjp(n, Z, x, Qp, o.spin, w.data(), g);
  if (!L.fukui_atom() || o.r == 0) return;
  // sum_i c_i r (1 - s) d(f+_i / F+) = r (1 - s) / F+ sum_j (c_j - cp) df+_j
  for (int i = 0; i < n; i++) u[i] = o.r * (1.0 - o.s) / o.Fp * (c[i] - cp);
  fpass(L.fsrc.get(), o.dfp);
  if (!pm) return;
  for (int i = 0; i < n; i++) u[i] = o.r * o.s / o.Fm * (c[i] - cm);
  fpass(L.fsrc_minus.get(), o.dfm);
}

// ---- combo components (feat/plato-lr-combo; after M. Langer, ipsum/PET-LR; combo/SPEC.md) ----------------------------
constexpr double kKE = 14.399645;
inline double softplus(double u) { return u > 30 ? u : std::log1p(std::exp(u)); }

// staircase occupation n(x) = sum_{k=-K}^{K-1} sigmoid((x - k - 1/2) / T) - K and n'(x)
inline void stair(int K, double T, double x, double &v, double &d) {
  v = -K, d = 0;
  for (int k = -K; k < K; k++) {
    const double sg = sig((x - k - 0.5) / T);
    v += sg, d += sg * (1.0 - sg) / T;
  }
}

// mu with sum_i c_i n(s_i - mu) = target (c = 1 when null): bisection, then safeguarded Newton to 1e-13
double solve_mu(const LrcSpec &L, const std::vector<double> &s, const std::vector<double> *c, double target) {
  const int n = static_cast<int>(s.size());
  double cs = 0;
  for (int i = 0; i < n; i++) cs += c ? (*c)[i] : 1.0;
  if (!(std::abs(target) < L.fermi_K * cs - 1e-9))
    throw std::runtime_error("lrc combo: the residual " + std::to_string(target) + " is outside the fill's range");
  auto res = [&](double mu, double &dres) {
    double v = 0;
    dres = 0;
    for (int i = 0; i < n; i++) {
      double o, d;
      stair(L.fermi_K, L.fermi_T, s[i] - mu, o, d);
      const double w = c ? (*c)[i] : 1.0;
      v += w * o, dres -= w * d;
    }
    return v - target;
  };
  double lo = *std::min_element(s.begin(), s.end()) - L.fermi_K - 1 - 40 * L.fermi_T;
  double hi = *std::max_element(s.begin(), s.end()) + L.fermi_K + 1 + 40 * L.fermi_T, d;
  for (int it = 0; it < 200 && hi - lo > 1e-12; it++) {
    const double mid = 0.5 * (lo + hi);
    (res(mid, d) > 0 ? lo : hi) = mid; // residual decreases with mu
  }
  double mu = 0.5 * (lo + hi);
  for (int it = 0; it < 60; it++) { // Newton to round-off (the FD tests see a root error / h)
    const double v = res(mu, d);
    if (v == 0 || d == 0) break;
    const double nm = mu - v / d;
    if (!(nm > lo - 1e-9 && nm < hi + 1e-9)) break;
    const double step = std::abs(nm - mu);
    mu = nm;
    if (step < 1e-15 * std::max(1.0, std::abs(mu))) break;
  }
  return mu;
}

// Gaussian-charge kernel K(r) = KE erf(r / sqrt2 sigma) / r and dK/dr (KE = 1: the coulomb flow weight)
inline void gkernel(double sigma, double r, double ke, double &k, double &dk) {
  const double a = 1.0 / (std::sqrt(2.0) * sigma), er = std::erf(a * r);
  k = ke * er / r;
  dk = ke * (2.0 * a / std::sqrt(M_PI) * std::exp(-a * a * r * r) / r - er / (r * r));
}
// flow coupling kappa_ij(r) and d kappa / dr
inline void flow_kappa(const LrcSpec &L, int ei, int ej, double r, double &kap, double &dkap) {
  const size_t E = L.fukui_Z.size();
  const double sp = softplus(0.5 * (L.flow_kappa[size_t(ei) * E + ej] + L.flow_kappa[size_t(ej) * E + ei]));
  double w, dw;
  if (L.flow_coulomb) gkernel(L.smear, r, 1.0, w, dw);
  else {
    const double t = r - L.flow_rc + 1.0;
    w = 1.0 - sstep(t), dw = -dsstep(t);
  }
  kap = sp * w, dkap = sp * dw;
}

// dense Cholesky of an SPD matrix (in place, lower) and the solve
void chol(std::vector<double> &A, int n) {
  for (int j = 0; j < n; j++) {
    double d = A[size_t(j) * n + j];
    for (int k = 0; k < j; k++) d -= A[size_t(j) * n + k] * A[size_t(j) * n + k];
    if (!(d > 0)) throw std::runtime_error("lrc combo qeq: J + K is not positive definite");
    A[size_t(j) * n + j] = std::sqrt(d);
    for (int i = j + 1; i < n; i++) {
      double v = A[size_t(i) * n + j];
      for (int k = 0; k < j; k++) v -= A[size_t(i) * n + k] * A[size_t(j) * n + k];
      A[size_t(i) * n + j] = v / A[size_t(j) * n + j];
    }
  }
}
std::vector<double> chol_solve(const std::vector<double> &Lc, int n, std::vector<double> b) {
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < i; k++) b[i] -= Lc[size_t(i) * n + k] * b[k];
    b[i] /= Lc[size_t(i) * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    for (int k = i + 1; k < n; k++) b[i] -= Lc[size_t(k) * n + i] * b[k];
    b[i] /= Lc[size_t(i) * n + i];
  }
  return b;
}
// QEq matrix A = diag(J) + K (Cholesky factor) and J
void qeq_matrix(const LrcSpec &L, int n, const int32_t *Z, const double *x, std::vector<double> &Lc, std::vector<double> &J) {
  if (n > 2000) throw std::runtime_error("lrc combo qeq: dense solve limited to 2000 atoms");
  Lc.assign(size_t(n) * n, 0.0), J.resize(n);
  const double self = kKE * std::sqrt(2.0 / M_PI) / L.smear;
  for (int i = 0; i < n; i++) {
    J[i] = softplus(L.qeq_eta[L.elem(Z[i])]);
    Lc[size_t(i) * n + i] = J[i] + self;
    for (int j = 0; j < i; j++) {
      double k, dk;
      gkernel(L.smear, dist(x, i, j), kKE, k, dk);
      Lc[size_t(i) * n + j] = Lc[size_t(j) * n + i] = k;
    }
  }
  chol(Lc, n);
}

// flow pass on q: dq (out), phi (out)
void flow_step(const LrcSpec &L, int n, const int32_t *Z, const double *x, const std::vector<double> &q,
               std::vector<double> &dq, std::vector<double> &phi) {
  phi.assign(n, 0.0), dq.assign(n, 0.0);
  std::vector<int> e(n);
  for (int i = 0; i < n; i++) e[i] = L.elem(Z[i]);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < i; j++) {
      double k, dk;
      gkernel(L.smear, dist(x, i, j), kKE, k, dk);
      phi[i] += k * q[j], phi[j] += k * q[i];
    }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < i; j++) {
      const double r = dist(x, i, j);
      if (!L.flow_coulomb && r >= L.flow_rc) continue;
      double kap, dkap;
      flow_kappa(L, e[i], e[j], r, kap, dkap);
      const double t = kap * (phi[j] - phi[i]);
      dq[i] += t, dq[j] -= t;
    }
}

LrcQ0 q0_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, double Q, int spin) {
  if (!L.combo()) return q0_graph_fukui(L, n, Z, x, Q, spin);
  LrcQ0 o;
  if (L.q0_ref == LrcSpec::Q0Ref::Fukui) o = q0_graph_fukui(L, n, Z, x, Q, spin);
  else {
    if (!L.qsrc) throw std::runtime_error("lrc graph: the charge pass (lrc_qmodel) is not loaded");
    o.Q = Q, o.spin = spin;
    if (n == 0) return o;
    const double Qp = L.qpass_neutral ? 0.0 : Q, B = L.fermi_K + 2.0;
    o.qraw = L.qsrc->raw(n, Z, x, Qp, spin);
    double sq = 0;
    for (int i = 0; i < n; i++) sq += o.qraw[i];
    o.r = Q - sq, o.q0 = o.qraw;
    o.occ.assign(n, 0.0), o.docc.assign(n, 0.0);
    auto scores = [&]() {
      if (!L.ssrc) throw std::runtime_error("lrc combo: the score pass (lrc_smodel) is not loaded");
      const std::vector<double> sr = L.ssrc->raw(n, Z, x, Qp, spin);
      o.score.resize(n);
      for (int i = 0; i < n; i++) o.score[i] = B * std::tanh(sr[i] / B);
    };
    switch (L.q0_ref) {
    case LrcSpec::Q0Ref::Fermi: {
      scores();
      o.mu = solve_mu(L, o.score, nullptr, o.r);
      for (int i = 0; i < n; i++) stair(L.fermi_K, L.fermi_T, o.score[i] - o.mu, o.occ[i], o.docc[i]), o.M += o.docc[i];
      for (int i = 0; i < n; i++) o.q0[i] += o.occ[i];
      break;
    }
    case LrcSpec::Q0Ref::Threshold: {
      scores();
      fukui_channel(L, n, Z, x, Qp, spin, false, o.fp, o.dfp);
      for (int i = 0; i < n; i++) {
        stair(L.fermi_K, L.fermi_T, o.score[i] - L.thr_theta[L.elem(Z[i])], o.occ[i], o.docc[i]);
        o.M += o.docc[i], o.sum_p += o.occ[i], o.Fp += o.fp[i];
      }
      o.wt.resize(n);
      for (int i = 0; i < n; i++) o.wt[i] = o.fp[i] / o.Fp, o.q0[i] += o.occ[i] + o.wt[i] * (o.r - o.sum_p);
      break;
    }
    case LrcSpec::Q0Ref::SpeciesFill: {
      o.score.resize(n), o.fp.resize(n); // fp: c_Z
      for (int i = 0; i < n; i++) {
        const int e = L.elem(Z[i]);
        o.score[i] = B * std::tanh(L.sf_score[e] / B), o.fp[i] = softplus(L.sf_scale[e]);
      }
      o.mu = solve_mu(L, o.score, &o.fp, o.r);
      for (int i = 0; i < n; i++) {
        stair(L.fermi_K, L.fermi_T, o.score[i] - o.mu, o.occ[i], o.docc[i]);
        o.M += o.fp[i] * o.docc[i], o.q0[i] += o.fp[i] * o.occ[i];
      }
      break;
    }
    case LrcSpec::Q0Ref::Qeq: {
      std::vector<double> Lc, J;
      qeq_matrix(L, n, Z, x, Lc, J);
      std::vector<double> b(n), one(n, 1.0);
      for (int i = 0; i < n; i++) b[i] = J[i] * o.qraw[i];
      const std::vector<double> y = chol_solve(Lc, n, b), z = chol_solve(Lc, n, one);
      double sy = 0, sz = 0;
      for (int i = 0; i < n; i++) sy += y[i], sz += z[i];
      const double lam = (Q - sy) / sz;
      for (int i = 0; i < n; i++) o.q0[i] = y[i] + lam * z[i];
      o.fp = J;
      break;
    }
    default: break;
    }
  }
  o.qref = o.q0;
  o.qflow.clear();
  for (int p = 0; p < L.flow_passes; p++) {
    std::vector<double> dq, phi;
    o.qflow.push_back(o.q0);
    flow_step(L, n, Z, x, o.q0, dq, phi);
    for (int i = 0; i < n; i++) o.q0[i] += dq[i];
  }
  return o;
}

// dE/dx of sum_i c_i q_i through the flows: c (in: dE/dq after the flows, out: dE/dq of the reference), g += positions
void flows_vjp(const LrcSpec &L, int n, const int32_t *Z, const double *x, const LrcQ0 &o, std::vector<double> &c,
               double *g) {
  std::vector<int> e(n);
  for (int i = 0; i < n; i++) e[i] = L.elem(Z[i]);
  for (int p = static_cast<int>(o.qflow.size()) - 1; p >= 0; p--) {
    const std::vector<double> &q = o.qflow[p];
    std::vector<double> dq, phi, psi(n, 0.0);
    flow_step(L, n, Z, x, q, dq, phi);
    // psi_j = dE/dphi_j = sum_i kappa_ij (c_i - c_j); the kappa(r) part of the position gradient
    for (int i = 0; i < n; i++)
      for (int j = 0; j < i; j++) {
        const double r = dist(x, i, j);
        if (!L.flow_coulomb && r >= L.flow_rc) continue;
        double kap, dkap;
        flow_kappa(L, e[i], e[j], r, kap, dkap);
        psi[j] += kap * (c[i] - c[j]), psi[i] += kap * (c[j] - c[i]);
        add_pair(g, x, i, j, r, (c[i] - c[j]) * (phi[j] - phi[i]) * dkap);
      }
    // phi = K0 q: dE/dq += K0 psi; the K(r) part
    std::vector<double> cn(c);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < i; j++) {
        const double r = dist(x, i, j);
        double k, dk;
        gkernel(L.smear, r, kKE, k, dk);
        cn[i] += k * psi[j], cn[j] += k * psi[i];
        add_pair(g, x, i, j, r, (psi[i] * q[j] + psi[j] * q[i]) * dk);
      }
    c = cn;
  }
}

void vjp_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, const LrcQ0 &o, const double *c0, double *g) {
  if (!L.combo()) return vjp_graph_fukui(L, n, Z, x, o, c0, g);
  if (n == 0) return;
  std::vector<double> c(c0, c0 + n);
  if (!o.qflow.empty()) flows_vjp(L, n, Z, x, o, c, g);
  if (L.q0_ref == LrcSpec::Q0Ref::Fukui) return vjp_graph_fukui(L, n, Z, x, o, c.data(), g);
  const double Qp = L.qpass_neutral ? 0.0 : o.Q, B = L.fermi_K + 2.0;
  std::vector<double> w(n), u(n, 0.0);
  double cb = 0;
  auto raw = [&]() {
    bool any = false;
    for (int i = 0; i < n; i++) any |= w[i] != 0;
    if (any) L.qsrc->raw_vjp(n, Z, x, Qp, o.spin, w.data(), g);
  };
  auto score = [&]() { // dE/ds_raw_j = n'_j (c_j - cb) (1 - (s_j / B)^2)
    bool any = false;
    for (int j = 0; j < n; j++) u[j] = o.docc[j] * (c[j] - cb) * (1.0 - (o.score[j] / B) * (o.score[j] / B)), any |= u[j] != 0;
    if (any) L.ssrc->raw_vjp(n, Z, x, Qp, o.spin, u.data(), g);
  };
  switch (L.q0_ref) {
  case LrcSpec::Q0Ref::Fermi:
    for (int i = 0; i < n; i++) cb += c[i] * o.docc[i];
    cb /= std::max(o.M, 1e-300);
    for (int i = 0; i < n; i++) w[i] = c[i] - cb;
    raw(), score();
    break;
  case LrcSpec::Q0Ref::Threshold: {
    for (int i = 0; i < n; i++) cb += c[i] * o.wt[i];
    for (int i = 0; i < n; i++) w[i] = c[i] - cb;
    raw(), score();
    const double rr = o.r - o.sum_p;
    if (L.fukui_atom() && rr != 0 && L.fsrc) { // r' d(f_i / F)/da, as the Fukui atom mode
      for (int i = 0; i < n; i++) u[i] = rr / o.Fp * (c[i] - cb) * o.dfp[i];
      L.fsrc->raw_vjp(n, Z, x, Qp, o.spin, u.data(), g);
    }
    break;
  }
  case LrcSpec::Q0Ref::SpeciesFill:
    for (int i = 0; i < n; i++) cb += c[i] * o.fp[i] * o.docc[i];
    cb /= std::max(o.M, 1e-300);
    for (int i = 0; i < n; i++) w[i] = c[i] - cb;
    raw();
    break;
  case LrcSpec::Q0Ref::Qeq: {
    std::vector<double> Lc, J, one(n, 1.0);
    qeq_matrix(L, n, Z, x, Lc, J);
    std::vector<double> v = chol_solve(Lc, n, c), z = chol_solve(Lc, n, one);
    double sv = 0, sz = 0;
    for (int i = 0; i < n; i++) sv += v[i], sz += z[i];
    for (int i = 0; i < n; i++) v[i] -= sv / sz * z[i];
    const std::vector<double> &q = o.qref;
    for (int i = 0; i < n; i++) {
      w[i] = J[i] * v[i];
      for (int j = 0; j < i; j++) {
        const double r = dist(x, i, j);
        double k, dk;
        gkernel(L.smear, r, kKE, k, dk);
        add_pair(g, x, i, j, r, -(v[i] * q[j] + v[j] * q[i]) * dk);
      }
    }
    raw();
    break;
  }
  default: break;
  }
}

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
  LrcProjection proj;
  std::vector<double> qp;
  const bool qproj = L.qproj && !L.fukui_cluster; // AC: the reference charges are already projected
  if (qproj) {
    proj = lrc_project(L, c.cl, n, Z, q);
    qp = proj.q, q = qp.data(), out.q = qp;
  }
  // dE/dq' (dq) -> dE/dq of the charge pass, and the link-state part of the projection into the position gradient
  auto unplateau = [&]() {
    if (qproj) lrc_project_vjp(L, c.cl, proj, n, x, dq, g.data());
  };
  // the switched inter-cluster pairs: (i, j, R, T, dT/dR, m = 1 - R_ab)
  struct P {
    int i, j;
    double R, T, dT, m;
  };
  std::vector<P> pairs;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const int a = c.cl.sn[i], b = c.cl.sn[j];
      if (a == b) continue;
      const double R = dist(x, i, j);
      if (R <= r0) continue;
      const double m = 1.0 - c.cl.R[size_t(a) * s + b];
      if (m == 0) continue;
      const double t = (R - r0) / S.lr_t;
      pairs.push_back({i, j, R, sstep(t), dsstep(t) / S.lr_t, m});
    }
  if (L.mp || L.pol) {
    std::vector<double> mu, th, Gm(3 * size_t(n), 0.0), Hm(9 * size_t(n), 0.0);
    if (L.mp) lrc_moments(L, n, Z, x, mu, th);
    // add one kernel's energy (weight scale) and gradients (fixed mu / Theta for the induced part: Hellmann-Feynman)
    auto accumulate = [&](const P &p, const PairK &k, double scale, bool moment_grads, bool energy) {
      const double w = KE * p.T * p.m * scale;
      if (energy) out.energy += w * k.e;
      if (!forces) return;
      for (int a = 0; a < 3; a++) {
        const double v = w * k.dr[a] + KE * scale * p.m * p.dT * k.e * (x[3 * p.j + a] - x[3 * p.i + a]) / p.R;
        g[3 * p.j + a] += v, g[3 * p.i + a] -= v;
        if (moment_grads) {
          Gm[3 * size_t(p.i) + a] += w * k.dmi[a], Gm[3 * size_t(p.j) + a] += w * k.dmj[a];
          for (int b = 0; b < 3; b++)
            Hm[9 * size_t(p.i) + 3 * a + b] += w * k.dTi[3 * a + b], Hm[9 * size_t(p.j) + 3 * a + b] += w * k.dTj[3 * a + b];
        }
      }
      dq[p.i] += w * k.dqi, dq[p.j] += w * k.dqj;
      const int a = c.cl.sn[p.i], b = c.cl.sn[p.j];
      W[size_t(a) * s + b] -= 0.5 * KE * scale * k.e * p.T, W[size_t(b) * s + a] -= 0.5 * KE * scale * k.e * p.T;
    };
    for (const P &p : pairs) {
      double r[3];
      for (int a = 0; a < 3; a++) r[a] = x[3 * p.j + a] - x[3 * p.i + a];
      const double *mi = L.mp ? &mu[3 * size_t(p.i)] : nullptr, *mj = L.mp ? &mu[3 * size_t(p.j)] : nullptr;
      const double *Ti = L.mp ? &th[9 * size_t(p.i)] : nullptr, *Tj = L.mp ? &th[9 * size_t(p.j)] : nullptr;
      accumulate(p, pair_kernel(r, q[p.i], q[p.j], mi, mj, Ti, Tj, true), 1.0, L.mp, true);
    }
    if (L.pol) {
      // induced atomic dipoles (inter-cluster only, additive isotropic alpha): (alpha^-1 - T) mu = F0, F0_i the switched
      // field of the other clusters' charges; E = -1/2 KE mu . F0 (Hellmann-Feynman forces: the gradient at fixed mu of
      // KE sum_pairs w [charge-dipole + dipole-dipole], the same kernel without q q)
      std::vector<int> id(n, -1), at;
      for (const P &p : pairs)
        for (int k : {p.i, p.j})
          if (id[k] < 0 && pol_alpha(L, Z[k]) > 0) id[k] = static_cast<int>(at.size()), at.push_back(k);
      const int m3 = 3 * static_cast<int>(at.size());
      if (m3 > 0) {
        std::vector<double> A(size_t(m3) * m3, 0.0), F0(m3, 0.0);
        for (size_t k = 0; k < at.size(); k++)
          for (int a = 0; a < 3; a++) A[(3 * k + a) * m3 + 3 * k + a] = 1.0 / pol_alpha(L, Z[at[k]]);
        for (const P &p : pairs) {
          const double w = p.T * p.m;
          double d[3];
          for (int a = 0; a < 3; a++) d[a] = x[3 * p.i + a] - x[3 * p.j + a]; // x_i - x_j
          const double R3 = p.R * p.R * p.R, R5 = R3 * p.R * p.R;
          const int ki = id[p.i], kj = id[p.j];
          for (int a = 0; a < 3; a++) {
            if (ki >= 0) F0[3 * ki + a] += w * q[p.j] * d[a] / R3;
            if (kj >= 0) F0[3 * kj + a] -= w * q[p.i] * d[a] / R3;
          }
          if (ki >= 0 && kj >= 0)
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++) {
                const double t = w * (3 * d[a] * d[b] / R5 - (a == b ? 1.0 / R3 : 0.0));
                A[(3 * ki + a) * m3 + 3 * kj + b] -= t, A[(3 * kj + b) * m3 + 3 * ki + a] -= t;
              }
        }
        // dense Cholesky (A is symmetric positive definite for the far-field couplings used here)
        std::vector<double> Lc(A), y(F0);
        for (int j = 0; j < m3; j++) {
          double d = Lc[size_t(j) * m3 + j];
          for (int k = 0; k < j; k++) d -= Lc[size_t(j) * m3 + k] * Lc[size_t(j) * m3 + k];
          if (!(d > 0)) throw std::runtime_error("lrc pol: polarisation matrix not positive definite (polarisation catastrophe)");
          Lc[size_t(j) * m3 + j] = std::sqrt(d);
          for (int i = j + 1; i < m3; i++) {
            double v = Lc[size_t(i) * m3 + j];
            for (int k = 0; k < j; k++) v -= Lc[size_t(i) * m3 + k] * Lc[size_t(j) * m3 + k];
            Lc[size_t(i) * m3 + j] = v / Lc[size_t(j) * m3 + j];
          }
        }
        for (int i = 0; i < m3; i++) {
          for (int k = 0; k < i; k++) y[i] -= Lc[size_t(i) * m3 + k] * y[k];
          y[i] /= Lc[size_t(i) * m3 + i];
        }
        for (int i = m3 - 1; i >= 0; i--) {
          for (int k = i + 1; k < m3; k++) y[i] -= Lc[size_t(k) * m3 + i] * y[k];
          y[i] /= Lc[size_t(i) * m3 + i];
        }
        double eind = 0;
        for (int k = 0; k < m3; k++) eind += y[k] * F0[k];
        out.energy += -0.5 * KE * eind;
        out.mu_ind.assign(3 * size_t(n), 0.0);
        for (size_t k = 0; k < at.size(); k++)
          for (int a = 0; a < 3; a++) out.mu_ind[3 * size_t(at[k]) + a] = y[3 * k + a];
        if (forces)
          for (const P &p : pairs) {
            double r[3];
            for (int a = 0; a < 3; a++) r[a] = x[3 * p.j + a] - x[3 * p.i + a];
            const double *mi = id[p.i] >= 0 ? &y[3 * size_t(id[p.i])] : nullptr;
            const double *mj = id[p.j] >= 0 ? &y[3 * size_t(id[p.j])] : nullptr;
            if (!mi && !mj) continue;
            accumulate(p, pair_kernel(r, q[p.i], q[p.j], mi, mj, nullptr, nullptr, false), 1.0, false, false);
          }
      }
    }
    if (!forces) return out;
    if (L.mp) lrc_moments_vjp(L, n, Z, x, Gm.data(), Hm.data(), g.data());
    unplateau();
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
  unplateau();
  lrc_clusters_vjp(L, c.cl, n, x, W, g.data());
  lrc_q0_vjp(L, S, n, Z, x, c.q0, dq.data(), g.data());
  for (size_t k = 0; k < g.size(); k++) out.forces[k] = -g[k];
  return out;
}
} // namespace

std::vector<double> lrc_project_charges(const LrcSpec &L, int n, const int32_t *Z, const double *x, const double *q) {
  const LrcClusters c = lrc_clusters(L, n, x);
  return lrc_project(L, c, n, Z, q).q;
}

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

std::vector<double> lrc_bec(const LrcSpec &L, const SqeSpec &S, const LrcQ0 &q, int n, const int32_t *Z, const double *x) {
  std::vector<double> out(9 * size_t(n), 0.0), c(n), g;
  for (int a = 0; a < 3; a++) {
    for (int k = 0; k < n; k++) c[k] = x[3 * k + a];
    g.assign(3 * size_t(n), 0.0);
    lrc_q0_vjp(L, S, n, Z, x, q, c.data(), g.data());
    for (int i = 0; i < n; i++)
      for (int b = 0; b < 3; b++) out[9 * size_t(i) + 3 * a + b] = g[3 * size_t(i) + b] + (a == b ? q.q0[i] : 0.0);
  }
  return out;
}

} // namespace mlipcpp::runtime
