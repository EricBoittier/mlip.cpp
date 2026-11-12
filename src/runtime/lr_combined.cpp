#include "lr_combined.h"

#include "core/gguf_loader.h"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

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
// g += dE/dx of a pair term E(r_ij); vir (3 x 3, may be null) += its strain derivative d (dE/dd), d = x_j - x_i
void add_pair(double *g, const double *x, int i, int j, double r, double dEdr, double *vir = nullptr) {
  double d[3];
  for (int k = 0; k < 3; k++) {
    d[k] = x[3 * j + k] - x[3 * i + k];
    const double f = dEdr * d[k] / r;
    g[3 * i + k] -= f, g[3 * j + k] += f;
  }
  if (vir)
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) vir[3 * a + b] += dEdr * d[a] * d[b] / r;
}

// g += dE/dx of weak link w's energy term with dE/dr = dEdr (cluster-frame positions x = c.xh, or the link's image
// displacement for layered clusters); vir (may be null) += its strain derivative
void link_grad(const LrcClusters &c, size_t w, const double *x, double dEdr, double *g, double *vir) {
  const auto &wk = c.weak[w];
  if (!c.layered) return add_pair(g, x, wk.i, wk.j, dist(x, wk.i, wk.j), dEdr, vir);
  const auto &d = c.wd[w];
  const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  for (int k = 0; k < 3; k++) {
    const double f = dEdr * d[k] / r;
    g[3 * wk.i + k] -= f, g[3 * wk.j + k] += f;
  }
  if (vir)
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) vir[3 * a + b] += dEdr * d[a] * d[b] / r;
}

// ---- periodic geometry ------------------------------------------------------------------------------------------------
// cell rows a_k; fractional f = x inv (x = f h); perp[k]: spacing of the lattice planes normal to a_k
struct Cell3 {
  double h[3][3], inv[3][3], V, perp[3];
};
Cell3 cell3(const double *cell) {
  Cell3 C;
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) C.h[a][b] = cell[3 * a + b];
  const auto &h = C.h;
  const double det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0]) +
                     h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
  if (!(std::abs(det) > 1e-9) || !std::isfinite(det)) throw std::runtime_error("lrc: singular periodic cell");
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++)
      C.inv[a][b] = (h[(b + 1) % 3][(a + 1) % 3] * h[(b + 2) % 3][(a + 2) % 3] -
                     h[(b + 1) % 3][(a + 2) % 3] * h[(b + 2) % 3][(a + 1) % 3]) / det;
  C.V = std::abs(det);
  for (int k = 0; k < 3; k++) {
    const double *u = h[(k + 1) % 3], *v = h[(k + 2) % 3];
    const double c0 = u[1] * v[2] - u[2] * v[1], c1 = u[2] * v[0] - u[0] * v[2], c2 = u[0] * v[1] - u[1] * v[0];
    C.perp[k] = C.V / std::sqrt(c0 * c0 + c1 * c1 + c2 * c2);
  }
  return C;
}

using I3 = std::array<long long, 3>;
inline bool nonzero(const I3 &v) { return v[0] || v[1] || v[2]; }

// every periodic pair closer than rc, once: visit(i, j, s, d, r) with d = x_j + s h - x_i, i < j, or i == j with s
// lexicographically positive (an atom and its own image); bins in fractional space, any cell shape and any rc
template <class F> void periodic_pairs(int n, const double *x, const Cell3 &C, double rc, F &&visit) {
  int nb[3], m[3];
  for (int k = 0; k < 3; k++) {
    nb[k] = std::clamp(static_cast<int>(C.perp[k] / rc), 1, 64);
    m[k] = static_cast<int>(std::ceil(rc * nb[k] / C.perp[k]));
  }
  std::vector<std::array<int, 3>> bin(n);
  std::vector<I3> w(n);
  std::vector<std::vector<int>> cells(size_t(nb[0]) * nb[1] * nb[2]);
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < 3; k++) {
      double f = 0;
      for (int a = 0; a < 3; a++) f += x[3 * i + a] * C.inv[a][k];
      const double fl = std::floor(f);
      w[i][k] = static_cast<long long>(fl);
      bin[i][k] = std::min(static_cast<int>((f - fl) * nb[k]), nb[k] - 1);
    }
    cells[(size_t(bin[i][0]) * nb[1] + bin[i][1]) * nb[2] + bin[i][2]].push_back(i);
  }
  auto fdiv = [](int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
  const double rc2 = rc * rc;
  for (int i = 0; i < n; i++)
    for (int o0 = -m[0]; o0 <= m[0]; o0++)
      for (int o1 = -m[1]; o1 <= m[1]; o1++)
        for (int o2 = -m[2]; o2 <= m[2]; o2++) {
          const int o[3] = {o0, o1, o2};
          int tb[3], sh[3];
          for (int k = 0; k < 3; k++) {
            const int t = bin[i][k] + o[k];
            sh[k] = fdiv(t, nb[k]), tb[k] = t - sh[k] * nb[k];
          }
          for (int j : cells[(size_t(tb[0]) * nb[1] + tb[1]) * nb[2] + tb[2]]) {
            if (j < i) continue;
            I3 s;
            for (int k = 0; k < 3; k++) s[k] = sh[k] + w[i][k] - w[j][k];
            if (j == i && !(s[0] > 0 || (s[0] == 0 && (s[1] > 0 || (s[1] == 0 && s[2] > 0))))) continue;
            double d[3], r2 = 0;
            for (int a = 0; a < 3; a++) {
              d[a] = x[3 * j + a] - x[3 * i + a] + s[0] * C.h[0][a] + s[1] * C.h[1][a] + s[2] * C.h[2][a];
              r2 += d[a] * d[a];
            }
            if (r2 < rc2 && r2 > 0) visit(i, j, s, d, std::sqrt(r2));
          }
        }
}

// integer lattice spanned by the wrap vectors of a cluster (net cell translations of its cycles), rows in echelon form
struct Wraps {
  long long b[3][3] = {};
  void add(I3 v) {
    for (int c = 0; c < 3; c++) {
      if (v[c] == 0) continue;
      if (b[c][c] == 0) {
        for (int k = 0; k < 3; k++) b[c][k] = v[c] < 0 ? -v[k] : v[k];
        return;
      }
      while (v[c] != 0) { // Euclid on the pivot column (unimodular row operations)
        const long long t = b[c][c] / v[c];
        for (int k = 0; k < 3; k++) b[c][k] -= t * v[k], std::swap(b[c][k], v[k]);
      }
      if (b[c][c] < 0)
        for (long long &e : b[c]) e = -e;
    }
  }
  void add(const Wraps &o) {
    for (const auto &r : o.b) add(I3{r[0], r[1], r[2]});
  }
  bool none() const { return !b[0][0] && !b[1][1] && !b[2][2]; }
  bool full() const { return b[0][0] == 1 && b[1][1] == 1 && b[2][2] == 1; } // every image is the same cluster
};

// union-find with cell offsets: phi(i) = the image of i in its root's frame; a link (i, 0) ~ (j, s) says
// phi(j) - phi(i) = s, and a link that closes a cycle with another offset is a wrap of the cluster
struct OffsetUF {
  std::vector<int> p;
  std::vector<I3> o; // phi(i) - phi(p[i])
  explicit OffsetUF(int n) : p(n), o(n, I3{0, 0, 0}) { std::iota(p.begin(), p.end(), 0); }
  std::pair<int, I3> find(int i) {
    std::vector<int> path;
    int r = i;
    while (p[r] != r) path.push_back(r), r = p[r];
    I3 up{0, 0, 0};
    for (size_t k = path.size(); k-- > 0;) {
      const int v = path[k];
      for (int d = 0; d < 3; d++) up[d] += o[v][d];
      o[v] = up, p[v] = r;
    }
    return {r, path.empty() ? I3{0, 0, 0} : o[i]};
  }
  // join the clusters of i and j through the link (i, 0) ~ (j, s); returns the wrap vector it closes (zero if none)
  I3 unite(int i, int j, const I3 &s, std::vector<Wraps> *wr = nullptr) {
    const auto [ri, fi] = find(i);
    const auto [rj, fj] = find(j);
    I3 w;
    for (int d = 0; d < 3; d++) w[d] = fi[d] + s[d] - fj[d];
    if (ri == rj) {
      if (wr && nonzero(w)) (*wr)[ri].add(w);
      return w;
    }
    p[rj] = ri, o[rj] = w;
    if (wr) (*wr)[ri].add((*wr)[rj]);
    return I3{0, 0, 0};
  }
};

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
    if (cm != "cluster" && cm != "global") throw std::runtime_error("lrc: lrc_cond must be cluster or global");
    L.cond_global = cm == "global";
    scalar("cond_w", L.cond_w);
    if (G.get_int32("pet.plato_lr.lrc_qplateau", 0) != 0)
      throw std::runtime_error("lrc_qplateau was removed (it lost sum q = Q and double-counted partly linked partners); "
                               "use lrc_qproj");
    L.qproj = G.get_int32("pet.plato_lr.lrc_qproj", 0) != 0;
    scalar("qproj_w", L.qproj_w);
    if (L.qproj && !(L.qproj_w > 0 && L.qproj_w <= 1)) throw std::runtime_error("lrc: lrc_qproj_w must be in (0, 1]");
    if (!(L.cond_w > 0 && L.cond_w <= 1)) throw std::runtime_error("lrc: lrc_cond_w must be in (0, 1]");
  }
  {
    const std::string im = G.get_string("pet.plato_lr.lrc_intra", "none");
    if (im != "none" && im != "ewald") throw std::runtime_error("lrc: lrc_intra must be none or ewald");
    L.intra = im == "ewald";
  }
  {
    const std::string sm = G.get_string("pet.plato_lr.lrc_slab", "none");
    if (sm == "none") L.slab = LrcSpec::Slab::None;
    else if (sm == "yb") L.slab = LrcSpec::Slab::YB;
    else if (sm == "ewald2d") L.slab = LrcSpec::Slab::Ewald2D;
    else throw std::runtime_error("lrc: lrc_slab must be none, yb or ewald2d");
    if (const auto v = get_f64(G, "pet.plato_lr.lrc_slab_vacuum"); !v.empty()) L.slab_vacuum = v[0];
    if (const auto v = get_f64(G, "pet.plato_lr.lrc_slab_gap"); !v.empty()) L.slab_gap = v[0];
    if ((L.slab_vacuum != 0 || L.slab_gap != 0) && L.slab == LrcSpec::Slab::None)
      throw std::runtime_error("lrc: lrc_slab_vacuum / lrc_slab_gap need lrc_slab = yb or ewald2d");
    if (!(L.slab_vacuum >= 0) || !(L.slab_gap >= 0)) throw std::runtime_error("lrc: lrc_slab_vacuum / _gap must be >= 0");
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
    if (L.intra && (L.mp || L.pol)) throw std::runtime_error("lrc_intra: not with lrc_mp / lrc_pol");
    if (L.slab != LrcSpec::Slab::None && (L.mp || L.pol)) throw std::runtime_error("lrc_slab: not with lrc_mp / lrc_pol");
    if (L.slab_gap > 0 && !(L.slab_gap > L.a_off && L.slab_gap > S.lr_R))
      throw std::runtime_error("lrc_slab_gap must exceed lrc_a_off and sqe_lr_R");
    return L;
  }
  if (L.intra) throw std::runtime_error("lrc_intra: needs the graph q0 mode");
  if (L.slab != LrcSpec::Slab::None) throw std::runtime_error("lrc_slab: needs the graph q0 mode");
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
  gguf_set_val_str(g, "pet.plato_lr.lrc_cond", cond_global ? "global" : "cluster");
  scalar("cond_w", cond_w);
  gguf_set_val_i32(g, "pet.plato_lr.lrc_qproj", qproj ? 1 : 0);
  scalar("qproj_w", qproj_w);
  if (intra) gguf_set_val_str(g, "pet.plato_lr.lrc_intra", "ewald"); // absent key = none (earlier GGUFs unchanged)
  if (slab != Slab::None) { // absent keys = slabs refused (earlier GGUFs unchanged)
    gguf_set_val_str(g, "pet.plato_lr.lrc_slab", slab == Slab::YB ? "yb" : "ewald2d");
    if (slab_vacuum != 0) scalar("slab_vacuum", slab_vacuum);
    if (slab_gap != 0) scalar("slab_gap", slab_gap);
  }
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

// An element absent from lrc_fukui_Z (e.g. the rare gases: the Fukui fits list only the elements in their data) takes
// phi = 0 (both channels), the value the fits start every element from (an element listed without data keeps it). It
// used to throw, which failed every structure with such an atom once it had more than one cluster (Ar ... Xe at 9 A).
double LrcSpec::fukui(int z) const {
  if (fukui_mode == FukuiMode::Uniform) return 1.0;
  const double p = fukui_phi_of(z);
  return (p > 30 ? p : std::log1p(std::exp(p))) + 1e-6;
}

double LrcSpec::fukui_phi_of(int z, bool minus) const {
  const std::vector<double> &phi = minus && !fukui_phi_minus.empty() ? fukui_phi_minus : fukui_phi;
  for (size_t k = 0; k < fukui_Z.size(); k++)
    if (fukui_Z[k] == z) return phi[k];
  return 0.0;
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
LrcQ0 q0_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell);
void vjp_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, const LrcQ0 &o, const double *c, double *g,
               double *vir);
} // namespace

LrcQ0 lrc_q0(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q, int spin,
             const double *cell) {
  if (L.graph) return q0_graph(L, n, Z, x, Q, spin, cell);
  if (cell) throw std::runtime_error("lrc: periodic reference charges need the graph q0 mode");
  return L.round || L.head_only ? q0_round(L, S, n, Z, x, Q) : q0_fermi2(L, S, n, Z, x, Q);
}
void lrc_q0_vjp(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, const LrcQ0 &q,
                const double *c, double *g, double *vir) {
  if (L.graph) vjp_graph(L, n, Z, x, q, c, g, vir);
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

namespace {
// super-edges, link probabilities and reliabilities from the super-nodes (c.s, c.sn) and the weak atom pairs, whose
// distances are those of the cluster-frame positions c.xh
// (layered clusters: disp[k] the image displacement of weak link k, stored in c.wd; its distance replaces c.xh's)
void finish_clusters(const LrcSpec &L, LrcClusters &c, const std::vector<std::pair<int, int>> &weak,
                     const std::vector<std::array<double, 3>> *disp = nullptr) {
  const double *x = c.xh.data();
  for (size_t l = 0; l < weak.size(); l++) {
    const int i = weak[l].first, j = weak[l].second;
    int a = c.sn[i], b = c.sn[j];
    if (a == b) continue;
    if (a > b) std::swap(a, b);
    int k = -1;
    for (size_t q = 0; q < c.se.size(); q++)
      if (c.se[q].first == a && c.se[q].second == b) k = int(q);
    if (k < 0) k = int(c.se.size()), c.se.push_back({a, b}), c.p.push_back(0.0);
    double r;
    if (disp) {
      const auto &d = (*disp)[l];
      r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      c.wd.push_back(d);
    } else {
      r = dist(x, i, j);
    }
    const double w = L.a_off - L.a_on, u = (r - L.a_on) / w;
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
}

// slabs: the zero row of a (T, T, F) cell (-1: none; two or more throw), and the unit normal / area of the other two
int zero_row(const double *cell) {
  int k = -1, m = 0;
  for (int r = 0; r < 3; r++)
    if (cell[3 * r] == 0 && cell[3 * r + 1] == 0 && cell[3 * r + 2] == 0) k = r, m++;
  if (m > 1) throw std::runtime_error("lrc: a cell with two or three open directions (a wire) is not supported");
  return k;
}
double plane_normal(const double *cell, int open, double *nrm) {
  const double *u = cell + 3 * ((open + 1) % 3), *v = cell + 3 * ((open + 2) % 3);
  const double c0 = u[1] * v[2] - u[2] * v[1], c1 = u[2] * v[0] - u[0] * v[2], c2 = u[0] * v[1] - u[1] * v[0];
  const double A = std::sqrt(c0 * c0 + c1 * c1 + c2 * c2);
  if (!(A > 1e-9)) throw std::runtime_error("lrc: degenerate slab cell (in-plane rows parallel)");
  nrm[0] = c0 / A, nrm[1] = c1 / A, nrm[2] = c2 / A;
  return A;
}
// (T, T, T) gap rule (lrc_slab_gap): the one lattice direction across which the atoms leave an empty layer >= gap (A),
// and per atom the multiple of that row that unwraps the slab into one piece; -1 when no or several directions have one
int gap_axis(double gap, int n, const double *x, const double *cell, std::vector<int32_t> &zshift) {
  const Cell3 C = cell3(cell);
  int axis = -1, hits = 0;
  double base = 0;
  for (int k = 0; k < 3; k++) {
    std::vector<double> u(n);
    for (int i = 0; i < n; i++) {
      double f = 0;
      for (int a = 0; a < 3; a++) f += x[3 * i + a] * C.inv[a][k];
      u[i] = f - std::floor(f);
    }
    std::sort(u.begin(), u.end());
    double best = n ? 1.0 - u[n - 1] + u[0] : 1.0, b = n ? u[0] : 0.0; // the gap through the cell boundary
    for (int i = 0; i + 1 < n; i++)
      if (u[i + 1] - u[i] > best) best = u[i + 1] - u[i], b = u[i + 1];
    if (best * C.perp[k] >= gap) axis = k, base = b, hits++;
  }
  if (hits != 1) return -1;
  zshift.assign(n, 0);
  for (int i = 0; i < n; i++) {
    double f = 0;
    for (int a = 0; a < 3; a++) f += x[3 * i + a] * C.inv[a][axis];
    zshift[i] = -static_cast<int32_t>(std::floor(f - base));
  }
  return axis;
}

// periodic: links through the images; super-nodes are the strong components of the tiling, either one bulk cluster
// (connected to all its images) or all finite, and then each weak component is unwrapped into c.xh. Slabs (lrc_slab):
// the same in the 2D tiling (bulk = one cluster through every in-plane image); with lrc_intra also clusters that wrap
// only part of the periodic directions (layered: weak links by image displacement, no unwrapping)
LrcClusters clusters_periodic(const LrcSpec &L, int n, const double *x, const double *cell) {
  LrcClusters c;
  c.cell.assign(cell, cell + 9);
  int open = zero_row(cell);
  std::vector<double> hs(cell, cell + 9); // the cell of the pair search
  if (open >= 0) {
    if (L.slab == LrcSpec::Slab::None)
      throw std::runtime_error("lrc: a slab (a cell with one open direction) needs pet.plato_lr.lrc_slab = yb or ewald2d");
    double nrm[3], lo = 0, hi = 0;
    plane_normal(cell, open, nrm);
    for (int i = 0; i < n; i++) {
      const double z = nrm[0] * x[3 * i] + nrm[1] * x[3 * i + 1] + nrm[2] * x[3 * i + 2];
      lo = i ? std::min(lo, z) : z, hi = i ? std::max(hi, z) : z;
    }
    for (int a = 0; a < 3; a++) hs[3 * open + a] = nrm[a] * (hi - lo + L.a_off + 1.0); // no open-direction image within a_off
  } else if (L.slab != LrcSpec::Slab::None && L.slab_gap > 0) {
    open = gap_axis(L.slab_gap, n, x, cell, c.zshift);
  }
  c.open = open;
  const Cell3 C = cell3(hs.data());
  OffsetUF u(n);
  std::vector<Wraps> wr(n);
  struct Link {
    int i, j;
    I3 s;
    std::array<double, 3> d;
  };
  std::vector<Link> weak;
  periodic_pairs(n, x, C, L.a_off, [&](int i, int j, const I3 &s, const double *d, double r) {
    if (r <= L.a_on) u.unite(i, j, s, &wr);
    else weak.push_back({i, j, s, {d[0], d[1], d[2]}});
  });
  std::vector<int> root(n);
  std::vector<I3> phi(n);
  for (int i = 0; i < n; i++) std::tie(root[i], phi[i]) = u.find(i);
  c.sn.resize(n);
  std::vector<int> lab(n, -1);
  c.s = 0;
  for (int i = 0; i < n; i++) { // numbered by first atom
    if (lab[root[i]] < 0) lab[root[i]] = c.s++;
    c.sn[i] = lab[root[i]];
  }
  // through every image of the periodic directions (a slab: the two in-plane ones)
  auto full = [&](const Wraps &w) {
    if (open < 0) return w.full();
    for (int k = 0; k < 3; k++)
      if (k != open && w.b[k][k] != 1) return false;
    return w.b[open][open] == 0;
  };
  if (c.s == 1 && full(wr[root[0]])) { // bulk: one cluster through every image
    c.bulk = true, c.xh.assign(x, x + 3 * size_t(n)), c.R.assign(1, 1.0);
    return c;
  }
  const bool may_layer = L.intra && L.slab != LrcSpec::Slab::None;
  bool layered = false;
  for (int i = 0; i < n; i++)
    if (!wr[root[i]].none()) {
      if (!may_layer)
        throw std::runtime_error(
            "lrc: a cluster wraps the periodic cell without being one bulk cluster (a slab, a wire, interpenetrating "
            "networks, or a network beside separate molecules); only bulk cells, where every atom has a neighbour within "
            "a_on = " + std::to_string(L.a_on) + " A through the images, and fully separated clusters are supported" +
            (L.slab != LrcSpec::Slab::None ? " (with lrc_slab: also any layered structure under lrc_intra)" : ""));
      layered = true;
    }
  // weak links between super-nodes, with the cell offset between the two clusters' frames
  OffsetUF v(c.s);
  std::vector<std::pair<int, int>> wk;
  const char *wrap = "lrc: clusters within a_off of their own periodic images, directly or through a chain of weak "
                     "links, are not supported (use a larger cell)";
  for (size_t l = 0; l < weak.size() && !layered; l++) {
    const Link &lk = weak[l];
    const int a = c.sn[lk.i], b = c.sn[lk.j];
    I3 d;
    for (int k = 0; k < 3; k++) d[k] = lk.s[k] - phi[lk.j][k] + phi[lk.i][k];
    bool bad = false;
    if (a == b) {
      if (nonzero(d)) bad = true;
      else continue;
    } else if (nonzero(v.unite(a, b, d))) {
      bad = true;
    }
    if (bad) {
      if (!may_layer) throw std::runtime_error(wrap);
      layered = true;
      break;
    }
    wk.push_back({lk.i, lk.j});
  }
  if (layered) { // links by image displacement; R on the quotient graph of the cell's super-nodes (self-links dropped)
    c.layered = true;
    c.xh.assign(x, x + 3 * size_t(n));
    std::vector<std::pair<int, int>> lk;
    std::vector<std::array<double, 3>> disp;
    for (const Link &l : weak)
      if (c.sn[l.i] != c.sn[l.j]) lk.push_back({l.i, l.j}), disp.push_back(l.d);
    finish_clusters(L, c, lk, &disp);
    return c;
  }
  c.xh.resize(3 * size_t(n));
  for (int i = 0; i < n; i++) {
    const I3 psi = v.find(c.sn[i]).second;
    for (int a = 0; a < 3; a++) {
      double t = x[3 * i + a];
      for (int k = 0; k < 3; k++) t += double(phi[i][k] + psi[k]) * C.h[k][a];
      c.xh[3 * i + a] = t;
    }
  }
  finish_clusters(L, c, wk);
  return c;
}
} // namespace

LrcClusters lrc_clusters(const LrcSpec &L, int n, const double *x, const double *cell) {
  if (cell) return clusters_periodic(L, n, x, cell);
  LrcClusters c;
  c.xh.assign(x, x + 3 * size_t(n));
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
  finish_clusters(L, c, weak);
  return c;
}

void lrc_clusters_vjp(const LrcSpec &L, const LrcClusters &c, int n, const double *x, const std::vector<double> &W,
                      double *g, double *vir) {
  (void)n;
  if (c.trivial()) return;
  if (!c.xh.empty()) x = c.xh.data(); // the links are differences of the cluster-frame positions
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
    link_grad(c, w, x, dp[wk.e] * others * wk.dA, g, vir);
  }
}

// ---- conditioning --------------------------------------------------------------------------------------------------
bool LrcCondition::needs_cond_forces() const {
  for (double v : dS)
    if (v != 0) return true;
  return false;
}

LrcCondition lrc_condition(const LrcSpec &L, const SqeSpec &S, int n, const int32_t *Z, const double *x, double Q,
                           int max_charge, int spin, const double *cell) {
  LrcCondition c;
  c.cl = lrc_clusters(L, n, x, cell);
  const int s = c.cl.s;
  c.Q0.assign(s, 0.0), c.Qloc.assign(s, 0.0), c.S.assign(s, 0.0), c.dS.assign(s, 0.0), c.k.assign(s, 0);
  if (c.cl.images() && (!L.graph || L.mp || L.pol))
    throw std::runtime_error("lrc: separated clusters in a periodic cell need the graph q0 mode without lrc_mp / "
                             "lrc_pol (bulk cells work in every mode)");
  if (L.intra && (!L.graph || L.mp || L.pol))
    throw std::runtime_error("lrc_intra: needs the graph q0 mode without lrc_mp / lrc_pol");
  if (s == 1 && !c.cl.images()) { // one open cluster or a bulk cell: conditioned on the structure's charge
    c.Q0[0] = c.Qloc[0] = Q;
    if (L.intra) c.q0 = lrc_q0(L, S, n, Z, x, Q, spin, cell); // the long range still acts inside (charge pass in the cell)
  } else { // (a single periodic cluster still meets its images: the charges are needed)
    c.q0 = lrc_q0(L, S, n, Z, x, Q, spin, cell);
    for (int i = 0; i < n; i++) c.Q0[c.cl.sn[i]] += c.q0.q0[i];
    for (int a = 0; a < s; a++)
      for (int b = 0; b < s; b++) c.Qloc[a] += c.cl.R[size_t(a) * s + b] * c.Q0[b];
    if (L.cond_global) c.Qloc.assign(s, Q); // global conditioning: no position dependence (dS = 0)
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
                          const double *x, const std::vector<double> &G, double *F, double *vir) {
  const int s = c.cl.s;
  if (s == 1 || L.cond_global) return; // (one periodic cluster: Q_loc = Q, no position dependence either)
  std::vector<double> g(3 * size_t(n), 0.0), W(size_t(s) * s, 0.0), cq(n, 0.0);
  std::vector<double> gs(s);
  for (int a = 0; a < s; a++) gs[a] = G[a] * c.dS[a]; // dE/dQloc_a
  for (int a = 0; a < s; a++)
    for (int b = 0; b < s; b++)
      if (a != b) W[size_t(a) * s + b] = gs[a] * c.Q0[b];
  lrc_clusters_vjp(L, c.cl, n, x, W, g.data(), vir);
  for (int i = 0; i < n; i++) {
    const int b = c.cl.sn[i];
    for (int a = 0; a < s; a++) cq[i] += gs[a] * c.cl.R[size_t(a) * s + b];
  }
  lrc_q0_vjp(L, S, n, Z, x, c.q0, cq.data(), g.data(), vir);
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
// [n]) += dE/df_i; vir (periodic, may be null) += the strain derivative of the link part
void lrc_project_vjp(const LrcSpec &L, const LrcClusters &c, const LrcProjection &o, int n, const double *x,
                     std::vector<double> &dq, double *g, double *df = nullptr, double *vir = nullptr) {
  const int s = c.s;
  if (!c.xh.empty()) x = c.xh.data();
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
    link_grad(c, w, x, dp[wk.e] * others * wk.dA, g, vir);
  }
}

// graph mode Fukui weights of one channel: f_i = softplus(phi_Z + a_i) + 1e-6 (a = 0 without a pass), df = df/da
void fukui_channel(const LrcSpec &L, int n, const int32_t *Z, const double *x, double Qp, int spin, const double *cell,
                   bool minus, std::vector<double> &f, std::vector<double> &df) {
  f.assign(n, 1.0), df.assign(n, 0.0);
  if (L.fukui_mode == LrcSpec::FukuiMode::Uniform) return;
  std::vector<double> a(n, 0.0);
  LrcQSource *src = minus ? L.fsrc_minus.get() : L.fsrc.get();
  if (L.fukui_atom()) {
    if (!src) throw std::runtime_error("lrc graph: the Fukui pass (lrc_fmodel" + std::string(minus ? "_minus" : "") +
                                       ") is not loaded");
    a = src->raw(n, Z, x, Qp, spin, cell);
  }
  for (int i = 0; i < n; i++) {
    const double u = L.fukui_phi_of(Z[i], minus) + a[i];
    f[i] = (u > 30 ? u : std::log1p(std::exp(u))) + 1e-6, df[i] = sig(u);
  }
}

// graph mode: q0 = q_raw + w (Q - sum q_raw), w = f / sum f (A+-: blend of f+ / F+ and f- / F- by the residual's sign);
// Fukui scope "cluster": the residual-routing soft-cluster projection of q_raw instead
LrcQ0 q0_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell) {
  if (!L.qsrc) throw std::runtime_error("lrc graph: the charge pass (lrc_qmodel) is not loaded");
  LrcQ0 o;
  o.Q = Q, o.spin = spin;
  if (cell) o.cell.assign(cell, cell + 9);
  if (n == 0) return o;
  const double Qp = L.qpass_neutral ? 0.0 : Q;
  o.qraw = L.qsrc->raw(n, Z, x, Qp, spin, cell);
  o.q0 = o.qraw;
  fukui_channel(L, n, Z, x, Qp, spin, cell, false, o.fp, o.dfp);
  double sq = 0;
  for (int i = 0; i < n; i++) sq += o.qraw[i], o.Fp += o.fp[i];
  o.r = Q - sq;
  if (L.fukui_cluster) {
    const LrcClusters c = lrc_clusters(L, n, x, cell);
    o.q0 = lrc_project(L, c, n, o.fp.data(), o.qraw.data(), L.fukui_cluster_w, true, Q).q;
    return o;
  }
  o.wt.resize(n);
  if (L.fukui_mode == LrcSpec::FukuiMode::AtomPM) {
    fukui_channel(L, n, Z, x, Qp, spin, cell, true, o.fm, o.dfm);
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
void vjp_graph(const LrcSpec &L, int n, const int32_t *Z, const double *x, const LrcQ0 &o, const double *c, double *g,
               double *vir) {
  const double Qp = L.qpass_neutral ? 0.0 : o.Q;
  const double *cell = o.cell.empty() ? nullptr : o.cell.data();
  std::vector<double> w(n), u(n, 0.0);
  bool any = false;
  auto fpass = [&](LrcQSource *src, const std::vector<double> &df) { // u: dE/df_i
    bool nz = false;
    for (int i = 0; i < n; i++) u[i] *= df[i], nz |= u[i] != 0;
    if (nz && src) src->raw_vjp(n, Z, x, Qp, o.spin, cell, u.data(), g, vir);
  };
  if (L.fukui_cluster) {
    const LrcClusters cl = lrc_clusters(L, n, x, cell);
    const LrcProjection pr = lrc_project(L, cl, n, o.fp.data(), o.qraw.data(), L.fukui_cluster_w, true, o.Q);
    std::vector<double> dq(c, c + n);
    lrc_project_vjp(L, cl, pr, n, x, dq, g, L.fukui_atom() ? u.data() : nullptr, vir);
    for (int i = 0; i < n; i++) any |= dq[i] != 0;
    if (any) L.qsrc->raw_vjp(n, Z, x, Qp, o.spin, cell, dq.data(), g, vir);
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
  if (any) L.qsrc->raw_vjp(n, Z, x, Qp, o.spin, cell, w.data(), g, vir);
  if (!L.fukui_atom() || o.r == 0) return;
  // sum_i c_i r (1 - s) d(f+_i / F+) = r (1 - s) / F+ sum_j (c_j - cp) df+_j
  for (int i = 0; i < n; i++) u[i] = o.r * (1.0 - o.s) / o.Fp * (c[i] - cp);
  fpass(L.fsrc.get(), o.dfp);
  if (!pm) return;
  for (int i = 0; i < n; i++) u[i] = o.r * o.s / o.Fm * (c[i] - cm);
  fpass(L.fsrc_minus.get(), o.dfm);
}

// periodic, separated clusters: the fixed-charge inter-cluster Coulomb of the tiling,
//   E = KE [ Ewald(q) - sum_{pairs r < lr_R} q_i q_j (1 - T) / r - sum_{i < j, R_ab > 0} q_i q_j T R_ab / r^_ij ]
// (every image pair counted once; r^ the distance in the unwrapped weak component, the only image of j linked to i).
// Ewald: point charges, tin-foil boundary, neutralising background for a charged cell; real-space cutoff balancing the
// real and reciprocal sums (at least lr_R) with erfc(alpha rc) = erfc(5) ~ 1e-12.
// lrc_intra: no linked-pair term (R = 0 in the energy), so also a bulk cell (one cluster through its images):
//   E = KE [ Ewald(q) - sum_{pairs r < lr_R} q_i q_j (1 - T) / r ]
LrcEnergy energy_images(const LrcSpec &L, const SqeSpec &S, const LrcCondition &c, int n, const int32_t *Z,
                        const double *x, bool forces) {
  constexpr double KE = 14.399645, acc = 5.0;
  LrcEnergy out;
  out.q0 = c.q0.q0, out.q = c.q0.q0;
  const int s = c.cl.s;
  const Cell3 C = cell3(c.cl.cell.data());
  const double r0 = S.lr_R - S.lr_t;
  std::vector<double> g(3 * size_t(n), 0.0), dq(n, 0.0), W(size_t(s) * s, 0.0), vir(9, 0.0);
  std::vector<double> q = c.q0.q0;
  LrcProjection proj;
  const bool qproj = L.qproj && !L.fukui_cluster; // AC: the reference charges are already projected
  if (qproj) proj = lrc_project(L, c.cl, n, Z, q.data()), q = proj.q, out.q = q;
  double E = 0, Qt = 0;
  for (double v : q) Qt += v;
  // one pair's energy q_i q_j u(r), j's image at d = x_j' - x_i, du = du/dr
  auto pair = [&](int i, int j, const double *d, double r, double u, double du) {
    E += q[i] * q[j] * u;
    dq[i] += q[j] * u, dq[j] += q[i] * u;
    if (!forces) return;
    const double f = q[i] * q[j] * du / r;
    for (int a = 0; a < 3; a++) {
      g[3 * j + a] += f * d[a], g[3 * i + a] -= f * d[a];
      for (int b = 0; b < 3; b++) vir[3 * a + b] += f * d[a] * d[b];
    }
  };
  auto sw = [&](double r, double &T, double &dT) {
    const double t = (r - r0) / S.lr_t;
    T = sstep(t), dT = dsstep(t) / S.lr_t;
  };
  const double rc = std::max(S.lr_R, acc * std::cbrt(C.V) / (std::sqrt(M_PI) * std::pow(double(n), 1.0 / 6.0)));
  const double al = acc / rc, kmax = 2 * acc * al;
  // real space: erfc(alpha r) / r, less the part of the bare 1 / r that the switch keeps out (1 - T inside lr_R)
  periodic_pairs(n, x, C, std::max(rc, S.lr_R), [&](int i, int j, const I3 &, const double *d, double r) {
    double u = 0, du = 0;
    if (r < rc) {
      const double e = std::erfc(al * r) / r;
      u += e, du -= (e + 2 * al / std::sqrt(M_PI) * std::exp(-al * al * r * r)) / r;
    }
    if (r < S.lr_R) {
      double T, dT;
      sw(r, T, dT);
      u -= (1 - T) / r, du += dT / r + (1 - T) / (r * r);
    }
    pair(i, j, d, r, u, du);
  });
  // self and background (- pi Q^2 / 2 V alpha^2: proportional to 1 / V)
  const double self = al / std::sqrt(M_PI), bg = M_PI / (2 * C.V * al * al);
  for (int i = 0; i < n; i++) E -= self * q[i] * q[i], dq[i] -= 2 * self * q[i] + 2 * bg * Qt;
  E -= bg * Qt * Qt;
  for (int a = 0; a < 3; a++) vir[4 * a] += bg * Qt * Qt;
  // reciprocal space over the half space of k (k and -k together): (4 pi / V) exp(-k^2 / 4 alpha^2) / k^2 |S(k)|^2
  int M[3];
  for (int k = 0; k < 3; k++)
    M[k] = static_cast<int>(kmax * std::sqrt(C.h[k][0] * C.h[k][0] + C.h[k][1] * C.h[k][1] + C.h[k][2] * C.h[k][2]) /
                            (2 * M_PI));
  if (double(2 * M[0] + 1) * (2 * M[1] + 1) * (2 * M[2] + 1) * n > 4e9)
    throw std::runtime_error("lrc: periodic long range too large for the direct Ewald sum (cell / atom count)");
  std::vector<std::complex<double>> ph[3]; // e^{i 2 pi m f_jk}, m in [-M_k, M_k]
  for (int k = 0; k < 3; k++) {
    const int w = 2 * M[k] + 1;
    ph[k].resize(size_t(n) * w);
    for (int j = 0; j < n; j++) {
      double f = 0;
      for (int a = 0; a < 3; a++) f += x[3 * j + a] * C.inv[a][k];
      for (int m = -M[k]; m <= M[k]; m++) ph[k][size_t(j) * w + m + M[k]] = std::polar(1.0, 2 * M_PI * m * f);
    }
  }
  std::vector<std::complex<double>> e(n);
  for (int m0 = 0; m0 <= M[0]; m0++)
    for (int m1 = -M[1]; m1 <= M[1]; m1++)
      for (int m2 = -M[2]; m2 <= M[2]; m2++) {
        if (m0 == 0 && (m1 < 0 || (m1 == 0 && m2 <= 0))) continue;
        double k[3], k2 = 0;
        for (int a = 0; a < 3; a++) {
          k[a] = 2 * M_PI * (m0 * C.inv[a][0] + m1 * C.inv[a][1] + m2 * C.inv[a][2]);
          k2 += k[a] * k[a];
        }
        if (k2 > kmax * kmax) continue;
        std::complex<double> Sk = 0;
        for (int j = 0; j < n; j++) {
          e[j] = ph[0][size_t(j) * (2 * M[0] + 1) + m0 + M[0]] * ph[1][size_t(j) * (2 * M[1] + 1) + m1 + M[1]] *
                 ph[2][size_t(j) * (2 * M[2] + 1) + m2 + M[2]];
          Sk += q[j] * e[j];
        }
        const double A = 4 * M_PI / C.V * std::exp(-k2 / (4 * al * al)) / k2, S2 = std::norm(Sk);
        E += A * S2;
        for (int j = 0; j < n; j++) {
          const std::complex<double> z = std::conj(Sk) * e[j];
          dq[j] += 2 * A * z.real();
          if (forces)
            for (int a = 0; a < 3; a++) g[3 * j + a] -= 2 * A * q[j] * z.imag() * k[a];
        }
        if (forces)
          for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++)
              vir[3 * a + b] += A * S2 * ((a == b ? -1.0 : 0.0) + 2 * k[a] * k[b] * (1 / (4 * al * al) + 1 / k2));
      }
  // the linked pairs: each cluster with itself (R = 1) and with the clusters of its weak component (not with lrc_intra)
  const double *xh = c.cl.xh.data();
  for (int i = 0; i < n && !L.intra; i++)
    for (int j = i + 1; j < n; j++) {
      const int a = c.cl.sn[i], b = c.cl.sn[j];
      const double R = a == b ? 1.0 : c.cl.R[size_t(a) * s + b];
      if (R == 0) continue;
      const double d[3] = {xh[3 * j] - xh[3 * i], xh[3 * j + 1] - xh[3 * i + 1], xh[3 * j + 2] - xh[3 * i + 2]};
      const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (r <= r0) continue;
      double T, dT;
      sw(r, T, dT);
      pair(i, j, d, r, -T * R / r, -R * (dT / r - T / (r * r)));
      if (a != b) W[size_t(a) * s + b] -= 0.5 * q[i] * q[j] * T / r, W[size_t(b) * s + a] -= 0.5 * q[i] * q[j] * T / r;
    }
  out.energy = KE * E;
  if (!forces) return out;
  for (std::vector<double> *v : {&g, &dq, &W, &vir})
    for (double &t : *v) t *= KE;
  if (qproj) lrc_project_vjp(L, c.cl, proj, n, x, dq, g.data(), nullptr, vir.data());
  if (!L.intra) lrc_clusters_vjp(L, c.cl, n, x, W, g.data(), vir.data());
  lrc_q0_vjp(L, S, n, Z, x, c.q0, dq.data(), g.data(), vir.data());
  out.forces.resize(g.size());
  for (size_t k = 0; k < g.size(); k++) out.forces[k] = -g[k];
  out.virial = vir;
  return out;
}

// ---- slabs (lrc_slab): the long-range sums of a 2D-periodic tiling ----------------------------------------------------
// geometry: unit normal, in-plane area, the elongated 3D cell h (in-plane rows as given; the open row n Lz: internal for
// (T, T, F), the given cell for the (T, T, T) gap rule unless ewald2d), positions unwrapped across the vacuum
struct SlabGeom {
  int open = -1;
  double nrm[3] = {0, 0, 0}, A = 0, Lz = 0;
  std::vector<double> h, xs;
};
SlabGeom slab_geom(const LrcSpec &L, const SqeSpec &S, const LrcClusters &c, int n, const double *x, double rc_min,
                   bool own_height) {
  SlabGeom g;
  g.open = c.open;
  g.h = c.cell;
  g.A = plane_normal(c.cell.data(), g.open, g.nrm);
  g.xs.assign(x, x + 3 * size_t(n));
  if (!c.zshift.empty())
    for (int i = 0; i < n; i++)
      for (int a = 0; a < 3; a++) g.xs[3 * i + a] += c.zshift[i] * c.cell[3 * g.open + a];
  const bool given = !c.zshift.empty() && !own_height; // (T, T, T) gap rule, Yeh-Berkowitz in the given cell
  if (given) {
    g.Lz = cell3(c.cell.data()).V / g.A;
    return g;
  }
  double lo = 0, hi = 0, lmax = 0;
  for (int i = 0; i < n; i++) {
    const double z = g.nrm[0] * g.xs[3 * i] + g.nrm[1] * g.xs[3 * i + 1] + g.nrm[2] * g.xs[3 * i + 2];
    lo = i ? std::min(lo, z) : z, hi = i ? std::max(hi, z) : z;
  }
  for (int r = 0; r < 3; r++)
    if (r != g.open) lmax = std::max(lmax, std::sqrt(c.cell[3 * r] * c.cell[3 * r] + c.cell[3 * r + 1] * c.cell[3 * r + 1] +
                                                     c.cell[3 * r + 2] * c.cell[3 * r + 2]));
  // vacuum: the z-images' interactions beyond the mean field fall as exp(-2 pi G / L_inplane) (4 L: ~1e-11); no z-image
  // inside the switch (lr_R) or the real-space cutoff; Lz on a 4 A grid (locally constant in the positions)
  const double G = std::max({L.slab_vacuum, 4.0 * lmax, S.lr_R + 1.0, rc_min + 1.0});
  g.Lz = 4.0 * std::ceil((hi - lo + G) / 4.0);
  for (int a = 0; a < 3; a++) g.h[3 * g.open + a] = g.nrm[a] * g.Lz;
  return g;
}

// e^{h z} erfc(h / 2 alpha + alpha z) without overflow (exp(h z + log erfc), 0 where erfc underflows)
inline double exp_erfc(double a, double u) {
  const double e = std::erfc(u);
  return e > 0 ? std::exp(a + std::log(e)) : 0.0;
}

// separated clusters / lrc_intra in a slab:
//   E = KE [ E_2D(q) - sum_{pairs r < lr_R} q_i q_j (1 - T) / r - sum_{i < j, R_ab > 0} q_i q_j T R_ab / r^_ij ]
// (the last term not with lrc_intra), E_2D the 2D-periodic Coulomb energy (lr_combined.h: Yeh-Berkowitz with the
// net-charge terms, or the exact 2D Ewald sum)
LrcEnergy energy_slab(const LrcSpec &L, const SqeSpec &S, const LrcCondition &c, int n, const int32_t *Z,
                      const double *x, bool forces) {
  constexpr double KE = 14.399645, acc = 5.0;
  LrcEnergy out;
  out.q0 = c.q0.q0, out.q = c.q0.q0;
  const int s = c.cl.s;
  const bool exact = L.slab == LrcSpec::Slab::Ewald2D;
  const double r0 = S.lr_R - S.lr_t;
  std::vector<double> g(3 * size_t(n), 0.0), dq(n, 0.0), W(size_t(s) * s, 0.0), vir(9, 0.0);
  std::vector<double> q = c.q0.q0;
  LrcProjection proj;
  const bool qproj = L.qproj && !L.fukui_cluster;
  if (qproj) proj = lrc_project(L, c.cl, n, Z, q.data()), q = proj.q, out.q = q;
  double E = 0, Qt = 0;
  for (double v : q) Qt += v;
  auto pair = [&](int i, int j, const double *d, double r, double u, double du) {
    E += q[i] * q[j] * u;
    dq[i] += q[j] * u, dq[j] += q[i] * u;
    if (!forces) return;
    const double f = q[i] * q[j] * du / r;
    for (int a = 0; a < 3; a++) {
      g[3 * j + a] += f * d[a], g[3 * i + a] -= f * d[a];
      for (int b = 0; b < 3; b++) vir[3 * a + b] += f * d[a] * d[b];
    }
  };
  auto sw = [&](double r, double &T, double &dT) {
    const double t = (r - r0) / S.lr_t;
    T = sstep(t), dT = dsstep(t) / S.lr_t;
  };
  // real-space cutoff: the exact sum needs no z-image inside it (the internal vacuum exceeds it)
  double rc = L.slab_rc;
  if (exact && !(rc > 0)) rc = std::max(S.lr_R, 10.0);
  const SlabGeom geo = slab_geom(L, S, c.cl, n, x, exact ? rc : 0.0, exact);
  const double *xs = geo.xs.data();
  const Cell3 C = cell3(geo.h.data());
  if (!(rc > 0)) rc = std::max(S.lr_R, acc * std::cbrt(C.V) / (std::sqrt(M_PI) * std::pow(double(n), 1.0 / 6.0)));
  const double al = acc / rc, kmax = 2 * acc * al;
  periodic_pairs(n, xs, C, std::max(rc, S.lr_R), [&](int i, int j, const I3 &, const double *d, double r) {
    double u = 0, du = 0;
    if (r < rc) {
      const double e = std::erfc(al * r) / r;
      u += e, du -= (e + 2 * al / std::sqrt(M_PI) * std::exp(-al * al * r * r)) / r;
    }
    if (r < S.lr_R) {
      double T, dT;
      sw(r, T, dT);
      u -= (1 - T) / r, du += dT / r + (1 - T) / (r * r);
    }
    pair(i, j, d, r, u, du);
  });
  const double self = al / std::sqrt(M_PI);
  for (int i = 0; i < n; i++) E -= self * q[i] * q[i], dq[i] -= 2 * self * q[i];
  const double *nv = geo.nrm;
  if (!exact) {
    // 3D Ewald of the elongated cell: background (- pi Q^2 / 2 V alpha^2) and the reciprocal sum, as energy_images
    const double bg = M_PI / (2 * C.V * al * al);
    for (int i = 0; i < n; i++) dq[i] -= 2 * bg * Qt;
    E -= bg * Qt * Qt;
    for (int a = 0; a < 3; a++) vir[4 * a] += bg * Qt * Qt;
    int M[3];
    for (int k = 0; k < 3; k++)
      M[k] = static_cast<int>(kmax * std::sqrt(C.h[k][0] * C.h[k][0] + C.h[k][1] * C.h[k][1] + C.h[k][2] * C.h[k][2]) /
                              (2 * M_PI));
    if (double(2 * M[0] + 1) * (2 * M[1] + 1) * (2 * M[2] + 1) * n > 4e9)
      throw std::runtime_error("lrc slab: long range too large for the direct Ewald sum (cell / atom count)");
    std::vector<std::complex<double>> ph[3];
    for (int k = 0; k < 3; k++) {
      const int w = 2 * M[k] + 1;
      ph[k].resize(size_t(n) * w);
      for (int j = 0; j < n; j++) {
        double f = 0;
        for (int a = 0; a < 3; a++) f += xs[3 * j + a] * C.inv[a][k];
        for (int m = -M[k]; m <= M[k]; m++) ph[k][size_t(j) * w + m + M[k]] = std::polar(1.0, 2 * M_PI * m * f);
      }
    }
    std::vector<std::complex<double>> e(n);
    for (int m0 = 0; m0 <= M[0]; m0++)
      for (int m1 = -M[1]; m1 <= M[1]; m1++)
        for (int m2 = -M[2]; m2 <= M[2]; m2++) {
          if (m0 == 0 && (m1 < 0 || (m1 == 0 && m2 <= 0))) continue;
          double k[3], k2 = 0;
          for (int a = 0; a < 3; a++) {
            k[a] = 2 * M_PI * (m0 * C.inv[a][0] + m1 * C.inv[a][1] + m2 * C.inv[a][2]);
            k2 += k[a] * k[a];
          }
          if (k2 > kmax * kmax) continue;
          std::complex<double> Sk = 0;
          for (int j = 0; j < n; j++) {
            e[j] = ph[0][size_t(j) * (2 * M[0] + 1) + m0 + M[0]] * ph[1][size_t(j) * (2 * M[1] + 1) + m1 + M[1]] *
                   ph[2][size_t(j) * (2 * M[2] + 1) + m2 + M[2]];
            Sk += q[j] * e[j];
          }
          const double A = 4 * M_PI / C.V * std::exp(-k2 / (4 * al * al)) / k2, S2 = std::norm(Sk);
          E += A * S2;
          for (int j = 0; j < n; j++) {
            const std::complex<double> z = std::conj(Sk) * e[j];
            dq[j] += 2 * A * z.real();
            if (forces)
              for (int a = 0; a < 3; a++) g[3 * j + a] -= 2 * A * q[j] * z.imag() * k[a];
          }
          if (forces)
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++)
                vir[3 * a + b] += A * S2 * ((a == b ? -1.0 : 0.0) + 2 * k[a] * k[b] * (1 / (4 * al * al) + 1 / k2));
        }
    // Yeh-Berkowitz with Ballenegger-Arnold-Cerda's net-charge terms (V = A Lz; origin independent):
    //   E_c = 2 pi / V [ M_z^2 - Q sum q z^2 - Q^2 Lz^2 / 12 ],  dE_c/d strain = E_c (2 n n^T - I)
    double Mz = 0, Sz2 = 0;
    std::vector<double> zz(n);
    for (int i = 0; i < n; i++) {
      zz[i] = nv[0] * xs[3 * i] + nv[1] * xs[3 * i + 1] + nv[2] * xs[3 * i + 2];
      Mz += q[i] * zz[i], Sz2 += q[i] * zz[i] * zz[i];
    }
    const double V = geo.A * geo.Lz, p = 2 * M_PI / V;
    const double Ec = p * (Mz * Mz - Qt * Sz2 - Qt * Qt * geo.Lz * geo.Lz / 12);
    E += Ec;
    for (int i = 0; i < n; i++) {
      dq[i] += p * (2 * Mz * zz[i] - Sz2 - Qt * zz[i] * zz[i] - Qt * geo.Lz * geo.Lz / 6);
      if (forces)
        for (int a = 0; a < 3; a++) g[3 * i + a] += p * 2 * q[i] * (Mz - Qt * zz[i]) * nv[a];
    }
    if (forces)
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) vir[3 * a + b] += Ec * (2 * nv[a] * nv[b] - (a == b ? 1.0 : 0.0));
  } else {
    // exact 2D Ewald (Parry), in the frame e1 = u / |u|, e2 = n x e1, n; half plane of in-plane wave vectors h:
    //   E_h = (pi / A) sum_{i,j} q_i q_j cos(h . rho_ij) Phi(h, z_ij) / h,
    //   Phi = e^{h z} erfc(h / 2a + a z) + e^{-h z} erfc(h / 2a - a z);
    //   E_0 = -(pi / A) sum_{i,j} q_i q_j [ z_ij erf(a z_ij) + exp(-a^2 z_ij^2) / (a sqrt(pi)) ]  (h = 0, divergence dropped)
    const double *uu = &geo.h[3 * ((geo.open + 1) % 3)], *vv = &geo.h[3 * ((geo.open + 2) % 3)];
    double B[3][3];
    const double ul = std::sqrt(uu[0] * uu[0] + uu[1] * uu[1] + uu[2] * uu[2]);
    for (int a = 0; a < 3; a++) B[0][a] = uu[a] / ul, B[2][a] = nv[a];
    B[1][0] = nv[1] * B[0][2] - nv[2] * B[0][1], B[1][1] = nv[2] * B[0][0] - nv[0] * B[0][2],
    B[1][2] = nv[0] * B[0][1] - nv[1] * B[0][0];
    auto dot = [](const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const double v0 = dot(vv, B[0]), v1 = dot(vv, B[1]), det = ul * v1, A = std::abs(det);
    const double b1[2] = {2 * M_PI * v1 / det, -2 * M_PI * v0 / det}, b2[2] = {0.0, 2 * M_PI * ul / det};
    std::vector<double> rx(n), ry(n), zc(n);
    for (int i = 0; i < n; i++) rx[i] = dot(&xs[3 * i], B[0]), ry[i] = dot(&xs[3 * i], B[1]), zc[i] = dot(&xs[3 * i], B[2]);
    std::vector<double> gc(3 * size_t(n), 0.0); // gradient in the frame
    double Wc[3][3] = {};
    const double hmax = kmax, vl = std::sqrt(v0 * v0 + v1 * v1), sp = std::sqrt(M_PI);
    const int M1 = static_cast<int>(std::ceil(hmax * ul / (2 * M_PI))), M2 = static_cast<int>(std::ceil(hmax * vl / (2 * M_PI)));
    std::vector<std::complex<double>> e(n);
    double Erec = 0;
    for (int m1 = 0; m1 <= M1; m1++)
      for (int m2 = -M2; m2 <= M2; m2++) {
        if (m1 == 0 && m2 <= 0) continue;
        const double hv[2] = {m1 * b1[0] + m2 * b2[0], m1 * b1[1] + m2 * b2[1]};
        const double hh = std::sqrt(hv[0] * hv[0] + hv[1] * hv[1]);
        if (hh > hmax) continue;
        for (int j = 0; j < n; j++) e[j] = std::polar(1.0, hv[0] * rx[j] + hv[1] * ry[j]);
        double dEdh = 0;
        const double Gh = std::exp(-hh * hh / (4 * al * al));
        { // i = j: Phi(h, 0) = 2 erfc(h / 2a)
          const double P0 = 2 * std::erfc(hh / (2 * al)), dP0 = -2 * Gh / (al * sp);
          for (int i = 0; i < n; i++) {
            const double t = M_PI / A * q[i] * q[i];
            Erec += t * P0 / hh, dq[i] += 2 * M_PI / A * q[i] * P0 / hh, dEdh += t * (dP0 / hh - P0 / (hh * hh));
          }
        }
        for (int i = 0; i < n; i++)
          for (int j = i + 1; j < n; j++) {
            const std::complex<double> z = e[j] * std::conj(e[i]); // e^{i h . rho_ij}
            const double cs = z.real(), sn = z.imag(), zij = zc[j] - zc[i];
            const double ep = exp_erfc(hh * zij, hh / (2 * al) + al * zij), em = exp_erfc(-hh * zij, hh / (2 * al) - al * zij);
            const double Phi = ep + em, Psi = ep - em, Gs = Gh * std::exp(-al * al * zij * zij);
            const double pre = 2 * M_PI / A, qq = q[i] * q[j];
            Erec += pre * qq * cs * Phi / hh;
            dq[i] += pre * q[j] * cs * Phi / hh, dq[j] += pre * q[i] * cs * Phi / hh;
            if (!forces) continue;
            const double dr = -pre * qq * sn * Phi / hh, dz = pre * qq * cs * Psi; // dE/d(h . rho), dE/dz
            const double d3[3] = {dr * hv[0], dr * hv[1], dz};
            for (int a = 0; a < 3; a++) gc[3 * j + a] += d3[a], gc[3 * i + a] -= d3[a];
            const double rij[2] = {rx[j] - rx[i], ry[j] - ry[i]};
            Wc[2][2] += dz * zij;
            for (int a = 0; a < 2; a++) Wc[a][2] += d3[a] * zij, Wc[2][a] += d3[a] * zij;
            (void)rij;
            dEdh += pre * qq * cs * ((zij * Psi - 2 * Gs / (al * sp)) / hh - Phi / (hh * hh));
          }
        if (forces)
          for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++) Wc[a][b] -= dEdh * hv[a] * hv[b] / hh;
      }
    double E0 = 0;
    for (int i = 0; i < n; i++) {
      const double t = M_PI / A / (al * sp);
      E0 -= t * q[i] * q[i], dq[i] -= 2 * t * q[i];
      for (int j = i + 1; j < n; j++) {
        const double zij = zc[j] - zc[i], er = std::erf(al * zij);
        const double f = zij * er + std::exp(-al * al * zij * zij) / (al * sp), pre = 2 * M_PI / A;
        E0 -= pre * q[i] * q[j] * f, dq[i] -= pre * q[j] * f, dq[j] -= pre * q[i] * f;
        if (!forces) continue;
        const double dz = -pre * q[i] * q[j] * er;
        gc[3 * j + 2] += dz, gc[3 * i + 2] -= dz, Wc[2][2] += dz * zij;
      }
    }
    E += Erec + E0;
    if (forces) {
      for (int a = 0; a < 2; a++) Wc[a][a] -= Erec + E0; // both ~ 1 / A
      for (int i = 0; i < n; i++)
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) g[3 * i + a] += B[b][a] * gc[3 * i + b];
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
          for (int k = 0; k < 3; k++)
            for (int l = 0; l < 3; l++) vir[3 * a + b] += B[k][a] * Wc[k][l] * B[l][b];
    }
  }
  // separated clusters (not lrc_intra): the linked pairs, as energy_images
  const double *xh = c.cl.xh.data();
  for (int i = 0; i < n && !L.intra; i++)
    for (int j = i + 1; j < n; j++) {
      const int a = c.cl.sn[i], b = c.cl.sn[j];
      const double R = a == b ? 1.0 : c.cl.R[size_t(a) * s + b];
      if (R == 0) continue;
      const double d[3] = {xh[3 * j] - xh[3 * i], xh[3 * j + 1] - xh[3 * i + 1], xh[3 * j + 2] - xh[3 * i + 2]};
      const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (r <= r0) continue;
      double T, dT;
      sw(r, T, dT);
      pair(i, j, d, r, -T * R / r, -R * (dT / r - T / (r * r)));
      if (a != b) W[size_t(a) * s + b] -= 0.5 * q[i] * q[j] * T / r, W[size_t(b) * s + a] -= 0.5 * q[i] * q[j] * T / r;
    }
  out.energy = KE * E;
  if (!forces) return out;
  for (std::vector<double> *v : {&g, &dq, &W, &vir})
    for (double &t : *v) t *= KE;
  if (qproj) lrc_project_vjp(L, c.cl, proj, n, x, dq, g.data(), nullptr, vir.data());
  if (!L.intra) lrc_clusters_vjp(L, c.cl, n, x, W, g.data(), vir.data());
  lrc_q0_vjp(L, S, n, Z, x, c.q0, dq.data(), g.data(), vir.data());
  out.forces.resize(g.size());
  for (size_t k = 0; k < g.size(); k++) out.forces[k] = -g[k];
  out.virial = vir;
  return out;
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
  // lrc_intra: every pair, inside clusters too, unmasked (m = 1: R = 0 in the energy), so no cluster gradient
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      const int a = c.cl.sn[i], b = c.cl.sn[j];
      if (a == b && !L.intra) continue;
      const double r = dist(x, i, j);
      if (r <= r0) continue;
      const double t = (r - r0) / S.lr_t;
      const double T = sstep(t), dT = dsstep(t) / S.lr_t;
      const double m = L.intra ? 1.0 : 1.0 - c.cl.R[size_t(a) * s + b];
      if (m == 0) continue;
      const double k = KE / r, e = q[i] * q[j] * k;
      out.energy += e * T * m;
      if (!forces) continue;
      add_pair(g.data(), x, i, j, r, q[i] * q[j] * m * (dT * k - T * k / r));
      dq[i] += q[j] * k * T * m, dq[j] += q[i] * k * T * m;
      if (!L.intra) W[size_t(a) * s + b] -= 0.5 * e * T, W[size_t(b) * s + a] -= 0.5 * e * T;
    }
  if (!forces) return out;
  unplateau();
  if (!L.intra) lrc_clusters_vjp(L, c.cl, n, x, W, g.data());
  lrc_q0_vjp(L, S, n, Z, x, c.q0, dq.data(), g.data());
  for (size_t k = 0; k < g.size(); k++) out.forces[k] = -g[k];
  return out;
}
} // namespace

std::vector<double> lrc_project_charges(const LrcSpec &L, int n, const int32_t *Z, const double *x, const double *q,
                                        const double *cell) {
  const LrcClusters c = lrc_clusters(L, n, x, cell);
  return lrc_project(L, c, n, Z, q).q;
}

LrcEnergy lrc_energy(const LrcSpec &L, const SqeSpec &S, const LrcCondition &c, int n, const int32_t *Z,
                     const double *x, double Q, bool forces) {
  LrcEnergy out;
  if (forces) out.forces.assign(3 * size_t(n), 0.0);
  if (n == 0) return out;
  if (L.intra) { // every pair beyond the switch, inside clusters and bulk cells too
    if (!L.graph || L.mp || L.pol) throw std::runtime_error("lrc_intra: needs the graph q0 mode without lrc_mp / lrc_pol");
    if (c.q0.q0.size() != size_t(n)) throw std::logic_error("lrc_intra: no reference charges in the conditioning");
    if (c.cl.slab()) return energy_slab(L, S, c, n, Z, x, forces);
    return c.cl.periodic() ? energy_images(L, S, c, n, Z, x, forces) : energy_fixed(L, S, c, n, Z, x, forces);
  }
  if (c.cl.s == 1 && !c.cl.images()) return out; // one cluster / bulk: E*(gamma) - E*(gamma) = 0 exactly
  if (c.cl.images()) {
    if (!L.graph || L.mp || L.pol)
      throw std::runtime_error("lrc: separated clusters in a periodic cell need the graph q0 mode without lrc_mp / lrc_pol");
    if (c.cl.slab()) return energy_slab(L, S, c, n, Z, x, forces);
    return energy_images(L, S, c, n, Z, x, forces);
  }
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
