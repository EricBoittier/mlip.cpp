#include "plato.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>
#include <nlohmann/json.hpp>

#ifdef MLIPCPP_HAVE_VESIN
#include <vesin.h>
#endif
#include "mlipcpp/neighbor_list.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace mlipcpp::plato {

using runtime::PlatoInputs;
using runtime::PlatoSpec;

namespace {

constexpr size_t GRAPH_SIZE = 32768;

bool primary(int i, int j, const std::array<int32_t, 3> &s) {
  if (i != j) return i < j;
  for (int c = 0; c < 3; c++)
    if (s[c] != 0) return s[c] > 0;
  return false;
}

} // namespace

// full list, grouped by centre and sorted, with reverse edges (as GraphModel's host_edges)
Edges neighbor_edges(const AtomicSystem &system, double cutoff, void *&vesin_state) {
  const int n = system.num_atoms();
  const float *x = system.positions();
  const Cell *cell = system.cell();
  double H[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  bool periodic[3] = {false, false, false};
  if (cell)
    for (int a = 0; a < 3; a++) {
      periodic[a] = cell->periodic[a];
      for (int b = 0; b < 3; b++) H[a][b] = cell->matrix[a][b];
    }
  std::vector<int32_t> pi, pj;
  std::vector<std::array<int32_t, 3>> ps;
#ifdef MLIPCPP_HAVE_VESIN
  if (!vesin_state) vesin_state = new VesinNeighborList();
  auto *list = static_cast<VesinNeighborList *>(vesin_state);
  std::vector<double> points(3 * size_t(n));
  for (size_t k = 0; k < points.size(); k++) points[k] = x[k];
  VesinOptions options{};
  options.cutoff = cutoff;
  options.full = true;
  options.sorted = false;
  options.algorithm = VesinAutoAlgorithm;
  options.skin = 0.0;
  options.n_threads = 0;
  options.return_shifts = true;
  const char *error = nullptr;
  if (vesin_neighbors(reinterpret_cast<const double(*)[3]>(points.data()), n, H, periodic, VesinDevice{VesinCPU, 0},
                      options, list, &error) != 0)
    throw std::runtime_error(std::string("vesin: ") + (error ? error : "unknown error"));
  pi.resize(list->length), pj.resize(list->length), ps.resize(list->length);
  for (size_t e = 0; e < list->length; e++) {
    pi[e] = static_cast<int32_t>(list->pairs[e][0]), pj[e] = static_cast<int32_t>(list->pairs[e][1]);
    ps[e] = {list->shifts[e][0], list->shifts[e][1], list->shifts[e][2]};
  }
#else
  (void)vesin_state;
  NeighborList nl = NeighborListBuilder(NeighborListOptions{static_cast<float>(cutoff), true, false}).build(system);
  pi = nl.centers, pj = nl.neighbors;
  ps = nl.cell_shifts.empty() ? std::vector<std::array<int32_t, 3>>(pi.size(), {0, 0, 0}) : nl.cell_shifts;
#endif
  Edges out;
  const size_t E = pi.size();
  out.start.assign(n + 1, 0);
  for (int32_t i : pi) out.start[i + 1]++;
  for (int i = 0; i < n; i++) out.start[i + 1] += out.start[i];
  out.j.resize(E), out.shift.resize(E), out.rev.assign(E, -1), out.v.resize(3 * E);
  {
    std::vector<int64_t> fill(out.start.begin(), out.start.end() - 1);
    for (size_t e = 0; e < E; e++) {
      const int64_t k = fill[pi[e]]++;
      out.j[k] = pj[e], out.shift[k] = ps[e];
    }
  }
  auto key = [&](int64_t e) { return std::tuple(out.j[e], out.shift[e][0], out.shift[e][1], out.shift[e][2]); };
  for (int i = 0; i < n; i++) {
    const int64_t b = out.start[i], c = out.start[i + 1];
    std::vector<int64_t> order(c - b);
    std::iota(order.begin(), order.end(), b);
    std::sort(order.begin(), order.end(), [&](int64_t p, int64_t q) { return key(p) < key(q); });
    std::vector<int32_t> j(c - b);
    std::vector<std::array<int32_t, 3>> s(c - b);
    for (int64_t k = 0; k < c - b; k++) j[k] = out.j[order[k]], s[k] = out.shift[order[k]];
    std::copy(j.begin(), j.end(), out.j.begin() + b);
    std::copy(s.begin(), s.end(), out.shift.begin() + b);
  }
  for (int i = 0; i < n; i++)
    for (int64_t e = out.start[i]; e < out.start[i + 1]; e++) {
      const int j = out.j[e];
      const auto &s = out.shift[e];
      const bool p = primary(i, j, s);
      const int a = p ? i : j, b2 = p ? j : i;
      const double sg = p ? 1.0 : -1.0;
      for (int c = 0; c < 3; c++) {
        const double t = sg * s[0] * H[0][c] + sg * s[1] * H[1][c] + sg * s[2] * H[2][c];
        out.v[3 * e + c] = static_cast<float>(sg * (double(x[3 * b2 + c]) - double(x[3 * a + c]) + t));
      }
      const auto want = std::tuple(i, -s[0], -s[1], -s[2]);
      int64_t lo = out.start[j], hi = out.start[j + 1];
      while (lo < hi) {
        const int64_t mid = (lo + hi) / 2;
        if (key(mid) < want) lo = mid + 1;
        else hi = mid;
      }
      if (lo < out.start[j + 1] && key(lo) == want) out.rev[e] = static_cast<int32_t>(lo);
      if (out.rev[e] < 0) throw std::runtime_error("Plato: no reverse edge");
    }
  return out;
}

void free_vesin(void *&vesin_state) {
#ifdef MLIPCPP_HAVE_VESIN
  if (vesin_state) {
    auto *l = static_cast<VesinNeighborList *>(vesin_state);
    vesin_free(l);
    delete l;
  }
#endif
  vesin_state = nullptr;
}

namespace {

void inverse3(const double A[3][3], double I[3][3], double &det) {
  det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  const double id = 1.0 / det;
  I[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * id;
  I[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * id;
  I[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * id;
  I[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * id;
  I[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * id;
  I[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * id;
  I[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * id;
  I[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * id;
  I[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * id;
}

// |a| of the float32 cell row, in double: the exact rule. PyTorch's float32 norm
// rounds (e.g. 30.0000000233 -> 30.0, so ceil gives 30 where this gives 31); its
// float64 model agrees with this
double row_norm(const float *a) {
  return std::sqrt(double(a[0]) * a[0] + double(a[1]) * a[1] + double(a[2]) * a[2]);
}
} // namespace
int mesh_points(const float *a, double spacing, bool round_fft) {
  const int m = std::max(8, static_cast<int>(std::ceil(row_norm(a) / spacing)));
  if (!round_fft) return m;
  for (int p = 8;; p *= 2)
    for (int f : {p, 5 * p / 4, 3 * p / 2})
      if (f >= m) return f;
}
namespace {

int env_mesh(int out[3]) {
  const char *e = std::getenv("MLIPCPP_PLATO_MESH");
  if (!e || !*e) return 0;
  int a = 0, b = 0, c = 0;
  const int k = std::sscanf(e, "%d,%d,%d", &a, &b, &c);
  if (k == 1) b = c = a;
  else if (k != 3) throw std::runtime_error("MLIPCPP_PLATO_MESH: M or M0,M1,M2");
  out[0] = a, out[1] = b, out[2] = c;
  return 1;
}

} // namespace

struct PlatoModel::Graph {
  ggml_context *ctx = nullptr;
  ggml_backend_buffer *inputs_buf = nullptr;
  ggml_cgraph *graph = nullptr, *forward = nullptr;
  PlatoInputs in;
  runtime::PlatoOutputs out;
  ggml_tensor *grad_ev = nullptr, *grad_pos = nullptr, *grad_eps = nullptr, *loss_grad = nullptr;
  std::vector<ggml_tensor *> inputs;
  // key
  int n = 0, max_len = 0, mesh[3] = {0, 0, 0};
  int64_t E = 0, nk = 0;
  bool periodic = false, mesh_on = false, les_on = false;
  ~Graph() {
    if (inputs_buf) ggml_backend_buffer_free(inputs_buf);
    if (ctx) ggml_free(ctx);
  }
};

PlatoModel::PlatoModel(PlatoOptions options) : options_(options) {
  if (const char *e = std::getenv("MLIPCPP_PLATO_FOLD_IN_GRAPH")) options_.fold_in_graph = options_.fold_in_graph || (*e && *e != '0');
  if (const char *e = std::getenv("MLIPCPP_PLATO_DECOMPOSED_LN")) options_.decomposed_layer_norm = options_.decomposed_layer_norm || (*e && *e != '0');
  if (!options_.fixed_mesh[0]) env_mesh(options_.fixed_mesh);
  // A/B of the parts (timing): MLIPCPP_PLATO_NO_MESH=1, MLIPCPP_PLATO_NO_LES=1
  if (const char *e = std::getenv("MLIPCPP_PLATO_NO_MESH")) options_.mesh = options_.mesh && !(*e && *e != '0');
  if (const char *e = std::getenv("MLIPCPP_PLATO_NO_LES")) options_.les = options_.les && !(*e && *e != '0');
}

PlatoModel::~PlatoModel() {
  graph_.reset();
  if (galloc_) ggml_gallocr_free(galloc_);
  if (buf_w_) ggml_backend_buffer_free(buf_w_);
  if (ctx_w_) ggml_free(ctx_w_);
  free_vesin(vesin_);
}

const char *PlatoModel::backend_name() const { return backend_ ? ggml_backend_name(backend_->primary()) : "none"; }

bool PlatoModel::load_from_gguf(const std::string &path) {
  ggml_context *host = nullptr;
  gguf_context *g = gguf_init_from_file(path.c_str(), {false, &host});
  if (!g) throw std::runtime_error("Plato: cannot read " + path);
  auto str = [&](const char *k) {
    const int64_t id = gguf_find_key(g, k);
    if (id < 0) throw std::runtime_error(std::string("Plato: no ") + k);
    return std::string(gguf_get_val_str(g, id));
  };
  if (str("general.architecture") != "plato-pet-lr") throw std::runtime_error("Plato: not a plato-pet-lr GGUF");
  const nlohmann::json hy = nlohmann::json::parse(str("plato.hypers"));
  const nlohmann::json info = nlohmann::json::parse(str("plato.dataset_info"));
  energy_scale_ = gguf_get_val_f64(g, gguf_find_key(g, "plato.energy_scale"));
  if (const int64_t k = gguf_find_key(g, "plato.mesh_round"); k >= 0) {
    const std::string r = gguf_get_val_str(g, k);
    if (r != "fft" && r != "none") throw std::runtime_error("Plato: unknown plato.mesh_round " + r);
    mesh_round_fft_ = r == "fft";
  }
  {
    const int64_t kt = gguf_find_key(g, "plato.composition_types"), ke = gguf_find_key(g, "plato.composition_energies");
    const size_t nc = gguf_get_arr_n(g, kt);
    const int32_t *zt = static_cast<const int32_t *>(gguf_get_arr_data(g, kt));
    const double *ze = static_cast<const double *>(gguf_get_arr_data(g, ke));
    for (size_t i = 0; i < nc; i++) composition_[zt[i]] = ze[i];
  }
  int idx = 0;
  for (int z : info["atomic_types"].get<std::vector<int>>()) species_index_[z] = idx++;
  if (hy["frame_mode"] != "attention") throw std::runtime_error("Plato: frame_mode must be attention");
  spec_.layers = hy["num_layers"], spec_.heads = hy["num_heads"], spec_.d = hy["d_model"], spec_.n_rbf = hy["num_rbf"];
  spec_.n_basis = hy["n_basis"], spec_.couple_at = hy["couple_at"], spec_.mesh_channels = hy["mesh_channels"];
  spec_.n_charges = hy["num_charges"], spec_.cutoff = hy["cutoff"], spec_.mesh_sigma = hy["mesh_sigma"];
  spec_.les_sigma = hy["les_sigma"], spec_.les_kcut = hy["les_kcut"];
  spec_.mesh = hy["mesh_coupling"].get<bool>() && options_.mesh;
  spec_.les = hy["les"].get<bool>() && options_.les;
  spec_.chi = spec_.les && hy.value("electronegativity", std::string("none")) == "linear";
  spec_.chi_bound = hy.value("dchi_bound", 0.0f);
  spec_.n_types = idx;
  spec_.fold_in_graph = options_.fold_in_graph;
  spec_.decomposed_layer_norm = options_.decomposed_layer_norm;
  mesh_spacing_ = hy["mesh_spacing"];
  {
    // plato.mesh_normalization, else the hypers' (a v6 checkpoint through an older exporter),
    // else per_point: every GGUF from before the setting (checkpoint v5 and older)
    std::string norm;
    if (const int64_t k = gguf_find_key(g, "plato.mesh_normalization"); k >= 0) norm = gguf_get_val_str(g, k);
    const std::string hnorm = hy.value("mesh_normalization", std::string());
    if (!norm.empty() && !hnorm.empty() && norm != hnorm)
      throw std::runtime_error("Plato: plato.mesh_normalization " + norm + " but the hypers say " + hnorm);
    if (norm.empty()) norm = hnorm;
    mesh_normalization_stated_ = !norm.empty();
    const std::string &want = options_.mesh_normalization;
    if (!want.empty()) {
      if (mesh_normalization_stated_ && want != norm)
        throw std::runtime_error("Plato: mesh_normalization " + want + " requested, the GGUF is " + norm);
      norm = want;
    }
    if (norm.empty()) norm = "per_point";
    if (norm != "physical" && norm != "per_point") throw std::runtime_error("Plato: unknown mesh_normalization " + norm);
    mesh_normalization_ = norm;
    spec_.mesh_physical = norm == "physical";
  }
  if (!hy["rope"].get<bool>()) throw std::runtime_error("Plato: needs rope");
  if (!hy["num_framed_layers"].is_null() && hy["num_framed_layers"].get<int>() != spec_.layers)
    throw std::runtime_error("Plato: num_framed_layers < num_layers is not built");

  // host copies of the float tensors, torch shapes
  std::map<std::string, std::vector<float>> raw;
  std::map<std::string, std::vector<int64_t>> shapes;
  const int64_t nt = gguf_get_n_tensors(g);
  for (int64_t i = 0; i < nt; i++) {
    const std::string name = gguf_get_tensor_name(g, i);
    ggml_tensor *t = ggml_get_tensor(host, name.c_str());
    if (t->type != GGML_TYPE_F32) continue;
    raw[name].assign(static_cast<float *>(t->data), static_cast<float *>(t->data) + ggml_nelements(t));
    std::vector<int64_t> s;
    for (int k = ggml_n_dims(t) - 1; k >= 0; k--) s.push_back(t->ne[k]);
    shapes[name] = s;
  }
  // frames and RoPE directions (buffers)
  const auto &GR = raw.at("G_R"), &Gw = raw.at("G_w"), &om = raw.at("omega");
  spec_.frames = static_cast<int>(Gw.size());
  spec_.frame_w.assign(Gw.begin(), Gw.end());
  spec_.frame_R.assign(GR.begin(), GR.end());
  spec_.omega.assign(om.begin(), om.end());
  const int F = spec_.frames, hd = spec_.head_dim() / 2;
  if (static_cast<int>(om.size()) != 3 * hd) throw std::runtime_error("Plato: omega size");

  std::map<std::string, std::vector<float>> derived;
  std::map<std::string, std::vector<int64_t>> dshape; // ggml ne
  std::vector<std::string> prefixes;
  if (!spec_.fold_in_graph)
    for (auto &[k, v] : runtime::plato_fold_weights(raw, shapes, prefixes)) {
      const auto &s = shapes.at(k.substr(5) + ".W");
      dshape[k] = {s[2], s[1]};
      derived[k] = std::move(v);
    }
  {
    std::vector<float> ang(3 * F * hd);
    for (int f = 0; f < F; f++)
      for (int c = 0; c < hd; c++)
        for (int x = 0; x < 3; x++) {
          double a = 0;
          for (int bb = 0; bb < 3; bb++) a += spec_.frame_R[(f * 3 + x) * 3 + bb] * spec_.omega[c * 3 + bb];
          ang[(f * hd + c) * 3 + x] = static_cast<float>(a);
        }
    derived["rope.angles"] = ang, dshape["rope.angles"] = {3, F * hd};
    for (int f = 0; f < F; f++) {
      std::vector<float> rt(9);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) rt[i * 3 + j] = static_cast<float>(spec_.frame_R[(f * 3 + j) * 3 + i]);
      const std::string k = "rope.frame_t." + std::to_string(f);
      derived[k] = rt, dshape[k] = {3, 3};
    }
    std::vector<float> mu(spec_.n_rbf);
    for (int k = 0; k < spec_.n_rbf; k++)
      mu[k] = static_cast<float>(spec_.cutoff * static_cast<float>(k) / static_cast<float>(spec_.n_rbf - 1));
    derived["rbf.mu"] = mu, dshape["rbf.mu"] = {spec_.n_rbf};
  }

  backend_ = BackendProvider::create(options_.backend, options_.device);
  const size_t n_w = raw.size() + derived.size();
  ctx_w_ = ggml_init({ggml_tensor_overhead() * (n_w + 8), nullptr, true});
  std::vector<std::pair<ggml_tensor *, const std::vector<float> *>> up;
  for (auto &[name, v] : raw) {
    ggml_tensor *src = ggml_get_tensor(host, name.c_str());
    ggml_tensor *t = ggml_new_tensor(ctx_w_, GGML_TYPE_F32, ggml_n_dims(src), src->ne);
    ggml_set_name(t, name.c_str());
    weights_[name] = t, up.emplace_back(t, &v);
  }
  for (auto &[name, v] : derived) {
    const auto &s = dshape.at(name);
    ggml_tensor *t = ggml_new_tensor(ctx_w_, GGML_TYPE_F32, static_cast<int>(s.size()), s.data());
    ggml_set_name(t, name.c_str());
    weights_[name] = t, up.emplace_back(t, &v);
  }
  buf_w_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w_, backend_->buffer_type());
  for (auto &[t, v] : up) ggml_backend_tensor_set(t, v->data(), 0, ggml_nbytes(t));
  gguf_free(g);
  ggml_free(host);
  return true;
}

ModelResult PlatoModel::predict(const AtomicSystem &system, bool compute_forces) {
  const int n = system.num_atoms();
  const int32_t *Z = system.atomic_numbers();
  const Cell *cell = system.cell();
  bool periodic = false;
  if (cell) {
    const int np = cell->periodic[0] + cell->periodic[1] + cell->periodic[2];
    if (np != 0 && np != 3 && (spec_.mesh || spec_.les))
      throw std::runtime_error("Plato-PET-LR long range needs full or no PBC");
    periodic = np == 3;
  }
  std::vector<int32_t> species(n);
  double composition = 0;
  for (int i = 0; i < n; i++) {
    auto it = species_index_.find(Z[i]);
    if (it == species_index_.end()) throw std::runtime_error("Plato: atomic number " + std::to_string(Z[i]));
    species[i] = it->second;
    if (auto c = composition_.find(Z[i]); c != composition_.end()) composition += c->second;
  }
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  const auto t0 = clk::now();
  const Edges ed = neighbor_edges(system, spec_.cutoff, vesin_);
  const auto t1 = clk::now();
  const int64_t E = static_cast<int64_t>(ed.j.size());
  int max_len = 1;
  for (int i = 0; i < n; i++) max_len = std::max<int>(max_len, 1 + static_cast<int>(ed.start[i + 1] - ed.start[i]));

  // cell-dependent long-range constants (cached on the cell)
  const bool mesh_on = periodic && spec_.mesh, les_on = spec_.les;
  int mesh[3] = {0, 0, 0};
  if (mesh_on)
    for (int a = 0; a < 3; a++)
      mesh[a] = options_.fixed_mesh[0] > 0
                    ? options_.fixed_mesh[a]
                    : mesh_points(cell->matrix[a], mesh_spacing_, mesh_round_fft_);
  CellConsts &K = consts_;
  bool same_cell = K.valid && K.periodic == periodic && K.mesh[0] == mesh[0] && K.mesh[1] == mesh[1] && K.mesh[2] == mesh[2];
  if (same_cell && periodic) same_cell = std::memcmp(K.cell, &cell->matrix[0][0], sizeof(K.cell)) == 0;
  if (!same_cell) {
    K = CellConsts{};
    K.valid = true, K.periodic = periodic;
    for (int a = 0; a < 3; a++) K.mesh[a] = mesh[a];
    if (periodic) {
      std::memcpy(K.cell, &cell->matrix[0][0], sizeof(K.cell));
      double C[3][3], det = 0;
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) C[a][b] = cell->matrix[a][b];
      inverse3(C, K.inv, det);
      K.volume = std::abs(det);
    }
    const auto &Ci = K.inv;
    if (mesh_on) {
      const int M0 = mesh[0], M1 = mesh[1], M2 = mesh[2], Mh2 = M2 / 2 + 1;
      const int64_t Mh = int64_t(M0) * M1 * Mh2;
      K.mesh_k.resize(3 * Mh), K.mesh_pad.assign(Mh, 0.0f), K.mesh_deconv.resize(Mh);
      auto freq = [](int i, int M) { return i <= (M - 1) / 2 ? i : i - M; };
      for (int i0 = 0; i0 < M0; i0++)
        for (int i1 = 0; i1 < M1; i1++)
          for (int i2 = 0; i2 < Mh2; i2++) {
            const int64_t e = (int64_t(i0) * M1 + i1) * Mh2 + i2;
            const double m[3] = {double(freq(i0, M0)), double(freq(i1, M1)), double(i2)};
            for (int b = 0; b < 3; b++)
              K.mesh_k[3 * e + b] = static_cast<float>(2 * M_PI * (m[0] * Ci[b][0] + m[1] * Ci[b][1] + m[2] * Ci[b][2]));
            double dc = 1;
            for (int a = 0; a < 3; a++) {
              const double f = (4 + 2 * std::cos(2 * M_PI * m[a] / mesh[a])) / 6;
              dc *= f * f;
            }
            const bool zero = i0 == 0 && i1 == 0 && i2 == 0;
            K.mesh_pad[e] = zero ? 1.0f : 0.0f;
            K.mesh_deconv[e] = zero ? 0.0f : static_cast<float>(1.0 / dc);
          }
    }
    if (les_on && periodic) {
      const double kc = spec_.les_kcut / spec_.les_sigma, wd = 1.0 / spec_.les_sigma;
      int nmax[3];
      for (int a = 0; a < 3; a++) nmax[a] = static_cast<int>(std::floor(kc * row_norm(cell->matrix[a]) / (2 * M_PI)));
      for (int a = -nmax[0]; a <= nmax[0]; a++)
        for (int b = -nmax[1]; b <= nmax[1]; b++)
          for (int c = -nmax[2]; c <= nmax[2]; c++) {
            if (!(a > 0 || (a == 0 && b > 0) || (a == 0 && b == 0 && c > 0))) continue;
            double k[3], k2 = 0;
            for (int q = 0; q < 3; q++) k[q] = 2 * M_PI * (a * Ci[q][0] + b * Ci[q][1] + c * Ci[q][2]), k2 += k[q] * k[q];
            if (!(k2 < kc * kc)) continue;
            for (int q = 0; q < 3; q++) K.les_k.push_back(static_cast<float>(k[q]));
            K.les_n.push_back(static_cast<float>(2 * M_PI * a)), K.les_n.push_back(static_cast<float>(2 * M_PI * b));
            K.les_n.push_back(static_cast<float>(2 * M_PI * c));
            K.les_taper.push_back(std::sqrt(k2) >= kc - wd ? 1.0f : 0.0f);
          }
    }
    consts_uploaded_ = false;
  }
  const double V = K.volume;
  const auto &Ci = K.inv;
  const int64_t nk = static_cast<int64_t>(K.les_taper.size());

  // the graph for this shape
  const bool backward = true;
  Graph *G = graph_.get();
  const bool same = G && G->n == n && G->E == E && G->max_len == max_len && G->periodic == periodic &&
                    G->mesh_on == mesh_on && G->les_on == les_on && G->nk == nk && G->mesh[0] == mesh[0] &&
                    G->mesh[1] == mesh[1] && G->mesh[2] == mesh[2];
  parts_.rebuilt = !same;
  if (!same) {
    consts_uploaded_ = false;
    graph_.reset();
    auto g = std::make_unique<Graph>();
    g->n = n, g->E = E, g->max_len = max_len, g->periodic = periodic, g->mesh_on = mesh_on, g->les_on = les_on, g->nk = nk;
    for (int a = 0; a < 3; a++) g->mesh[a] = mesh[a];
    g->ctx = ggml_init({ggml_tensor_overhead() * GRAPH_SIZE * 2 + ggml_graph_overhead_custom(GRAPH_SIZE, true) * 2,
                        nullptr, true});
    ggml_context *ctx = g->ctx;
    auto input = [&](ggml_type type, int64_t ne0, int64_t ne1, const char *name) {
      ggml_tensor *t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1) : ggml_new_tensor_1d(ctx, type, ne0);
      ggml_set_name(t, name);
      ggml_set_input(t);
      g->inputs.push_back(t);
      return t;
    };
    const int64_t T = n + E, hdF = int64_t(spec_.head_dim() / 2) * spec_.frames;
    PlatoInputs &in = g->in;
    in.species = input(GGML_TYPE_I32, n, 0, "species");
    in.edge_center_species = input(GGML_TYPE_I32, E, 0, "edge_center_species");
    in.neighbor_species = input(GGML_TYPE_I32, E, 0, "neighbor_species");
    in.edge_vectors = input(GGML_TYPE_F32, 3, E, "edge_vectors");
    in.reverse = input(GGML_TYPE_I32, T, 0, "reverse");
    in.offsets = input(GGML_TYPE_I32, n + 1, 0, "offsets");
    in.centers = input(GGML_TYPE_I32, E, 0, "centers");
    in.zero_ang = input(GGML_TYPE_F32, hdF, n, "zero_ang");
    in.rbf0 = input(GGML_TYPE_F32, spec_.n_rbf, 0, "rbf0");
    in.rel_identity = input(GGML_TYPE_F32, 9, 0, "rel_identity");
    in.positions = input(GGML_TYPE_F32, 3, n, "positions");
    in.strain = input(GGML_TYPE_F32, 3, 3, "strain");
    in.atom_inv_cell_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 3, n);
    ggml_set_name(in.atom_inv_cell_t, "atom_inv_cell_t"), ggml_set_input(in.atom_inv_cell_t), g->inputs.push_back(in.atom_inv_cell_t);
    in.inv_volume = input(GGML_TYPE_F32, 1, 1, "inv_volume");
    for (int a = 0; a < 3; a++) in.mesh_size[a] = mesh[a];
    if (mesh_on) {
      const int64_t Mh = int64_t(mesh[0]) * mesh[1] * (mesh[2] / 2 + 1);
      in.mesh_k = input(GGML_TYPE_F32, 3, Mh, "mesh_k");
      in.mesh_k0_pad = input(GGML_TYPE_F32, 1, Mh, "mesh_k0_pad");
      in.mesh_deconv = input(GGML_TYPE_F32, 1, Mh, "mesh_deconv");
    }
    if (les_on && periodic) {
      in.les_k = input(GGML_TYPE_F32, 3, nk, "les_k");
      in.les_n = input(GGML_TYPE_F32, 3, nk, "les_n");
      in.les_taper = input(GGML_TYPE_F32, 1, nk, "les_taper");
    }
    if (les_on && !periodic) in.offdiag = input(GGML_TYPE_F32, n, n, "offdiag");
    ggml_set_param(in.edge_vectors);
    ggml_set_param(in.positions);
    ggml_set_param(in.strain);
    g->out = runtime::build_plato(ctx, spec_, weights_, in, n, E, max_len, periodic);
    ggml_set_output(g->out.total);
    ggml_set_output(g->out.short_range);
    if (g->out.les) ggml_set_output(g->out.les);
    if (g->out.chi) ggml_set_output(g->out.chi);
    g->forward = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    ggml_build_forward_expand(g->forward, g->out.short_range);
    ggml_build_forward_expand(g->forward, g->out.total);
    g->graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, true);
    ggml_build_forward_expand(g->graph, g->out.short_range);
    ggml_build_forward_expand(g->graph, g->out.total);
    if (backward) {
      ggml_set_loss(g->out.total);
      ggml_build_backward_expand(ctx, g->graph, nullptr);
      g->grad_ev = ggml_graph_get_grad(g->graph, in.edge_vectors);
      g->grad_pos = ggml_graph_get_grad(g->graph, in.positions);
      g->grad_eps = ggml_graph_get_grad(g->graph, in.strain);
      for (ggml_tensor *t : {g->grad_ev, g->grad_pos, g->grad_eps}) // views: their sources too
        for (; t; t = t->view_src) ggml_set_output(t);
      g->loss_grad = ggml_graph_get_grad_acc(g->graph, g->out.total);
      if (g->loss_grad) {
        ggml_set_input(g->loss_grad);
        g->inputs.push_back(g->loss_grad);
      }
    }
    // inputs in a buffer of their own, the rest through the allocator
    ggml_backend_buffer_type_t buft = backend_->buffer_type();
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t size = 0;
    std::vector<size_t> offs;
    for (ggml_tensor *t : g->inputs) offs.push_back(size), size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
    g->inputs_buf = ggml_backend_buft_alloc_buffer(buft, std::max<size_t>(size, align));
    char *base = static_cast<char *>(ggml_backend_buffer_get_base(g->inputs_buf));
    for (size_t k = 0; k < g->inputs.size(); k++) ggml_backend_tensor_alloc(g->inputs_buf, g->inputs[k], base + offs[k]);
    if (!galloc_) galloc_ = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc_, g->graph)) throw std::runtime_error("Plato: cannot allocate the graph");
    graph_ = std::move(g);
    G = graph_.get();
  }

  // inputs
  const PlatoInputs &in = G->in;
  auto set = [](ggml_tensor *t, const void *data) { ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t)); };
  {
    std::vector<int32_t> ecs(E), nbs(E), cen(E), rev(n + E), off(n + 1);
    for (int i = 0; i < n; i++) {
      off[i] = static_cast<int32_t>(ed.start[i]);
      rev[i] = i;
      for (int64_t e = ed.start[i]; e < ed.start[i + 1]; e++)
        ecs[e] = species[i], cen[e] = i, nbs[e] = species[ed.j[e]], rev[n + e] = static_cast<int32_t>(n + ed.rev[e]);
    }
    off[n] = static_cast<int32_t>(E);
    set(in.species, species.data()), set(in.edge_center_species, ecs.data()), set(in.neighbor_species, nbs.data());
    set(in.centers, cen.data()), set(in.reverse, rev.data()), set(in.offsets, off.data());
    set(in.edge_vectors, ed.v.data());
    std::vector<float> zeros(ggml_nelements(in.zero_ang), 0.0f);
    set(in.zero_ang, zeros.data());
    std::vector<float> rbf0(spec_.n_rbf);
    const double d0 = 1e-6, w = spec_.cutoff / spec_.n_rbf; // sqrt(clamp(0, 1e-12))
    for (int k = 0; k < spec_.n_rbf; k++) {
      const double mu = spec_.cutoff * double(static_cast<float>(k) / static_cast<float>(spec_.n_rbf - 1));
      rbf0[k] = static_cast<float>(std::exp(-std::pow((d0 - mu) / w, 2)));
    }
    set(in.rbf0, rbf0.data());
    const float I9[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, Z9[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    set(in.rel_identity, I9), set(in.strain, Z9);
    std::vector<float> ict(9 * size_t(n));
    for (int i = 0; i < n; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) ict[9 * i + j * 3 + k] = static_cast<float>(Ci[k][j]);
    set(in.atom_inv_cell_t, ict.data());
    const float iv = V > 0 ? static_cast<float>(1.0 / V) : 0.0f;
    set(in.inv_volume, &iv);
    set(in.positions, system.positions());
    if (!consts_uploaded_) {
      if (in.mesh_k) set(in.mesh_k, K.mesh_k.data()), set(in.mesh_k0_pad, K.mesh_pad.data()), set(in.mesh_deconv, K.mesh_deconv.data());
      if (in.les_k) set(in.les_k, K.les_k.data()), set(in.les_n, K.les_n.data()), set(in.les_taper, K.les_taper.data());
      consts_uploaded_ = true;
    }
    if (in.offdiag) {
      std::vector<float> od(size_t(n) * n, 1.0f);
      for (int i = 0; i < n; i++) od[size_t(i) * n + i] = 0.0f;
      set(in.offdiag, od.data());
    }
    if (G->loss_grad) {
      const float one = 1.0f;
      set(G->loss_grad, &one);
    }
  }
  const auto t2 = clk::now();
  if (backend_->compute(compute_forces ? G->graph : G->forward) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("Plato: graph compute failed");
  const auto t3 = clk::now();
  parts_.nl_ms = ms(t0, t1), parts_.prep_ms = ms(t1, t2), parts_.compute_ms = ms(t2, t3);

  ModelResult r;
  float total = 0, sr = 0, les = 0; // one structure: [1, 1]
  ggml_backend_tensor_get(G->out.total, &total, 0, sizeof(float));
  ggml_backend_tensor_get(G->out.short_range, &sr, 0, sizeof(float));
  if (G->out.les) ggml_backend_tensor_get(G->out.les, &les, 0, sizeof(float));
  float chi = 0;
  if (G->out.chi) ggml_backend_tensor_get(G->out.chi, &chi, 0, sizeof(float));
  parts_.chi = energy_scale_ * chi;
  r.energy = static_cast<float>(energy_scale_ * total + composition);
  parts_.short_range = energy_scale_ * sr, parts_.les = energy_scale_ * les, parts_.composition = composition;
  for (int a = 0; a < 3; a++) parts_.mesh[a] = mesh[a];
  parts_.n_k = nk, parts_.edges = E, parts_.max_len = max_len;
  if (!compute_forces) {
    parts_.read_ms = ms(t3, clk::now());
    return r;
  }
  std::vector<float> gev(3 * E, 0.0f), gpos(3 * size_t(n), 0.0f), geps(9, 0.0f);
  if (G->grad_ev) ggml_backend_tensor_get(G->grad_ev, gev.data(), 0, gev.size() * sizeof(float));
  if (G->grad_pos) ggml_backend_tensor_get(G->grad_pos, gpos.data(), 0, gpos.size() * sizeof(float));
  if (G->grad_eps) ggml_backend_tensor_get(G->grad_eps, geps.data(), 0, geps.size() * sizeof(float));
  std::vector<double> f(3 * size_t(n), 0.0);
  double S[3][3] = {};
  for (int i = 0; i < n; i++)
    for (int c = 0; c < 3; c++) f[3 * i + c] -= gpos[3 * i + c];
  for (int i = 0; i < n; i++)
    for (int64_t e = ed.start[i]; e < ed.start[i + 1]; e++) {
      const int j = ed.j[e];
      for (int c = 0; c < 3; c++) {
        f[3 * i + c] += gev[3 * e + c], f[3 * j + c] -= gev[3 * e + c];
        for (int b = 0; b < 3; b++) S[c][b] += double(ed.v[3 * e + c]) * gev[3 * e + b];
      }
    }
  r.forces.resize(3 * size_t(n));
  for (size_t k = 0; k < f.size(); k++) r.forces[k] = static_cast<float>(energy_scale_ * f[k]);
  r.has_forces = true;
  if (periodic) {
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) S[a][b] = energy_scale_ * (S[a][b] + geps[a * 3 + b]) / V;
    r.stress = {float(S[0][0]), float(S[1][1]), float(S[2][2]), float(S[1][2]), float(S[0][2]), float(S[0][1])};
    r.has_stress = true;
  }
  parts_.read_ms = ms(t3, clk::now());
  return r;
}

} // namespace mlipcpp::plato
