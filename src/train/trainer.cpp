#include "train/trainer.h"
#include "train/lr_schedule.h"

#include "runtime/graph_jvp.h"

#ifdef MLIPCPP_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include <ggml-backend.h>
#include <ggml.h>
#include <ggml-impl.h> // ggml_graph_view, for profiling
#include <gguf.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <tuple>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <filesystem>
#include <functional>
#include <future>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace mlipcpp::train {

namespace {

constexpr size_t GRAPH_SIZE = 65536;

using clock_type = std::chrono::steady_clock;
double seconds_since(clock_type::time_point t0) {
  return std::chrono::duration<double>(clock_type::now() - t0).count();
}

ggml_tensor *input(ggml_context *ctx, ggml_type type, std::initializer_list<int64_t> ne,
                   const char *name) {
  ggml_tensor *t = nullptr;
  auto d = std::data(ne);
  switch (ne.size()) {
  case 1: t = ggml_new_tensor_1d(ctx, type, d[0]); break;
  case 2: t = ggml_new_tensor_2d(ctx, type, d[0], d[1]); break;
  case 3: t = ggml_new_tensor_3d(ctx, type, d[0], d[1], d[2]); break;
  default: throw std::logic_error("input rank");
  }
  ggml_set_name(t, name);
  ggml_set_input(t);
  return t;
}

/// metatrain's schedule: linear warmup from 0, then cosine to 0; step k
/// (0-based) runs at lr * factor(k).
double lr_factor(int64_t step, int64_t total, int64_t warmup) {
  if (step < warmup) return double(step) / double(std::max<int64_t>(1, warmup));
  const double progress = double(step - warmup) / double(std::max<int64_t>(1, total - warmup));
  return 0.5 * (1.0 + std::cos(std::numbers::pi * progress));
}

/// LOREM's schedule: 1 for the first `constant` steps, then linear to
/// `min_factor` at the end.
double lr_factor_linear(int64_t step, int64_t total, int64_t constant, double min_factor) {
  if (step < constant) return 1.0;
  const double progress = double(step - constant) / double(std::max<int64_t>(1, total - constant));
  return 1.0 + (min_factor - 1.0) * std::min(1.0, progress);
}


void write_gguf(const std::string &src, const std::string &dst, Worker &w,
                const std::vector<float> &flat, const TargetScaling &sc) {
  // Copy the source GGUF (metadata and every tensor) into a fresh context,
  // overwriting the trainable tensors' bytes in the loaded data context.
  // (gguf_set_tensor_data on the loaded context keeps the tensors' backend
  // buffer, so the writer would read the old buffer through a host pointer.)
  ggml_context *data_ctx = nullptr;
  gguf_context *in = gguf_init_from_file(src.c_str(), {/*no_alloc=*/false, &data_ctx});
  if (!in) throw std::runtime_error("cannot read " + src);
  size_t off = 0;
  for (ggml_tensor *p : w.params()) {
    ggml_tensor *t = ggml_get_tensor(data_ctx, ggml_get_name(p));
    if (!t || ggml_nbytes(t) != ggml_nbytes(p))
      throw std::runtime_error(std::string("write_gguf: tensor mismatch for ") + ggml_get_name(p));
    if (t->buffer) ggml_backend_tensor_set(t, flat.data() + off, 0, ggml_nbytes(t));
    else std::memcpy(t->data, flat.data() + off, ggml_nbytes(t));
    off += ggml_nelements(p);
  }
  gguf_context *g = gguf_init_empty();
  gguf_set_kv(g, in);
  for (int64_t i = 0; i < gguf_get_n_tensors(in); i++)
    gguf_add_tensor(g, ggml_get_tensor(data_ctx, gguf_get_tensor_name(in, i)));
  gguf_set_val_f32(g, "pet.energy_scale", static_cast<float>(sc.energy_scale));
  std::vector<int32_t> ck, fk;
  std::vector<float> cv, fv;
  for (auto [Z, c] : sc.composition) ck.push_back(Z), cv.push_back(static_cast<float>(c));
  for (auto [Z, s] : sc.force_scale) fk.push_back(Z), fv.push_back(static_cast<float>(s));
  gguf_set_arr_data(g, "pet.composition_keys", GGUF_TYPE_INT32, ck.data(), ck.size());
  gguf_set_arr_data(g, "pet.composition_values", GGUF_TYPE_FLOAT32, cv.data(), cv.size());
  gguf_set_arr_data(g, "pet.nc_force_scale_keys", GGUF_TYPE_INT32, fk.data(), fk.size());
  gguf_set_arr_data(g, "pet.nc_force_scale_values", GGUF_TYPE_FLOAT32, fv.data(), fv.size());
  const std::string tmp = dst + ".tmp";
  const bool ok = gguf_write_to_file(g, tmp.c_str(), false);
  gguf_free(g);
  gguf_free(in);
  ggml_free(data_ctx);
  if (!ok) throw std::runtime_error("cannot write " + dst);
  std::filesystem::rename(tmp, dst);
}

/// <output>.state of a chain of jobs: epochs done, the global step, the best
/// validation score and the optimizer state (Optimizer::get_state).
struct TrainState {
  int64_t epoch = 0, step = 0;
  double best = -1e300;
  std::vector<uint8_t> optimizer;
};
constexpr char state_magic[8] = {'M', 'L', 'P', 'S', 'T', 'A', 'T', '1'};

void write_state(const std::string &path, const TrainState &s) {
  const std::string tmp = path + ".tmp";
  std::unique_ptr<std::FILE, int (*)(std::FILE *)> f(std::fopen(tmp.c_str(), "wb"), std::fclose);
  const uint64_t n = s.optimizer.size();
  bool ok = f && std::fwrite(state_magic, 1, 8, f.get()) == 8 && std::fwrite(&s.epoch, 8, 1, f.get()) == 1 &&
            std::fwrite(&s.step, 8, 1, f.get()) == 1 && std::fwrite(&s.best, 8, 1, f.get()) == 1 &&
            std::fwrite(&n, 8, 1, f.get()) == 1 && std::fwrite(s.optimizer.data(), 1, n, f.get()) == n;
  ok = f && std::fclose(f.release()) == 0 && ok;
  if (!ok) throw std::runtime_error("cannot write " + path);
  std::filesystem::rename(tmp, path);
}

TrainState read_state(const std::string &path) {
  std::unique_ptr<std::FILE, int (*)(std::FILE *)> f(std::fopen(path.c_str(), "rb"), std::fclose);
  char magic[8];
  TrainState s;
  uint64_t n = 0;
  if (!f || std::fread(magic, 1, 8, f.get()) != 8 || std::memcmp(magic, state_magic, 8) != 0 ||
      std::fread(&s.epoch, 8, 1, f.get()) != 1 || std::fread(&s.step, 8, 1, f.get()) != 1 ||
      std::fread(&s.best, 8, 1, f.get()) != 1 || std::fread(&n, 8, 1, f.get()) != 1)
    throw std::runtime_error("cannot read the trainer state " + path);
  s.optimizer.resize(n);
  if (std::fread(s.optimizer.data(), 1, n, f.get()) != n) throw std::runtime_error(path + ": truncated");
  return s;
}

/// Online least squares for CostModel: decayed normal equations over the
/// measured micro-batch times, non-negative coefficients (features with a
/// negative fit are dropped, most negative first). The fixed cost is
/// floored at the measured per-micro-batch graph build + allocation, which
/// does not depend on its size: steps of a few large micro-batches leave the
/// intercept unresolved, a fit of ~0 ms let the plan split at every distinct
/// neighbour count (176059 / 176431: 280-560 micro-batches per worker at
/// batch 2048), and the fit then needed ~20 steps to recover. A weak ridge
/// pulls the other unresolved directions to the prior.
class CostFit {
public:
  explicit CostFit(const CostModel &prior) : prior_(prior) {}
  /// ms: the micro-batch's wall time; fixed_ms: its size-independent part
  void add(double n, double m, double ms, double fixed_ms) {
    const auto f = scaled(n, m);
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) A_[a][b] = decay_ * A_[a][b] + f[a] * f[b];
      y_[a] = decay_ * y_[a] + f[a] * ms;
    }
    fixed_sum_ = decay_ * fixed_sum_ + fixed_ms, fixed_w_ = decay_ * fixed_w_ + 1;
    n_++;
  }
  /// Fitted model, or false while too few samples / no admissible fit.
  bool fit(CostModel &out) const {
    if (n_ < 30) return false;
    const double floor = fixed_sum_ / fixed_w_;
    std::array<bool, 3> on = {true, true, true};
    std::array<double, 3> held = {}; // values of the features taken out of the solve
    for (int pass = 0; pass < 4; pass++) {
      std::array<double, 3> c{};
      if (!solve(on, held, c)) return false;
      if (on[0] && c[0] * scale_[0] < floor) { // the intercept first, at its floor
        on[0] = false, held[0] = floor / scale_[0];
        continue;
      }
      int worst = -1;
      for (int k = 1; k < 3; k++)
        if (on[k] && c[k] < 0 && (worst < 0 || c[k] < c[worst])) worst = k;
      if (worst < 0) {
        for (int k = 0; k < 3; k++) out.c[k] = (on[k] ? c[k] : held[k]) * scale_[k];
        return true;
      }
      on[worst] = false;
    }
    return false;
  }

private:
  static constexpr std::array<double, 3> scale_ = {1.0, 1e-6, 1e-4}; // features to O(1)
  std::array<double, 3> scaled(double n, double m) const {
    auto f = CostModel::features(n, m);
    for (int k = 0; k < 3; k++) f[k] *= scale_[k];
    return f;
  }
  bool solve(const std::array<bool, 3> &on, const std::array<double, 3> &held, std::array<double, 3> &c) const {
    std::vector<int> idx;
    for (int k = 0; k < 3; k++)
      if (on[k]) idx.push_back(k);
    const int d = static_cast<int>(idx.size());
    c = {};
    if (d == 0) return true;
    double M[3][4];
    for (int a = 0; a < d; a++) {
      for (int b = 0; b < d; b++) M[a][b] = A_[idx[a]][idx[b]] + (a == b ? ridge_ : 0.0);
      M[a][d] = y_[idx[a]] + ridge_ * prior_.c[idx[a]] / scale_[idx[a]];
      for (int k = 0; k < 3; k++)
        if (!on[k]) M[a][d] -= A_[idx[a]][k] * held[k];
    }
    for (int col = 0; col < d; col++) { // Gauss-Jordan, partial pivoting
      int piv = col;
      for (int r = col + 1; r < d; r++)
        if (std::abs(M[r][col]) > std::abs(M[piv][col])) piv = r;
      if (std::abs(M[piv][col]) < 1e-12) return false;
      std::swap(M[col], M[piv]);
      for (int r = 0; r < d; r++) {
        if (r == col) continue;
        const double f = M[r][col] / M[col][col];
        for (int k = col; k <= d; k++) M[r][k] -= f * M[col][k];
      }
    }
    for (int a = 0; a < d; a++) c[idx[a]] = M[a][d] / M[a][a];
    return std::isfinite(c[0]) && std::isfinite(c[1]) && std::isfinite(c[2]);
  }
  static constexpr double decay_ = 0.995; // ~200 micro-batches of memory
  static constexpr double ridge_ = 1.0;    // ~ one micro-batch with O(1) scaled features
  CostModel prior_;
  double A_[3][3] = {}, y_[3] = {};
  double fixed_sum_ = 0, fixed_w_ = 0;
  int64_t n_ = 0;
};

/// Run `f(w)` on every worker in its own thread.
template <typename F> void for_workers(std::vector<std::unique_ptr<Worker>> &ws, F f) {
  std::vector<std::thread> ts;
  for (size_t w = 0; w < ws.size(); w++) ts.emplace_back([&, w] { f(w); });
  for (auto &t : ts) t.join();
}

} // namespace

namespace {
ggml_tensor *find_weight(runtime::GraphModel &m, const std::string &name) {
  for (auto &[n, t] : m.weights())
    if (n == name) return t;
  throw std::runtime_error("trainable tensor " + name + " not in the GGUF");
}
} // namespace

AdaptiveCutoff adaptive_cutoff(const runtime::GraphModel &model) {
  const std::string &method = model.adaptive_cutoff_method();
  AdaptiveCutoff a{model.num_neighbors_adaptive(), model.cutoff_width_adaptive(), method == "grid"};
  if (a.enabled() && method != "solver" && method != "grid")
    throw std::runtime_error("adaptive cutoff: unknown method \"" + method + "\"");
  return a;
}

std::string cache_path(const std::string &cache_dir, const MemmapSplit &split, float cutoff,
                       const AdaptiveCutoff &adaptive) {
  if (cache_dir.empty()) return {};
  // keyed by content, not path: the data is staged to a per-job $TMPDIR
  uint64_t h = 1469598103934665603ULL; // FNV-1a
  auto mix = [&](const void *p, size_t n) {
    for (size_t k = 0; k < n; k++) h = (h ^ static_cast<const uint8_t *>(p)[k]) * 1099511628211ULL;
  };
  const int64_t ns = split.size(), N = split.total_atoms();
  mix(&ns, sizeof ns), mix(&N, sizeof N), mix(&cutoff, sizeof cutoff);
  if (adaptive.enabled()) mix(&adaptive.n_target, sizeof(double)), mix(&adaptive.width, sizeof(double));
  if (adaptive.enabled() && adaptive.grid) mix("grid", 4);
  mix(split.numbers(0), N * sizeof(int32_t));
  mix(split.positions(0), N * 3 * sizeof(float));
  mix(split.cell(0), ns * 9 * sizeof(float));
  return fmt::format("{}/nl-{:016x}.bin", cache_dir, h);
}

StepStats &StepStats::operator+=(const StepStats &o) {
  loss += o.loss, loss_energy += o.loss_energy, loss_force += o.loss_force;
  loss_nc += o.loss_nc, se_force_nc += o.se_force_nc;
  loss_charge += o.loss_charge, sum_charge_r += o.sum_charge_r;
  n_charge_r += o.n_charge_r, n_charge_skipped += o.n_charge_skipped;
  se_energy_per_atom += o.se_energy_per_atom, se_force += o.se_force;
  ae_energy_per_atom += o.ae_energy_per_atom, ae_force += o.ae_force;
  sum_e += o.sum_e, sum2_e += o.sum2_e, sum_f += o.sum_f, sum2_f += o.sum2_f;
  n_systems += o.n_systems, n_force_components += o.n_force_components;
  return *this;
}

namespace {
/// Muon applies to hidden matrices: 2-D, both dims >= 16, not an embedding
/// table (one-hot inputs: Adam, as in Muon's reference recipe).
bool is_muon_param(const ggml_tensor *p) {
  return ggml_n_dims(p) == 2 && std::min(p->ne[0], p->ne[1]) >= 16 &&
         std::string(ggml_get_name(p)).find("embed") == std::string::npos;
}
} // namespace

Worker::Worker(const std::string &gguf, int device, size_t cache_graphs) : cache_limit_(cache_graphs) {
  model_.set_backend_preference(device < 0 ? BackendPreference::CPU : BackendPreference::Auto);
  model_.set_device(std::max(device, 0));
  model_.load_from_gguf(gguf);
  if (!model_.forces_mode())
    throw std::runtime_error(gguf + ": training needs a forces-mode export");
  // energy-only exports (no --nc-forces) train with conservative forces only (Worker::run)
  if (model_.param_names().empty())
    throw std::runtime_error(gguf + ": no train.param_names; re-export with export_pet_gguf.py");
  for (const auto &name : model_.param_names()) {
    ggml_tensor *t = find_weight(model_, name);
    ggml_set_param(t);
    params_.push_back(t);
    n_params_ += ggml_nelements(t);
  }
  galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_.backend()));

  acc_ctx_ = ggml_init({ggml_tensor_overhead() * (params_.size() + 2), nullptr, true});
  acc_flat_ = ggml_new_tensor_1d(acc_ctx_, GGML_TYPE_F32, n_params_);
  ggml_set_name(acc_flat_, "grad_flat");
  // gradient layout: the Muon matrices first, grouped by shape (a group's
  // gradients are then one contiguous [ne0, ne1, n] block), then the rest
  std::vector<size_t> order(params_.size());
  for (size_t k = 0; k < order.size(); k++) order[k] = k;
  auto key = [&](size_t k) {
    const ggml_tensor *p = params_[k];
    return is_muon_param(p) ? std::make_tuple(0, p->ne[0], p->ne[1]) : std::make_tuple(1, int64_t(0), int64_t(0));
  };
  std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) { return key(i) < key(j); });
  grad_off_.resize(params_.size());
  size_t off = 0;
  for (size_t k : order) {
    ggml_tensor *p = params_[k];
    const size_t nb1 = p->ne[0] * sizeof(float), nb2 = nb1 * p->ne[1], nb3 = nb2 * p->ne[2];
    acc_[p] = ggml_view_4d(acc_ctx_, acc_flat_, p->ne[0], p->ne[1], p->ne[2], p->ne[3], nb1, nb2,
                           nb3, off * sizeof(float));
    grad_off_[k] = off;
    off += ggml_nelements(p);
  }
  acc_buf_ = ggml_backend_alloc_ctx_tensors(acc_ctx_, model_.backend());
  if (!acc_buf_) throw std::runtime_error("cannot allocate the gradient accumulator");
  zero_grads();
  // a byte view of the whole weight buffer, created after the accumulator
  // allocation so that it is placed on the weights, not allocated
  ggml_backend_buffer_t wbuf = model_.weight_buffer();
  weights_blob_ = ggml_new_tensor_1d(acc_ctx_, GGML_TYPE_I8,
                                     static_cast<int64_t>(ggml_backend_buffer_get_size(wbuf)));
  ggml_backend_tensor_alloc(wbuf, weights_blob_, ggml_backend_buffer_get_base(wbuf));
}

Worker::~Worker() {
  cache_.clear();
  if (galloc_) ggml_gallocr_free(galloc_);
  if (acc_buf_) ggml_backend_buffer_free(acc_buf_);
  if (acc_ctx_) ggml_free(acc_ctx_);
}

void Worker::zero_grads() { ggml_backend_buffer_clear(acc_buf_, 0); }

std::vector<float> Worker::get_grads() const {
  std::vector<float> flat(n_params_), g(n_params_);
  ggml_backend_tensor_get(acc_flat_, flat.data(), 0, ggml_nbytes(acc_flat_));
  size_t off = 0; // grouped layout -> parameter order
  for (size_t k = 0; k < params_.size(); k++) {
    const size_t n = ggml_nelements(params_[k]);
    std::copy_n(flat.begin() + grad_off_[k], n, g.begin() + off);
    off += n;
  }
  return g;
}

void Worker::set_params(const std::vector<float> &flat) {
  size_t off = 0;
  for (ggml_tensor *p : params_) {
    ggml_backend_tensor_set(p, flat.data() + off, 0, ggml_nbytes(p));
    off += ggml_nelements(p);
  }
}

std::vector<float> Worker::get_params() const {
  std::vector<float> flat(n_params_);
  size_t off = 0;
  for (ggml_tensor *p : params_) {
    ggml_backend_tensor_get(p, flat.data() + off, 0, ggml_nbytes(p));
    off += ggml_nelements(p);
  }
  return flat;
}

Worker::StepGraph::~StepGraph() {
  if (ctx) ggml_free(ctx);
  if (persist) ggml_backend_buffer_free(persist);
}

size_t Worker::compute_buffer_bytes() const { return ggml_gallocr_get_buffer_size(galloc_, 0); }

namespace {
void fill_constants(const std::vector<std::pair<ggml_tensor *, float>> &cs) {
  std::vector<float> staging;
  for (auto &[t, v] : cs) {
    if (!t->buffer) continue;
    staging.assign(ggml_nelements(t), v);
    ggml_backend_tensor_set(t, staging.data(), 0, ggml_nbytes(t));
  }
}

/// The targets' sums for R^2: physical energies per atom after the
/// composition (eV/atom) and force components (eV/A) of the real structures.
void add_target_sums(const Batch &b, double energy_scale, StepStats &st) {
  for (int k = 0; k < b.n_real_systems; k++) {
    const double y = double(b.energy_target[k]) * energy_scale;
    st.sum_e += y, st.sum2_e += y * y;
  }
  for (int a = 0; a < b.n_atoms; a++) {
    if (b.force_mask[a] == 0.0f) continue;
    for (int c = 0; c < 3; c++) {
      const double y = double(b.force_target[3 * a + c]) * b.force_scale[a];
      st.sum_f += y, st.sum2_f += y * y;
    }
  }
}

/// Loss normalisations of a micro-batch: (1/S, 1/3N) of the whole step batch.
std::pair<float, float> loss_norms(const Batch &b) {
  const float e_norm = b.energy_norm > 0 ? b.energy_norm : 1.0f / b.n_real_systems;
  float f_norm = b.force_norm;
  if (!(f_norm > 0)) {
    double n_real = 0;
    for (float w : b.force_mask) n_real += w;
    f_norm = static_cast<float>(1.0 / (3.0 * n_real));
  }
  return {e_norm, f_norm};
}
} // namespace

// (FLOPs, ideal bytes) of one node; views cost nothing
static std::pair<double, double> node_cost(const ggml_tensor *t) {
  switch (t->op) {
  case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
    return {0, 0};
  case GGML_OP_CPY: return {0, 2.0 * ggml_nbytes(t->src[0])};
  default: break;
  }
  double bytes = ggml_nbytes(t);
  for (const ggml_tensor *s : t->src)
    if (s) bytes += ggml_nbytes(s);
  const double e = ggml_nelements(t);
  switch (t->op) {
  case GGML_OP_MUL_MAT: return {2.0 * t->src[0]->ne[0] * e, bytes};
  case GGML_OP_OUT_PROD: return {2.0 * t->src[0]->ne[1] * e, bytes};
  case GGML_OP_CONT: case GGML_OP_DUP: case GGML_OP_GET_ROWS: case GGML_OP_CONCAT: return {0, bytes};
  case GGML_OP_SOFT_MAX: case GGML_OP_SOFT_MAX_BACK: return {5.0 * e, bytes};
  case GGML_OP_PET_ATTN: case GGML_OP_PET_ATTN_BACK: { // per (sequence, head, query, key), executed
    const ggml_tensor *x = t->src[0];
    const double H = t->op_params[0], d = x->ne[0] / H, L = x->ne[2]; // op_params[0]: n_head
    return {(t->op == GGML_OP_PET_ATTN ? 4 * d + 5 : 10 * d + 10) * x->ne[3] * H * L * L, bytes};
  }
  default: return {e, bytes};
  }
}

Worker::GraphCost Worker::graph_cost(int n, int m, int S, bool backward) {
  auto g = build_graph(n, m, S, backward ? BACKWARD : FORWARD, false);
  GraphCost c;
  for (int i = 0; i < ggml_graph_n_nodes(g->gf); i++) {
    auto [f, b] = node_cost(ggml_graph_node(g->gf, i));
    c.flops += f, c.bytes += b;
  }
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_.backend()));
  ggml_gallocr_reserve_n_size(ga, g->gf, nullptr, nullptr, &c.buffer);
  ggml_gallocr_free(ga);
  return c;
}

ggml_tensor *Worker::charge_corr_graph(StepGraph &g, ggml_context *ctx, int S, bool loss) {
  (void)S;
  if (!(charge_corr.weight > 0)) return nullptr;
  ggml_tensor *q = g.plato ? g.plato->charges() : nullptr;
  if (!q) throw std::runtime_error("--charge-corr-weight needs a Plato-PET-LR model with LES");
  const int64_t n = q->ne[1], G = g.plato->n_struct();
  g.qref = input(ctx, GGML_TYPE_F32, {1, n}, "ref_charge");
  g.qok = input(ctx, GGML_TYPE_F32, {1, G}, "ref_ok");
  g.qw = input(ctx, GGML_TYPE_F32, {1}, "charge_corr_weight");
  // the first latent charge channel (Pearson is shift and scale invariant: the charge unit
  // and the total-charge shift do not matter)
  ggml_tensor *q0 = ggml_cont(ctx, ggml_view_2d(ctx, q, 1, n, q->nb[1], 0));
  g.charge_r = g.plato->pearson(ctx, q0, g.qref); // [1, G]
  ggml_set_output(g.charge_r);
  if (!loss) return nullptr;
  ggml_tensor *hinge = ggml_mul(ctx, ggml_relu(ctx, ggml_scale_bias(ctx, g.charge_r, -1.0f, static_cast<float>(charge_corr.r_min))), g.qok);
  return g.charge_loss = ggml_mul(ctx, ggml_sum(ctx, hinge), g.qw);
}

void Worker::charge_corr_stats(StepGraph &g, const Batch &b, StepStats &st) {
  st.n_charge_skipped += b.n_ref_skipped;
  if (!g.charge_r) return;
  std::vector<float> r(ggml_nelements(g.charge_r));
  ggml_backend_tensor_get(g.charge_r, r.data(), 0, r.size() * sizeof(float));
  const double norm = b.charge_norm > 0 ? b.charge_norm : 1.0 / std::max<double>(1, std::accumulate(b.ref_ok.begin(), b.ref_ok.end(), 0.0));
  for (size_t k = 0; k < r.size() && k < b.ref_ok.size(); k++)
    if (b.ref_ok[k] != 0.0f) {
      st.sum_charge_r += r[k], st.n_charge_r++;
      st.loss_charge += charge_corr.weight * norm * std::max(0.0, charge_corr.r_min - r[k]);
    }
}

void Worker::plato_create(StepGraph &g, ggml_context *ctx, int n, bool grad) {
  std::vector<runtime::PetPlatoLRStructure> st;
  if (plato_batch_ && plato_batch_->n_atoms == n) {
    const Batch &b = *plato_batch_;
    int a0 = 0;
    for (int k = 0; k < b.n_real_systems; k++) {
      runtime::PetPlatoLRStructure s;
      s.a0 = a0, s.a1 = a0 + b.n_per_system[k], a0 = s.a1;
      s.periodic = b.periodic[k] != 0;
      for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) s.cell[r][c] = b.cells[k][3 * r + c];
      s.charge = b.system_charge[k];
      st.push_back(s);
    }
    // padding atoms (micro-batch / tile padding) belong to no structure: masked out of the long-range ops
    if (a0 > n) throw std::logic_error("Plato-PET-LR training: batch atom count");
  } else { // a shape without data (graph_cost): one open structure
    runtime::PetPlatoLRStructure s;
    s.a1 = n;
    st.push_back(s);
  }
  g.plato = std::make_unique<runtime::PetPlatoLR>();
  g.plato->create(ctx, model_.plato_lr(), st, n, grad);
}

std::unique_ptr<Worker::StepGraph> Worker::build_graph(int n, int m, int S, Mode mode, bool cached) {
  const bool backward = mode != FORWARD;
  const bool shared = mode == MIXED && shared_forward_; // + the FORCES graph on the same forward
  // the tangent graph and its reverse pass: several times the loss graph
  const size_t graph_size = mode == MIXED ? 8 * GRAPH_SIZE : GRAPH_SIZE;
  auto g = std::make_unique<StepGraph>();
  g->mode = mode;
  auto &interp = model_.interpreter();
  interp.set_dimension("n_atoms", n);
  interp.set_dimension("max_neighbors", m);
  interp.set_dimension("n_edges", int64_t(n) * m);
  interp.set_dimension("max_neighbors_plus_one", m + 1);

  // every shape has the same topology: the first build measures the context
  // (cached graphs keep theirs, so it is kept tight)
  size_t &ctx_bytes = ctx_bytes_[mode];
  const bool measure = ctx_bytes == 0;
  g->ctx = ggml_init({measure ? ggml_tensor_overhead() * graph_size * 2 +
                                    ggml_graph_overhead_custom(graph_size, true) * (shared ? 4 : 1)
                              : ctx_bytes,
                      nullptr, /*no_alloc=*/true});
  ggml_context *ctx = g->ctx;
  auto in = [&](ggml_type type, std::initializer_list<int64_t> ne, const char *name) {
    ggml_tensor *t = input(ctx, type, ne, name);
    g->inputs.push_back(t);
    return t;
  };
  // ggml shapes are the reversed PyTorch shapes; order = upload_inputs
  interp.set_input("species", in(GGML_TYPE_I32, {n}, "species"));
  interp.set_input("neighbor_species", in(GGML_TYPE_I32, {m, n}, "neighbor_species"));
  ggml_tensor *edges = in(GGML_TYPE_F32, {3, m, n}, "edge_vectors");
  interp.set_input("edge_vectors", edges);
  ggml_tensor *padding_mask = in(GGML_TYPE_F32, {m, n}, "padding_mask");
  interp.set_input("padding_mask", padding_mask);
  interp.set_input("reverse_neighbor_index",
                   in(GGML_TYPE_I32, {int64_t(n) * m}, "reverse_neighbor_index"));
  ggml_tensor *cv = in(GGML_TYPE_F32, {m, n}, "cutoff_values");
  interp.set_input("cutoff_values", cv);
  if (model_.conditioning().enabled) { // table rows of each atom's structure
    interp.set_input("charge_index", in(GGML_TYPE_I32, {n}, "charge_index"));
    interp.set_input("spin_index", in(GGML_TYPE_I32, {n}, "spin_index"));
  }
  if (model_.plato_lr().enabled) {
    plato_create(*g, ctx, n, mode == FORCES || shared);
    g->plato->set_padding_mask(padding_mask);
    interp.custom_op = [p = g->plato.get()](ggml_context *c, const runtime::GIRNode &node,
                                            const std::vector<ggml_tensor *> &in) { return p->build(c, node, in); };
  }
  struct ClearHook {
    runtime::GraphInterpreter &i;
    ~ClearHook() { i.custom_op = {}; }
  } clear_hook{interp};
  if (mode == FORCES) {
    // dE/d(edge vectors, cutoff values) of E = sum of the atomic energies;
    // the weights are not differentiated (their flags restored after)
    ggml_set_param(edges), ggml_set_param(cv);
    for (ggml_tensor *p : params_) p->flags &= ~GGML_TENSOR_FLAG_PARAM;
    g->out = interp.build(ctx);
    ggml_set_output(g->out); // (an energy-only export's [n]: the source of the view below, kept too)
    if (ggml_n_dims(g->out) == 1) g->out = ggml_reshape_2d(ctx, g->out, 1, n), ggml_set_output(g->out);
    charge_corr_graph(*g, ctx, S, false); // r for the metrics (the loss term is in MIXED)
    ggml_tensor *E = ggml_sum(ctx, ggml_cont(ctx, ggml_view_2d(ctx, g->out, 1, n, g->out->nb[1], 0)));
    g->gf = ggml_new_graph_custom(ctx, graph_size, true);
    ggml_build_forward_expand(g->gf, g->out);
    ggml_build_forward_expand(g->gf, E);
    if (g->charge_r) ggml_build_forward_expand(g->gf, g->charge_r);
    g->n_forward = ggml_graph_n_nodes(g->gf);
    ggml_set_loss(E);
    ggml_build_backward_expand(ctx, g->gf, nullptr);
    for (ggml_tensor *p : params_) ggml_set_param(p);
    g->grad_v = ggml_graph_get_grad(g->gf, edges);
    g->grad_c = ggml_graph_get_grad(g->gf, cv);
    if (g->plato) {
      g->grad_x = ggml_graph_get_grad(g->gf, g->plato->positions);
      if (g->grad_x) ggml_set_output(g->grad_x);
    }
    if (!g->grad_v) throw std::runtime_error("conservative forces: E does not depend on the edge vectors");
    ggml_set_output(g->grad_v);
    if (g->grad_c) ggml_set_output(g->grad_c);
    g->loss_grad = ggml_graph_get_grad_acc(g->gf, E);
    ggml_set_input(g->loss_grad);
  } else {
    ggml_tensor *atom_weight = in(GGML_TYPE_F32, {n, S}, "atom_weight");
    ggml_tensor *energy_target = in(GGML_TYPE_F32, {S, 1}, "energy_target");
    ggml_tensor *force_target = in(GGML_TYPE_F32, {3, n}, "force_target");
    ggml_tensor *force_mask = in(GGML_TYPE_F32, {1, n}, "force_mask");
    ggml_tensor *force_scale = in(GGML_TYPE_F32, {1, n}, "force_scale");
    // per-step loss normalisation (1/S, 1/3N of the whole step): inputs, so a
    // cached graph serves every step
    ggml_tensor *norms = in(GGML_TYPE_F32, {2}, "loss_norms");
    ggml_tensor *e_norm = ggml_view_1d(ctx, norms, 1, 0);
    ggml_tensor *f_norm = ggml_view_1d(ctx, norms, 1, sizeof(float));

    // MIXED: tangents of the edge vectors and cutoff values (the force
    // residual, Worker::run_conservative)
    ggml_tensor *dv = mode == MIXED ? in(GGML_TYPE_F32, {3, m, n}, "edge_tangent") : nullptr;
    ggml_tensor *dc = mode == MIXED ? in(GGML_TYPE_F32, {m, n}, "cutoff_tangent") : nullptr;
    if (mode == MIXED && g->plato) { // not in g->inputs: uploaded with the Plato inputs
      g->dx = input(ctx, GGML_TYPE_F32, {3, n}, "position_tangent");
      g->jvp_scale = input(ctx, GGML_TYPE_F32, {1}, "jvp_scale");
    }

    // out: [n_atoms, 4] = (energy | force) per atom; ggml ne = (4, n)
    ggml_tensor *out = interp.build(ctx);
    if (ggml_n_dims(out) == 1) out = ggml_reshape_2d(ctx, out, 1, n); // energy-only export: [n]
    ggml_tensor *energy = ggml_reshape_2d(
        ctx, ggml_cont(ctx, ggml_view_2d(ctx, out, 1, n, out->nb[1], 0)), n, 1);
    // energy-only exports (conservative training): no direct force head, its loss terms are 0
    ggml_tensor *force = out->ne[0] >= 4 ? ggml_cont(ctx, ggml_view_2d(ctx, out, 3, n, out->nb[1], sizeof(float)))
                                         : ggml_mul(ctx, force_target, force_mask);
    // metatrain's average_by_num_atoms: per-structure Σ_i e_i / N
    ggml_tensor *d_energy = ggml_sub(ctx, ggml_mul_mat(ctx, atom_weight, energy), energy_target);
    ggml_tensor *d_force = ggml_mul(ctx, ggml_sub(ctx, force, force_target), force_mask);
    // squared error, or Huber 0.5 c^2 + delta (|d| - |c|), c = clamp(d, -delta, delta)
    auto component = [&](ggml_tensor *d, double delta) {
      if (!loss.huber) return ggml_sum(ctx, ggml_sqr(ctx, d));
      const float dl = static_cast<float>(delta);
      ggml_tensor *c = ggml_clamp(ctx, d, -dl, dl);
      return ggml_sum(ctx, ggml_add(ctx, ggml_scale(ctx, ggml_sqr(ctx, c), 0.5f),
                                    ggml_scale(ctx, ggml_sub(ctx, ggml_abs(ctx, d), ggml_abs(ctx, c)), dl)));
    };
    ggml_tensor *loss_e = ggml_mul(ctx, component(d_energy, loss.delta_e), e_norm);
    ggml_tensor *loss_f = ggml_mul(ctx, component(d_force, loss.delta_f), f_norm);
    ggml_tensor *loss = ggml_add(ctx, loss_e, loss_f);
    if (ggml_tensor *lq = charge_corr_graph(*g, ctx, S, true)) loss = ggml_add(ctx, loss, lq);
    if (mode == MIXED) {
      // - <dE/dx, u> along the scaled residual u: its weight gradient is the
      // conservative force loss's (u held fixed)
      g->jvp_mark = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); // tensors are laid out in creation order
      runtime::GraphJVP jvp(ctx);
      jvp.seed(edges, dv);
      jvp.seed(cv, dc);
      if (g->dx) jvp.seed(g->plato->positions, g->dx);
      ggml_tensor *J = jvp(ggml_sum(ctx, energy));
      if (!J) throw std::runtime_error("conservative forces: E has no tangent");
      if (g->jvp_scale) J = ggml_mul(ctx, J, g->jvp_scale); // the tangents were normalised
      loss = ggml_sub(ctx, loss, J);
    }
    if (shared) {
      // stage 1, FORCES on these forward tensors: dE/d(edge vectors, cutoff values,
      // positions) of E = sum of the atomic energies, the weights not differentiated
      g->out = out;
      ggml_set_output(out);
      if (out->view_src) ggml_set_output(out->view_src); // (an energy-only export's [n], reshaped)
      ggml_tensor *E = ggml_sum(ctx, energy);
      // (flags before the expansion: ggml makes params nodes, the others leafs)
      for (ggml_tensor *p : params_) p->flags &= ~GGML_TENSOR_FLAG_PARAM;
      ggml_set_param(edges), ggml_set_param(cv);
      g->gforces = ggml_new_graph_custom(ctx, graph_size, true);
      ggml_build_forward_expand(g->gforces, out);
      ggml_build_forward_expand(g->gforces, E);
      if (g->charge_r) ggml_build_forward_expand(g->gforces, g->charge_r);
      g->n_forces_forward = ggml_graph_n_nodes(g->gforces);
      ggml_set_loss(E);
      ggml_build_backward_expand(ctx, g->gforces, nullptr);
      g->grad_v = ggml_graph_get_grad(g->gforces, edges);
      g->grad_c = ggml_graph_get_grad(g->gforces, cv);
      if (g->plato) g->grad_x = ggml_graph_get_grad(g->gforces, g->plato->positions);
      if (!g->grad_v) throw std::runtime_error("conservative forces: E does not depend on the edge vectors");
      for (ggml_tensor *t : {g->grad_v, g->grad_c, g->grad_x})
        if (t) ggml_set_output(t);
      ggml_tensor *seed = ggml_graph_get_grad_acc(g->gforces, E);
      ggml_set_input(seed);
      g->constants_extra.emplace_back(seed, 1.0f);
      // stage 2 differentiates the weights only (the tangents are its inputs)
      E->flags &= ~GGML_TENSOR_FLAG_LOSS;
      edges->flags &= ~GGML_TENSOR_FLAG_PARAM, cv->flags &= ~GGML_TENSOR_FLAG_PARAM;
      if (g->plato) {
        g->plato->positions->flags &= ~GGML_TENSOR_FLAG_PARAM;
        for (ggml_tensor *t : g->plato->strains()) t->flags &= ~GGML_TENSOR_FLAG_PARAM;
      }
      for (ggml_tensor *p : params_) ggml_set_param(p);
    }
    // physical force squared error, for the metrics (not differentiated)
    ggml_tensor *se_f = ggml_sum(ctx, ggml_sqr(ctx, ggml_mul(ctx, d_force, force_scale)));
    g->stats = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, loss, loss_e, 0), loss_f, 0), se_f, 0);
    ggml_set_output(g->stats);
    // per structure: energy error and the atom-mean physical force squared error
    g->per_atom = ggml_sum_rows(ctx, ggml_sqr(ctx, ggml_mul(ctx, d_force, force_scale)));
    ggml_set_output(g->per_atom);
    g->per_system = ggml_concat(ctx, d_energy, ggml_mul_mat(ctx, atom_weight, ggml_reshape_2d(ctx, g->per_atom, n, 1)), 1);
    ggml_set_output(g->per_system);

    g->gf = ggml_new_graph_custom(ctx, graph_size, backward);
    ggml_build_forward_expand(g->gf, g->stats);
    ggml_build_forward_expand(g->gf, g->per_system);
    g->n_forward = ggml_graph_n_nodes(g->gf);
    if (backward) {
      ggml_set_loss(loss);
      // parameter gradients accumulate in place into the persistent views
      std::vector<ggml_tensor *> accs(ggml_graph_n_nodes(g->gf), nullptr);
      for (int i = 0; i < ggml_graph_n_nodes(g->gf); i++) {
        auto it = acc_.find(ggml_graph_node(g->gf, i));
        if (it != acc_.end()) accs[i] = it->second;
      }
      ggml_build_backward_expand(ctx, g->gf, accs.data());
      g->loss_grad = ggml_graph_get_grad_acc(g->gf, loss);
      ggml_set_input(g->loss_grad);
    }
    if (shared) {
      // the schedule: stage 1's nodes, then gf's others (stage 2), expanded node by node (both
      // are in topological order) so that its use counts cover both stages: ggml-cuda fuses a
      // node into its consumer (and skips writing it) when its use count is 1, and a forward
      // value read again in stage 2 must not be fused away in stage 1. The stages are views.
      g->sched = ggml_new_graph_custom(ctx, 2 * graph_size, false);
      for (int i = 0; i < ggml_graph_n_nodes(g->gforces); i++) ggml_build_forward_expand(g->sched, ggml_graph_node(g->gforces, i));
      const int n1 = ggml_graph_n_nodes(g->sched);
      for (int i = 0; i < ggml_graph_n_nodes(g->gf); i++) ggml_build_forward_expand(g->sched, ggml_graph_node(g->gf, i));
      g->stage1 = std::make_unique<ggml_cgraph>(ggml_graph_view(g->sched, 0, n1));
      g->stage2 = std::make_unique<ggml_cgraph>(ggml_graph_view(g->sched, n1, ggml_graph_n_nodes(g->sched)));
      for (int i = 0; i < g->n_forces_forward; i++) g->forces_forward.push_back(ggml_graph_node(g->gforces, i));
      std::sort(g->forces_forward.begin(), g->forces_forward.end());
      for (int i = 0; i < g->n_forward; i++) g->mixed_forward.push_back(ggml_graph_node(g->gf, i));
      std::sort(g->mixed_forward.begin(), g->mixed_forward.end());
    }
  }
  g->constants = interp.take_constants();
  if (g->loss_grad) g->constants.emplace_back(g->loss_grad, 1.0f);
  g->constants.insert(g->constants.end(), g->constants_extra.begin(), g->constants_extra.end());
  for (auto &c : g->constants) ggml_set_input(c.first);

  if (cached) {
    // inputs (packed: one upload) and constants live in a buffer of their
    // own; the allocator skips tensors that already have memory
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(model_.backend());
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t size = 0;
    std::vector<std::pair<ggml_tensor *, size_t>> place;
    auto reserve = [&](ggml_tensor *t) {
      place.emplace_back(t, size);
      size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
    };
    for (ggml_tensor *t : g->inputs) reserve(t);
    g->blob_bytes = size;
    for (auto &c : g->constants) reserve(c.first);
    g->persist = ggml_backend_buft_alloc_buffer(buft, size);
    if (!g->persist) throw std::runtime_error("cannot allocate graph inputs");
    char *base = static_cast<char *>(ggml_backend_buffer_get_base(g->persist));
    g->blob = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(g->blob_bytes));
    ggml_backend_tensor_alloc(g->persist, g->blob, base);
    for (auto &[t, off] : place) ggml_backend_tensor_alloc(g->persist, t, base + off);
    fill_constants(g->constants);
  }
  if (measure) ctx_bytes = ggml_used_mem(ctx) * 5 / 4 + ggml_tensor_overhead() * 256;
  if (after_build) after_build(g->gf);
  return g;
}

Worker::StepGraph &Worker::cached_graph(int n, int m, int S, Mode mode) {
  const auto key = std::make_tuple(n, m, S, int(mode));
  auto it = cache_.find(key);
  if (it != cache_.end()) {
    cache_hits++;
    it->second->last_use = ++clock_;
    return *it->second;
  }
  cache_misses++;
  while (cache_.size() >= cache_limit_ && !cache_.empty()) // least recently used out
    cache_.erase(std::min_element(cache_.begin(), cache_.end(), [](auto &a, auto &b) {
      return a.second->last_use < b.second->last_use;
    }));
  auto g = build_graph(n, m, S, mode, true);
  // all cached graphs share the compute buffer (they run one at a time); when
  // it has to grow, the old buffer is freed and every other graph dangles
  const size_t before = compute_buffer_bytes();
  if (!ggml_gallocr_alloc_graph(galloc_, g->sched ? g->sched : g->gf))
    throw std::runtime_error(fmt::format("cannot allocate the graph ({} atoms x {} neighbours)", n, m));
  if (compute_buffer_bytes() != before && !cache_.empty()) cache_.clear(), cache_flushes++;
  g->last_use = ++clock_;
  return *(cache_[key] = std::move(g));
}

void Worker::upload_inputs(StepGraph &g, const Batch &b, const std::vector<float> *dv,
                           const std::vector<float> *dc) {
  auto [e_norm, f_norm] = loss_norms(b);
  if (conservative.on) e_norm *= conservative.energy, f_norm *= conservative.nc_force;
  const float norms[2] = {e_norm, f_norm};
  auto span = [](const auto &v) {
    return std::pair<const void *, size_t>(v.data(), v.size() * sizeof(v[0]));
  };
  // in build_graph's order: the model's inputs, the loss's, the tangents
  std::vector<std::pair<const void *, size_t>> src = {
      span(b.species),        span(b.neighbor_species), span(b.edge_vectors),
      span(b.padding_mask),   span(b.reverse_neighbor_index), span(b.cutoff_values)};
  if (const Conditioning &cond = model_.conditioning(); cond.enabled) {
    cond_rows_[0].resize(b.atom_charge.size()), cond_rows_[1].resize(b.atom_charge.size());
    for (size_t a = 0; a < b.atom_charge.size(); a++)
      std::tie(cond_rows_[0][a], cond_rows_[1][a]) = cond.rows(b.atom_charge[a], b.atom_spin[a]);
    src.insert(src.end(), {span(cond_rows_[0]), span(cond_rows_[1])});
  }
  if (g.inputs.size() > src.size())
    src.insert(src.end(), {span(b.atom_weight), span(b.energy_target), span(b.force_target),
                           span(b.force_mask), span(b.force_scale), {norms, sizeof norms}});
  if (dv) src.insert(src.end(), {span(*dv), span(*dc)});
  if (src.size() != g.inputs.size()) throw std::logic_error("upload_inputs: input count");
  if (g.qref) {
    std::vector<float> ref(ggml_nelements(g.qref), 0.0f), ok(ggml_nelements(g.qok), 0.0f);
    std::copy_n(b.ref_charge.begin(), std::min(ref.size(), b.ref_charge.size()), ref.begin());
    std::copy_n(b.ref_ok.begin(), std::min(ok.size(), b.ref_ok.size()), ok.begin());
    double n_ok = 0;
    for (float x : ok) n_ok += x;
    const float w = static_cast<float>(charge_corr.weight * (b.charge_norm > 0 ? b.charge_norm : 1.0 / std::max(1.0, n_ok)));
    ggml_backend_tensor_set(g.qref, ref.data(), 0, ggml_nbytes(g.qref));
    if (g.qok->buffer) ggml_backend_tensor_set(g.qok, ok.data(), 0, ggml_nbytes(g.qok)); // (FORCES graphs: r only)
    if (g.qw->buffer) ggml_backend_tensor_set(g.qw, &w, 0, sizeof(float));
  }
  if (g.plato) {
    g.plato->upload(b.positions.data());
    if (g.dx) {
      if (plato_dx_.size() != size_t(3) * b.n_atoms) throw std::logic_error("upload_inputs: position tangent");
      ggml_backend_tensor_set(g.dx, plato_dx_.data(), 0, ggml_nbytes(g.dx));
      ggml_backend_tensor_set(g.jvp_scale, &plato_jvp_scale_, 0, sizeof(float));
    }
  }
  for (size_t i = 0; i < g.inputs.size(); i++)
    if (ggml_nbytes(g.inputs[i]) != src[i].second)
      throw std::logic_error(std::string("upload size mismatch for ") + ggml_get_name(g.inputs[i]));
  if (g.blob) {
    staging_.resize(g.blob_bytes);
    const char *base = static_cast<const char *>(g.blob->data);
    for (size_t i = 0; i < g.inputs.size(); i++)
      std::memcpy(staging_.data() + (static_cast<const char *>(g.inputs[i]->data) - base),
                  src[i].first, src[i].second);
    ggml_backend_tensor_set(g.blob, staging_.data(), 0, g.blob_bytes);
  } else {
    for (size_t i = 0; i < g.inputs.size(); i++)
      ggml_backend_tensor_set(g.inputs[i], src[i].first, 0, src[i].second);
  }
}

Worker::StepGraph &Worker::graph(int n, int m, int S, Mode mode, std::unique_ptr<StepGraph> &once) {
  auto t0 = clock_type::now();
  if (cache_limit_ > 0 && !model_.plato_lr().enabled) { // Plato-PET-LR graphs depend on the batch
    StepGraph &g = cached_graph(n, m, S, mode);
    t_build += seconds_since(t0);
    return g;
  }
  once = build_graph(n, m, S, mode, false);
  t_build += seconds_since(t0);
  t0 = clock_type::now();
  if (!ggml_gallocr_alloc_graph(galloc_, once->sched ? once->sched : once->gf))
    throw std::runtime_error(fmt::format("cannot allocate the graph ({} atoms x {} neighbours)", n, m));
  fill_constants(once->constants);
  t_alloc += seconds_since(t0);
  return *once;
}

void Worker::compute(StepGraph &g, ggml_cgraph *which) {
  auto t0 = clock_type::now();
  ggml_cgraph *gf = which ? which : g.gf;
  if (profile) {
    const std::string kind = g.mode == FORCES ? "F." : g.mode == MIXED ? "M." : "";
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
      ggml_cgraph one = ggml_graph_view(gf, i, i + 1);
      auto tn = clock_type::now();
      if (ggml_backend_graph_compute(model_.backend(), &one) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph compute failed");
      ggml_backend_synchronize(model_.backend());
      const ggml_tensor *node = ggml_graph_node(gf, i);
      std::string phase;
      if (g.sched && gf == g.stage1.get())
        phase = std::string("F.") + (std::binary_search(g.forces_forward.begin(), g.forces_forward.end(), node) ? "fwd" : "bwd");
      else if (g.sched && gf != g.gf)
        phase = std::string("M.") + (!std::binary_search(g.mixed_forward.begin(), g.mixed_forward.end(), node) ? "bwd"
                                     : g.jvp_mark && node > g.jvp_mark                                       ? "tan"
                                                                                                             : "fwd");
      else phase = kind + (i >= g.n_forward ? "bwd" : g.jvp_mark && node > g.jvp_mark ? "tan" : "fwd");
      auto &s = profile_ops[phase + ":" + ggml_op_desc(node)];
      auto [f, b] = node_cost(node);
      const double ms = 1e3 * seconds_since(tn);
      s.ms += ms, s.count++, s.flops += f, s.bytes += b;
      {
        auto src_op = [&](int k) {
          const ggml_tensor *t = node->src[k];
          while (t && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE)) t = t->src[0];
          return t ? std::string(ggml_op_desc(t)) : std::string("-");
        };
        auto &p = profile_patterns[phase + ":" + ggml_op_desc(node) + "(" + src_op(0) + ", " + src_op(1) + ")"];
        p.ms += ms, p.count++, p.flops += f, p.bytes += b;
      }
      if (node->op == GGML_OP_CONT || node->op == GGML_OP_CPY) {
        // what the copy materialises: the chain of views down to the producer
        std::string key = phase + ":" + ggml_op_desc(node) + " <-";
        const ggml_tensor *t = node->src[0];
        for (int depth = 0; t && depth < 4; depth++, t = t->src[0]) {
          key += std::string(" ") + ggml_op_desc(t);
          if (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE &&
              t->op != GGML_OP_TRANSPOSE)
            break;
        }
        auto &c = profile_copies[key];
        c.ms += ms, c.count++, c.bytes += b;
      }
      profile_nodes.push_back({ms, fmt::format("{}:{} {} [{} {} {} {}]{}", phase,
                                               ggml_op_desc(node), node->name, node->ne[0], node->ne[1],
                                               node->ne[2], node->ne[3],
                                               ggml_is_contiguous(node) ? "" : " non-contiguous")});
    }
  } else if (ggml_backend_graph_compute(model_.backend(), gf) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("graph compute failed");
  }
  t_compute += seconds_since(t0);
  if (after_compute && !(g.sched && gf == g.stage1.get())) after_compute(g.gf);
}

StepStats Worker::run(const Batch &b, double energy_scale, bool backward) {
  t_build = t_alloc = t_upload = t_compute = t_download = t_cpu = 0;
  plato_batch_ = &b;
  if (conservative.on) return run_conservative(b, energy_scale, backward);
  if (!model_.nc_forces()) throw std::runtime_error("an energy-only model trains with conservative forces only");
  const int n = b.n_atoms, m = b.max_neighbors, S = b.n_systems;
  std::unique_ptr<StepGraph> once;
  StepGraph *g = &graph(n, m, S, backward ? BACKWARD : FORWARD, once);

  auto t0 = clock_type::now();
  upload_inputs(*g, b);
  t_upload = seconds_since(t0);
  compute(*g);

  t0 = clock_type::now();
  float stats[4];
  ggml_backend_tensor_get(g->stats, stats, 0, sizeof stats);
  StepStats st;
  st.loss = stats[0];
  st.loss_energy = stats[1];
  st.loss_force = stats[2];
  // Σ d_e^2 = loss_e / e_norm, in units of the energy scale
  st.se_energy_per_atom = double(stats[1]) / loss_norms(b).first * energy_scale * energy_scale;
  st.se_force = stats[3];
  st.n_systems = b.n_real_systems;
  charge_corr_stats(*g, b, st);
  for (float w : b.force_mask) st.n_force_components += 3 * static_cast<int64_t>(w);
  add_target_sums(b, energy_scale, st);
  if (system_errors) {
    std::vector<float> ps(2 * static_cast<size_t>(S));
    ggml_backend_tensor_get(g->per_system, ps.data(), 0, ps.size() * sizeof(float));
    for (int k = 0; k < b.n_real_systems; k++)
      system_errors->push_back({b.ids[k], -1, b.n_per_system[k], static_cast<float>(ps[k] * energy_scale),
                                ps[S + k] * b.n_per_system[k]});
    if (atom_errors) { // real atoms come first, structure by structure
      const int n_real = std::accumulate(b.n_per_system.begin(), b.n_per_system.begin() + b.n_real_systems, 0);
      const size_t at = atom_errors->size();
      atom_errors->resize(at + n_real);
      ggml_backend_tensor_get(g->per_atom, atom_errors->data() + at, 0, n_real * sizeof(float));
    }
  }
  t_download = seconds_since(t0);
  return st;
}

StepStats Worker::run_conservative(const Batch &b, double energy_scale, bool backward) {
  const int n = b.n_atoms, m = b.max_neighbors, S = b.n_systems;
  const auto [e_norm, f_norm] = loss_norms(b);
  const Conservative &w = conservative;
  plato_batch_ = &b;
  const bool nc = model_.nc_forces();
  const int ow = nc ? 4 : 1; // graph output columns: (energy | direct force) or energy

  // 1. forward + backward to the edge vectors and cutoff values; with the shared
  // forward, stage 1 of the MIXED graph (its tangent inputs are uploaded after 3.)
  const bool shared = backward && shared_forward_;
  std::unique_ptr<StepGraph> once;
  StepGraph &gf = graph(n, m, S, shared ? MIXED : FORCES, once);
  auto t0 = clock_type::now();
  if (shared) {
    std::vector<float> zv(3 * static_cast<size_t>(n) * m, 0.0f), zc(static_cast<size_t>(n) * m, 0.0f);
    if (gf.dx) plato_dx_.assign(3 * static_cast<size_t>(n), 0.0f), plato_jvp_scale_ = 1.0f;
    upload_inputs(gf, b, &zv, &zc);
  } else {
    upload_inputs(gf, b);
  }
  t_upload += seconds_since(t0);
  compute(gf, shared ? gf.stage1.get() : nullptr);
  t0 = clock_type::now();
  std::vector<float> out(ow * static_cast<size_t>(n)), gv(3 * static_cast<size_t>(n) * m), gc, gx;
  ggml_backend_tensor_get(gf.out, out.data(), 0, out.size() * sizeof(float));
  ggml_backend_tensor_get(gf.grad_v, gv.data(), 0, gv.size() * sizeof(float));
  const bool taper = !b.taper_center.empty();
  if (taper) {
    if (!gf.grad_c) throw std::runtime_error("conservative forces: no cutoff-value gradient");
    gc.resize(static_cast<size_t>(n) * m);
    ggml_backend_tensor_get(gf.grad_c, gc.data(), 0, gc.size() * sizeof(float));
  }
  StepStats st;
  charge_corr_stats(gf, b, st);
  if (gf.grad_x) { // Plato-PET-LR: the explicit position dependence (mesh, LES)
    gx.resize(3 * static_cast<size_t>(n));
    ggml_backend_tensor_get(gf.grad_x, gx.data(), 0, gx.size() * sizeof(float));
  }
  if (!shared) once.reset();
  t_download += seconds_since(t0);
  const auto t_host = clock_type::now();

  // 2. F = -dE/dx (scaled energy units): edge gradients to their atoms, and
  // through the adaptive radii, dE/dr_k = S_k = sum over edges touching k
  // of dE/dc / 2 (pair cutoffs (r_i + r_j) / 2), along the taper edges
  // (k -> l): dE/dv += S_k J v / |v| (runtime/graph_model.cpp's chain)
  auto real = [&](int f) { return b.padding_mask[f] < 0.5f && b.neighbor_atom[f] >= 0; };
  std::vector<double> F(3 * static_cast<size_t>(n), 0.0);
  for (int f = 0; f < n * m; f++) {
    if (!real(f)) continue;
    const int i = f / m, k = b.neighbor_atom[f];
    for (int c = 0; c < 3; c++) F[3 * i + c] += gv[3 * f + c], F[3 * k + c] -= gv[3 * f + c];
  }
  for (size_t k = 0; k < gx.size(); k++) F[k] -= gx[k];
  std::vector<double> Sk;
  if (taper) {
    Sk.assign(n, 0.0);
    for (int f = 0; f < n * m; f++)
      if (real(f)) Sk[f / m] += 0.5 * gc[f], Sk[b.neighbor_atom[f]] += 0.5 * gc[f];
    for (size_t t = 0; t < b.taper_center.size(); t++) {
      const int k = b.taper_center[t], l = b.taper_neighbor[t];
      const float *v = &b.taper_vector[3 * t];
      const double d = std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]);
      const double coef = d > 0 ? Sk[k] * b.taper_jacobian[t] / d : 0.0;
      for (int c = 0; c < 3; c++) F[3 * k + c] += coef * v[c], F[3 * l + c] -= coef * v[c];
    }
  }

  // the loss terms and metrics; force targets are physical F / s_Z, the
  // conservative ones F / s_E (the gradient of the scaled energy)
  st.n_systems = b.n_real_systems;
  std::vector<double> d_e(S, 0.0), u(3 * static_cast<size_t>(n), 0.0), fse(S, 0.0);
  for (int k = 0; k < S; k++) {
    for (int a = 0; a < n; a++) d_e[k] += double(b.atom_weight[static_cast<size_t>(k) * n + a]) * out[ow * a];
    d_e[k] -= b.energy_target[k];
    st.loss_energy += w.energy * e_norm * loss.value(d_e[k], loss.delta_e);
    st.se_energy_per_atom += d_e[k] * d_e[k] * energy_scale * energy_scale;
    if (k < b.n_real_systems) st.ae_energy_per_atom += std::abs(d_e[k]) * energy_scale;
  }
  add_target_sums(b, energy_scale, st);
  std::vector<float> per_atom(n, 0.0f);
  for (int a = 0, sys = 0, left = S ? b.n_per_system[0] : 0; a < n; a++) {
    if (b.force_mask[a] == 0.0f) continue;
    while (left == 0 && sys + 1 < S) left = b.n_per_system[++sys];
    left--;
    st.n_force_components += 3;
    double e2 = 0;
    for (int c = 0; c < 3; c++) {
      const double target = double(b.force_target[3 * a + c]) * b.force_scale[a];
      const double r = F[3 * a + c] - target / energy_scale, rnc = nc ? out[4 * a + 1 + c] - b.force_target[3 * a + c] : 0.0;
      st.loss_force += w.force * f_norm * loss.value(r, loss.delta_f);
      st.loss_nc += w.nc_force * f_norm * loss.value(rnc, loss.delta_f);
      st.se_force_nc += std::pow(rnc * b.force_scale[a], 2);
      e2 += std::pow(r * energy_scale, 2);
      st.ae_force += std::abs(r) * energy_scale;
      u[3 * a + c] = w.force * f_norm * loss.slope(r, loss.delta_f);
    }
    st.se_force += e2;
    per_atom[a] = static_cast<float>(e2);
    fse[sys] += e2;
  }
  st.loss = st.loss_energy + st.loss_force + st.loss_nc + st.loss_charge;
  if (system_errors) {
    for (int k = 0; k < b.n_real_systems; k++)
      system_errors->push_back({b.ids[k], -1, b.n_per_system[k], static_cast<float>(d_e[k] * energy_scale),
                                static_cast<float>(fse[k])});
    if (atom_errors) { // real atoms come first, structure by structure
      const int n_real = std::accumulate(b.n_per_system.begin(), b.n_per_system.begin() + b.n_real_systems, 0);
      atom_errors->insert(atom_errors->end(), per_atom.begin(), per_atom.begin() + n_real);
    }
  }
  if (!backward) return t_cpu = seconds_since(t_host), st;

  // 3. tangents along u (atoms): edge (i -> k) moves by u_k - u_i, a pair
  // cutoff by (dr_i + dr_k) / 2, dr_k = sum over taper edges (k -> l) of
  // J (v / |v|) . (u_l - u_k); then the loss minus <dE/dx, -u>, backward
  std::vector<float> dv(3 * static_cast<size_t>(n) * m, 0.0f), dc(static_cast<size_t>(n) * m, 0.0f);
  for (int f = 0; f < n * m; f++) {
    if (!real(f)) continue;
    const int i = f / m, k = b.neighbor_atom[f];
    for (int c = 0; c < 3; c++) dv[3 * f + c] = static_cast<float>(u[3 * k + c] - u[3 * i + c]);
  }
  if (taper) {
    std::vector<double> dr(n, 0.0);
    for (size_t t = 0; t < b.taper_center.size(); t++) {
      const int k = b.taper_center[t], l = b.taper_neighbor[t];
      const float *v = &b.taper_vector[3 * t];
      const double d = std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]);
      if (!(d > 0)) continue;
      double proj = 0;
      for (int c = 0; c < 3; c++) proj += v[c] * (u[3 * l + c] - u[3 * k + c]);
      dr[k] += b.taper_jacobian[t] * proj / d;
    }
    for (int f = 0; f < n * m; f++)
      if (real(f)) dc[f] = static_cast<float>(0.5 * (dr[f / m] + dr[b.neighbor_atom[f]]));
  }
  if (model_.plato_lr().enabled) {
    // the long-range ops' coordinate tangents are central differences (runtime/graph_jvp.h):
    // tangents of O(1), the J term scaled back
    double umax = 0;
    for (double x : u) umax = std::max(umax, std::abs(x));
    if (!(umax > 0)) umax = 1;
    const float inv = static_cast<float>(1.0 / umax);
    for (float &x : dv) x *= inv;
    for (float &x : dc) x *= inv;
    plato_dx_.resize(u.size());
    for (size_t k = 0; k < u.size(); k++) plato_dx_[k] = static_cast<float>(u[k] / umax);
    plato_jvp_scale_ = static_cast<float>(umax);
  }
  t_cpu = seconds_since(t_host);
  if (shared) { // stage 2: the tangents, then the rest of the MIXED graph
    t0 = clock_type::now();
    ggml_tensor *tv = gf.inputs[gf.inputs.size() - 2], *tc = gf.inputs.back();
    if (ggml_nbytes(tv) != dv.size() * sizeof(float) || ggml_nbytes(tc) != dc.size() * sizeof(float))
      throw std::logic_error("run_conservative: tangent inputs");
    ggml_backend_tensor_set(tv, dv.data(), 0, ggml_nbytes(tv));
    ggml_backend_tensor_set(tc, dc.data(), 0, ggml_nbytes(tc));
    if (gf.dx) {
      ggml_backend_tensor_set(gf.dx, plato_dx_.data(), 0, ggml_nbytes(gf.dx));
      ggml_backend_tensor_set(gf.jvp_scale, &plato_jvp_scale_, 0, sizeof(float));
    }
    t_upload += seconds_since(t0);
    compute(gf, gf.stage2.get());
    return st;
  }
  StepGraph &gm = graph(n, m, S, MIXED, once);
  t0 = clock_type::now();
  upload_inputs(gm, b, &dv, &dc);
  t_upload += seconds_since(t0);
  compute(gm);
  return st;
}

namespace {

constexpr float ns_a = 3.4445f, ns_b = -4.7750f, ns_c = 2.0315f;

/// Polar Express (Amsel et al. 2025): per-iteration minimax quintics, from
/// optimal_composition(l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2,
/// cushion=0.02) of github.com/NoahAmsel/PolarExpress (polar_express.py)
constexpr std::array<std::array<float, 3>, 10> polar_express = {{
    {8.237312490495555f, -23.1577474145582f, 16.680568411445915f},
    {4.082441999064829f, -2.8930477353325843f, 0.5252849256975644f},
    {3.9263479922546485f, -2.854746803476524f, 0.5318022422894979f},
    {3.2982187133085197f, -2.4245419810267106f, 0.4863200835884415f},
    {2.2970369434552587f, -1.6366255812590325f, 0.4002628455953631f},
    {1.8763805351440381f, -1.23478965777222f, 0.3589188750166826f},
    {1.856442348565278f, -1.2132449881003775f, 0.3568003487859341f},
    {1.85643356156706f, -1.2132336029471178f, 0.3567976320793584f},
    {1.8564311647224787f, -1.2132289037487436f, 0.3567953287942881f},
    {1.8749954775631825f, -1.249990955148181f, 0.37499547758499846f},
}};

/// (a, b, c) per Newton-Schulz iteration: Muon's fixed quintic, or with
/// MLIPCPP_NS_POLAR=1 the Polar Express sequence (the last one repeated past
/// 10). Its 1.01 norm cushion, X / (1.01 |X|), is folded into the first
/// iteration: p(x / s) = (a / s) x + (b / s^3) x^3 + (c / s^5) x^5.
std::vector<std::array<float, 3>> ns_coeffs(int steps) {
  static const bool polar = std::getenv("MLIPCPP_NS_POLAR") != nullptr;
  std::vector<std::array<float, 3>> v;
  for (int i = 0; i < steps; i++)
    v.push_back(polar ? polar_express[std::min<size_t>(i, polar_express.size() - 1)]
                      : std::array<float, 3>{ns_a, ns_b, ns_c});
  if (polar && steps > 0) {
    const float s = 1.01f;
    v[0] = {v[0][0] / s, v[0][1] / (s * s * s), v[0][2] / (s * s * s * s * s)};
  }
  return v;
}

/// ggml_opt_step_sgd as an in-place axpby on an intermediate: x <- x (1 - lr wd) - lr y.
/// The PARAM flag it asserts only matters to backward construction, and this
/// graph is forward only.
ggml_tensor *axpby_inplace(ggml_context *ctx, ggml_tensor *x, ggml_tensor *y, ggml_tensor *lr_wd) {
  x->flags |= GGML_TENSOR_FLAG_PARAM;
  return ggml_opt_step_sgd(ctx, x, y, lr_wd);
}

/// per iteration, (lr, wd) for both axpbys: B = c A^2 + b A, X = B X + a X
std::vector<float> ns_axpby_values(int steps) {
  std::vector<float> v;
  for (auto [a, b, c] : ns_coeffs(steps)) v.insert(v.end(), {-b, (1 - c) / -b, -a, 0.0f});
  return v;
}

/// Newton-Schulz quintic iteration (Muon): approximately U V^T of each
/// g[:, :, i] = U S V^T. ggml (ne0, ne1) = PyTorch (cols, rows); the Gram
/// matrix is taken over the shorter side. Same-shape matrices are stacked
/// along ne2 and orthogonalised together (batched mul_mat).
/// (The GPU default is the fused op instead: orthogonalize_all.)
/// `axpby` (optional): ns_axpby_values(steps) (4 per iteration); then each
/// iteration's two linear combinations run as in-place ggml_opt_step_sgd
/// (one kernel each instead of scale, scale, add / scale, add);
/// MLIPCPP_NS_SCALE_ADD=1: the separate ops (A/B).
ggml_tensor *orthogonalize(ggml_context *ctx, ggml_tensor *g, int steps, ggml_tensor *axpby = nullptr) {
  const std::vector<std::array<float, 3>> abc = ns_coeffs(steps);
  static const bool scale_add = std::getenv("MLIPCPP_NS_SCALE_ADD") != nullptr;
  if (scale_add) axpby = nullptr;
  const bool tall = g->ne[0] < g->ne[1]; // more rows than cols: work on g^T
  ggml_tensor *x = tall ? ggml_cont(ctx, ggml_transpose(ctx, g)) : g;
  const int64_t cols = x->ne[0], rows = x->ne[1], k = x->ne[2];
  // per-matrix Frobenius norm
  ggml_tensor *n2 = ggml_sum_rows(ctx, ggml_reshape_2d(ctx, ggml_sqr(ctx, x), cols * rows, k));
  x = ggml_div(ctx, x, ggml_reshape_3d(ctx, ggml_scale_bias(ctx, ggml_sqrt(ctx, n2), 1.0f, 1e-7f), 1, 1, k));
  for (int it = 0; it < steps; it++) {
    const auto [a, b, c] = abc[it];
    ggml_tensor *A = ggml_mul_mat(ctx, x, x); // X X^T
    if (axpby) {
      const size_t o = 4 * it * sizeof(float);
      ggml_tensor *B = axpby_inplace(ctx, ggml_mul_mat(ctx, A, A), A, ggml_view_1d(ctx, axpby, 2, o));
      ggml_tensor *BX = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)), B); // B symmetric
      x = axpby_inplace(ctx, BX, x, ggml_view_1d(ctx, axpby, 2, o + 2 * sizeof(float)));
      continue;
    }
    ggml_tensor *B = ggml_add(ctx, ggml_scale(ctx, A, b), ggml_scale(ctx, ggml_mul_mat(ctx, A, A), c));
    ggml_tensor *BX = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)), B); // B X (B symmetric)
    x = ggml_add(ctx, ggml_scale(ctx, x, a), BX);
  }
  return tall ? ggml_cont(ctx, ggml_transpose(ctx, x)) : x;
}
/// orthogonalize() of every stack in `us`. On a GPU backend one
/// ggml_newton_schulz_multi per GGML_MAX_SRC stacks instead (bf16 storage,
/// all the groups' GEMMs of a stage in one grouped cuBLAS call), each result a
/// view of its output: Muon step 3.01 -> 2.41 ms on the big model (bench
/// 145961). MLIPCPP_NS_OPS=1, or a CPU backend (the op's CPU kernel is a
/// reference): the ggml ops.
std::vector<ggml_tensor *> orthogonalize_all(ggml_context *ctx, ggml_backend_t backend, const std::vector<ggml_tensor *> &us,
                                             int steps, ggml_tensor *axpby) {
  std::vector<ggml_tensor *> os;
  const std::vector<std::array<float, 3>> abc = ns_coeffs(steps);
  ggml_backend_dev_t dev = ggml_backend_get_device(backend);
  if (std::getenv("MLIPCPP_NS_OPS") || steps > GGML_NS_MAX_STEPS || !dev ||
      ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
    for (ggml_tensor *u : us) os.push_back(orthogonalize(ctx, u, steps, axpby));
    return os;
  }
  for (size_t i0 = 0; i0 < us.size(); i0 += GGML_MAX_SRC) {
    std::vector<ggml_tensor *> xs(us.begin() + i0, us.begin() + std::min(us.size(), i0 + GGML_MAX_SRC));
    ggml_tensor *o = ggml_newton_schulz_multi_coeffs(ctx, xs.data(), static_cast<int>(xs.size()), steps, abc.data()->data());
    size_t off = 0;
    for (ggml_tensor *x : xs) {
      os.push_back(ggml_view_3d(ctx, o, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], off));
      off += ggml_nbytes(x);
    }
  }
  return os;
}
/// CUDA graph capture for a static graph computed every step (no-op off CUDA);
/// MLIPCPP_OPT_NO_GRAPH=1: off. Returns whether capture was allowed.
bool allow_cuda_graph(ggml_backend_t backend, ggml_cgraph *graph) {
  if (std::getenv("MLIPCPP_OPT_NO_GRAPH") || ggml_graph_n_nodes(graph) == 0) return false;
  ggml_backend_dev_t dev = ggml_backend_get_device(backend);
  if (!dev) return false;
  using allow_fn = void (*)(ggml_backend_t, const void *);
  auto allow = reinterpret_cast<allow_fn>(
      ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_allow"));
  if (allow) allow(backend, ggml_graph_node(graph, 0));
  return allow != nullptr;
}
} // namespace

#ifdef MLIPCPP_NCCL
namespace {
void nccl_check(ncclResult_t r, const char *what) {
  if (r != ncclSuccess) throw std::runtime_error(fmt::format("{}: {}", what, ncclGetErrorString(r)));
}
void cuda_check(cudaError_t r, const char *what) {
  if (r != cudaSuccess) throw std::runtime_error(fmt::format("{}: {}", what, cudaGetErrorString(r)));
}
} // namespace

// one communicator and one stream per device (workers_[w] on CUDA device w)
struct Optimizer::Nccl {
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
  explicit Nccl(int W) : comms(W), streams(W) {
    std::vector<int> devices(W);
    std::iota(devices.begin(), devices.end(), 0);
    nccl_check(ncclCommInitAll(comms.data(), W, devices.data()), "ncclCommInitAll");
    for (int d = 0; d < W; d++) {
      cuda_check(cudaSetDevice(d), "cudaSetDevice");
      cuda_check(cudaStreamCreateWithFlags(&streams[d], cudaStreamNonBlocking), "cudaStreamCreate");
    }
  }
  ~Nccl() {
    for (size_t d = 0; d < comms.size(); d++) {
      cudaSetDevice(static_cast<int>(d));
      cudaStreamDestroy(streams[d]);
      ncclCommDestroy(comms[d]);
    }
  }
};
#else
struct Optimizer::Nccl {};
#endif

namespace {
std::vector<Worker *> raw(std::vector<std::unique_ptr<Worker>> &workers) {
  std::vector<Worker *> out;
  for (auto &w : workers) out.push_back(w.get());
  return out;
}
} // namespace

Optimizer::Optimizer(std::vector<std::unique_ptr<Worker>> &workers, double clip, bool muon,
                     double muon_momentum, double muon_lr_scale, int ns_steps, bool allreduce)
    : Optimizer(raw(workers), clip, muon, muon_momentum, muon_lr_scale, ns_steps, allreduce) {}

Optimizer::Optimizer(const std::vector<Worker *> &workers, double clip, bool muon, double muon_momentum,
                     double muon_lr_scale, int ns_steps, bool allreduce)
    : workers_(workers), muon_lr_scale_(muon_lr_scale) {
  if (allreduce && workers_.size() > 1) {
#ifdef MLIPCPP_NCCL
    // one single-device optimizer per worker; the gradients are averaged in
    // place first (step_allreduce), so every replica takes the same step
    for (Worker *w : workers_)
      replicas_.push_back(std::make_unique<Optimizer>(std::vector<Worker *>{w}, clip, muon, muon_momentum,
                                                      muon_lr_scale, ns_steps, false));
    n_muon_tensors = replicas_[0]->n_muon_tensors;
    muon_groups = replicas_[0]->muon_groups;
    graph_captured_ = replicas_[0]->graph_captured_;
    nccl_ = std::make_unique<Nccl>(static_cast<int>(workers_.size()));
    int version = 0;
    ncclGetVersion(&version);
    allreduce_info = fmt::format("NCCL {} all-reduce (average) over {} devices, a replicated step on each",
                                 version, workers_.size());
    return;
#else
    throw std::runtime_error("--allreduce 1 needs a build with NCCL (module load nccl, then cmake)");
#endif
  }
  Worker &w0 = *workers_[0];
  const auto &params = w0.params();
  const int W = static_cast<int>(workers_.size());
  ggml_backend_t backend = w0.model().backend();
  ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

  // Muon groups: same-shape hidden matrices, orthogonalised together
  // (MLIPCPP_MUON_OLD=1: the per-matrix momentum / concat / update graph, A/B)
  static const bool old_muon = std::getenv("MLIPCPP_MUON_OLD") != nullptr;
  std::map<std::pair<int64_t, int64_t>, std::vector<size_t>> groups; // shape -> parameter indices
  if (muon)
    for (size_t k = 0; k < params.size(); k++)
      if (is_muon_param(params[k])) groups[{params[k]->ne[0], params[k]->ne[1]}].push_back(k);

  // the Muon matrices' owners: in grad_flat order (group by group), device 0
  // up to a share MLIPCPP_MUON_SPLIT (default 1/W) of the Newton-Schulz
  // FLOPs, the rest evenly; owned[group][device] = its matrices [a, b)
  const bool one_device = std::getenv("MLIPCPP_MUON_ONE_DEVICE") != nullptr;
  const char *share_env = std::getenv("MLIPCPP_MUON_SPLIT");
  const double share0 = share_env ? std::atof(share_env) : 1.0 / W;
  std::vector<std::vector<std::pair<size_t, size_t>>> owned;
  std::vector<double> dev_flops(W, 0.0);
  auto assign = [&](bool split) {
    std::vector<double> f; // per matrix
    for (auto &[shape, idx] : groups) {
      const double m = std::min(shape.first, shape.second), n = std::max(shape.first, shape.second);
      f.insert(f.end(), idx.size(), ns_steps * (4 * m * m * n + 2 * m * m * m));
    }
    const double total = std::accumulate(f.begin(), f.end(), 0.0);
    auto owner = [&](double x) {
      if (!split || x < share0 * total) return 0;
      return std::min(W - 1, 1 + static_cast<int>((x - share0 * total) / ((1 - share0) * total) * (W - 1)));
    };
    owned.clear(), std::fill(dev_flops.begin(), dev_flops.end(), 0.0);
    double F = 0;
    size_t j = 0;
    for (auto &[shape, idx] : groups) {
      auto &o = owned.emplace_back(W, std::make_pair(size_t(0), size_t(0)));
      for (size_t i = 0; i < idx.size(); i++, j++) {
        const int d = owner(F + f[j] / 2);
        F += f[j], dev_flops[d] += f[j];
        o[d] = o[d].first == o[d].second ? std::make_pair(i, i + 1) : std::make_pair(o[d].first, i + 1);
      }
    }
    return std::all_of(dev_flops.begin(), dev_flops.end(), [](double x) { return x > 0; });
  };
  const bool split = muon && !old_muon && W > 1 && !one_device && !groups.empty() && assign(true);
  if (!split) assign(false);
  if (split) {
    for (int d = 0; d < W; d++)
      muon_split += fmt::format("{}device {} {:.1f} GFLOP", d ? ", " : "", d, dev_flops[d] / 1e9);
  }
  auto group_offset = [&](const std::vector<size_t> &idx) { return w0.grad_offset(idx[0]); };

  // persistent state on device 0: Adam moments, staging for the other
  // workers' gradients, per Muon group the stacked momentum (of device 0's
  // matrices), hyper-parameters and constants; with the split, the clip
  // factor, the clipped mean gradient and the other devices' updates
  ctx_ = ggml_init({ggml_tensor_overhead() * (2 * params.size() + 2 * W + 2 * groups.size() + 16), nullptr, true});
  std::vector<ggml_tensor *> m, v;
  for (ggml_tensor *p : params) {
    const bool stacked = muon && is_muon_param(p) && !old_muon; // momentum in m_stack instead
    m.push_back(stacked ? nullptr : ggml_dup_tensor(ctx_, p));
    v.push_back(stacked || (muon && is_muon_param(p)) ? nullptr : ggml_dup_tensor(ctx_, p));
  }
  auto new_m_stack = [&](ggml_context *ctx, int d) { // [ne0, ne1, owned count] per group (nullptr: none)
    std::vector<ggml_tensor *> ms;
    size_t gi = 0;
    for (auto &[shape, idx] : groups) {
      const auto [a, b] = owned[gi++][d];
      ms.push_back(a == b ? nullptr : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, shape.first, shape.second, b - a));
      if (ms.back()) ggml_set_param(ms.back()); // stepped in place by ggml_opt_step_sgd
    }
    return ms;
  };
  std::vector<ggml_tensor *> m_stack = new_m_stack(ctx_, 0);
  for (int w = 1; w < W; w++) staging_.push_back(ggml_dup_tensor(ctx_, w0.grad_flat()));
  staged_.assign(W, 0);
  shards_.resize(split ? W - 1 : 0);
  for (size_t s = 0; s < shards_.size(); s++) {
    Shard &sh = shards_[s];
    sh.lo = SIZE_MAX;
    size_t gi = 0;
    for (auto &[shape, idx] : groups) {
      const auto [a, b] = owned[gi++][s + 1];
      const size_t mat = static_cast<size_t>(shape.first * shape.second);
      if (a < b) sh.lo = std::min(sh.lo, group_offset(idx) + a * mat), sh.hi = group_offset(idx) + b * mat;
    }
    sh.upd_in = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, sh.hi - sh.lo);
  }
  if (split) {
    clip_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
    gbuf_ = ggml_dup_tensor(ctx_, w0.grad_flat());
  }
  // one-copy weight broadcast: every parameter at the same offset of a
  // same-size weight buffer on every worker (same GGUF, same allocator)
  blob_broadcast_ = true;
  for (int w = 1; w < W; w++) {
    Worker &ww = *workers_[w];
    if (ggml_nbytes(ww.weights_blob()) != ggml_nbytes(w0.weights_blob())) blob_broadcast_ = false;
    for (size_t k = 0; k < params.size(); k++)
      if (static_cast<char *>(params[k]->data) - static_cast<char *>(w0.weights_blob()->data) !=
          static_cast<char *>(ww.params()[k]->data) - static_cast<char *>(ww.weights_blob()->data))
        blob_broadcast_ = false;
    if (split && ww.grad_offset(0) != w0.grad_offset(0)) throw std::runtime_error("workers' gradient layouts differ");
  }
  pars_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 7);
  ggml_tensor *one = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
  ggml_tensor *eps = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
  muon_lr_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
  // ggml_opt_step_sgd computes x (1 - lr wd) - lr g: with lr = -(1 - mu), wd = -1
  // that is the momentum mu x + (1 - mu) g
  const float mu = static_cast<float>(muon_momentum);
  const float mom_vals[2] = {-(1 - mu), -1.0f};
  const float nest_vals[2] = {-1.0f, mu > 0 ? (1 - mu) / mu - 1 : 0.0f}; // Nesterov axpby
  const std::vector<float> ns_vals = ns_axpby_values(ns_steps);         // Newton-Schulz axpbys, per iteration
  auto muon_constants = [&](ggml_context *ctx) {
    return std::array<ggml_tensor *, 3>{ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2), ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2),
                                        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * std::max(1, ns_steps))};
  };
  auto set_muon_constants = [&](const std::array<ggml_tensor *, 3> &c) {
    ggml_backend_tensor_set(c[0], mom_vals, 0, sizeof mom_vals);
    ggml_backend_tensor_set(c[1], nest_vals, 0, sizeof nest_vals);
    if (!ns_vals.empty()) ggml_backend_tensor_set(c[2], ns_vals.data(), 0, ns_vals.size() * sizeof(float));
  };
  const auto [mom_pars, nest_pars, ns_pars] = muon_constants(ctx_);
  // per group (lr * muon_lr_scale * rms, wd = 0) for the SGD update, set by step()
  muon_pars_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 2 * std::max<int64_t>(1, groups.size()));
  buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
  if (!buf_) throw std::runtime_error("cannot allocate the optimizer state");
  ggml_backend_buffer_clear(buf_, 0);
  const float c1 = 1.0f, ce = 1e-6f;
  ggml_backend_tensor_set(one, &c1, 0, sizeof c1);
  ggml_backend_tensor_set(eps, &ce, 0, sizeof ce);
  set_muon_constants({mom_pars, nest_pars, ns_pars});

  // Muon on a stack t of same-shape gradients (a view): one momentum step,
  // Nesterov (1 - mu) g + mu M divided by mu (Newton-Schulz normalises each
  // matrix first, so the common factor drops out; one in-place axpby on t,
  // read by nothing else after the momentum step); Newton-Schulz follows for
  // all the groups together (orthogonalize_all)
  auto muon_momentum_step = [&](ggml_context *ctx, ggml_tensor *t, ggml_tensor *M_state,
                                const std::array<ggml_tensor *, 3> &c) {
    ggml_tensor *M = ggml_opt_step_sgd(ctx, M_state, t, c[0]);
    return mu > 0 ? axpby_inplace(ctx, t, M, c[1]) : t;
  };

  // step graph: g = mean_w grad_w; c = min(1, clip / (|g| + 1e-6)); Adam(c g)
  // (with the split, g c is the norm graph's output gbuf_: the other devices
  // need c before device 0's step)
  const size_t n_nodes = (16 + 12 * ns_steps) * params.size() + 64;
  gctx_ = ggml_init({ggml_tensor_overhead() * n_nodes + 3 * ggml_graph_overhead_custom(n_nodes, false), nullptr, true});
  ggml_tensor *g = w0.grad_flat();
  for (ggml_tensor *st : staging_) g = ggml_add(gctx_, g, st);
  if (W > 1) g = ggml_scale(gctx_, g, 1.0f / W);
  norm_ = ggml_sqrt(gctx_, ggml_sum(gctx_, ggml_sqr(gctx_, g)));
  ggml_tensor *c = ggml_clamp(gctx_, ggml_scale(gctx_, ggml_div(gctx_, one, ggml_add(gctx_, norm_, eps)),
                                                static_cast<float>(clip)),
                              0.0f, 1.0f);
  ggml_set_output(norm_);
  ggml_tensor *gs = nullptr;
  graph_ = ggml_new_graph_custom(gctx_, n_nodes, false);
  if (split) {
    norm_graph_ = ggml_new_graph_custom(gctx_, n_nodes, false);
    ggml_build_forward_expand(norm_graph_, norm_);
    ggml_build_forward_expand(norm_graph_, ggml_cpy(gctx_, c, clip_));
    ggml_build_forward_expand(norm_graph_, ggml_cpy(gctx_, ggml_mul(gctx_, g, c), gbuf_));
    gs = gbuf_;
  } else {
    gs = ggml_mul(gctx_, g, c);
    ggml_build_forward_expand(graph_, norm_);
  }
  // Muon: (param, Nesterov momentum) per matrix shape, orthogonalised per group
  std::map<std::pair<int64_t, int64_t>, std::vector<std::pair<ggml_tensor *, ggml_tensor *>>> muon_groups;
  for (size_t k = 0; k < params.size(); k++) {
    ggml_tensor *p = params[k];
    const size_t off = w0.grad_offset(k);
    ggml_tensor *gk = ggml_view_4d(gctx_, gs, p->ne[0], p->ne[1], p->ne[2], p->ne[3], p->nb[1], p->nb[2],
                                   p->nb[3], off * sizeof(float));
    if (muon && is_muon_param(p) && !old_muon) {
      n_muon_tensors++; // stepped per group below
    } else if (muon && is_muon_param(p)) {
      // m[k] is the momentum: b = mu b + (1-mu) g; Nesterov u = (1-mu) g + mu b
      ggml_tensor *buf = ggml_add(gctx_, ggml_scale(gctx_, m[k], mu), ggml_scale(gctx_, gk, 1 - mu));
      ggml_build_forward_expand(graph_, ggml_cpy(gctx_, buf, m[k]));
      muon_groups[{p->ne[0], p->ne[1]}].emplace_back(
          p, ggml_add(gctx_, ggml_scale(gctx_, gk, 1 - mu), ggml_scale(gctx_, buf, mu)));
      n_muon_tensors++;
    } else {
      ggml_build_forward_expand(graph_, ggml_opt_step_adamw(gctx_, p, gk, m[k], v[k], pars_));
    }
  }
  // per group: the gradients are one contiguous block of gs (the worker's
  // grouped layout), so device 0's stack is a view; Muon, then per matrix one
  // in-place SGD step at lr * muon_lr_scale * 0.2 sqrt(max dim). The other
  // devices' updates arrive in their upd_in and are applied by apply_graph_
  if (split) apply_graph_ = ggml_new_graph_custom(gctx_, n_nodes, false);
  size_t gi = 0;
  std::vector<ggml_tensor *> us; // device 0's Nesterov stacks, orthogonalised together after the loop
  std::vector<std::pair<std::vector<ggml_tensor *>, ggml_tensor *>> steps; // per stack: (parameters, lr_wd)
  for (auto &[shape, idx] : old_muon ? decltype(groups){} : groups) {
    this->muon_groups.push_back(fmt::format("{}x{}x{}", shape.first, shape.second, idx.size()));
    const size_t mat = static_cast<size_t>(shape.first * shape.second), off0 = group_offset(idx);
    for (size_t i = 0; i < idx.size(); i++)
      if (w0.grad_offset(idx[i]) != off0 + i * mat) throw std::runtime_error("Muon group not contiguous in grad_flat");
    muon_rms_.push_back(0.2f * std::sqrt(static_cast<float>(std::max(shape.first, shape.second))));
    ggml_tensor *lr_wd = ggml_view_1d(gctx_, muon_pars_, 2, 2 * gi * sizeof(float));
    if (const auto [a, b] = owned[gi][0]; a < b) {
      ggml_tensor *t = ggml_view_3d(gctx_, gs, shape.first, shape.second, b - a, shape.first * sizeof(float),
                                    mat * sizeof(float), (off0 + a * mat) * sizeof(float));
      us.push_back(muon_momentum_step(gctx_, t, m_stack[gi], {mom_pars, nest_pars, ns_pars}));
      std::vector<ggml_tensor *> ps;
      for (size_t i = a; i < b; i++) ps.push_back(params[idx[i]]);
      steps.emplace_back(ps, lr_wd);
    }
    for (size_t s = 0; s < shards_.size(); s++)
      for (size_t i = owned[gi][s + 1].first; i < owned[gi][s + 1].second; i++) {
        ggml_tensor *p = params[idx[i]];
        ggml_tensor *ui = ggml_view_2d(gctx_, shards_[s].upd_in, p->ne[0], p->ne[1], p->nb[1],
                                       (off0 + i * mat - shards_[s].lo) * sizeof(float));
        ggml_build_forward_expand(apply_graph_, ggml_opt_step_sgd(gctx_, p, ui, lr_wd));
      }
    gi++;
  }
  const std::vector<ggml_tensor *> os = orthogonalize_all(gctx_, backend, us, ns_steps, ns_pars);
  for (size_t j = 0; j < os.size(); j++) {
    ggml_tensor *o = os[j];
    for (size_t i = 0; i < steps[j].first.size(); i++) {
      ggml_tensor *p = steps[j].first[i];
      ggml_tensor *oi = ggml_view_4d(gctx_, o, p->ne[0], p->ne[1], 1, 1, o->nb[1], o->nb[2], o->nb[3], i * o->nb[2]);
      ggml_build_forward_expand(graph_, ggml_opt_step_sgd(gctx_, p, oi, steps[j].second));
    }
  }
  for (auto &[shape, group] : muon_groups) {
    this->muon_groups.push_back(fmt::format("{}x{}x{}", shape.first, shape.second, group.size()));
    ggml_tensor *u = group[0].second;
    for (size_t i = 1; i < group.size(); i++) u = ggml_concat(gctx_, u, group[i].second, 2);
    ggml_tensor *o = orthogonalize(gctx_, ggml_cont(gctx_, u), ns_steps);
    // RMS of o ~ 1/sqrt(max dim); 0.2 sqrt(max dim) matches Adam's update RMS
    const float rms = 0.2f * std::sqrt(static_cast<float>(std::max(shape.first, shape.second)));
    for (size_t i = 0; i < group.size(); i++) {
      ggml_tensor *p = group[i].first;
      ggml_tensor *oi = ggml_view_2d(gctx_, o, p->ne[0], p->ne[1], o->nb[1], i * o->nb[2]);
      ggml_tensor *step = ggml_mul(gctx_, ggml_scale(gctx_, oi, rms), muon_lr_);
      ggml_build_forward_expand(graph_, ggml_cpy(gctx_, ggml_sub(gctx_, p, step), p));
    }
  }
  galloc_ = ggml_gallocr_new(buft);
  if (!ggml_gallocr_alloc_graph(galloc_, graph_)) throw std::runtime_error("cannot allocate the optimizer graph");
  // the step graphs are the same every step: CUDA graph capture for them
  // alone (every other graph runs uncaptured: dynamic shapes)
  graph_captured_ = allow_cuda_graph(backend, graph_);
  for (auto [gr, ga] : {std::make_pair(norm_graph_, &norm_galloc_), std::make_pair(apply_graph_, &apply_galloc_)})
    if (gr) {
      *ga = ggml_gallocr_new(buft);
      if (!ggml_gallocr_alloc_graph(*ga, gr)) throw std::runtime_error("cannot allocate the optimizer graph");
      allow_cuda_graph(backend, gr);
    }

  // the other devices: per shard, persistent state on device d (staging for
  // the other workers' gradients over [lo, hi), clip factor, momentum,
  // constants, the update) and its graph: mean, clip, Muon, update -> upd
  for (size_t s = 0; s < shards_.size(); s++) {
    Shard &sh = shards_[s];
    const int d = static_cast<int>(s) + 1;
    Worker &wd = *workers_[d];
    ggml_backend_t bd = wd.model().backend();
    ggml_backend_buffer_type_t bufd = ggml_backend_get_default_buffer_type(bd);
    const int64_t n = static_cast<int64_t>(sh.hi - sh.lo);
    sh.ctx = ggml_init({ggml_tensor_overhead() * (W + groups.size() + 8), nullptr, true});
    sh.vctx = ggml_init({ggml_tensor_overhead() * W, nullptr, true});
    for (int w = 0; w < W; w++)
      if (w != d) {
        sh.staging.push_back(ggml_new_tensor_1d(sh.ctx, GGML_TYPE_F32, n));
        sh.src.push_back(ggml_view_1d(sh.vctx, workers_[w]->grad_flat(), n, sh.lo * sizeof(float)));
        if (ggml_backend_view_init(sh.src.back()) != GGML_STATUS_SUCCESS)
          throw std::runtime_error("cannot view the gradient accumulator");
      }
    sh.clip = ggml_new_tensor_1d(sh.ctx, GGML_TYPE_F32, 1);
    sh.upd = ggml_new_tensor_1d(sh.ctx, GGML_TYPE_F32, n);
    std::vector<ggml_tensor *> md = new_m_stack(sh.ctx, d);
    const auto cd = muon_constants(sh.ctx);
    sh.buf = ggml_backend_alloc_ctx_tensors_from_buft(sh.ctx, bufd);
    if (!sh.buf) throw std::runtime_error("cannot allocate the optimizer state");
    ggml_backend_buffer_clear(sh.buf, 0);
    set_muon_constants(cd);

    const size_t nd = (32 + 16 * ns_steps) * groups.size() + 64;
    sh.gctx = ggml_init({ggml_tensor_overhead() * nd + ggml_graph_overhead_custom(nd, false), nullptr, true});
    ggml_tensor *gd = ggml_view_1d(sh.gctx, wd.grad_flat(), n, sh.lo * sizeof(float));
    for (ggml_tensor *st : sh.staging) gd = ggml_add(sh.gctx, gd, st);
    gd = ggml_mul(sh.gctx, ggml_scale(sh.gctx, gd, 1.0f / W), sh.clip);
    sh.graph = ggml_new_graph_custom(sh.gctx, nd, false);
    size_t gj = 0;
    std::vector<ggml_tensor *> ud, dst; // Nesterov stacks, their places in upd
    for (auto &[shape, idx] : groups) {
      const auto [a, b] = owned[gj][d];
      if (a < b) {
        const size_t mat = static_cast<size_t>(shape.first * shape.second),
                     off = (group_offset(idx) + a * mat - sh.lo) * sizeof(float);
        ggml_tensor *t = ggml_view_3d(sh.gctx, gd, shape.first, shape.second, b - a, shape.first * sizeof(float),
                                      mat * sizeof(float), off);
        ud.push_back(muon_momentum_step(sh.gctx, t, md[gj], cd));
        dst.push_back(ggml_view_3d(sh.gctx, sh.upd, shape.first, shape.second, b - a, shape.first * sizeof(float),
                                   mat * sizeof(float), off));
      }
      gj++;
    }
    const std::vector<ggml_tensor *> od = orthogonalize_all(sh.gctx, bd, ud, ns_steps, cd[2]);
    for (size_t j = 0; j < od.size(); j++) ggml_build_forward_expand(sh.graph, ggml_cpy(sh.gctx, od[j], dst[j]));
    sh.galloc = ggml_gallocr_new(bufd);
    if (!ggml_gallocr_alloc_graph(sh.galloc, sh.graph)) throw std::runtime_error("cannot allocate the optimizer graph");
    allow_cuda_graph(bd, sh.graph);
  }
}

Optimizer::~Optimizer() {
  for (Shard &sh : shards_) {
    if (sh.galloc) ggml_gallocr_free(sh.galloc);
    if (sh.buf) ggml_backend_buffer_free(sh.buf);
    for (ggml_context *c : {sh.gctx, sh.vctx, sh.ctx})
      if (c) ggml_free(c);
  }
  for (ggml_gallocr_t ga : {galloc_, norm_galloc_, apply_galloc_})
    if (ga) ggml_gallocr_free(ga);
  if (buf_) ggml_backend_buffer_free(buf_);
  if (gctx_) ggml_free(gctx_);
  if (ctx_) ggml_free(ctx_);
}

double Optimizer::step_allreduce(double lr) {
#ifdef MLIPCPP_NCCL
  const size_t W = workers_.size();
  for (Worker *w : workers_) ggml_backend_synchronize(w->model().backend()); // the gradients are complete
  nccl_check(ncclGroupStart(), "ncclGroupStart");
  for (size_t w = 0; w < W; w++) {
    ggml_tensor *g = workers_[w]->grad_flat();
    nccl_check(ncclAllReduce(g->data, g->data, ggml_nelements(g), ncclFloat, ncclAvg, nccl_->comms[w], nccl_->streams[w]),
               "ncclAllReduce");
  }
  nccl_check(ncclGroupEnd(), "ncclGroupEnd");
  for (size_t w = 0; w < W; w++) {
    cuda_check(cudaSetDevice(static_cast<int>(w)), "cudaSetDevice");
    cuda_check(cudaStreamSynchronize(nccl_->streams[w]), "ncclAllReduce");
  }
  // the replicas step in parallel, each on its own device
  std::vector<double> norms(W);
  std::vector<std::thread> pool;
  std::vector<std::exception_ptr> errors(W);
  for (size_t w = 0; w < W; w++)
    pool.emplace_back([&, w] {
      try {
        norms[w] = replicas_[w]->step(lr);
      } catch (...) {
        errors[w] = std::current_exception();
      }
    });
  for (auto &t : pool) t.join();
  for (auto &e : errors)
    if (e) std::rethrow_exception(e);
  // the same averaged gradient gives the same step everywhere; resync anyway
  if (++t_ % resync_every == 0) {
    const std::vector<float> p0 = workers_[0]->get_params();
    last_drift = 0;
    for (size_t w = 1; w < W; w++) {
      const std::vector<float> pw = workers_[w]->get_params();
      for (size_t i = 0; i < p0.size(); i++) last_drift = std::max(last_drift, double(std::abs(pw[i] - p0[i])));
      ggml_backend_tensor_copy(workers_[0]->weights_blob(), workers_[w]->weights_blob());
    }
  }
  return norms[0];
#else
  (void) lr;
  throw std::logic_error("step_allreduce without NCCL");
#endif
}

double Optimizer::step(double lr) {
  if (!replicas_.empty()) return step_allreduce(lr);
  Worker &w0 = *workers_[0];
  ++t_;
  const double b1 = 0.9, b2 = 0.999;
  const float pars[7] = {static_cast<float>(lr), static_cast<float>(b1), static_cast<float>(b2), 1e-8f, 0.0f,
                         static_cast<float>(1.0 / (1.0 - std::pow(b1, t_))),
                         static_cast<float>(1.0 / (1.0 - std::pow(b2, t_)))};
  ggml_backend_tensor_set(pars_, pars, 0, sizeof pars);
  const float mlr = static_cast<float>(lr * muon_lr_scale_);
  ggml_backend_tensor_set(muon_lr_, &mlr, 0, sizeof mlr);
  if (!muon_rms_.empty()) {
    std::vector<float> mp(2 * muon_rms_.size(), 0.0f); // (lr, wd = 0) per group
    for (size_t i = 0; i < muon_rms_.size(); i++) mp[2 * i] = mlr * muon_rms_[i];
    ggml_backend_tensor_set(muon_pars_, mp.data(), 0, mp.size() * sizeof(float));
  }
  for (size_t w = 0; w < workers_.size(); w++) stage(w);
  ggml_backend_t b0 = w0.model().backend();
  if (shards_.empty()) {
    if (ggml_backend_graph_compute(b0, graph_) != GGML_STATUS_SUCCESS) throw std::runtime_error("optimizer step failed");
  } else {
    // all queued on the device streams, the copies ordered by events: norm
    // and clip on device 0; the clip factor to the other devices, their
    // Muon; device 0's step; their updates back, applied on device 0
    auto run = [](ggml_backend_t b, ggml_cgraph *gr) {
      if (ggml_backend_graph_compute_async(b, gr) != GGML_STATUS_SUCCESS) throw std::runtime_error("optimizer step failed");
    };
    run(b0, norm_graph_);
    for (size_t s = 0; s < shards_.size(); s++) {
      ggml_backend_t bd = workers_[s + 1]->model().backend();
      ggml_backend_tensor_copy_async(b0, bd, clip_, shards_[s].clip);
      run(bd, shards_[s].graph);
    }
    run(b0, graph_);
    for (size_t s = 0; s < shards_.size(); s++)
      ggml_backend_tensor_copy_async(workers_[s + 1]->model().backend(), b0, shards_[s].upd, shards_[s].upd_in);
    run(b0, apply_graph_);
    for (auto &w : workers_) ggml_backend_synchronize(w->model().backend());
  }
  float norm = 0;
  ggml_backend_tensor_get(norm_, &norm, 0, sizeof norm);
  for (size_t w = 1; w < workers_.size(); w++) {
    if (blob_broadcast_) {
      ggml_backend_tensor_copy(w0.weights_blob(), workers_[w]->weights_blob());
    } else {
      for (size_t k = 0; k < w0.params().size(); k++)
        ggml_backend_tensor_copy(w0.params()[k], workers_[w]->params()[k]);
    }
  }
  for (auto &w : workers_) w->zero_grads();
  std::fill(staged_.begin(), staged_.end(), 0);
  return norm;
}

double Optimizer::profile(int reps, std::map<std::string, Worker::OpStats> &ops) {
  ggml_backend_t backend = workers_[0]->model().backend();
  step(1e-6); // warm-up
  ggml_backend_synchronize(backend);
  auto t0 = clock_type::now();
  for (int r = 0; r < reps; r++) step(1e-6);
  ggml_backend_synchronize(backend);
  const double ms = 1e3 * seconds_since(t0) / reps;
  for (int i = 0; i < ggml_graph_n_nodes(graph_); i++) {
    ggml_cgraph one = ggml_graph_view(graph_, i, i + 1);
    auto tn = clock_type::now();
    if (ggml_backend_graph_compute(backend, &one) != GGML_STATUS_SUCCESS)
      throw std::runtime_error("optimizer graph compute failed");
    ggml_backend_synchronize(backend);
    const ggml_tensor *node = ggml_graph_node(graph_, i);
    auto &s = ops[ggml_op_desc(node)];
    auto [f, b] = node_cost(node);
    s.ms += 1e3 * seconds_since(tn), s.count++, s.flops += f, s.bytes += b;
  }
  return ms;
}

void Optimizer::stage(size_t w) {
  if (!replicas_.empty() || staged_[w]) return; // all-reduce: nothing to stage
  if (w > 0) ggml_backend_tensor_copy(workers_[w]->grad_flat(), staging_[w - 1]); // peer copy
  for (size_t d = 1; d <= shards_.size(); d++)
    if (d != w) {
      const size_t k = w < d ? w : w - 1; // w's slot among d's other workers
      ggml_backend_tensor_copy(shards_[d - 1].src[k], shards_[d - 1].staging[k]);
    }
  staged_[w] = 1;
}

std::vector<uint8_t> Optimizer::get_state() const {
  // the step count, then every tensor of the state buffer (moments, momenta;
  // the staging and constants come along, they are overwritten anyway)
  if (!replicas_.empty()) {
    std::vector<uint8_t> s = replicas_[0]->get_state(); // the replicas step identically
    std::memcpy(s.data() + sizeof(int64_t), &t_, sizeof t_);
    return s;
  }
  if (!shards_.empty()) throw std::runtime_error("optimizer state: not with the Muon split over devices");
  std::vector<uint8_t> s(2 * sizeof(int64_t));
  std::memcpy(s.data(), &t_, sizeof t_);
  for (ggml_tensor *t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) {
    const size_t at = s.size();
    s.resize(at + ggml_nbytes(t));
    ggml_backend_tensor_get(t, s.data() + at, 0, ggml_nbytes(t));
  }
  return s;
}

void Optimizer::set_state(const std::vector<uint8_t> &s) {
  if (s.size() < 2 * sizeof(int64_t)) throw std::runtime_error("optimizer state: truncated");
  if (!replicas_.empty()) {
    for (auto &r : replicas_) r->set_state(s);
    std::memcpy(&t_, s.data() + sizeof(int64_t), sizeof t_);
    return;
  }
  if (!shards_.empty()) throw std::runtime_error("optimizer state: not with the Muon split over devices");
  size_t at = 2 * sizeof(int64_t);
  for (ggml_tensor *t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) at += ggml_nbytes(t);
  if (at != s.size()) throw std::runtime_error("optimizer state: size mismatch (another model or optimizer?)");
  std::memcpy(&t_, s.data(), sizeof t_);
  at = 2 * sizeof(int64_t);
  for (ggml_tensor *t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) {
    ggml_backend_tensor_set(t, s.data() + at, 0, ggml_nbytes(t));
    at += ggml_nbytes(t);
  }
}

namespace {
/// --lr-schedule: the groups' optimizer (GroupOptimizer) on device 0, the workers' gradients
/// averaged there first (peer copies), the weights copied back to the other workers after
struct GroupScheduled {
  std::vector<Worker *> workers;
  std::unique_ptr<GroupOptimizer> opt;
  ggml_context *vctx = nullptr, *sctx = nullptr, *gctx = nullptr;
  ggml_backend_buffer_t sbuf = nullptr;
  std::vector<ggml_tensor *> staging;
  ggml_cgraph *reduce = nullptr;
  ggml_gallocr_t galloc = nullptr;
  bool unmapped = false; // a GGUF without train.state_dict_names: GGUF names matched

  GroupScheduled(std::vector<std::unique_ptr<Worker>> &ws, LrSchedule sched, double clip, double mu, double scale,
                 int ns_steps) {
    for (auto &w : ws) workers.push_back(w.get());
    Worker &w0 = *workers[0];
    const auto &names = w0.model().state_dict_names();
    unmapped = names.empty();
    vctx = ggml_init({ggml_tensor_overhead() * (w0.params().size() + 4), nullptr, true});
    std::vector<GroupOptimizer::Param> ps;
    for (size_t k = 0; k < w0.params().size(); k++) {
      ggml_tensor *p = w0.params()[k];
      ggml_tensor *g = ggml_view_4d(vctx, w0.grad_flat(), p->ne[0], p->ne[1], p->ne[2], p->ne[3], p->nb[1], p->nb[2],
                                    p->nb[3], w0.grad_offset(k) * sizeof(float));
      if (ggml_backend_view_init(g) != GGML_STATUS_SUCCESS) throw std::runtime_error("cannot view the gradients");
      const std::string gguf = ggml_get_name(p);
      auto it = names.find(gguf);
      ps.push_back({p, g, it == names.end() ? gguf : it->second, is_muon_param(p)});
    }
    opt = std::make_unique<GroupOptimizer>(
        w0.model().backend(), std::move(ps), std::move(sched), clip, mu, scale,
        [ns_steps](ggml_context *ctx, ggml_tensor *g) { return orthogonalize(ctx, g, ns_steps); });
    if (workers.size() > 1) { // mean of the workers' gradients into worker 0's accumulator
      sctx = ggml_init({ggml_tensor_overhead() * workers.size(), nullptr, true});
      for (size_t w = 1; w < workers.size(); w++) staging.push_back(ggml_dup_tensor(sctx, w0.grad_flat()));
      sbuf = ggml_backend_alloc_ctx_tensors(sctx, w0.model().backend());
      const size_t nn = 4 * workers.size() + 16;
      gctx = ggml_init({ggml_tensor_overhead() * nn + ggml_graph_overhead_custom(nn, false), nullptr, true});
      ggml_tensor *s = w0.grad_flat();
      for (ggml_tensor *t : staging) s = ggml_add(gctx, s, t);
      reduce = ggml_new_graph_custom(gctx, nn, false);
      ggml_build_forward_expand(reduce, ggml_cpy(gctx, ggml_scale(gctx, s, 1.0f / float(workers.size())), w0.grad_flat()));
      galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w0.model().backend()));
      if (!ggml_gallocr_alloc_graph(galloc, reduce)) throw std::runtime_error("cannot allocate the gradient mean");
    }
  }
  ~GroupScheduled() {
    opt.reset();
    if (galloc) ggml_gallocr_free(galloc);
    if (gctx) ggml_free(gctx);
    if (sbuf) ggml_backend_buffer_free(sbuf);
    if (sctx) ggml_free(sctx);
    if (vctx) ggml_free(vctx);
  }
  double step(int64_t n, int64_t S, int64_t T) {
    Worker &w0 = *workers[0];
    if (reduce) {
      for (size_t w = 1; w < workers.size(); w++) ggml_backend_tensor_copy(workers[w]->grad_flat(), staging[w - 1]);
      if (ggml_backend_graph_compute(w0.model().backend(), reduce) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("gradient mean failed");
    }
    const double norm = opt->step(n, S, T);
    for (size_t w = 1; w < workers.size(); w++)
      for (size_t k = 0; k < w0.params().size(); k++) ggml_backend_tensor_copy(w0.params()[k], workers[w]->params()[k]);
    for (Worker *w : workers) w->zero_grads();
    return norm;
  }
  std::string summary(int64_t S, int64_t T) const {
    const LrSchedule &s = opt->schedule();
    const auto sizes = opt->group_sizes();
    std::vector<int> tensors(s.groups.size(), 0);
    for (int g : opt->group_of_param()) tensors[g]++;
    std::string out = fmt::format("base {:g}, min {:g}, {} after {} epochs, warmup {} epochs, {} steps/epoch, {} steps{}",
                                  s.base, s.min, s.shape, s.decay_after_epochs, s.warmup_epochs, S, T,
                                  unmapped ? " (GGUF without train.state_dict_names: GGUF names matched)" : "");
    for (size_t g = 0; g < s.groups.size(); g++) {
      const LrGroup &x = s.groups[g];
      out += fmt::format("\n  group {}: {} tensors, {} values; lr x{:g}, frozen {} epochs, warmup {} epochs, "
                         "weight decay {:g}, {}",
                         x.name, tensors[g], sizes[g], x.lr_mult, x.freeze_epochs, x.warmup_epochs, x.weight_decay,
                         x.muon ? "Muon (matrices) + AdamW" : "AdamW");
    }
    return out;
  }
  std::string lr_json(int64_t n, int64_t S, int64_t T) const {
    const LrSchedule &s = opt->schedule();
    std::string out = "{";
    for (size_t g = 0; g < s.groups.size(); g++)
      out += fmt::format("{}\"{}\":{}", g ? "," : "", s.groups[g].name, s.group_lr(int(g), n, S, T).first);
    return out + "}";
  }
};
} // namespace

void train(const TrainOptions &o) {
  auto t_setup = clock_type::now();
  MemmapSplit train_split(o.data_dir + "/train"), val_split(o.data_dir + "/val"),
      test_split(o.data_dir + "/test");
  fmt::print("data: {} train / {} val / {} test structures\n", train_split.size(),
             val_split.size(), test_split.size());

  // CUDA graphs pay off only for graphs that are computed again (tiles);
  // with dynamic shapes every step would capture into a new cache entry
  if (!(o.tile_nb_ratio > 1)) setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
  std::vector<std::unique_ptr<Worker>> workers;
  for (int w = 0; w < std::max(1, o.n_gpus); w++)
    workers.push_back(std::make_unique<Worker>(
        o.init_gguf, o.n_gpus == 0 ? -1 : w,
        o.tile_nb_ratio > 1 ? static_cast<size_t>(o.graph_cache) : 0));
  for (auto &w : workers) w->conservative = {o.conservative, o.energy_weight, o.force_weight, o.nc_force_weight};
  for (auto &w : workers) w->charge_corr = {o.charge_corr_weight, o.charge_corr_min};
  if (o.loss != "mse" && o.loss != "huber") throw std::runtime_error("--loss mse|huber, not " + o.loss);
  for (auto &w : workers) w->loss.huber = o.loss == "huber", w->loss.delta_e = o.huber_delta_e, w->loss.delta_f = o.huber_delta_f;
  if (o.loss == "huber")
    fmt::print("loss: Huber, delta {} (energy per atom / energy scale), {} (forces / their scale)\n", o.huber_delta_e, o.huber_delta_f);
  if (o.charge_corr_weight > 0) {
    if (o.charge_corr_target != "mulliken" && o.charge_corr_target != "lowdin" && o.charge_corr_target != "nbo")
      throw std::runtime_error("--charge-corr-target mulliken|lowdin|nbo, not " + o.charge_corr_target);
    if (!workers[0]->model().plato_lr().les)
      throw std::runtime_error("--charge-corr-weight needs a Plato-PET-LR model with LES latent charges");
    for (MemmapSplit *sp : {&train_split, &val_split, &test_split}) {
      const bool got = sp->load_partial_charges(o.charge_corr_target);
      int64_t ok = 0;
      for (int64_t s = 0; s < sp->size(); s++) ok += sp->partial_charges_ok(s);
      fmt::print("charge correlation: {}: {}_charge.bin {}, {} of {} structures taking part\n", sp->dir(),
                 o.charge_corr_target, got ? "read" : "absent", ok, sp->size());
    }
    fmt::print("charge correlation: weight {} x mean over structures of max(0, {} - r), r = Pearson(first latent "
               "charge channel, {} charges)\n", o.charge_corr_weight, o.charge_corr_min, o.charge_corr_target);
  }
  if (o.conservative)
    fmt::print("conservative forces -dE/dx: loss weights energy {}, forces {}, direct forces {}\n", o.energy_weight,
               o.force_weight, o.nc_force_weight);
  Worker &w0 = *workers[0];
  const float cutoff = w0.model().cutoff();
  const auto &species = w0.model().species_to_index();
  fmt::print("model: {} parameters in {} tensors, cutoff {} A, {} worker(s) on {}\n",
             w0.n_params(), w0.params().size(), cutoff, workers.size(),
             ggml_backend_name(w0.model().backend()));
  // charge / spin conditioning: every structure's values in the model's range
  // (an error now rather than mid-run), or a warning when the data has them
  // and the model does not read them
  if (const Conditioning &cond = w0.model().conditioning(); cond.enabled) {
    for (const MemmapSplit *s : {&train_split, &val_split, &test_split}) {
      int qmin = 0, qmax = 0, smin = 1, smax = 1;
      for (int64_t k = 0; k < s->size(); k++) {
        cond.check(s->charge(k), s->spin_multiplicity(k));
        qmin = std::min(qmin, s->charge(k)), qmax = std::max(qmax, s->charge(k));
        smin = std::min(smin, s->spin_multiplicity(k)), smax = std::max(smax, s->spin_multiplicity(k));
      }
      fmt::print("conditioning ({}): charge in [{}, {}], spin multiplicity in [{}, {}]{}\n", s->dir(), qmin, qmax,
                 smin, smax, s->has_conditioning_data() ? "" : " (no charge.bin / spin_multiplicity.bin)");
    }
  } else if (train_split.has_conditioning_data()) {
    fmt::print("warning: the data has charge / spin multiplicity, the model is not conditioned on them\n");
  }

  // --epochs 0 evaluates a trained model with the baselines stored in it
  const bool eval_only = o.epochs == 0;
  auto t0 = clock_type::now();
  TargetScaling scaling;
  if (eval_only || o.model_baselines) {
    for (auto [Z, c] : w0.model().composition_energies()) scaling.composition[Z] = c;
    for (auto [Z, f] : w0.model().nc_force_scales()) scaling.force_scale[Z] = f;
    scaling.energy_scale = w0.model().energy_scale();
  } else {
    scaling = TargetScaling::fit(train_split);
  }
  fmt::print("targets ({}): composition for {} species, energy scale {:.6g} eV/atom ({:.1f} s)\n",
             eval_only || o.model_baselines ? "from the model" : "fitted", scaling.composition.size(),
             scaling.energy_scale, seconds_since(t0));

  t0 = clock_type::now();
  const AdaptiveCutoff adaptive = adaptive_cutoff(w0.model());
  if (adaptive.enabled())
    fmt::print("adaptive cutoff: {} neighbours ({}, taper {} A) within {} A\n", adaptive.n_target,
               adaptive.grid ? "grid" : "solver", adaptive.width, cutoff);
  auto nl = [&](const MemmapSplit &s) {
    NeighborCache c = NeighborCache::load_or_build(s, cutoff, cache_path(o.cache_dir, s, cutoff, adaptive),
                                                   o.n_threads, adaptive);
    c.taper = o.conservative && adaptive.enabled(); // the radii's share of -dE/dx
    return c;
  };
  const NeighborCache nl_train = eval_only ? NeighborCache{} : nl(train_split),
                      nl_val = nl(val_split), nl_test = nl(test_split);
  fmt::print("neighbour lists ready ({:.1f} s)\n", seconds_since(t0));
  // structures used: all, or those within --max-neighbors and --max-attention
  // (a structure alone is one micro-batch: its memory ~ n (m+1)^2)
  // and not listed in --exclude DIR/<split>.npy
  auto kept = [&](const char *name, const MemmapSplit &s, const NeighborCache &c) {
    std::vector<char> excluded(c.max_neighbors.size(), 0);
    int64_t n_excluded = 0;
    if (!o.exclude.empty()) {
      for (int64_t i : read_npy_i64(fmt::format("{}/{}.npy", o.exclude, name)))
        if (i >= 0 && i < static_cast<int64_t>(excluded.size()) && !excluded[i]) excluded[i] = 1, n_excluded++;
      fmt::print("--exclude {}: {} of {} {} structures excluded\n", o.exclude, n_excluded, s.size(), name);
    }
    std::vector<int64_t> ids;
    for (int64_t i = 0; i < static_cast<int64_t>(c.max_neighbors.size()); i++) {
      const double m1 = c.max_neighbors[i] + 1.0;
      if (!excluded[i] && (o.max_neighbors <= 0 || c.max_neighbors[i] <= o.max_neighbors) &&
          (o.max_attention <= 0 || s.n_atoms(i) * m1 * m1 <= o.max_attention))
        ids.push_back(i);
    }
    if (o.max_neighbors > 0 || o.max_attention > 0)
      fmt::print("--max-neighbors {} --max-attention {:g}: {} of {} {} structures dropped\n", o.max_neighbors,
                 o.max_attention, s.size() - n_excluded - static_cast<int64_t>(ids.size()), s.size(), name);
    return ids;
  };
  const std::vector<int64_t> ids_train = eval_only ? std::vector<int64_t>{} : kept("train", train_split, nl_train),
                             ids_val = kept("val", val_split, nl_val), ids_test = kept("test", test_split, nl_test);

  const int W = static_cast<int>(workers.size()), B = o.batch_size;
  const Tiling tiling{o.tile_nb_ratio, o.tile_atom_ratio, B};
  const int64_t steps_per_epoch = static_cast<int64_t>(ids_train.size()) / (int64_t(B) * W); // drop_last
  const int64_t total_steps = o.epochs * steps_per_epoch;
  const int64_t warmup = static_cast<int64_t>(o.warmup_fraction * total_steps);
  if (o.schedule != "cosine" && o.schedule != "linear") throw std::runtime_error("--schedule must be cosine or linear");
  const bool linear = o.schedule == "linear";
  auto lr_at = [&](int64_t step) {
    return o.lr * (linear ? lr_factor_linear(step, total_steps, int64_t(o.decay_after) * steps_per_epoch, o.min_lr / o.lr)
                          : lr_factor(step, total_steps, warmup));
  };

  // --lr-schedule: per-group learning rates, the groups' optimizers (GroupScheduled)
  std::unique_ptr<GroupScheduled> grouped;
  if (!o.lr_schedule.empty()) {
    LrSchedule sched = LrSchedule::load(o.lr_schedule);
    if (o.dump_lr_schedule > 0) {
      fmt::print("{}\n", sched.table_json(o.dump_lr_schedule, steps_per_epoch, total_steps));
      return;
    }
    grouped = std::make_unique<GroupScheduled>(workers, std::move(sched), o.grad_clip, o.muon_momentum,
                                               o.muon_lr_scale, o.muon_ns_steps);
  } else if (o.dump_lr_schedule > 0) {
    throw std::runtime_error("--dump-lr-schedule needs --lr-schedule");
  }

  // identical parameters on every worker, then the device-side optimizer
  std::vector<float> params = w0.get_params();
  for (size_t w = 1; w < workers.size(); w++) workers[w]->set_params(params);
  if (o.optimizer != "adam" && o.optimizer != "muon")
    throw std::runtime_error("--optimizer must be adam or muon");
  // without --lr-schedule: exactly the optimizer as before
  std::unique_ptr<Optimizer> optimizer_owner =
      grouped ? nullptr
              : std::make_unique<Optimizer>(workers, o.grad_clip, o.optimizer == "muon", o.muon_momentum,
                                            o.muon_lr_scale, o.muon_ns_steps, o.allreduce);
  struct OptimizerFacade { // the calls the loop makes, to either optimizer
    Optimizer *plain;
    GroupScheduled *grouped;
    std::string allreduce_info, muon_split;
    size_t n_muon_tensors = 0;
    void stage(size_t w) {
      if (plain) plain->stage(w);
    }
    std::vector<uint8_t> get_state() const { return plain ? plain->get_state() : grouped->opt->get_state(); }
    void set_state(const std::vector<uint8_t> &s) { plain ? plain->set_state(s) : grouped->opt->set_state(s); }
  } optimizer{optimizer_owner.get(), grouped.get(), optimizer_owner ? optimizer_owner->allreduce_info : "",
              optimizer_owner ? optimizer_owner->muon_split : "", optimizer_owner ? optimizer_owner->n_muon_tensors : 0};
  if (!optimizer.allreduce_info.empty()) fmt::print("sync: {}\n", optimizer.allreduce_info);
  if (grouped) fmt::print("lr schedule {}: {}\n", o.lr_schedule, grouped->summary(steps_per_epoch, total_steps));
  fmt::print("optimizer: {}{}; {}\n", grouped ? std::string("per group (--lr-schedule)") : o.optimizer,
             !grouped && o.optimizer == "muon" ? fmt::format(" ({} tensors, momentum {}, lr scale {}, {} NS steps)",
                                                 optimizer.n_muon_tensors, o.muon_momentum,
                                                 o.muon_lr_scale, o.muon_ns_steps)
                                   : "",
             tiling.enabled() ? fmt::format("tiles: neighbour ratio {}, atom ratio {}, {} systems, "
                                            "graph cache {} shapes/device",
                                            o.tile_nb_ratio, o.tile_atom_ratio, B, o.graph_cache)
                              : std::string("dynamic shapes"));
  fmt::print("setup: {:.1f} s; {} steps/epoch of {} x {} structures, {}\n", seconds_since(t_setup), steps_per_epoch, W, B,
             linear ? fmt::format("lr {:g} for {} epochs, then linear to {:g}", o.lr, o.decay_after, o.min_lr)
                    : fmt::format("warmup {} steps, then cosine", warmup));
  TrainState state;
  if (!o.resume.empty()) {
    state = read_state(o.resume);
    optimizer.set_state(state.optimizer);
    fmt::print("resumed from {}: {} epochs done, step {}, best R2(E) + R2(F) {:.6g}%\n", o.resume, state.epoch,
               state.step, state.best);
  }
  if (!optimizer.muon_split.empty()) fmt::print("muon split: {}\n", optimizer.muon_split);

  // per-structure records of the workers (--dump-errors / --log-errors)
  std::vector<std::vector<SystemError>> errors(W);
  std::vector<std::vector<float>> atom_errors(W);
  auto collect_errors = [&](bool on, bool atoms = false) {
    for (int w = 0; w < W; w++) {
      workers[w]->system_errors = on ? &errors[w] : nullptr;
      workers[w]->atom_errors = on && atoms ? &atom_errors[w] : nullptr;
    }
  };
  auto drain_errors = [&](std::FILE *f, int32_t step) {
    for (auto &e : errors) {
      for (SystemError &r : e) r.step = step;
      if (!e.empty() && std::fwrite(e.data(), sizeof(SystemError), e.size(), f) != e.size())
        throw std::runtime_error("cannot write the per-structure errors");
      e.clear();
    }
  };
  auto evaluate = [&](const MemmapSplit &split, const NeighborCache &cache, const std::vector<int64_t> &use) {
    std::vector<StepStats> per(W);
    const int64_t n_use = static_cast<int64_t>(use.size()), n_batches = (n_use + B - 1) / B;
    for_workers(workers, [&](size_t w) {
      for (int64_t k = w; k < n_batches; k += W) {
        std::vector<int64_t> ids(use.begin() + k * B, use.begin() + std::min<int64_t>(n_use, (k + 1) * B));
        for (const Batch &b : make_microbatches(split, cache, ids, {}, species, scaling, cutoff,
                                                    o.mb_budget, o.mb_growth, tiling))
          per[w] += workers[w]->run(b, scaling.energy_scale, false);
      }
    });
    StepStats total;
    for (auto &p : per) total += p;
    return total;
  };
  auto rmse = [](const StepStats &s) {
    return std::pair{1e3 * std::sqrt(s.se_energy_per_atom / s.n_systems),
                     1e3 * std::sqrt(s.se_force / s.n_force_components)};
  };
  // MAE (conservative mode) and R^2 in % (energies per atom after the composition, force components pooled)
  auto mae = [](const StepStats &s) {
    return std::pair{1e3 * s.ae_energy_per_atom / s.n_systems, 1e3 * s.ae_force / s.n_force_components};
  };
  auto r2 = [](const StepStats &s) {
    auto one = [](double sse, double sum, double sum2, int64_t n) { return 100 * (1 - sse / (sum2 - sum * sum / n)); };
    return std::pair{one(s.se_energy_per_atom, s.sum_e, s.sum2_e, s.n_systems),
                     one(s.se_force, s.sum_f, s.sum2_f, s.n_force_components)};
  };
  std::unique_ptr<std::FILE, int (*)(std::FILE *)> jsonl(nullptr, std::fclose);
  if (!o.log_jsonl.empty() && !eval_only) {
    jsonl.reset(std::fopen(o.log_jsonl.c_str(), "a"));
    if (!jsonl) throw std::runtime_error("cannot open " + o.log_jsonl);
  }
  // one line of val metrics (and the `extra` fields) per epoch
  auto log_line = [&](int64_t epoch, const StepStats &val, const std::string &extra) {
    if (!jsonl) return;
    auto [ve, vf] = rmse(val);
    auto [ae, af] = mae(val);
    auto [re, rf] = r2(val);
    auto num = [](double x) { return std::isfinite(x) ? fmt::format("{}", x) : std::string("null"); };
    fmt::print(jsonl.get(),
               "{{\"epoch\":{}{},\"val_e_mae_mev_atom\":{},\"val_f_mae_mev_a\":{},\"val_e_rmse_mev_atom\":{},"
               "\"val_f_rmse_mev_a\":{},\"val_r2_e\":{},\"val_r2_f\":{}}}\n",
               epoch, extra, o.conservative ? num(ae) : "null", o.conservative ? num(af) : "null", num(ve), num(vf),
               num(re), num(rf));
    std::fflush(jsonl.get());
  };
  // conservative mode: the direct head's force RMSE too
  auto nc_rmse = [&](const StepStats &s) {
    return o.conservative ? fmt::format(", F_nc {:.2f} meV/A", 1e3 * std::sqrt(s.se_force_nc / s.n_force_components))
                          : std::string();
  };

  // step packing: the cost model is refitted to measured micro-batch times
  // after every step and shared with the packing threads
  const bool step_pack = o.pack == "step";
  if (!step_pack && o.pack != "worker") throw std::runtime_error("--pack must be step or worker");
  const Tiling step_tiling{o.tile_nb_ratio, o.tile_atom_ratio, W * B};
  std::mutex plan_mu;
  // the conservative step (two graphs, forces assembled on the CPU) costs
  // ~5x as much per micro-batch and ~15x per n(m+1) (176059, wall-clock)
  const CostModel plan_prior = o.conservative ? CostModel{{10.0, 4e-6, 3e-3}} : CostModel{};
  CostModel plan_cost = plan_prior;
  CostFit cost_fit(plan_prior);

  // batches of a step: structures perm[step*W*B, +W*B); worker w's rotation
  // stream covers its B of them (the same augmentation in both pack modes).
  // "worker": worker w gets its B structures; "step": all packed together
  auto make_step = [&](const std::vector<int64_t> &perm, int64_t step, uint64_t rng_seed) {
    if (step_pack) {
      std::vector<int64_t> ids(perm.begin() + step * W * B, perm.begin() + (step + 1) * W * B);
      std::vector<std::array<float, 9>> rot(ids.size());
      for (int w = 0; w < W; w++) {
        std::mt19937_64 rng(rng_seed * 0x9E3779B97F4A7C15ULL + w);
        for (int k = 0; k < B; k++) rot[w * B + k] = random_o3(rng);
      }
      CostModel cost;
      {
        std::lock_guard<std::mutex> lock(plan_mu);
        cost = plan_cost;
      }
      return make_step_batches(train_split, nl_train, ids, rot, species, scaling, cutoff, W, cost,
                               o.mb_budget, step_tiling);
    }
    std::vector<std::future<std::vector<Batch>>> fs;
    for (int w = 0; w < W; w++) {
      fs.push_back(std::async(std::launch::async, [&, w] {
        std::mt19937_64 rng(rng_seed * 0x9E3779B97F4A7C15ULL + w);
        std::vector<int64_t> ids(perm.begin() + (step * W + w) * B, perm.begin() + (step * W + w + 1) * B);
        std::vector<std::array<float, 9>> rot(B);
        for (auto &r : rot) r = random_o3(rng);
        return make_microbatches(train_split, nl_train, ids, rot, species, scaling, cutoff,
                                 o.mb_budget, o.mb_growth, tiling);
      }));
    }
    std::vector<std::vector<Batch>> out;
    for (auto &f : fs) out.push_back(f.get());
    return out;
  };

  // --log-errors: each training structure's errors before the update it takes part in
  std::unique_ptr<std::FILE, int (*)(std::FILE *)> log_file(nullptr, std::fclose);
  if (!o.log_errors.empty() && !eval_only) {
    log_file.reset(std::fopen(o.log_errors.c_str(), "wb"));
    if (!log_file) throw std::runtime_error("cannot open " + o.log_errors);
    fmt::print("--log-errors {}: per-structure training errors of the first {} epoch(s)\n", o.log_errors,
               o.log_errors_epochs);
  }

  int64_t global_step = state.step;
  if (jsonl && o.resume.empty()) log_line(-1, evaluate(val_split, nl_val, ids_val), ",\"init\":true"); // the initial model (epoch -1: before any training)
  const auto t_run = clock_type::now();
  for (int epoch = static_cast<int>(state.epoch); epoch < o.epochs && global_step != o.max_steps; epoch++) {
    const bool logging = log_file && epoch < o.log_errors_epochs;
    collect_errors(logging);
    std::vector<int64_t> perm = ids_train;
    std::shuffle(perm.begin(), perm.end(), std::mt19937_64(o.seed + epoch));

    auto t_epoch = clock_type::now();
    StepStats ep;
    double t_wait = 0, t_gpu = 0, t_sync = 0, t_build = 0, t_compute = 0;
    int64_t n_micro = 0;
    // packing runs `prefetch` steps ahead of the GPUs
    std::deque<std::future<std::vector<std::vector<Batch>>>> ahead;
    int64_t queued = 0;
    auto refill = [&] {
      while (queued < steps_per_epoch && static_cast<int>(ahead.size()) < std::max(1, o.prefetch)) {
        ahead.push_back(std::async(std::launch::async, make_step, std::cref(perm), queued,
                                   o.seed + epoch * 1000003ULL + queued));
        queued++;
      }
    };
    refill();
    for (int64_t step = 0; step < steps_per_epoch && global_step != o.max_steps; step++, global_step++) {
      auto t = clock_type::now();
      std::vector<std::vector<Batch>> batches = ahead.front().get();
      ahead.pop_front();
      refill();
      t_wait += seconds_since(t);

      t = clock_type::now();
      // each worker: its micro-batches, gradients summed (= full-batch gradient)
      std::vector<StepStats> st(W);
      std::vector<double> wb(W, 0.0), wc(W, 0.0);
      std::vector<std::vector<std::array<double, 4>>> samples(W); // (n, m, ms, fixed ms) per micro-batch
      for_workers(workers, [&](size_t w) {
        for (const Batch &mb : batches[w]) {
          // the cost model is fitted to the whole call, not just the GPU
          // compute: graph builds, allocation, transfers and (conservative)
          // the force assembly on the CPU are per micro-batch costs too
          const auto t_mb = clock_type::now();
          st[w] += workers[w]->run(mb, scaling.energy_scale, true);
          wb[w] += workers[w]->t_build, wc[w] += workers[w]->t_compute;
          samples[w].push_back({double(mb.n_atoms), double(mb.max_neighbors), 1e3 * seconds_since(t_mb),
                                1e3 * (workers[w]->t_build + workers[w]->t_alloc)});
        }
        optimizer.stage(w); // ship the gradients while the other workers still compute
      });
      if (logging) drain_errors(log_file.get(), static_cast<int32_t>(global_step));
      t_gpu += seconds_since(t);
      for (int w = 0; w < W; w++) t_build += wb[w] / W, t_compute += wc[w] / W;
      for (auto &mb : batches) n_micro += mb.size();

      // DDP: average, clip to the global norm, Adam -- all on device 0
      t = clock_type::now();
      const double lr = grouped ? grouped->opt->schedule().base_lr(global_step, steps_per_epoch, total_steps)
                                : lr_at(global_step);
      const double norm = grouped ? grouped->step(global_step, steps_per_epoch, total_steps)
                                  : optimizer.plain->step(lr);
      if (!std::isfinite(norm)) throw std::runtime_error(fmt::format("non-finite gradient at step {}", global_step));
      t_sync += seconds_since(t);
      if (step_pack) {
        for (auto &sw : samples)
          for (auto [n, m, ms, fixed] : sw) cost_fit.add(n, m, ms, fixed);
        CostModel fitted;
        const bool ok = cost_fit.fit(fitted);
        std::lock_guard<std::mutex> lock(plan_mu);
        if (ok) plan_cost = fitted;
      }

      StepStats step_total;
      for (auto &s : st) ep += s, step_total += s;
      if ((step + 1) % o.log_every == 0) {
        const double el = seconds_since(t_epoch);
        fmt::print("epoch {} step {}/{}: loss {:.4g} (E {:.4g}, F {:.4g}{}) lr {:.3g} |g| {:.3g} | "
                   "{:.1f} struct/s; per step: wait {:.1f} ms, gpu {:.1f} (build {:.1f}, compute {:.1f}), "
                   "sync {:.1f} ms; {:.2f} micro-batches/worker; graphs {} cached ({:.1f} GB buffer), {} misses, {} flushes\n",
                   epoch, step + 1, steps_per_epoch, step_total.loss / W, step_total.loss_energy / W,
                   step_total.loss_force / W,
                   (o.conservative ? fmt::format(", F_nc {:.4g}", step_total.loss_nc / W) : std::string()) +
                       (o.charge_corr_weight > 0
                            ? fmt::format(", q {:.4g}: mean r {:.4f}, {} skipped", step_total.loss_charge / W,
                                          step_total.sum_charge_r / std::max<int64_t>(1, step_total.n_charge_r),
                                          step_total.n_charge_skipped)
                            : std::string()),
                   lr, norm, (step + 1) * W * B / el, 1e3 * t_wait / (step + 1),
                   1e3 * t_gpu / (step + 1), 1e3 * t_build / (step + 1),
                   1e3 * t_compute / (step + 1), 1e3 * t_sync / (step + 1),
                   double(n_micro) / ((step + 1) * W), w0.cache_size(), w0.compute_buffer_bytes() / 1e9,
                   w0.cache_misses, w0.cache_flushes);
        if (step_pack) {
          std::lock_guard<std::mutex> lock(plan_mu);
          fmt::print("  packing cost model: {:.2f} ms + {:.3g} n(m+1)^2 + {:.3g} n(m+1)\n",
                     plan_cost.c[0], plan_cost.c[1], plan_cost.c[2]);
        }
        std::fflush(stdout);
      }
    }
    collect_errors(false);
    if (logging) std::fflush(log_file.get());
    const double t_train = seconds_since(t_epoch);
    const int64_t done = std::max<int64_t>(1, ep.n_systems / B);
    auto [tre, trf] = rmse(ep);

    auto t = clock_type::now();
    const StepStats val = evaluate(val_split, nl_val, ids_val);
    const double t_val = seconds_since(t);
    auto [ve, vf] = rmse(val);
    auto [re, rf] = r2(val);
    if (grouped) // the groups' learning rates of the epoch's last step
      fmt::print("epoch {} lr: {}\n", epoch, grouped->opt->describe(std::max<int64_t>(0, global_step - 1), steps_per_epoch, total_steps));
    fmt::print("epoch {} done: train {:.1f} s ({:.1f} struct/s, mean loss {:.4g}), RMSE E {:.2f} meV/atom, "
               "F {:.2f} meV/A{} | val ({:.1f} s) RMSE E {:.2f} meV/atom, F {:.2f} meV/A{}, R2 E {:.4f}% F {:.4f}%\n",
               epoch, t_train, ep.n_systems / t_train, ep.loss / done, tre, trf, nc_rmse(ep),
               t_val, ve, vf, nc_rmse(val), re, rf);
    if (o.charge_corr_weight > 0)
      fmt::print("epoch {} charge correlation ({}): mean r train {:.4f} ({} structures, {} skipped), val {:.4f} ({} structures)\n",
                 epoch, o.charge_corr_target, ep.sum_charge_r / std::max<int64_t>(1, ep.n_charge_r), ep.n_charge_r,
                 ep.n_charge_skipped, val.sum_charge_r / std::max<int64_t>(1, val.n_charge_r), val.n_charge_r);
    std::fflush(stdout);
    params = w0.get_params(); // the master copy lives on device 0
    const int64_t done_epochs = epoch + 1;
    // LOREM's checkpoint selection: the highest R^2(E) + R^2(F) on val/
    const bool best = re + rf > state.best;
    if (best) state.best = re + rf, write_gguf(o.init_gguf, o.output + ".best.gguf", w0, params, scaling);
    log_line(done_epochs, val,
             fmt::format(",\"step\":{},\"lr\":{},\"train_loss\":{},\"train_e_rmse_mev_atom\":{},\"train_f_rmse_mev_a\":{},"
                         "\"wall_s\":{},\"val_s\":{},\"best\":{}",
                         global_step, grouped ? grouped->opt->schedule().base_lr(global_step, steps_per_epoch, total_steps)
                                              : lr_at(global_step),
                         ep.loss / done, tre, trf, t_train, t_val, best) +
                 (grouped ? ",\"lr_groups\":" + grouped->lr_json(std::max<int64_t>(0, global_step - 1), steps_per_epoch, total_steps)
                          : std::string()));
    if (o.save_every <= 0)
      write_gguf(o.init_gguf, fmt::format("{}.epoch{}", o.output, epoch), w0, params, scaling);
    else if (done_epochs % o.save_every == 0)
      write_gguf(o.init_gguf, fmt::format("{}.epoch{:05d}.gguf", o.output, done_epochs), w0, params, scaling);
    const bool pause = o.max_hours > 0 && done_epochs < o.epochs && seconds_since(t_run) / 3600 > o.max_hours;
    if ((o.state_every > 0 && done_epochs % o.state_every == 0) || pause) {
      write_gguf(o.init_gguf, o.output, w0, params, scaling);
      state.epoch = done_epochs, state.step = global_step, state.optimizer = optimizer.get_state();
      write_state(o.output + ".state", state);
    }
    std::fflush(stdout);
    if (pause) {
      fmt::print("paused at epoch {} ({:.2f} h): wrote {} and {}.state\n", done_epochs, seconds_since(t_run) / 3600,
                 o.output, o.output);
      return;
    }
  }

  log_file.reset();
  // --dump-errors PREFIX: PREFIX.val.bin and PREFIX.test.bin (SystemError records)
  auto dump = [&](const char *name, auto &&run) {
    if (o.dump_errors.empty()) return run();
    collect_errors(true, true);
    auto st = run();
    collect_errors(false);
    const std::string path = fmt::format("{}.{}.bin", o.dump_errors, name),
                      apath = fmt::format("{}.{}.atoms.bin", o.dump_errors, name);
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> f(std::fopen(path.c_str(), "wb"), std::fclose),
        fa(std::fopen(apath.c_str(), "wb"), std::fclose);
    if (!f || !fa) throw std::runtime_error("cannot open " + path + " / " + apath);
    for (auto &a : atom_errors) { // the same worker order as drain_errors' records
      if (!a.empty() && std::fwrite(a.data(), sizeof(float), a.size(), fa.get()) != a.size())
        throw std::runtime_error("cannot write the per-atom errors");
      a.clear();
    }
    drain_errors(f.get(), -1);
    fmt::print("wrote {} and {}\n", path, apath);
    return st;
  };
  if (eval_only || !o.dump_errors.empty()) {
    const StepStats val = dump("val", [&] { return evaluate(val_split, nl_val, ids_val); });
    auto [ve, vf] = rmse(val);
    fmt::print("val: RMSE E {:.2f} meV/atom, F {:.2f} meV/A{} over {} structures\n", ve, vf, nc_rmse(val),
               ids_val.size());
  }
  const StepStats test = dump("test", [&] { return evaluate(test_split, nl_test, ids_test); });
  auto [te, tf] = rmse(test);
  fmt::print("test: RMSE E {:.2f} meV/atom, F {:.2f} meV/A{} over {} structures\n", te, tf, nc_rmse(test),
             test.n_systems);
  std::fflush(stdout);
  if (!eval_only) {
    params = w0.get_params();
    write_gguf(o.init_gguf, o.output, w0, params, scaling);
    fmt::print("wrote {}\n", o.output);
  }
}

} // namespace mlipcpp::train
