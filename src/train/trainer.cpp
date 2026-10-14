#include "train/trainer.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <ggml-impl.h> // ggml_graph_view, for profiling
#include <gguf.h>

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
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

std::string cache_path(const std::string &cache_dir, const MemmapSplit &split, float cutoff) {
  if (cache_dir.empty()) return {};
  // keyed by content, not path: the data is staged to a per-job $TMPDIR
  uint64_t h = 1469598103934665603ULL; // FNV-1a
  auto mix = [&](const void *p, size_t n) {
    for (size_t k = 0; k < n; k++) h = (h ^ static_cast<const uint8_t *>(p)[k]) * 1099511628211ULL;
  };
  const int64_t ns = split.size(), N = split.total_atoms();
  mix(&ns, sizeof ns), mix(&N, sizeof N), mix(&cutoff, sizeof cutoff);
  mix(split.numbers(0), N * sizeof(int32_t));
  mix(split.positions(0), N * 3 * sizeof(float));
  mix(split.cell(0), ns * 9 * sizeof(float));
  return fmt::format("{}/nl-{:016x}.bin", cache_dir, h);
}

StepStats &StepStats::operator+=(const StepStats &o) {
  loss += o.loss, loss_energy += o.loss_energy, loss_force += o.loss_force;
  se_energy_per_atom += o.se_energy_per_atom, se_force += o.se_force;
  n_systems += o.n_systems, n_force_components += o.n_force_components;
  return *this;
}

Worker::Worker(const std::string &gguf, int device, size_t cache_graphs) : cache_limit_(cache_graphs) {
  model_.set_backend_preference(device < 0 ? BackendPreference::CPU : BackendPreference::Auto);
  model_.set_device(std::max(device, 0));
  model_.load_from_gguf(gguf);
  if (!model_.nc_forces() || !model_.forces_mode())
    throw std::runtime_error(gguf + ": training needs a forces-mode export with --nc-forces");
  if (model_.param_names().empty())
    throw std::runtime_error(gguf + ": no train.param_names; re-export with export_pet_gguf.py");
  for (const auto &name : model_.param_names()) {
    ggml_tensor *t = find_weight(model_, name);
    ggml_set_param(t);
    params_.push_back(t);
    n_params_ += ggml_nelements(t);
  }
  galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_.backend()));

  acc_ctx_ = ggml_init({ggml_tensor_overhead() * (params_.size() + 1), nullptr, true});
  acc_flat_ = ggml_new_tensor_1d(acc_ctx_, GGML_TYPE_F32, n_params_);
  ggml_set_name(acc_flat_, "grad_flat");
  size_t off = 0;
  for (ggml_tensor *p : params_) {
    const size_t nb1 = p->ne[0] * sizeof(float), nb2 = nb1 * p->ne[1], nb3 = nb2 * p->ne[2];
    acc_[p] = ggml_view_4d(acc_ctx_, acc_flat_, p->ne[0], p->ne[1], p->ne[2], p->ne[3], nb1, nb2,
                           nb3, off * sizeof(float));
    off += ggml_nelements(p);
  }
  acc_buf_ = ggml_backend_alloc_ctx_tensors(acc_ctx_, model_.backend());
  if (!acc_buf_) throw std::runtime_error("cannot allocate the gradient accumulator");
  zero_grads();
}

Worker::~Worker() {
  cache_.clear();
  if (galloc_) ggml_gallocr_free(galloc_);
  if (acc_buf_) ggml_backend_buffer_free(acc_buf_);
  if (acc_ctx_) ggml_free(acc_ctx_);
}

void Worker::zero_grads() { ggml_backend_buffer_clear(acc_buf_, 0); }

std::vector<float> Worker::get_grads() const {
  std::vector<float> g(n_params_);
  ggml_backend_tensor_get(acc_flat_, g.data(), 0, ggml_nbytes(acc_flat_));
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

std::unique_ptr<Worker::StepGraph> Worker::build_graph(int n, int m, int S, bool backward, bool cached) {
  auto g = std::make_unique<StepGraph>();
  auto &interp = model_.interpreter();
  interp.set_dimension("n_atoms", n);
  interp.set_dimension("max_neighbors", m);
  interp.set_dimension("n_edges", int64_t(n) * m);
  interp.set_dimension("max_neighbors_plus_one", m + 1);

  // every shape has the same topology: the first build measures the context
  // (cached graphs keep theirs, so it is kept tight)
  size_t &ctx_bytes = ctx_bytes_[backward];
  const bool measure = ctx_bytes == 0;
  g->ctx = ggml_init({measure ? ggml_tensor_overhead() * GRAPH_SIZE * 2 +
                                    ggml_graph_overhead_custom(GRAPH_SIZE, true)
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
  interp.set_input("edge_vectors", in(GGML_TYPE_F32, {3, m, n}, "edge_vectors"));
  interp.set_input("padding_mask", in(GGML_TYPE_F32, {m, n}, "padding_mask"));
  interp.set_input("reverse_neighbor_index",
                   in(GGML_TYPE_I32, {int64_t(n) * m}, "reverse_neighbor_index"));
  interp.set_input("cutoff_values", in(GGML_TYPE_F32, {m, n}, "cutoff_values"));
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

  // out: [n_atoms, 4] = (energy | force) per atom; ggml ne = (4, n)
  ggml_tensor *out = interp.build(ctx);
  ggml_tensor *energy = ggml_reshape_2d(
      ctx, ggml_cont(ctx, ggml_view_2d(ctx, out, 1, n, out->nb[1], 0)), n, 1);
  ggml_tensor *force = ggml_cont(ctx, ggml_view_2d(ctx, out, 3, n, out->nb[1], sizeof(float)));
  // metatrain's average_by_num_atoms: per-structure Σ_i e_i / N
  ggml_tensor *d_energy = ggml_sub(ctx, ggml_mul_mat(ctx, atom_weight, energy), energy_target);
  ggml_tensor *d_force = ggml_mul(ctx, ggml_sub(ctx, force, force_target), force_mask);
  ggml_tensor *loss_e = ggml_mul(ctx, ggml_sum(ctx, ggml_sqr(ctx, d_energy)), e_norm);
  ggml_tensor *loss_f = ggml_mul(ctx, ggml_sum(ctx, ggml_sqr(ctx, d_force)), f_norm);
  ggml_tensor *loss = ggml_add(ctx, loss_e, loss_f);
  // physical force squared error, for the metrics (not differentiated)
  ggml_tensor *se_f = ggml_sum(ctx, ggml_sqr(ctx, ggml_mul(ctx, d_force, force_scale)));
  g->stats = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, loss, loss_e, 0), loss_f, 0), se_f, 0);
  ggml_set_output(g->stats);

  g->gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, backward);
  ggml_build_forward_expand(g->gf, g->stats);
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
  g->constants = interp.take_constants();
  if (g->loss_grad) g->constants.emplace_back(g->loss_grad, 1.0f);
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

Worker::StepGraph &Worker::cached_graph(int n, int m, int S, bool backward) {
  const auto key = std::make_tuple(n, m, S, backward);
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
  auto g = build_graph(n, m, S, backward, true);
  // all cached graphs share the compute buffer (they run one at a time); when
  // it has to grow, the old buffer is freed and every other graph dangles
  const size_t before = compute_buffer_bytes();
  if (!ggml_gallocr_alloc_graph(galloc_, g->gf))
    throw std::runtime_error(fmt::format("cannot allocate the graph ({} atoms x {} neighbours)", n, m));
  if (compute_buffer_bytes() != before && !cache_.empty()) cache_.clear(), cache_flushes++;
  g->last_use = ++clock_;
  return *(cache_[key] = std::move(g));
}

void Worker::upload_inputs(StepGraph &g, const Batch &b) {
  const auto [e_norm, f_norm] = loss_norms(b);
  const float norms[2] = {e_norm, f_norm};
  auto span = [](const auto &v) {
    return std::pair<const void *, size_t>(v.data(), v.size() * sizeof(v[0]));
  };
  const std::pair<const void *, size_t> src[] = {
      span(b.species),        span(b.neighbor_species), span(b.edge_vectors),
      span(b.padding_mask),   span(b.reverse_neighbor_index), span(b.cutoff_values),
      span(b.atom_weight),    span(b.energy_target),   span(b.force_target),
      span(b.force_mask),     span(b.force_scale),      {norms, sizeof norms}};
  if (std::size(src) != g.inputs.size()) throw std::logic_error("upload_inputs: input count");
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

StepStats Worker::run(const Batch &b, double energy_scale, bool backward) {
  auto t0 = clock_type::now();
  const int n = b.n_atoms, m = b.max_neighbors, S = b.n_systems;
  std::unique_ptr<StepGraph> once;
  StepGraph *g;
  if (cache_limit_ > 0) {
    g = &cached_graph(n, m, S, backward);
    t_build = seconds_since(t0), t_alloc = 0;
  } else {
    once = build_graph(n, m, S, backward, false);
    g = once.get();
    t_build = seconds_since(t0);
    t0 = clock_type::now();
    if (!ggml_gallocr_alloc_graph(galloc_, g->gf))
      throw std::runtime_error(fmt::format("cannot allocate the graph ({} atoms x {} neighbours)", n, m));
    fill_constants(g->constants);
    t_alloc = seconds_since(t0);
  }

  t0 = clock_type::now();
  upload_inputs(*g, b);
  t_upload = seconds_since(t0);

  t0 = clock_type::now();
  ggml_cgraph *gf = g->gf;
  if (profile) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
      ggml_cgraph one = ggml_graph_view(gf, i, i + 1);
      auto tn = clock_type::now();
      if (ggml_backend_graph_compute(model_.backend(), &one) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph compute failed");
      ggml_backend_synchronize(model_.backend());
      auto &[ms, count] = profile_ms[std::string(i < g->n_forward ? "fwd:" : "bwd:") + ggml_op_desc(ggml_graph_node(gf, i))];
      ms += 1e3 * seconds_since(tn), count++;
    }
  } else if (ggml_backend_graph_compute(model_.backend(), gf) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("graph compute failed");
  }
  t_compute = seconds_since(t0);
  if (after_compute) after_compute(gf);

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
  for (float w : b.force_mask) st.n_force_components += 3 * static_cast<int64_t>(w);
  t_download = seconds_since(t0);
  return st;
}

namespace {
/// Muon applies to hidden matrices: 2-D, both dims >= 16, not an embedding
/// table (one-hot inputs: Adam, as in Muon's reference recipe).
bool is_muon_param(const ggml_tensor *p) {
  return ggml_n_dims(p) == 2 && std::min(p->ne[0], p->ne[1]) >= 16 &&
         std::string(ggml_get_name(p)).find("embed") == std::string::npos;
}

/// Newton-Schulz quintic iteration (Muon): approximately U V^T of each
/// g[:, :, i] = U S V^T. ggml (ne0, ne1) = PyTorch (cols, rows); the Gram
/// matrix is taken over the shorter side. Same-shape matrices are stacked
/// along ne2 and orthogonalised together (batched mul_mat).
ggml_tensor *orthogonalize(ggml_context *ctx, ggml_tensor *g, int steps) {
  const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
  const bool tall = g->ne[0] < g->ne[1]; // more rows than cols: work on g^T
  ggml_tensor *x = tall ? ggml_cont(ctx, ggml_transpose(ctx, g)) : g;
  const int64_t cols = x->ne[0], rows = x->ne[1], k = x->ne[2];
  // per-matrix Frobenius norm
  ggml_tensor *n2 = ggml_sum_rows(ctx, ggml_reshape_2d(ctx, ggml_sqr(ctx, x), cols * rows, k));
  x = ggml_div(ctx, x, ggml_reshape_3d(ctx, ggml_scale_bias(ctx, ggml_sqrt(ctx, n2), 1.0f, 1e-7f), 1, 1, k));
  for (int it = 0; it < steps; it++) {
    ggml_tensor *A = ggml_mul_mat(ctx, x, x);                                      // X X^T
    ggml_tensor *B = ggml_add(ctx, ggml_scale(ctx, A, b), ggml_scale(ctx, ggml_mul_mat(ctx, A, A), c));
    ggml_tensor *BX = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)), B); // B X (B symmetric)
    x = ggml_add(ctx, ggml_scale(ctx, x, a), BX);
  }
  return tall ? ggml_cont(ctx, ggml_transpose(ctx, x)) : x;
}
} // namespace

Optimizer::Optimizer(std::vector<std::unique_ptr<Worker>> &workers, double clip, bool muon,
                     double muon_momentum, double muon_lr_scale, int ns_steps)
    : workers_(workers), muon_lr_scale_(muon_lr_scale) {
  Worker &w0 = *workers_[0];
  const auto &params = w0.params();
  const int W = static_cast<int>(workers_.size());
  ggml_backend_t backend = w0.model().backend();
  ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

  // persistent state on device 0: Adam moments, staging for the other
  // workers' gradients, hyper-parameters and constants
  ctx_ = ggml_init({ggml_tensor_overhead() * (2 * params.size() + W + 8), nullptr, true});
  std::vector<ggml_tensor *> m, v;
  for (ggml_tensor *p : params) {
    m.push_back(ggml_dup_tensor(ctx_, p));
    v.push_back(ggml_dup_tensor(ctx_, p));
  }
  for (int w = 1; w < W; w++) staging_.push_back(ggml_dup_tensor(ctx_, w0.grad_flat()));
  pars_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 7);
  ggml_tensor *one = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
  ggml_tensor *eps = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
  muon_lr_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
  buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
  if (!buf_) throw std::runtime_error("cannot allocate the optimizer state");
  ggml_backend_buffer_clear(buf_, 0);
  const float c1 = 1.0f, ce = 1e-6f;
  ggml_backend_tensor_set(one, &c1, 0, sizeof c1);
  ggml_backend_tensor_set(eps, &ce, 0, sizeof ce);

  // step graph: g = mean_w grad_w; c = min(1, clip / (|g| + 1e-6)); Adam(c g)
  const size_t n_nodes = (16 + 12 * ns_steps) * params.size() + 64;
  gctx_ = ggml_init({ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false),
                     nullptr, true});
  ggml_tensor *g = w0.grad_flat();
  for (ggml_tensor *st : staging_) g = ggml_add(gctx_, g, st);
  if (W > 1) g = ggml_scale(gctx_, g, 1.0f / W);
  norm_ = ggml_sqrt(gctx_, ggml_sum(gctx_, ggml_sqr(gctx_, g)));
  ggml_tensor *c = ggml_clamp(gctx_, ggml_scale(gctx_, ggml_div(gctx_, one, ggml_add(gctx_, norm_, eps)),
                                                static_cast<float>(clip)),
                              0.0f, 1.0f);
  ggml_tensor *gs = ggml_mul(gctx_, g, c);
  graph_ = ggml_new_graph_custom(gctx_, n_nodes, false);
  ggml_set_output(norm_);
  ggml_build_forward_expand(graph_, norm_);
  // Muon: (param, Nesterov momentum) per matrix shape, orthogonalised per group
  std::map<std::pair<int64_t, int64_t>, std::vector<std::pair<ggml_tensor *, ggml_tensor *>>> muon_groups;
  size_t off = 0;
  for (size_t k = 0; k < params.size(); k++) {
    ggml_tensor *p = params[k];
    ggml_tensor *gk = ggml_view_4d(gctx_, gs, p->ne[0], p->ne[1], p->ne[2], p->ne[3], p->nb[1],
                                   p->nb[2], p->nb[3], off * sizeof(float));
    if (muon && is_muon_param(p)) {
      // m[k] is the momentum: b = mu b + (1-mu) g; Nesterov u = (1-mu) g + mu b
      const float mu = static_cast<float>(muon_momentum);
      ggml_tensor *buf = ggml_add(gctx_, ggml_scale(gctx_, m[k], mu), ggml_scale(gctx_, gk, 1 - mu));
      ggml_build_forward_expand(graph_, ggml_cpy(gctx_, buf, m[k]));
      muon_groups[{p->ne[0], p->ne[1]}].emplace_back(
          p, ggml_add(gctx_, ggml_scale(gctx_, gk, 1 - mu), ggml_scale(gctx_, buf, mu)));
      n_muon_tensors++;
    } else {
      ggml_build_forward_expand(graph_, ggml_opt_step_adamw(gctx_, p, gk, m[k], v[k], pars_));
    }
    off += ggml_nelements(p);
  }
  for (auto &[shape, group] : muon_groups) {
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
}

Optimizer::~Optimizer() {
  if (galloc_) ggml_gallocr_free(galloc_);
  if (buf_) ggml_backend_buffer_free(buf_);
  if (gctx_) ggml_free(gctx_);
  if (ctx_) ggml_free(ctx_);
}

double Optimizer::step(double lr) {
  Worker &w0 = *workers_[0];
  ++t_;
  const double b1 = 0.9, b2 = 0.999;
  const float pars[7] = {static_cast<float>(lr), static_cast<float>(b1), static_cast<float>(b2), 1e-8f, 0.0f,
                         static_cast<float>(1.0 / (1.0 - std::pow(b1, t_))),
                         static_cast<float>(1.0 / (1.0 - std::pow(b2, t_)))};
  ggml_backend_tensor_set(pars_, pars, 0, sizeof pars);
  const float mlr = static_cast<float>(lr * muon_lr_scale_);
  ggml_backend_tensor_set(muon_lr_, &mlr, 0, sizeof mlr);
  for (size_t w = 1; w < workers_.size(); w++)
    ggml_backend_tensor_copy(workers_[w]->grad_flat(), staging_[w - 1]); // peer copy
  if (ggml_backend_graph_compute(w0.model().backend(), graph_) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("optimizer step failed");
  float norm = 0;
  ggml_backend_tensor_get(norm_, &norm, 0, sizeof norm);
  for (size_t w = 1; w < workers_.size(); w++)
    for (size_t k = 0; k < w0.params().size(); k++)
      ggml_backend_tensor_copy(w0.params()[k], workers_[w]->params()[k]);
  for (auto &w : workers_) w->zero_grads();
  return norm;
}

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
  Worker &w0 = *workers[0];
  const float cutoff = w0.model().cutoff();
  const auto &species = w0.model().species_to_index();
  fmt::print("model: {} parameters in {} tensors, cutoff {} A, {} worker(s) on {}\n",
             w0.n_params(), w0.params().size(), cutoff, workers.size(),
             ggml_backend_name(w0.model().backend()));

  // --epochs 0 evaluates a trained model with the baselines stored in it
  const bool eval_only = o.epochs == 0;
  auto t0 = clock_type::now();
  TargetScaling scaling;
  if (eval_only) {
    for (auto [Z, c] : w0.model().composition_energies()) scaling.composition[Z] = c;
    for (auto [Z, f] : w0.model().nc_force_scales()) scaling.force_scale[Z] = f;
    scaling.energy_scale = w0.model().energy_scale();
  } else {
    scaling = TargetScaling::fit(train_split);
  }
  fmt::print("targets ({}): composition for {} species, energy scale {:.6g} eV/atom ({:.1f} s)\n",
             eval_only ? "from the model" : "fitted", scaling.composition.size(),
             scaling.energy_scale, seconds_since(t0));

  t0 = clock_type::now();
  auto nl = [&](const MemmapSplit &s) {
    return NeighborCache::load_or_build(s, cutoff, cache_path(o.cache_dir, s, cutoff), o.n_threads);
  };
  const NeighborCache nl_train = eval_only ? NeighborCache{} : nl(train_split),
                      nl_val = nl(val_split), nl_test = nl(test_split);
  fmt::print("neighbour lists ready ({:.1f} s)\n", seconds_since(t0));

  const int W = static_cast<int>(workers.size()), B = o.batch_size;
  const Tiling tiling{o.tile_nb_ratio, o.tile_atom_ratio, B};
  const int64_t steps_per_epoch = train_split.size() / (int64_t(B) * W); // drop_last
  const int64_t total_steps = o.epochs * steps_per_epoch;
  const int64_t warmup = static_cast<int64_t>(o.warmup_fraction * total_steps);

  // identical parameters on every worker, then the device-side optimizer
  std::vector<float> params = w0.get_params();
  for (size_t w = 1; w < workers.size(); w++) workers[w]->set_params(params);
  if (o.optimizer != "adam" && o.optimizer != "muon")
    throw std::runtime_error("--optimizer must be adam or muon");
  Optimizer optimizer(workers, o.grad_clip, o.optimizer == "muon", o.muon_momentum,
                      o.muon_lr_scale, o.muon_ns_steps);
  fmt::print("optimizer: {}{}; {}\n", o.optimizer,
             o.optimizer == "muon" ? fmt::format(" ({} tensors, momentum {}, lr scale {}, {} NS steps)",
                                                 optimizer.n_muon_tensors, o.muon_momentum,
                                                 o.muon_lr_scale, o.muon_ns_steps)
                                   : "",
             tiling.enabled() ? fmt::format("tiles: neighbour ratio {}, atom ratio {}, {} systems, "
                                            "graph cache {} shapes/device",
                                            o.tile_nb_ratio, o.tile_atom_ratio, B, o.graph_cache)
                              : std::string("dynamic shapes"));
  fmt::print("setup: {:.1f} s; {} steps/epoch of {} x {} structures, warmup {} steps\n",
             seconds_since(t_setup), steps_per_epoch, W, B, warmup);

  auto evaluate = [&](const MemmapSplit &split, const NeighborCache &cache) {
    std::vector<StepStats> per(W);
    const int64_t n_batches = (split.size() + B - 1) / B;
    for_workers(workers, [&](size_t w) {
      for (int64_t k = w; k < n_batches; k += W) {
        std::vector<int64_t> ids;
        for (int64_t s = k * B; s < std::min<int64_t>(split.size(), (k + 1) * B); s++) ids.push_back(s);
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

  // batches of a step: worker w gets perm[(step*W + w)*B, +B), rotated
  auto make_step = [&](const std::vector<int64_t> &perm, int64_t step, uint64_t rng_seed) {
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

  int64_t global_step = 0;
  for (int epoch = 0; epoch < o.epochs && global_step != o.max_steps; epoch++) {
    std::vector<int64_t> perm(train_split.size());
    std::iota(perm.begin(), perm.end(), 0);
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
      for_workers(workers, [&](size_t w) {
        for (const Batch &mb : batches[w]) {
          st[w] += workers[w]->run(mb, scaling.energy_scale, true);
          wb[w] += workers[w]->t_build, wc[w] += workers[w]->t_compute;
        }
      });
      t_gpu += seconds_since(t);
      for (int w = 0; w < W; w++) t_build += wb[w] / W, t_compute += wc[w] / W;
      for (auto &mb : batches) n_micro += mb.size();

      // DDP: average, clip to the global norm, Adam -- all on device 0
      t = clock_type::now();
      const double lr = o.lr * lr_factor(global_step, total_steps, warmup);
      const double norm = optimizer.step(lr);
      if (!std::isfinite(norm)) throw std::runtime_error(fmt::format("non-finite gradient at step {}", global_step));
      t_sync += seconds_since(t);

      for (auto &s : st) ep += s;
      if ((step + 1) % o.log_every == 0) {
        const double el = seconds_since(t_epoch);
        fmt::print("epoch {} step {}/{}: loss {:.4g} (E {:.4g}, F {:.4g}) lr {:.3g} |g| {:.3g} | "
                   "{:.1f} struct/s; per step: wait {:.1f} ms, gpu {:.1f} (build {:.1f}, compute {:.1f}), "
                   "sync {:.1f} ms; {:.2f} micro-batches/worker; graphs {} cached ({:.1f} GB buffer), {} misses, {} flushes\n",
                   epoch, step + 1, steps_per_epoch, st[0].loss, st[0].loss_energy, st[0].loss_force,
                   lr, norm, (step + 1) * W * B / el, 1e3 * t_wait / (step + 1),
                   1e3 * t_gpu / (step + 1), 1e3 * t_build / (step + 1),
                   1e3 * t_compute / (step + 1), 1e3 * t_sync / (step + 1),
                   double(n_micro) / ((step + 1) * W), w0.cache_size(), w0.compute_buffer_bytes() / 1e9,
                   w0.cache_misses, w0.cache_flushes);
        std::fflush(stdout);
      }
    }
    const double t_train = seconds_since(t_epoch);
    const int64_t done = std::max<int64_t>(1, ep.n_systems / B);
    auto [tre, trf] = rmse(ep);

    auto t = clock_type::now();
    const StepStats val = evaluate(val_split, nl_val);
    auto [ve, vf] = rmse(val);
    fmt::print("epoch {} done: train {:.1f} s ({:.1f} struct/s, mean loss {:.4g}), RMSE E {:.2f} meV/atom, "
               "F {:.2f} meV/A | val ({:.1f} s) RMSE E {:.2f} meV/atom, F {:.2f} meV/A\n",
               epoch, t_train, ep.n_systems / t_train, ep.loss / done, tre, trf,
               seconds_since(t), ve, vf);
    std::fflush(stdout);
    params = w0.get_params(); // the master copy lives on device 0
    write_gguf(o.init_gguf, fmt::format("{}.epoch{}", o.output, epoch), w0, params, scaling);
    std::fflush(stdout);
  }

  if (eval_only) {
    auto [ve, vf] = rmse(evaluate(val_split, nl_val));
    fmt::print("val: RMSE E {:.2f} meV/atom, F {:.2f} meV/A over {} structures\n", ve, vf, val_split.size());
  }
  const StepStats test = evaluate(test_split, nl_test);
  auto [te, tf] = rmse(test);
  fmt::print("test: RMSE E {:.2f} meV/atom, F {:.2f} meV/A over {} structures\n", te, tf, test.n_systems);
  std::fflush(stdout);
  if (!eval_only) {
    params = w0.get_params();
    write_gguf(o.init_gguf, o.output, w0, params, scaling);
    fmt::print("wrote {}\n", o.output);
  }
}

} // namespace mlipcpp::train
