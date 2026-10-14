#include "train/trainer.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <ggml-impl.h> // ggml_graph_view, for profiling
#include <gguf.h>

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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

template <typename V> void upload(ggml_tensor *t, const std::vector<V> &v) {
  if (ggml_nbytes(t) != v.size() * sizeof(V))
    throw std::logic_error(std::string("upload size mismatch for ") + ggml_get_name(t));
  ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

float read_scalar(ggml_tensor *t) {
  float v;
  ggml_backend_tensor_get(t, &v, 0, sizeof v);
  return v;
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

Worker::Worker(const std::string &gguf, int device) {
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

StepStats Worker::run(const Batch &b, double energy_scale, bool backward) {
  auto t0 = clock_type::now();
  const int n = b.n_atoms, m = b.max_neighbors, S = b.n_systems;
  auto &interp = model_.interpreter();
  interp.set_dimension("n_atoms", n);
  interp.set_dimension("max_neighbors", m);
  interp.set_dimension("n_edges", int64_t(n) * m);
  interp.set_dimension("max_neighbors_plus_one", m + 1);

  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * GRAPH_SIZE * 2 +
                                     ggml_graph_overhead_custom(GRAPH_SIZE, true),
                                 nullptr, /*no_alloc=*/true});
  // ggml shapes are the reversed PyTorch shapes
  ggml_tensor *species = input(ctx, GGML_TYPE_I32, {n}, "species");
  ggml_tensor *neighbor_species = input(ctx, GGML_TYPE_I32, {m, n}, "neighbor_species");
  ggml_tensor *edge_vectors = input(ctx, GGML_TYPE_F32, {3, m, n}, "edge_vectors");
  ggml_tensor *padding_mask = input(ctx, GGML_TYPE_F32, {m, n}, "padding_mask");
  ggml_tensor *reverse = input(ctx, GGML_TYPE_I32, {int64_t(n) * m}, "reverse_neighbor_index");
  ggml_tensor *cutoff_values = input(ctx, GGML_TYPE_F32, {m, n}, "cutoff_values");
  ggml_tensor *atom_weight = input(ctx, GGML_TYPE_F32, {n, S}, "atom_weight");
  ggml_tensor *energy_target = input(ctx, GGML_TYPE_F32, {S, 1}, "energy_target");
  ggml_tensor *force_target = input(ctx, GGML_TYPE_F32, {3, n}, "force_target");
  interp.set_input("species", species);
  interp.set_input("neighbor_species", neighbor_species);
  interp.set_input("edge_vectors", edge_vectors);
  interp.set_input("padding_mask", padding_mask);
  interp.set_input("reverse_neighbor_index", reverse);
  interp.set_input("cutoff_values", cutoff_values);

  // out: [n_atoms, 4] = (energy | force) per atom; ggml ne = (4, n)
  ggml_tensor *out = interp.build(ctx);
  ggml_tensor *energy = ggml_reshape_2d(
      ctx, ggml_cont(ctx, ggml_view_2d(ctx, out, 1, n, out->nb[1], 0)), n, 1);
  ggml_tensor *force = ggml_cont(ctx, ggml_view_2d(ctx, out, 3, n, out->nb[1], sizeof(float)));
  // metatrain's average_by_num_atoms: per-structure Σ_i e_i / N
  ggml_tensor *d_energy = ggml_sub(ctx, ggml_mul_mat(ctx, atom_weight, energy), energy_target);
  ggml_tensor *d_force = ggml_sub(ctx, force, force_target);
  const float e_norm = b.energy_norm > 0 ? b.energy_norm : 1.0f / S;
  const float f_norm = b.force_norm > 0 ? b.force_norm : 1.0f / (3.0f * n);
  ggml_tensor *loss_e = ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, d_energy)), e_norm);
  ggml_tensor *loss_f = ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, d_force)), f_norm);
  ggml_tensor *loss = ggml_add(ctx, loss_e, loss_f);
  for (ggml_tensor *t : {d_energy, d_force, loss_e, loss_f, loss}) ggml_set_output(t);

  const bool train = backward;
  ggml_cgraph *gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, train);
  ggml_build_forward_expand(gf, loss);
  const int n_forward_nodes = ggml_graph_n_nodes(gf);
  ggml_tensor *loss_grad = nullptr;
  if (train) {
    ggml_set_loss(loss);
    // parameter gradients accumulate in place into the persistent views
    std::vector<ggml_tensor *> accs(ggml_graph_n_nodes(gf), nullptr);
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
      auto it = acc_.find(ggml_graph_node(gf, i));
      if (it != acc_.end()) accs[i] = it->second;
    }
    ggml_build_backward_expand(ctx, gf, accs.data());
    loss_grad = ggml_graph_get_grad_acc(gf, loss);
    ggml_set_input(loss_grad);
  }
  interp.mark_constants_as_inputs();
  if (after_build) after_build(gf);
  t_build = seconds_since(t0);

  t0 = clock_type::now();
  if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
    ggml_free(ctx);
    throw std::runtime_error(fmt::format("cannot allocate the graph ({} atoms x {} neighbours)", n, m));
  }
  t_alloc = seconds_since(t0);

  t0 = clock_type::now();
  upload(species, b.species);
  upload(neighbor_species, b.neighbor_species);
  upload(edge_vectors, b.edge_vectors);
  upload(padding_mask, b.padding_mask);
  upload(reverse, b.reverse_neighbor_index);
  upload(cutoff_values, b.cutoff_values);
  upload(atom_weight, b.atom_weight);
  upload(energy_target, b.energy_target);
  upload(force_target, b.force_target);
  interp.init_constants();
  if (loss_grad) {
    const float one = 1.0f;
    ggml_backend_tensor_set(loss_grad, &one, 0, sizeof one);
  }
  t_upload = seconds_since(t0);

  t0 = clock_type::now();
  if (profile) {
    const int n_fwd = n_forward_nodes;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
      ggml_cgraph one = ggml_graph_view(gf, i, i + 1);
      auto tn = clock_type::now();
      if (ggml_backend_graph_compute(model_.backend(), &one) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph compute failed");
      ggml_backend_synchronize(model_.backend());
      auto &[ms, count] = profile_ms[std::string(i < n_fwd ? "fwd:" : "bwd:") + ggml_op_desc(ggml_graph_node(gf, i))];
      ms += 1e3 * seconds_since(tn), count++;
    }
  } else if (ggml_backend_graph_compute(model_.backend(), gf) != GGML_STATUS_SUCCESS) {
    ggml_free(ctx);
    throw std::runtime_error("graph compute failed");
  }
  t_compute = seconds_since(t0);
  if (after_compute) after_compute(gf);

  t0 = clock_type::now();
  StepStats st;
  st.loss = read_scalar(loss);
  st.loss_energy = read_scalar(loss_e);
  st.loss_force = read_scalar(loss_f);
  std::vector<float> de(S), df(3 * n);
  ggml_backend_tensor_get(d_energy, de.data(), 0, de.size() * sizeof(float));
  ggml_backend_tensor_get(d_force, df.data(), 0, df.size() * sizeof(float));
  for (float d : de) st.se_energy_per_atom += std::pow(d * energy_scale, 2);
  for (int i = 0; i < n; i++)
    for (int k = 0; k < 3; k++) st.se_force += std::pow(double(df[3 * i + k]) * b.force_scale[i], 2);
  st.n_systems = S;
  st.n_force_components = 3 * int64_t(n);

  t_download = seconds_since(t0);
  ggml_free(ctx);
  return st;
}

Optimizer::Optimizer(std::vector<std::unique_ptr<Worker>> &workers, double clip)
    : workers_(workers) {
  Worker &w0 = *workers_[0];
  const auto &params = w0.params();
  const int W = static_cast<int>(workers_.size());
  ggml_backend_t backend = w0.model().backend();
  ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);

  // persistent state on device 0: Adam moments, staging for the other
  // workers' gradients, hyper-parameters and constants
  ctx_ = ggml_init({ggml_tensor_overhead() * (2 * params.size() + W + 4), nullptr, true});
  std::vector<ggml_tensor *> m, v;
  for (ggml_tensor *p : params) {
    m.push_back(ggml_dup_tensor(ctx_, p));
    v.push_back(ggml_dup_tensor(ctx_, p));
  }
  for (int w = 1; w < W; w++) staging_.push_back(ggml_dup_tensor(ctx_, w0.grad_flat()));
  pars_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 7);
  ggml_tensor *one = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
  ggml_tensor *eps = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, 1);
  buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
  if (!buf_) throw std::runtime_error("cannot allocate the optimizer state");
  ggml_backend_buffer_clear(buf_, 0);
  const float c1 = 1.0f, ce = 1e-6f;
  ggml_backend_tensor_set(one, &c1, 0, sizeof c1);
  ggml_backend_tensor_set(eps, &ce, 0, sizeof ce);

  // step graph: g = mean_w grad_w; c = min(1, clip / (|g| + 1e-6)); Adam(c g)
  gctx_ = ggml_init({ggml_tensor_overhead() * (4 * params.size() + 32) +
                         ggml_graph_overhead_custom(4 * params.size() + 32, false),
                     nullptr, true});
  ggml_tensor *g = w0.grad_flat();
  for (ggml_tensor *st : staging_) g = ggml_add(gctx_, g, st);
  if (W > 1) g = ggml_scale(gctx_, g, 1.0f / W);
  norm_ = ggml_sqrt(gctx_, ggml_sum(gctx_, ggml_sqr(gctx_, g)));
  ggml_tensor *c = ggml_clamp(gctx_, ggml_scale(gctx_, ggml_div(gctx_, one, ggml_add(gctx_, norm_, eps)),
                                                static_cast<float>(clip)),
                              0.0f, 1.0f);
  ggml_tensor *gs = ggml_mul(gctx_, g, c);
  graph_ = ggml_new_graph_custom(gctx_, 4 * params.size() + 32, false);
  ggml_set_output(norm_);
  ggml_build_forward_expand(graph_, norm_);
  size_t off = 0;
  for (size_t k = 0; k < params.size(); k++) {
    ggml_tensor *p = params[k];
    ggml_tensor *gk = ggml_view_4d(gctx_, gs, p->ne[0], p->ne[1], p->ne[2], p->ne[3], p->nb[1],
                                   p->nb[2], p->nb[3], off * sizeof(float));
    ggml_build_forward_expand(graph_, ggml_opt_step_adamw(gctx_, p, gk, m[k], v[k], pars_));
    off += ggml_nelements(p);
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

  std::vector<std::unique_ptr<Worker>> workers;
  for (int w = 0; w < std::max(1, o.n_gpus); w++)
    workers.push_back(std::make_unique<Worker>(o.init_gguf, o.n_gpus == 0 ? -1 : w));
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
  const int64_t steps_per_epoch = train_split.size() / (int64_t(B) * W); // drop_last
  const int64_t total_steps = o.epochs * steps_per_epoch;
  const int64_t warmup = static_cast<int64_t>(o.warmup_fraction * total_steps);

  // identical parameters on every worker, then the device-side optimizer
  std::vector<float> params = w0.get_params();
  for (size_t w = 1; w < workers.size(); w++) workers[w]->set_params(params);
  Optimizer optimizer(workers, o.grad_clip);
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
                                                    o.mb_budget, o.mb_growth))
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
                                 o.mb_budget, o.mb_growth);
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
    auto next = std::async(std::launch::async, make_step, std::cref(perm), 0, o.seed + epoch * 1000003ULL);
    for (int64_t step = 0; step < steps_per_epoch && global_step != o.max_steps; step++, global_step++) {
      auto t = clock_type::now();
      std::vector<std::vector<Batch>> batches = next.get();
      if (step + 1 < steps_per_epoch)
        next = std::async(std::launch::async, make_step, std::cref(perm), step + 1,
                          o.seed + epoch * 1000003ULL + step + 1);
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
                   "sync {:.1f} ms; {:.2f} micro-batches/worker\n",
                   epoch, step + 1, steps_per_epoch, st[0].loss, st[0].loss_energy, st[0].loss_force,
                   lr, norm, (step + 1) * W * B / el, 1e3 * t_wait / (step + 1),
                   1e3 * t_gpu / (step + 1), 1e3 * t_build / (step + 1),
                   1e3 * t_compute / (step + 1), 1e3 * t_sync / (step + 1),
                   double(n_micro) / ((step + 1) * W));
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
