#include "md/md_engine.h"

#include <cuda_runtime.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace mlipcpp::md {

namespace {

constexpr size_t GRAPH_SIZE = 65536;
constexpr double ACC = 9.64853321233100e-3; // eV / (Å amu) in Å / fs^2
constexpr double KB = 8.617333262e-5;       // eV / K

constexpr double MASSES[119] = { // ase.data.atomic_masses (IUPAC 2016), Z = 0..118
    1, 1.008, 4.0026, 6.94, 9.01218, 10.81, 12.011, 14.007, 15.999, 18.9984,
    20.1797, 22.9898, 24.305, 26.9815, 28.085, 30.9738, 32.06, 35.45, 39.948, 39.0983,
    40.078, 44.9559, 47.867, 50.9415, 51.9961, 54.938, 55.845, 58.9332, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.9216, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.9058,
    91.224, 92.9064, 95.95, 97.9072, 101.07, 102.906, 106.42, 107.868, 112.414, 114.818,
    118.71, 121.76, 127.6, 126.904, 131.293, 132.905, 137.327, 138.905, 140.116, 140.908,
    144.242, 144.913, 150.36, 151.964, 157.25, 158.925, 162.5, 164.93, 167.259, 168.934,
    173.054, 174.967, 178.49, 180.948, 183.84, 186.207, 190.23, 192.217, 195.084, 196.967,
    200.592, 204.38, 207.2, 208.98, 208.982, 209.987, 222.018, 223.02, 226.025, 227.028,
    232.038, 231.036, 238.029, 237.048, 244.064, 243.061, 247.07, 247.07, 251.08, 252.083,
    257.095, 258.098, 259.101, 262.11, 267.122, 268.126, 271.134, 270.133, 269.134, 278.156,
    281.165, 281.166, 285.177, 286.182, 289.19, 289.194, 293.204, 293.208, 294.214,
};

void check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("md: ") + what + ": " + cudaGetErrorString(e));
}

template <class T> void dalloc(T *&p, size_t n) {
  if (p) check(cudaFree(p), "free");
  p = nullptr;
  if (n) check(cudaMalloc(&p, n * sizeof(T)), "malloc");
}
template <class T> void upload(T *dst, const std::vector<T> &src) {
  check(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <class T> std::vector<T> download(const T *src, size_t n) {
  std::vector<T> out(n);
  if (n) check(cudaMemcpy(out.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return out;
}

ggml_tensor *input(ggml_context *ctx, ggml_type type, std::initializer_list<int64_t> ne, const char *name) {
  auto d = std::data(ne);
  ggml_tensor *t = ne.size() == 1   ? ggml_new_tensor_1d(ctx, type, d[0])
                   : ne.size() == 2 ? ggml_new_tensor_2d(ctx, type, d[0], d[1])
                                    : ggml_new_tensor_3d(ctx, type, d[0], d[1], d[2]);
  ggml_set_name(t, name);
  ggml_set_input(t);
  return t;
}

int round_up(int x, int m) { return (x + m - 1) / m * m; }

} // namespace

Engine::Engine(const std::string &gguf, int device, double skin) {
  // full FP32 by default: TF32 GEMMs (cuBLAS and ggml's mmf kernels) put
  // ~1e-3 relative noise on energies and forces, which heats NVE runs;
  // MLIPCPP_MD_TF32=1 keeps TF32
  if (!std::getenv("MLIPCPP_MD_TF32")) setenv("GGML_CUDA_NO_TF32", "1", 1);
  check(cudaSetDevice(device), "set device");
  model_.set_backend_preference(BackendPreference::CUDA);
  model_.set_device(device);
  model_.load_from_gguf(gguf);
  if (!model_.forces_mode()) throw std::runtime_error(gguf + ": MD needs a forces-mode export");
  if (model_.num_neighbors_adaptive() > 0 && model_.adaptive_cutoff_method() != "solver")
    throw std::runtime_error(gguf + ": MD implements the \"solver\" adaptive cutoff only, the model uses \"" +
                             model_.adaptive_cutoff_method() + "\"");
  cut_.cutoff = model_.cutoff();
  cut_.skin = skin;
  cut_.n_target = model_.num_neighbors_adaptive();
  cut_.width = model_.cutoff_width_adaptive();
  energy_scale_ = model_.energy_scale();
  galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model_.backend()));
  dalloc(st_.scalars, 3);
}

Engine::~Engine() {
  free_graph();
  gpu::free_cells(st_);
  if (galloc_) ggml_gallocr_free(galloc_);
  for (void *p : {(void *)st_.x, (void *)st_.v, (void *)st_.x_ref, (void *)st_.inv_mass, (void *)st_.species,
                  (void *)st_.nc_scale, (void *)st_.cand_j, (void *)st_.cand_rev, (void *)st_.count,
                  (void *)st_.cand_s, (void *)st_.cand_v, (void *)st_.cand_d, (void *)st_.cand_slot,
                  (void *)st_.nkept, (void *)st_.r, (void *)st_.root, (void *)st_.dn_root, (void *)st_.clamped,
                  (void *)st_.dEdr, (void *)st_.chain, (void *)st_.f_cons, (void *)st_.f_nc, (void *)st_.scalars})
    if (p) cudaFree(p);
}

void Engine::free_graph() {
  if (persist_) ggml_backend_buffer_free(persist_);
  if (flat_) ggml_backend_buffer_free(flat_);
  if (ctx_) ggml_free(ctx_);
  persist_ = nullptr, flat_ = nullptr, ctx_ = nullptr, gf_ = nullptr;
}

void Engine::build_graph(int M) {
  free_graph();
  const int n = st_.n;
  const bool adaptive = cut_.n_target > 0, nc = model_.nc_forces();
  auto &interp = model_.interpreter();
  interp.set_dimension("n_atoms", n);
  interp.set_dimension("max_neighbors", M);
  interp.set_dimension("n_edges", int64_t(n) * M);
  interp.set_dimension("max_neighbors_plus_one", M + 1);
  ctx_ = ggml_init({ggml_tensor_overhead() * GRAPH_SIZE * 2 + ggml_graph_overhead_custom(GRAPH_SIZE, true),
                    nullptr, /*no_alloc=*/true});
  std::vector<ggml_tensor *> inputs;
  auto in = [&](ggml_type t, std::initializer_list<int64_t> ne, const char *name) {
    inputs.push_back(input(ctx_, t, ne, name));
    return inputs.back();
  };
  ggml_tensor *species = in(GGML_TYPE_I32, {n}, "species");
  ggml_tensor *nspecies = in(GGML_TYPE_I32, {M, n}, "neighbor_species");
  ggml_tensor *ev = in(GGML_TYPE_F32, {3, M, n}, "edge_vectors");
  ggml_tensor *pm = in(GGML_TYPE_F32, {M, n}, "padding_mask");
  ggml_tensor *rev = in(GGML_TYPE_I32, {int64_t(n) * M}, "reverse_neighbor_index");
  ggml_tensor *cv = in(GGML_TYPE_F32, {M, n}, "cutoff_values");
  ggml_set_param(ev);
  if (adaptive) ggml_set_param(cv);
  for (ggml_tensor *t : inputs) interp.set_input(ggml_get_name(t), t);

  interp.set_input_gradients(true); // the graph is differentiated with respect to the inputs
  ggml_tensor *out = interp.build(ctx_);
  interp.set_input_gradients(false);
  if (!out) throw std::runtime_error("md: cannot build the model graph");
  // [1, n], contiguous; an output itself (a reshaped view of it would not
  // keep the allocator from reusing its memory in the backward pass)
  ggml_tensor *energy = nc ? ggml_cont(ctx_, ggml_view_2d(ctx_, out, 1, n, out->nb[1], 0)) : ggml_cont(ctx_, out);
  ggml_tensor *force = nc ? ggml_cont(ctx_, ggml_view_2d(ctx_, out, 3, n, out->nb[1], sizeof(float))) : nullptr;
  ggml_set_output(energy);
  if (force) ggml_set_output(force);
  ggml_tensor *total = ggml_sum(ctx_, energy);
  ggml_set_loss(total);

  gf_ = ggml_new_graph_custom(ctx_, GRAPH_SIZE, true);
  ggml_build_forward_expand(gf_, energy);
  if (force) ggml_build_forward_expand(gf_, force);
  ggml_build_forward_expand(gf_, total);
  const int n_forward = ggml_graph_n_nodes(gf_);
  ggml_build_backward_expand(ctx_, gf_, nullptr);
  ggml_tensor *loss_grad = ggml_graph_get_grad_acc(gf_, total);
  ggml_tensor *gev = ggml_graph_get_grad(gf_, ev), *gcv = adaptive ? ggml_graph_get_grad(gf_, cv) : nullptr;
  if (!gev) throw std::runtime_error("md: the energy does not depend on the edge vectors");
  ggml_set_output(gev);
  if (gcv) ggml_set_output(gcv);

  auto constants = interp.take_constants();
  constants.emplace_back(loss_grad, 1.0f);
  for (auto &c : constants) ggml_set_input(c.first);
  // inputs and constants in a buffer of their own (the allocator skips
  // tensors that already have memory), the rest through the allocator
  ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(model_.backend());
  const size_t align = ggml_backend_buft_get_alignment(buft);
  size_t size = 0;
  std::vector<std::pair<ggml_tensor *, size_t>> place;
  auto reserve = [&](ggml_tensor *t) {
    place.emplace_back(t, size);
    size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
  };
  for (ggml_tensor *t : inputs) reserve(t);
  for (auto &c : constants) reserve(c.first);
  persist_ = ggml_backend_buft_alloc_buffer(buft, size);
  if (!persist_) throw std::runtime_error("md: cannot allocate the graph inputs");
  char *base = static_cast<char *>(ggml_backend_buffer_get_base(persist_));
  for (auto &[t, off] : place) ggml_backend_tensor_alloc(persist_, t, base + off);
  // MLIPCPP_MD_NO_REUSE: every tensor gets its own memory (A/B of the
  // allocator's in-place reuse)
  static const bool no_reuse = std::getenv("MLIPCPP_MD_NO_REUSE") != nullptr;
  if (no_reuse) {
    if (!(flat_ = ggml_backend_alloc_ctx_tensors(ctx_, model_.backend())))
      throw std::runtime_error("md: cannot allocate the graph without reuse");
  } else if (!ggml_gallocr_alloc_graph(galloc_, gf_))
    throw std::runtime_error("md: cannot allocate the graph (" + std::to_string(n) + " atoms x " +
                             std::to_string(M) + " neighbours)");
  ggml_backend_buffer_clear(persist_, 0);
  std::vector<float> staging;
  for (auto &[t, v] : constants) {
    staging.assign(ggml_nelements(t), v);
    ggml_backend_tensor_set(t, staging.data(), 0, ggml_nbytes(t));
  }
  fwd_ = ggml_graph_view(gf_, 0, n_forward);

  g_.species = static_cast<int32_t *>(species->data);
  g_.neighbor_species = static_cast<int32_t *>(nspecies->data);
  g_.edge_vectors = static_cast<float *>(ev->data);
  g_.padding_mask = static_cast<float *>(pm->data);
  g_.reverse = static_cast<int32_t *>(rev->data);
  g_.cutoff_values = static_cast<float *>(cv->data);
  g_.energy = static_cast<const float *>(energy->data);
  g_.nc = force ? static_cast<const float *>(force->data) : nullptr;
  g_.grad_ev = static_cast<const float *>(gev->data);
  g_.grad_cv = gcv ? static_cast<const float *>(gcv->data) : nullptr;
  energy_t_ = energy;
  st_.M = M;
  graph_builds_++;
}

void Engine::alloc_atoms() {
  const size_t n = st_.n;
  dalloc(st_.x, 3 * n), dalloc(st_.v, 3 * n), dalloc(st_.x_ref, 3 * n), dalloc(st_.inv_mass, n);
  dalloc(st_.species, n), dalloc(st_.nc_scale, n), dalloc(st_.count, n), dalloc(st_.nkept, n);
  dalloc(st_.r, n), dalloc(st_.root, n), dalloc(st_.dn_root, n), dalloc(st_.clamped, n), dalloc(st_.dEdr, n);
  dalloc(st_.f_cons, 3 * n), dalloc(st_.f_nc, 3 * n);
  check(cudaMemset(st_.f_cons, 0, 3 * n * sizeof(double)), "memset");
  check(cudaMemset(st_.f_nc, 0, 3 * n * sizeof(double)), "memset");
}

void Engine::alloc_candidates(int K) {
  st_.K = K;
  const size_t E = size_t(st_.n) * K;
  dalloc(st_.cand_j, E), dalloc(st_.cand_rev, E), dalloc(st_.cand_s, E), dalloc(st_.cand_v, 3 * E);
  dalloc(st_.cand_d, E), dalloc(st_.cand_slot, E), dalloc(st_.chain, 3 * E);
}

void Engine::set_system(const AtomicSystem &system) {
  const int n = system.num_atoms();
  if (n < 2) throw std::runtime_error("md: need at least two atoms");
  st_.n = n;
  Z_.assign(system.atomic_numbers(), system.atomic_numbers() + n);
  std::vector<double> inv_mass(n);
  std::vector<int32_t> species(n);
  std::vector<float> nc_scale(n, 1.0f);
  composition_.assign(n, 0.0);
  for (int i = 0; i < n; i++) {
    const int z = Z_[i];
    auto s = model_.species_to_index().find(z);
    if (s == model_.species_to_index().end() || z <= 0 || z > 118)
      throw std::runtime_error("md: atomic number " + std::to_string(z) + " is not in the model");
    species[i] = s->second;
    inv_mass[i] = 1.0 / MASSES[z];
    if (auto c = model_.composition_energies().find(z); c != model_.composition_energies().end())
      composition_[i] = c->second;
    if (auto f = model_.nc_force_scales().find(z); f != model_.nc_force_scales().end()) nc_scale[i] = f->second;
  }
  std::fill(std::begin(st_.cell), std::end(st_.cell), 0.0);
  std::fill(std::begin(st_.pbc), std::end(st_.pbc), 0);
  std::fill(std::begin(st_.images), std::end(st_.images), 0);
  if (const Cell *c = system.cell()) {
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) st_.cell[3 * a + b] = c->matrix[a][b];
      st_.pbc[a] = c->periodic[a];
    }
    const double *m = st_.cell;
    auto cross = [&](int a, int b, double out[3]) {
      out[0] = m[3 * a + 1] * m[3 * b + 2] - m[3 * a + 2] * m[3 * b + 1];
      out[1] = m[3 * a + 2] * m[3 * b] - m[3 * a] * m[3 * b + 2];
      out[2] = m[3 * a] * m[3 * b + 1] - m[3 * a + 1] * m[3 * b];
    };
    double c12[3];
    cross(1, 2, c12);
    const double vol = std::abs(m[0] * c12[0] + m[1] * c12[1] + m[2] * c12[2]);
    for (int a = 0; a < 3; a++) {
      if (!st_.pbc[a]) continue;
      double w[3];
      cross((a + 1) % 3, (a + 2) % 3, w);
      const double height = vol / std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
      st_.images[a] = static_cast<int>(std::ceil((cut_.cutoff + cut_.skin) / height));
      if (st_.images[a] > 127) throw std::runtime_error("md: cell too thin for the cutoff");
    }
  }
  alloc_atoms();
  upload(st_.inv_mass, inv_mass), upload(st_.species, species), upload(st_.nc_scale, nc_scale);
  std::vector<double> x(3 * n);
  for (int k = 0; k < 3 * n; k++) x[k] = system.positions()[k];
  upload(st_.x, x);
  check(cudaMemset(st_.v, 0, 3 * n * sizeof(double)), "memset");
  if (st_.K == 0 || !st_.cand_j) alloc_candidates(64);
  else alloc_candidates(st_.K);
  free_graph();
  st_.M = 0;
  build_list();
}

void Engine::set_positions(const std::vector<double> &x) {
  if (x.size() != size_t(3 * st_.n)) throw std::runtime_error("md: positions size");
  upload(st_.x, x);
  build_list();
}

void Engine::set_velocities(const std::vector<double> &v) {
  if (v.size() != size_t(3 * st_.n)) throw std::runtime_error("md: velocities size");
  upload(st_.v, v);
}

void Engine::build_list() {
  for (;;) {
    gpu::reset_scalars(st_);
    gpu::build_list(st_, cut_);
    const int need = download(st_.scalars, 1)[0];
    if (need == 0) break;
    alloc_candidates(need + need / 4 + 8);
  }
  list_builds_++;
}

void Engine::maybe_rebuild_list() {
  const int32_t bits = download(st_.scalars + 2, 1)[0];
  float d2;
  std::memcpy(&d2, &bits, sizeof d2);
  if (d2 > 0.25 * cut_.skin * cut_.skin) build_list();
}

void Engine::prepare() {
  if (!ctx_) build_graph(16);
  for (;;) {
    gpu::prepare_inputs(st_, cut_, g_);
    max_kept_ = download(st_.scalars + 1, 1)[0];
    // MLIPCPP_MD_PAD: padded slots beyond the largest neighbour count (default 4)
    static const int pad = [] { const char *e = std::getenv("MLIPCPP_MD_PAD"); return e ? std::atoi(e) : 4; }();
    const int want = pad == 0 ? std::max(max_kept_, 1) : round_up(std::max(max_kept_, 1) + pad, 4);
    // grow at once; shrink only with a margin (each build costs a few ms)
    if (max_kept_ > st_.M || want + 8 < st_.M || (pad == 0 && want != st_.M)) build_graph(want);
    else break;
  }
}

double Engine::evaluate(bool conservative, bool want_energy) {
  prepare();
  if (!model_.nc_forces() && !conservative) throw std::runtime_error("md: the model has no direct force head");
  if (ggml_backend_graph_compute(model_.backend(), conservative ? gf_ : &fwd_) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("md: graph computation failed");
  if (model_.nc_forces()) gpu::nc_forces(st_, g_);
  if (conservative) gpu::conservative_forces(st_, cut_, g_, energy_scale_);
  if (!want_energy) return std::numeric_limits<double>::quiet_NaN();
  std::vector<float> e(st_.n);
  ggml_backend_tensor_get(energy_t_, e.data(), 0, e.size() * sizeof(float));
  double model = 0, comp = 0;
  for (int i = 0; i < st_.n; i++) model += e[i], comp += composition_[i];
  return energy_ = energy_scale_ * model + comp;
}

void Engine::step(Integrator integ, double dt, int inner, bool want_energy) {
  auto move = [&](double h) {
    check(cudaMemset(st_.scalars + 2, 0, sizeof(int32_t)), "memset");
    gpu::drift(st_, h);
    maybe_rebuild_list();
  };
  switch (integ) {
  case Integrator::Conservative:
  case Integrator::Direct: {
    const bool cons = integ == Integrator::Conservative;
    const double *f = cons ? st_.f_cons : st_.f_nc;
    gpu::kick(st_, f, 0.5, nullptr, 0, dt);
    move(dt);
    evaluate(cons, want_energy);
    gpu::kick(st_, f, 0.5, nullptr, 0, dt);
    break;
  }
  case Integrator::Respa: {
    // outer kicks with the slow force F_cons - F_nc, inner velocity Verlet on F_nc
    const double outer = inner * dt;
    gpu::kick(st_, st_.f_cons, 0.5, st_.f_nc, -0.5, outer);
    for (int m = 0; m < inner; m++) {
      gpu::kick(st_, st_.f_nc, 0.5, nullptr, 0, dt);
      move(dt);
      const bool last = m == inner - 1;
      evaluate(last, want_energy && last);
      gpu::kick(st_, st_.f_nc, 0.5, nullptr, 0, dt);
    }
    gpu::kick(st_, st_.f_cons, 0.5, st_.f_nc, -0.5, outer);
    break;
  }
  }
}

void Engine::maxwell_boltzmann(double target, uint64_t seed) {
  const int n = st_.n;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal;
  std::vector<double> v(3 * n), p(3, 0.0);
  double mtot = 0;
  for (int i = 0; i < n; i++) {
    const double m = MASSES[Z_[i]], s = std::sqrt(KB * target * ACC / m);
    for (int c = 0; c < 3; c++) v[3 * i + c] = s * normal(rng), p[c] += m * v[3 * i + c];
    mtot += m;
  }
  for (int i = 0; i < n; i++)
    for (int c = 0; c < 3; c++) v[3 * i + c] -= p[c] / mtot;
  set_velocities(v);
  const double t = temperature();
  if (t > 0)
    for (double &x : v) x *= std::sqrt(target / t);
  set_velocities(v);
}

double Engine::kinetic_energy() const {
  const auto v = velocities();
  double ke = 0;
  for (int i = 0; i < st_.n; i++)
    for (int c = 0; c < 3; c++) ke += 0.5 * MASSES[Z_[i]] * v[3 * i + c] * v[3 * i + c];
  return ke / ACC;
}

std::vector<int32_t> Engine::kept_per_atom() const { return download(st_.nkept, st_.n); }

Engine::Inputs Engine::inputs() const {
  const size_t n = st_.n, s = n * st_.M;
  return {st_.M, download(g_.species, n), download(g_.neighbor_species, s), download(g_.reverse, s),
          download(g_.edge_vectors, 3 * s), download(static_cast<const float *>(g_.padding_mask), s),
          download(static_cast<const float *>(g_.cutoff_values), s)};
}

int64_t Engine::n_edges() const {
  int64_t e = 0;
  for (int32_t k : download(st_.nkept, st_.n)) e += k;
  return e;
}

double Engine::temperature() const { return 2.0 * kinetic_energy() / ((3.0 * st_.n - 3.0) * KB); }

std::vector<double> Engine::positions() const { return download(st_.x, 3 * st_.n); }
std::vector<double> Engine::velocities() const { return download(st_.v, 3 * st_.n); }
std::vector<double> Engine::forces_conservative() const { return download(st_.f_cons, 3 * st_.n); }
std::vector<double> Engine::forces_direct() const { return download(st_.f_nc, 3 * st_.n); }

} // namespace mlipcpp::md
