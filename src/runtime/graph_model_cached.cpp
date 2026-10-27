// GraphModel::predict_cached: forces-mode PET graphs, built once per shape.
//
// Host side, per call: a full neighbour list (vesin when available), each
// atom's edges sorted by (neighbour, shift) so the slot order is
// deterministic, the edge vectors evaluated once per pair (i -> j and j -> i
// are exact negatives), the "solver" adaptive radii, the kept edges packed
// into [n_atoms, M] slots, and the forces / virial gathered per atom from the
// edge-vector gradients (no scatter). The per-atom loops run on OpenMP
// threads. The same layouts as the CUDA MD kernels (src/md/md_kernels.cu).

#include "graph_cache.h"
#include "graph_model.h"
#include "core/adaptive_cutoff.h"
#include "core/padding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#ifdef MLIPCPP_HAVE_VESIN
#include <vesin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef MLIPCPP_HAVE_CUDA_INPUTS
#include "md/md_kernels.h"
#include <cuda_runtime.h>
#include <ggml-backend-impl.h> // ggml_backend_event::context: the cudaEvent_t of a CUDA backend's event
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

namespace mlipcpp::runtime {

namespace {

constexpr size_t GRAPH_SIZE = 65536;

// atom i's edges are [start[i], start[i + 1]), sorted by (j, shift)
struct HostEdges {
  std::vector<int64_t> start;
  std::vector<int32_t> j, rev; // rev: index of j -> i with the opposite shift
  std::vector<std::array<int32_t, 3>> shift;
  std::vector<double> v, d; // [E, 3], [E]: x_j - x_i + shift . cell, |v| + 1e-15
};

int env_int(const char *name, int fallback) {
  const char *e = std::getenv(name);
  return e && *e ? std::atoi(e) : fallback;
}

bool primary(int i, int j, const std::array<int32_t, 3> &s) {
  if (i != j) return i < j;
  for (int c = 0; c < 3; c++)
    if (s[c] != 0) return s[c] > 0;
  return false;
}

#ifdef MLIPCPP_HAVE_CUDA_INPUTS
void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("GraphModel (CUDA): ") + what + ": " + cudaGetErrorString(e));
}
template <class T> void dalloc(T *&p, size_t count) {
  if (p) cudaFree(p);
  p = nullptr;
  if (count) cuda_check(cudaMalloc(&p, count * sizeof(T)), "malloc");
}
template <class T> void upload(T *device, const T *host, size_t count) {
  cuda_check(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <class T> void download(T *host, const T *device, size_t count) {
  cuda_check(cudaMemcpy(host, device, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
}

// predict_cached_device's buffers: the MD engine's state without the
// integrator, grown as needed and kept between calls
struct DeviceInputs {
  md::gpu::State st;
  size_t n_cap = 0, e_cap = 0;
  double *virial = nullptr; // [n, 6]
  // what the candidate list was built for: it is kept while these match and no
  // atom moved more than half the skin (md::gpu::list_valid)
  bool list_built = false;
  std::vector<int32_t> list_species;
  double list_cell[9] = {0};
  int list_pbc[3] = {0, 0, 0};

  // predict_device_step
  double *shift = nullptr;             // [n, 3]: what build_list's wrap added to the caller's positions
  int32_t *flags = nullptr;            // [4]: graph overflow, bad type, step good
  double *out = nullptr, *out_host = nullptr; // [16] the totals (out_host pinned)
  int32_t *type_species = nullptr;     // [n_types + 1]
  double *type_energy = nullptr;       // [n_types + 1] composition energy
  std::vector<int32_t> type_z;         // what type_species was made from
  int type_cap = 0;
  double list_skin = -1;
  bool have_sizes = false;             // last_total / last_max_kept from an earlier call
  int64_t last_total = 0;
  int last_max_kept = 0;
  ggml_backend_event_t ev_in = nullptr, ev_out = nullptr;

  void atoms(int n) {
    st.n = n;
    if (size_t(n) > n_cap) {
      n_cap = n;
      dalloc(st.x, 3 * n_cap), dalloc(st.x_ref, 3 * n_cap), dalloc(st.species, n_cap);
      dalloc(st.count, n_cap), dalloc(st.nkept, n_cap), dalloc(st.r, n_cap), dalloc(st.root, n_cap);
      dalloc(st.dn_root, n_cap), dalloc(st.clamped, n_cap), dalloc(st.dEdr, n_cap);
      dalloc(st.f_cons, 3 * n_cap), dalloc(virial, 6 * n_cap), dalloc(st.offsets, n_cap + 1);
      dalloc(shift, 3 * n_cap);
    }
    if (!st.scalars) dalloc(st.scalars, 3);
    candidates(st.K ? st.K : 64);
  }
  void candidates(int K) {
    st.K = K;
    const size_t E = size_t(st.n) * K;
    if (E <= e_cap) return;
    e_cap = E;
    dalloc(st.cand_j, E), dalloc(st.cand_rev, E), dalloc(st.cand_s, E), dalloc(st.cand_v, 3 * E);
    dalloc(st.cand_d, E), dalloc(st.cand_slot, E), dalloc(st.chain, 3 * E);
  }
  ~DeviceInputs() {
    md::gpu::free_cells(st);
    for (void *p : {(void *)shift, (void *)flags, (void *)out, (void *)type_species, (void *)type_energy})
      if (p) cudaFree(p);
    if (out_host) cudaFreeHost(out_host);
    if (ev_in) ggml_backend_event_free(ev_in);
    if (ev_out) ggml_backend_event_free(ev_out);
    for (void *p : {(void *)st.x, (void *)st.x_ref, (void *)st.species, (void *)st.count, (void *)st.nkept,
                    (void *)st.r, (void *)st.root, (void *)st.dn_root, (void *)st.clamped, (void *)st.dEdr,
                    (void *)st.f_cons, (void *)virial, (void *)st.offsets, (void *)st.scalars, (void *)st.cand_j, (void *)st.cand_rev,
                    (void *)st.cand_s, (void *)st.cand_v, (void *)st.cand_d, (void *)st.cand_slot, (void *)st.chain})
      if (p) cudaFree(p);
  }
};

// st's cell, periodicity and the periodic images within `reach` (as md::Engine::set_system); no cell: none
void set_device_cell(md::gpu::State &st, const double *cell, const int *pbc, double reach) {
  std::fill(std::begin(st.cell), std::end(st.cell), 0.0);
  std::fill(std::begin(st.pbc), std::end(st.pbc), 0);
  std::fill(std::begin(st.images), std::end(st.images), 0);
  if (!cell) return;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) st.cell[3 * a + b] = cell[3 * a + b];
    st.pbc[a] = pbc[a];
  }
  const double *m = st.cell;
  auto cross = [&](int a, int b, double out[3]) {
    out[0] = m[3 * a + 1] * m[3 * b + 2] - m[3 * a + 2] * m[3 * b + 1];
    out[1] = m[3 * a + 2] * m[3 * b] - m[3 * a] * m[3 * b + 2];
    out[2] = m[3 * a] * m[3 * b + 1] - m[3 * a + 1] * m[3 * b];
  };
  double c12[3];
  cross(1, 2, c12);
  const double volume = std::abs(m[0] * c12[0] + m[1] * c12[1] + m[2] * c12[2]);
  for (int a = 0; a < 3; a++) {
    if (!st.pbc[a]) continue;
    double w[3];
    cross((a + 1) % 3, (a + 2) % 3, w);
    const double height = volume / std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    st.images[a] = static_cast<int>(std::ceil(reach / height));
    if (st.images[a] > 127) throw std::runtime_error("GraphModel: cell too thin for the cutoff");
  }
}

md::gpu::Graph device_graph(const CachedGraph &g) {
  auto ptr = [](ggml_tensor *t) { return t ? t->data : nullptr; };
  md::gpu::Graph d;
  d.species = static_cast<int32_t *>(ptr(g.species));
  d.neighbor_species = static_cast<int32_t *>(ptr(g.neighbor_species));
  d.reverse = static_cast<int32_t *>(ptr(g.reverse));
  d.edge_vectors = static_cast<float *>(ptr(g.edge_vectors));
  d.padding_mask = static_cast<float *>(ptr(g.padding_mask));
  d.cutoff_values = static_cast<float *>(ptr(g.cutoff_values));
  d.energy = static_cast<const float *>(ptr(g.energy));
  d.nc = static_cast<const float *>(ptr(g.nc));
  d.grad_ev = static_cast<const float *>(ptr(g.grad_ev));
  d.grad_cv = static_cast<const float *>(ptr(g.grad_cv));
  d.offsets = static_cast<int32_t *>(ptr(g.offsets));
  d.centers = static_cast<int32_t *>(ptr(g.centers));
  d.e_cap = g.e_cap;
  return d;
}
#endif

// stress = sum of the per-atom virials / volume (nothing for a flat cell)
void set_stress(const Cell &cell, double scale, const std::array<double, 6> &w, ModelResult &result) {
  const auto &m = cell.matrix;
  const double volume = std::abs(double(m[0][0]) * (double(m[1][1]) * m[2][2] - double(m[1][2]) * m[2][1]) -
                                 double(m[0][1]) * (double(m[1][0]) * m[2][2] - double(m[1][2]) * m[2][0]) +
                                 double(m[0][2]) * (double(m[1][0]) * m[2][1] - double(m[1][1]) * m[2][0]));
  if (volume <= 1e-10) return;
  result.stress.resize(6);
  for (int c = 0; c < 6; c++) result.stress[c] = static_cast<float>(scale * w[c] / volume);
  result.has_stress = true;
}

} // namespace

void GraphModel::CachedGraphDeleter::operator()(CachedGraph *graph) const { delete graph; }

void GraphModel::release_cached_resources() {
  if (galloc_) ggml_gallocr_free(galloc_);
  galloc_ = nullptr;
#ifdef MLIPCPP_HAVE_VESIN
  if (vesin_) {
    auto *list = static_cast<VesinNeighborList *>(vesin_);
    vesin_free(list);
    delete list;
  }
#endif
  vesin_ = nullptr;
#ifdef MLIPCPP_HAVE_CUDA_INPUTS
  delete static_cast<DeviceInputs *>(device_inputs_);
#endif
  device_inputs_ = nullptr;
}

bool GraphModel::cached_path_supported() const {
  return forces_mode_ && !long_range_ &&
         (num_neighbors_adaptive_ <= 0.0f || adaptive_cutoff_method_ == "solver");
}

void GraphModel::build_cached_graph(int n, int M, bool exact_mask, bool backward) {
  cached_.reset();
  auto g = std::unique_ptr<CachedGraph, CachedGraphDeleter>(new CachedGraph());
  g->n_atoms = n, g->M = M, g->exact_mask = exact_mask, g->backward = backward;
  const bool adaptive = num_neighbors_adaptive_ > 0.0f;

  interp_.set_dimension("n_atoms", n);
  interp_.set_dimension("max_neighbors", M);
  interp_.set_dimension("n_edges", int64_t(n) * M);
  interp_.set_dimension("max_neighbors_plus_one", M + 1);
  interp_.set_exact_padding_mask(exact_mask);
  static const bool no_fuse = std::getenv("MLIPCPP_NO_FUSED_GLU") != nullptr;
  const std::string_view backend = ggml_backend_name(compute_backend_);
  interp_.set_fuse_glu(!no_fuse && (backend.rfind("CUDA", 0) == 0 || backend.rfind("CPU", 0) == 0));
  g->ctx = ggml_init({ggml_tensor_overhead() * GRAPH_SIZE * 2 + ggml_graph_overhead_custom(GRAPH_SIZE, true) +
                          ggml_graph_overhead_custom(GRAPH_SIZE, false),
                      nullptr, /*no_alloc=*/true});
  if (!g->ctx) throw std::runtime_error("GraphModel: cannot create the graph context");
  ggml_context *ctx = g->ctx;

  std::vector<ggml_tensor *> inputs;
  auto input = [&](ggml_type type, std::initializer_list<int64_t> ne, const char *name) {
    const int64_t *d = std::data(ne);
    ggml_tensor *t = ne.size() == 1   ? ggml_new_tensor_1d(ctx, type, d[0])
                     : ne.size() == 2 ? ggml_new_tensor_2d(ctx, type, d[0], d[1])
                                      : ggml_new_tensor_3d(ctx, type, d[0], d[1], d[2]);
    ggml_set_name(t, name);
    ggml_set_input(t);
    interp_.set_input(name, t);
    inputs.push_back(t);
    return t;
  };
  g->species = input(GGML_TYPE_I32, {n}, "species");
  g->neighbor_species = input(GGML_TYPE_I32, {M, n}, "neighbor_species");
  g->edge_vectors = input(GGML_TYPE_F32, {3, M, n}, "edge_vectors");
  g->padding_mask = input(GGML_TYPE_F32, {M, n}, "padding_mask");
  g->reverse = input(GGML_TYPE_I32, {int64_t(n) * M}, "reverse_neighbor_index");
  g->cutoff_values = input(GGML_TYPE_F32, {M, n}, "cutoff_values");
  if (conditioning_.enabled) {
    g->charge_index = input(GGML_TYPE_I32, {n}, "charge_index");
    g->spin_index = input(GGML_TYPE_I32, {n}, "spin_index");
  }
  if (backward) {
    ggml_set_param(g->edge_vectors);
    if (adaptive) ggml_set_param(g->cutoff_values);
  }

  interp_.coulomb_application = {};
  interp_.set_input_gradients(backward);
  ggml_tensor *out = nullptr;
  try {
    out = interp_.build(ctx);
  } catch (...) {
    interp_.set_input_gradients(false);
    interp_.set_fuse_glu(false);
    throw;
  }
  interp_.set_input_gradients(false);
  interp_.set_fuse_glu(false);
  if (!out) throw std::runtime_error("GraphModel: cannot build the model graph");
  // [1, n] contiguous outputs (a view of `out` would not keep the allocator
  // from reusing its memory in the backward pass)
  g->energy = nc_forces_ ? ggml_cont(ctx, ggml_view_2d(ctx, out, 1, n, out->nb[1], 0)) : ggml_cont(ctx, out);
  g->nc = nc_forces_ ? ggml_cont(ctx, ggml_view_2d(ctx, out, 3, n, out->nb[1], sizeof(float))) : nullptr;
  ggml_set_output(g->energy);
  if (g->nc) ggml_set_output(g->nc);

  g->graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, backward);
  ggml_build_forward_expand(g->graph, g->energy);
  if (g->nc) ggml_build_forward_expand(g->graph, g->nc);
  // the same expansion as the start of `graph`, so its allocation holds
  g->forward = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
  ggml_build_forward_expand(g->forward, g->energy);
  if (g->nc) ggml_build_forward_expand(g->forward, g->nc);
  auto constants = std::vector<std::pair<ggml_tensor *, float>>();
  if (backward) {
    ggml_tensor *total = g->loss = ggml_sum(ctx, g->energy);
    ggml_set_loss(total);
    ggml_build_forward_expand(g->graph, total);
    ggml_build_backward_expand(ctx, g->graph, nullptr);
    g->grad_ev = ggml_graph_get_grad(g->graph, g->edge_vectors);
    g->grad_cv = adaptive ? ggml_graph_get_grad(g->graph, g->cutoff_values) : nullptr;
    if (!g->grad_ev) throw std::runtime_error("GraphModel: the energy does not depend on the edge vectors");
    ggml_set_output(g->grad_ev);
    if (g->grad_cv) ggml_set_output(g->grad_cv);
    constants.emplace_back(ggml_graph_get_grad_acc(g->graph, total), 1.0f);
  }
  for (auto &c : interp_.take_constants()) constants.push_back(c);
  finish_cached_graph(std::move(g), inputs, std::move(constants));
}

void GraphModel::finish_cached_graph(std::unique_ptr<CachedGraph, CachedGraphDeleter> g,
                                     const std::vector<ggml_tensor *> &inputs,
                                     std::vector<std::pair<ggml_tensor *, float>> constants) {
  const int n = g->n_atoms, M = g->ragged ? g->max_len : g->M;
  for (auto &c : constants) ggml_set_input(c.first);

  // inputs and constants in a buffer of their own (the allocator skips
  // tensors that already have memory), the rest through the allocator
  ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(compute_backend_);
  const size_t align = ggml_backend_buft_get_alignment(buft);
  size_t size = 0;
  std::vector<std::pair<ggml_tensor *, size_t>> place;
  auto reserve = [&](ggml_tensor *t) {
    place.emplace_back(t, size);
    size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
  };
  for (ggml_tensor *t : inputs) reserve(t);
  for (auto &c : constants) reserve(c.first);
  g->inputs = ggml_backend_buft_alloc_buffer(buft, size);
  if (!g->inputs) throw std::runtime_error("GraphModel: cannot allocate the graph inputs");
  char *base = static_cast<char *>(ggml_backend_buffer_get_base(g->inputs));
  for (auto &[t, offset] : place) ggml_backend_tensor_alloc(g->inputs, t, base + offset);
  if (!galloc_) galloc_ = ggml_gallocr_new(buft);
  if (!ggml_gallocr_alloc_graph(galloc_, g->graph))
    throw std::runtime_error("GraphModel: cannot allocate the graph (" + std::to_string(n) + " atoms x " +
                             std::to_string(M) + " neighbours)");
  ggml_backend_buffer_clear(g->inputs, 0);
  if (std::getenv("MLIPCPP_PROFILE_GRAPH")) { // matrix-multiply work of the whole graph (2 K per output element)
    double mm = 0, op = 0;
    int n_mm = 0, n_op = 0;
    for (int i = 0; i < ggml_graph_n_nodes(g->graph); i++) {
      const ggml_tensor *t = ggml_graph_node(g->graph, i);
      if (t->op == GGML_OP_MUL_MAT) mm += 2.0 * t->src[0]->ne[0] * ggml_nelements(t), n_mm++;
      if (t->op == GGML_OP_OUT_PROD) op += 2.0 * t->src[0]->ne[1] * ggml_nelements(t), n_op++;
    }
    std::fprintf(stderr, "GRAPH_FLOPS n=%d M=%d nodes=%d mul_mat=%d (%.1f GFLOP) out_prod=%d (%.1f GFLOP)\n", n, M,
                 ggml_graph_n_nodes(g->graph), n_mm, mm / 1e9, n_op, op / 1e9);
  }
  std::vector<float> staging;
  for (auto &[t, value] : constants) {
    staging.assign(ggml_nelements(t), value);
    ggml_backend_tensor_set(t, staging.data(), 0, ggml_nbytes(t));
  }
  cached_ = std::move(g);
}

namespace {

HostEdges host_edges(const AtomicSystem &system, double cutoff, int n_threads, void *&vesin_state) {
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

  // raw full list: centre, neighbour, shift
  std::vector<int32_t> pi, pj;
  std::vector<std::array<int32_t, 3>> ps;
#ifdef MLIPCPP_HAVE_VESIN
  {
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
    options.n_threads = n_threads;
    options.return_shifts = true;
    const char *error = nullptr;
    if (vesin_neighbors(reinterpret_cast<const double(*)[3]>(points.data()), n, H, periodic,
                        VesinDevice{VesinCPU, 0}, options, list, &error) != 0)
      throw std::runtime_error(std::string("vesin: ") + (error ? error : "unknown error"));
    pi.resize(list->length), pj.resize(list->length), ps.resize(list->length);
    for (size_t e = 0; e < list->length; e++) {
      pi[e] = static_cast<int32_t>(list->pairs[e][0]);
      pj[e] = static_cast<int32_t>(list->pairs[e][1]);
      ps[e] = {list->shifts[e][0], list->shifts[e][1], list->shifts[e][2]};
    }
  }
#else
  {
    (void)vesin_state;
    NeighborList list = NeighborListBuilder(NeighborListOptions{static_cast<float>(cutoff), true, false}).build(system);
    pi = list.centers, pj = list.neighbors;
    ps = list.cell_shifts.empty() ? std::vector<std::array<int32_t, 3>>(pi.size(), {0, 0, 0}) : list.cell_shifts;
  }
#endif

  // group by centre (counting sort), then sort each atom's edges
  HostEdges out;
  const size_t E = pi.size();
  out.start.assign(n + 1, 0);
  for (int32_t i : pi) out.start[i + 1]++;
  for (int i = 0; i < n; i++) out.start[i + 1] += out.start[i];
  out.j.resize(E), out.shift.resize(E), out.rev.assign(E, -1), out.v.resize(3 * E), out.d.resize(E);
  {
    std::vector<int64_t> fill(out.start.begin(), out.start.end() - 1);
    for (size_t e = 0; e < E; e++) {
      const int64_t k = fill[pi[e]]++;
      out.j[k] = pj[e], out.shift[k] = ps[e];
    }
  }
  auto key = [&](int64_t e) { return std::tuple(out.j[e], out.shift[e][0], out.shift[e][1], out.shift[e][2]); };
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 32)
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
  // vectors (evaluated for the primary orientation of each pair, so that
  // i -> j and j -> i are exact negatives) and reverse edges
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 32)
  for (int i = 0; i < n; i++)
    for (int64_t e = out.start[i]; e < out.start[i + 1]; e++) {
      const int j = out.j[e];
      const auto &s = out.shift[e];
      const bool p = primary(i, j, s);
      const int a = p ? i : j, b = p ? j : i;
      const double sg = p ? 1.0 : -1.0;
      double v2 = 0;
      for (int c = 0; c < 3; c++) {
        const double t = sg * s[0] * H[0][c] + sg * s[1] * H[1][c] + sg * s[2] * H[2][c];
        const double v = sg * (double(x[3 * b + c]) - double(x[3 * a + c]) + t);
        out.v[3 * e + c] = v, v2 += v * v;
      }
      out.d[e] = std::sqrt(v2) + 1e-15; // as in training (train/dataset.cpp)
      const auto want = std::tuple(i, -s[0], -s[1], -s[2]);
      int64_t lo = out.start[j], hi = out.start[j + 1];
      while (lo < hi) {
        const int64_t mid = (lo + hi) / 2;
        if (key(mid) < want) lo = mid + 1;
        else hi = mid;
      }
      if (lo < out.start[j + 1] && key(lo) == want) out.rev[e] = static_cast<int32_t>(lo);
    }
  return out;
}

} // namespace

bool GraphModel::fit_cached_graph(int n, int max_kept) {
  // A system without any edge: see predict_single (exact padding mask, no
  // backward pass, zero forces and stress)
  const bool no_edges = max_kept == 0;
  const bool exact = no_edges && !std::getenv("MLIPCPP_PAD_EMPTY");
  const bool backward = !no_edges;
  // neighbour slots beyond the largest count (exact by default on the CPU,
  // where every slot costs compute; on a GPU a few spare slots avoid
  // rebuilding the graph whenever the largest count moves by one)
  static const int pad_env = env_int("MLIPCPP_PAD_NEIGHBORS", -1);
  const int pad = pad_env >= 0 ? pad_env : (backend_provider_->is_gpu() ? 4 : 0);
  const int need = std::max(max_kept, 1);
  if (cached_ && cached_->n_atoms == n && cached_->exact_mask == exact && cached_->backward == backward &&
      need <= cached_->M && cached_->M <= need + 2 * pad)
    return false;
  build_cached_graph(n, need + pad, exact, backward);
  return true;
}

void GraphModel::upload_conditioning(CachedGraph &g, int charge, int spin_multiplicity) const {
  if (!g.charge_index) return;
  const auto [ci, si] = conditioning_.rows(charge, spin_multiplicity);
  if (g.cond_set && g.cond_charge == charge && g.cond_spin == spin_multiplicity) return;
  std::vector<int32_t> c(g.n_atoms, ci), s(g.n_atoms, si);
  ggml_backend_tensor_set(g.charge_index, c.data(), 0, ggml_nbytes(g.charge_index));
  ggml_backend_tensor_set(g.spin_index, s.data(), 0, ggml_nbytes(g.spin_index));
  g.cond_set = true, g.cond_charge = charge, g.cond_spin = spin_multiplicity;
}

bool GraphModel::use_ragged() {
  if (ragged_state_ >= 0) return ragged_state_ == 1;
  ragged_state_ = 0;
  const char *e = std::getenv("MLIPCPP_RAGGED");
  if (!e || !*e || *e == '0') return false;
  std::string why;
  if (conditioning_.enabled) {
    why = "does not implement charge / spin conditioning";
  } else if (cutoff_function_ != "bump" || nc_forces_ || long_range_ || !forces_mode_) {
    why = "needs a forces-mode model with the bump cutoff, no direct forces and no long range";
  } else {
    ragged_weights_.clear();
    for (auto &[name, t] : weights()) ragged_weights_[name] = t;
    int heads = num_heads_;
    auto qkv = ragged_weights_.find("gnn_layers_0_trans_layers_0_attention_input_linear_weight");
    const int64_t d_pet = qkv == ragged_weights_.end() ? 0 : qkv->second->ne[0];
    if (heads <= 0 && d_pet > 0) {
      // the exported attention: SOFT_MAX(ADD(DIV(scores, sqrt(head size) * temperature), bias)),
      // temperature 1 (pet.num_heads in newer GGUFs)
      const GIRGraph &graph = interp_.graph();
      for (const GIRNode &node : graph.nodes) {
        if (node.op != "SOFT_MAX" || node.inputs.empty() || node.inputs[0].rfind("node:", 0) != 0) continue;
        const GIRNode *add = graph.get_node(std::stoi(node.inputs[0].substr(5)));
        if (!add || add->op != "ADD" || add->inputs.empty() || add->inputs[0].rfind("node:", 0) != 0) continue;
        const GIRNode *div = graph.get_node(std::stoi(add->inputs[0].substr(5)));
        if (!div || div->op != "DIV") continue;
        auto p = div->params.find("scalar");
        if (p == div->params.end() || !std::holds_alternative<double>(p->second)) continue;
        const double sd = std::get<double>(p->second);
        const int64_t d = std::llround(sd * sd);
        if (d > 0 && d_pet % d == 0 && std::abs(sd * sd - double(d)) < 1e-6 * double(d)) heads = int(d_pet / d);
        break;
      }
    }
    if (pet_ragged_spec(ragged_weights_, d_pet, heads, cutoff_width_, ragged_spec_, why)) {
      ragged_state_ = 1;
      // MLIPCPP_RAGGED_GRAPH_FOLD=1: the folded weight products as matmuls in the graph
      const char *f = std::getenv("MLIPCPP_RAGGED_GRAPH_FOLD");
      if ((!f || !*f || *f == '0') && !ctx_ragged_fold_ &&
          !pet_ragged_fold_weights(ggml_backend_buffer_get_type(weight_buffer_), ragged_spec_, ragged_weights_,
                                   &ctx_ragged_fold_, &ragged_fold_buffer_))
        std::fprintf(stderr, "mlipcpp: folded PET weights computed in the graph\n");
    }
  }
  if (ragged_state_ != 1) std::fprintf(stderr, "mlipcpp: MLIPCPP_RAGGED ignored (%s)\n", why.c_str());
  return ragged_state_ == 1;
}

void GraphModel::build_ragged_graph(int n, int64_t e_cap, int max_len, bool backward) {
  cached_.reset();
  auto g = std::unique_ptr<CachedGraph, CachedGraphDeleter>(new CachedGraph());
  g->n_atoms = n, g->M = 0, g->backward = backward;
  g->ragged = true, g->e_cap = e_cap, g->max_len = max_len;
  const bool adaptive = num_neighbors_adaptive_ > 0.0f;
  g->ctx = ggml_init({ggml_tensor_overhead() * GRAPH_SIZE * 2 + ggml_graph_overhead_custom(GRAPH_SIZE, true) +
                          ggml_graph_overhead_custom(GRAPH_SIZE, false),
                      nullptr, /*no_alloc=*/true});
  if (!g->ctx) throw std::runtime_error("GraphModel: cannot create the graph context");
  ggml_context *ctx = g->ctx;

  std::vector<ggml_tensor *> inputs;
  auto input = [&](ggml_type type, int64_t ne0, int64_t ne1, const char *name) {
    ggml_tensor *t = ne1 > 0 ? ggml_new_tensor_2d(ctx, type, ne0, ne1) : ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_name(t, name);
    ggml_set_input(t);
    inputs.push_back(t);
    return t;
  };
  PetRaggedInputs in;
  in.species = g->species = input(GGML_TYPE_I32, n, 0, "species");
  in.neighbor_species = g->neighbor_species = input(GGML_TYPE_I32, e_cap, 0, "neighbor_species");
  in.edge_vectors = g->edge_vectors = input(GGML_TYPE_F32, 3, e_cap, "edge_vectors");
  in.cutoff_values = g->cutoff_values = input(GGML_TYPE_F32, e_cap, 0, "cutoff_values");
  in.reverse = g->reverse = input(GGML_TYPE_I32, e_cap, 0, "reverse_edge");
  in.offsets = g->offsets = input(GGML_TYPE_I32, n + 1, 0, "offsets");
  in.centers = g->centers = input(GGML_TYPE_I32, e_cap, 0, "centers");
  if (backward) {
    ggml_set_param(g->edge_vectors);
    if (adaptive) ggml_set_param(g->cutoff_values);
  }

  const PetRaggedOutputs out = build_pet_ragged(ctx, ragged_spec_, ragged_weights_, in, n, e_cap, max_len);
  g->energy = out.atom_energy;
  g->loss = out.total;
  ggml_set_output(g->energy);

  g->graph = ggml_new_graph_custom(ctx, GRAPH_SIZE, backward);
  for (ggml_tensor *t : out.order) ggml_build_forward_expand(g->graph, t);
  ggml_build_forward_expand(g->graph, g->energy);
  ggml_build_forward_expand(g->graph, g->loss);
  g->forward = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
  for (ggml_tensor *t : out.order) ggml_build_forward_expand(g->forward, t);
  ggml_build_forward_expand(g->forward, g->energy);
  ggml_build_forward_expand(g->forward, g->loss);
  std::vector<std::pair<ggml_tensor *, float>> constants;
  if (backward) {
    ggml_set_loss(g->loss);
    ggml_build_backward_expand(ctx, g->graph, nullptr);
    g->grad_ev = ggml_graph_get_grad(g->graph, g->edge_vectors);
    g->grad_cv = adaptive ? ggml_graph_get_grad(g->graph, g->cutoff_values) : nullptr;
    if (!g->grad_ev) throw std::runtime_error("GraphModel: the energy does not depend on the edge vectors");
    ggml_set_output(g->grad_ev);
    if (g->grad_cv) ggml_set_output(g->grad_cv);
    constants.emplace_back(ggml_graph_get_grad_acc(g->graph, g->loss), 1.0f);
  }
  finish_cached_graph(std::move(g), inputs, std::move(constants));
}

bool GraphModel::fit_ragged_graph(int n, int64_t edges, int max_kept) {
  // spare rows and sequence length, so that the edge count of an MD step can
  // move a little without a rebuild
  const int64_t pad = std::max<int64_t>(64, edges / 32);
  const int need_len = 1 + std::max(max_kept, 1);
  if (cached_ && cached_->ragged && cached_->n_atoms == n && edges <= cached_->e_cap &&
      cached_->e_cap <= edges + 2 * pad && need_len <= cached_->max_len)
    return false;
  build_ragged_graph(n, std::max<int64_t>(edges + pad, 1), need_len + 4, true);
  return true;
}

void GraphModel::read_cached_outputs(const AtomicSystem &system, ModelResult &result) const {
  const CachedGraph &g = *cached_;
  const int n = g.n_atoms;
  const int32_t *Z = system.atomic_numbers();
  std::vector<float> atomic(n);
  ggml_backend_tensor_get(g.energy, atomic.data(), 0, atomic.size() * sizeof(float));
  double model = 0, composition = 0;
  for (int i = 0; i < n; i++) {
    model += atomic[i];
    if (auto c = composition_energies_.find(Z[i]); c != composition_energies_.end()) composition += c->second;
  }
  result.energy = static_cast<float>(energy_scale_ * model + composition);
  if (!g.nc) return;
  result.nc_forces.resize(3 * size_t(n));
  ggml_backend_tensor_get(g.nc, result.nc_forces.data(), 0, result.nc_forces.size() * sizeof(float));
  for (int i = 0; i < n; i++) {
    auto it = nc_force_scales_.find(Z[i]);
    const float s = it != nc_force_scales_.end() ? it->second : 1.0f;
    for (int c = 0; c < 3; c++) result.nc_forces[3 * i + c] *= s;
  }
  result.has_nc_forces = true;
}

ModelResult GraphModel::predict_cached(const AtomicSystem &system, bool compute_forces) {
  const int n = system.num_atoms();
  const int32_t *Z = system.atomic_numbers();
  const bool adaptive = num_neighbors_adaptive_ > 0.0f;
  const double cutoff = cutoff_, width = cutoff_width_adaptive_;
  const int nt = std::max(1, backend_provider_->n_threads());

  std::vector<int32_t> species(n);
  for (int i = 0; i < n; i++) {
    auto it = species_to_index_.find(Z[i]);
    if (it == species_to_index_.end())
      throw std::runtime_error("Atomic number " + std::to_string(Z[i]) + " not in species map");
    species[i] = it->second;
  }
#ifdef MLIPCPP_HAVE_CUDA_INPUTS
  static const bool host_inputs = std::getenv("MLIPCPP_HOST_INPUTS") != nullptr;
  if (!host_inputs && std::string_view(ggml_backend_name(compute_backend_)).rfind("CUDA", 0) == 0)
    return predict_cached_device(system, compute_forces, species);
#endif

  GraphProfile profile;
  const HostEdges edges = host_edges(system, cutoff, nt, vesin_);
  const int64_t E = static_cast<int64_t>(edges.j.size());
  profile.mark("neighbor_list");

  std::vector<adaptive::Radius> radii(n);
  if (adaptive) {
#pragma omp parallel for num_threads(nt) schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
      const double *d = edges.d.data() + edges.start[i];
      radii[i] = adaptive::solve(
          static_cast<int>(edges.start[i + 1] - edges.start[i]),
          [&](int k) { return d[k] < cutoff ? d[k] : 1e30; }, cutoff, num_neighbors_adaptive_, width);
    }
  }
  profile.mark("adaptive_cutoff");

  // kept edges and their slots; d (i -> j) == d (j -> i) and the pair cutoff
  // is symmetric, so both directions of a pair are kept or dropped together
  auto pair_cutoff = [&](int i, int j) { return adaptive ? 0.5 * (radii[i].r + radii[j].r) : cutoff; };
  std::vector<int32_t> slot(E, -1), nkept(n, 0);
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int i = 0; i < n; i++) {
    int k = 0;
    for (int64_t e = edges.start[i]; e < edges.start[i + 1]; e++) {
      const double d = edges.d[e];
      if (d < cutoff && edges.rev[e] >= 0 && (!adaptive || d <= pair_cutoff(i, edges.j[e]))) slot[e] = k++;
    }
    nkept[i] = k;
  }
  const int max_kept = n ? *std::max_element(nkept.begin(), nkept.end()) : 0;

  const bool no_edges = max_kept == 0, forces_requested = compute_forces;
  if (no_edges) compute_forces = false;
  // the row of atom i's first edge: i * M slots, or packed (ragged) rows
  const bool ragged = use_ragged();
  std::vector<int64_t> first(n + 1, 0);
  if (ragged) {
    for (int i = 0; i < n; i++) first[i + 1] = first[i] + nkept[i];
    if (fit_ragged_graph(n, first[n], max_kept)) profile.mark("build_graph");
  } else if (fit_cached_graph(n, max_kept)) {
    profile.mark("build_graph");
  }
  CachedGraph &g = *cached_;
  const int M = g.M;
  if (!ragged)
    for (int i = 0; i <= n; i++) first[i] = int64_t(i) * M;
  const size_t S = ragged ? size_t(g.e_cap) : size_t(n) * M;

  // pack; rows that hold no kept edge are edges at the cutoff (padding.h)
  std::vector<int32_t> ns(S, padding::kSpecies), rev(S, 0), centers(ragged ? S : 0, 0);
  std::vector<float> ev(3 * S), pm(ragged ? 0 : S, 1.0f), cv(S, static_cast<float>(cutoff));
  if (ragged)
    for (size_t f = first[n]; f < S; f++) rev[f] = static_cast<int32_t>(f), padding::edge(static_cast<float>(cutoff), &ev[3 * f]);
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int i = 0; i < n; i++) {
    if (!ragged)
      for (int s = nkept[i]; s < M; s++) padding::edge(static_cast<float>(cutoff), &ev[3 * (size_t(i) * M + s)]);
    for (int64_t e = edges.start[i]; e < edges.start[i + 1]; e++) {
      if (slot[e] < 0) continue;
      const int j = edges.j[e];
      const size_t f = first[i] + slot[e];
      ns[f] = species[j];
      for (int c = 0; c < 3; c++) ev[3 * f + c] = static_cast<float>(edges.v[3 * e + c]);
      if (!ragged) pm[f] = 0.0f;
      cv[f] = static_cast<float>(pair_cutoff(i, j));
      rev[f] = static_cast<int32_t>(first[j] + slot[edges.rev[e]]);
      if (ragged) centers[f] = i;
    }
  }
  ggml_backend_tensor_set(g.species, species.data(), 0, ggml_nbytes(g.species));
  ggml_backend_tensor_set(g.neighbor_species, ns.data(), 0, ggml_nbytes(g.neighbor_species));
  ggml_backend_tensor_set(g.edge_vectors, ev.data(), 0, ggml_nbytes(g.edge_vectors));
  if (!ragged) ggml_backend_tensor_set(g.padding_mask, pm.data(), 0, ggml_nbytes(g.padding_mask));
  ggml_backend_tensor_set(g.reverse, rev.data(), 0, ggml_nbytes(g.reverse));
  ggml_backend_tensor_set(g.cutoff_values, cv.data(), 0, ggml_nbytes(g.cutoff_values));
  upload_conditioning(g, system.charge(), system.spin_multiplicity());
  if (ragged) {
    std::vector<int32_t> off(first.begin(), first.end());
    ggml_backend_tensor_set(g.offsets, off.data(), 0, ggml_nbytes(g.offsets));
    ggml_backend_tensor_set(g.centers, centers.data(), 0, ggml_nbytes(g.centers));
  }
  profile.mark("inputs");

  if (backend_provider_->compute(compute_forces ? g.graph : g.forward) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("Graph computation failed");
  profile.mark("compute");

  ModelResult result;
  read_cached_outputs(system, result);

  const Cell *cell = system.cell();
  const bool periodic = cell && (cell->periodic[0] || cell->periodic[1] || cell->periodic[2]);
  if (compute_forces) {
    std::vector<float> G(3 * S), C;
    ggml_backend_tensor_get(g.grad_ev, G.data(), 0, G.size() * sizeof(float));
    if (g.grad_cv) {
      C.resize(S);
      ggml_backend_tensor_get(g.grad_cv, C.data(), 0, C.size() * sizeof(float));
    }
    // F_k = sum over k's slots of G[k, s] - G[reverse] (-dE/dx_k); the
    // adaptive cutoff adds the chain term of every edge within the cutoff,
    // dE/dv += dE/dr_k (dr_k / dd) v / d, with dE/dr_k = sum of
    // (C[k, s] + C[reverse]) / 2 (each pair cutoff is the mean of two radii)
    std::vector<double> F(3 * size_t(n), 0.0), chain(C.empty() ? 0 : 3 * E, 0.0);
    std::vector<std::array<double, 6>> virial(periodic ? n : 0, std::array<double, 6>{});
    auto add_virial = [&](int i, const double r[3], const double gr[3]) {
      auto &w = virial[i];
      w[0] += r[0] * gr[0], w[1] += r[1] * gr[1], w[2] += r[2] * gr[2];
      w[3] += 0.5 * (r[1] * gr[2] + r[2] * gr[1]);
      w[4] += 0.5 * (r[0] * gr[2] + r[2] * gr[0]);
      w[5] += 0.5 * (r[0] * gr[1] + r[1] * gr[0]);
    };
#pragma omp parallel for num_threads(nt) schedule(static)
    for (int k = 0; k < n; k++) {
      double dEdr = 0;
      for (int s = 0; s < nkept[k]; s++) {
        const size_t a = first[k] + s, b = rev[a];
        double ga[3], ra[3];
        for (int c = 0; c < 3; c++) {
          ga[c] = G[3 * a + c], ra[c] = ev[3 * a + c];
          F[3 * k + c] += ga[c] - double(G[3 * b + c]);
        }
        if (!C.empty()) dEdr += 0.5 * (double(C[a]) + double(C[b]));
        if (periodic) add_virial(k, ra, ga);
      }
      if (chain.empty()) continue;
      for (int64_t e = edges.start[k]; e < edges.start[k + 1]; e++) {
        const double d = edges.d[e];
        if (!(d < cutoff)) continue;
        const double coef = dEdr * adaptive::ddist(d, radii[k], width) / d;
        for (int c = 0; c < 3; c++) chain[3 * e + c] = coef * edges.v[3 * e + c];
        if (periodic) add_virial(k, &edges.v[3 * e], &chain[3 * e]);
      }
    }
    if (!chain.empty()) {
#pragma omp parallel for num_threads(nt) schedule(static)
      for (int k = 0; k < n; k++)
        for (int64_t e = edges.start[k]; e < edges.start[k + 1]; e++) {
          const int64_t b = edges.rev[e];
          if (b < 0) continue;
          for (int c = 0; c < 3; c++) F[3 * k + c] += chain[3 * e + c] - chain[3 * b + c];
        }
    }
    result.forces.resize(3 * size_t(n));
    for (size_t k = 0; k < F.size(); k++) result.forces[k] = static_cast<float>(energy_scale_ * F[k]);
    result.has_forces = true;

    if (periodic) {
      std::array<double, 6> w{};
      for (const auto &v : virial)
        for (int c = 0; c < 6; c++) w[c] += v[c];
      set_stress(*cell, energy_scale_, w, result);
    }
  }
  if (no_edges && forces_requested) { // energy independent of the positions
    result.forces.assign(3 * static_cast<size_t>(n), 0.0f);
    result.has_forces = true;
    if (cell) {
      result.stress.assign(6, 0.0f);
      result.has_stress = true;
    }
  }
  profile.mark("readback");
  return result;
}

#ifdef MLIPCPP_HAVE_CUDA_INPUTS
ModelResult GraphModel::predict_cached_device(const AtomicSystem &system, bool compute_forces,
                                              const std::vector<int32_t> &species) {
  GraphProfile profile;
  const int n = system.num_atoms();
  const bool adaptive = num_neighbors_adaptive_ > 0.0f;
  cuda_check(cudaSetDevice(device_), "set device");
  if (!device_inputs_) device_inputs_ = new DeviceInputs();
  auto &d = *static_cast<DeviceInputs *>(device_inputs_);
  md::gpu::State &st = d.st;
  st.stream = nullptr; // the legacy stream (predict_device_step sets the caller's)
  d.have_sizes = false, d.list_skin = -1; // predict_device_step: its list and wrap shifts are not this one's
  d.atoms(n);
  // a Verlet skin: successive calls on the same atoms (MD steps) reuse the
  // candidate list until an atom has moved half of it (MLIPCPP_NL_SKIN, Angstrom; 0: rebuild every call)
  static const double skin = [] {
    const char *e = std::getenv("MLIPCPP_NL_SKIN");
    return e && *e ? std::max(0.0, std::atof(e)) : 0.5;
  }();
  const md::gpu::Cutoff cut{cutoff_, skin, adaptive ? num_neighbors_adaptive_ : 0.0, cutoff_width_adaptive_};

  // cell and periodic images within the cutoff (as md::Engine::set_system)
  const Cell *cell = system.cell();
  if (cell) {
    double m[9];
    int pbc[3];
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) m[3 * a + b] = cell->matrix[a][b];
      pbc[a] = cell->periodic[a];
    }
    set_device_cell(st, m, pbc, cutoff_ + skin);
  } else {
    set_device_cell(st, nullptr, nullptr, cutoff_ + skin);
  }
  std::vector<double> x(system.positions(), system.positions() + 3 * size_t(n));
  upload(st.x, x.data(), x.size());
  upload(st.species, species.data(), species.size());
  const bool same_system = d.list_built && skin > 0 && d.list_species == species &&
                           std::equal(std::begin(st.cell), std::end(st.cell), std::begin(d.list_cell)) &&
                           std::equal(std::begin(st.pbc), std::end(st.pbc), std::begin(d.list_pbc));
  if (!(same_system && md::gpu::list_valid(st, skin))) {
    for (;;) {
      md::gpu::reset_scalars(st);
      md::gpu::build_list(st, cut);
      int32_t overflow = 0;
      download(&overflow, st.scalars, 1);
      if (overflow == 0) break;
      d.candidates(overflow + overflow / 4 + 8);
    }
    d.list_built = true, d.list_species = species;
    std::copy(std::begin(st.cell), std::end(st.cell), std::begin(d.list_cell));
    std::copy(std::begin(st.pbc), std::end(st.pbc), std::begin(d.list_pbc));
    profile.mark("neighbor_list");
  } else {
    profile.mark("neighbor_list_reused");
  }

  int max_kept = 0;
  md::gpu::Graph gg;
  if (use_ragged()) {
    // kept edges counted and scanned first; the graph sized to them, then packed
    md::gpu::count_kept(st, cut);
    int32_t total = 0;
    download(&total, st.offsets + n, 1);
    download(&max_kept, st.scalars + 1, 1);
    fit_ragged_graph(n, total, max_kept);
    gg = device_graph(*cached_);
    md::gpu::pack_ragged(st, cut, gg);
  } else {
    // pack into the current graph; rebuild it (and pack again) if the largest
    // neighbour count does not fit
    if (!cached_ || cached_->n_atoms != n || cached_->ragged) build_cached_graph(n, cached_ && !cached_->ragged ? cached_->M : 16, false, true);
    for (;;) {
      st.M = cached_->M;
      gg = device_graph(*cached_);
      md::gpu::prepare_inputs(st, cut, gg);
      download(&max_kept, st.scalars + 1, 1);
      if (!fit_cached_graph(n, max_kept)) break;
    }
  }
  md::gpu::synchronize();
  profile.mark("inputs");
  if (profile.enabled && !cached_->ragged) { // how full the n x M neighbour slots are (padding_mask: 0 = kept edge)
    std::vector<float> pm(ggml_nelements(cached_->padding_mask));
    ggml_backend_tensor_get(cached_->padding_mask, pm.data(), 0, ggml_nbytes(cached_->padding_mask));
    const size_t kept = std::count(pm.begin(), pm.end(), 0.0f);
    std::fprintf(stderr, "GRAPH_SLOTS n=%d M=%d max_kept=%d kept=%zu mean_kept=%.1f fill=%.1f%%\n", n, cached_->M,
                 max_kept, kept, double(kept) / n, 100.0 * kept / pm.size());
    profile.mark("slots_debug");
  }

  const bool no_edges = max_kept == 0, forces_requested = compute_forces;
  if (no_edges) compute_forces = false;
  CachedGraph &g = *cached_;
  upload_conditioning(g, system.charge(), system.spin_multiplicity());
  if (backend_provider_->compute(compute_forces ? g.graph : g.forward) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("Graph computation failed");
  profile.mark("compute");

  ModelResult result;
  read_cached_outputs(system, result);
  const bool periodic = cell && (cell->periodic[0] || cell->periodic[1] || cell->periodic[2]);
  if (compute_forces) {
    md::gpu::conservative_forces(st, cut, gg, energy_scale_);
    std::vector<double> F(3 * size_t(n));
    download(F.data(), st.f_cons, F.size());
    result.forces.assign(F.begin(), F.end());
    result.has_forces = true;
    if (periodic) {
      md::gpu::virial(st, cut, gg, d.virial);
      std::vector<double> v(6 * size_t(n));
      download(v.data(), d.virial, v.size());
      std::array<double, 6> w{};
      for (int i = 0; i < n; i++)
        for (int c = 0; c < 6; c++) w[c] += v[6 * i + c];
      set_stress(*cell, energy_scale_, w, result);
    }
  }
  if (no_edges && forces_requested) { // energy independent of the positions
    result.forces.assign(3 * static_cast<size_t>(n), 0.0f);
    result.has_forces = true;
    if (cell) {
      result.stress.assign(6, 0.0f);
      result.has_stress = true;
    }
  }
  profile.mark("readback");
  return result;
}
#endif

bool GraphModel::device_step_supported() const {
#ifdef MLIPCPP_HAVE_CUDA_INPUTS
  return cached_path_supported() && !nc_forces_ && compute_backend_ &&
         std::string_view(ggml_backend_name(compute_backend_)).rfind("CUDA", 0) == 0;
#else
  return false;
#endif
}

#ifndef MLIPCPP_HAVE_CUDA_INPUTS
GraphModel::DeviceTotals GraphModel::predict_device_step(const DeviceStep &) {
  throw std::runtime_error("GraphModel: device arrays need a CUDA build of mlip.cpp");
}
#else
// predict_cached_device for an engine whose arrays are on the GPU: the same
// kernels, on the caller's stream, without a download before the graph runs.
// Instead of reading back the kept-edge count (and the largest per atom) to
// size the graph, the graph is sized from the previous call's counts (with
// fit_ragged_graph's spare rows) and a guard kernel checks on the device that
// they fit; the list is not checked against the displacements before the step
// but along with the totals after it. The one download, the totals, says
// whether the step held; if not (graph too small, list stale) it is computed
// again before anything is added into the caller's forces (added by a kernel
// that reads the same verdict on the device).
GraphModel::DeviceTotals GraphModel::predict_device_step(const DeviceStep &in) {
  static const bool full_fp32 = [] { // as predict(): TF32 only with MLIPCPP_TF32
    if (std::getenv("MLIPCPP_TF32")) return false;
    setenv("GGML_CUDA_NO_TF32", "1", 0);
    return true;
  }();
  (void)full_fp32;
  if (!device_step_supported())
    throw std::runtime_error("GraphModel: device arrays need a forces-mode model of the cached path on CUDA");
  const int n = in.n_atoms;
  if (n <= 0 || !in.positions || !in.types || !in.forces || !in.type_atomic_numbers || in.n_types < 0)
    throw std::runtime_error("GraphModel: device step without atoms, positions, types, forces or type table");
  const bool adaptive = num_neighbors_adaptive_ > 0.0f;
  cuda_check(cudaSetDevice(device_), "set device");
  if (!device_inputs_) device_inputs_ = new DeviceInputs();
  auto &d = *static_cast<DeviceInputs *>(device_inputs_);
  md::gpu::State &st = d.st;
  cudaStream_t stream = static_cast<cudaStream_t>(in.stream);
  st.stream = in.stream;
  if (st.n != n) d.list_built = false, d.have_sizes = false;
  d.atoms(n);
  if (!d.flags) {
    dalloc(d.flags, 4), dalloc(d.out, 16);
    cuda_check(cudaMallocHost(&d.out_host, 16 * sizeof(double)), "pinned totals");
  }

  // atom type -> model species and composition energy, when the table changes
  const std::vector<int32_t> type_z(in.type_atomic_numbers, in.type_atomic_numbers + in.n_types + 1);
  if (type_z != d.type_z) {
    std::vector<int32_t> species(type_z.size(), -1);
    std::vector<double> e0(type_z.size(), 0.0);
    for (size_t t = 0; t < type_z.size(); t++) {
      if (auto it = species_to_index_.find(type_z[t]); it != species_to_index_.end()) species[t] = it->second;
      if (auto c = composition_energies_.find(type_z[t]); c != composition_energies_.end()) e0[t] = c->second;
    }
    if (int(type_z.size()) > d.type_cap) {
      d.type_cap = int(type_z.size());
      dalloc(d.type_species, d.type_cap), dalloc(d.type_energy, d.type_cap);
    }
    cuda_check(cudaMemcpyAsync(d.type_species, species.data(), species.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream), "type species");
    cuda_check(cudaMemcpyAsync(d.type_energy, e0.data(), e0.size() * sizeof(double), cudaMemcpyHostToDevice, stream),
               "type energies");
    cuda_check(cudaStreamSynchronize(stream), "type table"); // the host vectors go out of scope
    d.type_z = type_z;
  }

  static const double env_skin = [] {
    const char *e = std::getenv("MLIPCPP_NL_SKIN");
    return e && *e ? std::max(0.0, std::atof(e)) : 0.5;
  }();
  const double skin = in.skin >= 0 ? in.skin : env_skin;
  const md::gpu::Cutoff cut{cutoff_, skin, adaptive ? num_neighbors_adaptive_ : 0.0, cutoff_width_adaptive_};
  const bool periodic = in.cell && (in.pbc[0] || in.pbc[1] || in.pbc[2]);
  const int pbc[3] = {in.pbc[0], in.pbc[1], in.pbc[2]};
  set_device_cell(st, periodic ? in.cell : nullptr, pbc, cutoff_ + skin);
  const int64_t *xs = in.position_strides, *fs = in.force_strides;
  const bool ragged = use_ragged();

  DeviceTotals result;
  bool stale = false;
  for (int attempt = 1;; attempt++) {
    if (attempt > 4) throw std::runtime_error("GraphModel: device step did not settle (graph or list)");
    result.attempts = attempt;
    cuda_check(cudaMemsetAsync(d.flags, 0, 4 * sizeof(int32_t), stream), "flags");
    const bool same = d.list_built && skin == d.list_skin && !stale &&
                      std::equal(std::begin(st.cell), std::end(st.cell), std::begin(d.list_cell)) &&
                      std::equal(std::begin(st.pbc), std::end(st.pbc), std::begin(d.list_pbc));
    if (!same || (in.rebuild && attempt == 1)) {
      // the caller's positions as they are, wrapped by build_list; the wrap kept for the next calls
      md::gpu::load_positions(st, in.positions, xs[0], xs[1], nullptr, in.types, d.type_species, in.n_types,
                              d.flags + 1);
      for (;;) {
        md::gpu::reset_scalars(st);
        md::gpu::build_list(st, cut);
        int32_t overflow = 0;
        cuda_check(cudaMemcpyAsync(&overflow, st.scalars, sizeof overflow, cudaMemcpyDeviceToHost, stream), "overflow");
        cuda_check(cudaStreamSynchronize(stream), "overflow");
        if (overflow == 0) break;
        d.candidates(overflow + overflow / 4 + 8);
      }
      md::gpu::store_shift(st, in.positions, xs[0], xs[1], d.shift);
      d.list_built = true, d.list_skin = skin, stale = false, result.rebuilt = true;
      d.list_species.clear(); // predict_cached_device: not its list
      std::copy(std::begin(st.cell), std::end(st.cell), std::begin(d.list_cell));
      std::copy(std::begin(st.pbc), std::end(st.pbc), std::begin(d.list_pbc));
    } else {
      md::gpu::load_positions(st, in.positions, xs[0], xs[1], d.shift, in.types, d.type_species, in.n_types,
                              d.flags + 1);
      md::gpu::displacement(st);
    }

    md::gpu::Graph gg;
    const int32_t *total = nullptr;
    if (ragged) {
      md::gpu::count_kept(st, cut);
      if (!d.have_sizes) { // first call: the counts, to size the graph
        int32_t t = 0, m = 0;
        cuda_check(cudaMemcpyAsync(&t, st.offsets + n, sizeof t, cudaMemcpyDeviceToHost, stream), "total");
        cuda_check(cudaMemcpyAsync(&m, st.scalars + 1, sizeof m, cudaMemcpyDeviceToHost, stream), "max kept");
        cuda_check(cudaStreamSynchronize(stream), "sizes");
        d.last_total = t, d.last_max_kept = m, d.have_sizes = true;
      }
      fit_ragged_graph(n, d.last_total, d.last_max_kept);
      gg = device_graph(*cached_);
      md::gpu::pack_ragged_guarded(st, cut, gg, cached_->max_len, d.flags);
      total = st.offsets + n;
    } else {
      if (!cached_ || cached_->n_atoms != n || cached_->ragged)
        build_cached_graph(n, cached_ && !cached_->ragged ? cached_->M : 16, false, true);
      if (d.have_sizes) fit_cached_graph(n, std::max(d.last_max_kept, 1)); // always with the backward pass
      st.M = cached_->M;
      gg = device_graph(*cached_);
      md::gpu::prepare_inputs(st, cut, gg);
      md::gpu::slots_guard(st, d.flags);
    }

    CachedGraph &g = *cached_;
    upload_conditioning(g, in.charge, in.spin_multiplicity); // synchronous, only when they change
    if (g.direct < 0) { // every node on the compute backend: no scheduler, no host round trip
      g.direct = 1;
      for (int i = 0; i < ggml_graph_n_nodes(g.graph) && g.direct; i++) {
        const ggml_tensor *t = ggml_graph_node(g.graph, i);
        g.direct = ggml_backend_supports_op(compute_backend_, t);
        for (int k = 0; k < GGML_MAX_SRC && g.direct; k++) // and its operands in memory the backend reads
          if (const ggml_tensor *src = t->src[k]; src && src->buffer)
            g.direct = ggml_backend_supports_buft(compute_backend_, ggml_backend_buffer_get_type(src->buffer));
      }
      if (g.direct && !d.ev_in) {
        ggml_backend_dev_t dev = ggml_backend_get_device(compute_backend_);
        d.ev_in = dev ? ggml_backend_event_new(dev) : nullptr;
        d.ev_out = dev ? ggml_backend_event_new(dev) : nullptr;
      }
      if (!d.ev_in || !d.ev_out) g.direct = 0;
      if (std::getenv("MLIPCPP_PROFILE_GRAPH"))
        std::fprintf(stderr, "mlipcpp: device step, graph of %d nodes %s\n", ggml_graph_n_nodes(g.graph),
                     g.direct ? "on the backend's stream, after the caller's" : "through the scheduler (synchronous)");
    }
    if (g.direct) {
      cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(d.ev_in->context), stream), "event");
      ggml_backend_event_wait(compute_backend_, d.ev_in);
      if (ggml_backend_graph_compute_async(compute_backend_, g.graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Graph computation failed");
      ggml_backend_event_record(d.ev_out, compute_backend_);
      cuda_check(cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(d.ev_out->context), 0), "event wait");
    } else {
      cuda_check(cudaStreamSynchronize(stream), "inputs");
      if (backend_provider_->compute(g.graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Graph computation failed");
    }

    md::gpu::conservative_forces(st, cut, gg, energy_scale_, d.flags);
    if (in.virial) md::gpu::virial(st, cut, gg, d.virial, d.flags);
    md::gpu::totals(st, gg.energy, in.types, d.type_energy, in.n_types, energy_scale_, in.virial ? d.virial : nullptr,
                    total, 0.25 * skin * skin, d.flags, d.out);
    md::gpu::add_forces(st, in.forces, fs[0], fs[1], d.flags + 2);
    cuda_check(cudaMemcpyAsync(d.out_host, d.out, 12 * sizeof(double), cudaMemcpyDeviceToHost, stream), "totals");
    cuda_check(cudaStreamSynchronize(stream), "totals");

    const double *o = d.out_host;
    if (o[11] != 0) throw std::runtime_error("GraphModel: an atom type has no element of the model");
    if (ragged) d.last_total = int64_t(o[8]);
    d.last_max_kept = int(o[9]);
    const bool over = o[7] != 0, moved = o[10] > 0.25 * skin * skin;
    if (moved) stale = true; // an atom moved more than half the skin since the list was built
    if (over || moved) continue;
    result.energy = o[0];
    for (int c = 0; c < 6; c++) result.virial[c] = o[1 + c];
    return result;
  }
}
#endif

} // namespace mlipcpp::runtime
