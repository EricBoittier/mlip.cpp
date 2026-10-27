#include "train/conservative.h"
#include "core/p3m.h"
#include "runtime/graph_jvp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fmt/format.h>
#include <fstream>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-impl.h>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
namespace mlipcpp::train {
namespace {
void put(ggml_tensor *t, const std::vector<float> &v) {
  if (ggml_nelements(t) != int64_t(v.size()))
    throw std::runtime_error("Training tensor shape mismatch");
  ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
}
std::vector<float> get(ggml_tensor *t) {
  std::vector<float> v(ggml_nelements(t));
  ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
  return v;
}
void clear(ggml_tensor *t) {
  ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
}
struct Graph {
  ggml_context *ctx = nullptr;
  ggml_gallocr_t allocator = nullptr;
  std::vector<ggml_cgraph *> stages;
  std::map<std::string, ggml_tensor *> inputs;
  ggml_tensor *out = nullptr, *q = nullptr, *p = nullptr, *aq = nullptr,
              *pd = nullptr, *qd = nullptr, *adq = nullptr;
  ggml_cgraph *charge = nullptr, *force = nullptr, *tangent = nullptr,
              *mixed = nullptr, *force_boundary = nullptr,
              *mixed_boundary = nullptr;
  int force_forward = 0, mixed_forward = 0;
  int n = 0, m = 0;
  ~Graph() {
    if (allocator)
      ggml_gallocr_free(allocator);
    if (ctx)
      ggml_free(ctx);
  }
};
} // namespace
struct ConservativeTrainer::Impl {
  runtime::GraphModel &model;
  std::unique_ptr<Graph> cache;
  std::vector<ggml_tensor *> parameters;
  explicit Impl(runtime::GraphModel &m) : model(m) {
    if (!m.forces_mode() || m.nc_forces() || !m.uses_p3m() ||
        m.num_neighbors_adaptive() > 0)
      throw std::runtime_error(
          "Analytic trainer requires fixed-cutoff conservative PET-P3M");
    if (m.param_names().empty())
      throw std::runtime_error("No trainable parameter metadata");
    auto all = m.weights();
    for (auto &name : m.param_names()) {
      auto it = std::find_if(all.begin(), all.end(),
                             [&](auto &w) { return w.first == name; });
      if (it == all.end())
        throw std::runtime_error("Missing trainable weight: " + name);
      if (it->second->type != GGML_TYPE_F32)
        throw std::runtime_error("Training requires F32 weights");
      parameters.push_back(it->second);
    }
  }
  Graph &build(int n, int m) {
    if (cache && cache->n == n && cache->m == m)
      return *cache;
    cache.reset();
    auto g = std::make_unique<Graph>();
    g->n = n;
    g->m = m;
    g->ctx = ggml_init({256ull * 1024 * 1024, nullptr, true});
    if (!g->ctx)
      throw std::runtime_error("Training context allocation failed");
    auto *c = g->ctx;
    auto &in = g->inputs;
    auto input = [&](const char *name, ggml_type type,
                     std::initializer_list<int64_t> ne) {
      auto *t = ggml_new_tensor(c, type, ne.size(), ne.begin());
      ggml_set_input(t);
      ggml_set_name(t, name);
      in[name] = t;
      return t;
    };
    auto &interp = model.interpreter();
    interp.set_dimension("n_atoms", n);
    interp.set_dimension("max_neighbors", m);
    interp.set_dimension("n_edges", n * m);
    interp.set_dimension("max_neighbors_plus_one", m + 1);
    interp.set_input("species", input("species", GGML_TYPE_I32, {n}));
    interp.set_input("neighbor_species",
                     input("neighbor_species", GGML_TYPE_I32, {m, n}));
    auto *edges = input("edge_vectors", GGML_TYPE_F32, {3, m, n});
    ggml_set_param(edges);
    interp.set_input("edge_vectors", edges);
    interp.set_input("padding_mask",
                     input("padding_mask", GGML_TYPE_F32, {m, n}));
    interp.set_input("reverse_neighbor_index",
                     input("reverse_neighbor_index", GGML_TYPE_I32, {n * m}));
    interp.set_input("cutoff_values",
                     input("cutoff_values", GGML_TYPE_F32, {m, n}));
    auto *direction = input("direction", GGML_TYPE_F32, {3, m, n});
    auto *ecoef = input("energy_coeff", GGML_TYPE_F32, {1, n});
    interp.coulomb_application = [&](ggml_context *, ggml_tensor *q) {
      g->q = ggml_cont(c, q);
      ggml_set_output(g->q);
      g->p = input("potential", GGML_TYPE_F32, {q->ne[0], q->ne[1]});
      ggml_set_param(g->p);
      g->aq = input("charge_adjoint", GGML_TYPE_F32, {q->ne[0], q->ne[1]});
      g->pd = input("potential_tangent", GGML_TYPE_F32, {q->ne[0], q->ne[1]});
      g->adq = input("tangent_adjoint", GGML_TYPE_F32, {q->ne[0], q->ne[1]});
      return g->p;
    };
    try {
      g->out = interp.build(c);
    } catch (...) {
      interp.coulomb_application = {};
      throw;
    }
    interp.coulomb_application = {};
    if (!g->q)
      throw std::runtime_error("No P3M application in training graph");
    ggml_set_output(g->out);
    auto graph = [&](ggml_tensor *t, bool back, int *nf = nullptr) {
      auto *gr = ggml_new_graph_custom(c, 65536, back);
      ggml_build_forward_expand(gr, t);
      if (nf)
        *nf = ggml_graph_n_nodes(gr);
      if (back) {
        ggml_set_loss(t);
        ggml_build_backward_expand(c, gr, nullptr);
      }
      return gr;
    };
    g->charge = graph(g->q, false);
    auto inner = [&](ggml_tensor *a, ggml_tensor *b) {
      return ggml_sum(c, ggml_mul(c, a, b));
    };
    edges->flags &= ~GGML_TENSOR_FLAG_PARAM;
    g->force_boundary = graph(ggml_sum(c, g->out), true);
    ggml_set_param(edges);
    auto *energy = ggml_add(c, ggml_sum(c, g->out), inner(g->q, g->aq));
    g->force = graph(energy, true, &g->force_forward);
    runtime::GraphJVP jvp(c);
    jvp.seed(edges, direction);
    jvp.seed(g->p, g->pd);
    g->qd = jvp(g->q);
    if (!g->qd)
      throw std::runtime_error("Charges have no position dependence");
    ggml_set_output(g->qd);
    g->tangent = graph(g->qd, false);
    auto *outd = jvp(g->out);
    auto *loss =
        ggml_sub(c, inner(g->out, ggml_reshape(c, ecoef, g->out)),
                 ggml_scale(c, ggml_sum(c, outd), model.energy_scale()));

    // Force graph already retains its edge derivatives. Only weights and the
    // two P3M boundary inputs are parameters of the mixed-derivative graph.
    edges->flags &= ~GGML_TENSOR_FLAG_PARAM;
    ggml_set_param(g->pd);
    g->mixed_boundary = graph(ggml_dup(c, loss), true);
    loss = ggml_add(c, loss,
                    ggml_add(c, inner(g->q, g->aq), inner(g->qd, g->adq)));
    std::vector<int> flags;
    for (auto *p : parameters) {
      flags.push_back(p->flags);
      ggml_set_param(p);
    }
    try {
      g->mixed = graph(loss, true, &g->mixed_forward);
    } catch (...) {
      for (size_t i = 0; i < parameters.size(); ++i)
        parameters[i]->flags = flags[i];
      throw;
    }
    for (size_t i = 0; i < parameters.size(); ++i)
      parameters[i]->flags = flags[i];
    // Allocate the complete staged schedule once. Forward values remain live
    // until their last reverse use; temporary backward tensors share storage.
    // CPU-visible boundary tensors and loss seeds survive every stage.
    for (auto *gr :
         {g->force_boundary, g->force, g->mixed_boundary, g->mixed}) {
      for (int i = 0; i < ggml_graph_n_nodes(gr); ++i) {
        auto *node = ggml_graph_node(gr, i);
        if (node->flags & GGML_TENSOR_FLAG_LOSS) {
          auto *seed = ggml_graph_get_grad_acc(gr, node);
          if (seed) {
            ggml_set_input(seed);
            ggml_set_output(seed);
          }
        }
      }
    }
    for (auto *t : {ggml_graph_get_grad(g->force_boundary, g->p),
                    ggml_graph_get_grad(g->force, edges),
                    ggml_graph_get_grad(g->mixed_boundary, g->p),
                    ggml_graph_get_grad(g->mixed_boundary, g->pd)})
      ggml_set_output(t);
    for (auto *p : parameters)
      if (auto *t = ggml_graph_get_grad(g->mixed, p))
        ggml_set_output(t);
    auto constants = interp.take_constants();
    for (auto &item : constants) {
      ggml_set_input(item.first);
      ggml_set_output(item.first);
    }
    for (auto &item : in)
      ggml_set_output(item.second);
    auto *schedule = ggml_new_graph_custom(c, 65536, false);
    auto stage = [&](ggml_cgraph *gr, int start,
                     std::initializer_list<ggml_cgraph *> ready) {
      auto *view = ggml_new_graph_custom(c, 65536, false);
      std::unordered_set<ggml_tensor *> done;
      for (auto *r : ready)
        for (int i = 0; i < ggml_graph_n_nodes(r); ++i)
          done.insert(ggml_graph_node(r, i));
      for (int i = start; i < ggml_graph_n_nodes(gr); ++i)
        if (!done.count(ggml_graph_node(gr, i))) {
          auto *node = ggml_graph_node(gr, i);
          // Metal's default F32 GEMM converts operands to half internally.
          // Mixed derivatives need full precision, including attention
          // matrices.
          if (node->op == GGML_OP_MUL_MAT)
            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
          view->nodes[view->n_nodes++] = node;
          schedule->nodes[schedule->n_nodes++] = node;
        }
      g->stages.push_back(view);
    };
    stage(g->charge, 0, {});
    stage(g->force_boundary, 0, {g->charge});
    stage(g->force, g->force_forward, {});
    stage(g->tangent, 0, {g->charge, g->force_boundary});
    stage(g->mixed_boundary, 0, {g->charge, g->force_boundary, g->tangent});
    stage(g->mixed, g->mixed_forward, {});
    g->allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend()));
    if (!ggml_gallocr_alloc_graph(g->allocator, schedule))
      throw std::runtime_error("Training buffer allocation failed");
    for (auto &item : constants) {
      if (!item.first->data)
        continue;
      std::vector<float> value(ggml_nelements(item.first), item.second);
      put(item.first, value);
    }
    if (std::getenv("MLIPCPP_PROFILE_TRAIN"))
      fmt::print(
          stderr,
          "TRAIN_GRAPH atoms={} neighbors={} nodes={} buffer_MiB={:.1f}\n", n,
          m, schedule->n_nodes,
          ggml_gallocr_get_buffer_size(g->allocator, 0) / 1048576.0);
    cache = std::move(g);
    return *cache;
  }
  void compute(int stage, ggml_cgraph *reset = nullptr) {
    auto *g = cache->stages.at(stage);
    if (reset)
      ggml_graph_reset(reset);
    if (ggml_backend_graph_compute(model.backend(), g) != GGML_STATUS_SUCCESS)
      throw std::runtime_error("Conservative training graph failed");
  }
};
ConservativeTrainer::ConservativeTrainer(runtime::GraphModel &m)
    : impl(m.conditioning().enabled
               ? throw std::runtime_error("mlipcpp-train-lr: charge / spin conditioned models are not supported "
                                          "(use mlipcpp-train)")
               : std::make_unique<Impl>(m)) {}
ConservativeTrainer::~ConservativeTrainer() = default;
ConservativeResult
ConservativeTrainer::evaluate(const std::vector<AtomicSystem> &systems,
                              const std::vector<double> &targets,
                              const std::vector<float> &reference, double ew,
                              double fw, bool backward) {
  auto tick = std::chrono::steady_clock::now();
  auto mark = [&](const char *name) {
    auto now = std::chrono::steady_clock::now();
    if (std::getenv("MLIPCPP_PROFILE_TRAIN"))
      fmt::print(stderr, "TRAIN_PROFILE {}={:.3f} ms\n", name,
                 1000 * std::chrono::duration<double>(now - tick).count());
    tick = now;
  };
  if (!std::isfinite(ew) || !std::isfinite(fw) || ew < 0 || fw < 0)
    throw std::runtime_error("Invalid loss weights");
  if (systems.empty() || targets.size() != systems.size())
    throw std::runtime_error("Invalid training batch");
  auto dump = [&](const char *name, const std::vector<float> &v) {
    if (auto *dir = std::getenv("MLIPCPP_DUMP_TRAIN")) {
      std::ofstream file(std::string(dir) + "/" + name + ".bin",
                         std::ios::binary);
      file.write(reinterpret_cast<const char *>(v.data()),
                 v.size() * sizeof(float));
    }
  };
  auto &model = impl->model;
  std::vector<NeighborList> lists;
  std::vector<int> offsets{0};
  int n = 0, m = 1;
  for (auto &s : systems) {
    if (!s.num_atoms())
      throw std::runtime_error("Empty training system");
    auto nl = NeighborListBuilder({model.cutoff(), true, false}).build(s);
    std::vector<int> counts(s.num_atoms());
    for (auto i : nl.centers)
      m = std::max(m, ++counts[i]);
    n += s.num_atoms();
    offsets.push_back(n);
    lists.push_back(std::move(nl));
  }
  if (reference.size() != size_t(3 * n))
    throw std::runtime_error("Invalid reference force shape");
  m = ((m + 31) / 32) * 32;
  auto &g = impl->build(n, m);
  auto &in = g.inputs;
  std::vector<int32_t> species(n), nb(n * m), reverse(n * m),
      neighbor(n * m, -1);
  std::vector<float> edges(3 * n * m), mask(n * m, 1),
      cut(n * m, model.cutoff());
  // Masked edges still enter norm derivatives: use a finite nonzero dummy
  // displacement to avoid undefined 0/0 in mixed derivatives.
  for (int f = 0; f < n * m; ++f)
    edges[3 * f] = 1;
  for (size_t s = 0; s < systems.size(); ++s) {
    const int off = offsets[s];
    auto &sys = systems[s];
    auto &nl = lists[s];
    for (int i = 0; i < sys.num_atoms(); ++i)
      species[off + i] = model.species_to_index().at(sys.atomic_numbers()[i]);
    using Key = std::tuple<int, int, int, int, int>;
    std::map<Key, int> slots;
    std::vector<int> count(sys.num_atoms()), flat(nl.num_pairs());
    for (int e = 0; e < nl.num_pairs(); ++e) {
      int i = nl.centers[e], j = nl.neighbors[e],
          f = (off + i) * m + count[i]++;
      flat[e] = f;
      auto shift =
          nl.cell_shifts.empty() ? std::array<int32_t, 3>{} : nl.cell_shifts[e];
      slots[{i, j, shift[0], shift[1], shift[2]}] = f;
      neighbor[f] = off + j;
      nb[f] = species[off + j];
      mask[f] = 0;
      for (int k = 0; k < 3; ++k)
        edges[3 * f + k] = nl.edge_vectors[e][k];
    }
    for (int e = 0; e < nl.num_pairs(); ++e) {
      auto shift =
          nl.cell_shifts.empty() ? std::array<int32_t, 3>{} : nl.cell_shifts[e];
      reverse[flat[e]] = slots.at(
          {nl.neighbors[e], nl.centers[e], -shift[0], -shift[1], -shift[2]});
    }
  }
  auto puti = [&](const char *key, const std::vector<int32_t> &v) {
    ggml_backend_tensor_set(in.at(key), v.data(), 0,
                            v.size() * sizeof(int32_t));
  };
  puti("species", species);
  puti("neighbor_species", nb);
  puti("reverse_neighbor_index", reverse);
  put(in.at("edge_vectors"), edges);
  put(in.at("padding_mask"), mask);
  put(in.at("cutoff_values"), cut);
  clear(g.aq);
  clear(g.adq);
  clear(g.pd);
  clear(in.at("direction"));
  const int channels = g.q->ne[0];
  // Apply K or its position derivative separately to each cell in the batch.
  auto mesh = [&](const std::vector<float> &q, const std::vector<float> *adj,
                  const std::vector<double> *dir) {
    P3MResult result;
    for (size_t s = 0; s < systems.size(); ++s) {
      int begin = offsets[s] * channels, end = offsets[s + 1] * channels;
      std::vector<float> qs(q.begin() + begin, q.begin() + end), gs;
      std::vector<double> ds;
      if (adj)
        gs.assign(adj->begin() + begin, adj->begin() + end);
      if (dir)
        ds.assign(dir->begin() + 3 * offsets[s],
                  dir->begin() + 3 * offsets[s + 1]);
      auto r = p3m(systems[s], lists[s], model.coulomb_parameters(), channels,
                   qs, adj ? &gs : nullptr, dir ? &ds : nullptr, 2);
      result.potentials.insert(result.potentials.end(), r.potentials.begin(),
                               r.potentials.end());
      result.charge_gradient.insert(result.charge_gradient.end(),
                                    r.charge_gradient.begin(),
                                    r.charge_gradient.end());
      result.potential_tangent.insert(result.potential_tangent.end(),
                                      r.potential_tangent.begin(),
                                      r.potential_tangent.end());
      result.forces.insert(result.forces.end(), r.forces.begin(),
                           r.forces.end());
    }
    return result;
  };
  mark("pack_build_upload");
  impl->compute(0);
  mark("charge");
  auto q = get(g.q);
  auto p = mesh(q, nullptr, nullptr);
  put(g.p, p.potentials);
  mark("mesh_forward");
  impl->compute(1, g.force_boundary);
  mark("force_boundary");
  auto gp = get(ggml_graph_get_grad(g.force_boundary, g.p));
  dump("q", q);
  dump("gp", gp);
  auto pm = mesh(q, &gp, nullptr);
  put(g.aq, pm.charge_gradient);
  mark("mesh_force");
  impl->compute(2, g.force);
  mark("force_reverse");
  auto edgegrad = get(ggml_graph_get_grad(g.force, in.at("edge_vectors")));
  dump("edgegrad", edgegrad);
  ConservativeResult result;
  result.forces = pm.forces;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) {
      int f = i * m + j, k = neighbor[f];
      if (k < 0)
        continue;
      for (int a = 0; a < 3; ++a) {
        result.forces[3 * i + a] += edgegrad[3 * f + a];
        result.forces[3 * k + a] -= edgegrad[3 * f + a];
      }
    }
  const double scale = model.energy_scale();
  for (auto &f : result.forces)
    f *= scale;
  auto atomic = get(g.out);
  std::vector<float> ecoef(n);
  for (size_t s = 0; s < systems.size(); ++s) {
    double energy = 0;
    const int ns = offsets[s + 1] - offsets[s];
    for (int i = offsets[s]; i < offsets[s + 1]; ++i) {
      energy += scale * atomic[i];
      auto it = model.composition_energies().find(
          systems[s].atomic_numbers()[i - offsets[s]]);
      if (it != model.composition_energies().end())
        energy += it->second;
    }
    result.energies.push_back(energy);
    double err = (energy - targets[s]) / ns;
    result.energy_error += err / systems.size();
    result.loss += ew * err * err / systems.size();
    std::fill(ecoef.begin() + offsets[s], ecoef.begin() + offsets[s + 1],
              2 * ew * err * scale / (ns * systems.size()));
  }
  std::vector<double> direction(3 * n);
  for (int i = 0; i < 3 * n; ++i) {
    double r = result.forces[i] - reference[i];
    result.force_rmse += r * r / (3 * n);
    direction[i] = 2 * fw * r / (3 * n);
  }
  result.loss += fw * result.force_rmse;
  result.force_rmse = std::sqrt(result.force_rmse);
  if (!backward)
    return result;
  std::vector<float> edgedir(3 * n * m);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) {
      int f = i * m + j, k = neighbor[f];
      if (k < 0)
        continue;
      for (int a = 0; a < 3; ++a)
        edgedir[3 * f + a] = direction[3 * k + a] - direction[3 * i + a];
    }
  put(in.at("direction"), edgedir);
  put(in.at("energy_coeff"), ecoef);
  mark("loss_direction");
  impl->compute(3);
  auto qd = get(g.qd);
  dump("qd", qd);
  mark("charge_jvp");
  auto kdq = mesh(qd, nullptr, nullptr), dkq = mesh(q, nullptr, &direction);
  for (size_t i = 0; i < kdq.potentials.size(); ++i)
    kdq.potentials[i] += dkq.potential_tangent[i];
  put(g.pd, kdq.potentials);
  clear(g.aq);
  clear(g.adq);
  mark("mesh_jvp");
  impl->compute(4, g.mixed_boundary);
  mark("mixed_boundary");
  auto ap = get(ggml_graph_get_grad(g.mixed_boundary, g.p)),
       at = get(ggml_graph_get_grad(g.mixed_boundary, g.pd));
  dump("ap", ap);
  dump("at", at);
  auto kap = mesh(ap, nullptr, nullptr), kat = mesh(at, nullptr, &direction);
  for (size_t i = 0; i < kap.potentials.size(); ++i)
    kap.potentials[i] += kat.potential_tangent[i];
  put(g.aq, kap.potentials);
  put(g.adq, kat.potentials);
  mark("mesh_reverse");
  impl->compute(5, g.mixed);
  mark("mixed_reverse");
  for (auto *parameter : impl->parameters) {
    auto *grad = ggml_graph_get_grad(g.mixed, parameter);
    auto v =
        grad ? get(grad) : std::vector<float>(ggml_nelements(parameter), 0);
    result.gradient.insert(result.gradient.end(), v.begin(), v.end());
  }
  for (auto x : result.gradient)
    if (!std::isfinite(x))
      throw std::runtime_error("Nonfinite analytic gradient");
  return result;
}
} // namespace mlipcpp::train
