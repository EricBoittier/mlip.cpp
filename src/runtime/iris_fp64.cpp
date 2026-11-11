// fp64 evaluation of iris GGUFs (see iris_fp64.h). Mirrors GraphModel::predict_single_impl's iris path step by step;
// every value the fp32 path rounds to float on the host is kept in double here.
#include "iris_fp64.h"

#include "../core/adaptive_cutoff.h"
#include "../core/gguf_loader.h"
#include "../core/masses.h"
#include "../core/padding.h"
#include "gir64.h"
#include "iris_lr.h"

#include <ggml.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace mlipcpp::runtime::fp64 {

bool exec_fp64_requested() {
  const char *e = std::getenv("MLIPCPP_EXEC");
  return e && (std::strcmp(e, "fp64") == 0 || std::strcmp(e, "FP64") == 0);
}

System64 to_system64(const AtomicSystem &s) {
  System64 r;
  r.n = s.num_atoms();
  r.positions.assign(s.positions(), s.positions() + 3 * size_t(r.n));
  r.numbers.assign(s.atomic_numbers(), s.atomic_numbers() + r.n);
  if (const Cell *c = s.cell()) {
    r.has_cell = true;
    for (int a = 0; a < 3; a++) {
      r.pbc[a] = c->periodic[a];
      for (int b = 0; b < 3; b++) r.cell[a][b] = c->matrix[a][b];
    }
  }
  r.charge = s.charge(), r.spin_multiplicity = s.spin_multiplicity();
  if (s.has_dipole_origin()) r.dipole_origin = s.dipole_origin();
  if (s.has_field()) r.field = s.field();
  return r;
}

ModelResult to_model_result(const Result64 &r) {
  ModelResult m;
  m.energy = float(r.energy);
  m.atomic_energies.assign(r.atomic_energies.begin(), r.atomic_energies.end());
  m.forces.assign(r.forces.begin(), r.forces.end());
  m.stress.assign(r.stress.begin(), r.stress.end());
  m.virial = r.virial;
  m.has_forces = r.has_forces, m.has_stress = r.has_stress;
  if (r.has_charges) {
    m.charges.assign(r.charges.begin(), r.charges.end());
    m.dipole = r.dipole;
    m.fermi_level = r.fermi_level, m.metallicity = r.metallicity;
    m.has_charges = true;
    m.charge_residual = r.charge_residual, m.has_charge_residual = r.has_charge_residual;
    m.dipole_flow = r.dipole_flow;
    m.components = r.components, m.component_charges = r.component_charges;
  }
  return m;
}

using T64 = Tensor<double>;

class IrisDriver {
public:
  const GraphModel &m;
  GIRGraph g_open, g_pbc, g_scores, g_qeq_open, g_qeq_pbc, g_multi_open, g_multi_pbc;
  bool has_pbc = false, has_scores = false, has_qeq_open = false, has_qeq_pbc = false, has_multi_open = false,
       has_multi_pbc = false;
  std::map<std::string, T64> weights;
  double rms_eps = 2.220446049250313e-16;
  std::vector<std::pair<int, std::vector<double>>> col_req;
  Snapshot snap;
  // frozen Ewald parameters (his calculator, ModelOptions::iris_freeze_ewald)
  struct Frozen {
    bool valid = false;
    int n = 0;
    std::vector<int32_t> numbers;
    bool pbc[3] = {false, false, false};
    iris::EwaldFrozen f;
  } frozen;

  explicit IrisDriver(const GraphModel &model) : m(model) {
    const auto &L = m.iris_lr();
    GGUFLoader loader(m.gguf_path());
    auto parse = [&](const std::string &key, GIRGraph &g) {
      const std::string s = loader.get_string(key, "");
      if (s.empty()) return false;
      g = parse_gir_json(s);
      return true;
    };
    if (!parse("graph.json", g_open)) throw std::runtime_error("fp64: the GGUF has no graph.json");
    has_pbc = parse("graph_pbc.json", g_pbc);
    has_scores = parse("graph_scores.json", g_scores);
    has_qeq_open = parse("graph_qeq_apply.json", g_qeq_open);
    has_qeq_pbc = parse("graph_qeq_apply_pbc.json", g_qeq_pbc);
    has_multi_open = parse("graph_multi.json", g_multi_open);
    has_multi_pbc = parse("graph_multi_pbc.json", g_multi_pbc);
    for (const GIRGraph *g : {&g_open, &g_pbc, &g_scores, &g_qeq_open, &g_qeq_pbc, &g_multi_open, &g_multi_pbc})
      for (const auto &n : g->nodes)
        if (n.op.rfind("MLIPCPP_", 0) == 0) throw std::runtime_error("fp64: graph op " + n.op + " is not supported");
    if (m.plato_lr().enabled) throw std::runtime_error("fp64: Plato-PET-LR models are not supported");
    if (m.long_range() || m.uses_p3m()) throw std::runtime_error("fp64: metatrain PET-LR models are not supported");
    if (!m.forces_mode()) throw std::runtime_error("fp64: only forces-mode exports (cutoff_values input) are supported");
    if (m.nc_forces()) throw std::runtime_error("fp64: nc-force exports are not supported");
    (void)L;
    // weights: torch shapes from graph.weight_shapes, float32 data cast exactly
    nlohmann::json shapes;
    const std::string sj = loader.get_string("graph.weight_shapes", "");
    if (!sj.empty()) shapes = nlohmann::json::parse(sj);
    for (const auto &name : loader.get_tensor_names()) {
      ggml_tensor *t = loader.get_tensor(name);
      if (!t) continue;
      if (t->type != GGML_TYPE_F32) throw std::runtime_error("fp64: weight " + name + " is not F32");
      std::vector<int64_t> shape;
      if (shapes.contains(name)) shape = shapes[name].get<std::vector<int64_t>>();
      else
        for (int d = ggml_n_dims(t) - 1; d >= 0; d--) shape.push_back(t->ne[d]);
      const int64_t ne = ggml_nelements(t);
      int64_t want = 1;
      for (auto s : shape) want *= s;
      if (want != ne) throw std::runtime_error("fp64: weight " + name + " shape mismatch");
      const float *p = static_cast<const float *>(t->data);
      weights[name] = T64::from(shape, std::vector<double>(p, p + ne));
    }
  }

  template <class E> void load_weights(E &ex) {
    for (const auto &[k, v] : weights) ex.set_weight(k, v);
    ex.rms_eps_f32_replacement = rms_eps;
  }

  // the graph's declared inputs, from `in` (by name); a missing one is an error
  template <class E> static void set_inputs(E &ex, const GIRGraph &g, const std::map<std::string, T64> &in) {
    for (const auto &i : g.inputs) {
      auto it = in.find(i.name);
      if (it == in.end()) throw std::runtime_error("fp64: no value for graph input " + i.name);
      ex.set_input(i.name, it->second);
    }
  }

  Result64 predict(const System64 &s, bool compute_forces);
};

IrisFp64::IrisFp64(const GraphModel &model) : d_(std::make_unique<IrisDriver>(model)) {}
IrisFp64::~IrisFp64() = default;
Result64 IrisFp64::predict(const System64 &s, bool compute_forces) { return d_->predict(s, compute_forces); }
void IrisFp64::set_fp64_rms_eps(bool on) { d_->rms_eps = on ? 2.220446049250313e-16 : kF32Eps; }
void IrisFp64::request_column_gradient(int column, std::vector<double> w) { d_->col_req.emplace_back(column, std::move(w)); }
void IrisFp64::clear_column_gradients() { d_->col_req.clear(); }
const Snapshot &IrisFp64::last() const { return d_->snap; }
const std::map<std::string, Tensor<double>> &IrisFp64::weights() const { return d_->weights; }

namespace {
double det3(const double h[3][3]) {
  return h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0]) +
         h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
}

// pairs within `rc` (topology from the float builder at rc + skin; vectors and the cut in double)
struct Pairs {
  std::vector<int> c, j;
  std::vector<std::array<int, 3>> S;
  std::vector<std::array<double, 3>> v;
  std::vector<double> d;
};

Pairs pairs_within(const System64 &s, double rc) {
  std::vector<float> xf(s.positions.begin(), s.positions.end());
  std::optional<Cell> cell;
  if (s.has_cell) {
    float h[3][3];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) h[a][b] = float(s.cell[a][b]);
    cell.emplace(h, s.pbc[0], s.pbc[1], s.pbc[2]);
  }
  AtomicSystem sys(s.n, xf.data(), s.numbers.data(), cell ? &*cell : nullptr);
  const double skin = 0.05;
  NeighborList nl = NeighborListBuilder(NeighborListOptions{float(rc + skin), true, false}).build(sys);
  Pairs P;
  const bool shifts = !nl.cell_shifts.empty();
  for (int e = 0; e < nl.num_pairs(); e++) {
    const int i = nl.centers[e], j = nl.neighbors[e];
    std::array<int, 3> S = shifts ? std::array<int, 3>{nl.cell_shifts[e][0], nl.cell_shifts[e][1], nl.cell_shifts[e][2]}
                                  : std::array<int, 3>{0, 0, 0};
    std::array<double, 3> v;
    for (int b = 0; b < 3; b++) {
      v[b] = s.positions[3 * j + b] - s.positions[3 * i + b];
      for (int a = 0; a < 3; a++) v[b] += S[a] * s.cell[a][b];
    }
    // the float builder's convention (r_j - r_i + S h) checked against its own vector
    for (int b = 0; b < 3; b++)
      if (std::fabs(v[b] - nl.edge_vectors[e][b]) > 1e-3 * (1 + std::fabs(v[b])))
        throw std::runtime_error("fp64: neighbour list shift convention mismatch");
    const double d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (!(d2 < rc * rc) || !(d2 > 1e-16)) continue;
    P.c.push_back(i), P.j.push_back(j), P.S.push_back(S), P.v.push_back(v), P.d.push_back(std::sqrt(d2));
  }
  return P;
}
} // namespace

Result64 IrisDriver::predict(const System64 &s, bool compute_forces) {
  const auto &L = m.iris_lr();
  const int n = s.n;
  if (n < 1) throw std::runtime_error("fp64: no atoms");
  for (double x : s.positions)
    if (!std::isfinite(x)) throw std::runtime_error("fp64: non-finite positions");
  const double cutoff = m.cutoff();
  const bool periodic = s.has_cell && (s.pbc[0] || s.pbc[1] || s.pbc[2]);
  if (s.has_cell && std::fabs(det3(s.cell)) < 1e-12) throw std::runtime_error("fp64: singular cell");

  // --- neighbour list, adaptive cutoffs (double grid, as the iris path), pair cutoffs ---
  Pairs P = pairs_within(s, cutoff);
  Pairs full; // pre-filter list (the adaptive chain term)
  std::vector<double> pair_cut(P.c.size(), cutoff), cut_jac;
  bool have_adaptive = false;
  const float nna = m.num_neighbors_adaptive();
  if (nna > 0.0f) {
    const bool solver = m.adaptive_cutoff_method() == "solver";
    full = P;
    const int n_in = int(P.c.size());
    std::vector<int> canonical(n_in);
    std::iota(canonical.begin(), canonical.end(), 0);
    std::vector<std::vector<int>> by_center(n);
    for (int e = 0; e < n_in; e++) by_center[P.c[e]].push_back(e);
    std::map<std::tuple<int, int, int, int, int>, int> index;
    for (int e = 0; e < n_in; e++) index[{P.c[e], P.j[e], P.S[e][0], P.S[e][1], P.S[e][2]}] = e;
    for (int e = 0; e < n_in; e++) {
      auto it = index.find({P.j[e], P.c[e], -P.S[e][0], -P.S[e][1], -P.S[e][2]});
      if (it != index.end()) canonical[e] = std::min(e, it->second);
    }
    auto dist = [&](int e) { return P.d[e] + 1e-15; };
    std::vector<adaptive::Radius> radii(n);
    std::vector<size_t> off(n + 1, 0);
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + by_center[i].size();
    std::vector<double> gj(solver ? 0 : off.back());
    for (int i = 0; i < n; i++) {
      const auto &es = by_center[i];
      if (!solver)
        radii[i].r = adaptive::radius(true, int(es.size()), [&](int k) { return dist(es[k]); }, cutoff, nna,
                                      m.cutoff_width_adaptive(), gj.data() + off[i]);
      else
        radii[i] = adaptive::solve(int(es.size()), [&](int k) { return dist(es[k]); }, cutoff, nna,
                                   m.cutoff_width_adaptive());
    }
    cut_jac.assign(n_in, 0.0);
    if (!solver) {
      for (int i = 0; i < n; i++)
        for (size_t k = 0; k < by_center[i].size(); k++) cut_jac[by_center[i][k]] = gj[off[i] + k];
    } else {
      for (int e = 0; e < n_in; e++) cut_jac[e] = adaptive::ddist(dist(e), radii[P.c[e]], m.cutoff_width_adaptive());
    }
    std::vector<double> pc(n_in), dd(n_in);
    for (int e = 0; e < n_in; e++) pc[e] = 0.5 * (radii[P.c[e]].r + radii[P.j[e]].r), dd[e] = dist(e);
    Pairs K;
    pair_cut.clear();
    for (int e = 0; e < n_in; e++) {
      if (!(dd[canonical[e]] <= pc[canonical[e]])) continue;
      K.c.push_back(P.c[e]), K.j.push_back(P.j[e]), K.S.push_back(P.S[e]), K.v.push_back(P.v[e]), K.d.push_back(P.d[e]);
      pair_cut.push_back(pc[e]);
    }
    P = std::move(K);
    have_adaptive = true;
  }
  std::vector<int> cnt(n, 0);
  for (int c : P.c) cnt[c]++;
  int max_nb = *std::max_element(cnt.begin(), cnt.end());
  m.check_iris_overflow(max_nb);
  std::vector<int> comp; // conserving charges on several components: each atom's component (n_comp > 1)
  int n_comp = 0;
  if (L.enabled && L.conserving_charges) {
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> root = [&](int a) { return parent[a] == a ? a : parent[a] = root(parent[a]); };
    for (size_t e = 0; e < P.c.size(); e++) parent[root(P.c[e])] = root(P.j[e]);
    int comps = 0;
    for (int i = 0; i < n; i++) comps += root(i) == i;
    if (comps > 1 && !has_multi_open)
      throw std::runtime_error("iris PET-LR conserving_charges: " + std::to_string(comps) +
                               " components and no graph_multi.json (re-export)");
    if (comps > 1) { // ids numbered by first atom (his connected_components), as graph_model.cpp
      comp.assign(n, -1);
      std::vector<int> id_of(n, -1);
      for (int i = 0; i < n; i++) {
        const int r = root(i);
        if (id_of[r] < 0) id_of[r] = n_comp++;
        comp[i] = id_of[r];
      }
    }
    if (s.charge < double(n) * L.m_min || s.charge > double(n) * L.m_max)
      throw std::runtime_error("iris PET-LR conserving_charges: charge outside the charge-state range");
  }
  const bool no_edges = max_nb == 0;
  if (no_edges) {
    max_nb = 1;
    if (!L.enabled) compute_forces = false;
  }
  const int M = max_nb, slots = n * M;

  // --- inputs ---
  std::map<std::string, T64> in;
  const auto &sp = m.species_to_index();
  auto species_of = [&](int Z) {
    auto it = sp.find(Z);
    if (it == sp.end()) throw std::runtime_error("Atomic number " + std::to_string(Z) + " not in species map");
    return it->second;
  };
  std::vector<double> species(n), nspec(slots, padding::kSpecies), ev(3 * size_t(slots)), pm(slots, 1.0),
      cv(slots, cutoff), rni(slots, 0.0), nbr(slots, 0.0);
  for (int f = 0; f < slots; f++) ev[3 * f] = 0, ev[3 * f + 1] = 0, ev[3 * f + 2] = cutoff;
  std::vector<int> nb_atom(slots, -1);
  for (int i = 0; i < n; i++) species[i] = species_of(s.numbers[i]);
  std::map<std::tuple<int, int, int, int, int>, int> flat_of;
  std::vector<int> slot(n, 0);
  for (size_t e = 0; e < P.c.size(); e++) {
    const int i = P.c[e], j = P.j[e];
    const int sl = slot[i]++;
    if (sl >= M) continue;
    const int f = i * M + sl;
    flat_of[{i, j, P.S[e][0], P.S[e][1], P.S[e][2]}] = f;
    nspec[f] = species_of(s.numbers[j]);
    for (int b = 0; b < 3; b++) ev[3 * f + b] = P.v[e][b];
    pm[f] = 0.0;
    nb_atom[f] = j;
    cv[f] = pair_cut[e];
  }
  for (size_t e = 0; e < P.c.size(); e++) {
    auto a = flat_of.find({P.c[e], P.j[e], P.S[e][0], P.S[e][1], P.S[e][2]});
    if (a == flat_of.end()) continue;
    auto b = flat_of.find({P.j[e], P.c[e], -P.S[e][0], -P.S[e][1], -P.S[e][2]});
    if (b != flat_of.end()) rni[a->second] = b->second;
  }
  for (int f = 0; f < slots; f++) nbr[f] = std::max(nb_atom[f], 0);
  in["species"] = T64::from({n}, species);
  in["neighbor_species"] = T64::from({n, M}, nspec);
  in["edge_vectors"] = T64::from({n, M, 3}, ev);
  in["padding_mask"] = T64::from({n, M}, pm);
  in["reverse_neighbor_index"] = T64::from({n * M}, rni);
  in["cutoff_values"] = T64::from({n, M}, cv);
  const auto &cond = m.conditioning();
  if (cond.enabled) {
    const auto [ci, si] = cond.rows(s.charge, s.spin_multiplicity);
    in["charge_index"] = T64::from({n}, std::vector<double>(n, ci));
    in["spin_index"] = T64::from({n}, std::vector<double>(n, si));
  }

  // iris PET-LR inputs
  std::array<double, 3> field_E{0, 0, 0};
  bool field_on = false;
  // periodic: the Ewald inputs (graph_pbc.json)
  std::vector<std::array<int, 3>> ew_S, ew_kgrid;
  std::vector<char> ew_keep;
  double ew_h[3][3] = {};
  int ew_slab_axis = -1;
  int kl = 0;
  if (L.enabled) {
    if (periodic && !has_pbc) throw std::runtime_error("iris PET-LR: periodic structure but the GGUF has no graph_pbc.json");
    in["neighbor_index"] = T64::from({n, M}, nbr);
    in["positions"] = T64::from({n, 3}, s.positions);
    in["total_charge"] = T64::from({1}, {double(s.charge)});
    std::array<double, 3> r0{0, 0, 0};
    if (s.dipole_origin) r0 = *s.dipole_origin;
    else {
      double Mt = 0;
      for (int i = 0; i < n; i++) {
        const double mi = ASE_MASSES[std::clamp(s.numbers[i], 0, 118)];
        Mt += mi;
        for (int a = 0; a < 3; a++) r0[a] += mi * s.positions[3 * i + a];
      }
      for (auto &x : r0) x /= Mt;
    }
    in["dipole_origin"] = T64::from({3}, {r0[0], r0[1], r0[2]});
    in["mu_bisect"] = T64::from({1}, {0.0});
    in["gapped"] = T64::from({1}, {0.0});
    for (const char *k : {"m_round", "top", "bottom", "qeq_z", "qeq_lambda", "qeq_x"})
      in[k] = T64::from({n}, std::vector<double>(n, 0.0));
    if (n_comp > 1) {
      std::vector<double> oh(size_t(n) * n_comp, 0.0);
      for (int i = 0; i < n; i++) oh[size_t(i) * n_comp + comp[i]] = 1.0;
      in["comp_onehot"] = T64::from({n, n_comp}, oh);
      for (const char *k : {"comp_charge", "comp_mu", "comp_gapped"})
        in[k] = T64::from({n_comp}, std::vector<double>(n_comp, 0.0));
    }
    if (L.field) {
      if (s.field) field_E = *s.field;
      field_on = field_E[0] != 0 || field_E[1] != 0 || field_E[2] != 0;
      static const bool cons_only = [] {
        const char *e = std::getenv("MLIPCPP_FIELD_CONSERVATIVE_ONLY");
        return e && !std::strcmp(e, "1");
      }();
      const bool has_zf = std::any_of(L.columns.begin(), L.columns.end(), [](const auto &c) { return c.first == "zf_x"; });
      if (compute_forces && !cons_only && field_on && (periodic || have_adaptive))
        throw std::runtime_error("iris PET-LR: the field force E . Z0 is ported for open structures (hard cutoff) "
                                 "only; periodic cells: energies, charges and dipoles only (mlipcpp-eval --no-forces)");
      if (compute_forces && !cons_only && field_on && !has_zf)
        throw std::runtime_error("iris PET-LR: this GGUF has no zf_q0 / zf_x columns for the field force; re-export it");
      if (cons_only) field_on = false;
      in["field"] = T64::from({3}, {field_E[0], field_E[1], field_E[2]});
    }
    if (periodic) {
      bool pbc[3] = {s.pbc[0], s.pbc[1], s.pbc[2]};
      iris::EwaldSetup es;
      if (!L.freeze_ewald) {
        es = iris::ewald_setup(s.cell, pbc, s.positions.data(), n, L.num_k, L.smearing, L.slab_margin);
      } else {
        const bool same = frozen.valid && frozen.n == n && std::equal(s.numbers.begin(), s.numbers.end(), frozen.numbers.begin()) &&
                          frozen.pbc[0] == pbc[0] && frozen.pbc[1] == pbc[1] && frozen.pbc[2] == pbc[2];
        bool refreeze = !same;
        if (same && frozen.f.slab && iris::slab_thickness(s.cell, frozen.f.slab_axis, s.positions.data(), n) > frozen.f.thickness)
          refreeze = true;
        if (refreeze) {
          frozen.f = iris::ewald_freeze(s.cell, pbc, s.positions.data(), n, L.num_k, L.smearing, L.slab_margin);
          frozen.valid = true, frozen.n = n, frozen.numbers = s.numbers;
          for (int a = 0; a < 3; a++) frozen.pbc[a] = pbc[a];
        }
        es = iris::ewald_apply(frozen.f, s.cell, pbc);
      }
      if (L.long_range_transfer && es.slab)
        throw std::runtime_error("iris PET-LR: long_range_transfer on a slab (his ToSample raises)");
      const Pairs lr = pairs_within(s, es.cutoff + 0.5);
      std::vector<int> c2(n, 0);
      for (int i : lr.c) c2[i]++;
      kl = std::max(1, *std::max_element(c2.begin(), c2.end()));
      std::vector<double> idx(size_t(n) * kl, 0.0), shift(size_t(n) * kl * 3, 0.0), keep(size_t(n) * kl, 0.0);
      ew_S.assign(size_t(n) * kl, {0, 0, 0});
      ew_keep.assign(size_t(n) * kl, 0);
      std::fill(c2.begin(), c2.end(), 0);
      for (size_t p = 0; p < lr.c.size(); p++) {
        const int i = lr.c[p];
        const size_t sl = size_t(i) * kl + c2[i]++;
        idx[sl] = lr.j[p], keep[sl] = 1.0, ew_keep[sl] = 1;
        ew_S[sl] = lr.S[p];
        for (int b = 0; b < 3; b++) {
          double v = 0;
          for (int a = 0; a < 3; a++) v += lr.S[p][a] * es.cellc[a][b];
          shift[3 * sl + b] = v;
        }
      }
      ew_kgrid = es.k_grid, ew_slab_axis = es.slab ? es.slab_axis : -1;
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) ew_h[a][b] = es.cellc[a][b];
      const double det = det3(ew_h);
      double rec[3][3];
      for (int i = 0; i < 3; i++) {
        const double *aj = ew_h[(i + 1) % 3], *ak = ew_h[(i + 2) % 3];
        rec[i][0] = 2 * M_PI * (aj[1] * ak[2] - aj[2] * ak[1]) / det;
        rec[i][1] = 2 * M_PI * (aj[2] * ak[0] - aj[0] * ak[2]) / det;
        rec[i][2] = 2 * M_PI * (aj[0] * ak[1] - aj[1] * ak[0]) / det;
      }
      const int64_t K = int64_t(es.k_grid.size());
      std::vector<double> kv(3 * size_t(K));
      for (int64_t q = 0; q < K; q++)
        for (int b = 0; b < 3; b++)
          kv[3 * q + b] = es.k_grid[q][0] * rec[0][b] + es.k_grid[q][1] * rec[1][b] + es.k_grid[q][2] * rec[2][b];
      double normal[3] = {0, 0, 0}, basis = 0;
      if (es.slab) {
        const int k = es.slab_axis;
        const double *v1 = ew_h[(k + 1) % 3], *v2 = ew_h[(k + 2) % 3];
        double nh[3] = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
        const double nn = std::sqrt(nh[0] * nh[0] + nh[1] * nh[1] + nh[2] * nh[2]);
        for (int b = 0; b < 3; b++) normal[b] = nh[b] / nn;
        basis = std::fabs((ew_h[k][0] * nh[0] + ew_h[k][1] * nh[1] + ew_h[k][2] * nh[2]) / nn);
      }
      in["lr_index"] = T64::from({n, kl}, idx);
      in["lr_shift"] = T64::from({n, kl, 3}, shift);
      in["lr_keep"] = T64::from({n, kl}, keep);
      in["kvec"] = T64::from({K, 3}, kv);
      in["volume"] = T64::from({1}, {std::fabs(det)});
      // his Ewald batch is pinned to float32 even in his x64 runs (iris.pet.batching, jax-pme get_batch(dtype=
      // np.float32)): the smearing and the real-space cutoff the maths sees are float32-rounded (torch_grid.py
      // ewald_inputs does the same and = his JAX fp64 to 1e-14; in double they move periodic outputs by ~1e-11)
      in["smearing"] = T64::from({1}, {double(float(es.smearing))});
      in["cutoff"] = T64::from({1}, {double(float(es.cutoff))});
      in["slab"] = T64::from({1}, {es.slab ? 1.0 : 0.0});
      in["slab_normal"] = T64::from({3}, {normal[0], normal[1], normal[2]});
      in["slab_basis"] = T64::from({1}, {basis});
    }
  }
  auto dims = [&](auto &ex) {
    ex.set_dimension("n_atoms", n);
    ex.set_dimension("max_neighbors", M);
    ex.set_dimension("n_edges", int64_t(n) * M);
    ex.set_dimension("max_neighbors_plus_one", M + 1);
    if (periodic && L.enabled) {
      ex.set_dimension("n_lr", kl);
      ex.set_dimension("n_kvec", int64_t(ew_kgrid.size()));
    }
    if (n_comp > 1) ex.set_dimension("n_comp", n_comp);
  };
  const bool iris_qeq = L.enabled && L.reference == "qeq";
  if (n_comp > 1 && (periodic ? !has_multi_pbc : !has_multi_open))
    throw std::runtime_error("iris PET-LR: the GGUF lacks the graph for this structure");
  const GIRGraph &G = !L.enabled ? g_open : n_comp > 1 ? (periodic ? g_multi_pbc : g_multi_open) : periodic ? g_pbc : g_open;
  Result64 R;

  // --- scores pass, host bisection / QEq stage 1 ---
  std::vector<double> qeq_diag;
  std::unique_ptr<Executor<double>> qx;
  auto qeq_project = [](std::vector<double> v) {
    double mm = 0;
    for (double x : v) mm += x;
    mm /= double(std::max<size_t>(v.size(), 1));
    for (double &x : v) x -= mm;
    return v;
  };
  auto qeq_operator = [&](const std::vector<double> &x, std::vector<double> *diag) {
    in["qeq_x"] = T64::from({n}, x);
    qx->set_input("qeq_x", in["qeq_x"]);
    qx->forward(false);
    const T64 &o = qx->output();
    if (o.numel() != 2 * int64_t(n)) throw std::runtime_error("iris QEq: operator graph output size");
    std::vector<double> a(n);
    for (int i = 0; i < n; i++) a[i] = o.ptr()[2 * i];
    if (diag) {
      diag->resize(n);
      for (int i = 0; i < n; i++) (*diag)[i] = o.ptr()[2 * i + 1];
    }
    return a;
  };
  auto qeq_A = [&](const std::vector<double> &v) { return qeq_project(qeq_operator(qeq_project(v), nullptr)); };
  auto qeq_M = [&](const std::vector<double> &r) {
    std::vector<double> v(r.size());
    for (size_t i = 0; i < r.size(); i++) v[i] = r[i] / qeq_diag[i];
    return qeq_project(v);
  };
  // the scores graph runs for the references that need mu / chi (as GraphModel::load_from_gguf decides)
  // (conserving residual: the free charges p of the multi-component Q_c rule, on several components only)
  const bool needs_scores = L.enabled && ((L.reference != "species" && L.reference != "species_weighted" &&
                                           L.reference != "residual" && L.reference != "threshold" &&
                                           L.reference != "head") ||
                                          (L.reference == "residual" && n_comp > 1));
  if (needs_scores && !has_scores) throw std::runtime_error("iris PET-LR: the GGUF has no graph_scores.json");
  if (needs_scores) {
    Executor<double> sx(g_scores);
    load_weights(sx);
    dims(sx);
    set_inputs(sx, g_scores, in);
    sx.forward(false);
    const T64 &so = sx.output();
    if (iris_qeq) {
      if (so.numel() != n) throw std::runtime_error("iris QEq: scores graph output size");
      const GIRGraph &gq = periodic ? g_qeq_pbc : g_qeq_open;
      if (periodic ? !has_qeq_pbc : !has_qeq_open) throw std::runtime_error("iris QEq: no operator graph");
      qx = std::make_unique<Executor<double>>(gq);
      load_weights(*qx);
      dims(*qx);
      set_inputs(*qx, gq, in);
      const std::vector<double> uniform(n, double(s.charge) / n);
      std::vector<double> b = qeq_operator(uniform, &qeq_diag);
      for (int i = 0; i < n; i++) b[i] = -(b[i] + so.ptr()[i]);
      b = qeq_project(b);
      const iris::CGResult z = iris::cg_host(qeq_A, b, qeq_M, L.qeq_maxiter, f32_intent(float(L.qeq_tol)));
      R.qeq_forward_iterations = z.iterations;
      in["qeq_z"] = T64::from({n}, z.x);
    } else if (n_comp > 1) {
      // several components (graph_model.cpp): Q_c from mlip.cpp's R-only rule, then (fermi) one bisection per
      // component on its own atoms at its Q_c; mu_c and the tie weights in double
      if (so.numel() != n) throw std::runtime_error("iris PET-LR: scores graph output size");
      const int Cc = n_comp;
      const double Q = double(s.charge);
      iris::FermiConfig fc;
      fc.m_min = L.m_min, fc.m_max = L.m_max, fc.temperature = f32_intent(float(L.temperature));
      fc.iterations = L.iterations, fc.eps = f32_intent(float(L.eps));
      std::vector<double> qhat(Cc, 0.0), sd(so.ptr(), so.ptr() + n);
      std::vector<int> size(Cc, 0);
      for (int i = 0; i < n; i++) size[comp[i]]++;
      if (L.reference == "fermi") {
        const iris::FermiHost h = iris::fermi_host(sd, Q, fc);
        for (int i = 0; i < n; i++) qhat[comp[i]] += iris::occupation(sd[i] - h.mu, fc);
      } else {
        double psum = 0;
        for (int i = 0; i < n; i++) qhat[comp[i]] += sd[i], psum += sd[i];
        for (int c = 0; c < Cc; c++) qhat[c] += (Q - psum) * size[c] / n;
      }
      const std::vector<double> Qc = iris::largest_remainder(qhat, Q);
      for (int c = 0; c < Cc; c++)
        if (Qc[c] < double(size[c]) * L.m_min || Qc[c] > double(size[c]) * L.m_max)
          throw std::runtime_error("iris PET-LR conserving_charges: component charge outside its charge-state range");
      std::vector<double> mu(Cc, 0.0), gp(Cc, 0.0), mr(n, 0.0), tp(n, 0.0), bt(n, 0.0);
      if (L.reference == "fermi")
        for (int c = 0; c < Cc; c++) {
          std::vector<int> idx;
          std::vector<double> sc;
          for (int i = 0; i < n; i++)
            if (comp[i] == c) idx.push_back(i), sc.push_back(sd[i]);
          const iris::FermiHost h = iris::fermi_host(sc, Qc[c], fc);
          mu[c] = h.mu, gp[c] = h.gapped ? 1.0 : 0.0;
          int nt = 0, nb = 0;
          for (size_t k = 0; k < idx.size(); k++) nt += h.top[k] != 0, nb += h.bottom[k] != 0;
          for (size_t k = 0; k < idx.size(); k++) {
            mr[idx[k]] = h.m_round[k];
            tp[idx[k]] = h.top[k] != 0 ? 1.0 / nt : 0.0;
            bt[idx[k]] = h.bottom[k] != 0 ? 1.0 / nb : 0.0;
          }
        }
      in["comp_charge"] = T64::from({Cc}, Qc);
      in["comp_mu"] = T64::from({Cc}, mu);
      in["comp_gapped"] = T64::from({Cc}, gp);
      in["m_round"] = T64::from({n}, mr);
      in["top"] = T64::from({n}, tp);
      in["bottom"] = T64::from({n}, bt);
      R.components = comp, R.component_charges = Qc;
    } else {
      const bool weighted = L.reference == "species_fill";
      const int sc = weighted ? 2 : 1;
      if (so.numel() != int64_t(sc) * n) throw std::runtime_error("iris PET-LR: scores graph output size");
      std::vector<double> sd(n), wd(weighted ? n : 0);
      for (int i = 0; i < n; i++) {
        sd[i] = so.ptr()[size_t(i) * sc];
        if (weighted) wd[i] = so.ptr()[size_t(i) * sc + 1];
      }
      iris::FermiConfig fc;
      fc.m_min = L.m_min, fc.m_max = L.m_max, fc.temperature = f32_intent(float(L.temperature));
      fc.iterations = L.iterations, fc.eps = f32_intent(float(L.eps));
      const iris::FermiHost h = iris::fermi_host(sd, double(s.charge), fc, weighted ? &wd : nullptr);
      std::vector<double> top(n), bot(n), mr(n);
      int nt = 0, nbt = 0;
      for (int i = 0; i < n; i++) nt += h.top[i] != 0, nbt += h.bottom[i] != 0;
      for (int i = 0; i < n; i++) {
        top[i] = h.top[i] != 0 ? 1.0 / nt : 0.0; // 1 / ties in double (the float path rounds it)
        bot[i] = h.bottom[i] != 0 ? 1.0 / nbt : 0.0;
        mr[i] = h.m_round[i];
      }
      in["mu_bisect"] = T64::from({1}, {h.mu});
      in["gapped"] = T64::from({1}, {h.gapped ? 1.0 : 0.0});
      in["m_round"] = T64::from({n}, mr);
      in["top"] = T64::from({n}, top);
      in["bottom"] = T64::from({n}, bot);
    }
  }

  // MLIPCPP_FP64_DUMP=<file>: append this structure's graph inputs and output (JSON lines; diagnostics: the
  // eager torch module on exactly these inputs, scripts/pet_lr_port/fp64_torch_inputs.py)
  static const char *dump = [] { const char *e = std::getenv("MLIPCPP_FP64_DUMP"); return e && *e ? e : nullptr; }();
  auto dump_json = [&](const T64 *outp) {
    nlohmann::json j;
    for (const auto &i : G.inputs) {
      const T64 &t = in.at(i.name);
      j["inputs"][i.name] = {{"shape", t.shape}, {"data", *t.data}};
    }
    if (outp) j["output"] = {{"shape", outp->shape}, {"data", *outp->data}};
    if (FILE *fp = std::fopen(dump, "a")) {
      const std::string sj = j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
      std::fprintf(fp, "%s\n", sj.c_str());
      std::fclose(fp);
    }
  };

  // --- full graph ---
  Executor<double> ex(G);
  load_weights(ex);
  dims(ex);
  set_inputs(ex, G, in);
  snap.graph = &G;
  snap.inputs.clear();
  for (const auto &i : G.inputs) snap.inputs[i.name] = in.at(i.name);
  snap.dims = {{"n_atoms", n}, {"max_neighbors", M}, {"n_edges", int64_t(n) * M}, {"max_neighbors_plus_one", M + 1}};
  if (periodic && L.enabled) snap.dims["n_lr"] = kl, snap.dims["n_kvec"] = int64_t(ew_kgrid.size());
  ex.forward(compute_forces);
  const T64 &out = ex.output();
  const int C = L.enabled ? L.n_out : 1;
  if (out.numel() != int64_t(n) * C) throw std::runtime_error("fp64: graph output is not [n_atoms, n_out]");
  std::vector<double> aux(out.ptr(), out.ptr() + out.numel());
  if (dump) dump_json(&out);
  const double escale = m.energy_scale();

  // gradients of <cot, output> -> per-atom -d/dr (edge scatter + explicit positions), and the raw adjoints
  auto sweep = [&](const std::vector<double> &cot) { ex.backward(cot); };
  auto minus_dr = [&]() {
    std::vector<double> F(3 * size_t(n), 0.0);
    const auto &ge = ex.input_grad("edge_vectors");
    if (!ge.empty())
      for (int f = 0; f < slots; f++) {
        if (pm[f] > 0.5 || nb_atom[f] < 0) continue;
        const int ca = f / M, na = nb_atom[f];
        for (int k = 0; k < 3; k++) F[3 * ca + k] += ge[3 * f + k], F[3 * na + k] -= ge[3 * f + k];
      }
    const auto &gp = ex.input_grad("positions");
    if (!gp.empty())
      for (int i = 0; i < 3 * n; i++) F[i] -= gp[i];
    return F;
  };
  int cq = 8, cx = 8;
  if (L.enabled)
    for (int k = 0, col = 8; k < int(L.columns.size()); col += L.columns[k].second, k++) {
      if (L.columns[k].first == "zf_q0") cq = col;
      if (L.columns[k].first == "zf_x") cx = col;
    }
  std::vector<double> field_force;
  if (compute_forces) {
    if (field_on) { // his E . Z0 (see graph_model.cpp): two sweeps of sum_j a_j X_j
      const bool fermi = L.reference == "fermi";
      std::vector<double> c(n), w(n);
      double ctr[3] = {0, 0, 0}; // about the centroid, as graph_model.cpp (the force is origin-free)
      for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) ctr[k] += s.positions[3 * i + k] / n;
      for (int i = 0; i < n; i++) {
        c[i] = field_E[0] * (s.positions[3 * i] - ctr[0]) + field_E[1] * (s.positions[3 * i + 1] - ctr[1]) +
               field_E[2] * (s.positions[3 * i + 2] - ctr[2]);
        w[i] = fermi ? aux[size_t(i) * C + 7] : 1.0;
      }
      auto pass = [&](const std::vector<double> &a) {
        std::vector<double> cot(out.numel(), 0.0);
        for (int i = 0; i < n; i++) cot[size_t(i) * C + cx] = a[i];
        sweep(cot);
        return minus_dr();
      };
      std::vector<double> a(n);
      for (int i = 0; i < n; i++) a[i] = c[i] * w[i];
      const std::vector<double> FG = pass(a);
      const std::vector<double> FQ = pass(w);
      field_force.assign(3 * size_t(n), 0.0);
      for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++)
          field_force[3 * i + k] = field_E[k] * aux[size_t(i) * C + cq] - FG[3 * i + k] + c[i] * FQ[3 * i + k];
    }
    // -d(<cot, out>)/dr including the adaptive cutoffs' chain (for the column gradients; the energy's below keeps
    // its own pieces for the virial)
    auto minus_dr_full = [&]() {
      std::vector<double> F = minus_dr();
      const auto &Cg = ex.input_grad("cutoff_values");
      if (have_adaptive && !Cg.empty()) {
        std::vector<double> Sk(n, 0.0);
        for (int f = 0; f < slots; f++) {
          if (pm[f] > 0.5 || nb_atom[f] < 0) continue;
          Sk[f / M] += 0.5 * Cg[f], Sk[nb_atom[f]] += 0.5 * Cg[f];
        }
        for (size_t e = 0; e < full.c.size(); e++) {
          const double coef = full.d[e] > 0.0 ? Sk[full.c[e]] * cut_jac[e] / full.d[e] : 0.0;
          for (int c = 0; c < 3; c++) F[3 * full.c[e] + c] += coef * full.v[e][c], F[3 * full.j[e] + c] -= coef * full.v[e][c];
        }
      }
      return F;
    };
    for (const auto &[col, w] : col_req) {
      if (iris_qeq) throw std::runtime_error("fp64: column gradients of a QEq model are not supported");
      if (col < 0 || col >= C || int(w.size()) != n) throw std::runtime_error("fp64: column gradient request");
      std::vector<double> c(out.numel(), 0.0);
      for (int i = 0; i < n; i++) c[size_t(i) * C + col] = w[i];
      sweep(c);
      std::vector<double> g = minus_dr_full();
      for (auto &x : g) x = -x;
      R.column_gradients.push_back(std::move(g));
    }
    std::vector<double> cot(out.numel(), 0.0);
    for (int i = 0; i < n; i++) cot[size_t(i) * C] = 1.0;
    sweep(cot);
    if (iris_qeq) { // stage 3: lambda = cg(zbar), then the graph again with lambda; d/dr of sum E + sum lambda r
      const auto &zb = ex.input_grad("qeq_z");
      if (zb.empty()) throw std::runtime_error("iris QEq: no gradient for z");
      const iris::CGResult lam = iris::cg_host(qeq_A, zb, qeq_M, L.qeq_maxiter, f32_intent(float(L.qeq_tol)));
      R.qeq_cotangent_iterations = lam.iterations;
      in["qeq_lambda"] = T64::from({n}, lam.x);
      ex.set_input("qeq_lambda", in["qeq_lambda"]);
      ex.forward(true);
      for (int i = 0; i < n; i++) cot[size_t(i) * C + 6] = 1.0;
      sweep(cot);
    }
  } else if (iris_qeq) {
    (void)0;
  }

  // --- results ---
  R.atomic_energies.resize(n);
  double E = 0;
  for (int i = 0; i < n; i++) {
    R.atomic_energies[i] = aux[size_t(i) * C] * escale;
    E += aux[size_t(i) * C];
  }
  E *= escale;
  for (int i = 0; i < n; i++) {
    const auto &ce = m.composition_energies();
    auto it = ce.find(s.numbers[i]);
    if (it != ce.end()) E += double(it->second);
  }
  R.energy = E;

  if (compute_forces) {
    std::vector<double> F = minus_dr();
    const auto &ge = ex.input_grad("edge_vectors");
    // adaptive cutoffs: the chain through the atomic cutoffs (graph_model.cpp)
    std::vector<std::array<double, 3>> chain;
    if (have_adaptive) {
      const auto &Cg = ex.input_grad("cutoff_values");
      if (!Cg.empty()) {
        std::vector<double> Sk(n, 0.0);
        for (int f = 0; f < slots; f++) {
          if (pm[f] > 0.5 || nb_atom[f] < 0) continue;
          Sk[f / M] += 0.5 * Cg[f];
          Sk[nb_atom[f]] += 0.5 * Cg[f];
        }
        chain.resize(full.c.size());
        for (size_t e = 0; e < full.c.size(); e++) {
          const int k = full.c[e], l = full.j[e];
          const double d = full.d[e];
          const double coef = d > 0.0 ? Sk[k] * cut_jac[e] / d : 0.0;
          for (int c = 0; c < 3; c++) {
            chain[e][c] = coef * full.v[e][c];
            F[3 * k + c] += chain[e][c];
            F[3 * l + c] -= chain[e][c];
          }
        }
      }
    }
    for (auto &x : F) x *= escale;
    for (size_t i = 0; i < field_force.size(); i++) F[i] += field_force[i];
    R.forces = F;
    R.has_forces = true;
    if (periodic) {
      std::vector<double> W(9, 0.0);
      if (!ge.empty())
        for (int f = 0; f < slots; f++) {
          if (pm[f] > 0.5) continue;
          for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) W[3 * a + b] += ev[3 * f + a] * ge[3 * f + b];
        }
      for (size_t e = 0; e < chain.size(); e++)
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) W[3 * a + b] += full.v[e][a] * chain[e][b];
      if (L.enabled) { // the Ewald virial (graph_model.cpp): sum_i r_a dE/dr_b + (h^T dE/dh)_ab
        const auto &gp = ex.input_grad("positions");
        std::vector<double> w(9, 0.0);
        double Gm[3][3] = {};
        if (!gp.empty())
          for (int i = 0; i < n; i++)
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++) w[3 * a + b] += s.positions[3 * i + a] * gp[3 * i + b];
        auto grad = [&](const char *name, size_t len) {
          std::vector<double> d = ex.input_grad(name);
          if (d.empty()) d.assign(len, 0.0);
          return d;
        };
        const std::vector<double> gs = grad("lr_shift", ew_S.size() * 3);
        for (size_t sl = 0; sl < ew_S.size(); sl++)
          if (ew_keep[sl])
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++) Gm[a][b] += double(ew_S[sl][a]) * gs[3 * sl + b];
        const auto &h = ew_h;
        const double det = det3(h);
        double X[3][3];
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            const int i1 = (j + 1) % 3, i2 = (j + 2) % 3, j1 = (i + 1) % 3, j2 = (i + 2) % 3;
            X[i][j] = (h[i1][j1] * h[i2][j2] - h[i1][j2] * h[i2][j1]) / det;
          }
        const std::vector<double> gk = grad("kvec", ew_kgrid.size() * 3);
        double dR[3][3] = {};
        for (size_t q = 0; q < ew_kgrid.size(); q++)
          for (int i = 0; i < 3; i++)
            for (int b = 0; b < 3; b++) dR[i][b] += double(ew_kgrid[q][i]) * gk[3 * q + b];
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) {
            double v = 0;
            for (int p = 0; p < 3; p++)
              for (int q = 0; q < 3; q++) v += X[p][a] * 2 * M_PI * dR[q][p] * X[b][q];
            Gm[a][b] -= v;
          }
        const double gV = grad("volume", 1)[0];
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) Gm[a][b] += gV * std::fabs(det) * X[b][a];
        if (ew_slab_axis >= 0) {
          const int k = ew_slab_axis;
          const double *a1 = h[(k + 1) % 3], *a2 = h[(k + 2) % 3];
          const double c[3] = {a1[1] * a2[2] - a1[2] * a2[1], a1[2] * a2[0] - a1[0] * a2[2], a1[0] * a2[1] - a1[1] * a2[0]};
          const double cn = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
          const double nh[3] = {c[0] / cn, c[1] / cn, c[2] / cn};
          const double hk_n = h[k][0] * nh[0] + h[k][1] * nh[1] + h[k][2] * nh[2];
          const std::vector<double> gn0 = grad("slab_normal", 3);
          const double gb = grad("slab_basis", 1)[0];
          double gn[3], gc[3];
          for (int b = 0; b < 3; b++) gn[b] = gn0[b] + gb * (hk_n >= 0 ? 1.0 : -1.0) * h[k][b];
          const double ngn = nh[0] * gn[0] + nh[1] * gn[1] + nh[2] * gn[2];
          for (int b = 0; b < 3; b++) gc[b] = (gn[b] - nh[b] * ngn) / cn;
          const double r1[3] = {a2[1] * gc[2] - a2[2] * gc[1], a2[2] * gc[0] - a2[0] * gc[2], a2[0] * gc[1] - a2[1] * gc[0]};
          const double r2[3] = {gc[1] * a1[2] - gc[2] * a1[1], gc[2] * a1[0] - gc[0] * a1[2], gc[0] * a1[1] - gc[1] * a1[0]};
          for (int b = 0; b < 3; b++) Gm[(k + 1) % 3][b] += r1[b], Gm[(k + 2) % 3][b] += r2[b], Gm[k][b] = 0;
        }
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++)
            for (int A = 0; A < 3; A++) w[3 * a + b] += h[A][a] * Gm[A][b];
        for (int i = 0; i < 9; i++) W[i] += w[i];
      }
      for (auto &x : W) x *= escale;
      const double V = std::fabs(det3(s.cell));
      R.virial = W;
      R.stress = {W[0] / V, W[4] / V, W[8] / V, 0.5 * (W[5] + W[7]) / V, 0.5 * (W[2] + W[6]) / V, 0.5 * (W[1] + W[3]) / V};
      R.has_stress = true;
    }
  }
  if (L.enabled) {
    R.charges.resize(n);
    R.dipole.assign(3, 0.0);
    for (int i = 0; i < n; i++) {
      R.charges[i] = aux[size_t(i) * C + 1];
      for (int a = 0; a < 3; a++) R.dipole[a] += aux[size_t(i) * C + 2 + a];
      R.metallicity += aux[size_t(i) * C + 7];
    }
    int col = 8;
    for (const auto &[name, width] : L.columns) {
      if (name.rfind("zf_", 0) == 0) {
        col += width;
        continue;
      }
      std::vector<double> sum(width, 0.0);
      for (int i = 0; i < n; i++)
        for (int a = 0; a < width; a++) sum[a] += aux[size_t(i) * C + col + a];
      if (name == "charge_residual") R.charge_residual = sum[0], R.has_charge_residual = true;
      if (name == "dipole_flow") R.dipole_flow = sum;
      col += width;
    }
    R.fermi_level = aux[5];
    R.has_charges = true;
  }
  return R;
}

} // namespace mlipcpp::runtime::fp64
