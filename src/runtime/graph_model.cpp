#include "graph_model.h"
#include "graph_cache.h"
#include "core/adaptive_cutoff.h"
#include "core/gguf_loader.h"
#include "core/padding.h"
#include "core/p3m.h"

#include <nlohmann/json.hpp>

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <cmath>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

using json = nlohmann::json;

namespace mlipcpp::runtime {

namespace {

// Bump cutoff function
float cutoff_func_bump(float distance, float cutoff, float width) {
  float x = (distance - (cutoff - width)) / width;
  if (x <= 0.0f) return 1.0f;
  if (x >= 1.0f) return 0.0f;
  float tan_val = std::tan(static_cast<float>(M_PI) * x);
  return 0.5f * (1.0f + std::tanh(1.0f / tan_val));
}

// Cosine cutoff function
float cutoff_func_cosine(float distance, float cutoff, float width) {
  float x = (distance - (cutoff - width)) / width;
  if (x <= 0.0f) return 1.0f;
  if (x >= 1.0f) return 0.0f;
  return 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * x));
}

// Internal state from compute_adaptive_cutoffs that the chain-rule stress
// correction reuses (avoids re-deriving it for every edge perturbation).
struct AdaptiveCutoffCache {
  std::vector<float> probes;            // [n_probes]
  std::vector<double> diff;             // [n_atoms * n_probes]
  std::vector<double> width;            // [n_atoms * n_probes]
  std::vector<double> weights;          // [n_atoms * n_probes]   softmax over probes
  std::vector<float> atomic_cutoffs;    // [n_atoms]
  // X[k * n_probes + p] = ∂atomic_cutoffs[k] / ∂diff[k, p].
  // Populated by compute_adaptive_cutoffs_with_jacobian.
  std::vector<double> diff_jacobian;
  float effective_width = 0.0f;
  int n_probes = 0;
};

// Compute per-atom adaptive cutoffs. Mirrors metatrain.pet.modules
// .adaptive_cutoff.get_adaptive_cutoffs: probe-grid + Gaussian weighting,
// width per probe taken from the numerical gradient of (effective_nn -
// target + baseline) along the probe axis.
//
// Inputs:
//   centers, distances : edge index/value arrays of length n_edges
//   target             : num_neighbors_adaptive (e.g. 8.0)
//   n_atoms, max_cutoff: as passed by the caller
// Returns: vector of length n_atoms with per-atom adapted cutoff radii.
std::vector<float> compute_adaptive_cutoffs(
    const std::vector<int32_t> &centers,
    const std::vector<float> &distances,
    float target,
    int n_atoms,
    float max_cutoff,
    float min_cutoff = 0.5f,
    float effective_width = 1.0f) {
  const float probe_spacing = effective_width / 4.0f;
  std::vector<float> probe_cutoffs;
  for (float pc = min_cutoff; pc < max_cutoff; pc += probe_spacing) {
    probe_cutoffs.push_back(pc);
  }
  const int n_probes = static_cast<int>(probe_cutoffs.size());
  if (n_probes == 0 || n_atoms == 0) {
    return std::vector<float>(n_atoms, max_cutoff);
  }

  // diff[atom * n_probes + p] starts as effective_num_neighbors at probe p,
  // then we subtract `target` and add the baseline = target * (p/(n-1))^3.
  std::vector<double> diff(static_cast<size_t>(n_atoms) * n_probes, 0.0);

  // Accumulate effective neighbor count: f(d, probe, effective_width).
  const size_t n_edges = distances.size();
  for (int p = 0; p < n_probes; p++) {
    const float pc = probe_cutoffs[p];
    for (size_t e = 0; e < n_edges; e++) {
      const int a = centers[e];
      const float w = cutoff_func_bump(distances[e], pc, effective_width);
      diff[static_cast<size_t>(a) * n_probes + p] += static_cast<double>(w);
    }
  }

  // Subtract target, add baseline.
  for (int a = 0; a < n_atoms; a++) {
    for (int p = 0; p < n_probes; p++) {
      double x = (n_probes > 1)
                     ? static_cast<double>(p) / static_cast<double>(n_probes - 1)
                     : 0.0;
      double baseline = target * x * x * x;
      diff[static_cast<size_t>(a) * n_probes + p] += baseline - target;
    }
  }

  // Numerical gradient along probe axis (matches torch.gradient with default
  // edge_order=1: central differences interior, one-sided at boundaries).
  std::vector<double> grad(diff.size(), 0.0);
  for (int a = 0; a < n_atoms; a++) {
    const double *d = &diff[static_cast<size_t>(a) * n_probes];
    double *g = &grad[static_cast<size_t>(a) * n_probes];
    if (n_probes == 1) {
      g[0] = std::abs(d[0]) * 0.5;
    } else {
      g[0] = d[1] - d[0];
      g[n_probes - 1] = d[n_probes - 1] - d[n_probes - 2];
      for (int p = 1; p < n_probes - 1; p++) {
        g[p] = 0.5 * (d[p + 1] - d[p - 1]);
      }
    }
  }
  // width = max(|grad|, eps)
  const double eps = 1e-12;
  for (size_t i = 0; i < grad.size(); i++) {
    grad[i] = std::max(std::abs(grad[i]), eps);
  }

  // Gaussian weights: log = -0.5 * (diff/width)^2 ; subtract max per row;
  // exponentiate; normalize.
  std::vector<float> result(n_atoms, 0.0f);
  for (int a = 0; a < n_atoms; a++) {
    const double *d = &diff[static_cast<size_t>(a) * n_probes];
    const double *w = &grad[static_cast<size_t>(a) * n_probes];

    // First pass: compute logw and find max
    std::vector<double> logw(n_probes);
    double logw_max = -std::numeric_limits<double>::infinity();
    for (int p = 0; p < n_probes; p++) {
      double r = d[p] / w[p];
      logw[p] = -0.5 * r * r;
      if (logw[p] > logw_max) logw_max = logw[p];
    }
    // Second pass: exp(logw - max) and accumulate
    double sum_w = 0.0;
    double weighted_cutoff = 0.0;
    for (int p = 0; p < n_probes; p++) {
      double weight = std::exp(logw[p] - logw_max);
      sum_w += weight;
      weighted_cutoff += weight * probe_cutoffs[p];
    }
    result[a] = (sum_w > 0.0)
                    ? static_cast<float>(weighted_cutoff / sum_w)
                    : max_cutoff;
  }
  return result;
}

// Bump function value AND derivative w.r.t. distance.
//   f(d, rc, w) = 0.5 * (1 + tanh(1/tan(pi*x))), x = (d-(rc-w))/w
//   f'(d) = 0.5 * sech^2(...) * d/dx[1/tan(pi*x)] / w
//         = 0.5 * sech^2 * (-pi/sin^2(pi*x)) / w
// Returns 0 outside the active region (since the bump is 0 or 1 there).
double bump_derivative(float distance, float cutoff, float width) {
  double x = (static_cast<double>(distance) -
              (static_cast<double>(cutoff) - static_cast<double>(width))) /
             static_cast<double>(width);
  if (x <= 0.0 || x >= 1.0) return 0.0;
  double pi_x = M_PI * x;
  double s = std::sin(pi_x);
  double c = std::cos(pi_x);
  double t = c / s;                  // 1/tan(pi*x)
  double sech2 = 1.0 - std::tanh(t) * std::tanh(t);
  // dt/dx = -pi/sin^2(pi*x) ; dx/d(distance) = 1/width
  double dt_dx = -M_PI / (s * s);
  return 0.5 * sech2 * dt_dx / static_cast<double>(width);
}

// Same as compute_adaptive_cutoffs but stashes the intermediate state
// (eff_nn, diff, width, softmax weights, the diff-Jacobian X) needed by the
// stress chain-rule correction.
void compute_adaptive_cutoffs_with_state(
    const std::vector<int32_t> &centers,
    const std::vector<float> &distances,
    float target,
    int n_atoms,
    float max_cutoff,
    AdaptiveCutoffCache &out,
    float min_cutoff = 0.5f,
    float effective_width = 1.0f) {
  out = {};
  const float probe_spacing = effective_width / 4.0f;
  for (float pc = min_cutoff; pc < max_cutoff; pc += probe_spacing) {
    out.probes.push_back(pc);
  }
  const int n_probes = static_cast<int>(out.probes.size());
  out.n_probes = n_probes;
  out.effective_width = effective_width;
  out.atomic_cutoffs.assign(n_atoms, max_cutoff);
  if (n_probes == 0 || n_atoms == 0) return;

  // diff[a*n_probes + p] starts as effective_num_neighbors, then we subtract
  // target and add baseline (kept as in compute_adaptive_cutoffs).
  out.diff.assign(static_cast<size_t>(n_atoms) * n_probes, 0.0);
  const size_t n_edges = distances.size();
  for (int p = 0; p < n_probes; p++) {
    const float pc = out.probes[p];
    for (size_t e = 0; e < n_edges; e++) {
      const int a = centers[e];
      out.diff[static_cast<size_t>(a) * n_probes + p] +=
          static_cast<double>(cutoff_func_bump(distances[e], pc, effective_width));
    }
  }
  for (int a = 0; a < n_atoms; a++) {
    for (int p = 0; p < n_probes; p++) {
      double x = (n_probes > 1)
                     ? static_cast<double>(p) / static_cast<double>(n_probes - 1)
                     : 0.0;
      out.diff[static_cast<size_t>(a) * n_probes + p] +=
          static_cast<double>(target) * x * x * x -
          static_cast<double>(target);
    }
  }

  // grad along probe axis (centred for interior, one-sided at ends), then
  // width = max(|grad|, eps). We also remember sign(grad) per row so the
  // Jacobian can use it without a redundant sign step.
  out.width.assign(out.diff.size(), 0.0);
  std::vector<double> grad(out.diff.size(), 0.0);
  std::vector<double> grad_sign(out.diff.size(), 0.0);
  const double eps = 1e-12;
  for (int a = 0; a < n_atoms; a++) {
    const double *d = &out.diff[static_cast<size_t>(a) * n_probes];
    double *g = &grad[static_cast<size_t>(a) * n_probes];
    if (n_probes == 1) {
      g[0] = std::abs(d[0]) * 0.5;
    } else {
      g[0] = d[1] - d[0];
      g[n_probes - 1] = d[n_probes - 1] - d[n_probes - 2];
      for (int p = 1; p < n_probes - 1; p++) {
        g[p] = 0.5 * (d[p + 1] - d[p - 1]);
      }
    }
    for (int p = 0; p < n_probes; p++) {
      double abs_g = std::abs(g[p]);
      out.width[static_cast<size_t>(a) * n_probes + p] = std::max(abs_g, eps);
      grad_sign[static_cast<size_t>(a) * n_probes + p] = (g[p] >= 0.0) ? 1.0 : -1.0;
    }
  }

  // Softmax weights and atomic_cutoffs.
  out.weights.assign(out.diff.size(), 0.0);
  for (int a = 0; a < n_atoms; a++) {
    const double *d = &out.diff[static_cast<size_t>(a) * n_probes];
    const double *w = &out.width[static_cast<size_t>(a) * n_probes];
    double *we = &out.weights[static_cast<size_t>(a) * n_probes];

    std::vector<double> logw(n_probes);
    double logw_max = -std::numeric_limits<double>::infinity();
    for (int p = 0; p < n_probes; p++) {
      double r = d[p] / w[p];
      logw[p] = -0.5 * r * r;
      if (logw[p] > logw_max) logw_max = logw[p];
    }
    double sum_w = 0.0;
    double weighted_cutoff = 0.0;
    for (int p = 0; p < n_probes; p++) {
      we[p] = std::exp(logw[p] - logw_max);
      sum_w += we[p];
      weighted_cutoff += we[p] * out.probes[p];
    }
    if (sum_w > 0.0) {
      for (int p = 0; p < n_probes; p++) we[p] /= sum_w;
      out.atomic_cutoffs[a] = static_cast<float>(weighted_cutoff / sum_w);
    } else {
      // Degenerate: keep uniform so downstream divisions are safe.
      double uniform = 1.0 / std::max(1, n_probes);
      for (int p = 0; p < n_probes; p++) we[p] = uniform;
    }
  }

  // Diff-Jacobian X[a, p_target] = d(atomic_cutoffs[a])/d(diff[a, p_target]).
  // Only logits[a, r] depends on diff[a, *]; chain through w via
  //   d(a)/d(logits_r) = w_r * (probes_r - atomic_cutoffs[a]).
  // logits_r = -0.5 * (diff_r / width_r)^2; width_r is a numerical gradient
  // of diff and so depends on diff_{r-1}, diff_{r+1} (or endpoints).
  out.diff_jacobian.assign(out.diff.size(), 0.0);
  for (int a = 0; a < n_atoms; a++) {
    const double *d = &out.diff[static_cast<size_t>(a) * n_probes];
    const double *w_arr = &out.width[static_cast<size_t>(a) * n_probes];
    const double *we = &out.weights[static_cast<size_t>(a) * n_probes];
    const double *gs = &grad_sign[static_cast<size_t>(a) * n_probes];
    const double a_val = static_cast<double>(out.atomic_cutoffs[a]);
    double *X = &out.diff_jacobian[static_cast<size_t>(a) * n_probes];

    for (int r = 0; r < n_probes; r++) {
      double coef = we[r] * (static_cast<double>(out.probes[r]) - a_val);
      double w_r = w_arr[r];
      if (w_r <= eps) continue;
      double inv_w2 = 1.0 / (w_r * w_r);
      double d_over_w3 = (d[r] * d[r]) / (w_r * w_r * w_r);

      // direct contribution: d(logits_r)/d(diff_r) = -d_r / w_r^2
      X[r] += coef * (-d[r] * inv_w2);

      // width contribution: d(logits_r)/d(diff_target) =
      //     (d_r^2 / w_r^3) * d(width_r)/d(diff_target)
      // d(width_r)/d(diff_target) = sign(grad_r) * d(grad_r)/d(diff_target).
      double sgn = gs[r];
      if (n_probes == 1) {
        // only r=0 case, with d(grad_0)/d(diff_0) ~ 0.5*sign already baked in
        // Skip — degenerate.
      } else if (r == 0) {
        // grad_0 = diff_1 - diff_0
        double t = coef * d_over_w3;
        X[0] += t * sgn * (-1.0);
        X[1] += t * sgn * (+1.0);
      } else if (r == n_probes - 1) {
        // grad_{n-1} = diff_{n-1} - diff_{n-2}
        double t = coef * d_over_w3;
        X[n_probes - 1] += t * sgn * (+1.0);
        X[n_probes - 2] += t * sgn * (-1.0);
      } else {
        // grad_r = 0.5 * (diff_{r+1} - diff_{r-1})
        double t = coef * d_over_w3;
        X[r - 1] += t * sgn * (-0.5);
        X[r + 1] += t * sgn * (+0.5);
      }
    }
  }
}

} // namespace

// Context sizes
static constexpr size_t INPUT_CTX_SIZE = 16 * 1024 * 1024;   // 16 MB
static constexpr size_t COMPUTE_CTX_SIZE = 512 * 1024 * 1024; // 512 MB

GraphModel::GraphModel()
    : neighbor_builder_(NeighborListOptions{cutoff_, true, false}) {}

void GraphModel::PlainGraphDeleter::operator()(PlainGraph *graph) const { delete graph; }

GraphModel::~GraphModel() {
  plain_graphs_.clear();
  cached_.reset();
  release_cached_resources();
  if (ragged_fold_buffer_) ggml_backend_buffer_free(ragged_fold_buffer_);
  if (ctx_ragged_fold_) ggml_free(ctx_ragged_fold_);
  if (weight_buffer_) {
    ggml_backend_buffer_free(weight_buffer_);
  }
  if (ctx_weights_) {
    ggml_free(ctx_weights_);
  }
  // compute_backend_ is owned by backend_provider_; do not free here.
}

bool GraphModel::load_from_gguf(const std::string &path) {
  constexpr size_t TEMP_CTX_SIZE = 512 * 1024 * 1024;
  plain_graphs_.clear(); // graphs of the previous weights

  // Create temporary context with data allocation
  ggml_context *temp_ctx = ggml_init({TEMP_CTX_SIZE, nullptr, false});
  if (!temp_ctx) {
    throw std::runtime_error("Failed to create temporary context for loading");
  }

  GGUFLoader loader(path, temp_ctx);
  int n_tensors = static_cast<int>(loader.get_tensor_names().size());

  // Read model hyperparameters
  cutoff_ = loader.get_float32("pet.cutoff", 4.5f);
  cutoff_width_ = loader.get_float32("pet.cutoff_width", 0.2f);
  energy_scale_ = loader.get_float32("pet.energy_scale", 1.0f);
  cutoff_function_ = loader.get_string("pet.cutoff_function", "cosine");
  forces_mode_ = (loader.get_int32("pet.forces_mode", 0) != 0);
  nc_forces_ = (loader.get_int32("pet.nc_forces", 0) != 0);
  param_names_ = json::parse(loader.get_string("train.param_names", "[]"))
                     .get<std::vector<std::string>>();
  gguf_path_ = path;
  num_neighbors_adaptive_ = loader.get_float32("pet.num_neighbors_adaptive", 0.0f);
  cutoff_width_adaptive_ = loader.get_float32("pet.cutoff_width_adaptive", cutoff_width_);
  adaptive_cutoff_method_ = loader.get_string("pet.adaptive_cutoff_method", "grid");
  // stable bump (graph_interpreter.h: stable_bump): GGUF pet.cutoff_bump_stable (absent = 0), MLIPCPP_PET_STABLE_BUMP=0|1
  // overrides; default off, bit-identical
  interp_.stable_bump = loader.get_int32("pet.cutoff_bump_stable", 0) != 0;
  if (const char *e = std::getenv("MLIPCPP_PET_STABLE_BUMP"); e && *e) interp_.stable_bump = *e != '0';
  num_heads_ = loader.get_int32("pet.num_heads", 0);
  conditioning_.enabled = loader.get_int32("pet.system_conditioning", 0) != 0;
  plato_lr_ = PetPlatoLRSpec::from_gguf(loader);
  if (plato_lr_.lrc.enabled && plato_lr_.lrc.graph) { // the frozen charge pass of the lrc q0 mode "graph"
    std::string qp = plato_lr_.lrc.qmodel;
    if (const char *e = std::getenv("MLIPCPP_LRC_QMODEL"); e && *e) qp = e;
    if (!qp.empty() && qp[0] != '/') {
      const auto slash = path.find_last_of('/');
      if (slash != std::string::npos) qp = path.substr(0, slash + 1) + qp;
    }
    plato_lr_.lrc.qsrc = make_lrc_graph_charges(qp, backend_preference_, device_);
    // per-atom Fukui passes (lrc_fukui_mode atom / atom_pm), paths relative to the model like lrc_qmodel
    auto rel = [&](std::string p) {
      if (!p.empty() && p[0] != '/') {
        const auto slash = path.find_last_of('/');
        if (slash != std::string::npos) p = path.substr(0, slash + 1) + p;
      }
      return p;
    };
    if (!plato_lr_.lrc.fmodel.empty())
      plato_lr_.lrc.fsrc = make_lrc_graph_charges(rel(plato_lr_.lrc.fmodel), backend_preference_, device_);
    if (!plato_lr_.lrc.fmodel_minus.empty())
      plato_lr_.lrc.fsrc_minus = make_lrc_graph_charges(rel(plato_lr_.lrc.fmodel_minus), backend_preference_, device_);
  }
  if (plato_lr_.enabled && !forces_mode_)
    throw std::runtime_error("Plato-PET-LR GGUFs are forces-mode exports");
  conditioning_.max_charge = loader.get_int32("pet.max_charge", 10);
  conditioning_.max_spin_multiplicity = loader.get_int32("pet.max_spin_multiplicity", 10);
  state_dict_names_ = json::parse(loader.get_string("train.state_dict_names", "{}"))
                          .get<std::map<std::string, std::string>>();

  const auto lr_solver = loader.get_string("pet.long_range.solver", "");
  if (!lr_solver.empty() && lr_solver != "direct-ewald" && lr_solver != "p3m") {
    ggml_free(temp_ctx);
    throw std::runtime_error("Unsupported PET-LR solver: " + lr_solver);
  }
  long_range_ = !lr_solver.empty();
  p3m_ = lr_solver == "p3m";
  if (p3m_ && loader.get_int32("pet.long_range.interpolation_nodes", 5) != 5) {
    ggml_free(temp_ctx);
    throw std::runtime_error("P3M currently requires interpolation_nodes=5");
  }
  if (long_range_ && (num_neighbors_adaptive_ > 0 || nc_forces_)) {
    ggml_free(temp_ctx);
    throw std::runtime_error(
        "PET-LR requires fixed cutoff and conservative forces");
  }
  coulomb_parameters_ = {
      cutoff_, loader.get_float32("pet.long_range.smearing", 1.4f),
      loader.get_float32("pet.long_range.wavelength", 1.33f)};

  if (long_range_ && (!std::isfinite(coulomb_parameters_.cutoff) ||
                      coulomb_parameters_.cutoff <= 0 ||
                      !std::isfinite(coulomb_parameters_.smearing) ||
                      coulomb_parameters_.smearing <= 0 ||
                      !std::isfinite(coulomb_parameters_.wavelength) ||
                      coulomb_parameters_.wavelength <= 0)) {
    ggml_free(temp_ctx);
    throw std::runtime_error("Invalid PET-LR Coulomb parameters");
  }

  // Update neighbor list builder with loaded cutoff
  neighbor_builder_ = NeighborListBuilder(NeighborListOptions{cutoff_, true, false});

  // Load graph JSON
  std::string graph_json = loader.get_string("graph.json", "");
  if (graph_json.empty()) {
    ggml_free(temp_ctx);
    throw std::runtime_error("No graph.json found in GGUF file");
  }
  interp_.load_graph(graph_json);
  if ((interp_.graph().get_input("coulomb_matrix") != nullptr) != long_range_) {
    ggml_free(temp_ctx);
    throw std::runtime_error("PET-LR graph input and solver metadata disagree");
  }

  // Load species mapping
  auto species_map = loader.get_array_int32("pet.species_map");
  for (size_t i = 0; i + 1 < species_map.size(); i += 2) {
    species_to_index_[species_map[i]] = species_map[i + 1];
  }

  // Load composition energies
  auto comp_keys = loader.get_array_int32("pet.composition_keys");
  auto comp_vals = loader.get_array_float32("pet.composition_values");
  if (comp_keys.size() != comp_vals.size()) {
    ggml_free(temp_ctx);
    throw std::runtime_error(
        "GraphModel: composition_keys and composition_values mismatch");
  }
  for (size_t i = 0; i < comp_keys.size(); i++) {
    composition_energies_[comp_keys[i]] = comp_vals[i];
  }
  auto ncs_keys = loader.get_array_int32("pet.nc_force_scale_keys");
  auto ncs_vals = loader.get_array_float32("pet.nc_force_scale_values");
  for (size_t i = 0; i < std::min(ncs_keys.size(), ncs_vals.size()); i++) {
    nc_force_scales_[ncs_keys[i]] = ncs_vals[i];
  }

  // Create backend
  backend_provider_ = BackendProvider::create(backend_preference_, device_);

  // Load weight shapes from metadata (PyTorch shapes, need reversal for GGML)
  std::string shapes_json = loader.get_string("graph.weight_shapes", "");
  json weight_shapes;
  if (!shapes_json.empty()) {
    weight_shapes = json::parse(shapes_json);
  }

  // Create weight context (metadata only, no data allocation)
  size_t ctx_size = ggml_tensor_overhead() * static_cast<size_t>(n_tensors);
  ctx_weights_ = ggml_init({ctx_size, nullptr, true});
  if (!ctx_weights_) {
    ggml_free(temp_ctx);
    throw std::runtime_error("Failed to create weight context");
  }

  // Create weight tensors with correct GGML shapes (reversed PyTorch dims)
  for (const auto &name : loader.get_tensor_names()) {
    ggml_tensor *temp = loader.get_tensor(name);
    if (!temp) continue;

    ggml_tensor *t = nullptr;
    if (weight_shapes.contains(name)) {
      // Use PyTorch shape from metadata, reversed for GGML convention
      auto py_shape = weight_shapes[name].get<std::vector<int64_t>>();
      std::vector<int64_t> ggml_shape(py_shape.rbegin(), py_shape.rend());
      switch (ggml_shape.size()) {
      case 0:
        t = ggml_new_tensor_1d(ctx_weights_, GGML_TYPE_F32, 1);
        break;
      case 1:
        t = ggml_new_tensor_1d(ctx_weights_, GGML_TYPE_F32, ggml_shape[0]);
        break;
      case 2:
        t = ggml_new_tensor_2d(ctx_weights_, GGML_TYPE_F32, ggml_shape[0],
                               ggml_shape[1]);
        break;
      case 3:
        t = ggml_new_tensor_3d(ctx_weights_, GGML_TYPE_F32, ggml_shape[0],
                               ggml_shape[1], ggml_shape[2]);
        break;
      default:
        continue;
      }
    } else {
      // Fallback: use GGUF stored shape directly
      t = ggml_new_tensor(
          ctx_weights_, temp->type, ggml_n_dims(temp), temp->ne);
    }

    ggml_set_name(t, name.c_str());
    weight_names_.push_back(name);
  }

  // Allocate backend buffer for weights
  ggml_backend_buffer_type_t buft = backend_provider_->buffer_type();
  weight_buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights_, buft);
  if (!weight_buffer_) {
    ggml_free(temp_ctx);
    throw std::runtime_error("Failed to allocate weight buffer");
  }
  ggml_backend_buffer_set_usage(weight_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

  // Copy weight data and register with interpreter
  for (const auto &name : loader.get_tensor_names()) {
    ggml_tensor *temp = loader.get_tensor(name);
    ggml_tensor *weight = ggml_get_tensor(ctx_weights_, name.c_str());
    if (temp && weight) {
      ggml_backend_tensor_set(weight, temp->data, 0, ggml_nbytes(weight));
      interp_.set_weight(name, weight);
    }
  }

  ggml_free(temp_ctx);

  // Use primary backend (may be GPU) for compute; owned by BackendProvider.
  compute_backend_ = backend_provider_->primary();
  interp_.set_backend(compute_backend_);
  if (!compute_backend_) {
    throw std::runtime_error("Failed to get compute backend");
  }

  return true;
}

std::vector<std::pair<std::string, ggml_tensor *>> GraphModel::weights() const {
  std::vector<std::pair<std::string, ggml_tensor *>> out;
  for (const auto &name : weight_names_) {
    out.emplace_back(name, ggml_get_tensor(ctx_weights_, name.c_str()));
  }
  return out;
}

void GraphModel::load_graph_file(const std::string &path) {
  interp_.load_graph_file(path);
}

void GraphModel::set_weight(const std::string &name, ggml_tensor *tensor) {
  interp_.set_weight(name, tensor);
}

ModelResult GraphModel::predict(const AtomicSystem &system) {
  return predict_single(system, false);
}

ModelResult GraphModel::predict(const AtomicSystem &system,
                                bool compute_forces) {
  return predict_single(system, compute_forces);
}

ModelResult GraphModel::predict_with_capture(
    const AtomicSystem &system,
    const std::function<void(GraphInterpreter &)> &cb) {
  return predict_single(system, /*compute_forces=*/false,
                        cb ? std::function<void()>([this, &cb]() { cb(interp_); })
                           : nullptr);
}

ModelResult GraphModel::predict_with_parameter_gradients(
    const AtomicSystem &system, std::vector<float> &gradients) {
  if (param_names_.empty())
    throw std::runtime_error("Model has no train.param_names metadata");
  struct RestoreFlags {
    std::vector<std::pair<ggml_tensor *, int>> entries;
    ~RestoreFlags() {
      for (auto [tensor, flags] : entries) tensor->flags = flags;
    }
  } restore;
  const auto all = weights();
  for (const auto &name : param_names_) {
    auto it = std::find_if(all.begin(), all.end(), [&](const auto &w) { return w.first == name; });
    if (it == all.end()) throw std::runtime_error("Missing trainable tensor: " + name);
    restore.entries.emplace_back(it->second, it->second->flags);
    ggml_set_param(it->second);
  }
  return predict_single(system, true, nullptr, &gradients);
}

std::vector<int32_t> GraphModel::charge_rows(int n_atoms, int charge, int spin_multiplicity, bool unit) const {
  const int32_t ci = conditioning_.rows(charge, spin_multiplicity).first;
  std::vector<int32_t> c(n_atoms, ci);
  if (plato_lr_.lrc.enabled) { // lower rows (frag_cond_) / upper rows (lrc_hi_) of the local-charge blend
    const std::vector<int32_t> &src = unit ? lrc_hi_ : frag_cond_;
    if (!src.empty()) {
      if (src.size() != static_cast<size_t>(n_atoms)) throw std::logic_error("lrc: atom count");
      for (int i = 0; i < n_atoms; i++) c[i] = conditioning_.rows(src[i], spin_multiplicity).first;
    }
    return c;
  }
  // gated: charge_index keeps the structure's row, charge_index_unit gets the units' rows
  if (!frag_cond_.empty() && unit == plato_lr_.frag.cond_gate) {
    if (frag_cond_.size() != static_cast<size_t>(n_atoms)) throw std::logic_error("frag_charges: atom count");
    for (int i = 0; i < n_atoms; i++) c[i] = conditioning_.rows(frag_cond_[i], spin_multiplicity).first;
  }
  return c;
}

// Fragment-constrained charges (runtime/frag_charges.h): an open structure with formal charges
// and a model with pet.plato_lr.frag_charges gets per-atom (bonded-unit) conditioning and the
// beyond-R Coulomb term of the unit charges (host, float64), added to the energy and forces.
// Anything else is the plain model, unchanged.
// Split-charge equilibration (runtime/sqe.h): with pet.plato_lr.sqe the SQE energy and Hellmann-Feynman forces
// (host, float64) are added to the model's; q0 from the per-atom formal charges when the model says sqe_q0 formal and
// the structure has them (else Q / N). sqe_backbone_charge neutral: the backbone is evaluated at charge 0.
ModelResult GraphModel::predict_single(
    const AtomicSystem &system, bool compute_forces,
    const std::function<void()> &post_compute_hook,
    std::vector<float> *parameter_gradients) {
  if (plato_lr_.lrc.enabled && system.num_atoms() > 0)
    return predict_single_lrc(system, compute_forces, post_compute_hook, parameter_gradients);
  const SqeSpec &sq = plato_lr_.sqe;
  if (!sq.enabled || system.num_atoms() == 0)
    return predict_single_frag(system, compute_forces, post_compute_hook, parameter_gradients);
  bool periodic = false;
  if (const Cell *c = system.cell()) periodic = c->periodic[0] || c->periodic[1] || c->periodic[2];
  if (periodic) throw std::runtime_error("sqe: periodic structures are not implemented in mlip.cpp yet");
  const int n = system.num_atoms();
  std::vector<double> q0;
  if (sq.q0_formal && system.has_formal_charges()) {
    if (system.formal_charges().size() != static_cast<size_t>(n)) throw std::runtime_error("formal_charges: one value per atom");
    q0.assign(system.formal_charges().begin(), system.formal_charges().end());
  }
  const SqeOutput s = sqe_compute(sq, n, system.atomic_numbers(), system.positions(), double(system.charge()),
                                  q0.empty() ? nullptr : q0.data(), compute_forces);
  ModelResult r;
  if (sq.backbone_neutral && system.charge() != 0) {
    AtomicSystem neutral = system;
    neutral.set_charge(0);
    r = predict_single_frag(neutral, compute_forces, post_compute_hook, parameter_gradients);
  } else {
    r = predict_single_frag(system, compute_forces, post_compute_hook, parameter_gradients);
  }
  r.energy = static_cast<float>(double(r.energy) + s.energy);
  if (compute_forces && r.forces.size() == s.forces.size())
    for (size_t k = 0; k < s.forces.size(); k++) r.forces[k] = static_cast<float>(double(r.forces[k]) + s.forces[k]);
  r.sqe_charges.assign(s.q.begin(), s.q.end());
  return r;
}

namespace {
// lrc q0 mode "graph": a plain GraphModel whose per-atom outputs are the raw charges (e)
class LrcGraphCharges final : public LrcQSource {
public:
  explicit LrcGraphCharges(std::unique_ptr<GraphModel> m) : m_(std::move(m)) {}
  std::vector<double> raw(int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell) override {
    const AtomicSystem sys = system(n, Z, x, Q, spin, cell);
    m_->set_atom_energy_weights({});
    const ModelResult r = m_->predict(sys, false);
    return std::vector<double>(r.atomic_energies.begin(), r.atomic_energies.end());
  }
  void raw_vjp(int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell, const double *w,
               double *g, double *vir) override {
    const AtomicSystem sys = system(n, Z, x, Q, spin, cell);
    std::vector<float> wf(w, w + n);
    m_->set_atom_energy_weights(wf);
    struct Reset {
      GraphModel &m;
      ~Reset() { m.set_atom_energy_weights({}); }
    } reset{*m_};
    const ModelResult r = m_->predict(sys, true);
    if (r.forces.size() != 3 * size_t(n)) throw std::runtime_error("lrc graph: charge pass returned no gradient");
    for (size_t k = 0; k < r.forces.size(); k++) g[k] -= double(r.forces[k]);
    if (!vir || !cell) return;
    if (r.stress.size() != 6) throw std::runtime_error("lrc graph: charge pass returned no stress");
    // stress (Voigt xx yy zz yz xz xy) = (1 / V) dE/d strain (V of the cell the pass saw: a slab's filled open row)
    const std::array<double, 9> hf = filled(cell);
    const double V = std::abs(hf[0] * (hf[4] * hf[8] - hf[5] * hf[7]) - hf[1] * (hf[3] * hf[8] - hf[5] * hf[6]) +
                              hf[2] * (hf[3] * hf[7] - hf[4] * hf[6]));
    constexpr int va[6] = {0, 1, 2, 1, 0, 0}, vb[6] = {0, 1, 2, 2, 2, 1};
    for (int k = 0; k < 6; k++) {
      vir[3 * va[k] + vb[k]] += V * r.stress[k];
      if (va[k] != vb[k]) vir[3 * vb[k] + va[k]] += V * r.stress[k];
    }
  }

private:
  // a slab's zero (open) row replaced by a 1000 A normal (non-periodic there, as mlipcpp-train's data loader does)
  static std::array<double, 9> filled(const double *cell) {
    std::array<double, 9> h;
    std::copy(cell, cell + 9, h.begin());
    for (int k = 0; k < 3; k++) {
      if (h[3 * k] != 0 || h[3 * k + 1] != 0 || h[3 * k + 2] != 0) continue;
      const double *u = &h[3 * ((k + 1) % 3)], *v = &h[3 * ((k + 2) % 3)];
      const double w[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
      const double nw = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
      if (!(nw > 0)) throw std::runtime_error("lrc graph: a cell with more than one open direction");
      for (int a = 0; a < 3; a++) h[3 * k + a] = 1000.0 * w[a] / nw;
    }
    return h;
  }
  static AtomicSystem system(int n, const int32_t *Z, const double *x, double Q, int spin, const double *cell) {
    std::vector<float> xf(x, x + 3 * size_t(n));
    std::optional<Cell> c;
    if (cell) {
      bool pbc[3];
      for (int k = 0; k < 3; k++) pbc[k] = cell[3 * k] != 0 || cell[3 * k + 1] != 0 || cell[3 * k + 2] != 0;
      const std::array<double, 9> hf = filled(cell);
      float h[3][3];
      for (int a = 0; a < 9; a++) h[a / 3][a % 3] = static_cast<float>(hf[a]);
      c.emplace(h, pbc[0], pbc[1], pbc[2]);
    }
    AtomicSystem sys(n, xf.data(), Z, c ? &*c : nullptr);
    sys.set_charge(static_cast<int>(std::lround(Q)));
    sys.set_spin_multiplicity(spin);
    return sys;
  }
  std::unique_ptr<GraphModel> m_;
};
} // namespace

std::shared_ptr<LrcQSource> make_lrc_graph_charges(const std::string &path, BackendPreference pref, int device) {
  auto m = std::make_unique<GraphModel>();
  m->set_force_uncached(true); // per-atom outputs
  m->set_backend_preference(pref);
  m->set_device(device);
  if (!m->load_from_gguf(path)) throw std::runtime_error("lrc graph: cannot load the charge pass " + path);
  if (m->plato_lr().lrc.enabled) throw std::runtime_error("lrc graph: the charge pass must be a plain model");
  return std::make_shared<LrcGraphCharges>(std::move(m));
}

// Combined long-range model (runtime/lr_combined.h): every atom conditioned on its smooth cluster's charge (blend of two
// table rows), plus the inter-cluster SQE long-range energy. A one-cluster structure is the plain model (global charge,
// no term). Forces: the graph's at fixed conditioning, the long-range term's (Hellmann-Feynman + q0 and cluster chain
// rules), and the conditioning's position dependence with dE/dS per super-node from the graph's gradient w.r.t. the
// blend input (only for structures inside a cluster transition or with a fractional cluster charge;
// MLIPCPP_LRC_FD_DS=1: central differences in S instead, h = 0.05, for checks).
ModelResult GraphModel::predict_single_lrc(const AtomicSystem &system, bool compute_forces,
                                           const std::function<void()> &post_compute_hook,
                                           std::vector<float> *parameter_gradients) {
  const LrcSpec &L = plato_lr_.lrc;
  const SqeSpec &sq = plato_lr_.sqe;
  // lattice rows (lr_combined.h): fully periodic cells, or slabs (lrc_slab) with the open direction's row zero; hv the
  // cell as given (its volume normalises the long-range stress like the backbone's)
  std::vector<double> cell, hv;
  if (const Cell *c = system.cell()) {
    const int np = c->periodic[0] + c->periodic[1] + c->periodic[2];
    if (np == 2 && L.slab == LrcSpec::Slab::None)
      throw std::runtime_error("lrc: periodic in all three directions or none (slabs need pet.plato_lr.lrc_slab)");
    if (np != 0 && np != 3 && np != 2) throw std::runtime_error("lrc: periodic in all three directions, two (a slab) or none");
    if (np >= 2)
      for (int a = 0; a < 9; a++)
        hv.push_back(double(c->matrix[a / 3][a % 3])), cell.push_back(c->periodic[a / 3] ? hv.back() : 0.0);
  }
  const int n = system.num_atoms();
  std::vector<double> x(system.positions(), system.positions() + 3 * size_t(n));
  const LrcCondition cond = lrc_condition(L, sq, n, system.atomic_numbers(), x.data(), double(system.charge()),
                                          conditioning_.max_charge, system.spin_multiplicity(),
                                          cell.empty() ? nullptr : cell.data());
  struct Reset {
    GraphModel &m;
    ~Reset() { m.frag_cond_.clear(), m.lrc_hi_.clear(), m.lrc_blend_.clear(), m.lrc_drop_.clear(); }
  } reset{*this};
  // exact dE/dS (graph gradient w.r.t. the blend input) for structures in a transition / with a fractional cluster
  // charge; MLIPCPP_LRC_FD_DS=1: the earlier central differences in S (h = 0.05), for checks
  static const bool fd_ds = [] { const char *e = std::getenv("MLIPCPP_LRC_FD_DS"); return e && *e == '1'; }();
  // lrc_states: the backbone energy as the weighted sum over the link states' graphs (lr_combined.h)
  // stress of the long-range / conditioning / state-weight virial (dE/d strain, 3 x 3) added to a Voigt stress (/ V)
  auto add_virial = [&](std::vector<float> &stress, const std::vector<double> &vir) {
    const double *h = hv.data();
    const double V = std::abs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                              h[2] * (h[3] * h[7] - h[4] * h[6]));
    constexpr int va[6] = {0, 1, 2, 1, 0, 0}, vb[6] = {0, 1, 2, 2, 2, 1};
    for (int k = 0; k < 6; k++)
      stress[k] = static_cast<float>(double(stress[k]) + 0.5 * (vir[3 * va[k] + vb[k]] + vir[3 * vb[k] + va[k]]) / V);
  };
  if (L.states && cond.cl.s > 1) { // (periodic cells too: one state applies to every image, lr_combined.h)
    const LrcStates st = lrc_states(L, cond.cl);
    if (st.active) {
      if (post_compute_hook) throw std::runtime_error("lrc_states: no capture hook on a multi-state structure");
      if (fd_ds) throw std::runtime_error("lrc_states: MLIPCPP_LRC_FD_DS is not supported");
      const size_t V = st.copies.size();
      std::vector<double> E(V), F(compute_forces ? 3 * size_t(n) : 0, 0.0), Ea(n, 0.0), Fnc, pg;
      std::vector<double> Sw, virw(9, 0.0); // periodic: sum_v w_v stress_v (Voigt) and sum_v w_v (conditioning virial)_v
      ModelResult first;
      for (size_t v = 0; v < V; v++) {
        const double w = st.copies[v].w;
        const LrcCondition hc = lrc_state_condition(L, cond, st, v, double(system.charge()), conditioning_.max_charge);
        frag_cond_ = hc.row_charge, lrc_hi_ = hc.row_charge_hi, lrc_blend_ = hc.blend;
        lrc_drop_ = st.drop_groups(cond.cl, v);
        lrc_want_blend_grad_ = compute_forces && hc.needs_cond_forces();
        std::vector<float> pgv;
        ModelResult rv = predict_single_impl(system, compute_forces, nullptr, parameter_gradients ? &pgv : nullptr);
        lrc_want_blend_grad_ = false;
        E[v] = double(rv.energy);
        for (int i = 0; i < n && size_t(i) < rv.atomic_energies.size(); i++) Ea[i] += w * double(rv.atomic_energies[i]);
        if (parameter_gradients) {
          if (pg.empty()) pg.assign(pgv.size(), 0.0);
          for (size_t k = 0; k < pgv.size(); k++) pg[k] += w * double(pgv[k]);
        }
        if (rv.has_nc_forces) {
          if (Fnc.empty()) Fnc.assign(rv.nc_forces.size(), 0.0);
          for (size_t k = 0; k < Fnc.size(); k++) Fnc[k] += w * double(rv.nc_forces[k]);
        }
        if (compute_forces) {
          std::vector<double> Fv(3 * size_t(n), 0.0);
          if (rv.forces.size() == Fv.size())
            for (size_t k = 0; k < Fv.size(); k++) Fv[k] = double(rv.forces[k]);
          static const bool no_cf = [] { const char *e = std::getenv("MLIPCPP_LRC_NO_CONDF"); return e && *e == '1'; }();
          std::vector<double> vv(9, 0.0);
          if (hc.needs_cond_forces() && !no_cf) {
            if (lrc_blend_grad_.size() != static_cast<size_t>(n)) throw std::runtime_error("lrc_states: no dE/dS");
            std::vector<double> G(hc.cl.s, 0.0);
            for (int i = 0; i < n; i++) G[hc.cl.sn[i]] += lrc_blend_grad_[i];
            lrc_condition_forces(L, sq, hc, n, system.atomic_numbers(), x.data(), G, Fv.data(),
                                 cell.empty() ? nullptr : vv.data());
          }
          for (size_t k = 0; k < Fv.size(); k++) F[k] += w * Fv[k];
          if (!cell.empty()) {
            for (int k = 0; k < 9; k++) virw[k] += w * vv[k];
            if (rv.stress.size() == 6) {
              if (Sw.empty()) Sw.assign(6, 0.0);
              for (int k = 0; k < 6; k++) Sw[k] += w * double(rv.stress[k]);
            }
          }
        }
        if (v == 0) first = std::move(rv);
      }
      // sum_v w_v E_v about a common reference (sum_v w_v = 1; the copies' energies are close)
      const double Eref = E[0];
      double Emix = 0;
      std::vector<double> dE(V);
      for (size_t v = 0; v < V; v++) dE[v] = E[v] - Eref, Emix += st.copies[v].w * dE[v];
      ModelResult r = std::move(first);
      r.atomic_energies.assign(Ea.begin(), Ea.end());
      if (!Fnc.empty()) r.nc_forces.assign(Fnc.begin(), Fnc.end());
      if (parameter_gradients) parameter_gradients->assign(pg.begin(), pg.end());
      LrcEnergy le = lrc_energy(L, sq, cond, n, system.atomic_numbers(), x.data(), double(system.charge()),
                                compute_forces);
      static const bool no_lr = [] { const char *e = std::getenv("MLIPCPP_LRC_NO_LR"); return e && *e == '1'; }();
      if (no_lr) le.energy = 0, std::fill(le.forces.begin(), le.forces.end(), 0.0);
      r.energy = static_cast<float>(Eref + Emix + le.energy);
      r.sqe_charges.assign(le.q.begin(), le.q.end());
      if (compute_forces) {
        std::vector<double> g(3 * size_t(n), 0.0), vs(9, 0.0); // d/dx sum_v E_v w_v (and d/d strain)
        lrc_states_vjp(L, cond.cl, st, dE, g.data(), cell.empty() ? nullptr : vs.data());
        r.forces.assign(3 * size_t(n), 0.0f);
        for (size_t k = 0; k < F.size(); k++)
          r.forces[k] = static_cast<float>(F[k] - g[k] + (k < le.forces.size() ? le.forces[k] : 0.0));
        r.has_forces = true;
        if (!cell.empty() && Sw.size() == 6 && r.stress.size() == 6) { // sum_v w_v sigma_v + the virials / V
          for (int k = 0; k < 6; k++) r.stress[k] = static_cast<float>(Sw[k]);
          for (int k = 0; k < 9; k++) vs[k] += virw[k] + (le.virial.size() == 9 && !no_lr ? le.virial[k] : 0.0);
          add_virial(r.stress, vs);
        }
      }
      return r;
    }
  }
  if (cond.cl.s > 1) frag_cond_ = cond.row_charge, lrc_hi_ = cond.row_charge_hi, lrc_blend_ = cond.blend;
  lrc_want_blend_grad_ = compute_forces && cond.cl.s > 1 && cond.needs_cond_forces() && !fd_ds;
  struct WantReset {
    bool &f;
    ~WantReset() { f = false; }
  } want_reset{lrc_want_blend_grad_};
  ModelResult r = predict_single_impl(system, compute_forces, post_compute_hook, parameter_gradients);
  lrc_want_blend_grad_ = false;
  const std::vector<double> blend_grad = lrc_blend_grad_;
  // one open cluster or a bulk cell: the plain model (lrc_intra: the long range still acts inside)
  if (cond.cl.s == 1 && !cond.cl.images() && !L.intra) return r;
  LrcEnergy le = lrc_energy(L, sq, cond, n, system.atomic_numbers(), x.data(), double(system.charge()), compute_forces);
  // MLIPCPP_LRC_NO_LR=1: drop the long-range energy and forces (diagnostics only: ablations on fixed weights)
  static const bool no_lr = [] { const char *e = std::getenv("MLIPCPP_LRC_NO_LR"); return e && *e == '1'; }();
  if (no_lr) le.energy = 0, std::fill(le.forces.begin(), le.forces.end(), 0.0);
  r.energy = static_cast<float>(double(r.energy) + le.energy);
  r.sqe_charges.assign(le.q.begin(), le.q.end());
  if (!compute_forces || r.forces.size() != le.forces.size()) return r;
  std::vector<double> F(le.forces), vir(le.virial);
  vir.resize(9, 0.0);
  // dE/dS from the graph, also for a structure without any edge (a backward pass for the blend alone there); the
  // central differences in S (h = 0.05) only on request: their truncation error was ~3 eV/A on bare ions (E04)
  // MLIPCPP_LRC_NO_CONDF=1: drop the conditioning forces (diagnostics only: force attribution on fixed weights)
  static const bool no_cf = [] { const char *e = std::getenv("MLIPCPP_LRC_NO_CONDF"); return e && *e == '1'; }();
  if (no_cf) {
  } else if (cond.needs_cond_forces() && !fd_ds) {
    if (blend_grad.size() != static_cast<size_t>(n)) throw std::runtime_error("lrc: no dE/dS from the graph");
    std::vector<double> G(cond.cl.s, 0.0);
    for (int i = 0; i < n; i++) G[cond.cl.sn[i]] += blend_grad[i];
    lrc_condition_forces(L, sq, cond, n, system.atomic_numbers(), x.data(), G, F.data(), vir.data());
  } else if (cond.needs_cond_forces()) {
    const double h = 5e-2; // E is smooth in S; float32 energies: noise ~1e-5 eV / 2h
    std::vector<double> G(cond.cl.s, 0.0);
    const std::vector<float> base = cond.blend;
    for (int a = 0; a < cond.cl.s; a++) {
      if (cond.dS[a] == 0) continue;
      double Epm[2];
      for (int sgn = 0; sgn < 2; sgn++) {
        lrc_blend_ = base;
        for (int i = 0; i < n; i++)
          if (cond.cl.sn[i] == a) lrc_blend_[i] = static_cast<float>(cond.S[a] + (sgn ? -h : h));
        Epm[sgn] = predict_single_impl(system, false, nullptr, nullptr).energy;
      }
      G[a] = (Epm[0] - Epm[1]) / (2 * h);
    }
    lrc_blend_ = base;
    lrc_condition_forces(L, sq, cond, n, system.atomic_numbers(), x.data(), G, F.data(), vir.data());
  }
  for (size_t k = 0; k < F.size(); k++) r.forces[k] = static_cast<float>(double(r.forces[k]) + F[k]);
  if (!cell.empty() && r.stress.size() == 6) add_virial(r.stress, vir); // the long-range and conditioning virial
  return r;
}

ModelResult GraphModel::predict_single_frag(
    const AtomicSystem &system, bool compute_forces,
    const std::function<void()> &post_compute_hook,
    std::vector<float> *parameter_gradients) {
  bool open = !system.cell();
  if (const Cell *c = system.cell()) open = !c->periodic[0] && !c->periodic[1] && !c->periodic[2];
  const int n = system.num_atoms();
  if (!(plato_lr_.frag.enabled && conditioning_.enabled && open && system.has_formal_charges() && n > 0))
    return predict_single_impl(system, compute_forces, post_compute_hook, parameter_gradients);
  if (system.formal_charges().size() != static_cast<size_t>(n))
    throw std::runtime_error("formal_charges: one value per atom");
  const std::vector<int32_t> units = frag_bonded_units(n, system.atomic_numbers(), system.positions());
  std::vector<double> q;
  std::vector<int32_t> unit_charge;
  frag_unit_charges(n, units, system.formal_charges().data(), unit_charge, q);
  struct Reset {
    std::vector<int32_t> &v;
    ~Reset() { v.clear(); }
  } reset{frag_cond_};
  frag_cond_ = unit_charge;
  ModelResult r = predict_single_impl(system, compute_forces, post_compute_hook, parameter_gradients);
  std::vector<double> F(compute_forces ? 3 * size_t(n) : 0);
  const double E = frag_coulomb(plato_lr_.frag, n, system.positions(), q.data(), compute_forces ? F.data() : nullptr);
  r.energy = static_cast<float>(double(r.energy) + E);
  if (compute_forces && r.forces.size() == F.size())
    for (size_t k = 0; k < F.size(); k++) r.forces[k] = static_cast<float>(double(r.forces[k]) + F[k]);
  return r;
}

ModelResult GraphModel::predict_single_impl(
    const AtomicSystem &system, bool compute_forces,
    const std::function<void()> &post_compute_hook,
    std::vector<float> *parameter_gradients) {
  // Full FP32 on CUDA, as the MD engine: TF32 GEMMs (cuBLAS and ggml's mmf
  // kernels) put energies up to 14 meV/atom off the CPU / metatrain values on
  // MPtrj test structures, at no measurable speed gain here. MLIPCPP_TF32=1
  // keeps TF32. (A process-wide setting, read by the CUDA kernels per call.)
  static const bool full_fp32 = [] {
    if (std::getenv("MLIPCPP_TF32")) return false;
    setenv("GGML_CUDA_NO_TF32", "1", 0);
    return true;
  }();
  (void)full_fp32;
  static const bool no_cache = [] {
    const char *e = std::getenv("MLIPCPP_NO_GRAPH_CACHE");
    return e && *e && *e != '0';
  }();
  // per-atom outputs / weights and dE/d(blend) come from the plain graph only
  if (!no_cache && !post_compute_hook && !parameter_gradients && cached_path_supported() &&
      system.num_atoms() > 0 && !force_uncached_ && atom_w_.empty() && !lrc_want_blend_grad_ && lrc_drop_.empty())
    return predict_cached(system, compute_forces);
  GraphProfile profile;
  if (compute_forces && !forces_mode_) {
    throw std::runtime_error(
        "GraphModel: forces requested but model was exported with "
        "--no-forces. Re-export without --no-forces to enable forces.");
  }

  const int n_atoms = static_cast<int>(system.num_atoms());
  const int32_t *atomic_numbers = system.atomic_numbers();

  // Build neighbor list at the global model cutoff (the search radius).
  // Validate LR geometry before building the full, unfiltered neighbor list.
  if (long_range_ && n_atoms < 1)
    throw std::runtime_error("PET-LR requires at least one atom");
  if (long_range_ && !p3m_ && n_atoms > 512)
    throw std::runtime_error(
        "PET-LR dense reference solver supports 1..512 atoms");
  if (long_range_ && system.cell()) {
    int count = 0;
    for (bool b : system.cell()->periodic)
      count += b;
    const auto &c = system.cell()->matrix;
    double determinant =
        double(c[0][0]) *
            (double(c[1][1]) * c[2][2] - double(c[1][2]) * c[2][1]) -
        double(c[0][1]) *
            (double(c[1][0]) * c[2][2] - double(c[1][2]) * c[2][0]) +
        double(c[0][2]) *
            (double(c[1][0]) * c[2][1] - double(c[1][1]) * c[2][0]);
    if (!std::isfinite(determinant) || std::abs(determinant) < 1e-12)
      throw std::runtime_error(
          "PET-LR requires a nonsingular cell when a Cell is supplied");
    if (count != 0 && count != 3)
      throw std::runtime_error(
          "PET-LR supports nonperiodic or fully periodic cells only");
  }
  if (long_range_) {
    for (int i = 0; i < 3 * n_atoms; ++i)
      if (!std::isfinite(system.positions()[i]))
        throw std::runtime_error("PET-LR requires finite positions");
  }
  NeighborList nlist = neighbor_builder_.build(system);
  CoulombResult coulomb;
  if (long_range_ && !p3m_)
    coulomb = coulomb_matrix(system, nlist, coulomb_parameters_);

  // Per-pair cutoffs: default to global cutoff. When the graph itself does
  // adaptive cutoff (adaptive_cutoff_in_graph_), C++ leaves pair_cutoffs at
  // the global cutoff and skips filtering — the graph computes per-pair
  // cutoffs from edge_distances and applies them via cutoff_factors.
  // Otherwise, when num_neighbors_adaptive_ > 0, C++ does the legacy
  // filtering and passes per-pair cutoffs as a constant input.
  std::vector<float> pair_cutoffs(nlist.num_pairs(), cutoff_);

  // Pre-filter (full) neighbor list and adaptive-cutoff state are kept here
  // so the stress chain-rule correction can use them after the backward pass
  // (the model only sees the post-filter list, but atomic_cutoffs is computed
  // on the full list).
  NeighborList full_nlist;
  AdaptiveCutoffCache adapt_cache;
  bool have_adaptive = false;

  // d atomic_cutoff[center] / d distance, per edge of the full list (the
  // implicit position dependence the graph does not see)
  std::vector<double> cutoff_jacobian;
  const bool solver = adaptive_cutoff_method_ == "solver";

  if (num_neighbors_adaptive_ > 0.0f) {
    full_nlist = nlist;  // copy of pre-filter list
    const int n_in = nlist.num_pairs();
    std::vector<float> atomic_cutoffs;
    std::vector<int> canonical(n_in);  // the edge of the pair (i->j, j->i) whose distance decides
    std::iota(canonical.begin(), canonical.end(), 0);
    if (solver) {
      // as in training (train/dataset.cpp): radii in double from the edges
      // of each centre, one keep decision per edge pair
      std::vector<std::vector<int>> by_center(n_atoms);
      for (int e = 0; e < n_in; e++) by_center[nlist.centers[e]].push_back(e);
      const bool shifts = !nlist.cell_shifts.empty();
      auto key = [&](int i, int j, int s0, int s1, int s2) { return std::tuple{i, j, s0, s1, s2}; };
      std::map<std::tuple<int, int, int, int, int>, int> index;
      for (int e = 0; e < n_in; e++) {
        const auto s = shifts ? nlist.cell_shifts[e] : std::array<int32_t, 3>{0, 0, 0};
        index[key(nlist.centers[e], nlist.neighbors[e], s[0], s[1], s[2])] = e;
      }
      for (int e = 0; e < n_in; e++) {
        const auto s = shifts ? nlist.cell_shifts[e] : std::array<int32_t, 3>{0, 0, 0};
        auto it = index.find(key(nlist.neighbors[e], nlist.centers[e], -s[0], -s[1], -s[2]));
        if (it != index.end()) canonical[e] = std::min(e, it->second);
      }
      auto dist = [&](int e) {
        const auto &v = nlist.edge_vectors[e];
        return std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]) + 1e-15;
      };
      std::vector<adaptive::Radius> radii(n_atoms);
      atomic_cutoffs.resize(n_atoms);
      for (int i = 0; i < n_atoms; i++) {
        const auto &es = by_center[i];
        radii[i] = adaptive::solve(static_cast<int>(es.size()), [&](int k) { return dist(es[k]); },
                                   cutoff_, num_neighbors_adaptive_, cutoff_width_adaptive_);
        atomic_cutoffs[i] = static_cast<float>(radii[i].r);
      }
      cutoff_jacobian.resize(n_in);
      for (int e = 0; e < n_in; e++)
        cutoff_jacobian[e] = adaptive::ddist(dist(e), radii[nlist.centers[e]], cutoff_width_adaptive_);
      // the keep test and pair cutoff in double, as training caches them
      std::vector<double> pc(n_in), d(n_in);
      for (int e = 0; e < n_in; e++)
        pc[e] = 0.5 * (radii[nlist.centers[e]].r + radii[nlist.neighbors[e]].r), d[e] = dist(e);
      std::vector<char> keep(n_in);
      for (int e = 0; e < n_in; e++) keep[e] = d[canonical[e]] <= pc[canonical[e]];
      int n_out = 0;
      std::vector<float> kept_pair_cutoffs;
      for (int e = 0; e < n_in; e++) {
        if (!keep[e]) continue;
        nlist.centers[n_out] = nlist.centers[e];
        nlist.neighbors[n_out] = nlist.neighbors[e];
        nlist.edge_vectors[n_out] = nlist.edge_vectors[e];
        nlist.distances[n_out] = nlist.distances[e];
        if (shifts) nlist.cell_shifts[n_out] = nlist.cell_shifts[e];
        kept_pair_cutoffs.push_back(static_cast<float>(pc[e]));
        ++n_out;
      }
      nlist.centers.resize(n_out);
      nlist.neighbors.resize(n_out);
      nlist.edge_vectors.resize(n_out);
      nlist.distances.resize(n_out);
      if (shifts) nlist.cell_shifts.resize(n_out);
      pair_cutoffs = std::move(kept_pair_cutoffs);
    } else {
    compute_adaptive_cutoffs_with_state(
        nlist.centers, nlist.distances,
        num_neighbors_adaptive_, n_atoms, cutoff_, adapt_cache,
        /*min_cutoff=*/0.5f,
        /*effective_width=*/cutoff_width_adaptive_);
    atomic_cutoffs = adapt_cache.atomic_cutoffs;

    // Compact nlist in place: keep edge e iff distances[e] <= pair_cutoff
    std::vector<float> kept_pair_cutoffs;
    kept_pair_cutoffs.reserve(n_in);
    int n_out = 0;
    for (int e = 0; e < n_in; e++) {
      const int i = nlist.centers[e];
      const int j = nlist.neighbors[e];
      const float pc = 0.5f * (atomic_cutoffs[i] + atomic_cutoffs[j]);
      if (nlist.distances[e] > pc) continue;
      if (n_out != e) {
        nlist.centers[n_out] = nlist.centers[e];
        nlist.neighbors[n_out] = nlist.neighbors[e];
        nlist.edge_vectors[n_out] = nlist.edge_vectors[e];
        nlist.distances[n_out] = nlist.distances[e];
        if (!nlist.cell_shifts.empty()) {
          nlist.cell_shifts[n_out] = nlist.cell_shifts[e];
        }
      }
      kept_pair_cutoffs.push_back(pc);
      ++n_out;
    }
    nlist.centers.resize(n_out);
    nlist.neighbors.resize(n_out);
    nlist.edge_vectors.resize(n_out);
    nlist.distances.resize(n_out);
    if (!nlist.cell_shifts.empty()) nlist.cell_shifts.resize(n_out);
    pair_cutoffs = std::move(kept_pair_cutoffs);
    // J[e] = sum_p X[k, p] * bump'(d, probe_p)
    cutoff_jacobian.resize(n_in);
    for (int e = 0; e < n_in; e++) {
      const int k = full_nlist.centers[e];
      const float d = full_nlist.distances[e];
      double J = 0.0;
      if (d > 0.0f) {
        const double *Xk = &adapt_cache.diff_jacobian[static_cast<size_t>(k) * adapt_cache.n_probes];
        for (int p = 0; p < adapt_cache.n_probes; p++)
          J += Xk[p] * bump_derivative(d, adapt_cache.probes[p], adapt_cache.effective_width);
      }
      cutoff_jacobian[e] = J;
    }
    }
    have_adaptive = true;
  }

  // lrc_states: the copy's graph has no edge between groups its link state separates (the radii, pair cutoffs and the
  // radius chain above stay those of the full structure)
  if (!lrc_drop_.empty()) {
    if (lrc_drop_.size() != static_cast<size_t>(n_atoms)) throw std::logic_error("lrc_states: drop groups");
    auto dropped = [&](int i, int j) {
      const int32_t a = lrc_drop_[i], b = lrc_drop_[j];
      return a >= 0 && b >= 0 && a != b && (a >> 20) == (b >> 20);
    };
    const bool shifts = !nlist.cell_shifts.empty();
    int n_out = 0;
    for (int e = 0; e < nlist.num_pairs(); e++) {
      if (dropped(nlist.centers[e], nlist.neighbors[e])) continue;
      nlist.centers[n_out] = nlist.centers[e];
      nlist.neighbors[n_out] = nlist.neighbors[e];
      nlist.edge_vectors[n_out] = nlist.edge_vectors[e];
      nlist.distances[n_out] = nlist.distances[e];
      if (shifts) nlist.cell_shifts[n_out] = nlist.cell_shifts[e];
      pair_cutoffs[n_out] = pair_cutoffs[e];
      ++n_out;
    }
    nlist.centers.resize(n_out);
    nlist.neighbors.resize(n_out);
    nlist.edge_vectors.resize(n_out);
    nlist.distances.resize(n_out);
    if (shifts) nlist.cell_shifts.resize(n_out);
    pair_cutoffs.resize(n_out);
  }

  // Count max neighbors after any adaptive filtering
  std::vector<int> neighbor_counts(n_atoms, 0);
  for (int e = 0; e < nlist.num_pairs(); e++) {
    neighbor_counts[nlist.centers[e]]++;
  }
  int max_neighbors = 0;
  for (int i = 0; i < n_atoms; i++) {
    max_neighbors = std::max(max_neighbors, neighbor_counts[i]);
  }
  // A system without any edge: metatrain has no neighbour slot, so the
  // attention sees the central token alone. ggml needs at least one slot, and
  // through PET's finite -34.5 bias a padded slot still competes with the
  // central token (isolated atoms up to eV off); so it is masked exactly
  // (-inf) here. The energy then does not depend on the positions: no
  // backward pass for the forces, which are zero (with stress). The combined
  // long-range model's conditioning blend still needs dE/dS (its positions
  // dependence goes through the clusters): a backward pass for the blend alone.
  const bool no_edges = max_neighbors == 0, forces_requested = compute_forces;
  const bool blend_only = no_edges && forces_requested && lrc_want_blend_grad_;
  if (no_edges) {
    max_neighbors = 1;
    compute_forces = false;
  }
  const bool backward = compute_forces || blend_only;
  const bool exact_mask = no_edges && !std::getenv("MLIPCPP_PAD_EMPTY");
  const int total_slots = n_atoms * max_neighbors;

  int periodic_dims = 0;
  if (const Cell *c = system.cell()) periodic_dims = c->periodic[0] + c->periodic[1] + c->periodic[2];
  PetPlatoLRStructure plato_st;
  if (plato_lr_.enabled) {
    plato_st.a0 = 0, plato_st.a1 = n_atoms;
    if (const Cell *c = system.cell()) {
      if (periodic_dims != 0 && periodic_dims != 3 && plato_lr_.long_range())
        throw std::runtime_error("Plato-PET-LR long range needs full or no periodicity");
      plato_st.periodic = periodic_dims == 3;
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) plato_st.cell[a][b] = c->matrix[a][b];
    }
    plato_st.charge = system.charge();
  }

  // The graph of this exact shape (graph_cache.h PlainGraph): built on the first call, later calls only upload its
  // inputs. Separated clusters of the combined long-range model come this way on every call (charge pass and its
  // vector-Jacobian product, dE/d blend, link-state copies), where the rebuild (graph, backward graph, device buffers,
  // one synchronous upload per constant) cost several times the compute and serialised processes sharing the GPU.
  const bool keep_graph = !no_cache && !post_compute_hook && !parameter_gradients && !long_range_;
  const PlainGraph::Key key{n_atoms,
                            max_neighbors,
                            periodic_dims,
                            compute_forces,
                            backward,
                            backward && lrc_want_blend_grad_,
                            compute_forces && !atom_w_.empty(),
                            exact_mask};
  std::unique_ptr<PlainGraph, PlainGraphDeleter> fresh;
  PlainGraph *pg = nullptr;
  if (keep_graph) {
    for (size_t k = 0; k < plain_graphs_.size(); k++) {
      if (plain_graphs_[k]->key != key) continue;
      if (plain_graphs_[k]->plato && !plain_graphs_[k]->plato->refresh({plato_st})) { // another mesh / k set
        plain_graphs_.erase(plain_graphs_.begin() + static_cast<std::ptrdiff_t>(k));
        break;
      }
      std::rotate(plain_graphs_.begin(), plain_graphs_.begin() + static_cast<std::ptrdiff_t>(k),
                  plain_graphs_.begin() + static_cast<std::ptrdiff_t>(k) + 1);
      pg = plain_graphs_.front().get();
      break;
    }
  }
  const bool rebuilt = pg == nullptr;
  if (profile.enabled)
    std::fprintf(stderr, "GRAPH_PLAIN n=%d M=%d forces=%d backward=%d blend_grad=%d atom_w=%d no_edges=%d %s\n", n_atoms,
                 max_neighbors, int(key.forces), int(key.backward), int(key.blend_grad), int(key.atom_w),
                 int(key.no_edges), rebuilt ? (keep_graph ? "build" : "build-uncached") : "reuse");

  // --- Pack neighbor list data ---
  std::vector<int32_t> species_data(n_atoms);
  for (int i = 0; i < n_atoms; i++) {
    auto it = species_to_index_.find(atomic_numbers[i]);
    if (it == species_to_index_.end()) {
      throw std::runtime_error(
          "Atomic number " + std::to_string(atomic_numbers[i]) +
          " not in species map");
    }
    species_data[i] = it->second;
  }

  std::vector<int32_t> ns_data(total_slots, padding::kSpecies);
  // padded slots: core/padding.h (an ordinary edge at the cutoff, as metatrain)
  std::vector<float> ev_data(total_slots * 3);
  std::vector<float> ed_data(total_slots, cutoff_);
  for (int f = 0; f < total_slots; f++) padding::edge(cutoff_, &ev_data[3 * f]);
  std::vector<float> pm_data(total_slots, 1.0f);  // 1.0 = padded
  std::vector<float> cf_data(total_slots, 0.0f);
  std::vector<float> cv_data(total_slots, cutoff_);
  std::vector<int32_t> rni_data(total_slots, 0);
  std::vector<int> neighbor_atoms_vec(total_slots, -1);

  // Build edge mapping
  using EdgeKey = std::tuple<int, int, int, int, int>;
  std::map<EdgeKey, int> edge_to_flat_idx;
  std::vector<int> slot_indices(n_atoms, 0);
  bool has_cell_shifts = !nlist.cell_shifts.empty();

  for (int e = 0; e < nlist.num_pairs(); e++) {
    int i = nlist.centers[e];
    int j = nlist.neighbors[e];
    int slot = slot_indices[i]++;
    if (slot >= max_neighbors) continue;

    int flat_idx = i * max_neighbors + slot;

    int sa = 0, sb = 0, sc = 0;
    if (has_cell_shifts) {
      sa = nlist.cell_shifts[e][0];
      sb = nlist.cell_shifts[e][1];
      sc = nlist.cell_shifts[e][2];
    }
    edge_to_flat_idx[{i, j, sa, sb, sc}] = flat_idx;

    auto it = species_to_index_.find(atomic_numbers[j]);
    if (it == species_to_index_.end()) {
      throw std::runtime_error(
          "Atomic number " + std::to_string(atomic_numbers[j]) +
          " not in species map");
    }
    ns_data[flat_idx] = it->second;

    const auto &ev = nlist.edge_vectors[e];
    int ev_idx = i * (max_neighbors * 3) + slot * 3;
    ev_data[ev_idx + 0] = ev[0];
    ev_data[ev_idx + 1] = ev[1];
    ev_data[ev_idx + 2] = ev[2];

    ed_data[flat_idx] = nlist.distances[e];
    pm_data[flat_idx] = 0.0f;  // valid edge
    neighbor_atoms_vec[flat_idx] = j;

    float r = nlist.distances[e];
    float pc = pair_cutoffs[e];
    cv_data[flat_idx] = pc;
    if (cutoff_function_ == "bump") {
      cf_data[flat_idx] = cutoff_func_bump(r, pc, cutoff_width_);
    } else {
      cf_data[flat_idx] = cutoff_func_cosine(r, pc, cutoff_width_);
    }
  }

  // Build reverse neighbor index
  for (int e = 0; e < nlist.num_pairs(); e++) {
    int i = nlist.centers[e];
    int j = nlist.neighbors[e];
    int sa = 0, sb = 0, sc = 0;
    if (has_cell_shifts) {
      sa = nlist.cell_shifts[e][0];
      sb = nlist.cell_shifts[e][1];
      sc = nlist.cell_shifts[e][2];
    }
    auto it_ij = edge_to_flat_idx.find({i, j, sa, sb, sc});
    if (it_ij == edge_to_flat_idx.end()) continue;
    auto it_ji = edge_to_flat_idx.find({j, i, -sa, -sb, -sc});
    if (it_ji != edge_to_flat_idx.end()) {
      rni_data[it_ij->second] = it_ji->second;
    }
  }

  ggml_tensor *species_t = nullptr, *neighbor_species_t = nullptr, *edge_vectors_t = nullptr,
              *padding_mask_t = nullptr, *reverse_neighbor_index_t = nullptr, *coulomb_t = nullptr,
              *charge_index_t = nullptr, *spin_index_t = nullptr, *charge_index_unit_t = nullptr,
              *cond_blend_t = nullptr, *atom_w_t = nullptr, *edge_distances_t = nullptr,
              *cutoff_factors_t = nullptr, *cutoff_values_t = nullptr, *output = nullptr, *nc_output = nullptr,
              *latent_q = nullptr;
  ggml_tensor *lr_charges = nullptr, *lr_potentials = nullptr,
              *lr_charge_adjoint = nullptr;
  ggml_cgraph *cgraph = nullptr, *charge_graph = nullptr;
  ggml_tensor *loss_t = nullptr;
  if (rebuilt) {
  fresh.reset(new PlainGraph());
  pg = fresh.get();
  pg->key = key;
  interp_.set_exact_padding_mask(exact_mask);

  // Set symbolic dimensions for this system
  interp_.set_dimension("n_atoms", n_atoms);
  interp_.set_dimension("max_neighbors", max_neighbors);
  interp_.set_dimension("n_edges", n_atoms * max_neighbors);
  interp_.set_dimension("max_neighbors_plus_one", max_neighbors + 1);

  // --- Create input context ---
  ggml_context *input_ctx = pg->input_ctx = ggml_init({INPUT_CTX_SIZE, nullptr, true});
  if (!input_ctx) {
    throw std::runtime_error("Failed to create input context");
  }

  // Create input tensors
  species_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
  ggml_set_name(species_t, "species");

  neighbor_species_t =
      ggml_new_tensor_2d(input_ctx, GGML_TYPE_I32, max_neighbors, n_atoms);
  ggml_set_name(neighbor_species_t, "neighbor_species");

  edge_vectors_t =
      ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, 3, max_neighbors, n_atoms);
  ggml_set_name(edge_vectors_t, "edge_vectors");

  padding_mask_t =
      ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(padding_mask_t, "padding_mask");

  reverse_neighbor_index_t =
      ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, total_slots);
  ggml_set_name(reverse_neighbor_index_t, "reverse_neighbor_index");

  if (long_range_ && !p3m_) {
    coulomb_t = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, n_atoms, n_atoms);
    ggml_set_name(coulomb_t, "coulomb_matrix");
    if (compute_forces)
      ggml_set_param(coulomb_t);
  }

  // charge / spin conditioning: the table rows of the structure, per atom
  if (conditioning_.enabled && (plato_lr_.frag.cond_gate || plato_lr_.lrc.enabled)) {
    charge_index_unit_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
    ggml_set_name(charge_index_unit_t, "charge_index_unit");
  }
  if (conditioning_.enabled && plato_lr_.lrc.enabled) {
    cond_blend_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, n_atoms);
    ggml_set_name(cond_blend_t, "cond_blend");
    if (backward && lrc_want_blend_grad_) ggml_set_param(cond_blend_t); // dE/dS (exact conditioning forces)
  }
  if (compute_forces && !atom_w_.empty()) {
    if (atom_w_.size() != static_cast<size_t>(n_atoms)) throw std::logic_error("atom energy weights: atom count");
    atom_w_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, n_atoms);
    ggml_set_name(atom_w_t, "atom_energy_weights");
  }
  if (conditioning_.enabled) {
    charge_index_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
    ggml_set_name(charge_index_t, "charge_index");
    spin_index_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
    ggml_set_name(spin_index_t, "spin_index");
  }

  // Mode-specific inputs
  if (!forces_mode_) {
    edge_distances_t =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(edge_distances_t, "edge_distances");

    cutoff_factors_t =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(cutoff_factors_t, "cutoff_factors");
  } else {
    cutoff_values_t =
        ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
    ggml_set_name(cutoff_values_t, "cutoff_values");
  }

  // Mark edge_vectors as parameter for gradient computation
  if (compute_forces) {
    ggml_set_param(edge_vectors_t);
    // Adaptive cutoff models also need ∂E/∂cutoff_values to recover the
    // missing chain in the virial stress (correction added in C++).
    if (cutoff_values_t && num_neighbors_adaptive_ > 0.0f) {
      ggml_set_param(cutoff_values_t);
    }
  }

  // Plato-PET-LR: positions (and strain) inputs of the long-range ops, read by the
  // interpreter's MLIPCPP_* builders
  if (plato_lr_.enabled) {
    pg->plato = std::make_unique<PetPlatoLR>();
    pg->plato->create(input_ctx, plato_lr_, {plato_st}, n_atoms, compute_forces && plato_lr_.long_range());
  }
  PetPlatoLR *plato = pg->plato.get();

  // Allocate input buffer
  ggml_backend_buffer_t input_buffer = pg->input_buffer =
      ggml_backend_alloc_ctx_tensors(input_ctx, compute_backend_);
  if (!input_buffer) {
    throw std::runtime_error("Failed to allocate input buffer");
  }

  if (plato) {
    interp_.custom_op = [plato](ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in) {
      return plato->build(ctx, node, in);
    };
  }

  // Register the inputs
  if (coulomb_t) interp_.set_input("coulomb_matrix", coulomb_t);
  if (conditioning_.enabled) {
    interp_.set_input("charge_index", charge_index_t);
    interp_.set_input("spin_index", spin_index_t);
    if (charge_index_unit_t) {
      interp_.set_input("charge_index_unit", charge_index_unit_t);
      if (cond_blend_t) {
        interp_.set_input("cond_blend", cond_blend_t);
        install_lrc_blend(interp_, cond_blend_t);
      } else {
        install_frag_cond_gate(interp_);
      }
    }
  }
  interp_.set_input("species", species_t);
  interp_.set_input("neighbor_species", neighbor_species_t);
  interp_.set_input("edge_vectors", edge_vectors_t);
  interp_.set_input("padding_mask", padding_mask_t);
  if (plato) plato->set_padding_mask(padding_mask_t), plato->set_fused_frames(true);
  interp_.set_input("reverse_neighbor_index", reverse_neighbor_index_t);
  if (!forces_mode_) {
    interp_.set_input("edge_distances", edge_distances_t);
    interp_.set_input("cutoff_factors", cutoff_factors_t);
  } else {
    interp_.set_input("cutoff_values", cutoff_values_t);
  }

  profile.mark("inputs");
  // --- Build ---
  ggml_context *compute_ctx = pg->compute_ctx = ggml_init({COMPUTE_CTX_SIZE, nullptr, true});
  if (!compute_ctx) {
    throw std::runtime_error("Failed to create compute context");
  }

  interp_.coulomb_application = {};
  if (p3m_) {
    interp_.coulomb_application = [&](ggml_context *ctx, ggml_tensor *charges) {
      if (lr_charges)
        throw std::runtime_error(
            "Multiple PET-LR applications are unsupported");
      lr_charges = ggml_cont(ctx, charges);
      ggml_set_output(lr_charges);
      lr_potentials = ggml_dup_tensor(ctx, lr_charges);
      ggml_set_input(lr_potentials);
      if (compute_forces) {
        ggml_set_param(lr_potentials);
        lr_charge_adjoint = ggml_dup_tensor(ctx, lr_charges);
        ggml_set_input(lr_charge_adjoint);
      }
      return lr_potentials;
    };
  }
  interp_.set_input_gradients(compute_forces);
  try {
    output = interp_.build(compute_ctx);
  } catch (...) {
    interp_.coulomb_application = {};
    interp_.custom_op = {};
    interp_.post_node = {};
    interp_.set_input_gradients(false);
    throw;
  }
  interp_.coulomb_application = {};
  interp_.custom_op = {};
  interp_.post_node = {};
  interp_.set_input_gradients(false);
  if (p3m_ && !lr_charges)
    throw std::runtime_error("Missing PET-LR charge application");
  if (!output) {
    throw std::runtime_error("Failed to build computation graph");
  }
  // nc-force models output [n_atoms, 4] = (energy | force): split the columns
  if (nc_forces_) {
    ggml_tensor *full = output;
    output = ggml_cont(compute_ctx, ggml_view_2d(compute_ctx, full, 1, n_atoms, full->nb[1], 0));
    nc_output = ggml_cont(compute_ctx, ggml_view_2d(compute_ctx, full, 3, n_atoms, full->nb[1],
                                                    sizeof(float)));
    ggml_set_output(nc_output);
  }
  ggml_set_output(output);
  latent_q = plato ? plato->charges() : nullptr;
  if (latent_q) ggml_set_output(latent_q);

  if (backward) {
    // Build forward + backward graph
    ggml_tensor *total_energy = loss_t = atom_w_t ? ggml_sum(compute_ctx, ggml_mul(compute_ctx, output,
                                                                          ggml_reshape(compute_ctx, atom_w_t, output)))
                                         : ggml_sum(compute_ctx, output);
    // Seed the backbone with the native solver's charge adjoint in a second
    // backward evaluation. Potentials are independent inputs to the readout.
    if (lr_charge_adjoint)
      total_energy =
          ggml_add(compute_ctx, total_energy,
                   ggml_sum(compute_ctx, ggml_mul(compute_ctx, lr_charges,
                                                  lr_charge_adjoint)));
    loss_t = total_energy;
    ggml_set_loss(total_energy);
    ggml_set_output(total_energy);

    cgraph = ggml_new_graph_custom(compute_ctx, 32768, true);
    ggml_build_forward_expand(cgraph, output);
    if (nc_output) ggml_build_forward_expand(cgraph, nc_output);
    ggml_build_forward_expand(cgraph, total_energy);
    ggml_build_backward_expand(compute_ctx, cgraph, nullptr);

    if (lr_potentials && compute_forces) {
      auto *grad = ggml_graph_get_grad(cgraph, lr_potentials);
      if (!grad)
        throw std::runtime_error("Missing P3M potential gradient");
      ggml_set_output(grad);
    }
    if (coulomb_t) {
      auto *lr_grad = ggml_graph_get_grad(cgraph, coulomb_t);
      if (lr_grad)
        ggml_set_output(lr_grad);
    }
    if (cutoff_values_t && (cutoff_values_t->flags & GGML_TENSOR_FLAG_PARAM))
      if (ggml_tensor *g = ggml_graph_get_grad(cgraph, cutoff_values_t)) ggml_set_output(g); // (read back below)
    ggml_tensor *grad = ggml_graph_get_grad(cgraph, edge_vectors_t);
    if (grad) {
      ggml_set_output(grad);
    } else {
      compute_forces = false;
    }
    if (plato)
      for (ggml_tensor *p : {plato->positions, plato->strain})
        if (ggml_tensor *g = p ? ggml_graph_get_grad(cgraph, p) : nullptr)
          for (; g; g = g->view_src) ggml_set_output(g);
    if (cond_blend_t && (cond_blend_t->flags & GGML_TENSOR_FLAG_PARAM))
      if (ggml_tensor *g = ggml_graph_get_grad(cgraph, cond_blend_t)) ggml_set_output(g);
  } else {
    cgraph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(cgraph, output);
    if (nc_output) ggml_build_forward_expand(cgraph, nc_output);
  }

  if (lr_charges) {
    charge_graph = ggml_new_graph_custom(compute_ctx, 32768, false);
    ggml_build_forward_expand(charge_graph, lr_charges);
  }
  size_t compute_bytes = 0;
  if (keep_graph) {
    // a kept graph: its own allocator (memory reused along the graph, as the cached path's; every tensor read back
    // is an output), the constants and the loss seed in a buffer of their own, which the allocator does not reuse
    std::vector<std::pair<ggml_tensor *, float>> constants = interp_.take_constants();
    if (loss_t)
      if (ggml_tensor *acc = ggml_graph_get_grad_acc(cgraph, loss_t)) constants.emplace_back(acc, 1.0f);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(compute_backend_);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t size = 0;
    std::vector<size_t> offset;
    for (auto &c : constants) {
      ggml_set_input(c.first);
      offset.push_back(size);
      size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, c.first), align);
    }
    if (!constants.empty()) {
      pg->const_buffer = ggml_backend_buft_alloc_buffer(buft, size);
      if (!pg->const_buffer) throw std::runtime_error("Failed to allocate the graph constants");
      char *base = static_cast<char *>(ggml_backend_buffer_get_base(pg->const_buffer));
      for (size_t k = 0; k < constants.size(); k++)
        ggml_backend_tensor_alloc(pg->const_buffer, constants[k].first, base + offset[k]);
    }
    pg->galloc = ggml_gallocr_new(buft);
    if (!pg->galloc || !ggml_gallocr_alloc_graph(pg->galloc, cgraph))
      throw std::runtime_error("Failed to allocate compute buffer");
    std::vector<float> staging;
    for (auto &[t, value] : constants) {
      staging.assign(ggml_nelements(t), value);
      ggml_backend_tensor_set(t, staging.data(), 0, ggml_nbytes(t));
    }
    compute_bytes = size + ggml_gallocr_get_buffer_size(pg->galloc, 0);
  } else {
    ggml_backend_buffer_t compute_buffer = pg->compute_buffer =
        ggml_backend_alloc_ctx_tensors(compute_ctx, compute_backend_);
    if (!compute_buffer) {
      throw std::runtime_error("Failed to allocate compute buffer");
    }
    interp_.init_constants();
    compute_bytes = ggml_backend_buffer_get_size(compute_buffer);
  }
  pg->species = species_t, pg->neighbor_species = neighbor_species_t, pg->edge_vectors = edge_vectors_t;
  pg->padding_mask = padding_mask_t, pg->reverse = reverse_neighbor_index_t;
  pg->charge_index = charge_index_t, pg->spin_index = spin_index_t, pg->charge_index_unit = charge_index_unit_t;
  pg->cond_blend = cond_blend_t, pg->atom_w = atom_w_t;
  pg->edge_distances = edge_distances_t, pg->cutoff_factors = cutoff_factors_t, pg->cutoff_values = cutoff_values_t;
  pg->output = output, pg->nc_output = nc_output, pg->latent_q = latent_q, pg->cgraph = cgraph;
  pg->forces = compute_forces;
  pg->bytes = ggml_backend_buffer_get_size(input_buffer) + compute_bytes;
  if (profile.enabled) std::fprintf(stderr, "GRAPH_PLAIN bytes=%.1f MiB nodes=%d\n", pg->bytes / 1048576.0, ggml_graph_n_nodes(cgraph));
  // keep it (most recent first; at most kMaxPlainGraphs graphs and kMaxPlainBytes device bytes)
  constexpr size_t kMaxPlainGraphs = 8, kMaxPlainBytes = size_t(512) << 20;
  if (keep_graph && pg->bytes <= kMaxPlainBytes) {
    plain_graphs_.insert(plain_graphs_.begin(), std::move(fresh));
    size_t total = 0, k = 0;
    for (; k < plain_graphs_.size(); k++) {
      total += plain_graphs_[k]->bytes;
      if (k >= kMaxPlainGraphs || total > kMaxPlainBytes) break;
    }
    plain_graphs_.resize(k);
  }
  } else {
    species_t = pg->species, neighbor_species_t = pg->neighbor_species, edge_vectors_t = pg->edge_vectors;
    padding_mask_t = pg->padding_mask, reverse_neighbor_index_t = pg->reverse;
    charge_index_t = pg->charge_index, spin_index_t = pg->spin_index, charge_index_unit_t = pg->charge_index_unit;
    cond_blend_t = pg->cond_blend, atom_w_t = pg->atom_w;
    edge_distances_t = pg->edge_distances, cutoff_factors_t = pg->cutoff_factors, cutoff_values_t = pg->cutoff_values;
    output = pg->output, nc_output = pg->nc_output, latent_q = pg->latent_q, cgraph = pg->cgraph;
    compute_forces = pg->forces;
    profile.mark("inputs");
  }
  PetPlatoLR *plato = pg->plato.get();

  // --- Upload the inputs (every call) ---
  // asynchronous copies on the compute backend's stream and one synchronisation (a synchronous copy per tensor waited
  // for a time slice of a shared GPU each)
  auto set_in = [&](ggml_tensor *t, const void *data, size_t bytes) {
    ggml_backend_tensor_set_async(compute_backend_, t, data, 0, bytes);
  };
  set_in(species_t, species_data.data(), species_data.size() * sizeof(int32_t));
  set_in(neighbor_species_t, ns_data.data(), ns_data.size() * sizeof(int32_t));
  set_in(edge_vectors_t, ev_data.data(), ev_data.size() * sizeof(float));
  set_in(padding_mask_t, pm_data.data(), pm_data.size() * sizeof(float));
  set_in(reverse_neighbor_index_t, rni_data.data(), rni_data.size() * sizeof(int32_t));
  if (coulomb_t) set_in(coulomb_t, coulomb.matrix.data(), coulomb.matrix.size() * sizeof(float));
  if (atom_w_t) {
    if (atom_w_.size() != static_cast<size_t>(n_atoms)) throw std::logic_error("atom energy weights: atom count");
    set_in(atom_w_t, atom_w_.data(), atom_w_.size() * sizeof(float));
  }
  std::vector<int32_t> cond_rows, cond_spin, cond_rows_unit; // (alive until the synchronisation below)
  std::vector<float> cond_bl;
  if (conditioning_.enabled) {
    const auto [ci, si] = conditioning_.rows(system.charge(), system.spin_multiplicity());
    (void)ci;
    cond_rows = charge_rows(n_atoms, system.charge(), system.spin_multiplicity()), cond_spin.assign(n_atoms, si);
    set_in(charge_index_t, cond_rows.data(), cond_rows.size() * sizeof(int32_t));
    set_in(spin_index_t, cond_spin.data(), cond_spin.size() * sizeof(int32_t));
    if (charge_index_unit_t) {
      cond_rows_unit = charge_rows(n_atoms, system.charge(), system.spin_multiplicity(), true);
      set_in(charge_index_unit_t, cond_rows_unit.data(), cond_rows_unit.size() * sizeof(int32_t));
      if (cond_blend_t) {
        cond_bl.assign(n_atoms, 0.0f);
        if (!lrc_blend_.empty()) cond_bl = lrc_blend_;
        set_in(cond_blend_t, cond_bl.data(), cond_bl.size() * sizeof(float));
      }
    }
  }
  if (!forces_mode_) {
    set_in(edge_distances_t, ed_data.data(), ed_data.size() * sizeof(float));
    set_in(cutoff_factors_t, cf_data.data(), cf_data.size() * sizeof(float));
  } else {
    set_in(cutoff_values_t, cv_data.data(), cv_data.size() * sizeof(float));
  }
  if (plato && plato->needs_neighbor_index()) {
    std::vector<int32_t> nbr(neighbor_atoms_vec.begin(), neighbor_atoms_vec.end());
    for (auto &j : nbr) j = std::max(j, 0);
    plato->set_neighbor_index(std::move(nbr));
  }
  if (plato) plato->upload(system.positions(), compute_backend_); // the inputs made while building too
  ggml_backend_synchronize(compute_backend_);
  profile.mark(rebuilt ? "build_allocate" : "upload");

  P3MResult mesh_result;
  std::vector<float> charge_values;
  if (charge_graph) {
    if (backend_provider_->compute(charge_graph) !=
        GGML_STATUS_SUCCESS)
      throw std::runtime_error("PET-LR charge graph failed");
    profile.mark("charge_forward");
    charge_values.resize(ggml_nelements(lr_charges));
    ggml_backend_tensor_get(lr_charges, charge_values.data(), 0,
                            ggml_nbytes(lr_charges));
    mesh_result = p3m(system, nlist, coulomb_parameters_, lr_charges->ne[0],
                      charge_values);
    ggml_backend_tensor_set(lr_potentials, mesh_result.potentials.data(), 0,
                            ggml_nbytes(lr_potentials));
    if (lr_charge_adjoint)
      ggml_backend_tensor_memset(lr_charge_adjoint, 0, 0,
                                 ggml_nbytes(lr_charge_adjoint));
  }
  profile.mark("p3m_forward");
  if (backward) {
    ggml_graph_reset(cgraph);
  }

  ggml_status status = backend_provider_->compute(cgraph);
  if (status != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("Graph computation failed");
  }

  profile.mark("forward_backward");
  if (lr_potentials && compute_forces) {
    std::vector<float> adjoint(charge_values.size());
    ggml_backend_tensor_get(ggml_graph_get_grad(cgraph, lr_potentials),
                            adjoint.data(), 0, adjoint.size() * sizeof(float));
    mesh_result = p3m(system, nlist, coulomb_parameters_, lr_charges->ne[0],
                      charge_values, &adjoint);
    ggml_backend_tensor_set(lr_charge_adjoint,
                            mesh_result.charge_gradient.data(), 0,
                            ggml_nbytes(lr_charge_adjoint));
    profile.mark("p3m_backward");
    ggml_graph_reset(cgraph);
    if (backend_provider_->compute(cgraph) !=
        GGML_STATUS_SUCCESS)
      throw std::runtime_error("PET-LR charge backward graph failed");
  }

  profile.mark("charge_backward");
  if (parameter_gradients) {
    parameter_gradients->clear();
    const auto all = weights();
    for (const auto &name : param_names_) {
      auto it = std::find_if(all.begin(), all.end(), [&](const auto &w) { return w.first == name; });
      auto *parameter = it->second;
      auto *gradient = ggml_graph_get_grad(cgraph, parameter);
      const size_t offset = parameter_gradients->size();
      parameter_gradients->resize(offset + ggml_nelements(parameter), 0.0f);
      if (gradient) {
        ggml_backend_tensor_get(gradient, parameter_gradients->data() + offset, 0,
                                ggml_nbytes(gradient));
        for (size_t i = offset; i < parameter_gradients->size(); ++i)
          (*parameter_gradients)[i] *= energy_scale_;
      }
    }
  }

  // --- Extract results ---
  ModelResult result;

  if (latent_q) read_latent_charges(latent_q, result);

  // Get atomic energies
  std::vector<float> atomic_energies(n_atoms);
  ggml_backend_tensor_get(output, atomic_energies.data(), 0,
                          n_atoms * sizeof(float));

  result.atomic_energies.resize(n_atoms);
  for (int i = 0; i < n_atoms; i++) result.atomic_energies[i] = atomic_energies[i] * energy_scale_;
  lrc_blend_grad_.clear();
  if (backward && cond_blend_t && (cond_blend_t->flags & GGML_TENSOR_FLAG_PARAM)) {
    lrc_blend_grad_.assign(n_atoms, 0.0);
    if (ggml_tensor *g = ggml_graph_get_grad(cgraph, cond_blend_t)) {
      std::vector<float> gb(n_atoms);
      ggml_backend_tensor_get(g, gb.data(), 0, n_atoms * sizeof(float));
      for (int i = 0; i < n_atoms; i++) {
        if (!std::isfinite(gb[i])) throw std::runtime_error("lrc: non-finite dE/dS from the graph");
        lrc_blend_grad_[i] = double(gb[i]) * energy_scale_;
      }
    }
  }

  // Sum and scale
  float model_energy = 0.0f;
  for (int i = 0; i < n_atoms; i++) {
    model_energy += atomic_energies[i];
  }
  float scaled_energy = model_energy * energy_scale_;

  // Add composition energies
  float composition_energy = 0.0f;
  for (int i = 0; i < n_atoms; i++) {
    auto it = composition_energies_.find(atomic_numbers[i]);
    if (it != composition_energies_.end()) {
      composition_energy += it->second;
    }
  }

  result.energy = scaled_energy + composition_energy;

  if (nc_output) {
    result.nc_forces.resize(n_atoms * 3);
    ggml_backend_tensor_get(nc_output, result.nc_forces.data(), 0,
                            result.nc_forces.size() * sizeof(float));
    for (int i = 0; i < n_atoms; i++) {
      auto it = nc_force_scales_.find(atomic_numbers[i]);
      const float s = it != nc_force_scales_.end() ? it->second : 1.0f;
      for (int k = 0; k < 3; k++) result.nc_forces[3 * i + k] *= s;
    }
    result.has_nc_forces = true;
  }

  // Extract forces
  if (compute_forces) {
    ggml_tensor *grad_tensor = ggml_graph_get_grad(cgraph, edge_vectors_t);
    if (grad_tensor && grad_tensor->data) {
      std::vector<float> grad_data(ggml_nelements(grad_tensor));
      ggml_backend_tensor_get(grad_tensor, grad_data.data(), 0,
                              ggml_nbytes(grad_tensor));

      // Scatter edge gradients to per-atom forces
      result.forces.resize(n_atoms * 3, 0.0f);
      const int stride_slot = 3;
      const int stride_atom = 3 * max_neighbors;

      for (int ca = 0; ca < n_atoms; ca++) {
        for (int slot = 0; slot < max_neighbors; slot++) {
          int flat_idx = ca * max_neighbors + slot;
          if (pm_data[flat_idx] > 0.5f) continue;

          int na = neighbor_atoms_vec[flat_idx];
          if (na < 0) continue;

          int base = slot * stride_slot + ca * stride_atom;
          float gx = grad_data[0 + base];
          float gy = grad_data[1 + base];
          float gz = grad_data[2 + base];

          result.forces[ca * 3 + 0] += gx;
          result.forces[ca * 3 + 1] += gy;
          result.forces[ca * 3 + 2] += gz;

          result.forces[na * 3 + 0] -= gx;
          result.forces[na * 3 + 1] -= gy;
          result.forces[na * 3 + 2] -= gz;
        }
      }

      // Adaptive cutoff: E depends on positions through atomic_cutoffs too.
      // S[k] = sum over edges touching k of dE/dcutoff_values (each edge's
      // pair cutoff is the mean of its two atomic cutoffs, hence the 1/2);
      // edge e = (k -> l) of the full list then adds g_e = S[k] J[e] v_e / d_e
      // to dE/dv_e, like the graph's own edge-vector gradient.
      std::vector<std::array<double, 3>> chain_grad;
      if (have_adaptive && cutoff_values_t) {
        ggml_tensor *cv_grad = ggml_graph_get_grad(cgraph, cutoff_values_t);
        if (cv_grad && cv_grad->data) {
          std::vector<float> C(ggml_nelements(cv_grad));
          ggml_backend_tensor_get(cv_grad, C.data(), 0, ggml_nbytes(cv_grad));
          std::vector<double> S(n_atoms, 0.0);
          for (int flat = 0; flat < total_slots; flat++) {
            if (pm_data[flat] > 0.5f || neighbor_atoms_vec[flat] < 0) continue;
            S[flat / max_neighbors] += 0.5 * C[flat];
            S[neighbor_atoms_vec[flat]] += 0.5 * C[flat];
          }
          chain_grad.resize(full_nlist.num_pairs());
          for (int e = 0; e < full_nlist.num_pairs(); e++) {
            const int k = full_nlist.centers[e], l = full_nlist.neighbors[e];
            const auto &v = full_nlist.edge_vectors[e];
            const double d = std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]);
            const double coef = d > 0.0 ? S[k] * cutoff_jacobian[e] / d : 0.0;
            for (int c = 0; c < 3; c++) {
              chain_grad[e][c] = coef * v[c];
              result.forces[k * 3 + c] += static_cast<float>(chain_grad[e][c]);
              result.forces[l * 3 + c] -= static_cast<float>(chain_grad[e][c]);
            }
          }
        }
      }

      // Plato-PET-LR: the explicit position and strain dependence of the long-range ops
      std::vector<float> plato_geps;
      if (plato) {
        if (ggml_tensor *g = ggml_graph_get_grad(cgraph, plato->positions)) {
          std::vector<float> gp(3 * size_t(n_atoms));
          ggml_backend_tensor_get(g, gp.data(), 0, gp.size() * sizeof(float));
          for (int i = 0; i < n_atoms * 3; i++) result.forces[i] -= gp[i];
        }
        if (ggml_tensor *g = plato->strain ? ggml_graph_get_grad(cgraph, plato->strain) : nullptr) {
          plato_geps.resize(9);
          ggml_backend_tensor_get(g, plato_geps.data(), 0, 9 * sizeof(float));
        }
      }

      // Apply energy scale
      for (int i = 0; i < n_atoms * 3; i++) {
        result.forces[i] *= energy_scale_;
      }

      result.has_forces = true;

      // Virial stress for periodic systems: σ_ab = (1/V) Σ_edges r_a ⊗ ∂E/∂r_b.
      // Off-diagonal Voigt components are symmetrized.
      const Cell *cell = system.cell();
      if (cell &&
          (cell->periodic[0] || cell->periodic[1] || cell->periodic[2])) {
        const auto &m = cell->matrix;
        float vol = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        vol = std::abs(vol);

        if (vol > 1e-10f) {
          result.stress.assign(6, 0.0f);
          for (int ca = 0; ca < n_atoms; ca++) {
            for (int slot = 0; slot < max_neighbors; slot++) {
              int flat_idx = ca * max_neighbors + slot;
              if (pm_data[flat_idx] > 0.5f) continue;

              int base = slot * stride_slot + ca * stride_atom;
              float ex = ev_data[base + 0];
              float ey = ev_data[base + 1];
              float ez = ev_data[base + 2];
              float gx = grad_data[base + 0];
              float gy = grad_data[base + 1];
              float gz = grad_data[base + 2];

              result.stress[0] += ex * gx;                    // xx
              result.stress[1] += ey * gy;                    // yy
              result.stress[2] += ez * gz;                    // zz
              result.stress[3] += 0.5f * (ey * gz + ez * gy); // yz
              result.stress[4] += 0.5f * (ex * gz + ez * gx); // xz
              result.stress[5] += 0.5f * (ex * gy + ey * gx); // xy
            }
          }
          for (int e = 0; e < static_cast<int>(chain_grad.size()); e++) {
            const auto &r = full_nlist.edge_vectors[e];
            const auto &g = chain_grad[e];
            result.stress[0] += static_cast<float>(r[0] * g[0]);
            result.stress[1] += static_cast<float>(r[1] * g[1]);
            result.stress[2] += static_cast<float>(r[2] * g[2]);
            result.stress[3] += static_cast<float>(0.5 * (r[1] * g[2] + r[2] * g[1]));
            result.stress[4] += static_cast<float>(0.5 * (r[0] * g[2] + r[2] * g[0]));
            result.stress[5] += static_cast<float>(0.5 * (r[0] * g[1] + r[1] * g[0]));
          }

          if (!plato_geps.empty()) { // row-major dE/d eps
            const auto &q = plato_geps;
            result.stress[0] += q[0], result.stress[1] += q[4], result.stress[2] += q[8];
            result.stress[3] += 0.5f * (q[5] + q[7]);
            result.stress[4] += 0.5f * (q[2] + q[6]);
            result.stress[5] += 0.5f * (q[1] + q[3]);
          }
          for (int i = 0; i < 6; i++) {
            result.stress[i] *= energy_scale_ / vol;
          }
          result.has_stress = true;
        }
      }
    }
  }

  if (compute_forces && lr_potentials) {
    for (size_t i = 0; i < mesh_result.forces.size(); ++i)
      result.forces[i] += energy_scale_ * mesh_result.forces[i];
    for (size_t i = 0; i < mesh_result.stress.size(); ++i)
      result.stress[i] += energy_scale_ * mesh_result.stress[i];
  }
  if (compute_forces && coulomb_t) {
    auto *grad = ggml_graph_get_grad(cgraph, coulomb_t);
    if (!grad) {
      throw std::runtime_error("Missing PET-LR Coulomb gradient");
    }
    std::vector<float> adjoint(size_t(n_atoms) * n_atoms);
    ggml_backend_tensor_get(grad, adjoint.data(), 0,
                            adjoint.size() * sizeof(float));
    auto correction =
        coulomb_matrix(system, nlist, coulomb_parameters_, &adjoint);
    for (size_t i = 0; i < correction.forces.size(); ++i)
      result.forces[i] += energy_scale_ * correction.forces[i];
    for (size_t i = 0; i < correction.stress.size(); ++i)
      result.stress[i] += energy_scale_ * correction.stress[i];
  }

  if (no_edges && forces_requested) { // energy independent of the positions
    result.forces.assign(3 * static_cast<size_t>(n_atoms), 0.0f);
    result.has_forces = true;
    if (system.cell()) {
      result.stress.assign(6, 0.0f);
      result.has_stress = true;
    }
  }

  profile.mark("readback");
  // Hook for callers that want to inspect intermediate tensors. Runs while
  // the compute buffer (and therefore every node_outputs_ tensor) is still
  // alive on the active backend.
  if (post_compute_hook) {
    post_compute_hook();
  }

  return result;
}

} // namespace mlipcpp::runtime
