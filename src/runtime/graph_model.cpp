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

GraphModel::~GraphModel() {
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
  num_heads_ = loader.get_int32("pet.num_heads", 0);
  conditioning_.enabled = loader.get_int32("pet.system_conditioning", 0) != 0;
  plato_lr_ = PetPlatoLRSpec::from_gguf(loader);
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

ModelResult GraphModel::predict_single(
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
  if (!no_cache && !post_compute_hook && !parameter_gradients && cached_path_supported() &&
      system.num_atoms() > 0)
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
  // backward pass, forces and stress are zero.
  const bool no_edges = max_neighbors == 0, forces_requested = compute_forces;
  if (no_edges) {
    max_neighbors = 1;
    compute_forces = false;
  }
  interp_.set_exact_padding_mask(no_edges && !std::getenv("MLIPCPP_PAD_EMPTY"));

  // Set symbolic dimensions for this system
  interp_.set_dimension("n_atoms", n_atoms);
  interp_.set_dimension("max_neighbors", max_neighbors);
  interp_.set_dimension("n_edges", n_atoms * max_neighbors);
  interp_.set_dimension("max_neighbors_plus_one", max_neighbors + 1);

  const int total_slots = n_atoms * max_neighbors;

  // --- Create input context ---
  ggml_context *input_ctx = ggml_init({INPUT_CTX_SIZE, nullptr, true});
  std::unique_ptr<ggml_context, decltype(&ggml_free)> input_owner(input_ctx,
                                                                  ggml_free);
  if (!input_ctx) {
    throw std::runtime_error("Failed to create input context");
  }

  // Create input tensors
  ggml_tensor *species_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
  ggml_set_name(species_t, "species");

  ggml_tensor *neighbor_species_t =
      ggml_new_tensor_2d(input_ctx, GGML_TYPE_I32, max_neighbors, n_atoms);
  ggml_set_name(neighbor_species_t, "neighbor_species");

  ggml_tensor *edge_vectors_t =
      ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, 3, max_neighbors, n_atoms);
  ggml_set_name(edge_vectors_t, "edge_vectors");

  ggml_tensor *padding_mask_t =
      ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, max_neighbors, n_atoms);
  ggml_set_name(padding_mask_t, "padding_mask");

  ggml_tensor *reverse_neighbor_index_t =
      ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, total_slots);
  ggml_set_name(reverse_neighbor_index_t, "reverse_neighbor_index");

  ggml_tensor *coulomb_t = nullptr;
  if (long_range_ && !p3m_) {
    coulomb_t = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, n_atoms, n_atoms);
    ggml_set_name(coulomb_t, "coulomb_matrix");
    if (compute_forces)
      ggml_set_param(coulomb_t);
  }

  // charge / spin conditioning: the table rows of the structure, per atom
  ggml_tensor *charge_index_t = nullptr, *spin_index_t = nullptr;
  if (conditioning_.enabled) {
    charge_index_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
    ggml_set_name(charge_index_t, "charge_index");
    spin_index_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
    ggml_set_name(spin_index_t, "spin_index");
  }

  // Mode-specific inputs
  ggml_tensor *edge_distances_t = nullptr;
  ggml_tensor *cutoff_factors_t = nullptr;
  ggml_tensor *cutoff_values_t = nullptr;

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
  std::unique_ptr<PetPlatoLR> plato;
  if (plato_lr_.enabled) {
    PetPlatoLRStructure st;
    st.a0 = 0, st.a1 = n_atoms;
    if (const Cell *c = system.cell()) {
      const int np = c->periodic[0] + c->periodic[1] + c->periodic[2];
      if (np != 0 && np != 3 && plato_lr_.long_range())
        throw std::runtime_error("Plato-PET-LR long range needs full or no periodicity");
      st.periodic = np == 3;
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) st.cell[a][b] = c->matrix[a][b];
    }
    st.charge = system.charge();
    plato = std::make_unique<PetPlatoLR>();
    plato->create(input_ctx, plato_lr_, {st}, n_atoms, compute_forces && plato_lr_.long_range());
  }

  // Allocate input buffer
  ggml_backend_buffer_t input_buffer =
      ggml_backend_alloc_ctx_tensors(input_ctx, compute_backend_);
  std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
      input_buffer_owner(input_buffer, ggml_backend_buffer_free);
  if (!input_buffer) {
    throw std::runtime_error("Failed to allocate input buffer");
  }

  if (plato) {
    std::vector<float> pos(system.positions(), system.positions() + 3 * size_t(n_atoms));
    plato->upload(pos.data());
    interp_.custom_op = [&](ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in) {
      return plato->build(ctx, node, in);
    };
  }

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

  // Copy data to tensors
  ggml_backend_tensor_set(species_t, species_data.data(), 0,
                          species_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(neighbor_species_t, ns_data.data(), 0,
                          ns_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(edge_vectors_t, ev_data.data(), 0,
                          ev_data.size() * sizeof(float));
  ggml_backend_tensor_set(padding_mask_t, pm_data.data(), 0,
                          pm_data.size() * sizeof(float));
  ggml_backend_tensor_set(reverse_neighbor_index_t, rni_data.data(), 0,
                          rni_data.size() * sizeof(int32_t));

  if (coulomb_t) {
    ggml_backend_tensor_set(coulomb_t, coulomb.matrix.data(), 0,
                            coulomb.matrix.size() * sizeof(float));
    interp_.set_input("coulomb_matrix", coulomb_t);
  }

  if (conditioning_.enabled) {
    const auto [ci, si] = conditioning_.rows(system.charge(), system.spin_multiplicity());
    std::vector<int32_t> c(n_atoms, ci), sp(n_atoms, si);
    ggml_backend_tensor_set(charge_index_t, c.data(), 0, c.size() * sizeof(int32_t));
    ggml_backend_tensor_set(spin_index_t, sp.data(), 0, sp.size() * sizeof(int32_t));
    interp_.set_input("charge_index", charge_index_t);
    interp_.set_input("spin_index", spin_index_t);
  }

  // Register common inputs
  interp_.set_input("species", species_t);
  interp_.set_input("neighbor_species", neighbor_species_t);
  interp_.set_input("edge_vectors", edge_vectors_t);
  interp_.set_input("padding_mask", padding_mask_t);
  if (plato) plato->set_padding_mask(padding_mask_t), plato->set_fused_frames(true);
  interp_.set_input("reverse_neighbor_index", reverse_neighbor_index_t);

  // Register mode-specific inputs
  if (!forces_mode_) {
    ggml_backend_tensor_set(edge_distances_t, ed_data.data(), 0,
                            ed_data.size() * sizeof(float));
    ggml_backend_tensor_set(cutoff_factors_t, cf_data.data(), 0,
                            cf_data.size() * sizeof(float));
    interp_.set_input("edge_distances", edge_distances_t);
    interp_.set_input("cutoff_factors", cutoff_factors_t);
  } else {
    ggml_backend_tensor_set(cutoff_values_t, cv_data.data(), 0,
                            cv_data.size() * sizeof(float));
    interp_.set_input("cutoff_values", cutoff_values_t);
  }

  profile.mark("inputs");
  // --- Build and compute ---
  ggml_context *compute_ctx = ggml_init({COMPUTE_CTX_SIZE, nullptr, true});
  std::unique_ptr<ggml_context, decltype(&ggml_free)> compute_owner(compute_ctx,
                                                                    ggml_free);
  if (!compute_ctx) {
    throw std::runtime_error("Failed to create compute context");
  }

  ggml_tensor *lr_charges = nullptr, *lr_potentials = nullptr,
              *lr_charge_adjoint = nullptr;
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
  ggml_tensor *output = nullptr;
  interp_.set_input_gradients(compute_forces);
  try {
    output = interp_.build(compute_ctx);
  } catch (...) {
    interp_.coulomb_application = {};
    interp_.custom_op = {};
    interp_.set_input_gradients(false);
    throw;
  }
  interp_.coulomb_application = {};
  interp_.custom_op = {};
  interp_.set_input_gradients(false);
  if (p3m_ && !lr_charges)
    throw std::runtime_error("Missing PET-LR charge application");
  if (!output) {
    throw std::runtime_error("Failed to build computation graph");
  }
  // nc-force models output [n_atoms, 4] = (energy | force): split the columns
  ggml_tensor *nc_output = nullptr;
  if (nc_forces_) {
    ggml_tensor *full = output;
    output = ggml_cont(compute_ctx, ggml_view_2d(compute_ctx, full, 1, n_atoms, full->nb[1], 0));
    nc_output = ggml_cont(compute_ctx, ggml_view_2d(compute_ctx, full, 3, n_atoms, full->nb[1],
                                                    sizeof(float)));
    ggml_set_output(nc_output);
  }
  ggml_set_output(output);
  ggml_tensor *latent_q = plato ? plato->charges() : nullptr;
  if (latent_q) ggml_set_output(latent_q);

  ggml_cgraph *cgraph = nullptr;

  if (compute_forces) {
    // Build forward + backward graph
    ggml_tensor *total_energy = ggml_sum(compute_ctx, output);
    // Seed the backbone with the native solver's charge adjoint in a second
    // backward evaluation. Potentials are independent inputs to the readout.
    if (lr_charge_adjoint)
      total_energy =
          ggml_add(compute_ctx, total_energy,
                   ggml_sum(compute_ctx, ggml_mul(compute_ctx, lr_charges,
                                                  lr_charge_adjoint)));
    ggml_set_loss(total_energy);
    ggml_set_output(total_energy);

    cgraph = ggml_new_graph_custom(compute_ctx, 32768, true);
    ggml_build_forward_expand(cgraph, output);
    if (nc_output) ggml_build_forward_expand(cgraph, nc_output);
    ggml_build_forward_expand(cgraph, total_energy);
    ggml_build_backward_expand(compute_ctx, cgraph, nullptr);

    if (lr_potentials) {
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
  } else {
    cgraph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(cgraph, output);
    if (nc_output) ggml_build_forward_expand(cgraph, nc_output);
  }

  ggml_cgraph *charge_graph = nullptr;
  if (lr_charges) {
    charge_graph = ggml_new_graph_custom(compute_ctx, 32768, false);
    ggml_build_forward_expand(charge_graph, lr_charges);
  }
  ggml_backend_buffer_t compute_buffer =
      ggml_backend_alloc_ctx_tensors(compute_ctx, compute_backend_);
  std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
      compute_buffer_owner(compute_buffer, ggml_backend_buffer_free);
  if (!compute_buffer) {
    throw std::runtime_error("Failed to allocate compute buffer");
  }

  interp_.init_constants();
  if (plato) plato->upload(system.positions()); // again: the inputs made while building
  profile.mark("build_allocate");

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
  if (compute_forces) {
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
