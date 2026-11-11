#include "graph_model.h"
#include "graph_cache.h"
#include "core/adaptive_cutoff.h"
#include "core/gguf_loader.h"
#include "core/padding.h"
#include "core/p3m.h"
#include "core/masses.h"
#include "iris_lr.h"

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
#include <functional>
#include <numeric>
#include <cmath>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

using json = nlohmann::json;


namespace {
/// Precise transcendentals on CUDA (cmake/patches/ggml-cuda-precise-unary.patch): ggml-cuda builds with
/// -use_fast_math, whose __logf keeps only an absolute accuracy of 2^-21.4 near 1, so log(1 + u) (the log1p of
/// softplus) loses all relative accuracy at small u, and whose __sinf / __cosf degrade outside [-pi, pi] (the
/// Ewald structure factors), and whose __expf in SIGMOID loses accuracy with |x| (the Fermi occupations at small T),
/// with 2-ulp approximate division (DIV: __fdiv_rn when flagged, ggml-cuda-precise-sigmoid-div.patch). Flags every
/// LOG / SIN / COS / EXP / SIGMOID / DIV node of `g` (forward and backward) for the
/// double-precision kernels, and (opt-in, MLIPCPP_PRECISE_SUM=1) every SUM / SUM_ROWS for double accumulation
/// (ggml-cuda-precise-sum.patch); the CPU backend's kernels are already accurate and ignore the flag. Other models
/// never call this (their CUDA kernels stay the fast ones, bit for bit). MLIPCPP_PRECISE_MATH=0 turns it off.
void mark_precise_math(ggml_cgraph *g) {
  if (!g) return;
  constexpr int slot = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
  for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
    ggml_tensor *t = ggml_graph_node(g, i);
    // MLIPCPP_PRECISE_MATH_SKIP=sigmoid,div,...: leave those op kinds on the fast kernels (bisection diagnostics)
    static const std::string skip = [] { const char *e = std::getenv("MLIPCPP_PRECISE_MATH_SKIP"); return e ? e : ""; }();
    // SIGMOID is NOT flagged by default: the precise kernel is correctly rounded (op test) and removes the 4-13x local
    // error of the Fermi occupations, but moved no verdict and made seeded lr_both ion mu worse on CUDA (median 1.5e-7
    // -> 2.1e-7, P 0.35-0.57 -> 0.90-0.93; bisected 2026-11-11, job 4891356): MLIPCPP_PRECISE_SIGMOID=1 turns it on
    static const bool sig = [] { const char *e = std::getenv("MLIPCPP_PRECISE_SIGMOID"); return e && !std::strcmp(e, "1"); }();
    // SUM / SUM_ROWS in double (as ggml-cpu) is NOT flagged by default either: it reaches the potential sums (their
    // local errors drop to the torch fp32 level) but moved no verdict (CUDA charges P 0.58-0.75 unchanged, median q
    // error 8.5e-7 -> 8.8e-7; job 4891425, 2026-11-11): MLIPCPP_PRECISE_SUM=1 turns it on
    static const bool psum = [] { const char *e = std::getenv("MLIPCPP_PRECISE_SUM"); return e && !std::strcmp(e, "1"); }();
    // MUL_MAT accumulating in double (ggml-cuda-precise-mulmat.patch): a DIAGNOSTIC, off by default: MLIPCPP_PRECISE_MATMUL=1
    // flags every F32 MUL_MAT, MLIPCPP_PRECISE_MATMUL_K=<K> only those reducing over K (e.g. 1036, the transfer MLP's
    // first LINEAR of the 512-wide checkpoints: the CUDA charge-error candidate of 2026-11-11)
    static const long pmm = [] {
      const char *e = std::getenv("MLIPCPP_PRECISE_MATMUL"), *k = std::getenv("MLIPCPP_PRECISE_MATMUL_K");
      return k && *k ? std::atol(k) : (e && !std::strcmp(e, "1") ? -1L : 0L);
    }();
    if (t->op == GGML_OP_MUL_MAT && pmm != 0 && (pmm < 0 || t->src[0]->ne[0] == pmm)) t->op_params[slot] = 1;
    auto on = [](const char *k) {
      if (!std::strcmp(k, "sigmoid") && !sig) return false;
      if (!std::strcmp(k, "sum") && !psum) return false;
      return skip.find(k) == std::string::npos;
    };
    const bool is_unary = t->op == GGML_OP_UNARY;
    const bool unary = is_unary && ((ggml_get_unary_op(t) == GGML_UNARY_OP_EXP && on("exp")) ||
                                    (ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID && on("sigmoid")));
    // SUM / SUM_ROWS: double accumulation (ggml-cuda-precise-sum.patch), as ggml-cpu (ggml_float) does
    const bool sum = (t->op == GGML_OP_SUM || t->op == GGML_OP_SUM_ROWS) && on("sum");
    if (sum || (t->op == GGML_OP_LOG && on("log")) || (t->op == GGML_OP_SIN && on("sin")) ||
        (t->op == GGML_OP_COS && on("cos")) || (t->op == GGML_OP_DIV && on("div")) || unary)
      t->op_params[slot] = 1;
  }
}
} // namespace

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
  iris_lr_ = {};
  iris_lr_.enabled = loader.get_int32("pet.iris_lr", 0) != 0;
  iris_max_neighbors_ = loader.get_int32("pet.iris.max_neighbors", 0);
  precise_math_gguf_ = loader.get_int32("pet.precise_math", 0) != 0;
  if (iris_lr_.enabled) {
    iris_lr_.n_out = loader.get_int32("pet.iris_lr.n_out", 8);
    iris_lr_.reference = loader.get_string("pet.iris_lr.reference", "");
    iris_lr_.m_min = loader.get_int32("pet.iris_lr.charge_state_min", -3);
    iris_lr_.m_max = loader.get_int32("pet.iris_lr.charge_state_max", 3);
    iris_lr_.temperature = loader.get_float32("pet.iris_lr.charge_temperature", 0.1f);
    iris_lr_.iterations = loader.get_int32("pet.iris_lr.fermi_iterations", 40);
    iris_lr_.eps = loader.get_float32("pet.iris_lr.fermi_eps", 1e-6f);
    iris_lr_.num_k = loader.get_int32("pet.iris_lr.num_k", 0);
    iris_lr_.smearing = loader.get_float32("pet.iris_lr.smearing", 0.0f);
    iris_lr_.slab_margin = loader.get_float32("pet.iris_lr.slab_margin", 2.0f);
    iris_lr_.long_range_transfer = loader.get_int32("pet.iris_lr.long_range_transfer", 0) != 0;
    iris_lr_.conserving_charges = loader.get_int32("pet.iris_lr.conserving_charges", 0) != 0;
    if (loader.get_int32("pet.iris_lr.halfspace", 1) != 1)
      throw std::runtime_error("pet.iris_lr.halfspace = 0: only his default halfspace k grid is ported");
    // species_weighted / species_fill: his branch feat/long-range-transfer (3ed96b0); residual: his
    // feat/conserving-charges (a98c2f6), conserving_charges only
    if (iris_lr_.reference != "fermi" && iris_lr_.reference != "species" && iris_lr_.reference != "both" &&
        iris_lr_.reference != "species_weighted" && iris_lr_.reference != "species_fill" &&
        !(iris_lr_.reference == "residual" && iris_lr_.conserving_charges))
      throw std::runtime_error("pet.iris_lr.reference: " + iris_lr_.reference);
    if (iris_lr_.n_out != 8) throw std::runtime_error("pet.iris_lr.n_out != 8");
  }
  param_names_ = json::parse(loader.get_string("train.param_names", "[]"))
                     .get<std::vector<std::string>>();
  gguf_path_ = path;
  num_neighbors_adaptive_ = loader.get_float32("pet.num_neighbors_adaptive", 0.0f);
  cutoff_width_adaptive_ = loader.get_float32("pet.cutoff_width_adaptive", cutoff_width_);
  adaptive_cutoff_method_ = loader.get_string("pet.adaptive_cutoff_method", "grid");
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
  if (const std::string keys = loader.get_string("pet.conditioning_keys", ""); !keys.empty()) {
    conditioning_.use_charge = conditioning_.use_spin = false;
    size_t a = 0;
    while (a <= keys.size()) {
      const size_t b = std::min(keys.find(',', a), keys.size());
      const std::string k = keys.substr(a, b - a);
      if (k == "charge") conditioning_.use_charge = true;
      else if (k == "spin_multiplicity") conditioning_.use_spin = true;
      else throw std::runtime_error("pet.conditioning_keys: unknown key '" + k + "'");
      a = b + 1;
    }
  }
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
  if (iris_lr_.enabled) {
    iris_lr_.graph_open = graph_json;
    iris_lr_.graph_pbc = loader.get_string("graph_pbc.json", "");
    iris_lr_.pbc_loaded = false;
  }
  scores_interp_ = GraphInterpreter();
  if (iris_lr_.enabled && iris_lr_.reference != "species" && iris_lr_.reference != "species_weighted" &&
      iris_lr_.reference != "residual") {
    const std::string sj = loader.get_string("graph_scores.json", "");
    if (sj.empty()) {
      ggml_free(temp_ctx);
      throw std::runtime_error("pet.iris_lr: the fermi reference needs graph_scores.json");
    }
    scores_interp_.load_graph(sj);
  }
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
      if (scores_interp_.has_graph()) scores_interp_.set_weight(name, weight);
    }
  }

  ggml_free(temp_ctx);

  // Use primary backend (may be GPU) for compute; owned by BackendProvider.
  compute_backend_ = backend_provider_->primary();
  interp_.set_backend(compute_backend_);
  scores_interp_.set_backend(compute_backend_);
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

bool GraphModel::precise_math() const {
  // MLIPCPP_PRECISE_MATH=0 / 1 forces the CUDA precise-math flags off / on for every graph (diagnostics); default:
  // iris graphs (the PET-LR ones, and iris trunks exported with pet.precise_math = 1)
  static const int env = [] {
    const char *e = std::getenv("MLIPCPP_PRECISE_MATH");
    return !e || !*e ? -1 : std::strcmp(e, "0") ? 1 : 0;
  }();
  if (env >= 0) return env == 1;
  return iris_lr_.enabled || precise_math_gguf_;
}

void GraphModel::check_iris_overflow(int max_selected) const {
  // his Calculator.calculate raises when the forward's selection needs more than max_neighbors slots for one atom
  // (iris.pet.predict truncate_edges / pack_edges overflow): with differentiable (adaptive) cutoffs the pairs the
  // adaptive cutoff keeps, with hard cutoffs every pair within the cutoff. mlip.cpp would keep them all and silently
  // differ from every run of his; so it raises too (GGUF pet.iris.max_neighbors, ModelOptions::iris_max_neighbors)
  if (iris_max_neighbors_ > 0 && max_selected > iris_max_neighbors_)
    throw std::runtime_error("an atom has more than max_neighbors=" + std::to_string(iris_max_neighbors_) +
                             " neighbours selected within the cutoff " + std::to_string(cutoff_) + " (" +
                             std::to_string(max_selected) + ")");
}

void GraphModel::mark_precise(ggml_cgraph *g) const {
  if (precise_math()) mark_precise_math(g);
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
    ~Reset() { m.frag_cond_.clear(), m.lrc_hi_.clear(), m.lrc_blend_.clear(); }
  } reset{*this};
  if (cond.cl.s > 1) frag_cond_ = cond.row_charge, lrc_hi_ = cond.row_charge_hi, lrc_blend_ = cond.blend;
  // exact dE/dS (graph gradient w.r.t. the blend input) for structures in a transition / with a fractional cluster
  // charge; MLIPCPP_LRC_FD_DS=1: the earlier central differences in S (h = 0.05), for checks
  static const bool fd_ds = [] { const char *e = std::getenv("MLIPCPP_LRC_FD_DS"); return e && *e == '1'; }();
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
  const LrcEnergy le = lrc_energy(L, sq, cond, n, system.atomic_numbers(), x.data(), double(system.charge()),
                                  compute_forces);
  r.energy = static_cast<float>(double(r.energy) + le.energy);
  r.sqe_charges.assign(le.q.begin(), le.q.end());
  if (!compute_forces || r.forces.size() != le.forces.size()) return r;
  std::vector<double> F(le.forces), vir(le.virial);
  vir.resize(9, 0.0);
  // dE/dS from the graph, also for a structure without any edge (a backward pass for the blend alone there); the
  // central differences in S (h = 0.05) only on request: their truncation error was ~3 eV/A on bare ions (E04)
  if (cond.needs_cond_forces() && !fd_ds) {
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
  if (!cell.empty() && r.stress.size() == 6) { // the long-range and conditioning virial (Voigt, / V)
    const double *h = hv.data();
    const double V = std::abs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                              h[2] * (h[3] * h[7] - h[4] * h[6]));
    constexpr int va[6] = {0, 1, 2, 1, 0, 0}, vb[6] = {0, 1, 2, 2, 2, 1};
    for (int k = 0; k < 6; k++)
      r.stress[k] = static_cast<float>(double(r.stress[k]) + 0.5 * (vir[3 * va[k] + vb[k]] + vir[3 * vb[k] + va[k]]) / V);
  }
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
  // MLIPCPP_NODE_GRAD_DUMP=<file>: append every GIR node's forward value and dE/d(node) (the backward graph's
  // accumulated gradient) per structure (diagnostics, parity work: compare_node_grads_iris.py); plain path only
  static const char *node_grad_dump = [] {
    const char *e = std::getenv("MLIPCPP_NODE_GRAD_DUMP");
    return e && *e ? e : nullptr;
  }();
  if (!no_cache && !node_grad_dump && !post_compute_hook && !parameter_gradients && cached_path_supported() &&
      system.num_atoms() > 0 && !force_uncached_ && atom_w_.empty() && !lrc_want_blend_grad_)
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
    // iris PET-LR: the grid in double (adaptive::radius, as the cached path), clamped bump as pet-jax
    // (MLIPCPP_GRID_DOUBLE=1: the same for any grid model, diagnostics of the float grid below)
    static const bool grid_double_env = [] { const char *e = std::getenv("MLIPCPP_GRID_DOUBLE"); return e && *e == '1'; }();
    const bool grid_double = (iris_lr_.enabled || grid_double_env) && !solver;
    if (solver || grid_double) {
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
      std::vector<size_t> grid_off(n_atoms + 1, 0);
      for (int i = 0; i < n_atoms; i++) grid_off[i + 1] = grid_off[i] + by_center[i].size();
      std::vector<double> grid_jac(grid_double ? grid_off.back() : 0);
      for (int i = 0; i < n_atoms; i++) {
        const auto &es = by_center[i];
        if (grid_double)
          radii[i].r = adaptive::radius(true, static_cast<int>(es.size()), [&](int k) { return dist(es[k]); },
                                        cutoff_, num_neighbors_adaptive_, cutoff_width_adaptive_,
                                        grid_jac.data() + grid_off[i]);
        else
          radii[i] = adaptive::solve(static_cast<int>(es.size()), [&](int k) { return dist(es[k]); },
                                     cutoff_, num_neighbors_adaptive_, cutoff_width_adaptive_);
        atomic_cutoffs[i] = static_cast<float>(radii[i].r);
      }
      cutoff_jacobian.resize(n_in);
      if (grid_double) {
        for (int i = 0; i < n_atoms; i++)
          for (size_t k = 0; k < by_center[i].size(); k++) cutoff_jacobian[by_center[i][k]] = grid_jac[grid_off[i] + k];
      } else {
        for (int e = 0; e < n_in; e++)
          cutoff_jacobian[e] = adaptive::ddist(dist(e), radii[nlist.centers[e]], cutoff_width_adaptive_);
      }
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
  check_iris_overflow(max_neighbors);
  if (iris_lr_.enabled && iris_lr_.conserving_charges) {
    // his conserving-charges branch (iris.pet.sample component_charges): one Fermi level / residual correction per
    // connected component of the mutual edge graph, Q_c from fragment labels, or the structure charge for a structure
    // of ONE component; mlip.cpp has no fragment input, so (as his prep without labels) more components raise. Also
    // his range check: Q outside [n m_min, n m_max]
    std::vector<int> parent(n_atoms);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> root = [&](int a) { return parent[a] == a ? a : parent[a] = root(parent[a]); };
    for (int e = 0; e < nlist.num_pairs(); e++) parent[root(nlist.centers[e])] = root(nlist.neighbors[e]);
    int comps = 0;
    for (int i = 0; i < n_atoms; i++) comps += root(i) == i;
    if (comps > 1)
      throw std::runtime_error("iris PET-LR conserving_charges: " + std::to_string(comps) +
                               " components without fragment labels (his sample prep raises)");
    const double Q = system.charge();
    if (Q < double(n_atoms) * iris_lr_.m_min || Q > double(n_atoms) * iris_lr_.m_max)
      throw std::runtime_error("iris PET-LR conserving_charges: charge outside the charge-state range");
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
    if (!iris_lr_.enabled) compute_forces = false; // iris PET-LR: the Coulomb energy still moves the atoms
  }
  const bool backward = compute_forces || blend_only;
  // iris PET-LR graphs mask padded tokens themselves (-1e30, as pet-jax's flax mask); the exact mask's log(0) = -inf
  // would meet that WHERE (lowered as c a + (1 - c) b) as 0 * -inf = NaN
  interp_.set_exact_padding_mask(no_edges && !iris_lr_.enabled && !std::getenv("MLIPCPP_PAD_EMPTY"));

  // Set symbolic dimensions for this system
  interp_.set_dimension("n_atoms", n_atoms);
  interp_.set_dimension("max_neighbors", max_neighbors);
  interp_.set_dimension("n_edges", n_atoms * max_neighbors);
  interp_.set_dimension("max_neighbors_plus_one", max_neighbors + 1);
  if (iris_lr_.enabled && scores_interp_.has_graph()) {
    scores_interp_.set_dimension("n_atoms", n_atoms);
    scores_interp_.set_dimension("max_neighbors", max_neighbors);
    scores_interp_.set_dimension("n_edges", n_atoms * max_neighbors);
    scores_interp_.set_dimension("max_neighbors_plus_one", max_neighbors + 1);
  }

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
  ggml_tensor *charge_index_t = nullptr, *spin_index_t = nullptr, *charge_index_unit_t = nullptr;
  ggml_tensor *cond_blend_t = nullptr;
  if (conditioning_.enabled && (plato_lr_.frag.cond_gate || plato_lr_.lrc.enabled)) {
    charge_index_unit_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_I32, n_atoms);
    ggml_set_name(charge_index_unit_t, "charge_index_unit");
  }
  if (conditioning_.enabled && plato_lr_.lrc.enabled) {
    cond_blend_t = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, n_atoms);
    ggml_set_name(cond_blend_t, "cond_blend");
    if (backward && lrc_want_blend_grad_) ggml_set_param(cond_blend_t); // dE/dS (exact conditioning forces)
  }
  ggml_tensor *atom_w_t = nullptr;
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

  // iris PET-LR (pet.iris_lr): the LR branch's own inputs; positions are a gradient input
  ggml_tensor *iris_nbr = nullptr, *iris_pos = nullptr, *iris_Q = nullptr, *iris_mu = nullptr,
              *iris_gapped = nullptr, *iris_mround = nullptr, *iris_top = nullptr, *iris_bottom = nullptr,
              *iris_r0 = nullptr;
  // periodic cells: his jax-pme Ewald (graph_pbc.json): the real-space list at rc + skin in [n, n_lr] slots, the
  // halfspace k vectors, the volume; lattice shifts S @ cell, k vectors and volume are gradient inputs (virial)
  ggml_tensor *ew_index = nullptr, *ew_shift = nullptr, *ew_keep = nullptr, *ew_kvec = nullptr, *ew_vol = nullptr,
              *ew_smear = nullptr, *ew_cut = nullptr, *ew_slab = nullptr, *ew_normal = nullptr, *ew_basis = nullptr;
  std::vector<int32_t> ew_index_d;
  std::vector<float> ew_shift_d, ew_keep_d, ew_kvec_d, ew_scalars(5, 0.f), ew_normal_d(3, 0.f);
  std::vector<int32_t> ew_S;                // integer lattice shift of each slot (virial)
  std::vector<std::array<int, 3>> ew_kgrid; // integer k of each k vector (virial)
  double ew_h[3][3] = {};                   // the cell the Ewald maths uses
  int ew_slab_axis = -1;
  bool iris_pbc = false;
  if (iris_lr_.enabled) {
    const Cell *c = system.cell();
    iris_pbc = c && (c->periodic[0] || c->periodic[1] || c->periodic[2]);
    if (iris_pbc && iris_lr_.graph_pbc.empty())
      throw std::runtime_error("iris PET-LR: periodic structure but the GGUF has no graph_pbc.json (re-export)");
    if (iris_pbc != iris_lr_.pbc_loaded) {
      interp_.load_graph(iris_pbc ? iris_lr_.graph_pbc : iris_lr_.graph_open);
      iris_lr_.pbc_loaded = iris_pbc;
    }
    if (iris_pbc) {
      double cell[3][3];
      bool pbc[3];
      for (int a = 0; a < 3; a++) {
        pbc[a] = c->periodic[a];
        for (int b = 0; b < 3; b++) cell[a][b] = c->matrix[a][b];
      }
      std::vector<double> pos(system.positions(), system.positions() + 3 * size_t(n_atoms));
      iris::EwaldSetup es;
      if (!iris_lr_.freeze_ewald) {
        es = iris::ewald_setup(cell, pbc, pos.data(), n_atoms, iris_lr_.num_k, iris_lr_.smearing,
                               iris_lr_.slab_margin);
      } else { // his Calculator: frozen at the first call of a system, refrozen when a slab outgrows its margin
        auto &fz = iris_lr_.frozen;
        const int32_t *Z = system.atomic_numbers();
        const bool same = fz.valid && fz.n == n_atoms && std::equal(Z, Z + n_atoms, fz.numbers.begin()) &&
                          fz.pbc[0] == pbc[0] && fz.pbc[1] == pbc[1] && fz.pbc[2] == pbc[2];
        iris::EwaldFrozen f;
        bool refreeze = !same;
        if (same) {
          f = iris::EwaldFrozen::from_vec(fz.params);
          if (f.slab && iris::slab_thickness(cell, f.slab_axis, pos.data(), n_atoms) > f.thickness) {
            std::fprintf(stderr, "[mlipcpp iris] the slab outgrew its margin: new Ewald parameters\n");
            refreeze = true;
          }
        }
        if (refreeze) {
          f = iris::ewald_freeze(cell, pbc, pos.data(), n_atoms, iris_lr_.num_k, iris_lr_.smearing,
                                 iris_lr_.slab_margin);
          fz.valid = true, fz.n = n_atoms, fz.numbers.assign(Z, Z + n_atoms);
          for (int a = 0; a < 3; a++) fz.pbc[a] = pbc[a];
          fz.params = f.to_vec();
        }
        es = iris::ewald_apply(f, cell, pbc);
      }
      // full real-space list at rc + skin (his half list at rc + skin, both directions); r < rc masked in-graph
      NeighborList lr = NeighborListBuilder(NeighborListOptions{float(es.cutoff + 0.5), true, false}).build(system);
      std::vector<int> cnt(n_atoms, 0);
      for (int p = 0; p < lr.num_pairs(); p++) cnt[lr.centers[p]]++;
      const int kl = std::max(1, *std::max_element(cnt.begin(), cnt.end()));
      ew_index_d.assign(size_t(n_atoms) * kl, 0), ew_keep_d.assign(size_t(n_atoms) * kl, 0.f);
      ew_shift_d.assign(size_t(n_atoms) * kl * 3, 0.f);
      ew_S.assign(size_t(n_atoms) * kl * 3, 0);
      // his ToSample raises on a slab. On a 3-D cell the transfer kernel is the k-space sum at the batch's Ewald
      // smearing (jax-pme TiledEwald: lr_k2(sr_batch.smearing, k2), a host constant of the batch), so his stress is
      // the strain derivative at FIXED smearing, which is what the host virial assembles (no smearing gradient)
      if (iris_lr_.long_range_transfer && es.slab)
        throw std::runtime_error("iris PET-LR: long_range_transfer on a slab (his ToSample raises)");
      ew_kgrid = es.k_grid, ew_slab_axis = es.slab ? es.slab_axis : -1;
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) ew_h[a][b] = es.cellc[a][b];
      std::fill(cnt.begin(), cnt.end(), 0);
      for (int p = 0; p < lr.num_pairs(); p++) {
        const int i = lr.centers[p];
        const size_t sl = size_t(i) * kl + cnt[i]++;
        ew_index_d[sl] = lr.neighbors[p], ew_keep_d[sl] = 1.f;
        for (int a = 0; a < 3; a++) ew_S[3 * sl + a] = lr.cell_shifts[p][a];
        for (int b = 0; b < 3; b++) {
          double v = 0;
          for (int a = 0; a < 3; a++) v += lr.cell_shifts[p][a] * es.cellc[a][b];
          ew_shift_d[3 * sl + b] = float(v);
        }
      }
      const auto &h = es.cellc;
      const double det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) -
                         h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0]) +
                         h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
      double rec[3][3]; // rows b_i = 2 pi (a_j x a_k) / det: 2 pi inv(cell)^T
      for (int i = 0; i < 3; i++) {
        const double *aj = h[(i + 1) % 3], *ak = h[(i + 2) % 3];
        rec[i][0] = 2 * M_PI * (aj[1] * ak[2] - aj[2] * ak[1]) / det;
        rec[i][1] = 2 * M_PI * (aj[2] * ak[0] - aj[0] * ak[2]) / det;
        rec[i][2] = 2 * M_PI * (aj[0] * ak[1] - aj[1] * ak[0]) / det;
      }
      ew_kvec_d.resize(3 * es.k_grid.size());
      for (size_t m = 0; m < es.k_grid.size(); m++)
        for (int b = 0; b < 3; b++)
          ew_kvec_d[3 * m + b] = float(es.k_grid[m][0] * rec[0][b] + es.k_grid[m][1] * rec[1][b] + es.k_grid[m][2] * rec[2][b]);
      ew_scalars = {float(std::fabs(det)), float(es.smearing), float(es.cutoff), es.slab ? 1.f : 0.f, 0.f};
      if (es.slab) {
        const int k = es.slab_axis;
        const double *v1 = h[(k + 1) % 3], *v2 = h[(k + 2) % 3];
        double nh[3] = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
        const double nn = std::sqrt(nh[0] * nh[0] + nh[1] * nh[1] + nh[2] * nh[2]);
        for (int b = 0; b < 3; b++) ew_normal_d[b] = float(nh[b] / nn);
        ew_scalars[4] = float(std::fabs((h[k][0] * nh[0] + h[k][1] * nh[1] + h[k][2] * nh[2]) / nn));
      }
      interp_.set_dimension("n_lr", kl);
      interp_.set_dimension("n_kvec", int64_t(es.k_grid.size()));
    }
  }
  if (iris_lr_.enabled) {
    auto mk = [&](ggml_tensor *t, const char *name) { ggml_set_name(t, name); return t; };
    iris_nbr = mk(ggml_new_tensor_2d(input_ctx, GGML_TYPE_I32, max_neighbors, n_atoms), "neighbor_index");
    iris_pos = mk(ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, 3, n_atoms), "positions");
    if (compute_forces) ggml_set_param(iris_pos);
    iris_Q = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 1), "total_charge");
    iris_mu = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 1), "mu_bisect");
    iris_gapped = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 1), "gapped");
    iris_mround = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, n_atoms), "m_round");
    iris_top = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, n_atoms), "top");
    iris_bottom = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, n_atoms), "bottom");
    iris_r0 = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 3), "dipole_origin");
    if (iris_pbc) {
      const int64_t kl = int64_t(ew_keep_d.size()) / n_atoms, K = int64_t(ew_kvec_d.size()) / 3;
      ew_index = mk(ggml_new_tensor_2d(input_ctx, GGML_TYPE_I32, kl, n_atoms), "lr_index");
      ew_shift = mk(ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, 3, kl, n_atoms), "lr_shift");
      ew_keep = mk(ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, kl, n_atoms), "lr_keep");
      ew_kvec = mk(ggml_new_tensor_2d(input_ctx, GGML_TYPE_F32, 3, K), "kvec");
      ew_vol = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 1), "volume");
      ew_smear = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 1), "smearing");
      ew_cut = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 1), "cutoff");
      ew_slab = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 1), "slab");
      ew_normal = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 3), "slab_normal");
      ew_basis = mk(ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, 1), "slab_basis");
      if (compute_forces)
        for (ggml_tensor *t : {ew_shift, ew_kvec, ew_vol, ew_normal, ew_basis}) ggml_set_param(t);
    }
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

  if (atom_w_t) ggml_backend_tensor_set(atom_w_t, atom_w_.data(), 0, atom_w_.size() * sizeof(float));
  if (conditioning_.enabled) {
    const auto [ci, si] = conditioning_.rows(system.charge(), system.spin_multiplicity());
    (void)ci;
    std::vector<int32_t> c = charge_rows(n_atoms, system.charge(), system.spin_multiplicity()), sp(n_atoms, si);
    ggml_backend_tensor_set(charge_index_t, c.data(), 0, c.size() * sizeof(int32_t));
    ggml_backend_tensor_set(spin_index_t, sp.data(), 0, sp.size() * sizeof(int32_t));
    interp_.set_input("charge_index", charge_index_t);
    interp_.set_input("spin_index", spin_index_t);
    if (charge_index_unit_t) {
      const std::vector<int32_t> cu = charge_rows(n_atoms, system.charge(), system.spin_multiplicity(), true);
      ggml_backend_tensor_set(charge_index_unit_t, cu.data(), 0, cu.size() * sizeof(int32_t));
      interp_.set_input("charge_index_unit", charge_index_unit_t);
      if (cond_blend_t) {
        std::vector<float> bl(n_atoms, 0.0f);
        if (!lrc_blend_.empty()) bl = lrc_blend_;
        ggml_backend_tensor_set(cond_blend_t, bl.data(), 0, bl.size() * sizeof(float));
        interp_.set_input("cond_blend", cond_blend_t);
        install_lrc_blend(interp_, cond_blend_t);
      } else {
        install_frag_cond_gate(interp_);
      }
    }
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

  if (iris_lr_.enabled) {
    std::vector<int32_t> nbr(total_slots);
    for (int f = 0; f < total_slots; f++) nbr[f] = std::max(neighbor_atoms_vec[f], 0); // padded: weight 0
    ggml_backend_tensor_set(iris_nbr, nbr.data(), 0, nbr.size() * sizeof(int32_t));
    ggml_backend_tensor_set(iris_pos, system.positions(), 0, 3 * size_t(n_atoms) * sizeof(float));
    const float Q = static_cast<float>(system.charge());
    ggml_backend_tensor_set(iris_Q, &Q, 0, sizeof(float));
    std::array<double, 3> r0{0, 0, 0};
    if (system.has_dipole_origin()) {
      r0 = system.dipole_origin();
    } else { // his calculator: atoms.get_center_of_mass() (ASE masses)
      double M = 0;
      for (int i = 0; i < n_atoms; i++) {
        const double m = ASE_MASSES[std::clamp(atomic_numbers[i], 0, 118)];
        M += m;
        for (int a = 0; a < 3; a++) r0[a] += m * system.positions()[3 * i + a];
      }
      for (auto &x : r0) x /= M;
    }
    const float r0f[3] = {float(r0[0]), float(r0[1]), float(r0[2])};
    ggml_backend_tensor_set(iris_r0, r0f, 0, sizeof(r0f));
    std::vector<float> zeros(n_atoms, 0.f);
    const float z1 = 0.f;
    for (ggml_tensor *t : {iris_mu, iris_gapped}) ggml_backend_tensor_set(t, &z1, 0, sizeof(float));
    for (ggml_tensor *t : {iris_mround, iris_top, iris_bottom})
      ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size() * sizeof(float));
    for (ggml_tensor *t : {iris_nbr, iris_pos, iris_Q, iris_mu, iris_gapped, iris_mround, iris_top, iris_bottom, iris_r0})
      interp_.set_input(ggml_get_name(t), t);
    if (iris_pbc) {
      ggml_backend_tensor_set(ew_index, ew_index_d.data(), 0, ew_index_d.size() * sizeof(int32_t));
      ggml_backend_tensor_set(ew_shift, ew_shift_d.data(), 0, ew_shift_d.size() * sizeof(float));
      ggml_backend_tensor_set(ew_keep, ew_keep_d.data(), 0, ew_keep_d.size() * sizeof(float));
      ggml_backend_tensor_set(ew_kvec, ew_kvec_d.data(), 0, ew_kvec_d.size() * sizeof(float));
      ggml_tensor *sc[5] = {ew_vol, ew_smear, ew_cut, ew_slab, ew_basis};
      for (int i = 0; i < 5; i++) ggml_backend_tensor_set(sc[i], &ew_scalars[i], 0, sizeof(float));
      ggml_backend_tensor_set(ew_normal, ew_normal_d.data(), 0, 3 * sizeof(float));
      for (ggml_tensor *t : {ew_index, ew_shift, ew_keep, ew_kvec, ew_vol, ew_smear, ew_cut, ew_slab, ew_normal, ew_basis})
        interp_.set_input(ggml_get_name(t), t);
    }
    // the scores graph reads the trunk's inputs, by name
    if (scores_interp_.has_graph())
      for (const auto &in : scores_interp_.graph().inputs) {
        ggml_tensor *t = ggml_get_tensor(input_ctx, in.name.c_str());
        if (!t) throw std::runtime_error("iris PET-LR scores graph: no input " + in.name);
        scores_interp_.set_input(in.name, t);
      }
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
  ggml_tensor *iris_scores = nullptr, *iris_aux = nullptr;
  if (iris_lr_.enabled && output) {
    if (scores_interp_.has_graph()) {
      iris_scores = scores_interp_.build(compute_ctx);
      if (!iris_scores) throw std::runtime_error("iris PET-LR: scores graph failed to build");
      ggml_set_output(iris_scores);
    }
    // output [n, n_out]: column 0 is the energy (the only column differentiated), the rest is read back
    if (output->ne[0] != iris_lr_.n_out || output->ne[1] != n_atoms)
      throw std::runtime_error("iris PET-LR: graph output is not [n_atoms, n_out]");
    iris_aux = ggml_cont(compute_ctx, output);
    ggml_set_output(iris_aux);
    output = ggml_cont(compute_ctx, ggml_view_2d(compute_ctx, iris_aux, 1, n_atoms, iris_aux->nb[1], 0));
  }
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

  if (backward) {
    // Build forward + backward graph
    ggml_tensor *total_energy = atom_w_t ? ggml_sum(compute_ctx, ggml_mul(compute_ctx, output,
                                                                          ggml_reshape(compute_ctx, atom_w_t, output)))
                                         : ggml_sum(compute_ctx, output);
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
    ggml_tensor *grad = ggml_graph_get_grad(cgraph, edge_vectors_t);
    ggml_tensor *pos_grad = iris_pos ? ggml_graph_get_grad(cgraph, iris_pos) : nullptr;
    if (pos_grad) ggml_set_output(pos_grad);
    for (ggml_tensor *t : {ew_shift, ew_kvec, ew_vol, ew_normal, ew_basis})
      if (ggml_tensor *g = t ? ggml_graph_get_grad(cgraph, t) : nullptr) ggml_set_output(g);
    if (grad) {
      ggml_set_output(grad);
    } else if (!pos_grad) {
      compute_forces = false;
    }
    if (plato)
      for (ggml_tensor *p : {plato->positions, plato->strain})
        if (ggml_tensor *g = p ? ggml_graph_get_grad(cgraph, p) : nullptr)
          for (; g; g = g->view_src) ggml_set_output(g);
    if (cond_blend_t && (cond_blend_t->flags & GGML_TENSOR_FLAG_PARAM))
      if (ggml_tensor *g = ggml_graph_get_grad(cgraph, cond_blend_t)) ggml_set_output(g);
    if (node_grad_dump) // keep every forward value and gradient alive (no buffer reuse, no in-place ops)
      interp_.capture_all_outputs([&](int, const std::string &, ggml_tensor *t) {
        ggml_set_output(t);
        if (ggml_tensor *g = ggml_graph_get_grad(cgraph, t)) ggml_set_output(g);
      });
  } else {
    cgraph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(cgraph, output);
    if (nc_output) ggml_build_forward_expand(cgraph, nc_output);
  }

  if (precise_math()) mark_precise_math(cgraph);

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
  if (iris_scores) scores_interp_.init_constants();
  if (plato && plato->needs_neighbor_index()) {
    std::vector<int32_t> nbr(neighbor_atoms_vec.begin(), neighbor_atoms_vec.end());
    for (auto &j : nbr) j = std::max(j, 0);
    plato->set_neighbor_index(std::move(nbr));
  }
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
  if (iris_lr_.enabled) {
    // two-pass mu: the scores graph forward, the bisection on the host (his stop_gradient), then the full graph
    if (iris_scores) {
      ggml_cgraph *sg = ggml_new_graph_custom(compute_ctx, 32768, false);
      ggml_build_forward_expand(sg, iris_scores);
      if (precise_math()) mark_precise_math(sg);
      if (backend_provider_->compute(sg) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("iris PET-LR: scores graph failed");
      // species_fill: the scores graph returns [n, 2] = (s, c), c the Fermi level's weights
      const bool weighted = iris_lr_.reference == "species_fill";
      const int sc = weighted ? 2 : 1;
      if (ggml_nelements(iris_scores) != int64_t(sc) * n_atoms)
        throw std::runtime_error("iris PET-LR: scores graph output size");
      std::vector<float> sf(size_t(sc) * n_atoms);
      ggml_backend_tensor_get(iris_scores, sf.data(), 0, sf.size() * sizeof(float));
      std::vector<double> sd(n_atoms), wd(weighted ? n_atoms : 0);
      for (int i = 0; i < n_atoms; i++) {
        sd[i] = sf[size_t(i) * sc];
        if (weighted) wd[i] = sf[size_t(i) * sc + 1];
      }
      iris::FermiConfig fc;
      fc.m_min = iris_lr_.m_min, fc.m_max = iris_lr_.m_max, fc.temperature = iris_lr_.temperature;
      fc.iterations = iris_lr_.iterations, fc.eps = iris_lr_.eps;
      const iris::FermiHost h = iris::fermi_host(sd, double(system.charge()), fc, weighted ? &wd : nullptr);
      const float mu = float(h.mu), gap = h.gapped ? 1.f : 0.f;
      ggml_backend_tensor_set(iris_mu, &mu, 0, sizeof(float));
      ggml_backend_tensor_set(iris_gapped, &gap, 0, sizeof(float));
      ggml_backend_tensor_set(iris_mround, h.m_round.data(), 0, n_atoms * sizeof(float));
      ggml_backend_tensor_set(iris_top, h.top.data(), 0, n_atoms * sizeof(float));
      ggml_backend_tensor_set(iris_bottom, h.bottom.data(), 0, n_atoms * sizeof(float));
    }
    profile.mark("iris_scores");
  }
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
  // MLIPCPP_ATOMIC_E_DUMP=<file>: append n and the per-atom graph energies (unscaled) + energy_scale (diagnostics)
  if (static const char *ep = std::getenv("MLIPCPP_ATOMIC_E_DUMP"); ep && *ep)
    if (FILE *fp = std::fopen(ep, "ab")) {
      const int32_t n32 = n_atoms;
      const float sc = energy_scale_;
      std::fwrite(&n32, 4, 1, fp);
      std::fwrite(atomic_energies.data(), 4, n_atoms, fp);
      std::fwrite(&sc, 4, 1, fp);
      std::fclose(fp);
    }
  float model_energy = 0.0f;
  static const bool energy_double = [] { const char *e = std::getenv("MLIPCPP_ENERGY_DOUBLE"); return e && *e == '1'; }();
  if (energy_double) { // diagnostics (parity work): the per-atom sum in double, rounded once
    double acc = 0.0;
    for (int i = 0; i < n_atoms; i++) acc += atomic_energies[i];
    model_energy = static_cast<float>(acc);
  } else
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

      if (iris_pos) // iris PET-LR: the explicit positions dependence (Coulomb, SR kernel, dipole lever)
        if (ggml_tensor *g = ggml_graph_get_grad(cgraph, iris_pos)) {
          std::vector<float> gp(3 * size_t(n_atoms));
          ggml_backend_tensor_get(g, gp.data(), 0, gp.size() * sizeof(float));
          for (int i = 0; i < n_atoms * 3; i++) result.forces[i] -= gp[i];
          if (iris_pbc) { // the Ewald virial (row-major, as plato_geps): his predict stress
            // sum_i r_a dE/dr_b + (h^T dE/dh)_ab, with dE/dh through the graph inputs that depend on the cell h the
            // Ewald maths uses: shifts S @ h, k = n @ 2 pi h^-T, V = |det h|, a slab's normal and basis; a slab's
            // effective (non-periodic) row is a constant (his compose_cell). The edge-vector terms follow below.
            // scripts/pet_lr_port/check_pbc64.py checks this sum against the strain derivative (fp64, 1e-15).
            std::vector<double> w(9, 0.0);
            double G[3][3] = {};
            auto grad = [&](ggml_tensor *t, size_t n) {
              std::vector<float> d(n, 0.f);
              if (ggml_tensor *g = t ? ggml_graph_get_grad(cgraph, t) : nullptr)
                ggml_backend_tensor_get(g, d.data(), 0, n * sizeof(float));
              return d;
            };
            const float *x = system.positions();
            for (int i = 0; i < n_atoms; i++)
              for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) w[3 * a + b] += double(x[3 * i + a]) * gp[3 * i + b];
            const std::vector<float> gs = grad(ew_shift, ew_shift_d.size());
            for (size_t sl = 0; sl < ew_keep_d.size(); sl++)
              if (ew_keep_d[sl] > 0.f)
                for (int a = 0; a < 3; a++)
                  for (int b = 0; b < 3; b++) G[a][b] += double(ew_S[3 * sl + a]) * gs[3 * sl + b];
            const auto &h = ew_h;
            const double det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) -
                               h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0]) +
                               h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
            double X[3][3]; // inv(h)
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++) {
                const int i1 = (j + 1) % 3, i2 = (j + 2) % 3, j1 = (i + 1) % 3, j2 = (i + 2) % 3;
                X[i][j] = (h[i1][j1] * h[i2][j2] - h[i1][j2] * h[i2][j1]) / det;
              }
            const std::vector<float> gk = grad(ew_kvec, ew_kvec_d.size());
            double dR[3][3] = {}; // dE/dR, R = 2 pi h^-T (rows), k = n @ R
            for (size_t m = 0; m < ew_kgrid.size(); m++)
              for (int i = 0; i < 3; i++)
                for (int b = 0; b < 3; b++) dR[i][b] += double(ew_kgrid[m][i]) * gk[3 * m + b];
            // dE/dX = 2 pi dR^T; dE/dh = -X^T (dE/dX) X^T
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++) {
                double v = 0;
                for (int p = 0; p < 3; p++)
                  for (int q = 0; q < 3; q++) v += X[p][a] * 2 * M_PI * dR[q][p] * X[b][q];
                G[a][b] -= v;
              }
            const double gV = grad(ew_vol, 1)[0];
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++) G[a][b] += gV * std::fabs(det) * X[b][a];
            if (ew_slab_axis >= 0) { // n = c / |c|, c = a1 x a2; basis = |h_k . n|
              const int k = ew_slab_axis;
              const double *a1 = h[(k + 1) % 3], *a2 = h[(k + 2) % 3];
              const double c[3] = {a1[1] * a2[2] - a1[2] * a2[1], a1[2] * a2[0] - a1[0] * a2[2],
                                   a1[0] * a2[1] - a1[1] * a2[0]};
              const double cn = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
              const double nh[3] = {c[0] / cn, c[1] / cn, c[2] / cn};
              const double hk_n = h[k][0] * nh[0] + h[k][1] * nh[1] + h[k][2] * nh[2];
              const std::vector<float> gn0 = grad(ew_normal, 3);
              const double gb = grad(ew_basis, 1)[0];
              double gn[3], gc[3];
              for (int b = 0; b < 3; b++) gn[b] = gn0[b] + gb * (hk_n >= 0 ? 1.0 : -1.0) * h[k][b];
              const double ngn = nh[0] * gn[0] + nh[1] * gn[1] + nh[2] * gn[2];
              for (int b = 0; b < 3; b++) gc[b] = (gn[b] - nh[b] * ngn) / cn;
              const double r1[3] = {a2[1] * gc[2] - a2[2] * gc[1], a2[2] * gc[0] - a2[0] * gc[2], a2[0] * gc[1] - a2[1] * gc[0]};
              const double r2[3] = {gc[1] * a1[2] - gc[2] * a1[1], gc[2] * a1[0] - gc[0] * a1[2], gc[0] * a1[1] - gc[1] * a1[0]};
              for (int b = 0; b < 3; b++) G[(k + 1) % 3][b] += r1[b], G[(k + 2) % 3][b] += r2[b], G[k][b] = 0;
            }
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++)
                for (int A = 0; A < 3; A++) w[3 * a + b] += h[A][a] * G[A][b];
            plato_geps.assign(w.begin(), w.end()); // (no Plato branch in an iris model: the slot is free)
          }
        }

      // MLIPCPP_FORCE_SPLIT=<file>: append this structure's force pieces (diagnostics, parity work): n, k, the
      // slot neighbours, padding, dE/d edge_vectors [n, k, 3], dE/d cutoff_values [n, k], dE/d positions [n, 3]
      // and the per-atom chain term of the adaptive cutoffs [n, 3], all unscaled (energy_scale written last)
      static const char *split_path = std::getenv("MLIPCPP_FORCE_SPLIT");
      if (split_path && *split_path) {
        if (FILE *fp = std::fopen(split_path, "ab")) {
          const int32_t hdr[2] = {n_atoms, max_neighbors};
          std::fwrite(hdr, sizeof(int32_t), 2, fp);
          std::vector<int32_t> nb(neighbor_atoms_vec.begin(), neighbor_atoms_vec.begin() + total_slots);
          std::fwrite(nb.data(), sizeof(int32_t), total_slots, fp);
          std::fwrite(pm_data.data(), sizeof(float), total_slots, fp);
          std::fwrite(grad_data.data(), sizeof(float), size_t(total_slots) * 3, fp);
          std::vector<float> cvg(total_slots, 0.0f), gpos(3 * size_t(n_atoms), 0.0f), fcut(3 * size_t(n_atoms), 0.0f);
          if (ggml_tensor *g = cutoff_values_t ? ggml_graph_get_grad(cgraph, cutoff_values_t) : nullptr)
            if (g->data) ggml_backend_tensor_get(g, cvg.data(), 0, ggml_nbytes(g));
          if (ggml_tensor *g = iris_pos ? ggml_graph_get_grad(cgraph, iris_pos) : nullptr)
            ggml_backend_tensor_get(g, gpos.data(), 0, gpos.size() * sizeof(float));
          for (size_t e = 0; e < chain_grad.size(); e++)
            for (int c = 0; c < 3; c++) {
              fcut[full_nlist.centers[e] * 3 + c] += float(chain_grad[e][c]);
              fcut[full_nlist.neighbors[e] * 3 + c] -= float(chain_grad[e][c]);
            }
          std::fwrite(cvg.data(), sizeof(float), total_slots, fp);
          std::fwrite(gpos.data(), sizeof(float), gpos.size(), fp);
          std::fwrite(fcut.data(), sizeof(float), fcut.size(), fp);
          const float sc = energy_scale_;
          std::fwrite(&sc, sizeof(float), 1, fp);
          std::fwrite(ev_data.data(), sizeof(float), size_t(total_slots) * 3, fp); // the graph's inputs
          std::fwrite(cv_data.data(), sizeof(float), total_slots, fp);
          std::fclose(fp);
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
          // the full virial W_ab = sum_edges r_a dE/dr_b (row-major, eV, double): PET is not exactly rotation
          // invariant, so W has an antisymmetric part that the Voigt stress drops (parity outputs)
          result.virial.assign(9, 0.0);
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
              const double e3[3] = {ex, ey, ez}, g3[3] = {gx, gy, gz};
              for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) result.virial[3 * a + b] += e3[a] * g3[b];
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
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++) result.virial[3 * a + b] += double(r[a]) * double(g[b]);
          }

          if (!plato_geps.empty()) { // row-major dE/d eps
            const auto &q = plato_geps;
            result.stress[0] += q[0], result.stress[1] += q[4], result.stress[2] += q[8];
            result.stress[3] += 0.5f * (q[5] + q[7]);
            result.stress[4] += 0.5f * (q[2] + q[6]);
            result.stress[5] += 0.5f * (q[1] + q[3]);
            for (int i = 0; i < 9; i++) result.virial[i] += q[i];
          }
          for (int i = 0; i < 6; i++) {
            result.stress[i] *= energy_scale_ / vol;
          }
          for (auto &w : result.virial) w *= energy_scale_;
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

  if (iris_aux) {
    std::vector<float> aux(size_t(n_atoms) * iris_lr_.n_out);
    ggml_backend_tensor_get(iris_aux, aux.data(), 0, aux.size() * sizeof(float));
    const int C = iris_lr_.n_out;
    result.charges.resize(n_atoms);
    result.dipole.assign(3, 0.0);
    result.metallicity = 0;
    for (int i = 0; i < n_atoms; i++) {
      result.charges[i] = aux[size_t(i) * C + 1];
      for (int a = 0; a < 3; a++) result.dipole[a] += aux[size_t(i) * C + 2 + a];
      result.metallicity += aux[size_t(i) * C + 7];
    }
    result.fermi_level = n_atoms ? aux[5] : 0.0;
    result.has_charges = true;
  }

  if (no_edges && forces_requested && !iris_lr_.enabled) { // energy independent of the positions
    result.forces.assign(3 * static_cast<size_t>(n_atoms), 0.0f);
    result.has_forces = true;
    if (system.cell()) {
      result.stress.assign(6, 0.0f);
      result.has_stress = true;
    }
  }

  profile.mark("readback");
  if (node_grad_dump && backward) {
    // per structure: int32 'FRME', int32 count; per entry: int32 id, int32 name length, name, value block, int32
    // has_grad, [grad block]; block: int32 ndims, int64 ne[ndims], int64 nb[ndims] (byte strides: views are written
    // as their raw extent), int32 dtype (0 f32, 1 i32, 2 other), int64 nbytes, data
    std::vector<std::tuple<int, std::string, ggml_tensor *>> entries;
    interp_.capture_all_outputs([&](int id, const std::string &name, ggml_tensor *t) { entries.emplace_back(id, name, t); });
    if (FILE *fp = std::fopen(node_grad_dump, "ab")) {
      const int32_t hdr[2] = {0x454D5246, int32_t(entries.size())};
      std::fwrite(hdr, 4, 2, fp);
      auto block = [fp](ggml_tensor *t) {
        const int32_t nd = ggml_n_dims(t);
        const int32_t dt = t->type == GGML_TYPE_F32 ? 0 : t->type == GGML_TYPE_I32 ? 1 : 2;
        std::vector<int64_t> ne(t->ne, t->ne + nd), nb(t->nb, t->nb + nd);
        std::vector<char> data((t->data || t->view_src) ? ggml_nbytes(t) : 0);
        if (!data.empty()) ggml_backend_tensor_get(t, data.data(), 0, data.size());
        const int64_t n = int64_t(data.size());
        std::fwrite(&nd, 4, 1, fp), std::fwrite(ne.data(), 8, nd, fp), std::fwrite(nb.data(), 8, nd, fp);
        std::fwrite(&dt, 4, 1, fp), std::fwrite(&n, 8, 1, fp), std::fwrite(data.data(), 1, data.size(), fp);
      };
      for (const auto &[id, name, t] : entries) {
        const int32_t id32 = id, nl = int32_t(name.size());
        std::fwrite(&id32, 4, 1, fp), std::fwrite(&nl, 4, 1, fp), std::fwrite(name.data(), 1, nl, fp);
        block(t);
        ggml_tensor *g = ggml_graph_get_grad(cgraph, t);
        const int32_t has = g && (g->data || g->view_src) ? 1 : 0;
        std::fwrite(&has, 4, 1, fp);
        if (has) block(g);
      }
      std::fclose(fp);
    }
  }
  // Hook for callers that want to inspect intermediate tensors. Runs while
  // the compute buffer (and therefore every node_outputs_ tensor) is still
  // alive on the active backend.
  if (post_compute_hook) {
    post_compute_hook();
  }

  return result;
}

} // namespace mlipcpp::runtime
