#include "train/lr_schedule.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace mlipcpp::train {

namespace {

// Python's round(): half to even (the default floating-point rounding mode)
int64_t py_round(double x) { return static_cast<int64_t>(std::nearbyint(x)); }

// fnmatch.fnmatchcase: '*', '?', '[seq]', '[!seq]' on the whole string
bool glob(const char *p, const char *s) {
  for (; *p; p++, s++) {
    if (*p == '*') {
      while (*p == '*') p++;
      if (!*p) return true;
      for (; *s; s++)
        if (glob(p, s)) return true;
      return false;
    }
    if (!*s) return false;
    if (*p == '?') continue;
    if (*p == '[') {
      const char *q = p + 1;
      const bool neg = *q == '!';
      if (neg) q++;
      bool hit = false;
      const char *start = q;
      for (; *q && (*q != ']' || q == start); q++) {
        if (q[1] == '-' && q[2] && q[2] != ']') {
          hit |= *s >= *q && *s <= q[2];
          q += 2;
        } else {
          hit |= *s == *q;
        }
      }
      if (!*q) { // no closing bracket: a literal '['
        if (*s != '[') return false;
        continue;
      }
      if (hit == neg) return false;
      p = q;
      continue;
    }
    if (*p != *s) return false;
  }
  return !*s;
}

} // namespace

LrSchedule LrSchedule::load(const std::string &path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read the lr schedule " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return parse(ss.str());
}

LrSchedule LrSchedule::parse(const std::string &text) {
  const nlohmann::json j = nlohmann::json::parse(text);
  LrSchedule s;
  if (!j.contains("base_lr")) throw std::runtime_error("lr schedule: base_lr is required");
  s.base = j["base_lr"].get<double>();
  s.min = j.value("min_lr", 0.0);
  s.shape = j.value("shape", std::string("cosine"));
  if (s.shape != "cosine" && s.shape != "linear" && s.shape != "constant")
    throw std::runtime_error("lr schedule: unknown shape '" + s.shape + "'");
  s.warmup_epochs = j.value("warmup_epochs", 0.0);
  s.decay_after_epochs = j.value("decay_after_epochs", 0.0);
  s.default_group = j.value("default_group", std::string("default"));
  for (const auto &g : j.value("groups", nlohmann::json::array())) {
    LrGroup x;
    if (!g.contains("name")) throw std::runtime_error("lr schedule: every group needs a name");
    x.name = g["name"].get<std::string>();
    x.match = g.value("match", std::vector<std::string>{});
    x.lr_mult = g.value("lr_mult", 1.0);
    x.freeze_epochs = g.value("freeze_epochs", 0.0);
    x.warmup_epochs = g.value("warmup_epochs", 0.0);
    x.weight_decay = g.value("weight_decay", 0.0);
    const std::string opt = g.value("optimizer", std::string("adam"));
    if (opt != "adam" && opt != "muon") throw std::runtime_error("lr schedule: unknown optimizer '" + opt + "'");
    x.muon = opt == "muon";
    if (s.group_index(x.name) >= 0) throw std::runtime_error("lr schedule: group '" + x.name + "' twice");
    s.groups.push_back(std::move(x));
  }
  if (s.group_index(s.default_group) < 0) {
    LrGroup d;
    d.name = s.default_group;
    s.groups.push_back(d);
  }
  return s;
}

bool LrSchedule::matches(const std::string &name, const std::string &pattern) {
  if (pattern.find('*') != std::string::npos) return glob(pattern.c_str(), name.c_str());
  return name.find(pattern) != std::string::npos;
}

int LrSchedule::group_index(const std::string &name) const {
  for (size_t g = 0; g < groups.size(); g++)
    if (groups[g].name == name) return static_cast<int>(g);
  return -1;
}

int LrSchedule::group_of(const std::string &name) const {
  for (size_t g = 0; g < groups.size(); g++)
    for (const auto &p : groups[g].match)
      if (matches(name, p)) return static_cast<int>(g);
  return group_index(default_group);
}

double LrSchedule::base_lr(int64_t step, int64_t S, int64_t total) const {
  const int64_t W = py_round(warmup_epochs * double(S));
  const int64_t D = std::max(W, py_round(decay_after_epochs * double(S)));
  if (step < W) return base * std::min(1.0, double(step + 1) / double(W));
  if (step < D || shape == "constant") return base;
  const double p = std::min(1.0, double(step - D) / double(std::max<int64_t>(1, total - 1 - D)));
  if (shape == "linear") return base - (base - min) * p;
  return min + (base - min) * 0.5 * (1.0 + std::cos(M_PI * p));
}

std::pair<double, bool> LrSchedule::group_lr(int g, int64_t step, int64_t S, int64_t total) const {
  const LrGroup &x = groups.at(g);
  const int64_t F = py_round(x.freeze_epochs * double(S));
  if (step < F) return {0.0, true};
  const int64_t W = py_round(x.warmup_epochs * double(S));
  const double ramp = W == 0 ? 1.0 : std::min(1.0, double(step - F + 1) / double(W));
  return {base_lr(step, S, total) * x.lr_mult * ramp, false};
}

std::string LrSchedule::table_json(int64_t n, int64_t S, int64_t total) const {
  // %.17g: the doubles exactly (json.dump writes repr, the shortest round trip)
  std::string out = "{\"steps_per_epoch\": " + std::to_string(S) + ", \"total\": " + std::to_string(total) +
                    ", \"lr\": {";
  std::string fr = "\"frozen\": {";
  char buf[64];
  for (size_t g = 0; g < groups.size(); g++) {
    out += (g ? ", \"" : "\"") + groups[g].name + "\": [";
    fr += (g ? ", \"" : "\"") + groups[g].name + "\": [";
    for (int64_t k = 0; k < n; k++) {
      const auto [lr, frozen] = group_lr(static_cast<int>(g), k, S, total);
      std::snprintf(buf, sizeof buf, "%s%.17g", k ? ", " : "", lr);
      out += buf;
      fr += std::string(k ? ", " : "") + (frozen ? "true" : "false");
    }
    out += "]", fr += "]";
  }
  return out + "}, " + fr + "}}";
}

// ---------------------------------------------------------------------------

GroupOptimizer::GroupOptimizer(ggml_backend *backend, std::vector<Param> params, LrSchedule schedule, double clip,
                               double muon_momentum, double muon_lr_scale, Orthogonalize orthogonalize)
    : backend_(backend), params_(std::move(params)), schedule_(std::move(schedule)), clip_(clip), mu_(muon_momentum),
      muon_lr_scale_(muon_lr_scale), orthogonalize_(std::move(orthogonalize)) {
  const size_t P = params_.size(), G = schedule_.groups.size();
  for (const Param &p : params_) {
    const int g = schedule_.group_of(p.name);
    group_.push_back(g);
    use_muon_.push_back(schedule_.groups[g].muon && p.muon_matrix);
    rms_.push_back(0.2f * std::sqrt(static_cast<float>(std::max(p.param->ne[0], p.param->ne[1]))));
    if (use_muon_.back() && !orthogonalize_) throw std::runtime_error("lr schedule: Muon group without Newton-Schulz");
  }
  t_.assign(G, 0);
  ctx_state_ = ggml_init({ggml_tensor_overhead() * (3 * P + G + 4), nullptr, true});
  for (size_t k = 0; k < P; k++) {
    m_.push_back(ggml_dup_tensor(ctx_state_, params_[k].param));
    v_.push_back(use_muon_[k] ? nullptr : ggml_dup_tensor(ctx_state_, params_[k].param));
    muon_pars_.push_back(use_muon_[k] ? ggml_new_tensor_1d(ctx_state_, GGML_TYPE_F32, 2) : nullptr);
  }
  for (size_t g = 0; g < G; g++) adam_pars_.push_back(ggml_new_tensor_1d(ctx_state_, GGML_TYPE_F32, 7));
  clip_t_ = ggml_new_tensor_1d(ctx_state_, GGML_TYPE_F32, 1);
  buf_state_ = ggml_backend_alloc_ctx_tensors(ctx_state_, backend_);
  if (!buf_state_) throw std::runtime_error("lr schedule: cannot allocate the optimizer state");
  ggml_backend_buffer_clear(buf_state_, 0);
  const float c = static_cast<float>(clip_);
  ggml_backend_tensor_set(clip_t_, &c, 0, sizeof c);
}

GroupOptimizer::~GroupOptimizer() {
  if (galloc_) ggml_gallocr_free(galloc_);
  if (ctx_graph_) ggml_free(ctx_graph_);
  if (buf_state_) ggml_backend_buffer_free(buf_state_);
  if (ctx_state_) ggml_free(ctx_state_);
}

std::vector<size_t> GroupOptimizer::group_sizes() const {
  std::vector<size_t> n(schedule_.groups.size(), 0);
  for (size_t k = 0; k < params_.size(); k++) n[group_[k]] += ggml_nelements(params_[k].param);
  return n;
}

void GroupOptimizer::build(const std::vector<bool> &frozen) {
  if (galloc_) ggml_gallocr_free(galloc_), galloc_ = nullptr;
  if (ctx_graph_) ggml_free(ctx_graph_), ctx_graph_ = nullptr;
  const size_t n_nodes = 96 * params_.size() + 64;
  ctx_graph_ = ggml_init({ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true});
  ggml_context *ctx = ctx_graph_;
  graph_ = ggml_new_graph_custom(ctx, n_nodes, false);
  // the norm of the non-frozen gradients (frozen ones are dropped before the clipping)
  ggml_tensor *sumsq = nullptr;
  for (size_t k = 0; k < params_.size(); k++) {
    if (frozen[group_[k]]) continue;
    ggml_tensor *s2 = ggml_sum(ctx, ggml_sqr(ctx, params_[k].grad));
    sumsq = sumsq ? ggml_add(ctx, sumsq, s2) : s2;
  }
  if (!sumsq) { // everything frozen
    norm_ = nullptr;
    built_for_ = frozen;
    return;
  }
  norm_ = ggml_sqrt(ctx, sumsq);
  ggml_set_output(norm_);
  ggml_build_forward_expand(graph_, norm_);
  // c = min(1, clip / (|g| + 1e-6)), as both trainers
  ggml_tensor *c = ggml_clamp(ctx, ggml_div(ctx, clip_t_, ggml_scale_bias(ctx, norm_, 1.0f, 1e-6f)), 0.0f, 1.0f);
  const float mu = static_cast<float>(mu_);
  for (size_t k = 0; k < params_.size(); k++) {
    if (frozen[group_[k]]) continue;
    ggml_tensor *p = params_[k].param;
    ggml_tensor *g = ggml_mul(ctx, params_[k].grad, c);
    if (!use_muon_[k]) {
      ggml_build_forward_expand(graph_, ggml_opt_step_adamw(ctx, p, g, m_[k], v_[k], adam_pars_[group_[k]]));
      continue;
    }
    // Muon: momentum b = mu b + (1 - mu) g, Nesterov u = (1 - mu) g + mu b, Newton-Schulz of the
    // stack [ne0, ne1, k], decoupled weight decay x (1 - lr wd) and the step lr' o
    ggml_tensor *b = ggml_add(ctx, ggml_scale(ctx, m_[k], mu), ggml_scale(ctx, g, 1.0f - mu));
    ggml_build_forward_expand(graph_, ggml_cpy(ctx, b, m_[k]));
    ggml_tensor *u = ggml_add(ctx, ggml_scale(ctx, g, 1.0f - mu), ggml_scale(ctx, b, mu));
    ggml_tensor *o = orthogonalize_(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, u), u->ne[0], u->ne[1],
                                                       ggml_nelements(u) / (u->ne[0] * u->ne[1])));
    ggml_build_forward_expand(graph_, ggml_opt_step_sgd(ctx, p, ggml_reshape(ctx, ggml_cont(ctx, o), p), muon_pars_[k]));
  }
  galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  if (!ggml_gallocr_alloc_graph(galloc_, graph_)) throw std::runtime_error("lr schedule: cannot allocate the step graph");
  built_for_ = frozen;
}

double GroupOptimizer::step(int64_t n, int64_t S, int64_t total) {
  const size_t G = schedule_.groups.size();
  std::vector<bool> frozen(G);
  std::vector<double> lr(G);
  for (size_t g = 0; g < G; g++) {
    const auto [l, f] = schedule_.group_lr(static_cast<int>(g), n, S, total);
    lr[g] = l, frozen[g] = f;
  }
  if (frozen != built_for_ || !ctx_graph_) build(frozen);
  if (!norm_) return 0.0;
  const double b1 = 0.9, b2 = 0.999;
  for (size_t g = 0; g < G; g++) {
    if (frozen[g]) continue;
    const int64_t t = ++t_[g];
    const float pars[7] = {static_cast<float>(lr[g]), static_cast<float>(b1), static_cast<float>(b2), 1e-8f,
                           static_cast<float>(schedule_.groups[g].weight_decay),
                           static_cast<float>(1.0 / (1.0 - std::pow(b1, double(t)))),
                           static_cast<float>(1.0 / (1.0 - std::pow(b2, double(t))))};
    ggml_backend_tensor_set(adam_pars_[g], pars, 0, sizeof pars);
  }
  for (size_t k = 0; k < params_.size(); k++) {
    if (!use_muon_[k] || frozen[group_[k]]) continue;
    const double l = lr[group_[k]], wd = schedule_.groups[group_[k]].weight_decay;
    const double step_lr = l * muon_lr_scale_ * rms_[k];
    // ggml_opt_step_sgd: x (1 - a w) - a o; a w = lr wd (torch Muon: decay with the lr itself)
    const float lw[2] = {static_cast<float>(step_lr), static_cast<float>(step_lr > 0 ? l * wd / step_lr : 0.0)};
    ggml_backend_tensor_set(muon_pars_[k], lw, 0, sizeof lw);
  }
  if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("lr schedule: optimizer step failed");
  float norm = 0;
  ggml_backend_tensor_get(norm_, &norm, 0, sizeof norm);
  return norm;
}

std::string GroupOptimizer::describe(int64_t n, int64_t S, int64_t total) const {
  std::string out;
  char buf[160];
  for (size_t g = 0; g < schedule_.groups.size(); g++) {
    const auto [lr, frozen] = schedule_.group_lr(static_cast<int>(g), n, S, total);
    std::snprintf(buf, sizeof buf, "%s%s %.4g%s", g ? ", " : "", schedule_.groups[g].name.c_str(), lr,
                  frozen ? " (frozen)" : "");
    out += buf;
  }
  return out;
}

std::vector<uint8_t> GroupOptimizer::get_state() const {
  // magic, group count, per-group step counts, then the state tensors (moments, momenta)
  const uint64_t magic = 0x3150524753524c4dULL; // "MLRSGRP1"
  std::vector<uint8_t> s(2 * sizeof(uint64_t) + t_.size() * sizeof(int64_t));
  const uint64_t ng = t_.size();
  std::memcpy(s.data(), &magic, 8), std::memcpy(s.data() + 8, &ng, 8);
  std::memcpy(s.data() + 16, t_.data(), t_.size() * sizeof(int64_t));
  for (ggml_tensor *t : m_) {
    const size_t at = s.size();
    s.resize(at + ggml_nbytes(t));
    ggml_backend_tensor_get(t, s.data() + at, 0, ggml_nbytes(t));
  }
  for (ggml_tensor *t : v_) {
    if (!t) continue;
    const size_t at = s.size();
    s.resize(at + ggml_nbytes(t));
    ggml_backend_tensor_get(t, s.data() + at, 0, ggml_nbytes(t));
  }
  return s;
}

void GroupOptimizer::set_state(const std::vector<uint8_t> &s) {
  uint64_t magic = 0, ng = 0;
  if (s.size() < 16) throw std::runtime_error("lr schedule state: truncated");
  std::memcpy(&magic, s.data(), 8), std::memcpy(&ng, s.data() + 8, 8);
  if (magic != 0x3150524753524c4dULL) throw std::runtime_error("lr schedule state: not a grouped optimizer state "
                                                               "(the run did not use --lr-schedule?)");
  if (ng != t_.size()) throw std::runtime_error("lr schedule state: another set of groups");
  size_t at = 16 + ng * sizeof(int64_t);
  size_t need = at;
  for (ggml_tensor *t : m_) need += ggml_nbytes(t);
  for (ggml_tensor *t : v_)
    if (t) need += ggml_nbytes(t);
  if (need != s.size()) throw std::runtime_error("lr schedule state: size mismatch (another model or schedule?)");
  std::memcpy(t_.data(), s.data() + 16, ng * sizeof(int64_t));
  for (ggml_tensor *t : m_) ggml_backend_tensor_set(t, s.data() + at, 0, ggml_nbytes(t)), at += ggml_nbytes(t);
  for (ggml_tensor *t : v_)
    if (t) ggml_backend_tensor_set(t, s.data() + at, 0, ggml_nbytes(t)), at += ggml_nbytes(t);
}

} // namespace mlipcpp::train
