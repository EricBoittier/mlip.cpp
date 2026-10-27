#pragma once
// Per-group learning-rate schedules (mlipcpp-train / mlipcpp-train-plato --lr-schedule FILE),
// the semantics of metatrain's src/metatrain/pet/modules/lr_schedule.py (EricBoittier/metatrain
// feat/pet-plato-lr; golden values tests/resources/lr_schedule_golden.json):
//
//   { "base_lr": 1e-4, "min_lr": 1e-6, "shape": "cosine" | "linear" | "constant",
//     "warmup_epochs": 1, "decay_after_epochs": 10,
//     "groups": [ {"name": "new", "match": ["*_gate", "les."], "lr_mult": 1.0, "freeze_epochs": 0,
//                  "warmup_epochs": 2, "weight_decay": 0.0, "optimizer": "adam" | "muon"}, ... ],
//     "default_group": "backbone" }
//
// S steps per epoch, T steps in the run, step n from 0 at the start of this run (a fine-tune
// from --init starts at 0; --resume continues the count):
//   base b(n) = base_lr min(1, (n + 1) / W) for n < W = round(S warmup_epochs); base_lr until
//   D = max(W, round(S decay_after_epochs)); then the shape from base_lr at D to min_lr at T - 1,
//   p = min(1, (n - D) / max(1, T - 1 - D)): cosine min + (base - min)(1 + cos(pi p)) / 2,
//   linear base - (base - min) p, constant base.
//   group g: frozen while n < F = round(S freeze_epochs) (no update, no optimizer state, its
//   gradient left out of the clipping norm); then lr = b(n) lr_mult min(1, (n - F + 1) / W_g),
//   W_g = round(S warmup_epochs_g) (0: ramp 1).
//   matching on metatrain state-dict names: a pattern with '*' is an fnmatch glob on the whole
//   name, any other pattern a substring; first matching group wins; unmatched names go to
//   default_group (a group of the list, or a new one with the defaults).
//   group defaults: lr_mult 1, freeze_epochs 0, warmup_epochs 0, weight_decay 0, optimizer adam;
//   "muon": Muon for the trainer's Muon matrices, AdamW for the group's other tensors.
// round() is Python's (half to even), as the reference.

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_gallocr;

namespace mlipcpp::train {

struct LrGroup {
  std::string name;
  std::vector<std::string> match;
  double lr_mult = 1.0, freeze_epochs = 0.0, warmup_epochs = 0.0, weight_decay = 0.0;
  bool muon = false;
};

struct LrSchedule {
  double base = 1e-4, min = 0.0, warmup_epochs = 0.0, decay_after_epochs = 0.0;
  std::string shape = "cosine";
  std::vector<LrGroup> groups; // the default group included
  std::string default_group = "default";

  static LrSchedule load(const std::string &path); // JSON file
  static LrSchedule parse(const std::string &json);
  static bool matches(const std::string &name, const std::string &pattern);
  int group_of(const std::string &state_dict_name) const; // index into groups
  int group_index(const std::string &name) const;         // -1 if none
  double base_lr(int64_t step, int64_t steps_per_epoch, int64_t total) const;
  std::pair<double, bool> group_lr(int g, int64_t step, int64_t steps_per_epoch, int64_t total) const;
  /// {"lr": {group: [...]}, "frozen": {group: [...]}} for steps [0, n) (the golden file's tables)
  std::string table_json(int64_t n, int64_t steps_per_epoch, int64_t total) const;
};

/// Adam(W) / Muon per parameter group with the group's lr, weight decay and its own step
/// count (bias corrections: a group unfrozen at step F starts its Adam at t = 1, as torch's
/// per-parameter state); global-norm clipping over the non-frozen gradients; frozen groups
/// get no update and keep no state. One device: the parameters, their gradients and the
/// state on `backend`. The step graph is rebuilt when the set of frozen groups changes.
class GroupOptimizer {
public:
  struct Param {
    ggml_tensor *param = nullptr, *grad = nullptr; // grad: same shape, on the backend (a view is fine)
    std::string name;                              // state-dict name (matching)
    bool muon_matrix = false;                      // a Muon matrix (stack) of the trainer
  };
  /// orthogonalize: the trainer's Newton-Schulz of a stack [ne0, ne1, k] (Muon groups)
  using Orthogonalize = std::function<ggml_tensor *(ggml_context *, ggml_tensor *)>;
  GroupOptimizer(ggml_backend *backend, std::vector<Param> params, LrSchedule schedule, double clip,
                 double muon_momentum = 0.95, double muon_lr_scale = 1.0, Orthogonalize orthogonalize = {});
  ~GroupOptimizer();
  GroupOptimizer(const GroupOptimizer &) = delete;
  GroupOptimizer &operator=(const GroupOptimizer &) = delete;

  /// step n (0-based) of a run of `total` steps, `steps_per_epoch` per epoch; returns the
  /// gradient norm before clipping (of the non-frozen groups)
  double step(int64_t n, int64_t steps_per_epoch, int64_t total);
  const LrSchedule &schedule() const { return schedule_; }
  const std::vector<int> &group_of_param() const { return group_; }
  std::vector<size_t> group_sizes() const; // values per group
  /// "name: lr (frozen)" of every group at step n
  std::string describe(int64_t n, int64_t steps_per_epoch, int64_t total) const;
  std::vector<uint8_t> get_state() const;
  void set_state(const std::vector<uint8_t> &s);

private:
  ggml_backend *backend_;
  std::vector<Param> params_;
  LrSchedule schedule_;
  double clip_, mu_, muon_lr_scale_;
  Orthogonalize orthogonalize_;
  std::vector<int> group_;            // per parameter
  std::vector<bool> use_muon_;        // per parameter
  std::vector<int64_t> t_;            // per group: steps taken
  std::vector<float> rms_;            // per parameter (Muon): 0.2 sqrt(max dim)
  ggml_context *ctx_state_ = nullptr; // m, v (Adam) / momentum (Muon) per parameter; per-group pars
  ggml_backend_buffer *buf_state_ = nullptr;
  std::vector<ggml_tensor *> m_, v_, muon_pars_; // muon_pars_: per parameter (lr, wd)
  std::vector<ggml_tensor *> adam_pars_;         // per group [7]
  ggml_tensor *clip_t_ = nullptr;
  std::vector<bool> built_for_; // frozen flags the graph was built for
  ggml_context *ctx_graph_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  ggml_gallocr *galloc_ = nullptr;
  ggml_tensor *norm_ = nullptr;
  void build(const std::vector<bool> &frozen);
};

} // namespace mlipcpp::train
