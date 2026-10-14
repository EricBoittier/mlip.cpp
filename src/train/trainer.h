#pragma once

#include "runtime/graph_model.h"
#include "train/batch.h"

#include <ggml-alloc.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlipcpp::train {

/// metatrain PET defaults, with the batch size counted per GPU as in DDP.
struct TrainOptions {
  std::string data_dir;   // holds train/ val/ test/ (metatrain MemmapDataset)
  std::string init_gguf;  // PET exported with --nc-forces (forces mode)
  std::string output;     // trained GGUF; <output>.epoch<k> after each epoch
  std::string cache_dir;  // neighbour-list cache ("" = none)
  int epochs = 10;
  int batch_size = 64; // structures per GPU and step
  int n_gpus = 1;      // 0 = CPU
  double lr = 1e-4;
  double warmup_fraction = 0.01;
  double grad_clip = 1.0;
  uint64_t seed = 42;
  int n_threads = 8;    // batch preparation and neighbour lists
  int64_t max_steps = -1; // stop early (tests, profiling)
  int log_every = 100;
  // micro-batching (see make_microbatches): exact full-batch gradients
  double mb_growth = 4.0;  // max ratio of padded neighbour counts in one micro-batch
  double mb_budget = 5e7;  // max n_atoms * (max_nb+1)^2 per micro-batch
};

/// Squared errors of one batch: scaled (the loss) and physical (metrics).
struct StepStats {
  double loss = 0, loss_energy = 0, loss_force = 0;
  double se_energy_per_atom = 0; // Σ_s (ΔE_s / N_s)^2, eV^2
  double se_force = 0;           // Σ_i,k ΔF_ik^2, (eV/Å)^2
  int64_t n_systems = 0, n_force_components = 0;

  StepStats &operator+=(const StepStats &o);
};

/// One model replica on one device. Builds the loss graph of a batch,
/// optionally its backward pass, and computes it with a reused graph
/// allocator. Weights stay resident on the device.
class Worker {
public:
  Worker(const std::string &gguf, int device); // device < 0: CPU
  ~Worker();

  /// Forward, and backward if `backward`: the parameter gradients are added
  /// in place to a device-side accumulator (zero_grads / get_grads), so
  /// micro-batches accumulate without a host round trip. `energy_scale`
  /// converts scaled energy errors into physical ones for the metrics.
  StepStats run(const Batch &batch, double energy_scale, bool backward);

  void zero_grads();
  std::vector<float> get_grads() const; // flat, in parameter order
  ggml_tensor *grad_flat() const { return acc_flat_; }

  void set_params(const std::vector<float> &flat);
  std::vector<float> get_params() const;

  runtime::GraphModel &model() { return model_; }
  const std::vector<ggml_tensor *> &params() const { return params_; }
  size_t n_params() const { return n_params_; }

  /// Optional hooks for diagnostics: after the backward graph is built
  /// (before allocation) and after compute.
  std::function<void(ggml_cgraph *)> after_build, after_compute;

  /// Profiling: compute node by node (synchronised) and accumulate wall
  /// time per op into `profile_ms` ("fwd:OP" / "bwd:OP"). Slow; diagnostics only.
  bool profile = false;
  std::map<std::string, std::pair<double, int64_t>> profile_ms;

  /// Per-phase wall time of the last run(), in seconds.
  double t_build = 0, t_alloc = 0, t_upload = 0, t_compute = 0, t_download = 0;

private:
  runtime::GraphModel model_;
  std::vector<ggml_tensor *> params_;
  size_t n_params_ = 0;
  ggml_gallocr_t galloc_ = nullptr;
  // gradient accumulator: one flat F32 tensor, viewed per parameter
  ggml_context *acc_ctx_ = nullptr;
  ggml_backend_buffer_t acc_buf_ = nullptr;
  ggml_tensor *acc_flat_ = nullptr;
  std::map<ggml_tensor *, ggml_tensor *> acc_; // parameter -> its view
};

/// Data-parallel torch.optim.Adam (no weight decay) on worker 0's device:
/// sums the workers' accumulated gradients (peer copy), averages, clips to
/// the global norm and steps every parameter with ggml_opt_step_adamw in one
/// static graph, then copies the parameters to the other workers and zeroes
/// all accumulators.
class Optimizer {
public:
  Optimizer(std::vector<std::unique_ptr<Worker>> &workers, double clip);
  ~Optimizer();
  /// One step at learning rate `lr`; returns the global gradient norm (of
  /// the averaged gradient, before clipping).
  double step(double lr);

private:
  std::vector<std::unique_ptr<Worker>> &workers_;
  ggml_context *ctx_ = nullptr, *gctx_ = nullptr;
  ggml_backend_buffer_t buf_ = nullptr;
  ggml_gallocr_t galloc_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  std::vector<ggml_tensor *> staging_; // other workers' gradients, on device 0
  ggml_tensor *pars_ = nullptr, *norm_ = nullptr;
  int64_t t_ = 0;
};

/// Neighbour-list cache file of `split` under `cache_dir` ("" if no dir).
std::string cache_path(const std::string &cache_dir, const MemmapSplit &split, float cutoff);

void train(const TrainOptions &opts);

} // namespace mlipcpp::train
