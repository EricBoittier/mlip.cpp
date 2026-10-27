#pragma once

#include "runtime/graph_model.h"
#include "train/batch.h"

#include <ggml-alloc.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
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
  // "cosine" (metatrain: warmup_fraction, then cosine to 0) or "linear"
  // (LOREM: constant for decay_after epochs, then linear to min_lr at the end)
  std::string schedule = "cosine";
  int decay_after = 10;
  double min_lr = 1e-6;
  double grad_clip = 1.0;
  uint64_t seed = 42;
  int n_threads = 8;    // batch preparation and neighbour lists
  int64_t max_steps = -1; // stop early (tests, profiling)
  int log_every = 100;
  // micro-batching (see make_microbatches): exact full-batch gradients
  double mb_growth = 4.0;  // max ratio of padded neighbour counts in one micro-batch
  double mb_budget = 5e7;  // max n_atoms * (max_nb+1)^2 per micro-batch
  int max_neighbors = 0;   // drop structures with an atom above this many neighbours (0 = keep all)
  double max_attention = 0; // drop structures with n_atoms * (max_nb+1)^2 above this (0 = keep all)
  // "step": all workers' structures packed together by a fitted cost model
  // (make_step_batches; loss = full-step mean); "worker": each worker packs
  // its own batch (make_microbatches; DDP mean of per-worker means)
  std::string pack = "step";
  // fixed-shape tiles + cached graphs (see Tiling); 0 = dynamic shapes
  double tile_nb_ratio = 0, tile_atom_ratio = 1.25;
  int prefetch = 4;       // steps packed ahead of the GPUs
  int graph_cache = 512; // cached graph shapes per device (they share one compute buffer)
  // optimizer: "adam" (torch.optim.Adam) or "muon" (2-D hidden weights; Adam for the rest)
  std::string optimizer = "adam";
  double muon_momentum = 0.95;
  double muon_lr_scale = 1.0; // Muon lr = lr * this (updates are RMS-matched to Adam)
  int muon_ns_steps = 5;
  // per-structure errors (SystemError records, see write_errors): of val/ and
  // test/ at the end ("" = off), and online during the first
  // log_errors_epochs epochs of training, before each update (prequential)
  std::string dump_errors, log_errors;
  int log_errors_epochs = 1;
  // drop structures: DIR/{train,val,test}.npy hold split indices ("" = none)
  std::string exclude;
  // conservative forces -dE/dx (Worker::Conservative): the force loss is on
  // them, the direct head keeps a term of its own; loss weights of the
  // energy, the conservative and the direct forces
  bool conservative = false;
  double energy_weight = 1.0, force_weight = 1.0, nc_force_weight = 1.0;
  // fine-tuning: keep the composition and scales stored in --init instead of
  // fitting them on the training set
  bool model_baselines = false;
  // checkpoints: save_every > 0 writes <output>.epoch%05d.gguf every that many
  // epochs (numbered by epochs done) instead of <output>.epoch<k> after every
  // epoch; <output>.best.gguf is the epoch with the highest R^2(E) + R^2(F) on
  // val/ (LOREM's criterion); log_jsonl: one line of val metrics per epoch
  int save_every = 0;
  std::string log_jsonl;
  // chains of jobs: <output>.state (optimizer state, step, epoch, best score)
  // with the model in <output>, every state_every epochs and when stopping
  // after the epoch in which max_hours passed ("paused at epoch k"); resume:
  // --init <output> --resume <output>.state
  int state_every = 0;
  double max_hours = 0;
  std::string resume;
  // several GPUs: average the gradients with one NCCL all-reduce and step a
  // replica of the optimizer on every device (no gradient staging on device
  // 0, no weight broadcast); default with a build that has NCCL
#ifdef MLIPCPP_NCCL
  bool allreduce = true;
#else
  bool allreduce = false;
#endif
};

/// One structure's errors: energy per atom (prediction - target, eV/atom) and
/// the summed squared force error over its atoms ((eV/A)^2). 24-byte records;
/// numpy dtype [('id','<i8'),('step','<i4'),('n','<i4'),('de','<f4'),('fse','<f4')]
/// (step: the training step, -1 for an evaluation). --dump-errors also writes
/// PREFIX.<split>.atoms.bin: float32 |dF|^2 of every atom, record by record,
/// each structure's atoms in its memmap order.
struct SystemError {
  int64_t id;
  int32_t step, n;
  float de, fse;
};
static_assert(sizeof(SystemError) == 24);

/// Squared errors of one batch: scaled (the loss) and physical (metrics).
struct StepStats {
  double loss = 0, loss_energy = 0, loss_force = 0;
  double loss_nc = 0, se_force_nc = 0; // conservative mode: the direct head's term and (eV/Å)^2
  double se_energy_per_atom = 0; // Σ_s (ΔE_s / N_s)^2, eV^2
  double se_force = 0;           // Σ_i,k ΔF_ik^2, (eV/Å)^2 (conservative mode: of -dE/dx)
  // MAE and R^2 = 1 - SSE / SST (energies per atom after the composition,
  // force components pooled): Σ|ΔE_s / N_s|, Σ|ΔF_ik| (conservative mode
  // only), and the sums of the targets and of their squares
  double ae_energy_per_atom = 0, ae_force = 0;
  double sum_e = 0, sum2_e = 0, sum_f = 0, sum2_f = 0;
  int64_t n_systems = 0, n_force_components = 0;

  StepStats &operator+=(const StepStats &o);
};

/// One model replica on one device. Builds the loss graph of a batch,
/// optionally its backward pass, and computes it with a reused graph
/// allocator. Weights stay resident on the device.
class Worker {
public:
  /// `cache_graphs` > 0 keeps the graph of up to that many batch shapes for
  /// replay (CUDA graphs), least recently used first out. All cached graphs
  /// share one compute buffer (they run one at a time); each keeps its
  /// inputs and constants in a small buffer of its own.
  Worker(const std::string &gguf, int device, size_t cache_graphs = 0); // device < 0: CPU
  ~Worker();

  /// Forward, and backward if `backward`: the parameter gradients are added
  /// in place to a device-side accumulator (zero_grads / get_grads), so
  /// micro-batches accumulate without a host round trip. `energy_scale`
  /// converts scaled energy errors into physical ones for the metrics.
  StepStats run(const Batch &batch, double energy_scale, bool backward);

  void zero_grads();
  std::vector<float> get_grads() const; // flat, in parameter order
  ggml_tensor *grad_flat() const { return acc_flat_; }
  /// offset (in floats) of parameter k's gradient in grad_flat: Muon
  /// matrices of one shape are adjacent, so not parameter order
  size_t grad_offset(size_t k) const { return grad_off_[k]; }

  void set_params(const std::vector<float> &flat);
  /// The whole weight buffer as one byte tensor (for one-copy broadcasts).
  ggml_tensor *weights_blob() const { return weights_blob_; }
  std::vector<float> get_params() const;

  runtime::GraphModel &model() { return model_; }
  const std::vector<ggml_tensor *> &params() const { return params_; }
  size_t n_params() const { return n_params_; }

  /// Optional hooks for diagnostics: after the backward graph is built
  /// (before allocation) and after compute.
  std::function<void(ggml_cgraph *)> after_build, after_compute;

  /// Conservative forces F = -dE/dx. run() then computes two graphs per
  /// batch: the forward pass and its backward pass to the edge vectors and
  /// the cutoff values, for the forces (on the host: edge gradients to
  /// atoms, plus the adaptive radii's chain through Batch::taper_*); then the
  /// loss with the forward-mode tangent of E along the force residual
  /// (runtime/graph_jvp.h), whose weight gradient is the force loss's, and
  /// its backward pass to the weights (accumulated as in the direct mode).
  /// Loss = energy * L_E + force * L_F(-dE/dx) + nc_force * L_F(direct head).
  struct Conservative {
    bool on = false;
    double energy = 1.0, force = 1.0, nc_force = 1.0;
  } conservative;

  /// Profiling: compute node by node (synchronised) and accumulate, per op
  /// ("fwd:OP" / "bwd:OP"; conservative mode "F.fwd:OP" / "F.bwd:OP" for the
  /// FORCES graph and "M.fwd" / "M.tan" / "M.bwd" for MIXED's primal forward,
  /// tangent and backward), wall time, FLOPs and ideal bytes (every operand
  /// read and the result written once). Slow; diagnostics only.
  struct OpStats {
    double ms = 0, flops = 0, bytes = 0;
    int64_t count = 0;
  };
  bool profile = false;
  /// when set, run() appends one record per real structure (step = -1)
  std::vector<SystemError> *system_errors = nullptr;
  /// when set too, run() appends each real atom's |dF|^2 ((eV/A)^2), in the order of the records
  std::vector<float> *atom_errors = nullptr;
  std::map<std::string, OpStats> profile_ops;
  std::map<std::string, OpStats> profile_copies; // CONT / CPY nodes by the view chain they copy
  std::map<std::string, OpStats> profile_patterns; // "phase:OP(src0 op, src1 op)", fusion candidates
  std::vector<std::pair<double, std::string>> profile_nodes; // (ms, "fwd:OP name [shape]") per node

  /// FLOPs, ideal bytes and compute-buffer size of one batch shape's graph:
  /// built and measured, never allocated or computed.
  struct GraphCost {
    double flops = 0, bytes = 0;
    size_t buffer = 0;
  };
  GraphCost graph_cost(int n, int m, int S, bool backward = true);

  /// Per-phase wall time of the last run(), in seconds; t_cpu: conservative
  /// mode's host work (forces from the edge gradients, loss, tangents).
  double t_build = 0, t_alloc = 0, t_upload = 0, t_compute = 0, t_download = 0, t_cpu = 0;
  int64_t cache_hits = 0, cache_misses = 0, cache_flushes = 0;
  size_t compute_buffer_bytes() const;
  size_t cache_size() const { return cache_.size(); }

private:
  /// The loss graph of one batch shape, with its inputs and outputs.
  struct StepGraph {
    ggml_context *ctx = nullptr;
    ggml_cgraph *gf = nullptr;
    std::vector<ggml_tensor *> inputs; // in upload order (see upload_inputs)
    // (loss, loss_e, loss_f, physical force squared error): one download
    ggml_tensor *stats = nullptr, *loss_grad = nullptr;
    ggml_tensor *per_system = nullptr; // [S, 2]: d_energy (scaled, per atom), mean |dF|^2 per atom (physical)
    ggml_tensor *per_atom = nullptr;   // [1, n]: |dF|^2 per atom (physical)
    ggml_tensor *out = nullptr, *grad_v = nullptr, *grad_c = nullptr; // FORCES: (e | F_nc) per atom, dE/dv, dE/dc
    int n_forward = 0;
    // profile labels: the graph kind, and in MIXED the first tensor made
    // after the primal forward (nodes past it in the forward part are the tangent)
    int mode = 0;
    const ggml_tensor *jvp_mark = nullptr;
    std::vector<std::pair<ggml_tensor *, float>> constants;
    // cached graphs: inputs packed at the start of `persist` (one upload
    // through `blob`), then the constants (filled once)
    ggml_backend_buffer_t persist = nullptr;
    ggml_tensor *blob = nullptr;
    size_t blob_bytes = 0;
    uint64_t last_use = 0;
    ~StepGraph();
  };
  /// graph kinds: the loss (FORWARD / BACKWARD to the weights); conservative
  /// mode's FORCES (backward to the edge vectors and cutoff values) and MIXED
  /// (the loss with the tangent term, backward to the weights)
  enum Mode { FORWARD, BACKWARD, FORCES, MIXED };
  std::unique_ptr<StepGraph> build_graph(int n, int m, int S, Mode mode, bool cached);
  StepGraph &graph(int n, int m, int S, Mode mode, std::unique_ptr<StepGraph> &once);
  StepGraph &cached_graph(int n, int m, int S, Mode mode);
  void upload_inputs(StepGraph &g, const Batch &b, const std::vector<float> *dv = nullptr,
                     const std::vector<float> *dc = nullptr);
  void compute(StepGraph &g);
  StepStats run_conservative(const Batch &b, double energy_scale, bool backward);

  std::map<std::tuple<int, int, int, int>, std::unique_ptr<StepGraph>> cache_;
  size_t cache_limit_ = 0;
  uint64_t clock_ = 0;
  size_t ctx_bytes_[4] = {0, 0, 0, 0};  // graph context size per Mode; measured on the first build
  std::vector<uint8_t> staging_;   // host side of the input blob
  std::vector<int32_t> cond_rows_[2]; // charge / spin table rows of the batch's atoms (conditioned models)

  runtime::GraphModel model_;
  std::vector<ggml_tensor *> params_;
  ggml_tensor *weights_blob_ = nullptr;
  size_t n_params_ = 0;
  ggml_gallocr_t galloc_ = nullptr;
  // gradient accumulator: one flat F32 tensor, viewed per parameter
  ggml_context *acc_ctx_ = nullptr;
  ggml_backend_buffer_t acc_buf_ = nullptr;
  ggml_tensor *acc_flat_ = nullptr;
  std::vector<size_t> grad_off_; // per parameter, in grad_flat
  std::map<ggml_tensor *, ggml_tensor *> acc_; // parameter -> its view
};

/// Data-parallel torch.optim.Adam (no weight decay) on worker 0's device:
/// sums the workers' accumulated gradients (peer copy), averages, clips to
/// the global norm and steps every parameter with ggml_opt_step_adamw in one
/// static graph, then copies the parameters to the other workers and zeroes
/// all accumulators.
class Optimizer {
public:
  /// `muon`: 2-D hidden weights (both dims >= 16, not embeddings) take Muon
  /// steps -- momentum, Newton-Schulz orthogonalisation, update RMS matched
  /// to Adam (0.2 sqrt(max dim)) -- at lr * muon_lr_scale; the rest Adam.
  /// `allreduce` (a build with NCCL, several workers): one replica of this
  /// optimizer per device, each stepping its worker's parameters after an
  /// NCCL all-reduce (average) of the gradients in place; every
  /// `resync_every` steps device 0's weights are copied to the others anyway.
  Optimizer(std::vector<std::unique_ptr<Worker>> &workers, double clip, bool muon = false,
            double muon_momentum = 0.95, double muon_lr_scale = 1.0, int ns_steps = 5, bool allreduce = false);
  Optimizer(const std::vector<Worker *> &workers, double clip, bool muon, double muon_momentum,
            double muon_lr_scale, int ns_steps, bool allreduce);
  ~Optimizer();
  /// Copy worker w's (w > 0) gradients to device 0 now (and, with the Muon
  /// split, w's gradients over each other device's range there), from w's
  /// thread as soon as it is done, so the copy overlaps the other workers' compute.
  void stage(size_t w);
  /// One step at learning rate `lr` (gradients not staged yet are copied
  /// first); returns the global gradient norm (of the averaged gradient,
  /// before clipping). Updated weights go to the other workers as one copy
  /// of the weight buffer when the layouts match.
  double step(double lr);
  /// Profiling only (steps the parameters): mean wall time of `reps` whole
  /// steps, then the step graph node by node (synchronised), by op.
  double profile(int reps, std::map<std::string, Worker::OpStats> &ops);
  /// The optimizer state for resuming (step count, Adam moments, Muon
  /// momenta: device 0's buffer; not with the Muon split over devices).
  std::vector<uint8_t> get_state() const;
  void set_state(const std::vector<uint8_t> &state);
  std::vector<std::string> muon_groups; // "ne0 x ne1 x count" per orthogonalised group
  std::string muon_split;               // Newton-Schulz FLOPs per device ("" if not split)
  std::string allreduce_info;           // "" without the all-reduce
  int64_t resync_every = 1000;
  double last_drift = 0;                // max |w_d - w_0| over the replicas at the last resync

private:
  std::vector<Worker *> workers_;
  std::vector<std::unique_ptr<Optimizer>> replicas_; // all-reduce mode: one per device
  struct Nccl;
  std::unique_ptr<Nccl> nccl_;
  double step_allreduce(double lr);
  ggml_context *ctx_ = nullptr, *gctx_ = nullptr;
  ggml_backend_buffer_t buf_ = nullptr;
  ggml_gallocr_t galloc_ = nullptr;
  ggml_cgraph *graph_ = nullptr;
  std::vector<ggml_tensor *> staging_; // other workers' gradients, on device 0
  ggml_tensor *pars_ = nullptr, *norm_ = nullptr, *muon_lr_ = nullptr;
  std::vector<char> staged_;   // per worker: gradients already on device 0
  bool blob_broadcast_ = false; // weight buffers laid out identically
  double muon_lr_scale_ = 1.0;
  ggml_tensor *muon_pars_ = nullptr; // per Muon group (lr, wd)
  std::vector<float> muon_rms_;      // per Muon group, the update RMS factor
  int64_t t_ = 0;
  // Muon split over the devices: device d > 0 owns the Muon matrices in the
  // grad_flat range [lo, hi) (contiguous, balanced by Newton-Schulz FLOPs):
  // it receives the other workers' gradients over that range, takes the
  // momentum / Newton-Schulz steps there and sends the orthogonalised update
  // back to device 0, which applies it. Device 0 keeps Adam, the global norm
  // and its own range. MLIPCPP_MUON_ONE_DEVICE=1: everything on device 0.
  struct Shard {
    size_t lo = 0, hi = 0;
    ggml_context *ctx = nullptr, *gctx = nullptr, *vctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph *graph = nullptr;
    std::vector<ggml_tensor *> staging, src; // per other worker: its gradients here / its grad_flat over [lo, hi)
    ggml_tensor *clip = nullptr, *upd = nullptr; // on device d
    ggml_tensor *upd_in = nullptr;               // on device 0
  };
  std::vector<Shard> shards_; // devices 1..W-1
  ggml_cgraph *norm_graph_ = nullptr, *apply_graph_ = nullptr;
  ggml_gallocr_t norm_galloc_ = nullptr, apply_galloc_ = nullptr;
  ggml_tensor *clip_ = nullptr, *gbuf_ = nullptr; // device 0: clip factor, clipped mean gradient

public:
  bool graph_captured_ = false; // the step graph runs as a CUDA graph

public:
  size_t n_muon_tensors = 0;
};

/// Neighbour-list cache file of `split` under `cache_dir` ("" if no dir).
std::string cache_path(const std::string &cache_dir, const MemmapSplit &split, float cutoff,
                       const AdaptiveCutoff &adaptive = {});
/// The model's adaptive-cutoff settings (training implements the "solver" method only).
AdaptiveCutoff adaptive_cutoff(const runtime::GraphModel &model);

void train(const TrainOptions &opts);

} // namespace mlipcpp::train
