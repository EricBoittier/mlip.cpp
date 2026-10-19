// Train PET (energy + direct non-conservative forces, or with --conservative
// -dE/dx as well) on a metatrain MemmapDataset, data-parallel over GPUs. See
// src/train/trainer.h.
#include "train/crash.h"
#include "train/trainer.h"

#include <fmt/core.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  mlipcpp::train::install_crash_backtrace();
  mlipcpp::train::TrainOptions o;
  auto usage = [&] {
    fmt::print(stderr,
               "usage: {} --data DIR --init INIT.gguf -o OUT.gguf [--epochs {}] [--batch-size {}]\n"
               "          [--gpus {}] [--lr {}] [--warmup-fraction {}] [--grad-clip {}] [--seed {}]\n"
               "          [--threads {}] [--cache-dir DIR] [--max-steps N] [--log-every {}]\n"
               "          [--mb-growth {}] [--mb-budget {}] [--max-neighbors N] [--max-attention A] [--tile-nb-ratio R] [--tile-atom-ratio {}]\n"
               "          [--graph-cache {}] [--prefetch {}] [--pack step|worker] [--optimizer adam|muon] [--muon-momentum {}]\n"
               "          [--muon-lr-scale {}] [--muon-ns-steps {}]\n"
               "          [--dump-errors PREFIX] [--log-errors FILE] [--log-errors-epochs {}] [--exclude DIR]\n"
               "          [--conservative] [--energy-weight {}] [--force-weight {}] [--nc-force-weight {}] [--model-baselines]\n"
               "--dump-errors writes PREFIX.{{val,test}}.bin, --log-errors the training structures' errors\n"
               "before each update (24-byte records, see SystemError); --exclude DIR drops the split\n"
               "indices in DIR/{{train,val,test}}.npy;\n"
               "--tile-nb-ratio > 1 pads micro-batches to a geometric grid of shapes and\n"
               "replays cached graphs (CUDA graphs);\n"
               "--gpus 0 trains on the CPU; --batch-size counts structures per GPU;\n"
               "--conservative puts the force loss on -dE/dx (forward-mode tangents, about 3x the\n"
               "step), the direct head keeping a term of its own (weights: --*-weight);\n"
               "--model-baselines (fine-tuning) keeps --init's composition and scales;\n"
               "--epochs 0 only evaluates --init on val/ and test/ (-o is ignored).\n",
               argv[0], o.epochs, o.batch_size, o.n_gpus, o.lr, o.warmup_fraction,
               o.grad_clip, o.seed, o.n_threads, o.log_every, o.mb_growth, o.mb_budget,
               o.tile_atom_ratio, o.graph_cache, o.prefetch, o.muon_momentum, o.muon_lr_scale, o.muon_ns_steps,
               o.log_errors_epochs, o.energy_weight, o.force_weight, o.nc_force_weight);
    std::exit(2);
  };
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : (usage(), ""); };
    if (a == "--data") o.data_dir = val();
    else if (a == "--init") o.init_gguf = val();
    else if (a == "-o" || a == "--output") o.output = val();
    else if (a == "--cache-dir") o.cache_dir = val();
    else if (a == "--epochs") o.epochs = std::stoi(val());
    else if (a == "--batch-size") o.batch_size = std::stoi(val());
    else if (a == "--gpus") o.n_gpus = std::stoi(val());
    else if (a == "--lr") o.lr = std::stod(val());
    else if (a == "--warmup-fraction") o.warmup_fraction = std::stod(val());
    else if (a == "--grad-clip") o.grad_clip = std::stod(val());
    else if (a == "--seed") o.seed = std::stoull(val());
    else if (a == "--threads") o.n_threads = std::stoi(val());
    else if (a == "--max-steps") o.max_steps = std::stoll(val());
    else if (a == "--log-every") o.log_every = std::stoi(val());
    else if (a == "--mb-growth") o.mb_growth = std::stod(val());
    else if (a == "--mb-budget") o.mb_budget = std::stod(val());
    else if (a == "--max-neighbors") o.max_neighbors = std::stoi(val());
    else if (a == "--max-attention") o.max_attention = std::stod(val());
    else if (a == "--tile-nb-ratio") o.tile_nb_ratio = std::stod(val());
    else if (a == "--tile-atom-ratio") o.tile_atom_ratio = std::stod(val());
    else if (a == "--pack") o.pack = val();
    else if (a == "--prefetch") o.prefetch = std::stoi(val());
    else if (a == "--graph-cache") o.graph_cache = std::stoi(val());
    else if (a == "--optimizer") o.optimizer = val();
    else if (a == "--muon-momentum") o.muon_momentum = std::stod(val());
    else if (a == "--muon-lr-scale") o.muon_lr_scale = std::stod(val());
    else if (a == "--muon-ns-steps") o.muon_ns_steps = std::stoi(val());
    else if (a == "--dump-errors") o.dump_errors = val();
    else if (a == "--log-errors") o.log_errors = val();
    else if (a == "--log-errors-epochs") o.log_errors_epochs = std::stoi(val());
    else if (a == "--exclude") o.exclude = val();
    else if (a == "--conservative") o.conservative = true;
    else if (a == "--energy-weight") o.energy_weight = std::stod(val());
    else if (a == "--force-weight") o.force_weight = std::stod(val());
    else if (a == "--nc-force-weight") o.nc_force_weight = std::stod(val());
    else if (a == "--model-baselines") o.model_baselines = true;
    else usage();
  }
  if (o.data_dir.empty() || o.init_gguf.empty() || o.output.empty()) usage();
  try {
    mlipcpp::train::train(o);
  } catch (const std::exception &e) {
    fmt::print(stderr, "error: {}\n", e.what());
    return 1;
  }
}
