// Train PET (energy + direct non-conservative forces, or with --conservative
// -dE/dx as well) on a metatrain MemmapDataset, data-parallel over GPUs. See
// src/train/trainer.h.
#include "train/crash.h"
#include "train/trainer.h"
#include "train/lr_schedule.h"

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
               "          [--gpus {}] [--lr {}] [--warmup-fraction {}] [--schedule cosine|linear] [--decay-after {}]\n"
               "          [--min-lr {}] [--grad-clip {}] [--seed {}]\n"
               "          [--threads {}] [--cache-dir DIR] [--max-steps N] [--log-every {}]\n"
               "          [--mb-growth {}] [--mb-budget {}] [--max-neighbors N] [--max-attention A] [--tile-nb-ratio R] [--tile-atom-ratio {}]\n"
               "          [--graph-cache {}] [--prefetch {}] [--pack step|worker] [--optimizer adam|muon] [--muon-momentum {}]\n"
               "          [--muon-lr-scale {}] [--muon-ns-steps {}]\n"
               "          [--dump-errors PREFIX] [--log-errors FILE] [--log-errors-epochs {}] [--exclude DIR]\n"
               "          [--conservative] [--energy-weight {}] [--force-weight {}] [--nc-force-weight {}] [--model-baselines]\n"
               "          [--allreduce {}] [--save-every N] [--log JSONL] [--state-every N] [--max-hours H] [--resume STATE]\n"
               "          [--lr-schedule FILE.json] [--dump-lr-schedule N [--dump-steps-per-epoch S]]\n"
               "          [--charge-corr-weight W] [--charge-corr-target mulliken|lowdin|nbo] [--charge-corr-min R] [--frag-charge-weight W] [--frag-charges [--frag-cond-gate]] [--sqe [--sqe-q0-weight W]]\n"
               "          [--loss mse|huber] [--huber-delta-e D] [--huber-delta-f D]\n"
               "          [--force-scale-global] [--energy-sw FILE.f32] [--lr-energy-grad exact|legacy]\n"
               "--force-scale-global: one force-loss scale (RMS of all training force components) for every species;\n"
               "--energy-sw FILE: per-structure energy-loss weights w (raw float32, one per training-split row), training\n"
               "steps only, as a residual scale sqrt(w) (MSE: w d^2);\n"
               "--lr-energy-grad exact (default) | legacy: the host-side energies (lrc / SQE) in the energy-loss gradient of\n"
               "the conservative step (legacy: backbone residual only, as before; the reported loss always includes them);\n"
               "--loss huber: torch.nn.HuberLoss per component (metatrain's huber; 0.5 d^2 below delta, then linear), deltas in\n"
               "the loss's units: energy per atom over the energy scale (default 0.01), forces over their scale (0.02);\n"
               "--charge-corr-weight W (Plato-PET-LR with LES; default 0: off): an auxiliary loss on the latent charges,\n"
               "W x mean over structures of max(0, R - r), r the per-structure Pearson correlation of the first latent\n"
               "charge channel with the DFT partial charges <split>/<target>_charge.bin (float32 per atom; --charge-corr-target,\n"
               "default mulliken). A hinge on r: it rewards correlation, the magnitudes stay free; R (--charge-corr-min,\n"
               "default 1) is the correlation above which a structure adds nothing, so R = 1 gives W (1 - r). Structures\n"
               "without the file, with < 3 atoms, a constant reference or any NaN (nbo) are left out (counted as skipped);\n"
               "--frag-charge-weight W (Plato-PET-LR with LES; default 0: off): W x mean over fragments of the squared error of\n"
               "the fragment's summed latent charge (e) against its summed --charge-corr-target charges, structures with two or\n"
               "more bonded fragments only (<split>/fragment.bin, int32 per atom: scripts/plato_lr/fragments.py);\n"
               "--frag-charges (Plato-PET-LR with conditioning; on by itself for a pet.plato_lr.frag_charges model):\n"
               "fragment-constrained charges from <split>/fragunit.bin, fragq.bin, frag_ok.bin: every atom conditioned on its\n"
               "bonded unit's charge and the beyond-R Coulomb term of the unit charges added (taken out of the targets);\n"
               "frag_ok 0: whole-structure conditioning, no term; the saved GGUFs get pet.plato_lr.frag_*;\n"
               "--sqe (a pet.plato_lr.sqe model, keys from scripts/plato_lr/add_sqe_keys.py; conservative forces): train the\n"
               "bounded split-charge equilibration parameters (host float64, Hellmann-Feynman forces, implicit gradients) with\n"
               "a host Adam at the lr of the --lr-schedule group matching 'sqe.'; without --sqe a sqe model keeps them fixed;\n"
               "--sqe-q0-weight W (sqe_q0 learned): supervise the learned reference charges by the unit labels (fragunit.bin,\n"
               "fragq.bin, frag_ok.bin), W x mean over units of (sum q0 - Q_u)^2 per structure; training only;\n"
               "--frag-cond-gate: keep the structure-charge conditioning c and add g (c_unit - c), g the zero-initialised\n"
               "per-channel tensor plato_frag_cond_gate (scripts/plato_lr/add_frag_keys.py --cond-gate);\n"
               "--tail-data DIR --tail-weight W [--tail-floor-mev 2.585 --tail-delta 1] (conservative): the physics-\n"
               "  normalised tail loss on pulled-apart scans (train/tail_loss.h; DIR a split + tail_pairs.tsv), one scan per\n"
               "  step, classes in turn: W x mean_p Huber((dE_model - dE_DFT) / max(|dE_ref|, floor)), no absolute energies;\n"
               "--checkpoint-steps N: also OUT.step%06d.gguf every N steps;\n"
               "--lr-schedule: per-group learning rates (src/train/lr_schedule.h), replaces --lr, --warmup-fraction,\n"
               "--schedule, --decay-after, --min-lr and --optimizer; groups matched on metatrain state-dict names;\n"
               "--dump-lr-schedule N prints the groups' lr of the first N steps as JSON and stops (with\n"
               "--dump-steps-per-epoch S: without data, S steps per epoch, --epochs epochs);\n"
               "--dump-errors writes PREFIX.{{val,test}}.bin, --log-errors the training structures' errors\n"
               "before each update (24-byte records, see SystemError); --exclude DIR drops the split\n"
               "indices in DIR/{{train,val,test}}.npy;\n"
               "--tile-nb-ratio > 1 pads micro-batches to a geometric grid of shapes and\n"
               "replays cached graphs (CUDA graphs);\n"
               "--gpus 0 trains on the CPU; --batch-size counts structures per GPU;\n"
               "--conservative puts the force loss on -dE/dx (forward-mode tangents, about 3x the\n"
               "step), the direct head keeping a term of its own (weights: --*-weight);\n"
               "--model-baselines (fine-tuning) keeps --init's composition and scales;\n"
               "--allreduce 1 (NCCL builds, the default there) averages the gradients with one\n"
               "all-reduce and steps a replica of the optimizer on every GPU;\n"
               "--schedule linear: lr constant for --decay-after epochs, then linear to --min-lr (LOREM);\n"
               "OUT.best.gguf: the epoch with the highest R^2(E) + R^2(F) on val/; --log: val metrics per epoch;\n"
               "--save-every N: OUT.epoch%05d.gguf every N epochs (default: OUT.epoch<k> after each);\n"
               "--state-every N / --max-hours H: OUT and OUT.state every N epochs and when pausing after H hours;\n"
               "continue with --init OUT --resume OUT.state;\n"
               "--epochs 0 only evaluates --init on val/ and test/ (-o is ignored).\n",
               argv[0], o.epochs, o.batch_size, o.n_gpus, o.lr, o.warmup_fraction, o.decay_after, o.min_lr,
               o.grad_clip, o.seed, o.n_threads, o.log_every, o.mb_growth, o.mb_budget,
               o.tile_atom_ratio, o.graph_cache, o.prefetch, o.muon_momentum, o.muon_lr_scale, o.muon_ns_steps,
               o.log_errors_epochs, o.energy_weight, o.force_weight, o.nc_force_weight, int(o.allreduce));
    std::exit(2);
  };
  int64_t dump_spe = 0;
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
    else if (a == "--schedule") o.schedule = val();
    else if (a == "--decay-after") o.decay_after = std::stoi(val());
    else if (a == "--min-lr") o.min_lr = std::stod(val());
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
    else if (a == "--force-scale-global") o.force_scale_global = true;
    else if (a == "--energy-sw") o.energy_sw = val();
    else if (a == "--lr-energy-grad") o.lr_energy_grad = val();
    else if (a == "--charge-corr-weight") o.charge_corr_weight = std::stod(val());
    else if (a == "--loss") o.loss = val();
    else if (a == "--huber-delta-e") o.huber_delta_e = std::stod(val());
    else if (a == "--huber-delta-f") o.huber_delta_f = std::stod(val());
    else if (a == "--charge-corr-target") o.charge_corr_target = val();
    else if (a == "--charge-corr-min") o.charge_corr_min = std::stod(val());
    else if (a == "--frag-charge-weight") o.frag_charge_weight = std::stod(val());
    else if (a == "--frag-charges") o.frag_charges = true;
    else if (a == "--frag-cond-gate") o.frag_cond_gate = true;
    else if (a == "--tail-data") o.tail_data = val();
    else if (a == "--tail-weight") o.tail_weight = std::stod(val());
    else if (a == "--tail-floor-mev") o.tail_floor_mev = std::stod(val());
    else if (a == "--tail-delta") o.tail_delta = std::stod(val());
    else if (a == "--checkpoint-steps") o.checkpoint_steps = std::stoll(val());
    else if (a == "--sqe") o.sqe = true;
    else if (a == "--sqe-q0-weight") o.sqe_q0_weight = std::stod(val());
    else if (a == "--allreduce") o.allreduce = std::stoi(val()) != 0;
    else if (a == "--save-every") o.save_every = std::stoi(val());
    else if (a == "--log") o.log_jsonl = val();
    else if (a == "--state-every") o.state_every = std::stoi(val());
    else if (a == "--max-hours") o.max_hours = std::stod(val());
    else if (a == "--resume") o.resume = val();
    else if (a == "--lr-schedule") o.lr_schedule = val();
    else if (a == "--dump-lr-schedule") o.dump_lr_schedule = std::stoll(val());
    else if (a == "--dump-steps-per-epoch") dump_spe = std::stoll(val());
    else usage();
  }
  // --lr-schedule F --dump-lr-schedule N --dump-steps-per-epoch S [--epochs E]: the table
  // without data or model (the golden test); with data, the run's own steps per epoch
  if (o.dump_lr_schedule > 0 && dump_spe > 0) {
    if (o.lr_schedule.empty()) usage();
    const auto s = mlipcpp::train::LrSchedule::load(o.lr_schedule);
    fmt::print("{}\n", s.table_json(o.dump_lr_schedule, dump_spe, dump_spe * o.epochs));
    return 0;
  }
  if (o.data_dir.empty() || o.init_gguf.empty() || o.output.empty()) usage();
  try {
    mlipcpp::train::train(o);
  } catch (const std::exception &e) {
    fmt::print(stderr, "error: {}\n", e.what());
    return 1;
  }
}
