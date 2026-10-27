# Training PET with `mlipcpp-train`

`mlipcpp-train` trains and fine-tunes metatrain PET models on CUDA GPUs (or
the CPU) in C++/ggml. Its gradients are checked against PyTorch. It reads
metatrain's memmap datasets and writes GGUF files that `mlipcpp-md`,
`mlipcpp-predict` and the Python package load directly.

This page covers usage. Measured speeds are in [Performance](#performance).
They come from lyra (RTX PRO 6000 Blackwell) on MAD-CORE.

## Build

```bash
module load gcc cuda cmake nccl     # NCCL enables the multi-GPU all-reduce
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF
cmake --build build -j --target mlipcpp-train mlipcpp-train-check
```

`MLIPCPP_BUILD_TRAINING` is on by default (POSIX only). On a cluster, build
inside a job, not on the login node.

## Inputs

**Data**: a directory with `train/`, `val/` and `test/`, each a metatrain
`MemmapDataset` split. Energies and forces are read from it. Forces are the
`non_conservative_force` target.

**Initial model**: a PET GGUF with the direct-force head. Either export a
fresh metatrain initialisation:

```bash
# from the directory holding train/ val/ test/: the PET `mtt train options.yaml` would start from
python scripts/train/make_init.py options.yaml -o init.pt
python scripts/export_pytorch/export_pet_gguf.py --model init.pt --nc-forces -o init.gguf
```

or start from any GGUF that `mlipcpp-train` wrote (fine-tuning, see below).

## Quick start

```bash
mlipcpp-train --data DATA --init init.gguf -o model.gguf \
    --gpus 2 --batch-size 64 --epochs 10 --optimizer muon --muon-lr-scale 10 \
    --threads 32 --cache-dir /scratch/nl-cache
```

- `--batch-size` counts structures **per GPU**. A step holds `gpus x batch-size` structures.
- `--cache-dir` stores neighbour lists across runs, keyed by dataset and
  cutoff. The first epoch of a new dataset builds them. Later runs start
  immediately.
- After each epoch the model is written as `model.gguf.epoch<k>`, and at the
  end as `model.gguf`. The final evaluation is on `test/`.
- `--epochs 0` only evaluates `--init` on val and test.

## Long runs

```bash
mlipcpp-train --data DATA --init init.gguf -o model.gguf --conservative --nc-force-weight 0 \
    --batch-size 32 --lr 1e-4 --schedule linear --decay-after 10 --min-lr 1e-6 --grad-clip 1e30 \
    --epochs 4000 --save-every 100 --state-every 10 --max-hours 23.3 --log log.jsonl
# next job, after "paused at epoch k":
mlipcpp-train ... --init model.gguf --resume model.gguf.state
```

- `--schedule linear`: the lr is constant for `--decay-after` epochs, then
  linear to `--min-lr` at the last step (LOREM's schedule). The default is
  metatrain's warmup and cosine.
- `model.gguf.best.gguf` is the epoch with the highest R²(E) + R²(F) on `val/`
  (R² = 1 − SSE/SST, energies per atom after the composition, force
  components pooled).
- `--log FILE` appends one JSON line per epoch: val E/F MAE (conservative
  mode), RMSE and R² in %, train loss and RMSE, lr, times. Line 0 is the
  initial model.
- `--save-every N` writes `model.gguf.epoch%05d.gguf` (numbered by epochs
  done) every N epochs instead of a file after every epoch.
- `--state-every N` writes `model.gguf` and `model.gguf.state` (the Adam
  moments, step, epoch, best score) every N epochs. With `--max-hours H` the
  run stops after the epoch in which H hours passed and writes both. A resumed
  run continues with the same shuffles and rotations (they are seeded per
  epoch and step). The Muon split over devices has no state support.
- `scripts/train/xyz_to_memmap.py` writes an extended-XYZ file as a split;
  `scripts/train/pet_long*.sbatch` are the bio_dimers runs on lyra.

## Recommended settings

### Optimizer

Use **Muon with `--muon-lr-scale 10`** at the base lr (1e-4). In every
from-scratch comparison so far it beat Adam at equal epochs. On MAD-CORE
2^18 over 10 epochs, test RMSE E / F (meV/atom, meV/Å):

| optimizer | E | F |
|---|---|---|
| Adam 1e-4 | 291.8 | 1417 |
| Adam 3e-4 | 238.8 | 1058 |
| Muon x3 | 204.9 | 903 |
| **Muon x10** | **182.5** | **828** |

The Muon step costs about 2 ms more than Adam's. With several GPUs it is
split across them. Polar Express coefficients (`MLIPCPP_NS_POLAR=1`) tied with
the fixed quintic and are opt-in.

### Batch size and packing

Keep the default `--pack step`. All structures of a step are sorted by
neighbour count and cut into micro-batches by a cost model fitted online to
the measured time per micro-batch. The gradients are exactly those of the
full batch, however it is cut. Guidance:

- **Direct forces:** 64 per GPU is a good default. Throughput depends mostly
  on the structure size of the dataset, not on the batch.
- **Conservative forces (`--conservative`):** use **512–1024 per GPU**.
  Throughput rises up to about 512 and then levels off (see the table below).
  Large batches mean fewer optimizer steps per epoch, so raise the learning
  rate: Muon x10 at lr 1e-4 is the current choice.
- Each log line is followed by a "packing cost model" line with the fitted
  fixed and per-edge costs. A fixed cost near 0 ms with tens of micro-batches
  per worker means the plan is splitting too much. Report it.
- Tiling (`--tile-nb-ratio 1.25`) pads micro-batches onto a grid of shapes and
  replays CUDA graphs. It is correct but slower, and uses much more memory.
  Leave it off.

### Large or dense structures

`--max-attention 5e7` drops training structures whose attention size
`n_atoms x (max_neighbours + 1)^2` exceeds the micro-batch budget.
`--max-neighbors N` drops structures with any atom above N neighbours. With
the adaptive cutoff (`cutoff_function: adaptive` in the metatrain options)
the neighbour counts stay small and neither limit is normally needed.

### Multi-GPU

`--gpus N` uses N devices of one node. A build with NCCL uses `--allreduce 1`
(the default): one all-reduce of the gradients, and each GPU steps its own
copy of the optimizer. On 8 GPUs this takes about 2–3 ms per step, against
10.6 ms for the old device-0 reduction.

## Conservative fine-tuning

`--conservative` puts the force loss on `F = -dE/dx`. Its weight gradient
comes from a forward-mode tangent of E along the force residual, followed by
a backward pass, which equals PyTorch's double backward to 4e-6 in FP32. The
direct-force head keeps a loss term of its own. The adaptive-cutoff chain
term is included.

```bash
mlipcpp-train --data DATA --init pretrained.gguf -o cons.gguf --gpus 8 \
    --conservative --model-baselines \
    --energy-weight 0.1 --force-weight 5 --nc-force-weight 1 \
    --batch-size 512 --optimizer muon --lr 1e-4 --muon-lr-scale 10 --epochs 3
```

- **`--model-baselines` is required for fine-tunes.** It keeps the
  composition energies and scales stored in `--init`. Without it they are
  refitted on the training set, and with a high learning rate that has
  wrecked a pretrained model.
- Loss weights: `--energy-weight` (E), `--force-weight` (−dE/dx) and
  `--nc-force-weight` (direct head). In metatrain-based fine-tunes, a large
  force weight traded energy for conservative forces (E 93 → 202 meV/atom,
  F_cons 2475 → 791 meV/Å). Raising the energy weight recovered part of the
  energy.
- Pretrain with direct forces first, then fine-tune conservatively. The
  conservative step costs about 3.5x as much GPU time per structure.
- Check the result in MD: `mlipcpp-md check` / `run` (NVE drift with
  conservative forces).

## Monitoring and per-structure errors

A log line per `--log-every` steps:

```
epoch 0 step 468/64225: loss ... | 1889.4 struct/s; per step: wait 0.1 ms, gpu 133.2 (build 8.8, compute 106.6), sync 2.1 ms; 1.66 micro-batches/worker; ...
  packing cost model: 7.96 ms + 1.41e-06 n(m+1)^2 + 0.00272 n(m+1)
```

- `wait`: the GPUs waited for batch preparation. This should be about 0.
  Otherwise raise `--threads` or `--prefetch`.
- `gpu`: the slowest worker's time on its micro-batches. `build` is graph
  construction, `compute` is GPU time, and the remainder is host work,
  transfers and imbalance.
- `sync`: the gradient all-reduce and the optimizer step.

`--dump-errors PREFIX` writes the per-structure val and test errors at the
end. `--log-errors FILE` records each training structure's error before the
update it takes part in, for the first `--log-errors-epochs` epochs. These
are 24-byte `SystemError` records, used for label-quality screening.
`--exclude DIR` drops the split indices listed in `DIR/{train,val,test}.npy`.

## Checking a build

`mlipcpp-train-check` runs one batch through a model, by default against a
PyTorch reference dump:

```bash
# gradients vs PyTorch (scripts/train/check_gradients.py), micro-batch and tile consistency
GGML_CUDA_NO_TF32=1 mlipcpp-train-check --data DATA --init init.gguf --out chk --n 16 --gpus 1 \
    [--conservative 1 --energy-weight 0.1 --force-weight 5 --nc-force-weight 1]
# per-phase wall time and the node-by-node op profile of one step
mlipcpp-train-check --data DATA --init model.gguf --out prof --n 256 --gpus 1 --profile 1 --tol 1 \
    --conservative 1 --energy-weight 0.1 --force-weight 5 --nc-force-weight 1
```

Gate on freshly initialised models. A trained model's gradients depend
slightly on how the batch is cut, through the padded attention keys, in
metatrain as well. TF32 alone moves gradients by about 1e-3, so gates set
`GGML_CUDA_NO_TF32=1`. In conservative mode, the profile labels ops by graph:
`F.fwd`/`F.bwd` is the forces graph, and `M.fwd`/`M.tan`/`M.bwd` is the loss
graph's primal forward, tangent and backward.

`mlipcpp-jvp-check` tests the forward-mode tangent rules on the PET graph
against the backward pass and central differences.

## Performance

MAD-CORE, big PET (d_pet 256, 16 heads, 4 attention layers) with the
adaptive cutoff, lyra RTX PRO 6000, 2 GPUs unless noted.

| run | struct/s per GPU |
|---|---|
| direct forces, 2^24, batch 64 (145880) | 2710 |
| conservative, batch 32, 8 GPU, old reduction (175917) | 553 |
| conservative, batch 32 | 677 |
| conservative, batch 128 | 934 |
| conservative, batch 512 | 1109 |
| conservative, batch 1024 | 1165 |
| conservative, batch 2048 | 1141 |

For direct-force training of the base PET on MAD-CORE 2^20, metatrain was
about 2.5x slower per epoch than `mlipcpp-train` and used about 20x the GPU
memory (90.8 vs 4.6 GB; runs 145166 / 145179).

### Where a conservative step's time goes

The profile of one 256-structure step from 2^24 on one GPU
(`mlipcpp-train-check --profile 1 --conservative 1`, job 176448) breaks down
as follows. Of 258 ms wall time, 228 ms is GPU compute, 21 ms graph
construction, 4 ms allocation and 0.6 ms host work. GPU compute is 4.0x the
direct-force step on the same batch (57 ms). The step launches 24.6k kernels
(8.1k for direct forces) and moves 286 GB in ideal memory traffic (64 GB).
At this model size it is limited by memory bandwidth, not by its GEMMs.
Node-by-node shares:

| phase | share | what it is |
|---|---|---|
| `M.bwd` | 48 % | backward of the loss graph through the tangent (weight gradients) |
| `F.bwd` | 16 % | backward to the edge vectors and cutoffs (dE/dx) |
| `M.tan` | 16 % | forward-mode tangent of E |
| `F.fwd` | 11 % | forward of the forces graph |
| `M.fwd` | 9 % | the same forward, recomputed by the loss graph |

The largest single items are the weight-gradient GEMMs (`M.bwd:OUT_PROD`, at
46 TFLOP/s), then the elementwise `MUL` / `ADD` / `SCALE` / `REPEAT_BACK`
nodes of the two backward passes, then the `CONT` copies that the tangent
rules materialise (`M.tan:CONT`, 17.5 GB). The fused attention tangent ops
take about 7 %.

## Environment switches

These are for A/B tests and diagnostics. The defaults are the tested paths.

| variable | effect |
|---|---|
| `GGML_CUDA_NO_TF32=1` | FP32 GEMMs (gradient gates) |
| `GGML_PET_ATTN_BF16=1` | bf16 tensor-core attention kernels (forward and backward, not the tangent) |
| `MLIPCPP_NS_POLAR=1` | Polar Express Newton–Schulz coefficients |
| `GGML_NS_F32=1` | FP32 Newton–Schulz (default bf16 storage) |
| `MLIPCPP_JVP_UNFUSED_ATTN=1` | unfused attention tangent (checks) |
| `MLIPCPP_NO_FUSED_ATTN=1` | unfused attention (checks) |
| `MLIPCPP_PAD_FLOOR=x` | padded-key attention floor (diagnostic) |
