# Plato-PET-LR in mlip.cpp

metatrain's `experimental.plato_pet_lr` (branch `feat/plato-cheap-frames`) in the
configuration `plato-D2-att`: D2 (Klein four-group) frames used only inside the
attention (`frame_mode: attention`), the learned mesh coupling after layer
`couple_at`, and the LES energy of latent charges. Phase 1 (this page): exact
inference (energy, forces, stress) with ggml autodiff. Phase 2: training in
`mlipcpp-train` (plan at the end).

## Files

| file | what |
|---|---|
| `scripts/plato/export_plato_gguf.py` | checkpoint -> GGUF: every state-dict tensor verbatim (name, dtype, bits), hypers and dataset info as JSON, composition and scale as float64 metadata |
| `scripts/plato/gguf_to_plato_ckpt.py` | GGUF -> metatrain checkpoint (`--check orig.ckpt`: bit for bit) |
| `scripts/plato/plato_reference.py` | PyTorch float32 / float64 references: energy, forces, stress per model part (full, no LES, no mesh, short range), D2 rotations, a generic rotation, open and triclinic cells, a 2x2x2 tile |
| `cmake/patches/ggml-plato.patch` | the ggml ops below, CPU + CUDA |
| `src/runtime/plato_graph.{h,cpp}` | the model as one ggml graph on packed tokens |
| `src/models/plato/plato.{h,cpp}` | `PlatoModel` (the `Model` interface; `load_model` on a `plato-pet-lr` GGUF) |
| `src/bin/mlipcpp_plato.cpp` | `mlipcpp-plato check` (parity) and `bench` (timing) |

```bash
python scripts/plato/export_plato_gguf.py model.ckpt -o model.gguf
python scripts/plato/gguf_to_plato_ckpt.py model.gguf -o back.ckpt --check model.ckpt
python scripts/plato/plato_reference.py model.ckpt test.xyz ref --rt back.ckpt
mlipcpp-plato check model.gguf ref --backend cuda      # GGML_CUDA_NO_TF32=1: FP32 GEMMs
mlipcpp-plato bench model.gguf ref/structures.xyz --tile 8 --backend cuda
```

## The graph

Tokens are packed as in the ragged PET (`pet_ragged.h`): the n centre tokens,
then the E edge tokens grouped by centre, columns of `[d, n + E]`; no padding.
With one token frame every `FrameLinear` is a linear map
`W_eff = sum_b phi_b(I) W_b`. By default the folds are computed once on the host
in double (`fold.<prefix>` weights); `MLIPCPP_PLATO_FOLD_IN_GRAPH=1` builds them in
the graph from the raw `W` and `phi` (the training layout: gradients reach the
checkpoint's own parameters). The GGUF always holds the raw parameters.

- edge embedding `sum_f w_f SiLU(W R_f^T r) + rad(rbf(d)) + sp_nbr + sp_center`,
  the frames as four 3x3 products of the edge vectors;
- RoPE angles `r . R_f omega` as one `[3, F d_h/2]` product (the centres: 0);
- `ggml_plato_attn`: per head `sum_f w_f softmax_t(scale rope_f(q) . rope_f(k)
  + log f_c) v`, frames averaged before `@ v`, on the ragged sequences; its
  backward returns d qkv, d angles (so forces reach the RoPE) and d bias;
- mesh coupling: `ggml_plato_spread` (order-4 B-splines, the stencil of
  `bspline4_stencil`), `ggml_plato_fft3` (rfftn), the learned filter
  `4 pi / V exp(-sigma^2 k^2 / 2) / k^2 (1 + filt(|k|)) / deconv` as plain ops,
  irfftn, `ggml_plato_gather`;
- LES: the windowed Ewald sum `4 pi / V sum_k g(k) |S(k)|^2` minus the Gaussian
  self term (plain ops: phases `2 pi n . frac`, cos, sin, two products), or the
  `erf(r / sqrt(2) sigma) / r` pair sum (`ggml_plato_erf`) for open systems.

Forces and stress come from one backward pass w.r.t. the edge vectors, the
positions (through the fractional coordinates of mesh and LES) and a strain
tensor `eps` (zeros): under strain the fractional coordinates and phases are
invariant, and only `k (I + eps)^-T` and `V det(I + eps)` change; both enter to
first order, which is exact for the derivative at `eps = 0`.
`stress = (sum_e v_e (x) dE/dv_e + dE/d eps) / V`, as metatomic's strain gradient.

### ggml ops (`ggml-plato.patch`)

| op | forward | gradient |
|---|---|---|
| `PLATO_ATTN` | frame-averaged RoPE attention (CPU reference; CUDA: a block per sequence, rotated keys/queries in shared memory, head size 8/16/32, F in {1, 3, 4, 6, 12}) | `PLATO_ATTN_BACK`: d qkv, d angles, d bias |
| `PLATO_FFT3` | modes 0 rfftn, 1 irfftn, 2 / 3 their adjoints (CPU: mixed-radix, double lines; CUDA: cuFFT R2C, C2C over axes 0 and 1 + C2R) | the adjoint mode (0 <-> 3, 1 <-> 2): linear, so differentiable to any order |
| `PLATO_SPREAD` | B-spline charge assignment `[C, n] -> [M2, M1, M0, C]` | d v = gather; d frac = sum_c v gather' |
| `PLATO_GATHER` | interpolation `-> [C, n]`; with `deriv`, d/d frac `[C, 3, n]` | d mesh = spread; d frac = sum_c g gather'; the `deriv` form has none yet |
| `PLATO_ERF` | erf | `2 / sqrt(pi) exp(-x^2)` as ops |
| `GET_ROWS_BACK` | (upstream) | its gradient, a `GET_ROWS` (the readout `h = tok_i + sum_e f_c tok_e`) |

## Mesh size

PyTorch uses `max(8, ceil(|a_i| / mesh_spacing))` per cell vector, from the
detached cell. For an integer-length box the norm sits on an integer: a generic
rotation of the 30 A cube gives `|a_2| = 30.0000000233` from the float32 cell,
mlip.cpp (norm in double, like PyTorch float64) takes 31 points where PyTorch
float32 (its norm rounds to 30.0) takes 30. The trained model is sensitive to
this: 30 vs 31 points changes the cc0 dimer energy by about 50 meV, and a 32^3
mesh by 240 meV. `PlatoOptions::fixed_mesh` / `MLIPCPP_PLATO_MESH="M"` or
`"M0,M1,M2"` fixes the mesh (off by default, so the rule above is reproduced).
A fixed mesh makes the energy a smooth function of the cell, but the model was
trained with the varying rule.

## Phase 1 results (lyra, 2026-10-23)

PyTorch references: `plato_reference.py` on cosmopc7 (CPU, torch 2.13), float32 and
float64, positions and cells rounded to float32 first. Structures: six test dimers
(14-25 atoms, 30 A boxes), the three D2 rotations of the CC dimer, a generic rotation,
the CC dimer open (no PBC) and in a triclinic cell, and the CC dimer tiled 2x2x2
(160 atoms, 60 A). Each with the full model, without LES, without the mesh, and short
range only. mlip.cpp float32 on CUDA with FP32 GEMMs (`GGML_CUDA_NO_TF32=1`):

| | max over structures and parts (vs PyTorch float64) |
|---|---|
| energy | 0.016 meV (160 atoms), <= 0.002 meV (20 atoms) |
| forces | max abs 1.0e-6 eV/A (RTX PRO 6000), 2.5e-6 (B200) |
| stress | 1.3e-9 eV/A^3 |

PyTorch float32 itself is further from float64 (energy up to 0.12 meV at 160 atoms,
forces 9e-6 eV/A). The CPU backend: energies the same, forces up to 4.5e-5 eV/A
(LES phases through the CPU's float32 cos/sin). TF32 GEMMs (the ggml default):
energies 0.02-1.0 meV, forces up to 3.4e-4 eV/A (the geometry and long-range products
are kept in FP32 under TF32, `mm_f32`; without that TF32 cost 8 meV).

D2 invariance: the three group rotations of the CC dimer change the energy by
<= 2.4e-7 eV (CPU), <= 7e-7 eV (CUDA): float32 rounding, the frames summed in another
order (PyTorch float32: 3.6e-7 eV; float64: 6e-16 eV). A generic rotation changes it by
42 meV (10 meV approximate invariance plus the 30 -> 31 mesh jump).

GGUF round trip: checkpoint -> GGUF -> checkpoint is bit-identical for all 210
state-dict tensors (float32 weights, the float64 / int64 / uint8 buffers) and gives
identical PyTorch float32 predictions on every reference structure.

Timing: ms per energy + forces call, float32 (FP32 GEMMs), CUDA, mlip.cpp's host
neighbour list excluded as in the PyTorch numbers (TorchScript `model.pt`, float32,
median of 30 / 10 calls, its neighbour lists excluded), same node and job:

| model, atoms | RTX PRO 6000 (180850, lr011) mlip.cpp / PyTorch | B200 (180857, lb002) mlip.cpp / PyTorch |
|---|---|---|
| D2-att, 20 | 2.01 / 11.19 | 2.16 / 17.12 |
| D2-att, 160 | 7.82 / 13.69 | 5.32 / 17.76 |
| D2-att-sr, 20 | 1.25 / 8.08 | 1.42 / 12.80 |
| D2-att-sr, 160 | 1.35 / 8.22 | 1.55 / 12.96 |
| D2-att-sr, 10,240 | 14.07 / 44.80 | 14.71 / 38.04 |
| pet-s (study), 20 / 160 / 10,240, PyTorch | 7.08 / 7.13 / 36.79 | 10.10 / 10.36 / 35.73 |
| pet-s in mlip.cpp (GraphModel, padded path, host NL included) | 1.57 / 1.81 / 30.95 | 1.81 / 1.99 / 26.43 |

mlip.cpp's host neighbour list (vesin, CPU) adds 0.3 ms at 20 atoms and 7.2 ms at
10,240. At 160 atoms the LES Ewald sum (394k k vectors x 160 atoms) is 5.7 of the
7.8 ms (RTX PRO 6000; without LES 2.1 ms, short range alone 1.3 ms).

Not done in phase 1: the device neighbour list and the LAMMPS kk path (the
`DeviceStep` interface is GraphModel's), P3M for the LES energy of large cells (the
plain Ewald sum is O(n n_k); at 10,240 atoms in a 240 A box it is 25M k vectors, as
in PyTorch, which is why the study timed the -sr twin there), per-atom energies,
multi-structure inference batches (the graph is batched, `PlatoModel` passes one).

## Phase 2: training (`mlipcpp-train-plato`)

```bash
python scripts/plato/plato_dataset.py train.xyz train.bin --energy e_res --forces f
mlipcpp-train-plato --init model.gguf --train train.bin --val val.bin -o out.gguf \
    --epochs 3 --batch-size 16 --lr 1e-4 --backend cuda        # GGML_CUDA_NO_TF32=1
python scripts/plato/gguf_to_plato_ckpt.py out.gguf -o out.ckpt # back to metatrain
# gradient check against PyTorch (double backward of metatrain's loss)
python scripts/plato/plato_grad_reference.py model.ckpt train.xyz ref --indices 0 1 2 3
mlipcpp-train-plato --init model.gguf --train train.bin --check ref --backend cuda
```

What it does now:

- the loss of metatrain's plato_pet_lr trainer: MSE of the energies per atom and of
  the position gradients (no stress in bio_dimers; the graph has the strain input for
  a stress term), targets minus the composition and divided by the energy scale,
  Adam (torch defaults), gradient-norm clipping 1.0, linear warmup + cosine, O(3)
  augmentation (Haar rotation times +-1). The composition and scale stay those of the
  GGUF (metatrain refits them on the same training set).
- parameters: the checkpoint's own (`W`, `phi` of every FrameLinear folded in the
  graph, LayerNorms of primitive ops), so the GGUF goes back to metatrain without
  loss (`gguf_to_plato_ckpt.py`, checked bit for bit) and mlipcpp-train restarts and
  fine-tunes are usable in metatrain/metatomic.
- conservative forces without second derivatives through the custom ops: the force
  term's parameter gradient is the derivative of the energy along the residual
  direction `u = dL/d(dE/dx)`, by a central difference of the parameter gradients
  (`T(+-) = sum_b (1 +- eps a_b) E_b(x +- eps u)`, `a_b = dL/dE_b`; 3 graph evaluations
  per batch). Error vs PyTorch: 6e-4 relative on CUDA (cosine 0.9999998), 2e-3 on
  the CPU, the float32 roundoff of the two gradients; the energy-only gradient is
  exact to 5e-5. Choosing `--fd-eps` 3e-3 to 1e-2 A is at the noise floor.
- one ggml graph per batch: all structures' tokens packed, the mesh coupling with
  each structure's channels (`C x G` channels on one mesh, a filter per structure from
  its rotated cell and strain) and the Ewald sum per structure (charges as `[n, G]`).
  A batch needs one mesh size and one integer k set (all bio_dimers boxes are 30 A
  cubes); the mesh is taken from the unrotated cell, so an augmentation rotation does
  not change it (metatrain's rule can jump to 31 under rotation: a deliberate
  difference).
- no direct-force head was needed.

Restart of the 40-epoch plato-D2-att weights, 3 epochs, lr 1e-4 (cosine, warmup 0.01),
batch 16, augmentation, RTX PRO 6000, both jobs at once on the same node (lr011, one
GPU each, shared CPUs), mlip.cpp FP32 GEMMs, metatrain float32 (`mtt train --restart`
of the GGUF round-trip checkpoint: fresh Adam and schedule):

| epoch | mlip.cpp val E / F MAE (meV/atom, meV/A) | metatrain val E / F MAE |
|---|---|---|
| start | 0.986 / 5.556 | 0.986 / 5.556 (its epoch-37 best) |
| 1 | 1.686 / 5.782 | 1.072 / 6.000 |
| 2 | 0.880 / 5.555 | 0.948 / 5.590 |
| 3 | 0.832 / 5.457 | 0.991 / 5.497 |
| s per epoch | 61 (train) + ~3 (val) | 148-160 (train + val) |

Different shuffles and augmentation draws, so only the trend is comparable: both
recover from the fresh-optimizer kick and end slightly below the start in forces.
mlip.cpp is 2.4x faster per epoch here; of its 61 s, 35 s is GPU compute (3
evaluations per step), 12 s graph construction and inputs on the host, 4 s gradient
download.

### Next steps for phase 2

1. Exact conservative gradients: forward-mode tangents (`runtime/graph_jvp.h`, as
   mlipcpp-train does for PET): rules for `PLATO_ATTN` (a fused JVP kernel and its
   backward, as `PET_ATTN_JVP*`, or the unfused per-frame expression on the padded
   layout, max_len ~13 here), `PLATO_FFT3` (linear: the same op), `PLATO_SPREAD` /
   `GATHER` (tangent in the values: the same op; in the positions: the `deriv`
   gather, whose backward needs order-2 B-spline weights, a new gather mode),
   `PLATO_ERF`, `GET_ROWS_BACK` (linear). Removes the 2e-3 / 6e-4 FD noise and one
   evaluation.
2. Speed: the parameter gradient accumulated and Adam stepped on the device (now
   downloaded, 4 s/epoch), graph reuse across batches of the same shape bucket
   (padding atoms/edges; 12 s/epoch of construction), a fused LES op.
3. Data: metatrain memmap datasets (`train/dataset.cpp`) with stress, mixed cells in
   one batch (micro-batches per mesh size / k set), the neighbour-list cache.
4. Integration into `mlipcpp-train` proper (multi-GPU all-reduce, Muon, logging and
   per-structure error dumps) once the step is a single graph.
5. Electronegativity term `E_chi = sum_i chi_i q_i` (being added in metatrain): a
   per-species table (`get_rows` on species) plus a small head on `h` (a FrameMLP as the
   charge head), dotted with the centred charges `PlatoOutputs::charges` and added to
   `energies`; ordinary ggml ops, first-order differentiable, and the FD-conservative
   step covers it unchanged. Not in phase 2 until its metatrain results are in.
