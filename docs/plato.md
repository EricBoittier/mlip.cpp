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
