# mlipcpp

> **WARNING: Pre-alpha software**
> This project is in early development. APIs may change without notice, and there may be bugs or incomplete features. Not recommended for production use.

Standalone C++ implementation of Machine Learning Interatomic Potentials (MLIPs) using [ggml](https://github.com/ggml-org/ggml).

Currently supports [PET/uPET](https://github.com/lab-cosmo/pet-mad) models (energy, forces, stresses).

## Quick start (Python)

```bash
# Install the package
pip install .

# Download and convert a model to GGUF
uv run scripts/convert_models.py --models pet-mad-s
```

```python
import numpy as np
import mlipcpp

# Load a model
model = mlipcpp.Predictor("gguf/pet-mad-s.gguf")
print(f"Model type: {model.model_type}, cutoff: {model.cutoff} A")

# Water molecule
positions = np.array([
    [0.000,  0.000, 0.000],  # O
    [0.757,  0.586, 0.000],  # H
    [-0.757, 0.586, 0.000],  # H
], dtype=np.float32)
atomic_numbers = np.array([8, 1, 1], dtype=np.int32)

# Predict energy
result = model.predict(positions, atomic_numbers, compute_forces=False)
print(f"Energy: {result.energy:.4f} eV")
# => Energy: -14.3693 eV

# Predict energy + forces
result = model.predict(positions, atomic_numbers, compute_forces=True)
print(f"Energy: {result.energy:.4f} eV")
forces = np.array(result.forces).reshape(-1, 3)
print(f"Forces (eV/A):\n{forces}")
```

### ASE integration

```python
from ase.io import read
from mlipcpp.ase import MLIPCalculator

atoms = read("structure.xyz")
atoms.calc = MLIPCalculator("gguf/pet-mad-s.gguf")
print(f"Energy: {atoms.get_potential_energy():.4f} eV")
```

## Converting models

Download and convert uPET models from HuggingFace to GGUF format:

```bash
# Convert all available models
uv run scripts/convert_models.py

# Convert a specific model
uv run scripts/convert_models.py --models pet-mad-s

# List available models
uv run scripts/convert_models.py --list
```

Default models: `pet-mad-s`, `pet-oam-l`, `pet-omad-xs`, `pet-omad-s`, `pet-omat-xs`, `pet-omat-s`, `pet-spice-s`

Use `--all` to also convert larger variants: `pet-oam-xl`, `pet-omad-l`, `pet-omat-m`, `pet-omat-l`, `pet-omat-xl`, `pet-omatpes-l`, `pet-spice-l`

## Building from source

### Dependencies

- [ggml](https://github.com/ggml-org/ggml) - Tensor library (fetched automatically via CMake)
- [fmt](https://github.com/fmtlib/fmt) - Formatting library (fetched automatically)

**Note:** This project uses a [modified fork of ggml](https://github.com/peterspackman/ggml) with additional backpropagation support for `CONCAT` and `CLAMP` operations, required for force/stress computation.

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

On CPU, `-DMLIPCPP_USE_BLAS=ON` (with `-DGGML_BLAS_VENDOR=OpenBLAS`,
`Intel10_64_dyn` for MKL, ...) runs the matrix products of the graph models
on BLAS and the rest on ggml's CPU backend, e.g. 1.6x faster energies and
1.3x faster forces for a 64-atom PET with MKL on 12 Zen 3 cores. Set
`MLIPCPP_NUM_THREADS` for the number of threads (default 4, shared by BLAS),
and `MLIPCPP_NO_ACCEL=1` to run without BLAS.

### C++ CLI

```bash
# Energy only
./build/bin/graph_inference gguf/pet-mad-s.gguf structure.xyz

# With forces
./build/bin/graph_inference gguf/pet-mad-s.gguf structure.xyz --forces

# With forces and stress (periodic systems)
./build/bin/graph_inference gguf/pet-mad-s.gguf structure.xyz --forces --stress
```

## metatomic plugin

`-DMLIPCPP_BUILD_METATOMIC=ON` builds `mlipcpp-metatomic.so`, a
[metatomic](https://docs.metatensor.org/metatomic/) plugin that lets any
engine using the metatomic C API load `*.gguf` models (energy, forces, stress
and non-conservative forces). See [docs/metatomic.md](docs/metatomic.md).

## Training PET on GPUs

`mlipcpp-train` trains and fine-tunes metatrain PET models on CUDA (one node,
any number of GPUs), with direct or conservative (-dE/dx) forces. See
[docs/training.md](docs/training.md) for usage, recommended settings and
measured speeds.

## PET with long-range features (experimental)

PET checkpoints trained with `long_range.enable: true` can be exported with:

```bash
python scripts/export_pytorch/export_pet_gguf.py \
  --model pet-lr.ckpt --long-range-solver p3m -o pet-lr.gguf
./build/bin/graph_inference pet-lr.gguf structure.xyz --forces --stress
```

The P3M path applies the excluded Coulomb potential directly to the learned charge
channels, without allocating an N-by-N matrix or imposing the reference solver's
512-atom limit. It uses Apple Accelerate FFTs on macOS (a portable CPU FFT
elsewhere), fifth-order charge assignment, and
torch-pme's potential influence function, self term, and neutralizing background.
Both residual and feedforward PET support energies, conservative forces, and
periodic stress, including gradients through the learned charges and mesh weights.
No torch-pme dependency is needed at inference time.

For fully periodic systems, electrostatics costs `O(C * (N + M log M))` at fixed
neighbor density, with `C` charge channels and `M` mesh points; working storage is
`O(N*C + M)`. Mesh dimensions follow torch-pme's power-of-two rule using the
checkpoint's `kspace_resolution`. Triclinic cells are supported. This initial CPU
implementation processes channels sequentially and recomputes the PET graph to
propagate charge gradients. It guards allocations at 8 million mesh points and
1024 points per axis. These are mesh resource limits, not atom-count limits.

Nonperiodic systems use direct excluded Coulomb interactions with linear storage
but quadratic work. Fixed cutoffs and `interpolation_nodes: 5` are required;
adaptive cutoffs, partial periodicity, and non-conservative force heads
are not supported for PET-LR. Exporting requires the checkpoint's Python
dependencies. The existing `--long-range-solver direct-ewald` option remains a
small-system reference (maximum 512 atoms and `N*N*number_of_k_vectors <= 100,000,000`).
Its periodic results follow Ewald rather than P3M. Re-export an existing checkpoint
with `--long-range-solver p3m` to select the mesh path explicitly.

### Experimental native PET-LR training on Metal

With `MLIPCPP_BUILD_TRAINING=ON` and `MLIPCPP_USE_METAL=ON`,
`mlipcpp-train-lr` can fine-tune an exported conservative PET-LR checkpoint:

```bash
./build/bin/mlipcpp-train-lr --init pet-lr.gguf --data data/train \
  --backend metal --check-gradient
./build/bin/mlipcpp-train-lr --init pet-lr.gguf --data data/train \
  --backend metal --steps 20 --lr 1e-4 --batch-size 3 --output trained-lr.gguf
./build/bin/mlipcpp-train-lr --init trained-lr.gguf --data data/val \
  --backend metal --evaluate
```

The data argument is one native memmap split. Its historically named
`non_conservative_force.bin` supplies reference forces; this trainer predicts
**conservative forces from the energy**, with no separate force head. Energy
and force losses are the mean squared energy error per atom and mean squared
force-component error, weighted by `--energy-weight` and `--force-weight`
(both default to 1). Checkpoint composition energies and scaling are retained;
initialize them before exporting a new model.

The default `--method analytic` computes force-loss parameter gradients by
forward directional differentiation followed by reverse differentiation. For
`d = dL/dF`, it differentiates `-D_R E[d]` with respect to the weights, including
both charge and position derivatives of the P3M mesh, exclusion, self and
background terms. No displaced geometries are used. The network runs in
C++/ggml on Metal or CPU; P3M uses two CPU workers and Adam runs on the CPU.
The fused SiLU and softmax derivative operations have analytic backward rules.
Training explicitly requests full-F32 matrix multiplication on Metal; the
default ggml matrix kernel may otherwise convert F32 operands to half internally,
which is insufficiently accurate for these mixed derivatives.

`--batch-size` packs independent structures into a disjoint neural graph, with
separate P3M cells. Energy loss is averaged over structures; force loss is
averaged over all force components. A bounded cache retains the most recent
atom/neighbour shape (neighbours rounded up to a multiple of 32). Inputs,
constants, and gradient seeds remain persistent, while temporary storage is
reused across the six execution stages. Repeated shapes require no graph rebuild
or device allocation. Changing the atom count or neighbour bucket rebuilds the
cache. This path currently requires an F32, fixed-cutoff conservative P3M export.

`--method fd` retains the single-structure directional finite-difference method
for diagnostics. `--fd-step` bounds its largest Cartesian displacement in
angstroms (default 0.01). `--check-gradient` checks one structure against a
numerical parameter-direction loss derivative and an independent finite-
difference mixed derivative, failing above 5% relative discrepancy. In FD mode,
the latter check instead halves the displacement. For stronger validation,
`--dump-gradient` exports every named parameter gradient for an independent
analytic reference; small force-only gradients can make FD checks noisy.

For longer experiments, pass `--validation data/val --eval-every 255` to
evaluate the full validation split and save numbered checkpoints every 255
updates. The initial model is evaluated too; `<output>.best.gguf` retains the
lowest weighted validation MSE seen during the run. Optimizer state stays live
between these evaluations. With `--check-gradient`, `--index` selects a training
structure and `--dump-gradient gradients.json` saves named parameter gradients
for comparison with an independent implementation.

`MLIPCPP_PROFILE_TRAIN=1` prints stage timings and allocated graph memory for
the analytic trainer. `MLIPCPP_PROFILE_GRAPH=1` profiles the inference/FD path.
On Apple platforms, `-DMLIPCPP_ACCELERATE_FFT=OFF` selects the portable FFT.
`--threads` controls native CPU neural-graph execution (default 2); benchmark
several settings rather than comparing Metal only against one CPU thread.

```bash
python scripts/benchmark_conservative.py --binary ./build/bin/mlipcpp-train-lr \
  --init pet-lr.gguf --data data/train --output benchmark-results \
  --steps 22 --warmup 2 --batches 1 3 --cpu-threads 2 4 8
```

This runs CPU and Metal sequentially from identical weights, with the same
seed, optimizer, loss, and batches. It records full process times and warmed
update times, including packing, P3M, gradient readback and Adam updates.
Validation and checkpoint writing are outside the per-update timer. Throughput
is reported in structures/second, so different batch sizes are comparable.
Metal support alone does not establish a speed advantage: the small width-16
water model still spends substantial time in P3M and many small GPU operations.
Compare with an analytic PyTorch CPU baseline as well as the native CPU backend.

Measured on an 8-core M1 Pro with 16 GiB RAM, the 11,794-parameter width-16
Cheng-water model (192 atoms/structure) gave the following warmed training
seconds per structure. Each run used 22 updates, excluded the first two, and
included P3M and Adam; all started from the same weights and shuffled order.

| Backend | Batch 1 | Batch 3 |
| --- | ---: | ---: |
| Native CPU, 4 threads | 0.381 | 0.379 |
| Metal, full F32 | 0.396 | 0.409 |
| PyTorch CPU analytic, 2 threads | 0.160 | 0.160 |

Analytic Metal is about twice as fast as the earlier 0.785 s/structure FD Metal
path, but **does not meet the faster-than-CPU goal for this model**. Batch three
uses about 603 MiB of graph storage and does not improve Metal throughput.
Independent force-only gradients on sampled initial/trained water checkpoints
agree with PyTorch autograd within 7e-6 relative L2 error. These are performance
and derivative checks, not a new convergence study.

The experimental trainer uses gradient clipping at norm 1, CPU Adam, shuffled
sampling and GGUF checkpoints; it does not resume optimizer state. The existing
`mlipcpp-train` remains the separate direct-force-head trainer. A successful
short run verifies execution, not a converged molecular-dynamics model.

Validation against metatrain and torch-pme includes both PET featurizers, isolated
atoms, distant pairs, triclinic cells, periodic images, and a 729-atom periodic cell:

```bash
python tests/generate_pet_lr_reference.py /tmp/pet-p3m-reference --solver p3m
MLIPCPP_PET_LR_REFERENCE=/tmp/pet-p3m-reference/reference.json \
  ./build/tests/test_coulomb
```

Build with `-DMLIPCPP_BUILD_TESTS=ON`. Without Python, the test executable still
checks analytic charge, force, and stress derivatives against finite differences
and exercises the mesh solver above 512 atoms. Use `--solver direct-ewald` to
regenerate the dense reference fixtures.

## API

C, C++, Fortran, and Python APIs are provided. See `examples/` for usage.

## License

BSD 3-Clause
