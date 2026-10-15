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

### C++ CLI

```bash
# Energy only
./build/bin/graph_inference gguf/pet-mad-s.gguf structure.xyz

# With forces
./build/bin/graph_inference gguf/pet-mad-s.gguf structure.xyz --forces

# With forces and stress (periodic systems)
./build/bin/graph_inference gguf/pet-mad-s.gguf structure.xyz --forces --stress
```

## PET with long-range features (experimental)

PET checkpoints trained with `long_range.enable: true` can be exported with:

```bash
python scripts/export_pytorch/export_pet_gguf.py \
  --model pet-lr.ckpt --long-range-solver p3m -o pet-lr.gguf
./build/bin/graph_inference pet-lr.gguf structure.xyz --forces --stress
```

The P3M path applies the excluded Coulomb potential directly to the learned charge
channels, without allocating an N-by-N matrix or imposing the reference solver's
512-atom limit. It uses a portable CPU FFT, fifth-order charge assignment, and
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
adaptive cutoffs, partial periodicity, non-conservative force heads, and C++
training are not supported for PET-LR. Exporting requires the checkpoint's Python
dependencies. The existing `--long-range-solver direct-ewald` option remains a
small-system reference (maximum 512 atoms and `N*N*number_of_k_vectors <= 100,000,000`).
Its periodic results follow Ewald rather than P3M. Re-export an existing checkpoint
with `--long-range-solver p3m` to select the mesh path explicitly.

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
