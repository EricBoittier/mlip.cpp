# metatomic plugin

`mlipcpp-metatomic.so` exposes GGUF models to any engine using the
[metatomic](https://docs.metatensor.org/metatomic/) C API (`metatomic-core`):
the engine loads the plugin, then loads a `*.gguf` file as a regular metatomic
model. No PyTorch is involved; the model runs on mlipcpp/ggml.

## Build

The plugin needs an installed `metatomic-core` (C/C++ library and CMake
package, which itself needs `metatensor-core`):

```bash
cmake -S . -B build -DMLIPCPP_BUILD_METATOMIC=ON \
    -DCMAKE_PREFIX_PATH="<metatomic prefix>;<metatensor prefix>"
cmake --build build --target mlipcpp-metatomic metatomic_plugin
```

`libmlipcpp` and ggml are linked statically into the plugin (the option turns
on position independent code), and only the metatomic entry point
`mta_plugin_init` is exported, so the plugin does not clash with another copy
of ggml in the same process. GPU backends follow the usual
`-DMLIPCPP_USE_CUDA=ON` etc.

## Use

```c
mta_load_plugin("build/lib/mlipcpp-metatomic.so");

mta_model_t model = {0};
mta_load_model("ft-huber-3e-5.gguf", "{\"backend\": \"cpu\"}", NULL, &model);

mta_execute_model(model, systems, n_systems, NULL, requested_outputs_json,
                  /*check_consistency*/ false, outputs, n_outputs);
```

`examples/metatomic_plugin.c` is a complete program doing this for a cluster,
a slab and a bulk cell, checking gradients against finite differences:

```bash
./build/bin/metatomic_plugin build/lib/mlipcpp-metatomic.so model.gguf '{"backend": "cpu"}'
```

From Python, with the `metatomic-core` package:

```python
import metatomic as mta

mta.load_plugin("build/lib/mlipcpp-metatomic.so")
model = mta.load_model("model.gguf", {"dtype": "float64"})
```

The plugin answers to any `load_from` ending in `.gguf` (plugin name
`mlipcpp`).

### Options

All optional, given as a JSON object of strings:

| option | values | default |
| --- | --- | --- |
| `backend` | `auto`, `cpu`, `cuda`, `hip`, `metal`, `vulkan`, `sycl`, ... | `auto` |
| `dtype` | `float64`, `float32`: dtype of the systems and outputs | `float64` |
| `interaction_range` | Angstrom | cutoff x number of message passing layers |

The computation itself is float32 whatever `dtype` is.

On CPU, ggml uses 4 threads unless `MLIPCPP_NUM_THREADS` is set in the
environment (this applies to every mlipcpp entry point, not only the plugin).

### Outputs

| output | samples | gradients |
| --- | --- | --- |
| `energy` (eV) | per system | `positions`, `strain` |
| `non_conservative_force` (eV/Angstrom) | per atom | none |

`non_conservative_force` is only advertised for models with a direct force
head (`pet.nc_forces` in the GGUF metadata). Per-atom energies are not
available, and `selected_atoms` must select every atom for the energy (it is
used as a filter for `non_conservative_force`).

The model builds its own neighbor list and requests no pair list from the
engine. Systems must be on CPU; units are converted by `mta_execute_model`
(the model works in Angstrom and eV). Non-periodic directions (zero cell
vectors in metatomic) are supported, including mixed periodicity such as
slabs.
