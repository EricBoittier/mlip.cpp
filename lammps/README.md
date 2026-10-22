# pair_style mlipcpp for LAMMPS

mlip.cpp GGUF models (PET-MAD and friends) in LAMMPS: package `ML-MLIPCPP`,
compiled into LAMMPS like pet-kokkos's `ML-PET`, talking to mlip.cpp through
its C API (`include/mlipcpp/mlipcpp.h`) in one shared library,
`libmlipcpp_c.so`.

## Build

```sh
# 1. mlip.cpp, with the shared C API
cmake -S mlip.cpp -B mlip.cpp/build -DCMAKE_BUILD_TYPE=Release -DMLIPCPP_USE_CUDA=ON -DMLIPCPP_BUILD_LAMMPS=ON
cmake --build mlip.cpp/build --target mlipcpp_c

# 2. the package into a LAMMPS tree (CMake builds only; leaves ML-PET and others alone)
mlip.cpp/lammps/patch_lammps.sh /path/to/lammps          # or --link to symlink

# 3. LAMMPS, with or without KOKKOS
cmake -S lammps/cmake -B build -D PKG_ML-MLIPCPP=ON [-D MLIPCPP_LIBRARY=/path/to/libmlipcpp_c.so] ...
```

`MLIPCPP_DIR` (the mlip.cpp tree, set by `patch_lammps.sh`) and
`MLIPCPP_LIBRARY` (default `MLIPCPP_DIR/build/lib/libmlipcpp_c.so`) locate it.
mlip.cpp keeps its own compiler settings (C++20, its CUDA and ggml); LAMMPS
compiles the pair style with its own (nvcc_wrapper and C++17 under KOKKOS)
against the C header only.

## Run

```
units       metal
pair_style  mlipcpp pet-mad-s-v1.6.0.gguf [backend auto|cpu|cuda|...]
pair_coeff  * * Pd                     # an element per atom type
```

One MPI rank. Each step the owned atoms, LAMMPS's box (as the cell) and its
periodicity go to mlip.cpp, which builds its own neighbour list (on the GPU,
reused between steps with a Verlet skin, `MLIPCPP_NL_SKIN`), so the pair style
requests no LAMMPS list. Energy, forces and the global virial (from mlip.cpp's
stress for a periodic box, `sum x (x) F` otherwise) come back; per-atom energy
and virial are not available. mlip.cpp's switches (`MLIPCPP_RAGGED=1`,
`MLIPCPP_TF32=1`, ...) are read from the environment. Under the KOKKOS package
(`-k on g 1 -sf kk`) it runs as a host pair style: positions go to the host and
forces back each step.
