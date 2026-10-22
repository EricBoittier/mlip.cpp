# ML-MLIPCPP: pair_style mlipcpp, mlip.cpp GGUF models through mlip.cpp's C API.
# Put in place by mlip.cpp/lammps/patch_lammps.sh.
#
#   MLIPCPP_DIR      the mlip.cpp source tree (default: the one that patched this tree)
#   MLIPCPP_LIBRARY  libmlipcpp_c.so, from an mlip.cpp build configured with
#                    -D MLIPCPP_BUILD_LAMMPS=ON (default: MLIPCPP_DIR/build/lib)
#
# mlip.cpp is built on its own (C++20, its own CUDA and ggml): LAMMPS compiles
# the pair style with its compiler and standard (nvcc_wrapper and C++17 under
# KOKKOS) against the C header only, and links the one shared library.
set(MLIPCPP_DIR "@MLIPCPP_DIR@" CACHE PATH "mlip.cpp source tree")
if(NOT EXISTS "${MLIPCPP_DIR}/include/mlipcpp/mlipcpp.h")
  message(FATAL_ERROR "ML-MLIPCPP: MLIPCPP_DIR (${MLIPCPP_DIR}) is not an mlip.cpp source tree")
endif()
find_library(MLIPCPP_LIBRARY mlipcpp_c HINTS "${MLIPCPP_DIR}/build/lib" DOC "libmlipcpp_c.so (mlip.cpp, -D MLIPCPP_BUILD_LAMMPS=ON)")
if(NOT MLIPCPP_LIBRARY)
  message(FATAL_ERROR "ML-MLIPCPP: libmlipcpp_c not found: build mlip.cpp with -D MLIPCPP_BUILD_LAMMPS=ON "
                      "and pass -D MLIPCPP_LIBRARY=/path/to/libmlipcpp_c.so")
endif()
message(STATUS "ML-MLIPCPP: ${MLIPCPP_LIBRARY}")
target_include_directories(lammps PRIVATE "${MLIPCPP_DIR}/include")
target_link_libraries(lammps PRIVATE "${MLIPCPP_LIBRARY}")
