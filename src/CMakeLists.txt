# =============================================================================
# mlipcpp library sources
# =============================================================================

# Core components
set(MLIPCPP_CORE_SOURCES
    core/coulomb.cpp
    core/p3m.cpp
    core/system.cpp
    core/neighbor_list.cpp
    core/gguf_loader.cpp
    core/backend.cpp
    core/ggml_utils.cpp
    core/timer.cpp
)

# Model implementations
set(MLIPCPP_MODEL_SOURCES
    models/model.cpp
    models/pet/pet.cpp
    models/pet/pet_batch.cpp
    models/pet/pet_layers.cpp
    models/plato/plato.cpp
)

# I/O and utilities
set(MLIPCPP_IO_SOURCES
    io/xyz.cpp
)

# Runtime (graph interpreter)
set(MLIPCPP_RUNTIME_SOURCES
    runtime/graph_ir.cpp
    runtime/graph_interpreter.cpp
    runtime/graph_model.cpp
    runtime/graph_model_cached.cpp
    runtime/pet_ragged.cpp
    runtime/plato_graph.cpp
)

# Utilities
set(MLIPCPP_UTIL_SOURCES
    ggml_attention.cpp
)

# API bindings
set(MLIPCPP_API_SOURCES
    api/c/mlipcpp_api.cpp
    api/cpp/mlipcpp_cpp.cpp
)

# Combine all sources (prepend src/ directory)
set(MLIPCPP_SOURCES)
foreach(src IN LISTS
    MLIPCPP_CORE_SOURCES
    MLIPCPP_MODEL_SOURCES
    MLIPCPP_IO_SOURCES
    MLIPCPP_RUNTIME_SOURCES
    MLIPCPP_UTIL_SOURCES
    MLIPCPP_API_SOURCES)
    list(APPEND MLIPCPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${src})
endforeach()

# Create library target
add_library(mlipcpp ${MLIPCPP_SOURCES})
add_library(mlipcpp::mlipcpp ALIAS mlipcpp)

if(APPLE)
    option(MLIPCPP_ACCELERATE_FFT "Use Accelerate for P3M FFTs on Apple platforms" ON)
    if(MLIPCPP_ACCELERATE_FFT)
        find_library(MLIPCPP_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        target_link_libraries(mlipcpp PRIVATE ${MLIPCPP_ACCELERATE_FRAMEWORK})
        target_compile_definitions(mlipcpp PRIVATE MLIPCPP_ACCELERATE_FFT)
    endif()
endif()

target_include_directories(mlipcpp
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ggml_SOURCE_DIR}/include
)

target_link_libraries(mlipcpp
    PRIVATE
        ggml
        nlohmann_json::nlohmann_json
        $<$<NOT:$<BOOL:${EMSCRIPTEN}>>:fmt::fmt>
)

if(TARGET vesin_objects)
    target_link_libraries(mlipcpp PRIVATE vesin_objects)
    target_compile_definitions(mlipcpp PRIVATE MLIPCPP_HAVE_VESIN)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(mlipcpp PRIVATE OpenMP::OpenMP_CXX)
endif()
# CUDA: GraphModel::predict_cached builds the model inputs and gathers the
# forces on the device, with the MD engine's kernels
if(MLIPCPP_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(mlipcpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/md/md_kernels.cu)
    set_target_properties(mlipcpp PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(mlipcpp PRIVATE CUDA::cudart)
    target_compile_definitions(mlipcpp PRIVATE MLIPCPP_HAVE_CUDA_INPUTS)
    # GraphModel::predict_device_step orders the backend's stream after the caller's
    # through ggml backend events (their cudaEvent_t: ggml-backend-impl.h)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/runtime/graph_model_cached.cpp
                                PROPERTIES INCLUDE_DIRECTORIES ${ggml_SOURCE_DIR}/src)
endif()

target_compile_options(mlipcpp PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

set_target_properties(mlipcpp PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME mlipcpp
)

# =============================================================================
# metatomic plugin (optional)
# =============================================================================
if(MLIPCPP_BUILD_METATOMIC)
    add_library(mlipcpp-metatomic SHARED api/metatomic/mlipcpp_metatomic.cpp)
    target_link_libraries(mlipcpp-metatomic PRIVATE mlipcpp ggml metatomic)
    target_include_directories(mlipcpp-metatomic PRIVATE ${ggml_SOURCE_DIR}/include)
    # only export the plugin entry point, keep our ggml copy private
    set_target_properties(mlipcpp-metatomic PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    if(UNIX AND NOT APPLE)
        target_link_options(mlipcpp-metatomic PRIVATE -Wl,--exclude-libs,ALL)
    endif()

    include(GNUInstallDirs)
    install(TARGETS mlipcpp-metatomic
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# =============================================================================
# Shared C API for LAMMPS (optional)
# =============================================================================
# pair_style mlipcpp (lammps/) is compiled into LAMMPS, with LAMMPS's compiler
# and C++ standard (nvcc_wrapper, C++17 under KOKKOS), so it talks to mlip.cpp
# through the C header only: one shared library with libmlipcpp and ggml inside
# and only the mlipcpp_* C symbols exported. The C API source is compiled into
# it directly, so the static library's copy is never pulled in.
if(MLIPCPP_BUILD_LAMMPS)
    add_library(mlipcpp_c SHARED api/c/mlipcpp_api.cpp)
    target_include_directories(mlipcpp_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${ggml_SOURCE_DIR}/include)
    target_link_libraries(mlipcpp_c PRIVATE mlipcpp ggml nlohmann_json::nlohmann_json fmt::fmt)
    target_include_directories(mlipcpp_c PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    if(UNIX AND NOT APPLE)
        target_link_options(mlipcpp_c PRIVATE
            -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/api/c/mlipcpp_c.map)
        set_target_properties(mlipcpp_c PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/api/c/mlipcpp_c.map)
    endif()
    include(GNUInstallDirs)
    install(TARGETS mlipcpp_c
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# =============================================================================
# Fortran bindings (optional)
# =============================================================================
if(MLIPCPP_BUILD_FORTRAN)
    add_library(mlipcpp_fortran
        api/fortran/mlipcpp.f90
    )
    add_library(mlipcpp::fortran ALIAS mlipcpp_fortran)

    target_link_libraries(mlipcpp_fortran PUBLIC mlipcpp)

    set_target_properties(mlipcpp_fortran PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/fortran_modules
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME fortran
    )

    target_include_directories(mlipcpp_fortran
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/fortran_modules>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/mlipcpp/fortran>
    )
endif()

# =============================================================================
# PET training (mlipcpp-train)
# =============================================================================
if(MLIPCPP_BUILD_TRAINING AND UNIX AND NOT EMSCRIPTEN)
    find_package(Threads REQUIRED)
    add_library(mlipcpp_train STATIC
        train/conservative.cpp
        train/dataset.cpp
        train/batch.cpp
        train/trainer.cpp
    )
    target_include_directories(mlipcpp_train PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ggml_SOURCE_DIR}/include
                               PRIVATE ${ggml_SOURCE_DIR}/src) # ggml-impl.h: ggml_graph_view
    target_link_libraries(mlipcpp_train PUBLIC mlipcpp ggml fmt::fmt nlohmann_json::nlohmann_json Threads::Threads)
    target_compile_options(mlipcpp_train PRIVATE -Wall -Wextra)
    # NCCL (e.g. `module load nccl`): the multi-GPU gradient all-reduce with a
    # replicated optimizer step (Optimizer, --allreduce)
    if(MLIPCPP_USE_CUDA)
        find_path(NCCL_INCLUDE_DIR nccl.h HINTS $ENV{NCCL_ROOT}/include $ENV{NCCL_HOME}/include)
        find_library(NCCL_LIBRARY nccl HINTS $ENV{NCCL_ROOT}/lib $ENV{NCCL_HOME}/lib)
        if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
            find_package(CUDAToolkit REQUIRED)
            target_compile_definitions(mlipcpp_train PUBLIC MLIPCPP_NCCL)
            target_include_directories(mlipcpp_train PRIVATE ${NCCL_INCLUDE_DIR})
            target_link_libraries(mlipcpp_train PUBLIC ${NCCL_LIBRARY} CUDA::cudart)
            message(STATUS "mlipcpp-train: NCCL all-reduce (${NCCL_LIBRARY})")
        endif()
    endif()

    add_executable(mlipcpp-train bin/mlipcpp_train.cpp)
    add_executable(mlipcpp-train-lr bin/mlipcpp_train_lr.cpp)
    target_link_libraries(mlipcpp-train-lr PRIVATE mlipcpp_train)
    target_link_libraries(mlipcpp-train PRIVATE mlipcpp_train)
    target_link_options(mlipcpp-train PRIVATE -rdynamic) # names in crash backtraces

    add_executable(mlipcpp-op-test bin/mlipcpp_op_test.cpp)
    target_link_libraries(mlipcpp-op-test PRIVATE mlipcpp_train)

    add_executable(mlipcpp-attn-test bin/mlipcpp_attn_test.cpp)
    target_link_libraries(mlipcpp-attn-test PRIVATE mlipcpp_train)

    add_executable(mlipcpp-wgrad-test bin/mlipcpp_wgrad_test.cpp)
    target_link_libraries(mlipcpp-wgrad-test PRIVATE mlipcpp_train)

    # Plato-PET-LR parity against PyTorch references and timing
    add_executable(mlipcpp-plato bin/mlipcpp_plato.cpp)
    target_link_libraries(mlipcpp-plato PRIVATE mlipcpp_train)
    add_executable(mlipcpp-train-plato bin/mlipcpp_train_plato.cpp)
    target_link_libraries(mlipcpp-train-plato PRIVATE mlipcpp_train)
    if(MLIPCPP_USE_CUDA AND NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        target_include_directories(mlipcpp-train-plato PRIVATE ${NCCL_INCLUDE_DIR})
        target_link_libraries(mlipcpp-train-plato PRIVATE CUDA::cudart)
    endif()

    add_executable(mlipcpp-predict bin/mlipcpp_predict.cpp)
    target_link_libraries(mlipcpp-predict PRIVATE mlipcpp_train)

    add_executable(mlipcpp-jvp-check bin/mlipcpp_jvp_check.cpp)
    target_link_libraries(mlipcpp-jvp-check PRIVATE mlipcpp_train)
    target_link_options(mlipcpp-jvp-check PRIVATE -rdynamic)

    add_executable(mlipcpp-train-check bin/mlipcpp_train_check.cpp)
    target_link_libraries(mlipcpp-train-check PRIVATE mlipcpp_train)
    target_link_options(mlipcpp-train-check PRIVATE -rdynamic)

    # GPU molecular dynamics (mlipcpp-md): CUDA kernels next to the ggml graph
    if(MLIPCPP_USE_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        add_library(mlipcpp_md STATIC md/md_engine.cpp) # md/md_kernels.cu: in mlipcpp
        target_include_directories(mlipcpp_md PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ggml_SOURCE_DIR}/include
                                   ${ggml_SOURCE_DIR}/src) # ggml-impl.h: ggml_graph_view
        target_link_libraries(mlipcpp_md PUBLIC mlipcpp ggml CUDA::cudart)
        set_target_properties(mlipcpp_md PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_compile_options(mlipcpp_md PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

        add_executable(mlipcpp-md bin/mlipcpp_md.cpp)
        target_link_libraries(mlipcpp-md PRIVATE mlipcpp_md mlipcpp_train)
        target_link_options(mlipcpp-md PRIVATE -rdynamic)
    endif()
endif()
