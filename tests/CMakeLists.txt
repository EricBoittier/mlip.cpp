# Test executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# PET comprehensive test
add_executable(test_pet
    test_pet.cpp
)

target_link_libraries(test_pet
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_pet
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/models/pet
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

# GGML layout test (standalone, no Catch2)
add_executable(test_ggml_layout
    test_ggml_layout.cpp
)

target_link_libraries(test_ggml_layout
    PRIVATE
        ggml
        fmt::fmt
)

target_include_directories(test_ggml_layout
    PRIVATE
        ${ggml_SOURCE_DIR}/include
)

# PET gradient test
add_executable(test_pet_gradients
    test_pet_gradients.cpp
)

target_link_libraries(test_pet_gradients
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_pet_gradients
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/models/pet
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

# IO tests (no model file needed)
add_executable(test_io
    test_io.cpp
)

target_link_libraries(test_io
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        fmt::fmt
)

# Neighbor list tests (no model file needed)
add_executable(test_neighbor_list
    test_neighbor_list.cpp
)

target_link_libraries(test_neighbor_list
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        fmt::fmt
)

# System/Cell tests (no model file needed)
add_executable(test_system
    test_system.cpp
)

target_link_libraries(test_system
    PRIVATE
        mlipcpp
        fmt::fmt
        Catch2::Catch2WithMain
)

# Graph interpreter tests
add_executable(test_graph_interpreter
    test_graph_interpreter.cpp
)

target_link_libraries(test_graph_interpreter
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_graph_interpreter
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

# Plato LES by PME: ggml ops against the direct Ewald sum
add_executable(test_plato_pme
    test_plato_pme.cpp
)

target_link_libraries(test_plato_pme
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_plato_pme
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

# Auto-export vs manual comparison tests
add_executable(test_auto_vs_manual
    test_auto_vs_manual.cpp
)

target_link_libraries(test_auto_vs_manual
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_auto_vs_manual
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/models/pet
)

# Register with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(test_pet)
catch_discover_tests(test_pet_gradients)
catch_discover_tests(test_io)
catch_discover_tests(test_neighbor_list)
catch_discover_tests(test_system)
catch_discover_tests(test_plato_pme)
catch_discover_tests(test_graph_interpreter)
catch_discover_tests(test_auto_vs_manual)

# Full export integration test
add_executable(test_full_export
    test_full_export.cpp
)

target_link_libraries(test_full_export
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_full_export
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

catch_discover_tests(test_full_export)

# GraphModel tests
add_executable(test_graph_model
    test_graph_model.cpp
)

target_link_libraries(test_graph_model
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
        ggml
        fmt::fmt
)

target_include_directories(test_graph_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
)

catch_discover_tests(test_graph_model)

# Backend-selection unit tests (use only the public C API)
add_executable(test_backend_selection
    test_backend_selection.cpp
)

target_link_libraries(test_backend_selection
    PRIVATE
        mlipcpp
        Catch2::Catch2WithMain
)

target_compile_definitions(test_backend_selection
    PRIVATE
        MLIPCPP_TEST_MODEL_DEFAULT="${CMAKE_BINARY_DIR}/tests/gguf/pet-auto.gguf"
)

catch_discover_tests(test_backend_selection)

add_executable(test_coulomb test_coulomb.cpp)
target_link_libraries(test_coulomb PRIVATE mlipcpp ggml Catch2::Catch2WithMain nlohmann_json::nlohmann_json)
target_include_directories(test_coulomb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
catch_discover_tests(test_coulomb)

if(MLIPCPP_BUILD_TRAINING)
    add_executable(test_conservative test_conservative.cpp)
    target_link_libraries(test_conservative PRIVATE mlipcpp_train Catch2::Catch2WithMain)
    catch_discover_tests(test_conservative)
endif()

# Charge / spin conditioning inputs (no model file needed)
add_executable(test_conditioning test_conditioning.cpp)
target_link_libraries(test_conditioning PRIVATE mlipcpp Catch2::Catch2WithMain fmt::fmt)
target_include_directories(test_conditioning PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/)
catch_discover_tests(test_conditioning)

# Per-group learning-rate schedules against metatrain's golden tables
if(TARGET mlipcpp_train)
    add_executable(test_lr_schedule test_lr_schedule.cpp)
    target_link_libraries(test_lr_schedule PRIVATE mlipcpp_train Catch2::Catch2WithMain fmt::fmt)
    target_compile_definitions(test_lr_schedule PRIVATE MLIPCPP_TEST_RESOURCES="${CMAKE_CURRENT_SOURCE_DIR}/resources")
    catch_discover_tests(test_lr_schedule)
endif()

# PET's "grid" adaptive cutoff against metatrain's golden values (and, with
# MLIPCPP_TEST_GRID_GGUF, a grid model's cached path)
add_executable(test_adaptive_grid test_adaptive_grid.cpp)
target_link_libraries(test_adaptive_grid PRIVATE mlipcpp ggml fmt::fmt Catch2::Catch2WithMain nlohmann_json::nlohmann_json)
target_include_directories(test_adaptive_grid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src/)
target_compile_definitions(test_adaptive_grid PRIVATE MLIPCPP_TEST_RESOURCES="${CMAKE_CURRENT_SOURCE_DIR}/resources")
catch_discover_tests(test_adaptive_grid)

# Per-model CUDA matmul precision (MLIPCPP_PRECISION_TF32 / _F32_EXACT): the same model
# in FP32 and in TF32 in one process against FP32-only and MLIPCPP_TF32=1 processes
# (tests/precision_check.py; skipped without MLIPCPP_PRECISION_TEST_MODEL or numpy)
set(MLIPCPP_PRECISION_TEST_MODEL "" CACHE FILEPATH "pet-graph GGUF for the precision_per_model test")
set(MLIPCPP_PRECISION_TEST_XYZ "${CMAKE_CURRENT_SOURCE_DIR}/../geometries/water.xyz" CACHE FILEPATH
    "structure for the precision_per_model test")
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND TARGET mlipcpp-precision-check)
    if(MLIPCPP_USE_CUDA)
        set(_prec_backend cuda)
    else()
        set(_prec_backend cpu)
    endif()
    set(_prec_model "${MLIPCPP_PRECISION_TEST_MODEL}")
    if(NOT _prec_model)
        set(_prec_model "${CMAKE_CURRENT_BINARY_DIR}/no-precision-test-model.gguf")
    endif()
    add_test(NAME precision_per_model
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/precision_check.py
                     $<TARGET_FILE:mlipcpp-precision-check> ${_prec_model} ${MLIPCPP_PRECISION_TEST_XYZ}
                     ${CMAKE_CURRENT_BINARY_DIR}/precision_check --backend ${_prec_backend} --reps 1 --warmup 0)
    set_tests_properties(precision_per_model PROPERTIES SKIP_RETURN_CODE 77)
endif()
