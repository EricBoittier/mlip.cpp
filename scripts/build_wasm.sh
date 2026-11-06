#!/usr/bin/env bash
set -e

# Build mlipcpp for WebAssembly using Emscripten
# Requires: Emscripten SDK installed and activated (source emsdk_env.sh)
#
# Options:
#   --webgpu     Enable WebGPU backend via emdawnwebgpu
#   --asyncify   Use ASYNCIFY instead of JSPI (broader browser compat, slower)
#   --threads    pthread build (needs a cross-origin isolated page; set Module.ENV.MLIPCPP_NUM_THREADS)

BUILD_DIR="${BUILD_DIR:-wasm}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
USE_WEBGPU=OFF
USE_ASYNCIFY=OFF
USE_THREADS=OFF

for arg in "$@"; do
    case "$arg" in
        --webgpu)   USE_WEBGPU=ON ;;
        --asyncify) USE_ASYNCIFY=ON ;;
        --threads)  USE_THREADS=ON ;;
        -h|--help)
            sed -n '4,11p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

if ! command -v emcmake &> /dev/null; then
    echo "Error: emcmake not found. Please install and activate Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk && ./emsdk install latest && ./emsdk activate latest"
    echo "  source emsdk_env.sh"
    exit 1
fi

echo "=== Building mlipcpp for WebAssembly ==="
echo "Build directory: ${BUILD_DIR}"
echo "Build type:      ${BUILD_TYPE}"
echo "WebGPU:          ${USE_WEBGPU}"
echo "Async strategy:  $([ $USE_ASYNCIFY = ON ] && echo ASYNCIFY || echo JSPI)"

emcmake cmake . -B"${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DMLIPCPP_USE_WEBGPU=${USE_WEBGPU} \
    -DMLIPCPP_WASM_ASYNCIFY=${USE_ASYNCIFY} \
    -DMLIPCPP_WASM_THREADS=${USE_THREADS} \
    -GNinja

cmake --build "${BUILD_DIR}" --target mlipcpp_wasm

echo ""
echo "=== Build complete ==="
echo "Output files:"
echo "  ${BUILD_DIR}/bin/mlipcpp_wasm.js"
