#pragma once

// Per-model CUDA matmul precision in one process (MatmulPrecision, model.h).
//
// ggml reads GGML_CUDA_NO_TF32 once, at the first CUDA product, so the process
// decides between full FP32 (GGML_CUDA_NO_TF32 set) and
// TF32 once: commit(), called when the first graph is built. It runs TF32 when
// MLIPCPP_TF32 is set or some model asked for TF32 (request_tf32) before then.
// In a TF32 process a model in FP32 has its graphs tagged (tag_graph): every
// MUL_MAT and OUT_PROD GGML_PREC_F32, which the patched ggml runs in full FP32
// (cuBLAS default math, no TF32 mmf / cuBLASLt kernels), and the ragged PET
// attention without its TF32 tensor-core kernels.

#include <mlipcpp/model.h>

struct ggml_cgraph;
typedef struct ggml_backend *ggml_backend_t;

namespace mlipcpp::matmul_precision {

/// A model asks for TF32: the process will run TF32 if not committed yet;
/// throws std::runtime_error if it committed to full FP32
void request_tf32();

/// The process decision (idempotent): sets GGML_CUDA_NO_TF32 unless MLIPCPP_TF32
/// is set or TF32 was requested. True if the CUDA products run in TF32.
bool commit();

/// Whether the decision was made, and (if so) whether the process runs TF32
bool committed();

/// Whether a model of this precision runs TF32 products in a TF32 process
bool model_tf32(MatmulPrecision precision);

/// Tags every product of `graph` GGML_PREC_F32 if the model runs in FP32 while
/// the process runs TF32 on CUDA (`backend`); otherwise leaves the graph as
/// built. Commits for Default and TF32 models only if `commit_now`. Returns the
/// number of nodes tagged.
int tag_graph(ggml_cgraph *graph, MatmulPrecision precision, ggml_backend_t backend,
              bool commit_now = true);

} // namespace mlipcpp::matmul_precision
