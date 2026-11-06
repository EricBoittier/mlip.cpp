#include "matmul_precision.h"

#include "log.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace mlipcpp {

void Model::set_matmul_precision(MatmulPrecision precision) {
  if (precision == MatmulPrecision::TF32) matmul_precision::request_tf32();
  matmul_precision_ = precision;
}

namespace matmul_precision {
namespace {

struct State {
  std::mutex mutex;
  bool requested = false; // some model asked for TF32
  bool committed = false;
  bool tf32 = false; // the committed decision
};

State &state() {
  static State s;
  return s;
}

bool cuda_available() {
#ifdef __EMSCRIPTEN__
  return false;
#else
  return ggml_backend_reg_by_name("CUDA") != nullptr;
#endif
}

bool is_cuda(ggml_backend_t backend) {
  return backend && std::strncmp(ggml_backend_name(backend), "CUDA", 4) == 0;
}

} // namespace

void request_tf32() {
  if (!cuda_available()) return; // TF32 is a CUDA setting: nothing to decide
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.committed && !s.tf32)
    throw std::runtime_error(
        "mlipcpp: a model asks for TF32 matmuls, but the process already runs its CUDA "
        "matmuls in full FP32 (GGML_CUDA_NO_TF32, read once by ggml at the first product). "
        "Set the precision of every model (load them) before the first predict, or set "
        "MLIPCPP_TF32=1.");
  s.requested = true;
}

bool commit() {
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!s.committed) {
    // full FP32 unless asked for TF32 (an explicit GGML_CUDA_NO_TF32 is kept)
    if (!std::getenv("MLIPCPP_TF32") && !s.requested) setenv("GGML_CUDA_NO_TF32", "1", 0);
    s.tf32 = std::getenv("GGML_CUDA_NO_TF32") == nullptr;
    s.committed = true;
    if (s.requested && !s.tf32)
      log::warn("mlipcpp: TF32 requested by a model, but GGML_CUDA_NO_TF32 is set: every "
                "CUDA matmul runs in full FP32");
  }
  return s.tf32;
}

bool committed() {
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.committed;
}

bool model_tf32(MatmulPrecision precision) {
  switch (precision) {
  case MatmulPrecision::TF32: return true;
  case MatmulPrecision::FP32: return false;
  default: return std::getenv("MLIPCPP_TF32") != nullptr;
  }
}

int tag_graph(ggml_cgraph *graph, MatmulPrecision precision, ggml_backend_t backend,
              bool commit_now) {
  if (!graph || !is_cuda(backend)) return 0;
  bool process_tf32;
  if (commit_now) {
    process_tf32 = commit();
  } else if (committed()) {
    process_tf32 = commit(); // returns the decision
  } else if (precision == MatmulPrecision::FP32) {
    process_tf32 = std::getenv("GGML_CUDA_NO_TF32") == nullptr; // e.g. training
  } else {
    return 0;
  }
  if (!process_tf32 || model_tf32(precision)) return 0;
  int tagged = 0;
  for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
    ggml_tensor *t = ggml_graph_node(graph, i);
    switch (t->op) {
    case GGML_OP_MUL_MAT:
      ggml_mul_mat_set_prec(t, GGML_PREC_F32);
      ggml_mul_mat_set_grad_prec(t, GGML_PREC_F32);
      tagged++;
      break;
    case GGML_OP_OUT_PROD: // the gradients of products (GGML_PREC_F32: full FP32)
      t->op_params[0] = GGML_PREC_F32;
      tagged++;
      break;
    case GGML_OP_PET_ATTN_RAGGED:
    case GGML_OP_PET_ATTN_RAGGED_BACK: // no TF32 tensor-core kernels
      t->op_params[3] = GGML_PREC_F32;
      tagged++;
      break;
    default: break;
    }
  }
  return tagged;
}

} // namespace matmul_precision
} // namespace mlipcpp
