#pragma once
#include <cstring>
#include <ggml.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mlipcpp::runtime {
// Forward directional differentiation. Tangents are ordinary differentiable
// ggml expressions: reversing the result gives mixed energy derivatives.
// The fused SiLU/softmax derivative operations have explicit backward rules.
// Missing tangents mean zero.
class GraphJVP {
  ggml_context *c;
  std::unordered_map<ggml_tensor *, ggml_tensor *> tangents;
  ggml_tensor *add(ggml_tensor *a, ggml_tensor *b) {
    return a && b ? ggml_add(c, a, b) : a ? a : b;
  }
  ggml_tensor *mul(ggml_tensor *a, ggml_tensor *b) {
    return a && b ? ggml_mul(c, a, b) : nullptr;
  }
  ggml_tensor *neg(ggml_tensor *a) { return a ? ggml_neg(c, a) : nullptr; }
  ggml_tensor *zero(ggml_tensor *a) {
    // scale(x,0) is unsafe for masked -infinity values.
    return ggml_fill(c, ggml_dup_tensor(c, a), 0);
  }

public:
  explicit GraphJVP(ggml_context *ctx) : c(ctx) {}
  void seed(ggml_tensor *x, ggml_tensor *v) { tangents[x] = v; }
  ggml_tensor *operator()(ggml_tensor *x) {
    if (!x)
      return nullptr;
    if (auto it = tangents.find(x); it != tangents.end())
      return it->second;
    auto *a = x->src[0], *b = x->src[1];
    auto *u = (*this)(a), *v = (*this)(b);
    ggml_tensor *t = nullptr;
    if (!u && !v)
      return tangents[x] = nullptr;
    float f;
    std::memcpy(&f, x->op_params, sizeof(f));
    switch (x->op) {
    case GGML_OP_ADD:
      t = add(u, v);
      break;
    case GGML_OP_ADD1:
      t = u && v ? ggml_add(c, u, v) : u ? u : ggml_repeat(c, v, x);
      break;
    case GGML_OP_SUB:
      t = add(u, neg(v));
      break;
    case GGML_OP_MUL:
      t = add(mul(u, b), mul(a, v));
      break;
    case GGML_OP_DIV:
      t = add(u ? ggml_div(c, u, b) : nullptr,
              v ? neg(ggml_div(c, mul(x, v), b)) : nullptr);
      break;
    case GGML_OP_CLAMP: {
      float hi;
      std::memcpy(&hi, x->op_params + 1, sizeof(hi));
      auto *lo_mask = ggml_step(c, ggml_scale_bias(c, a, 1, -f));
      auto *hi_mask = ggml_step(c, ggml_scale_bias(c, a, -1, hi));
      t = mul(u, mul(lo_mask, hi_mask));
      break;
    }
    case GGML_OP_SCALE:
      t = u ? ggml_scale(c, u, f) : nullptr;
      break;
    case GGML_OP_SQR:
      t = ggml_scale(c, mul(a, u), 2);
      break;
    case GGML_OP_SQRT:
      t = ggml_scale(c, ggml_div(c, u, x), .5f);
      break;
    case GGML_OP_LOG:
      t = ggml_div(c, u, a);
      break;
    case GGML_OP_SIN:
      t = mul(ggml_cos(c, a), u);
      break;
    case GGML_OP_COS:
      t = neg(mul(ggml_sin(c, a), u));
      break;
    case GGML_OP_SUM:
      t = ggml_sum(c, u);
      break;
    case GGML_OP_SUM_ROWS:
      t = ggml_sum_rows(c, u);
      break;
    case GGML_OP_MEAN:
      t = ggml_mean(c, u);
      break;
    case GGML_OP_REPEAT:
      t = u ? ggml_repeat(c, u, x) : nullptr;
      break;
    case GGML_OP_REPEAT_BACK:
      t = u ? ggml_repeat_back(c, u, x) : nullptr;
      break;
    case GGML_OP_MUL_MAT:
      t = add(u ? ggml_mul_mat(c, u, b) : nullptr,
              v ? ggml_mul_mat(c, a, v) : nullptr);
      break;
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_CPY:
      t = u ? ggml_reshape(c, ggml_cont(c, u), x) : nullptr;
      break;
    case GGML_OP_RESHAPE:
      t = ggml_reshape(c, ggml_cont(c, u), x);
      break;
    case GGML_OP_VIEW: {
      size_t offset;
      std::memcpy(&offset, x->op_params, sizeof(offset));
      // Preserve source layout: a tangent may have been formed contiguously.
      if (!ggml_is_contiguous(a))
        throw std::runtime_error("JVP view of noncontiguous source");
      t = ggml_view_4d(c, ggml_cont(c, u), x->ne[0], x->ne[1], x->ne[2],
                       x->ne[3], x->nb[1], x->nb[2], x->nb[3], offset);
      break;
    }
    case GGML_OP_PERMUTE:
      t = ggml_permute(c, u, x->op_params[0], x->op_params[1], x->op_params[2],
                       x->op_params[3]);
      break;
    case GGML_OP_TRANSPOSE:
      t = ggml_transpose(c, u);
      break;
    case GGML_OP_GET_ROWS:
      t = u ? ggml_get_rows(c, u, b) : nullptr;
      break;
    case GGML_OP_CONCAT:
      t = ggml_concat(c, u ? u : zero(a), v ? v : zero(b), x->op_params[0]);
      break;
    case GGML_OP_SOFT_MAX: {
      float bias;
      std::memcpy(&bias, x->op_params + 1, sizeof(bias));
      if (bias != 0 || v)
        throw std::runtime_error("JVP softmax with differentiable mask/ALiBi");
      t = ggml_soft_max_ext_back(c, u, x, f, 0);
      break;
    }
    case GGML_OP_UNARY:
      switch (ggml_get_unary_op(x)) {
      case GGML_UNARY_OP_NEG:
        t = neg(u);
        break;
      case GGML_UNARY_OP_STEP:
      case GGML_UNARY_OP_SGN:
        break;
      case GGML_UNARY_OP_ABS:
        t = mul(ggml_sgn(c, a), u);
        break;
      case GGML_UNARY_OP_RELU:
        t = mul(ggml_step(c, a), u);
        break;
      case GGML_UNARY_OP_EXP:
        t = mul(x, u);
        break;
      case GGML_UNARY_OP_SIGMOID:
        t = mul(mul(x, ggml_scale_bias(c, x, -1, 1)), u);
        break;
      case GGML_UNARY_OP_SILU: {
        t = ggml_silu_back(c, ggml_cont(c, u), ggml_cont(c, a));
        break;
      }
      case GGML_UNARY_OP_TANH:
        t = mul(ggml_scale_bias(c, ggml_sqr(c, x), -1, 1), u);
        break;
      default:
        throw std::runtime_error("Unsupported JVP unary: " +
                                 std::string(ggml_op_desc(x)));
      }
      break;
    default:
      throw std::runtime_error("Unsupported JVP op: " +
                               std::string(ggml_op_desc(x)));
    }
    if (t && !ggml_are_same_shape(t, x))
      t = ggml_repeat(c, t, x);
    return tangents[x] = t;
  }
};
} // namespace mlipcpp::runtime
