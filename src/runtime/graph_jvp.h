#pragma once
#include <cstdlib>
#include <cstring>
#include <ggml.h>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mlipcpp::runtime {
// Forward directional differentiation. Tangents are ordinary differentiable
// ggml expressions: reversing the result gives mixed energy derivatives.
// The fused SiLU/softmax derivative operations have explicit backward rules.
// A null tangent means zero (the value does not depend on the seeds); an op
// without a rule, or a tangent reaching an operand a rule ignores, throws.
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
  // a / b with b broadcast explicitly: ggml's DIV backward needs equal shapes
  ggml_tensor *div(ggml_tensor *a, ggml_tensor *b) {
    return ggml_div(c, a, ggml_are_same_shape(a, b) ? b : ggml_repeat(c, b, a));
  }
  ggml_tensor *zero(ggml_tensor *a) {
    // scale(x,0) is unsafe for masked -infinity values.
    return ggml_fill(c, ggml_dup_tensor(c, a), 0);
  }
  // Fused PET attention r = [o | lse] (ggml_pet_attn: qkv [D, 3, L, N], bias
  // [L, L, 1, N], r [D + H, L, N]), differentiated through the unfused
  // expression: S = scale K^T Q + bias, P = softmax(S), o = V P per head,
  //   dS = scale (dK^T Q + K^T dQ) + dbias, dP = P (dS - sum(P dS)),
  //   do = V dP + dV P, dlse = sum(P dS).
  // P is recomputed as an ordinary softmax and stored ([L, L, H, N]), so the
  // reverse pass through the tangent uses the standard backward rules.
  ggml_tensor *pet_attn(ggml_tensor *r, ggml_tensor *dqkv, ggml_tensor *dbias) {
    auto *qkv = r->src[0], *bias = r->src[1];
    if (qkv->type != GGML_TYPE_F32)
      throw std::runtime_error("JVP of PET attention needs F32 qkv");
    const int H = r->op_params[0];
    float scale;
    std::memcpy(&scale, r->op_params + 1, sizeof(scale));
    const int64_t D = qkv->ne[0], d = D / H, L = qkv->ne[2], N = qkv->ne[3];
    // slot s of a [D, 3, L, N] tensor as heads [d, L, H, N]
    auto heads = [&](ggml_tensor *t, int s) {
      t = ggml_view_4d(c, t, d, H, L, N, d * sizeof(float), t->nb[2], t->nb[3],
                       s * t->nb[1]);
      return ggml_cont(c, ggml_permute(c, t, 0, 2, 1, 3));
    };
    auto *Q = heads(qkv, 0), *K = heads(qkv, 1), *V = heads(qkv, 2);
    auto *P = ggml_soft_max(c, ggml_add(c, ggml_scale(c, ggml_mul_mat(c, K, Q), scale),
                                        bias)); // [L_k, L_q, H, N]
    auto *Vt = ggml_cont(c, ggml_transpose(c, V)); // [L_k, d, H, N]
    ggml_tensor *dS = dbias ? ggml_repeat(c, dbias, P) : nullptr, *dO = nullptr;
    if (dqkv) {
      dqkv = ggml_cont(c, dqkv);
      auto *dQ = heads(dqkv, 0), *dK = heads(dqkv, 1), *dV = heads(dqkv, 2);
      dS = add(dS, ggml_scale(c, ggml_add(c, ggml_mul_mat(c, dK, Q),
                                          ggml_mul_mat(c, K, dQ)),
                              scale));
      dO = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, dV)), P);
    }
    dO = add(dO, ggml_mul_mat(c, Vt, ggml_soft_max_ext_back(c, dS, P, 1, 0)));
    // [d, L, H, N] -> [D, L, N]; dlse [1, L, H, N] -> [H, L, N]
    dO = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, dO, 0, 2, 1, 3)), D, L, N);
    auto *dlse = ggml_sum_rows(c, ggml_mul(c, P, dS));
    dlse = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, dlse, 1, 2, 0, 3)), H, L, N);
    return ggml_concat(c, dO, dlse, 0);
  }

  // throw, or (diagnostics) count the op and treat its tangent as zero
  ggml_tensor *unsupported(const std::string &what) {
    if (!missing)
      throw std::runtime_error("JVP: " + what);
    (*missing)[what]++;
    return nullptr;
  }

public:
  explicit GraphJVP(ggml_context *ctx) : c(ctx) {}
  std::map<std::string, int> *missing = nullptr;
  // PET attention through ggml_pet_attn_jvp (nothing L x L stored), else the
  // unfused expression (pet_attn above); MLIPCPP_JVP_UNFUSED_ATTN=1: unfused
  bool fused_attention = std::getenv("MLIPCPP_JVP_UNFUSED_ATTN") == nullptr;
  void seed(ggml_tensor *x, ggml_tensor *v) { tangents[x] = v; }
  ggml_tensor *operator()(ggml_tensor *x) {
    if (!x)
      return nullptr;
    if (auto it = tangents.find(x); it != tangents.end())
      return it->second;
    auto *a = x->src[0], *b = x->src[1];
    auto *u = (*this)(a), *v = (*this)(b);
    ggml_tensor *t = nullptr;
    bool extra = false; // a tangent on src[2..]: only rules that read them may proceed
    for (int i = 2; i < GGML_MAX_SRC; i++)
      extra = (*this)(x->src[i]) || extra;
    if (!u && !v && !extra)
      return tangents[x] = nullptr;
    if (extra)
      return tangents[x] = unsupported("tangent on an extra operand of " +
                                       std::string(ggml_op_desc(x)));
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
      t = add(u ? div(u, b) : nullptr, v ? neg(div(mul(x, v), b)) : nullptr);
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
      if (!ggml_is_contiguous(a)) {
        t = unsupported("VIEW of a noncontiguous source");
        break;
      }
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
      // y = softmax(f a + mask): dy = y (w - sum(y w)), w = f u + dmask
      float bias;
      std::memcpy(&bias, x->op_params + 1, sizeof(bias));
      if (bias != 0)
        throw std::runtime_error("JVP softmax with ALiBi");
      if (!v)
        t = ggml_soft_max_ext_back(c, u, x, f, 0);
      else
        t = ggml_soft_max_ext_back(
            c, add(u ? ggml_scale(c, u, f) : nullptr, ggml_repeat(c, v, x)), x,
            1, 0);
      break;
    }
    case GGML_OP_RMS_NORM: {
      // y = a / s, s = sqrt(mean(a^2) + eps): dy = (u - y mean(y u)) / s
      const float inv = 1.0f / a->ne[0];
      auto *s = ggml_sqrt(c, ggml_scale_bias(c, ggml_sum_rows(c, ggml_sqr(c, a)), inv, f));
      auto *yu = ggml_scale(c, ggml_sum_rows(c, ggml_mul(c, x, u)), inv);
      t = div(ggml_sub(c, u, ggml_mul(c, x, yu)), s);
      break;
    }
    case GGML_OP_NORM: {
      // y = (a - mean(a)) / s, s = sqrt(var(a) + eps):
      // dy = (u - mean(u) - y mean(y u)) / s
      const float inv = 1.0f / a->ne[0];
      auto *d = ggml_sub(c, a, ggml_scale(c, ggml_sum_rows(c, a), inv));
      auto *s = ggml_sqrt(c, ggml_scale_bias(c, ggml_sum_rows(c, ggml_sqr(c, d)), inv, f));
      auto *yu = ggml_scale(c, ggml_sum_rows(c, ggml_mul(c, x, u)), inv);
      auto *um = ggml_scale(c, ggml_sum_rows(c, u), inv);
      t = div(ggml_sub(c, ggml_sub(c, u, um), ggml_mul(c, x, yu)), s);
      break;
    }
    case GGML_OP_PET_ATTN:
      if (fused_attention && a->type == GGML_TYPE_F32) {
        float scale;
        std::memcpy(&scale, x->op_params + 1, sizeof(scale));
        t = ggml_pet_attn_jvp(c, a, b, u ? ggml_cont(c, u) : nullptr,
                              v ? ggml_cont(c, v) : nullptr, x->op_params[0],
                              scale);
      } else {
        t = pet_attn(x, u, v);
      }
      break;
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
        t = unsupported(std::string("unary ") + ggml_op_desc(x));
      }
      break;
    default:
      t = unsupported(ggml_op_desc(x));
    }
    if (t && !ggml_are_same_shape(t, x))
      t = ggml_repeat(c, t, x);
    return tangents[x] = t;
  }
};
} // namespace mlipcpp::runtime
