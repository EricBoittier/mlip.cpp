// PyTorch custom operators around mlip.cpp's fused PET attention kernels.
//
//   torch.ops.mlipcpp.pet_attention(qkv, bias, heads, scale) -> (o, lse)
//     qkv [N, L, 3*D] or [N, L, 3, D] (D = heads*d, d in {8, 16, 32, 64}),
//     bias [N, L, L] or [N, 1, L, L] (bias[n][query][key], shared by heads);
//     o [N, L, D], lse [N, L, heads]; float32 (fast kernels) or float64.
//
// Autograd is implemented in C++ (works from TorchScript and from C++ hosts
// such as metatomic in LAMMPS, no Python needed):
//   backward       = pet_attention_backward (fused kernels, P recomputed from lse)
//   double backward = pet_attention_jvp (grad w.r.t. the incoming gradient) and
//                    pet_attention_jvp_backward (grad w.r.t. qkv and bias), i.e.
//   d/dx <w, J(x)^T u> = d/dx <u, J(x) w>, so the forward-mode kernels of
//   mlip.cpp give the second derivatives without storing any L x L matrix.
// Third derivatives are not implemented (an error is raised).
// Meta kernels make the ops traceable by torch.compile / FakeTensor.
#include "pet_attn.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cstdlib>
#include <tuple>

using at::Tensor;
using c10::optional;

namespace {

struct Dims {
  int64_t N;
  int64_t L, H, d, D;
};

Dims dims_of(const Tensor &qkv, const Tensor &bias, int64_t heads) {
  TORCH_CHECK(qkv.dim() == 3 || qkv.dim() == 4, "pet_attention: qkv must be [N, L, 3*D] or [N, L, 3, D]");
  TORCH_CHECK(heads > 0, "pet_attention: heads must be positive");
  const int64_t N = qkv.size(0), L = qkv.size(1);
  const int64_t D3 = qkv.dim() == 3 ? qkv.size(2) : qkv.size(2) * qkv.size(3);
  TORCH_CHECK(qkv.dim() == 3 || qkv.size(2) == 3, "pet_attention: qkv [N, L, 3, D] expected");
  TORCH_CHECK(D3 % (3 * heads) == 0, "pet_attention: qkv last dims must be 3 * heads * d");
  const int64_t D = D3 / 3, d = D / heads;
  TORCH_CHECK(mlipcpp_attn::head_dim_supported((int)d), "pet_attention: head dim must be 8, 16, 32 or 64, got ", d);
  TORCH_CHECK(bias.numel() == N * L * L &&
                  (bias.dim() == 3 || (bias.dim() == 4 && bias.size(1) == 1)) &&
                  bias.size(0) == N && bias.size(-1) == L && bias.size(-2) == L,
              "pet_attention: bias must be [N, L, L] or [N, 1, L, L]");
  TORCH_CHECK(qkv.scalar_type() == bias.scalar_type(), "pet_attention: qkv and bias dtypes differ");
  TORCH_CHECK(qkv.scalar_type() == at::kFloat || qkv.scalar_type() == at::kDouble,
              "pet_attention: float32 or float64 only");
  TORCH_CHECK(L <= (1 << 28), "pet_attention: sequence too long");
  return {N, L, heads, d, D};
}

void check_like(const Tensor &t, const Tensor &ref, const char *what) {
  TORCH_CHECK(t.numel() == ref.numel() && t.scalar_type() == ref.scalar_type() && t.device() == ref.device(),
              "pet_attention: ", what, " does not match its reference tensor");
}

int variant() {
  const char *e = std::getenv("MLIPCPP_ATTN_V1");
  return e && std::atoi(e) != 0 ? 1 : 0;
}

template <typename T> const T *ptr(const Tensor &t) { return t.defined() ? t.data_ptr<T>() : nullptr; }
template <typename T> const T *ptr(const optional<Tensor> &t) {
  return t.has_value() && t->defined() ? t->data_ptr<T>() : nullptr;
}
Tensor cont(const optional<Tensor> &t) {
  return t.has_value() && t->defined() ? t->contiguous() : Tensor();
}

// ---- CUDA kernels ----------------------------------------------------------

std::tuple<Tensor, Tensor> pet_attention_cuda(const Tensor &qkv_, const Tensor &bias_, int64_t heads, double scale) {
  const Dims s = dims_of(qkv_, bias_, heads);
  TORCH_CHECK(qkv_.is_cuda() && bias_.is_cuda(), "pet_attention: CUDA tensors expected");
  const c10::cuda::CUDAGuard guard(qkv_.device());
  const Tensor qkv = qkv_.contiguous(), bias = bias_.contiguous();
  Tensor o = at::empty({s.N, s.L, s.D}, qkv.options());
  Tensor lse = at::empty({s.N, s.L, s.H}, qkv.options());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES(qkv.scalar_type(), "pet_attention", [&] {
    mlipcpp_attn::forward<scalar_t>(ptr<scalar_t>(qkv), ptr<scalar_t>(bias), o.data_ptr<scalar_t>(),
                                    lse.data_ptr<scalar_t>(), s.N, (int)s.L, (int)s.H, (int)s.d, scale, stream,
                                    variant());
  });
  return {o, lse};
}

std::tuple<Tensor, Tensor> pet_attention_backward_cuda(const Tensor &qkv_, const Tensor &bias_, const Tensor &o_,
                                                       const Tensor &lse_, const Tensor &go_,
                                                       const optional<Tensor> &glse_, int64_t heads, double scale) {
  const Dims s = dims_of(qkv_, bias_, heads);
  const c10::cuda::CUDAGuard guard(qkv_.device());
  const Tensor qkv = qkv_.contiguous(), bias = bias_.contiguous(), o = o_.contiguous(), lse = lse_.contiguous(),
               go = go_.contiguous(), glse = cont(glse_);
  TORCH_CHECK(o.numel() == s.N * s.L * s.D && go.numel() == o.numel() && lse.numel() == s.N * s.L * s.H &&
                  (!glse.defined() || glse.numel() == lse.numel()),
              "pet_attention_backward: shapes");
  Tensor dqkv = at::empty(qkv_.sizes(), qkv.options());
  Tensor dbias = at::empty(bias_.sizes(), bias.options());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES(qkv.scalar_type(), "pet_attention_backward", [&] {
    mlipcpp_attn::backward<scalar_t>(ptr<scalar_t>(qkv), ptr<scalar_t>(bias), ptr<scalar_t>(o), ptr<scalar_t>(lse),
                                     ptr<scalar_t>(go), ptr<scalar_t>(glse), dqkv.data_ptr<scalar_t>(),
                                     dbias.data_ptr<scalar_t>(), s.N, (int)s.L, (int)s.H, (int)s.d, scale, stream,
                                     variant());
  });
  return {dqkv, dbias};
}

std::tuple<Tensor, Tensor> pet_attention_jvp_cuda(const Tensor &qkv_, const Tensor &bias_,
                                                  const optional<Tensor> &tqkv_, const optional<Tensor> &tbias_,
                                                  int64_t heads, double scale) {
  const Dims s = dims_of(qkv_, bias_, heads);
  const c10::cuda::CUDAGuard guard(qkv_.device());
  const Tensor qkv = qkv_.contiguous(), bias = bias_.contiguous(), tqkv = cont(tqkv_), tbias = cont(tbias_);
  if (tqkv.defined()) check_like(tqkv, qkv, "tqkv");
  if (tbias.defined()) check_like(tbias, bias, "tbias");
  Tensor to = at::empty({s.N, s.L, s.D}, qkv.options());
  Tensor tlse = at::empty({s.N, s.L, s.H}, qkv.options());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES(qkv.scalar_type(), "pet_attention_jvp", [&] {
    mlipcpp_attn::jvp<scalar_t>(ptr<scalar_t>(qkv), ptr<scalar_t>(bias), ptr<scalar_t>(tqkv), ptr<scalar_t>(tbias),
                                to.data_ptr<scalar_t>(), tlse.data_ptr<scalar_t>(), s.N, (int)s.L, (int)s.H,
                                (int)s.d, scale, stream);
  });
  return {to, tlse};
}

// gradients of <(go, glse), jvp(qkv, bias; tqkv, tbias)> w.r.t. qkv, bias,
// tqkv, tbias
std::tuple<Tensor, Tensor, Tensor, Tensor>
pet_attention_jvp_backward_cuda(const Tensor &qkv_, const Tensor &bias_, const Tensor &go_,
                                const optional<Tensor> &glse_, const optional<Tensor> &tqkv_,
                                const optional<Tensor> &tbias_, int64_t heads, double scale) {
  const Dims s = dims_of(qkv_, bias_, heads);
  const c10::cuda::CUDAGuard guard(qkv_.device());
  const Tensor qkv = qkv_.contiguous(), bias = bias_.contiguous(), go = go_.contiguous(), glse = cont(glse_),
               tqkv = cont(tqkv_), tbias = cont(tbias_);
  TORCH_CHECK(go.numel() == s.N * s.L * s.D && (!glse.defined() || glse.numel() == s.N * s.L * s.H),
              "pet_attention_jvp_backward: shapes");
  if (tqkv.defined()) check_like(tqkv, qkv, "tqkv");
  if (tbias.defined()) check_like(tbias, bias, "tbias");
  Tensor gx = at::empty({s.N, s.L, 6, s.D}, qkv.options());
  Tensor gb = at::empty({s.N, 2, s.L, s.L}, qkv.options());
  const bool dbl = qkv.scalar_type() == at::kDouble;
  Tensor stats = at::empty({(int64_t)mlipcpp_attn::jvp_stats_bytes(s.N * s.H * s.L, dbl)},
                           qkv.options().dtype(at::kByte));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES(qkv.scalar_type(), "pet_attention_jvp_backward", [&] {
    mlipcpp_attn::jvp_backward<scalar_t>(ptr<scalar_t>(qkv), ptr<scalar_t>(bias), ptr<scalar_t>(go),
                                         ptr<scalar_t>(glse), ptr<scalar_t>(tqkv), ptr<scalar_t>(tbias),
                                         stats.data_ptr(), gx.data_ptr<scalar_t>(), gb.data_ptr<scalar_t>(), s.N,
                                         (int)s.L, (int)s.H, (int)s.d, scale, stream);
  });
  return {gx.narrow(2, 0, 3).reshape(qkv_.sizes()), gb.select(1, 0).reshape(bias_.sizes()),
          gx.narrow(2, 3, 3).reshape(qkv_.sizes()), gb.select(1, 1).reshape(bias_.sizes())};
}

// everything the backward of PetAttnBackwardFn needs, in one op: with
// u = (grad_o, grad_lse) and w = (tqkv, tbias),
//   (to, tlse) = J w,  (g_qkv, g_bias) = d/d(qkv, bias) <u, J w>
std::tuple<Tensor, Tensor, Tensor, Tensor>
pet_attention_double_backward_cuda(const Tensor &qkv_, const Tensor &bias_, const Tensor &go_,
                                   const optional<Tensor> &glse_, const optional<Tensor> &tqkv_,
                                   const optional<Tensor> &tbias_, int64_t heads, double scale) {
  const Dims s = dims_of(qkv_, bias_, heads);
  const c10::cuda::CUDAGuard guard(qkv_.device());
  const Tensor qkv = qkv_.contiguous(), bias = bias_.contiguous(), go = go_.contiguous(), glse = cont(glse_),
               tqkv = cont(tqkv_), tbias = cont(tbias_);
  TORCH_CHECK(go.numel() == s.N * s.L * s.D && (!glse.defined() || glse.numel() == s.N * s.L * s.H),
              "pet_attention_double_backward: shapes");
  if (tqkv.defined()) check_like(tqkv, qkv, "tqkv");
  if (tbias.defined()) check_like(tbias, bias, "tbias");
  Tensor to = at::empty({s.N, s.L, s.D}, qkv.options());
  Tensor tlse = at::empty({s.N, s.L, s.H}, qkv.options());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  bool fused = false;
  AT_DISPATCH_FLOATING_TYPES(qkv.scalar_type(), "pet_attention_double_backward", [&] {
    fused = mlipcpp_attn::double_backward_fused_ok<scalar_t>((int)s.L, (int)s.d);
  });
  if (fused) {
    Tensor gx = at::empty(qkv_.sizes(), qkv.options());
    Tensor gb = at::empty(bias_.sizes(), bias.options());
    AT_DISPATCH_FLOATING_TYPES(qkv.scalar_type(), "pet_attention_double_backward", [&] {
      mlipcpp_attn::double_backward<scalar_t>(ptr<scalar_t>(qkv), ptr<scalar_t>(bias), ptr<scalar_t>(go),
                                              ptr<scalar_t>(glse), ptr<scalar_t>(tqkv), ptr<scalar_t>(tbias),
                                              to.data_ptr<scalar_t>(), tlse.data_ptr<scalar_t>(),
                                              gx.data_ptr<scalar_t>(), gb.data_ptr<scalar_t>(), s.N, (int)s.L,
                                              (int)s.H, (int)s.d, scale, stream);
    });
    return {to, tlse, gx, gb};
  }
  // long sequences: the ggml JVP kernels (one thread per row, nothing L x L stored)
  auto [to2, tlse2] = pet_attention_jvp_cuda(qkv_, bias_, tqkv_, tbias_, heads, scale);
  auto r = pet_attention_jvp_backward_cuda(qkv_, bias_, go_, glse_, tqkv_, tbias_, heads, scale);
  return {to2, tlse2, std::get<0>(r), std::get<1>(r)};
}

// ---- Meta (shape-only) kernels: FakeTensor / torch.compile ------------------

std::tuple<Tensor, Tensor> pet_attention_meta(const Tensor &qkv, const Tensor &bias, int64_t heads, double) {
  const Dims s = dims_of(qkv, bias, heads);
  return {at::empty({s.N, s.L, s.D}, qkv.options()), at::empty({s.N, s.L, s.H}, qkv.options())};
}
std::tuple<Tensor, Tensor> pet_attention_backward_meta(const Tensor &qkv, const Tensor &bias, const Tensor &,
                                                       const Tensor &, const Tensor &, const optional<Tensor> &,
                                                       int64_t, double) {
  return {at::empty(qkv.sizes(), qkv.options()), at::empty(bias.sizes(), bias.options())};
}
std::tuple<Tensor, Tensor> pet_attention_jvp_meta(const Tensor &qkv, const Tensor &bias, const optional<Tensor> &,
                                                  const optional<Tensor> &, int64_t heads, double) {
  const Dims s = dims_of(qkv, bias, heads);
  return {at::empty({s.N, s.L, s.D}, qkv.options()), at::empty({s.N, s.L, s.H}, qkv.options())};
}
std::tuple<Tensor, Tensor, Tensor, Tensor>
pet_attention_jvp_backward_meta(const Tensor &qkv, const Tensor &bias, const Tensor &, const optional<Tensor> &,
                                const optional<Tensor> &, const optional<Tensor> &, int64_t, double) {
  return {at::empty(qkv.sizes(), qkv.options()), at::empty(bias.sizes(), bias.options()),
          at::empty(qkv.sizes(), qkv.options()), at::empty(bias.sizes(), bias.options())};
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
pet_attention_double_backward_meta(const Tensor &qkv, const Tensor &bias, const Tensor &, const optional<Tensor> &,
                                   const optional<Tensor> &, const optional<Tensor> &, int64_t heads, double) {
  const Dims s = dims_of(qkv, bias, heads);
  return {at::empty({s.N, s.L, s.D}, qkv.options()), at::empty({s.N, s.L, s.H}, qkv.options()),
          at::empty(qkv.sizes(), qkv.options()), at::empty(bias.sizes(), bias.options())};
}

// ---- autograd ----------------------------------------------------------------

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

const c10::TypedOperatorHandle<std::tuple<Tensor, Tensor>(const Tensor &, const Tensor &, int64_t, double)> &
op_fwd() {
  static auto h = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("mlipcpp::pet_attention", "")
                      .typed<std::tuple<Tensor, Tensor>(const Tensor &, const Tensor &, int64_t, double)>();
  return h;
}
using BwdSig = std::tuple<Tensor, Tensor>(const Tensor &, const Tensor &, const Tensor &, const Tensor &,
                                          const Tensor &, const optional<Tensor> &, int64_t, double);
using JvpSig = std::tuple<Tensor, Tensor>(const Tensor &, const Tensor &, const optional<Tensor> &,
                                          const optional<Tensor> &, int64_t, double);
using JvpBwdSig = std::tuple<Tensor, Tensor, Tensor, Tensor>(const Tensor &, const Tensor &, const Tensor &,
                                                             const optional<Tensor> &, const optional<Tensor> &,
                                                             const optional<Tensor> &, int64_t, double);
const c10::TypedOperatorHandle<BwdSig> &op_bwd() {
  static auto h = c10::Dispatcher::singleton().findSchemaOrThrow("mlipcpp::pet_attention_backward", "").typed<BwdSig>();
  return h;
}
const c10::TypedOperatorHandle<JvpSig> &op_jvp() {
  static auto h = c10::Dispatcher::singleton().findSchemaOrThrow("mlipcpp::pet_attention_jvp", "").typed<JvpSig>();
  return h;
}
const c10::TypedOperatorHandle<JvpBwdSig> &op_dbl_bwd() {
  static auto h =
      c10::Dispatcher::singleton().findSchemaOrThrow("mlipcpp::pet_attention_double_backward", "").typed<JvpBwdSig>();
  return h;
}
const c10::TypedOperatorHandle<JvpBwdSig> &op_jvp_bwd() {
  static auto h =
      c10::Dispatcher::singleton().findSchemaOrThrow("mlipcpp::pet_attention_jvp_backward", "").typed<JvpBwdSig>();
  return h;
}
optional<Tensor> opt(const Tensor &t) { return t.defined() ? optional<Tensor>(t) : c10::nullopt; }

// (qkv, bias, o, lse, go, glse) -> (dqkv, dbias); its own backward uses the JVP
class PetAttnBackwardFn : public torch::autograd::Function<PetAttnBackwardFn> {
public:
  static variable_list forward(AutogradContext *ctx, const Tensor &qkv, const Tensor &bias, const Tensor &o,
                               const Tensor &lse, const Tensor &go, const Tensor &glse, int64_t heads,
                               double scale) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto [dqkv, dbias] = op_bwd().call(qkv, bias, o, lse, go, opt(glse), heads, scale);
    ctx->save_for_backward({qkv, bias, go, glse});
    ctx->saved_data["heads"] = heads;
    ctx->saved_data["scale"] = scale;
    return {dqkv, dbias};
  }
  static variable_list backward(AutogradContext *ctx, variable_list grads) {
    // grads: w = (wqkv, wbias), cotangents of (dqkv, dbias)
    TORCH_CHECK(!c10::GradMode::is_enabled(),
                "mlipcpp::pet_attention: third derivatives are not implemented "
                "(backward of the double backward with create_graph=True)");
    auto saved = ctx->get_saved_variables();
    const Tensor &qkv = saved[0], &bias = saved[1], &go = saved[2], &glse = saved[3];
    const int64_t heads = ctx->saved_data["heads"].toInt();
    const double scale = ctx->saved_data["scale"].toDouble();
    const Tensor &wq = grads[0], &wb = grads[1];
    if (!wq.defined() && !wb.defined()) return variable_list(8);
    auto [to, tlse, g_qkv, g_bias] = op_dbl_bwd().call(qkv, bias, go, opt(glse), opt(wq), opt(wb), heads, scale);
    Tensor g_go = to.view(go.sizes()), g_glse = glse.defined() ? tlse.view(glse.sizes()) : Tensor();
    // o and lse are functions of (qkv, bias): the JVP kernels differentiate
    // through the recomputed P, so nothing flows through o and lse here
    return {g_qkv, g_bias, Tensor(), Tensor(), g_go, g_glse, Tensor(), Tensor()};
  }
};

class PetAttnFn : public torch::autograd::Function<PetAttnFn> {
public:
  static variable_list forward(AutogradContext *ctx, const Tensor &qkv, const Tensor &bias, int64_t heads,
                               double scale) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto [o, lse] = op_fwd().call(qkv, bias, heads, scale);
    ctx->save_for_backward({qkv, bias, o, lse});
    ctx->saved_data["heads"] = heads;
    ctx->saved_data["scale"] = scale;
    return {o, lse};
  }
  static variable_list backward(AutogradContext *ctx, variable_list grads) {
    auto saved = ctx->get_saved_variables();
    const int64_t heads = ctx->saved_data["heads"].toInt();
    const double scale = ctx->saved_data["scale"].toDouble();
    Tensor go = grads[0], glse = grads[1];
    if (!go.defined() && !glse.defined()) return variable_list(4);
    if (!go.defined()) go = at::zeros_like(saved[2]);
    auto r = PetAttnBackwardFn::apply(saved[0], saved[1], saved[2], saved[3], go, glse, heads, scale);
    return {r[0], r[1], Tensor(), Tensor()};
  }
};

std::tuple<Tensor, Tensor> pet_attention_autograd(const Tensor &qkv, const Tensor &bias, int64_t heads,
                                                  double scale) {
  auto r = PetAttnFn::apply(qkv, bias, heads, scale);
  return {r[0], r[1]};
}

} // namespace

TORCH_LIBRARY(mlipcpp, m) {
  m.def("pet_attention(Tensor qkv, Tensor bias, int heads, float scale) -> (Tensor, Tensor)");
  m.def("pet_attention_backward(Tensor qkv, Tensor bias, Tensor o, Tensor lse, Tensor grad_o, Tensor? grad_lse, "
        "int heads, float scale) -> (Tensor, Tensor)");
  m.def("pet_attention_jvp(Tensor qkv, Tensor bias, Tensor? tqkv, Tensor? tbias, int heads, float scale) "
        "-> (Tensor, Tensor)");
  m.def("pet_attention_jvp_backward(Tensor qkv, Tensor bias, Tensor grad_o, Tensor? grad_lse, Tensor? tqkv, "
        "Tensor? tbias, int heads, float scale) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("pet_attention_double_backward(Tensor qkv, Tensor bias, Tensor grad_o, Tensor? grad_lse, Tensor? tqkv, "
        "Tensor? tbias, int heads, float scale) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mlipcpp, CUDA, m) {
  m.impl("pet_attention", &pet_attention_cuda);
  m.impl("pet_attention_backward", &pet_attention_backward_cuda);
  m.impl("pet_attention_jvp", &pet_attention_jvp_cuda);
  m.impl("pet_attention_jvp_backward", &pet_attention_jvp_backward_cuda);
  m.impl("pet_attention_double_backward", &pet_attention_double_backward_cuda);
}

TORCH_LIBRARY_IMPL(mlipcpp, Meta, m) {
  m.impl("pet_attention", &pet_attention_meta);
  m.impl("pet_attention_backward", &pet_attention_backward_meta);
  m.impl("pet_attention_jvp", &pet_attention_jvp_meta);
  m.impl("pet_attention_jvp_backward", &pet_attention_jvp_backward_meta);
  m.impl("pet_attention_double_backward", &pet_attention_double_backward_meta);
}

TORCH_LIBRARY_IMPL(mlipcpp, Autograd, m) { m.impl("pet_attention", &pet_attention_autograd); }
