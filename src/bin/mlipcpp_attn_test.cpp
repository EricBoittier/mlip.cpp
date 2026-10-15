// Fused PET attention (ggml_pet_attn) check and timing: the output and the
// q/k/v gradients of L = sum(r * attn(qkv, bias)) on the best backend, against
// a double-precision reference, for every CUDA kernel variant (switched by
// environment: default, GGML_PET_ATTN_V1=1, GGML_PET_ATTN_BF16=1; "+f16": qkv
// cast to F16 before the op, as with MLIPCPP_ATTN_F16=1), then the forward and
// forward+backward times on the shapes PET produces.
#include "core/backend.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace mlipcpp;

namespace {

struct Problem {
  int64_t H, d, L, N;
  std::vector<float> qkv, bias, r; // qkv [D,3,L,N], bias [L,L,N], r [D+H,L,N]
};

Problem make_problem(int64_t H, int64_t d, int64_t L, int64_t N) {
  Problem p{H, d, L, N, {}, {}, {}};
  const int64_t D = H * d;
  std::mt19937 rng(L * 1000 + N);
  std::normal_distribution<float> U(0, 1);
  p.qkv.resize(D * 3 * L * N), p.bias.resize(L * L * N), p.r.resize((D + H) * L * N);
  for (auto &v : p.qkv) v = U(rng);
  // PET's bias is log(cutoff weight): mostly O(1), some strongly suppressed pairs
  for (auto &v : p.bias) v = (rng() % 8 == 0) ? -30.0f + U(rng) : U(rng);
  for (int64_t row = 0; row < L * N; row++)
    for (int64_t c = 0; c < D + H; c++) p.r[row * (D + H) + c] = c < D ? U(rng) : 0.0f; // lse part has no gradient
  return p;
}

// output o [D, L, N] and gradient of qkv, double
void reference(const Problem &p, std::vector<double> &o, std::vector<double> &g) {
  const int64_t H = p.H, d = p.d, L = p.L, D = H * d;
  const double scale = 1 / std::sqrt(double(d));
  o.assign(D * L * p.N, 0), g.assign(D * 3 * L * p.N, 0);
  std::vector<double> P(L * L), dS(L * L);
  for (int64_t n = 0; n < p.N; n++)
    for (int64_t h = 0; h < H; h++) {
      auto x = [&](int64_t i, int part, int64_t c) { return double(p.qkv[((n * L + i) * 3 + part) * D + h * d + c]); };
      auto dO = [&](int64_t i, int64_t c) { return double(p.r[(n * L + i) * (D + H) + h * d + c]); };
      auto gx = [&](int64_t i, int part, int64_t c) -> double & { return g[((n * L + i) * 3 + part) * D + h * d + c]; };
      for (int64_t i = 0; i < L; i++) {
        double mx = -1e300;
        for (int64_t j = 0; j < L; j++) {
          double s = 0;
          for (int64_t c = 0; c < d; c++) s += x(i, 0, c) * x(j, 1, c);
          P[i * L + j] = scale * s + p.bias[(n * L + i) * L + j];
          mx = std::max(mx, P[i * L + j]);
        }
        double sum = 0;
        for (int64_t j = 0; j < L; j++) sum += (P[i * L + j] = std::exp(P[i * L + j] - mx));
        for (int64_t j = 0; j < L; j++) P[i * L + j] /= sum;
        double Di = 0;
        for (int64_t j = 0; j < L; j++) {
          double dp = 0;
          for (int64_t c = 0; c < d; c++) {
            o[(n * L + i) * D + h * d + c] += P[i * L + j] * x(j, 2, c);
            dp += dO(i, c) * x(j, 2, c);
          }
          dS[i * L + j] = dp;
          Di += P[i * L + j] * dp;
        }
        for (int64_t j = 0; j < L; j++) dS[i * L + j] = P[i * L + j] * (dS[i * L + j] - Di);
      }
      for (int64_t i = 0; i < L; i++)
        for (int64_t j = 0; j < L; j++)
          for (int64_t c = 0; c < d; c++) {
            gx(i, 0, c) += scale * dS[i * L + j] * x(j, 1, c);
            gx(j, 1, c) += scale * dS[i * L + j] * x(i, 0, c);
            gx(j, 2, c) += P[i * L + j] * dO(i, c);
          }
    }
}

struct Result {
  std::vector<float> o, g; // o [D+H, L, N], g like qkv
  double t_fwd = 0, t_all = 0; // ms per call
};

// reps = 0: one evaluation, no timing; f16: the op reads an F16 copy of qkv
Result run(BackendProvider &bp, const Problem &p, int reps, bool f16) {
  const int64_t D = p.H * p.d;
  const float scale = 1.0f / std::sqrt(float(p.d));
  ggml_context *ctx = ggml_init({ggml_tensor_overhead() * 64 + 2 * ggml_graph_overhead_custom(64, true), nullptr, true});
  ggml_tensor *qkv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 3, p.L, p.N);
  ggml_tensor *bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, p.L, p.L, 1, p.N);
  ggml_tensor *r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D + p.H, p.L, p.N);
  for (ggml_tensor *t : {qkv, bias, r}) ggml_set_input(t);
  ggml_set_param(qkv);
  ggml_tensor *y = ggml_pet_attn(ctx, f16 ? ggml_cast(ctx, qkv, GGML_TYPE_F16) : qkv, bias, int(p.H), scale);
  ggml_set_output(y);
  ggml_tensor *loss = ggml_sum(ctx, ggml_mul(ctx, y, r));
  ggml_set_loss(loss);

  auto set_inputs = [&] {
    ggml_backend_tensor_set(qkv, p.qkv.data(), 0, ggml_nbytes(qkv));
    ggml_backend_tensor_set(bias, p.bias.data(), 0, ggml_nbytes(bias));
    if (r->buffer) ggml_backend_tensor_set(r, p.r.data(), 0, ggml_nbytes(r)); // not in the forward graph
  };
  auto time = [&](ggml_cgraph *gf) {
    ggml_backend_graph_compute(bp.primary(), gf); // warm-up
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) ggml_backend_graph_compute(bp.primary(), gf);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
  };
  Result res;
  ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(bp.primary());

  // forward only (allocated first: the full graph re-places every tensor)
  if (reps > 0) {
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t ga = ggml_gallocr_new(bt);
    ggml_gallocr_alloc_graph(ga, gf);
    set_inputs();
    res.t_fwd = time(gf);
    ggml_gallocr_free(ga);
    // the tensors still point into the freed buffer: unplace them, or the
    // full graph's allocator would take them as already allocated
    for (ggml_tensor *t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) t->data = nullptr, t->buffer = nullptr;
  }

  ggml_cgraph *gb = ggml_new_graph_custom(ctx, 64, true);
  ggml_build_forward_expand(gb, loss);
  ggml_build_backward_expand(ctx, gb, nullptr);
  ggml_tensor *gq = ggml_graph_get_grad(gb, qkv);
  for (ggml_tensor *v = gq; v; v = v->view_src) ggml_set_output(v);
  ggml_tensor *lg = ggml_graph_get_grad_acc(gb, loss);
  ggml_set_input(lg);
  ggml_gallocr_t ga = ggml_gallocr_new(bt);
  ggml_gallocr_alloc_graph(ga, gb);
  set_inputs();
  const float one = 1;
  ggml_backend_tensor_set(lg, &one, 0, 4);
  ggml_backend_graph_compute(bp.primary(), gb);
  res.o.resize(ggml_nelements(y)), res.g.resize(ggml_nelements(qkv));
  ggml_backend_tensor_get(y, res.o.data(), 0, ggml_nbytes(y));
  ggml_backend_tensor_get(gq, res.g.data(), 0, ggml_nbytes(gq));
  if (reps > 0) res.t_all = time(gb);
  ggml_gallocr_free(ga);
  ggml_free(ctx);
  return res;
}

// max |a - b| / max |b| over the rows' first D of every `stride` values
double err(const std::vector<float> &a, const std::vector<double> &b, int64_t stride_a, int64_t stride_b, int64_t D,
           int64_t off = 0) {
  double e = 0, m = 0;
  for (size_t row = 0; row < b.size() / stride_b; row++)
    for (int64_t c = 0; c < D; c++) {
      e = std::max(e, std::abs(a[row * stride_a + off + c] - b[row * stride_b + off + c]));
      m = std::max(m, std::abs(b[row * stride_b + off + c]));
    }
  return e / m;
}

struct Variant {
  const char *name, *env;
  bool f16;
  double tol; // fp32 kernels to fp32 rounding, F16 storage to 2^-11, bf16 to 2^-8
};
const Variant VARIANTS[] = {{"v2", "", false, 1e-4},
                            {"v1", "GGML_PET_ATTN_V1", false, 1e-4},
                            {"bf16", "GGML_PET_ATTN_BF16", false, 2e-2},
                            {"v2+f16", "", true, 5e-3},
                            {"bf16+f16", "GGML_PET_ATTN_BF16", true, 2e-2}};
constexpr int NV = sizeof(VARIANTS) / sizeof(VARIANTS[0]);

void use_variant(int v) {
  unsetenv("GGML_PET_ATTN_V1"), unsetenv("GGML_PET_ATTN_BF16");
  if (*VARIANTS[v].env) setenv(VARIANTS[v].env, "1", 1);
}

} // namespace

int main(int argc, char **argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 20;
  // every run builds a new graph, often at the addresses of the last one:
  // ggml's CUDA graph cache would replay the previous variant's kernels
  setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
  auto bp = BackendProvider::create(BackendPreference::Auto);
  fmt::print("backend {}\n", bp->name());
  int bad = 0;

  fmt::print("== accuracy (max abs error / max abs value; o, dq, dk, dv)\n");
  for (auto [H, d, L, N] : {std::tuple<int64_t, int64_t, int64_t, int64_t>{16, 16, 37, 3}, {16, 16, 129, 2},
                            {8, 32, 70, 2}, {4, 64, 50, 2}, {16, 16, 683, 1}}) {
    const Problem p = make_problem(H, d, L, N);
    std::vector<double> o, g;
    reference(p, o, g);
    const int64_t D = H * d;
    for (int v = 0; v < NV; v++) {
      use_variant(v);
      const Result r = run(*bp, p, 0, VARIANTS[v].f16);
      const double e[4] = {err(r.o, o, D + H, D, D), err(r.g, g, 3 * D, 3 * D, D, 0), err(r.g, g, 3 * D, 3 * D, D, D),
                           err(r.g, g, 3 * D, 3 * D, D, 2 * D)};
      const bool ok = std::max({e[0], e[1], e[2], e[3]}) < VARIANTS[v].tol;
      bad += !ok;
      fmt::print("H {:2} d {:2} L {:4} N {:2}  {:<8} {:.2e} {:.2e} {:.2e} {:.2e}  {}\n", H, d, L, N, VARIANTS[v].name, e[0],
                 e[1], e[2], e[3], ok ? "ok" : "WRONG");
    }
  }

  if (reps > 0) {
    fmt::print("== timing (ms per call: forward, forward+backward; {} reps)\n", reps);
    // shapes of the d_pet 256 / 16-head model at 6 A (profile of 145277) and
    // adaptive-cutoff sized sequences
    for (auto [H, d, L, N] : {std::tuple<int64_t, int64_t, int64_t, int64_t>{16, 16, 683, 4}, {16, 16, 462, 15},
                              {16, 16, 177, 30}, {16, 16, 74, 30}, {16, 16, 24, 1000}, {16, 16, 40, 500}}) {
      const Problem p = make_problem(H, d, L, N);
      fmt::print("H {:2} d {:2} L {:4} N {:4}", H, d, L, N);
      double base = 0;
      for (int v = 0; v < NV; v++) {
        use_variant(v);
        const Result r = run(*bp, p, reps, VARIANTS[v].f16);
        if (v == 0) base = r.t_all;
        fmt::print("  {} {:.3f} {:.3f}{}", VARIANTS[v].name, r.t_fwd, r.t_all,
                   v ? fmt::format(" (x{:.2f})", base / r.t_all) : std::string());
      }
      fmt::print("\n");
    }
  }
  return bad ? 1 : 0;
}
