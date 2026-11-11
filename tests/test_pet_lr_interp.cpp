// PET-LR port (after M. Langer, ipsum/PET-LR): unit tests of the GIR interpreter additions the iris LR graph needs,
// each on a hand-written graph, forward against closed forms and the ggml backward against analytic gradients
// (linear ops) or central differences (erf):
//  - INDEX of a 1-D source by a 1-D index (per-atom / per-slot gathers), repeated indices accumulate;
//  - MATMUL of a matrix by a vector (torch [m, k] @ [k] -> [m]; the open Coulomb kernel on the charges);
//  - REPEAT whose extra operands are shape sources ("input:..." of a symbolic-size torch .expand);
//  - SUB / DIV with a broadcast second operand (ggml's SUB / DIV backward lack repeat_back for src1);
//  - UNARY_ERF (Gaussian charges);
//  - the GIR JSON number parser with an explicit exponent sign (Python writes 1e+30).
#include <catch2/catch_test_macros.hpp>

#include "../src/runtime/graph_interpreter.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>

using namespace mlipcpp::runtime;

namespace {

struct In {
  std::string name;
  ggml_type type;
  std::vector<int64_t> ne; // ggml order
  std::vector<float> f;    // F32 data
  std::vector<int32_t> i;  // I32 data
  bool param = false;
};

struct Run {
  std::vector<float> out;
  std::vector<int64_t> out_ne;
  std::map<std::string, std::vector<float>> grad;
};

// build the graph, loss = sum(out * w) with w[j] = 0.3 + 0.1 j (so every output element has its own weight)
Run run(const std::string &json, const std::vector<In> &ins, bool backward = true) {
  GraphInterpreter interp;
  interp.load_graph(json);
  ggml_context *ctx = ggml_init({64 * 1024 * 1024, nullptr, true});
  std::map<std::string, ggml_tensor *> t;
  for (const auto &x : ins) {
    ggml_tensor *v = ggml_new_tensor(ctx, x.type, int(x.ne.size()), x.ne.data());
    ggml_set_input(v);
    if (x.param) ggml_set_param(v);
    interp.set_input(x.name, v);
    t[x.name] = v;
  }
  ggml_tensor *out = interp.build(ctx);
  REQUIRE(out != nullptr);
  ggml_tensor *w = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, out->ne);
  ggml_tensor *loss = ggml_sum(ctx, ggml_mul(ctx, out, w));
  ggml_set_output(out);
  ggml_set_loss(loss);
  ggml_cgraph *g = ggml_new_graph_custom(ctx, 4096, true);
  ggml_build_forward_expand(g, loss);
  if (backward) ggml_build_backward_expand(ctx, g, nullptr);
  ggml_backend_t cpu = ggml_backend_cpu_init();
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
  REQUIRE(buf != nullptr);
  for (const auto &x : ins) {
    if (x.type == GGML_TYPE_F32) ggml_backend_tensor_set(t[x.name], x.f.data(), 0, x.f.size() * 4);
    else ggml_backend_tensor_set(t[x.name], x.i.data(), 0, x.i.size() * 4);
  }
  std::vector<float> wv(ggml_nelements(w));
  for (size_t j = 0; j < wv.size(); j++) wv[j] = 0.3f + 0.1f * float(j);
  ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * 4);
  interp.init_constants();
  if (backward) ggml_graph_reset(g);
  REQUIRE(ggml_backend_graph_compute(cpu, g) == GGML_STATUS_SUCCESS);
  Run r;
  r.out.resize(ggml_nelements(out));
  ggml_backend_tensor_get(out, r.out.data(), 0, r.out.size() * 4);
  for (int d = 0; d < ggml_n_dims(out); d++) r.out_ne.push_back(out->ne[d]);
  if (backward)
    for (const auto &x : ins)
      if (x.param) {
        ggml_tensor *gr = ggml_graph_get_grad(g, t[x.name]);
        REQUIRE(gr != nullptr);
        r.grad[x.name].resize(ggml_nelements(gr));
        ggml_backend_tensor_get(gr, r.grad[x.name].data(), 0, r.grad[x.name].size() * 4);
      }
  ggml_backend_buffer_free(buf);
  ggml_backend_free(cpu);
  ggml_free(ctx);
  return r;
}

double wgt(size_t j) { return 0.3 + 0.1 * double(j); }

std::string graph1(const std::string &inputs, const std::string &node, const std::string &extra = "") {
  return R"({"version":"1.0.0","model_type":"test","inputs":[)" + inputs +
         R"(],"outputs":[{"name":"y","node_ref":"node:0"}],"nodes":[)" + node + extra + "]}";
}

} // namespace

TEST_CASE("pet_lr_interp: INDEX of a 1-D source by a 1-D index, backward accumulates repeats", "[pet_lr_port]") {
  const std::string j = graph1(R"({"name":"x","dtype":"f32","shape":[5]},{"name":"idx","dtype":"i32","shape":[7]})",
                               R"({"id":0,"op":"INDEX","name":"index","inputs":["input:x","input:idx"],"output_shape":[7],"output_dtype":"f32"})");
  const std::vector<float> x = {1.5f, -2.0f, 0.25f, 4.0f, -0.5f};
  const std::vector<int32_t> idx = {3, 0, 3, 4, 1, 3, 0};
  Run r = run(j, {{"x", GGML_TYPE_F32, {5}, x, {}, true}, {"idx", GGML_TYPE_I32, {7}, {}, idx}});
  REQUIRE(r.out_ne == std::vector<int64_t>{7});
  for (int m = 0; m < 7; m++) REQUIRE(r.out[m] == x[idx[m]]);
  std::vector<double> gx(5, 0.0);
  for (int m = 0; m < 7; m++) gx[idx[m]] += wgt(m);
  for (int a = 0; a < 5; a++) REQUIRE(std::abs(r.grad["x"][a] - gx[a]) < 1e-6);
}

TEST_CASE("pet_lr_interp: MATMUL matrix @ vector ([m, k] @ [k] -> [m]) and both gradients", "[pet_lr_port]") {
  const std::string j = graph1(R"({"name":"A","dtype":"f32","shape":[3,4]},{"name":"v","dtype":"f32","shape":[4]})",
                               R"({"id":0,"op":"MATMUL","name":"matmul","inputs":["input:A","input:v"],"output_shape":[3],"output_dtype":"f32"})");
  std::vector<float> A(12), v = {0.5f, -1.0f, 2.0f, 0.75f};
  for (int q = 0; q < 12; q++) A[q] = 0.1f * float(q) - 0.4f; // torch row-major [3, 4] = ggml [4, 3]
  Run r = run(j, {{"A", GGML_TYPE_F32, {4, 3}, A, {}, true}, {"v", GGML_TYPE_F32, {4}, v, {}, true}});
  REQUIRE(r.out_ne == std::vector<int64_t>{3});
  for (int m = 0; m < 3; m++) {
    double y = 0;
    for (int k = 0; k < 4; k++) y += double(A[m * 4 + k]) * v[k];
    REQUIRE(std::abs(r.out[m] - y) < 1e-6);
  }
  for (int m = 0; m < 3; m++)
    for (int k = 0; k < 4; k++) REQUIRE(std::abs(r.grad["A"][m * 4 + k] - wgt(m) * v[k]) < 1e-6);
  for (int k = 0; k < 4; k++) {
    double g = 0;
    for (int m = 0; m < 3; m++) g += wgt(m) * A[m * 4 + k];
    REQUIRE(std::abs(r.grad["v"][k] - g) < 1e-6);
  }
}

TEST_CASE("pet_lr_interp: REPEAT with shape-source operands (symbolic .expand) and its backward", "[pet_lr_port]") {
  // torch: x [4, 1, 3].expand(n, n, 3), the export carries input:species twice as the sources of n
  const std::string j = graph1(R"({"name":"x","dtype":"f32","shape":[4,1,3]},{"name":"species","dtype":"i32","shape":[4]})",
                               R"({"id":0,"op":"REPEAT","name":"expand","inputs":["input:x","input:species","input:species"],"output_shape":[4,4,3],"output_dtype":"f32","params":{"shape":[3]}})");
  std::vector<float> x(12);
  for (int q = 0; q < 12; q++) x[q] = float(q) * 0.5f - 1.0f;
  Run r = run(j, {{"x", GGML_TYPE_F32, {3, 1, 4}, x, {}, true}, {"species", GGML_TYPE_I32, {4}, {}, {1, 8, 8, 1}}});
  REQUIRE(r.out_ne == std::vector<int64_t>{3, 4, 4});
  std::vector<double> gx(12, 0.0);
  for (int a = 0; a < 4; a++)
    for (int b = 0; b < 4; b++)
      for (int c = 0; c < 3; c++) {
        const size_t o = (size_t(a) * 4 + b) * 3 + c;
        REQUIRE(r.out[o] == x[a * 3 + c]);
        gx[a * 3 + c] += wgt(o);
      }
  for (int q = 0; q < 12; q++) REQUIRE(std::abs(r.grad["x"][q] - gx[q]) < 1e-5);
}

TEST_CASE("pet_lr_interp: SUB / DIV with a broadcast second operand, gradients of both operands", "[pet_lr_port]") {
  const std::vector<float> a = {1.0f, -2.0f, 3.5f, 0.5f, -1.25f, 2.0f, 4.0f, -3.0f};
  for (const std::string op : {"SUB", "DIV"}) {
    SECTION(op + " [8] by [1] (s - mu)") {
      const std::string j = graph1(R"({"name":"a","dtype":"f32","shape":[8]},{"name":"b","dtype":"f32","shape":[1]})",
                                   R"({"id":0,"op":")" + op + R"(","name":"o","inputs":["input:a","input:b"],"output_shape":[8],"output_dtype":"f32"})");
      const float b = 1.75f;
      Run r = run(j, {{"a", GGML_TYPE_F32, {8}, a, {}, true}, {"b", GGML_TYPE_F32, {1}, {b}, {}, true}});
      double gb = 0;
      for (int q = 0; q < 8; q++) {
        const double y = op == "SUB" ? double(a[q]) - b : double(a[q]) / b;
        REQUIRE(std::abs(r.out[q] - y) < 1e-6);
        const double ga = op == "SUB" ? wgt(q) : wgt(q) / b;
        REQUIRE(std::abs(r.grad["a"][q] - ga) < 1e-6);
        gb += op == "SUB" ? -wgt(q) : -wgt(q) * a[q] / (double(b) * b);
      }
      REQUIRE(std::abs(r.grad["b"][0] - gb) < 1e-5);
    }
    SECTION(op + " [2, 4] by [4] (rows)") {
      const std::string j = graph1(R"({"name":"a","dtype":"f32","shape":[2,4]},{"name":"b","dtype":"f32","shape":[4]})",
                                   R"({"id":0,"op":")" + op + R"(","name":"o","inputs":["input:a","input:b"],"output_shape":[2,4],"output_dtype":"f32"})");
      const std::vector<float> b = {0.5f, -1.5f, 2.0f, 1.25f};
      Run r = run(j, {{"a", GGML_TYPE_F32, {4, 2}, a, {}, true}, {"b", GGML_TYPE_F32, {4}, b, {}, true}});
      std::vector<double> gb(4, 0.0);
      for (int m = 0; m < 2; m++)
        for (int k = 0; k < 4; k++) {
          const int q = m * 4 + k;
          const double y = op == "SUB" ? double(a[q]) - b[k] : double(a[q]) / b[k];
          REQUIRE(std::abs(r.out[q] - y) < 1e-6);
          REQUIRE(std::abs(r.grad["a"][q] - (op == "SUB" ? wgt(q) : wgt(q) / b[k])) < 1e-6);
          gb[k] += op == "SUB" ? -wgt(q) : -wgt(q) * a[q] / (double(b[k]) * b[k]);
        }
      for (int k = 0; k < 4; k++) REQUIRE(std::abs(r.grad["b"][k] - gb[k]) < 1e-5);
    }
  }
}

TEST_CASE("pet_lr_interp: UNARY_ERF forward and backward (central differences)", "[pet_lr_port]") {
  const std::string j = graph1(R"({"name":"x","dtype":"f32","shape":[9]})",
                               R"({"id":0,"op":"UNARY_ERF","name":"erf","inputs":["input:x"],"output_shape":[9],"output_dtype":"f32"})");
  std::vector<float> x = {-3.0f, -1.5f, -0.7f, -0.1f, 0.0f, 0.2f, 0.9f, 1.8f, 4.0f};
  Run r = run(j, {{"x", GGML_TYPE_F32, {9}, x, {}, true}});
  const double h = 1e-3;
  for (int q = 0; q < 9; q++) {
    REQUIRE(std::abs(r.out[q] - std::erf(double(x[q]))) < 2e-7);
    const double analytic = wgt(q) * 2.0 / std::sqrt(M_PI) * std::exp(-double(x[q]) * x[q]);
    REQUIRE(std::abs(r.grad["x"][q] - analytic) < 1e-6);
    // central differences of the graph's own forward (fp32: h = 1e-3, tolerance 1e-3 relative)
    auto f = [&](float xv) {
      std::vector<float> xx = x;
      xx[q] = xv;
      return double(run(j, {{"x", GGML_TYPE_F32, {9}, xx}}, false).out[q]);
    };
    const double fd = wgt(q) * (f(float(x[q] + h)) - f(float(x[q] - h))) / (2 * h);
    REQUIRE(std::abs(r.grad["x"][q] - fd) < 1e-3 * std::max(1.0, std::abs(fd)) + 2e-4);
  }
}

TEST_CASE("pet_lr_interp: GIR numbers with an explicit exponent sign (1e+30, 2.5E+2, 1e-5)", "[pet_lr_port]") {
  // y = (x - 2.5E+2) / 1e+30 * ... checked through the scalar params of SUB / DIV
  const std::string j = R"({"version":"1.0.0","model_type":"test","inputs":[{"name":"x","dtype":"f32","shape":[2]}],
    "outputs":[{"name":"y","node_ref":"node:2"}],"nodes":[
    {"id":0,"op":"SUB","name":"s","inputs":["input:x"],"output_shape":[2],"output_dtype":"f32","params":{"scalar":2.5E+2}},
    {"id":1,"op":"DIV","name":"d","inputs":["node:0"],"output_shape":[2],"output_dtype":"f32","params":{"scalar":1e+30}},
    {"id":2,"op":"DIV","name":"e","inputs":["node:1"],"output_shape":[2],"output_dtype":"f32","params":{"scalar":1e-30}}]})";
  Run r = run(j, {{"x", GGML_TYPE_F32, {2}, {251.0f, 0.0f}, {}, true}});
  REQUIRE(std::abs(r.out[0] - 1.0) < 1e-5);
  REQUIRE(std::abs(r.out[1] + 250.0) < 1e-3);
}

// ggml-cuda-precise-unary.patch: LOG / SIN / COS / EXP flagged precise (op_params[last] = 1, set for iris PET-LR
// graphs by graph_model.cpp) run in double precision on CUDA; the default fast-math kernels keep __logf, whose error
// is ABSOLUTE (2^-21.4) near 1: log(1 + u) at small u (the log1p of softplus) had relative errors up to ~1 and made
// the seeded species preset 5x his floor on CUDA. On the GPU backend (skipped without one): flagged log(1 + u),
// sin / cos at |x| ~ 100 and exp agree with libm in double to fp32 rounding.
#include <ggml.h>
TEST_CASE("pet_lr_interp: precise LOG / SIN / COS / EXP on the GPU backend (flagged ops)", "[pet_lr_port]") {
  ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
  if (!gpu) {
    SUCCEED("no GPU backend");
    return;
  }
  const int n = 64;
  std::vector<float> w(n), a(n), e(n);
  for (int i = 0; i < n; i++) {
    w[i] = float(1.0 + std::pow(10.0, -7.0 + 6.0 * i / (n - 1))); // 1 + u, u in [1e-7, 1e-1]
    a[i] = float(-120.0 + 240.0 * i / (n - 1));
    e[i] = float(-30.0 + 40.0 * i / (n - 1));
  }
  ggml_context *ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
  ggml_tensor *tw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n), *ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n),
              *te = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
  ggml_tensor *outs[5] = {ggml_log(ctx, tw), ggml_sin(ctx, ta), ggml_cos(ctx, ta), ggml_exp(ctx, te), ggml_log(ctx, tw)};
  constexpr int slot = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
  for (int k = 0; k < 4; k++) outs[k]->op_params[slot] = 1; // outs[4]: the default (fast) log, for the record
  ggml_cgraph *g = ggml_new_graph(ctx);
  for (ggml_tensor *o : outs) ggml_set_output(o), ggml_build_forward_expand(g, o);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
  ggml_backend_tensor_set(tw, w.data(), 0, n * 4);
  ggml_backend_tensor_set(ta, a.data(), 0, n * 4);
  ggml_backend_tensor_set(te, e.data(), 0, n * 4);
  REQUIRE(ggml_backend_graph_compute(gpu, g) == GGML_STATUS_SUCCESS);
  std::vector<float> r[5];
  for (int k = 0; k < 5; k++) r[k].resize(n), ggml_backend_tensor_get(outs[k], r[k].data(), 0, n * 4);
  double rl = 0, rl_fast = 0, as = 0, ac = 0, re = 0;
  for (int i = 0; i < n; i++) {
    const double L = std::log(double(w[i]));
    rl = std::max(rl, std::fabs(r[0][i] - L) / L);
    rl_fast = std::max(rl_fast, std::fabs(r[4][i] - L) / L);
    as = std::max(as, std::fabs(r[1][i] - std::sin(double(a[i]))));
    ac = std::max(ac, std::fabs(r[2][i] - std::cos(double(a[i]))));
    re = std::max(re, std::fabs(r[3][i] - std::exp(double(e[i]))) / std::exp(double(e[i])));
  }
  INFO("log(1 + u) relative: flagged " << rl << ", default " << rl_fast << "; sin " << as << ", cos " << ac
                                       << " (absolute); exp relative " << re);
  CHECK(rl < 1.2e-7);
  CHECK(as < 1.2e-7);
  CHECK(ac < 1.2e-7);
  CHECK(re < 1.2e-7);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(gpu);
}
