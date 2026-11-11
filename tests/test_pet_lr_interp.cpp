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

// ggml-cuda-precise-sigmoid-div.patch: SIGMOID (the Fermi occupations, arguments (s - mu) / T up to ~50 at T = 0.02)
// in double and DIV as IEEE round-to-nearest (__fdiv_rn) when flagged; the default kernels keep __expf (error growing
// with |x|) and the 2-ulp approximate division of -use_fast_math. Found by the per-node fp64 oracle
// (compare_node_grads_iris.py): CUDA's sigmoid of the occupation 4-10x the local error of a correctly rounded fp32 op.
TEST_CASE("pet_lr_interp: precise SIGMOID / DIV on the GPU backend (flagged ops)", "[pet_lr_port]") {
  ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
  if (!gpu) {
    SUCCEED("no GPU backend");
    return;
  }
  const int n = 256;
  std::vector<float> x(n), y(n);
  for (int i = 0; i < n; i++) {
    x[i] = float(-60.0 + 120.0 * i / (n - 1) + 1e-3 * std::sin(7.0 * i)); // sigmoid(x) and x / y
    y[i] = float(0.0173 + 0.37 * std::fabs(std::cos(3.0 * i)));
  }
  ggml_context *ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
  ggml_tensor *tx = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n), *ty = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
  ggml_tensor *outs[4] = {ggml_sigmoid(ctx, tx), ggml_div(ctx, tx, ty), ggml_sigmoid(ctx, tx), ggml_div(ctx, tx, ty)};
  constexpr int slot = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
  for (int k = 0; k < 2; k++) outs[k]->op_params[slot] = 1; // outs[2, 3]: the default (fast) kernels, for the record
  ggml_cgraph *g = ggml_new_graph(ctx);
  for (ggml_tensor *o : outs) ggml_set_output(o), ggml_build_forward_expand(g, o);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
  ggml_backend_tensor_set(tx, x.data(), 0, n * 4);
  ggml_backend_tensor_set(ty, y.data(), 0, n * 4);
  REQUIRE(ggml_backend_graph_compute(gpu, g) == GGML_STATUS_SUCCESS);
  std::vector<float> r[4];
  for (int k = 0; k < 4; k++) r[k].resize(n), ggml_backend_tensor_get(outs[k], r[k].data(), 0, n * 4);
  double rs = 0, rs_fast = 0, rd = 0, rd_fast = 0;
  int div_exact = 0;
  for (int i = 0; i < n; i++) {
    const double S = 1.0 / (1.0 + std::exp(-double(x[i])));
    rs = std::max(rs, std::fabs(r[0][i] - S) / S), rs_fast = std::max(rs_fast, std::fabs(r[2][i] - S) / S);
    const double D = double(x[i]) / double(y[i]);
    rd = std::max(rd, std::fabs(r[1][i] - D) / std::fabs(D)), rd_fast = std::max(rd_fast, std::fabs(r[3][i] - D) / std::fabs(D));
    div_exact += r[1][i] == x[i] / y[i]; // the CPU's (correctly rounded) fp32 division
  }
  INFO("sigmoid relative: flagged " << rs << ", default " << rs_fast << "; division relative: flagged " << rd
                                    << ", default " << rd_fast << "; flagged = CPU fp32 division on " << div_exact << "/" << n);
  CHECK(rs < 1.2e-7);
  CHECK(rd < 6.1e-8);
  CHECK(div_exact == n);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(gpu);
}

// ggml-cuda-precise-sqrt.patch: SQRT as IEEE round-to-nearest (__fsqrt_rn) when flagged; the default kernel is the
// approximate square root of -use_fast_math (-prec-sqrt=false). Found by the signed energy budget of the per-node fp64
// oracle (compare_node_grads_iris.py --budget): the edge-length SQRT of the les-water ladder presets carried a
// systematic CUDA-only energy shift (7x the CPU's signed contribution).
TEST_CASE("pet_lr_interp: precise SQRT on the GPU backend (flagged op)", "[pet_lr_port]") {
  ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
  if (!gpu) {
    SUCCEED("no GPU backend");
    return;
  }
  const int n = 4096;
  std::vector<float> x(n);
  for (int i = 0; i < n; i++) x[i] = float(0.25 + 36.0 * i / (n - 1) + 1e-4 * std::sin(13.0 * i)); // r^2 in [0.25, 36]
  ggml_context *ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
  ggml_tensor *tx = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
  ggml_tensor *outs[2] = {ggml_sqrt(ctx, tx), ggml_sqrt(ctx, tx)};
  constexpr int slot = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
  outs[0]->op_params[slot] = 1; // outs[1]: the default (fast) kernel, for the record
  ggml_cgraph *g = ggml_new_graph(ctx);
  for (ggml_tensor *o : outs) ggml_set_output(o), ggml_build_forward_expand(g, o);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
  ggml_backend_tensor_set(tx, x.data(), 0, n * 4);
  REQUIRE(ggml_backend_graph_compute(gpu, g) == GGML_STATUS_SUCCESS);
  std::vector<float> r[2];
  for (int k = 0; k < 2; k++) r[k].resize(n), ggml_backend_tensor_get(outs[k], r[k].data(), 0, n * 4);
  int exact = 0, exact_fast = 0;
  double bias_fast = 0;
  for (int i = 0; i < n; i++) {
    const float c = std::sqrt(x[i]); // the CPU's correctly rounded fp32 sqrt
    exact += r[0][i] == c, exact_fast += r[1][i] == c;
    bias_fast += (double(r[1][i]) - std::sqrt(double(x[i]))) / std::sqrt(double(x[i]));
  }
  INFO("flagged = CPU sqrtf on " << exact << "/" << n << "; default on " << exact_fast << "/" << n
                                 << " (mean relative error of the default " << bias_fast / n << ")");
  CHECK(exact == n);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(gpu);
}

// ggml-cuda-precise-sum.patch: SUM / SUM_ROWS accumulate in double when flagged (ggml-cpu always does: ggml_float in
// ggml_vec_sum_f32); the default CUDA kernels accumulate in float (cub / the warp-tree reduce_rows). Candidate from the
// per-node fp64 oracle: the Ewald / Coulomb potential sums of the charge path at 2.5-7x the local error of a torch fp32
// sum on CUDA. Rows with cancellation (alternating large terms): flagged = the host double sum rounded once, on every row.
TEST_CASE("pet_lr_interp: precise SUM / SUM_ROWS on the GPU backend (flagged ops)", "[pet_lr_port]") {
  ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
  if (!gpu) {
    SUCCEED("no GPU backend");
    return;
  }
  const int nc = 1500, nr = 300; // ncols >= 1024: the 128-thread default kernel, and nr / nsm < 2: the 512-thread one
  std::vector<float> x(size_t(nc) * nr);
  for (int r = 0; r < nr; r++)
    for (int c = 0; c < nc; c++)
      x[size_t(r) * nc + c] = float((c % 2 ? -1.0 : 1.0) * (1.0 + 1e3 * std::fabs(std::sin(0.37 * c + r))) +
                                    1e-2 * std::cos(1.3 * c * (r + 1)));
  ggml_context *ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
  ggml_tensor *tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
  ggml_tensor *outs[4] = {ggml_sum_rows(ctx, tx), ggml_sum(ctx, tx), ggml_sum_rows(ctx, tx), ggml_sum(ctx, tx)};
  constexpr int slot = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
  for (int k = 0; k < 2; k++) outs[k]->op_params[slot] = 1; // outs[2, 3]: the default (float) kernels, for the record
  ggml_cgraph *g = ggml_new_graph(ctx);
  for (ggml_tensor *o : outs) ggml_set_output(o), ggml_build_forward_expand(g, o);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
  ggml_backend_tensor_set(tx, x.data(), 0, x.size() * 4);
  REQUIRE(ggml_backend_graph_compute(gpu, g) == GGML_STATUS_SUCCESS);
  std::vector<float> rows(nr), rows_fast(nr);
  float tot = 0, tot_fast = 0;
  ggml_backend_tensor_get(outs[0], rows.data(), 0, nr * 4);
  ggml_backend_tensor_get(outs[2], rows_fast.data(), 0, nr * 4);
  ggml_backend_tensor_get(outs[1], &tot, 0, 4);
  ggml_backend_tensor_get(outs[3], &tot_fast, 0, 4);
  int exact = 0, exact_fast = 0;
  double all = 0;
  for (int r = 0; r < nr; r++) {
    double s = 0;
    for (int c = 0; c < nc; c++) s += double(x[size_t(r) * nc + c]);
    all += s;
    exact += rows[r] == float(s), exact_fast += rows_fast[r] == float(s);
  }
  INFO("SUM_ROWS = host double sum rounded: flagged " << exact << "/" << nr << ", default " << exact_fast << "/" << nr
                                                      << "; SUM flagged " << tot << " default " << tot_fast << " host " << float(all));
  CHECK(exact == nr);
  CHECK(tot == float(all));
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(gpu);
}

TEST_CASE("pet_lr_interp: precise MUL_MAT (double accumulation) on the GPU backend, K = 1036 (flagged ops)",
          "[pet_lr_port]") {
  // ggml-cuda-precise-mulmat.patch: an F32 MUL_MAT flagged precise (op_params[last] = 1; graph_model.cpp under
  // MLIPCPP_PRECISE_MATMUL / MLIPCPP_PRECISE_MATMUL_K, opt-in) sums the K products in double and rounds once. K = 1036
  // is the transfer MLP's first LINEAR (2 * 512 + 8 + 4 inputs), where the default cuBLAS / mmf GEMM's float
  // reduction carried the CUDA charge excess. Checked: flagged = the host double sum rounded, on every element, for
  // a plain product, a broadcast batch (src0 shared across ne2) and a transposed (non-contiguous) src1 view; the
  // default kernel is recorded, not required
  ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
  if (!gpu) {
    SUCCEED("no GPU backend");
    return;
  }
  const int K = 1036, M = 64, N = 40, B = 3;
  auto val = [](int a, int b, double s) {
    return float((((a * 7 + b) % 3) - 1) * (1.0 + 1e2 * std::fabs(std::sin(s * a + 0.11 * b))) + 1e-3 * std::cos(a + 2.0 * b));
  };
  std::vector<float> w(size_t(K) * M), x(size_t(K) * N * B), xt(size_t(N) * K);
  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++) w[size_t(m) * K + k] = val(k, m, 0.37);
  for (int b = 0; b < B; b++)
    for (int n = 0; n < N; n++)
      for (int k = 0; k < K; k++) x[(size_t(b) * N + n) * K + k] = val(k + 13 * b, n, 0.71);
  for (int n = 0; n < N; n++) // xt [K, N] row-major: element (k, n) at k * N + n; viewed transposed as [K, N]
    for (int k = 0; k < K; k++) xt[size_t(k) * N + n] = x[size_t(n) * K + k];
  ggml_context *ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
  ggml_tensor *tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
  ggml_tensor *tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, B);
  ggml_tensor *txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, K);
  ggml_tensor *txv = ggml_transpose(ctx, txt); // [K, N], nb[0] = N * 4: non-contiguous
  ggml_tensor *outs[4] = {ggml_mul_mat(ctx, tw, tx), ggml_mul_mat(ctx, tw, txv), ggml_mul_mat(ctx, tw, tx),
                          ggml_mul_mat(ctx, tw, txv)};
  constexpr int slot = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
  outs[0]->op_params[slot] = 1, outs[1]->op_params[slot] = 1; // outs[2, 3]: the default kernels, for the record
  ggml_cgraph *g = ggml_new_graph(ctx);
  for (int i = 0; i < 4; i++) ggml_set_output(outs[i]), ggml_build_forward_expand(g, outs[i]);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
  ggml_backend_tensor_set(tw, w.data(), 0, w.size() * 4);
  ggml_backend_tensor_set(tx, x.data(), 0, x.size() * 4);
  ggml_backend_tensor_set(txt, xt.data(), 0, xt.size() * 4);
  REQUIRE(ggml_backend_graph_compute(gpu, g) == GGML_STATUS_SUCCESS);
  std::vector<float> o0(size_t(M) * N * B), o1(size_t(M) * N), o2(o0.size()), o3(o1.size());
  ggml_backend_tensor_get(outs[0], o0.data(), 0, o0.size() * 4);
  ggml_backend_tensor_get(outs[1], o1.data(), 0, o1.size() * 4);
  ggml_backend_tensor_get(outs[2], o2.data(), 0, o2.size() * 4);
  ggml_backend_tensor_get(outs[3], o3.data(), 0, o3.size() * 4);
  int exact = 0, exact_t = 0, exact_fast = 0, exact_fast_t = 0;
  double worst_fast = 0;
  for (int b = 0; b < B; b++)
    for (int n = 0; n < N; n++)
      for (int m = 0; m < M; m++) {
        double s = 0;
        for (int k = 0; k < K; k++) s += double(w[size_t(m) * K + k]) * double(x[(size_t(b) * N + n) * K + k]);
        const size_t i = (size_t(b) * N + n) * M + m; // dst [M, N, B]
        exact += o0[i] == float(s), exact_fast += o2[i] == float(s);
        worst_fast = std::max(worst_fast, std::fabs(double(o2[i]) - s) / std::max(std::fabs(s), 1e-30));
        if (b == 0) exact_t += o1[i] == float(s), exact_fast_t += o3[i] == float(s);
      }
  INFO("flagged = host double sum rounded: " << exact << "/" << M * N * B << " (batched), " << exact_t << "/" << M * N
       << " (transposed src1); default kernel " << exact_fast << "/" << M * N * B << ", " << exact_fast_t << "/" << M * N
       << ", worst relative error " << worst_fast);
  CHECK(exact == M * N * B);
  CHECK(exact_t == M * N);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  ggml_backend_free(gpu);
}
