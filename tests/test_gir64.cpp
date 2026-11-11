// fp64 executor (runtime/gir64.h, runtime/iris_fp64.h): op-level checks of every op the iris graphs use (reverse =
// central differences, forward mode = reverse, forward-over-reverse = differences of the reverse), and graph-level
// checks on the tiny seeded iris GGUFs (forces / virial / charge gradients = central differences of the fp64
// energies / charges, forward mode on the real graphs, the trainer's mixed derivative d/dtheta <dE/dx, v>).
#include "../src/runtime/gir64.h"
#include "../src/runtime/graph_model.h"
#include "../src/runtime/iris_fp64.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace mlipcpp;
using namespace mlipcpp::runtime;
using namespace mlipcpp::runtime::fp64;

namespace {
std::string resource(const char *name) { return std::string(MLIPCPP_TEST_RESOURCES) + "/" + name; }
using T64 = Tensor<double>;

std::string shp(const std::vector<int64_t> &s) {
  std::string r = "[";
  for (size_t i = 0; i < s.size(); i++) r += (i ? "," : "") + std::to_string(s[i]);
  return r + "]";
}

struct OpCase {
  std::string label;
  std::string nodes;                    // JSON array body of the nodes
  int out = 1;                          // output node id
  std::vector<int64_t> out_shape;
  std::map<std::string, T64> f, i;      // float inputs (differentiated), integer / mask inputs
  std::map<std::string, T64> w;         // weights (differentiated)
};

std::string graph_json(const OpCase &c) {
  std::string in;
  for (const auto *m : {&c.f, &c.i})
    for (const auto &[k, v] : *m)
      in += std::string(in.empty() ? "" : ",") + "{\"name\":\"" + k + "\",\"dtype\":\"" + (m == &c.f ? "f32" : "i32") +
            "\",\"shape\":" + shp(v.shape) + "}";
  return "{\"version\":\"1\",\"model_type\":\"test\",\"inputs\":[" + in + "],\"outputs\":[{\"name\":\"out\",\"node_ref\":"
         "\"node:" + std::to_string(c.out) + "\",\"dtype\":\"f32\",\"shape\":" + shp(c.out_shape) + "}],\"nodes\":[" +
         c.nodes + "]}";
}

T64 rnd(std::vector<int64_t> s, std::mt19937 &g, double lo = -1, double hi = 1) {
  T64 t = T64::zeros(s);
  std::uniform_real_distribution<double> u(lo, hi);
  for (auto &x : *t.data) x = u(g);
  return t;
}

Tensor<Dual> dual(const T64 &t, const std::vector<double> *tan = nullptr) {
  Tensor<Dual> d = Tensor<Dual>::zeros(t.shape);
  for (size_t k = 0; k < t.data->size(); k++) (*d.data)[k] = Dual((*t.data)[k], tan ? (*tan)[k] : 0.0);
  return d;
}

// reverse vs central differences, forward mode vs reverse, forward-over-reverse vs differences of the reverse
void check_op(const OpCase &c, double tol = 1e-7) {
  INFO(c.label);
  const GIRGraph g = parse_gir_json(graph_json(c));
  std::mt19937 gen(7);
  auto eval = [&](const std::map<std::string, T64> &f, const std::map<std::string, T64> &w, const std::vector<double> *cot,
                  std::map<std::string, std::vector<double>> *grads) {
    Executor<double> ex(g);
    for (const auto &[k, v] : f) ex.set_input(k, v);
    for (const auto &[k, v] : c.i) ex.set_input(k, v);
    for (const auto &[k, v] : w) ex.set_weight(k, v);
    ex.forward(true);
    const T64 &o = ex.output();
    REQUIRE(o.shape == c.out_shape);
    double L = 0;
    for (int64_t k = 0; k < o.numel(); k++) L += (*cot)[k] * o.ptr()[k];
    if (grads) {
      ex.backward(*cot, true);
      for (const auto &[k, v] : f) {
        auto gk = ex.input_grad(k);
        if (gk.empty()) gk.assign(v.numel(), 0.0);
        (*grads)[k] = gk;
      }
      for (const auto &[k, v] : w) {
        auto gk = ex.weight_grad(k);
        if (gk.empty()) gk.assign(v.numel(), 0.0);
        (*grads)["w:" + k] = gk;
      }
    }
    return L;
  };
  // the cotangent: random over the output
  std::vector<double> cot;
  {
    Executor<double> ex(g);
    for (const auto &[k, v] : c.f) ex.set_input(k, v);
    for (const auto &[k, v] : c.i) ex.set_input(k, v);
    for (const auto &[k, v] : c.w) ex.set_weight(k, v);
    ex.forward(true);
    std::uniform_real_distribution<double> u(-1, 1);
    cot.resize(ex.output().numel());
    for (auto &x : cot) x = u(gen);
  }
  std::map<std::string, std::vector<double>> G;
  eval(c.f, c.w, &cot, &G);
  const double h = 1e-6;
  auto deep = [](const std::map<std::string, T64> &m) {
    std::map<std::string, T64> r;
    for (const auto &[k, v] : m) r[k] = T64::from(v.shape, *v.data);
    return r;
  };
  // 1. reverse = central differences (every element of every float input and weight)
  for (int pass = 0; pass < 2; pass++) {
    const auto &src = pass == 0 ? c.f : c.w;
    for (const auto &[k, v] : src) {
      const auto &gk = G[(pass ? "w:" : "") + k];
      double gmax = 1e-12;
      for (double x : gk) gmax = std::max(gmax, std::fabs(x));
      for (int64_t e = 0; e < v.numel(); e++) {
        auto fp = deep(c.f), fm = deep(c.f), wp = deep(c.w), wm = deep(c.w);
        (pass == 0 ? fp : wp)[k].ptr()[e] += h;
        (pass == 0 ? fm : wm)[k].ptr()[e] -= h;
        const double fd = (eval(fp, wp, &cot, nullptr) - eval(fm, wm, &cot, nullptr)) / (2 * h);
        INFO((pass ? "weight " : "input ") << k << "[" << e << "] reverse " << gk[e] << " fd " << fd);
        CHECK(std::fabs(fd - gk[e]) <= tol * gmax + 1e-9);
      }
    }
  }
  // 2. forward mode: <cot, J v> = <J^T cot, v> (one random tangent on every float input and weight)
  std::map<std::string, std::vector<double>> V;
  std::uniform_real_distribution<double> u(-1, 1);
  for (const auto &[k, v] : c.f) {
    V[k].resize(v.numel());
    for (auto &x : V[k]) x = u(gen);
  }
  for (const auto &[k, v] : c.w) {
    V["w:" + k].resize(v.numel());
    for (auto &x : V["w:" + k]) x = u(gen);
  }
  Executor<Dual> dx(g);
  for (const auto &[k, v] : c.f) dx.set_input(k, dual(v, &V[k]));
  for (const auto &[k, v] : c.i) dx.set_input(k, dual(v));
  for (const auto &[k, v] : c.w) dx.set_weight(k, dual(v, &V["w:" + k]));
  dx.forward(true);
  double jvp = 0, vjp = 0, scale = 1e-12;
  for (int64_t k = 0; k < dx.output().numel(); k++) jvp += cot[k] * dx.output().ptr()[k].d;
  for (const auto &[k, gk] : G)
    for (size_t e = 0; e < gk.size(); e++) vjp += gk[e] * V[k][e], scale += std::fabs(gk[e] * V[k][e]);
  INFO("jvp " << jvp << " vjp " << vjp);
  CHECK(std::fabs(jvp - vjp) <= 1e-12 * scale + 1e-14);
  // 3. forward-over-reverse: tangent of the reverse gradients = d/de grad(x + e v) (central differences)
  std::vector<Dual> dcot(cot.begin(), cot.end());
  dx.backward(dcot, true);
  auto shifted = [&](double s) {
    auto f = deep(c.f), w = deep(c.w);
    for (auto &[k, v] : f)
      for (int64_t e = 0; e < v.numel(); e++) v.ptr()[e] += s * V[k][e];
    for (auto &[k, v] : w)
      for (int64_t e = 0; e < v.numel(); e++) v.ptr()[e] += s * V["w:" + k][e];
    std::map<std::string, std::vector<double>> gg;
    eval(f, w, &cot, &gg);
    return gg;
  };
  const double hh = 1e-5;
  auto gp = shifted(hh), gm = shifted(-hh);
  for (const auto &[k, gk] : G) {
    const bool wt = k.rfind("w:", 0) == 0;
    const auto &dg = wt ? dx.weight_grad(k.substr(2)) : dx.input_grad(k);
    double m = 1e-12;
    for (size_t e = 0; e < gk.size(); e++) m = std::max(m, std::fabs((gp[k][e] - gm[k][e]) / (2 * hh)));
    for (size_t e = 0; e < gk.size(); e++) {
      const double fd = (gp[k][e] - gm[k][e]) / (2 * hh), an = dg.empty() ? 0.0 : dg[e].d;
      INFO("hvp " << k << "[" << e << "] " << an << " fd " << fd);
      CHECK(std::fabs(fd - an) <= 1e-6 * m + 1e-8);
    }
  }
}

std::string node(int id, const std::string &op, const std::vector<std::string> &in, const std::vector<int64_t> &out,
                 const std::string &params = "{}") {
  std::string ins;
  for (const auto &s : in) ins += std::string(ins.empty() ? "" : ",") + "\"" + s + "\"";
  return "{\"id\":" + std::to_string(id) + ",\"op\":\"" + op + "\",\"name\":\"n" + std::to_string(id) +
         "\",\"inputs\":[" + ins + "],\"output_shape\":" + shp(out) + ",\"output_dtype\":\"f32\",\"params\":" + params + "}";
}
} // namespace

TEST_CASE("fp64 executor: every op of the iris graphs, reverse = central differences, JVP = VJP, HVP", "[fp64]") {
  std::mt19937 g(1);
  auto f = [&](std::vector<int64_t> s, double lo = -1, double hi = 1) { return rnd(std::move(s), g, lo, hi); };
  std::vector<OpCase> cases;
  // elementwise binaries with torch broadcasting, scalar forms
  for (const char *op : {"ADD", "SUB", "MUL", "DIV"}) {
    cases.push_back({std::string(op) + " broadcast", node(1, op, {"input:a", "input:b"}, {2, 3, 4}), 1, {2, 3, 4},
                     {{"a", f({2, 3, 4})}, {"b", f({3, 1}, 0.5, 1.5)}}, {}, {}});
    cases.push_back({std::string(op) + " scalar", node(1, op, {"input:a"}, {2, 3}, "{\"scalar\":1.7}"), 1, {2, 3},
                     {{"a", f({2, 3})}}, {}, {}});
  }
  cases.push_back({"MUL broadcast both", node(1, "MUL", {"input:a", "input:b"}, {2, 3, 4}), 1, {2, 3, 4},
                   {{"a", f({2, 1, 4})}, {"b", f({3, 1})}}, {}, {}});
  // unaries (domains away from kinks / poles)
  const std::vector<std::pair<std::string, std::pair<double, double>>> un = {
      {"SQRT", {0.5, 2}}, {"LOG", {0.5, 2}}, {"UNARY_EXP", {-1, 1}}, {"UNARY_NEG", {-1, 1}}, {"UNARY_ABS", {0.2, 1}},
      {"UNARY_RELU", {0.2, 1}}, {"UNARY_SIGMOID", {-2, 2}}, {"UNARY_SILU", {-2, 2}}, {"UNARY_TANH", {-2, 2}},
      {"UNARY_ERF", {-2, 2}}, {"COS", {-2, 2}}, {"SIN", {-2, 2}}, {"SQR", {-1, 1}}};
  for (const auto &[op, d] : un)
    cases.push_back({op, node(1, op, {"input:a"}, {3, 4}), 1, {3, 4}, {{"a", f({3, 4}, d.first, d.second)}}, {}, {}});
  cases.push_back({"UNARY_ABS negative", node(1, "UNARY_ABS", {"input:a"}, {5}), 1, {5}, {{"a", f({5}, -1, -0.2)}}, {}, {}});
  cases.push_back({"SCALE", node(1, "SCALE", {"input:a"}, {3}, "{\"scale\":-2.5}"), 1, {3}, {{"a", f({3})}}, {}, {}});
  for (double e : {2.0, 0.5, 3.0, -1.5})
    cases.push_back({"POW " + std::to_string(e), node(1, "POW", {"input:a"}, {4}, "{\"exponent\":" + std::to_string(e) + "}"),
                     1, {4}, {{"a", f({4}, 0.5, 2)}}, {}, {}});
  {
    T64 a = T64::from({6}, {-2.0, -0.4, 0.1, 0.6, 1.4, 3.0});
    cases.push_back({"CLAMP", node(1, "CLAMP", {"input:a"}, {6}, "{\"min\":-0.5,\"max\":1.0}"), 1, {6}, {{"a", a}}, {}, {}});
  }
  // matrix products
  cases.push_back({"MATMUL batched broadcast", node(1, "MATMUL", {"input:a", "input:b"}, {2, 2, 3, 5}), 1, {2, 2, 3, 5},
                   {{"a", f({2, 1, 3, 4})}, {"b", f({1, 2, 4, 5})}}, {}, {}});
  cases.push_back({"MATMUL matrix @ vector", node(1, "MATMUL", {"input:a", "input:b"}, {3}), 1, {3},
                   {{"a", f({3, 4})}, {"b", f({4})}}, {}, {}});
  cases.push_back({"MATMUL 2-D", node(1, "MATMUL", {"input:a", "input:b"}, {3, 2}), 1, {3, 2},
                   {{"a", f({3, 4})}, {"b", f({4, 2})}}, {}, {}});
  cases.push_back({"LINEAR + bias", node(1, "LINEAR", {"input:x", "weight:W", "weight:b"}, {2, 3, 5}), 1, {2, 3, 5},
                   {{"x", f({2, 3, 4})}}, {}, {{"W", f({5, 4})}, {"b", f({5})}}});
  // shape ops
  cases.push_back({"RESHAPE", node(1, "RESHAPE", {"input:a"}, {6, 2}, "{\"shape\":[6,2]}"), 1, {6, 2}, {{"a", f({3, 4})}}, {}, {}});
  cases.push_back({"VIEW", node(1, "VIEW", {"input:a"}, {4, 3}, "{\"shape\":[4,-1]}"), 1, {4, 3}, {{"a", f({3, 4})}}, {}, {}});
  cases.push_back({"SPLIT + VIEW index (chunk of the last dim)",
                   node(1, "SPLIT", {"input:a"}, {}, "{\"shape\":[2]}") + "," +
                       node(2, "VIEW", {"node:1"}, {3, 2}, "{\"index\":1,\"shape\":[3,2]}"),
                   2, {3, 2}, {{"a", f({3, 4})}}, {}, {}});
  cases.push_back({"CHUNK + VIEW index (rows)",
                   node(1, "CHUNK", {"input:a"}, {}, "{\"num_chunks\":2,\"dim\":-1}") + "," +
                       node(2, "VIEW", {"node:1"}, {4, 1, 3}, "{\"index\":0,\"shape\":[4,1,3]}"),
                   2, {4, 1, 3}, {{"a", f({4, 2, 3})}}, {}, {}});
  cases.push_back({"SELECT", node(1, "SELECT", {"input:a"}, {2, 4}, "{\"dim\":1,\"index\":2}"), 1, {2, 4},
                   {{"a", f({2, 3, 4})}}, {}, {}});
  cases.push_back({"SLICE", node(1, "SLICE", {"input:a"}, {2, 2, 4}, "{\"dim\":1,\"start\":1,\"end\":3}"), 1, {2, 2, 4},
                   {{"a", f({2, 3, 4})}}, {}, {}});
  cases.push_back({"SLICE open end", node(1, "SLICE", {"input:a"}, {2, 3}, "{\"dim\":0,\"start\":0,\"end\":9223372036854775807}"),
                   1, {2, 3}, {{"a", f({2, 3})}}, {}, {}});
  cases.push_back({"PERMUTE", node(1, "PERMUTE", {"input:a"}, {4, 2, 3}, "{\"axes\":[2,0,1]}"), 1, {4, 2, 3},
                   {{"a", f({2, 3, 4})}}, {}, {}});
  cases.push_back({"TRANSPOSE", node(1, "TRANSPOSE", {"input:a"}, {2, 4, 3}, "{\"dims\":[-2,-1]}"), 1, {2, 4, 3},
                   {{"a", f({2, 3, 4})}}, {}, {}});
  cases.push_back({"CONCAT", node(1, "CONCAT", {"input:a", "input:b"}, {2, 5}, "{\"dim\":1}"), 1, {2, 5},
                   {{"a", f({2, 3})}, {"b", f({2, 2})}}, {}, {}});
  cases.push_back({"SUM_ROWS dim", node(1, "SUM_ROWS", {"input:a"}, {2, 4}, "{\"dims\":[1],\"keepdim\":false}"), 1, {2, 4},
                   {{"a", f({2, 3, 4})}}, {}, {}});
  cases.push_back({"SUM_ROWS keepdim", node(1, "SUM_ROWS", {"input:a"}, {2, 3, 1}, "{\"dims\":[-1],\"keepdim\":true}"), 1,
                   {2, 3, 1}, {{"a", f({2, 3, 4})}}, {}, {}});
  cases.push_back({"SUM_ROWS to scalar", node(1, "SUM_ROWS", {"input:a"}, {}, "{\"keepdim\":false}"), 1, {},
                   {{"a", f({5})}}, {}, {}});
  cases.push_back({"REPEAT", node(1, "REPEAT", {"input:a"}, {3, 2, 4}, "{\"shape\":[3]}"), 1, {3, 2, 4},
                   {{"a", f({1, 2, 4})}}, {}, {}});
  cases.push_back({"SOFT_MAX", node(1, "SOFT_MAX", {"input:a"}, {2, 3, 5}, "{\"dim\":-1}"), 1, {2, 3, 5},
                   {{"a", f({2, 3, 5}, -3, 3)}}, {}, {}});
  cases.push_back({"LAYER_NORM", node(1, "LAYER_NORM", {"input:a", "weight:w", "weight:b"}, {3, 6}, "{\"eps\":1e-5}"),
                   1, {3, 6}, {{"a", f({3, 6})}}, {}, {{"w", f({6})}, {"b", f({6})}}});
  cases.push_back({"RMS_NORM (f32 eps -> f64 eps)",
                   node(1, "RMS_NORM", {"input:a", "weight:w"}, {3, 6}, "{\"eps\":1.1920928955078125e-07}"), 1, {3, 6},
                   {{"a", f({3, 6})}}, {}, {{"w", f({6})}}});
  // gathers / masks (integer and mask inputs are not differentiated)
  cases.push_back({"GET_ROWS", node(1, "GET_ROWS", {"weight:E", "input:idx"}, {2, 3, 4}), 1, {2, 3, 4}, {},
                   {{"idx", T64::from({2, 3}, {0, 2, 1, 2, 2, 0})}}, {{"E", f({3, 4})}}});
  cases.push_back({"INDEX 1-D", node(1, "INDEX", {"input:a", "input:idx"}, {5}), 1, {5}, {{"a", f({4})}},
                   {{"idx", T64::from({5}, {3, 0, 0, 2, 1})}}, {}});
  cases.push_back({"INDEX rows", node(1, "INDEX", {"input:a", "input:idx"}, {3, 2}), 1, {3, 2}, {{"a", f({4, 2})}},
                   {{"idx", T64::from({3}, {3, 1, 3})}}, {}});
  cases.push_back({"WHERE", node(1, "WHERE", {"input:c", "input:a", "input:b"}, {2, 3}), 1, {2, 3},
                   {{"a", f({2, 3})}, {"b", f({2, 3})}}, {{"c", T64::from({2, 3}, {1, 0, 0, 1, 1, 0})}}, {}});
  cases.push_back({"INDEX_PUT", node(1, "INDEX_PUT", {"input:a", "input:c", "input:v"}, {2, 3}), 1, {2, 3},
                   {{"a", f({2, 3})}, {"v", f({1})}}, {{"c", T64::from({2, 3}, {1, 0, 0, 1, 1, 0})}}, {}});
  cases.push_back({"LE / GE / BITWISE_NOT feed a WHERE",
                   node(1, "LE", {"input:a"}, {2, 3}, "{\"scalar\":0.0}") + "," +
                       node(2, "BITWISE_NOT", {"node:1"}, {2, 3}) + "," + node(3, "WHERE", {"node:2", "input:a", "input:b"}, {2, 3}) +
                       "," + node(4, "GE", {"input:b", "input:a"}, {2, 3}) + "," + node(5, "MUL", {"node:3", "node:4"}, {2, 3}),
                   5, {2, 3}, {{"a", f({2, 3})}, {"b", f({2, 3})}}, {}, {}});
  cases.push_back({"NEW_ONES + NEW_ZEROS", node(1, "NEW_ONES", {"input:a"}, {2, 3}, "{\"shape\":[2,3]}") + "," +
                                               node(2, "NEW_ZEROS", {"input:a"}, {2, 3}, "{\"shape\":[2,3]}") + "," +
                                               node(3, "ADD", {"node:1", "input:a"}, {2, 3}) + "," +
                                               node(4, "SUB", {"node:3", "node:2"}, {2, 3}),
                   4, {2, 3}, {{"a", f({2, 3})}}, {}, {}});
  for (const auto &c : cases) check_op(c);
}

TEST_CASE("fp64 executor: f32_intent recovers the exporter's decimals", "[fp64]") {
  CHECK(f32_intent(0.1f) == 0.1);
  CHECK(f32_intent(0.02f) == 0.02);
  CHECK(f32_intent(1e-5f) == 1e-5);
  CHECK(f32_intent(1e-6f) == 1e-6);
  CHECK(f32_intent(4.5f) == 4.5);
}

namespace {
// a small molecule-ish cluster (no species outside the tiny maps: H C N O)
System64 cluster(int n, uint32_t seed) {
  System64 s;
  s.n = n;
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> u(-0.15, 0.15);
  const int Z[4] = {8, 1, 6, 7};
  for (int i = 0; i < n; i++) { // a jittered 3 x 2 x 2 grid of spacing 1.4 A
    s.numbers.push_back(Z[i % 4]);
    const int ix[3] = {i % 3, (i / 3) % 2, i / 6};
    for (int c = 0; c < 3; c++) s.positions.push_back(1.4 * ix[c] + u(g));
  }
  return s;
}
// 24 waters in a 9 A cube (as test_pet_lr_port's water_box), as doubles; slab: the z row non-periodic
System64 water(bool slab) {
  System64 s;
  const double L = 9.0, a = L / 3.0;
  uint32_t r = 12345;
  auto rnd = [&r] { r = r * 1664525u + 1013904223u; return (r >> 8) / double(1u << 24) - 0.5; };
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) {
        if (s.positions.size() / 3 >= 72) break;
        const double o[3] = {(i + 0.5) * a + 0.3 * rnd(), (j + 0.5) * a + 0.3 * rnd(), (k + 0.5) * a + 0.3 * rnd()};
        const double h1[3] = {0.757, 0.586, 0.1 * rnd()}, h2[3] = {-0.757, 0.586, 0.1 * rnd()};
        for (int c = 0; c < 3; c++) s.positions.push_back(o[c]);
        for (int c = 0; c < 3; c++) s.positions.push_back(o[c] + h1[c]);
        for (int c = 0; c < 3; c++) s.positions.push_back(o[c] + h2[c]);
        s.numbers.insert(s.numbers.end(), {8, 1, 1});
      }
  s.n = int(s.numbers.size());
  s.has_cell = true;
  for (int c = 0; c < 9; c++) s.cell[c / 3][c % 3] = c % 4 == 0 ? L : 0.0;
  s.pbc[0] = s.pbc[1] = true, s.pbc[2] = !slab;
  if (slab) s.cell[2][2] = 30.0;
  return s;
}

const std::vector<std::pair<std::string, std::string>> &tiny_lr() {
  static const std::vector<std::pair<std::string, std::string>> v = {
      {"fermi", "petlr_tiny_fermi.gguf"},       {"fermi gapped T 0.02", "petlr_tiny_cold.gguf"},
      {"species", "petlr_tiny_species.gguf"},   {"both (Gaussian charges)", "petlr_tiny_both.gguf"},
      {"species_weighted", "petlr_tiny_weighted.gguf"}, {"species_fill", "petlr_tiny_fill.gguf"},
      {"species_fill + LRT", "petlr_tiny_lrt.gguf"},    {"conserving fermi", "petlr_tiny_ccfermi.gguf"},
      {"conserving residual", "petlr_tiny_ccresidual.gguf"}, {"qeq", "petlr_tiny_qeq.gguf"},
      {"threshold", "petlr_tiny_thr.gguf"},     {"conserving fermi + field (zero)", "petlr_tiny_ccfield.gguf"},
      {"les ladder + LR head", "petlr_les_ladder_lr.gguf"}, {"trunk only", "petlr_tiny_trunk.gguf"},
      {"conserving + induced dipoles (zero field)", "petlr_tiny_ccind.gguf"},
      {"les ladder (trunk)", "petlr_les_ladder.gguf"}};
  return v;
}

double rel_err(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 1e-300, d = 0;
  for (size_t k = 0; k < a.size(); k++) m = std::max(m, std::fabs(b[k])), d = std::max(d, std::fabs(a[k] - b[k]));
  return d / m;
}
} // namespace

TEST_CASE("fp64 executor: forces = -dE/dx and charge gradients = d(w.q)/dx (central differences), open, every tiny "
          "iris GGUF", "[fp64]") {
  for (const auto &[label, file] : tiny_lr()) {
    INFO(label);
    GraphModel m;
    REQUIRE(m.load_from_gguf(resource(file.c_str())));
    m.set_iris_qeq_solver(500, 1e-14); // a converged cg: the implicit derivative is the derivative of the solve
    IrisFp64 x(m);
    // cc: one component per structure (a compact cluster)
    const System64 s0 = cluster(10, 3);
    std::vector<double> w(s0.n);
    for (int i = 0; i < s0.n; i++) w[i] = std::sin(1.3 * i + 0.2);
    const bool lr = m.iris_lr().enabled, qeq = lr && m.iris_lr().reference == "qeq";
    if (lr && !qeq) x.request_column_gradient(1, w);
    const Result64 r = x.predict(s0, true);
    x.clear_column_gradients();
    REQUIRE(r.has_forces);
    const double h = 1e-5;
    std::vector<double> fd(3 * s0.n), gq(3 * s0.n);
    for (int k = 0; k < 3 * s0.n; k++) {
      System64 p = s0, q = s0;
      p.positions[k] += h, q.positions[k] -= h;
      const Result64 rp = x.predict(p, false), rq = x.predict(q, false);
      fd[k] = -(rp.energy - rq.energy) / (2 * h);
      if (lr) {
        double a = 0, b = 0;
        for (int i = 0; i < s0.n; i++) a += w[i] * rp.charges[i], b += w[i] * rq.charges[i];
        gq[k] = (a - b) / (2 * h);
      }
    }
    const double eF = rel_err(r.forces, fd);
    INFO("forces vs FD: " << eF);
    CHECK(eF < 1e-7);
    if (lr && !qeq) {
      const double eq = rel_err(r.column_gradients.at(0), gq);
      INFO("charge gradient vs FD: " << eq);
      CHECK(eq < 1e-7);
    }
    if (lr) {
      double qs = 0;
      for (double q : r.charges) qs += q;
      if (m.iris_lr().reference != "head") CHECK(std::fabs(qs - s0.charge) < 1e-10);
    }
  }
}

TEST_CASE("fp64 executor: periodic forces and the virial = dE/d eps (central differences, frozen Ewald parameters), "
          "3-D cell and slab", "[fp64]") {
  for (const auto &[label, file] : tiny_lr()) {
    for (bool slab : {false, true}) {
      INFO(label << (slab ? " slab" : " 3-D"));
      GraphModel m;
      REQUIRE(m.load_from_gguf(resource(file.c_str())));
      if (m.iris_lr().enabled && m.iris_lr().num_k == 0) continue;  // no periodic graph
      if (m.iris_lr().enabled && m.iris_lr().conserving_charges) continue; // one component only: a box has many
      if (slab && m.iris_lr().long_range_transfer) continue;               // his ToSample raises on a slab
      if (m.iris_lr().field) continue;
      m.set_iris_qeq_solver(500, 1e-14);
      m.set_iris_freeze_ewald(true);
      IrisFp64 x(m);
      const System64 s0 = water(slab);
      const Result64 r = x.predict(s0, true);
      REQUIRE(r.has_stress);
      const double h = 1e-5;
      // forces on a handful of coordinates
      double ef = 0, fm = 0;
      for (int k : {0, 4, 17, 50, 133, 200}) {
        System64 p = s0, q = s0;
        p.positions[k] += h, q.positions[k] -= h;
        const double fd = -(x.predict(p, false).energy - x.predict(q, false).energy) / (2 * h);
        ef = std::max(ef, std::fabs(fd - r.forces[k]));
      }
      for (double v : r.forces) fm = std::max(fm, std::fabs(v));
      INFO("forces vs FD " << ef / fm);
      CHECK(ef / fm < 1e-7);
      // virial: x -> x (I + eps), h -> h (I + eps) (slab: the open row too; the frozen effective cell holds). Step
      // 1e-7: his real-space Ewald sum is truncated at r < rc (erfc(rc / sqrt2 s) ~ 6e-5, a jump), and a strain of
      // 1e-5 moves ~1e-4 A at rc, where some of the ~15k pairs of the box cross it (FD errors ~1e-3 relative)
      const double hs = 1e-7;
      double ew = 0, wm = 0;
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) {
          auto strained = [&](double e) {
            System64 t = s0;
            for (int i = 0; i < t.n; i++) t.positions[3 * i + b] += e * s0.positions[3 * i + a];
            for (int c = 0; c < 3; c++) t.cell[c][b] += e * s0.cell[c][a];
            return x.predict(t, false).energy;
          };
          const double fd = (strained(hs) - strained(-hs)) / (2 * hs);
          ew = std::max(ew, std::fabs(fd - r.virial[3 * a + b]));
          wm = std::max(wm, std::fabs(r.virial[3 * a + b]));
        }
      INFO("virial vs FD " << ew / wm << " (max |W| " << wm << ")");
      CHECK(ew / wm < 1e-6);
    }
  }
}

TEST_CASE("fp64 executor: forward mode on the real graphs (JVP = VJP) and the trainer's mixed derivative "
          "d/dtheta <dE/dx, v> (forward-over-reverse = differences of the weight gradients)", "[fp64]") {
  for (const char *file : {"petlr_tiny_fermi.gguf", "petlr_tiny_both.gguf", "petlr_tiny_trunk.gguf"}) {
    INFO(file);
    GraphModel m;
    REQUIRE(m.load_from_gguf(resource(file)));
    IrisFp64 x(m);
    x.predict(cluster(9, 5), false);
    const Snapshot &sn = x.last();
    REQUIRE(sn.graph);
    const GIRGraph &G = *sn.graph;
    std::mt19937 gen(11);
    std::uniform_real_distribution<double> u(-1, 1);
    // tangents on the edge vectors (real slots) and the positions, as the trainer seeds its JVP
    std::map<std::string, std::vector<double>> V;
    const T64 &pm = sn.inputs.at("padding_mask");
    for (const char *k : {"edge_vectors", "positions", "cutoff_values"}) {
      if (!sn.inputs.count(k)) continue;
      const T64 &t = sn.inputs.at(k);
      V[k].resize(t.numel());
      for (int64_t e = 0; e < t.numel(); e++) {
        const bool pad = std::string(k) != "positions" && pm.ptr()[std::string(k) == "edge_vectors" ? e / 3 : e] > 0.5;
        V[k][e] = pad ? 0.0 : u(gen);
      }
    }
    auto setup = [&](auto &ex, double shift) {
      using E = std::decay_t<decltype(ex)>;
      constexpr bool dual = std::is_same_v<E, Executor<Dual>>;
      for (const auto &[k, v] : sn.dims) ex.set_dimension(k, v);
      for (const auto &[k, w] : x.weights()) {
        if constexpr (dual) ex.set_weight(k, ::dual(w));
        else ex.set_weight(k, w);
      }
      for (const auto &[k, t] : sn.inputs) {
        T64 c = T64::from(t.shape, *t.data);
        if (V.count(k))
          for (int64_t e = 0; e < c.numel(); e++) c.ptr()[e] += shift * V[k][e];
        if constexpr (dual) ex.set_input(k, ::dual(c, V.count(k) ? &V[k] : nullptr));
        else ex.set_input(k, c);
      }
    };
    const int C = m.iris_lr().enabled ? m.iris_lr().n_out : 1;
    Executor<double> ex(G);
    setup(ex, 0.0);
    ex.forward(true);
    const int64_t n = ex.output().numel() / C;
    std::vector<double> cot(ex.output().numel(), 0.0);
    for (int64_t i = 0; i < n; i++) cot[i * C] = 1.0;
    ex.backward(cot, true);
    double vjp = 0, sc = 1e-300;
    for (const auto &[k, v] : V) {
      const auto &gk = ex.input_grad(k);
      for (size_t e = 0; e < v.size() && !gk.empty(); e++) vjp += gk[e] * v[e], sc += std::fabs(gk[e] * v[e]);
    }
    Executor<Dual> dx(G);
    setup(dx, 0.0);
    dx.forward(true);
    double jvp = 0;
    for (int64_t i = 0; i < n; i++) jvp += dx.output().ptr()[i * C].d;
    INFO("JVP " << jvp << " VJP " << vjp);
    CHECK(std::fabs(jvp - vjp) <= 1e-12 * sc);
    // d/dtheta <dE/dx, v>: the weight gradients' tangents vs central differences along v
    std::vector<Dual> dcot(cot.begin(), cot.end());
    dx.backward(dcot, true);
    const double h = 1e-5;
    Executor<double> ep(G), em(G);
    setup(ep, h), setup(em, -h);
    ep.forward(true), em.forward(true);
    ep.backward(cot, true), em.backward(cot, true);
    double err = 0, mag = 1e-300;
    for (const auto &[k, w] : x.weights()) {
      const auto &a = dx.weight_grad(k);
      const auto &gp = ep.weight_grad(k), &gm = em.weight_grad(k);
      if (gp.empty()) continue;
      for (size_t e = 0; e < gp.size(); e++) {
        const double fd = (gp[e] - gm[e]) / (2 * h), an = a.empty() ? 0.0 : a[e].d;
        err = std::max(err, std::fabs(fd - an)), mag = std::max(mag, std::fabs(fd));
      }
    }
    INFO("mixed derivative vs FD: " << err / mag << " (max " << mag << ")");
    CHECK(err / mag < 1e-6);
  }
}
