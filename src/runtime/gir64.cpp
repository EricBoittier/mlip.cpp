// fp64 executor of a GIR graph (see gir64.h). Torch semantics on row-major tensors; every op checks its result
// shape against the node's output_shape, and an op or parameter outside the inventory below throws (no silent
// fallback to fp32 anywhere).
#include "gir64.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mlipcpp::runtime::fp64 {

using std::cos;
using std::erf;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
using std::tanh;

double f32_intent(float x) {
  if (!std::isfinite(x) || x == 0.0f) return double(x);
  char buf[64];
  for (int p = 1; p <= 9; p++) {
    std::snprintf(buf, sizeof buf, "%.*g", p, double(x));
    if (std::strtof(buf, nullptr) == x) return std::strtod(buf, nullptr);
  }
  return double(x);
}

enum class Op {
  ADD, SUB, MUL, DIV, MATMUL, LINEAR, RESHAPE, VIEW, SELECT, PERMUTE, TRANSPOSE, CONT, SCALE, SQR, SQRT, LOG,
  SUM_ROWS, REPEAT, CLAMP, SOFT_MAX, SILU, RELU, ERF, TANH, EXP, NEG, ABS, SIGMOID, LAYER_NORM, RMS_NORM, CONCAT,
  GET_ROWS, NEW_ZEROS, NEW_ONES, SLICE, SPLIT, CHUNK, BITWISE_NOT, INDEX, INDEX_PUT, LE, GE, WHERE, COS, SIN, POW,
  MASKED_FILL, IDENTITY
};

struct NodeInfo {
  Op op;
  bool has_scalar = false;
  double scalar = 0, eps = 1e-5, exponent = 2, lo = -INFINITY, hi = INFINITY, scale = 1, fill = 0;
  int64_t dim = 0, index = -1, start = 0, end = std::numeric_limits<int64_t>::max();
  bool has_dim = false, has_start = false, has_end = false, has_dims = false, keepdim = false;
  std::vector<int64_t> shape, axes, dims;
  bool shape_has_negative = false;
};

namespace {

Op parse_op(const GIRNode &n) {
  static const std::map<std::string, Op> m = {
      {"ADD", Op::ADD}, {"SUB", Op::SUB}, {"MUL", Op::MUL}, {"DIV", Op::DIV}, {"MATMUL", Op::MATMUL},
      {"LINEAR", Op::LINEAR}, {"RESHAPE", Op::RESHAPE}, {"VIEW", Op::VIEW}, {"SELECT", Op::SELECT},
      {"PERMUTE", Op::PERMUTE}, {"TRANSPOSE", Op::TRANSPOSE}, {"CONT", Op::CONT}, {"SCALE", Op::SCALE},
      {"SQR", Op::SQR}, {"SQRT", Op::SQRT}, {"LOG", Op::LOG}, {"SUM_ROWS", Op::SUM_ROWS}, {"REPEAT", Op::REPEAT},
      {"CLAMP", Op::CLAMP}, {"SOFT_MAX", Op::SOFT_MAX}, {"UNARY_SILU", Op::SILU}, {"UNARY_RELU", Op::RELU},
      {"UNARY_ERF", Op::ERF}, {"UNARY_TANH", Op::TANH}, {"UNARY_EXP", Op::EXP}, {"UNARY_NEG", Op::NEG},
      {"UNARY_ABS", Op::ABS}, {"UNARY_SIGMOID", Op::SIGMOID}, {"LAYER_NORM", Op::LAYER_NORM},
      {"RMS_NORM", Op::RMS_NORM}, {"CONCAT", Op::CONCAT}, {"GET_ROWS", Op::GET_ROWS}, {"NEW_ZEROS", Op::NEW_ZEROS},
      {"NEW_ONES", Op::NEW_ONES}, {"SLICE", Op::SLICE}, {"SPLIT", Op::SPLIT}, {"CHUNK", Op::CHUNK},
      {"BITWISE_NOT", Op::BITWISE_NOT}, {"INDEX", Op::INDEX}, {"INDEX_PUT", Op::INDEX_PUT}, {"LE", Op::LE},
      {"GE", Op::GE}, {"WHERE", Op::WHERE}, {"COS", Op::COS}, {"SIN", Op::SIN}, {"POW", Op::POW},
      {"MASKED_FILL", Op::MASKED_FILL}};
  if (n.op == "DECOMPOSE" && n.name.find("norm") != std::string::npos) return Op::LAYER_NORM;
  auto it = m.find(n.op);
  if (it == m.end())
    throw std::runtime_error("fp64 executor: op " + n.op + " (node " + std::to_string(n.id) + " \"" + n.name +
                             "\") is not supported");
  return it->second;
}

bool has(const GIRNode &n, const char *k) { return n.params.count(k) > 0; }
double pdouble(const GIRNode &n, const char *k, double d) {
  auto it = n.params.find(k);
  if (it == n.params.end()) return d;
  if (auto p = std::get_if<double>(&it->second)) return *p;
  if (auto p = std::get_if<int64_t>(&it->second)) return double(*p);
  if (auto p = std::get_if<std::string>(&it->second)) { // "-inf" / "inf"
    if (*p == "-inf") return -INFINITY;
    if (*p == "inf") return INFINITY;
  }
  return d;
}
int64_t pint(const GIRNode &n, const char *k, int64_t d) {
  auto it = n.params.find(k);
  if (it == n.params.end()) return d;
  if (auto p = std::get_if<int64_t>(&it->second)) return *p;
  if (auto p = std::get_if<double>(&it->second)) { // saturate (INT64_MAX, a slice's open end, arrives as a double)
    if (*p >= 9.2e18) return std::numeric_limits<int64_t>::max();
    if (*p <= -9.2e18) return std::numeric_limits<int64_t>::min();
    return int64_t(*p);
  }
  return d;
}
std::vector<int64_t> pints(const GIRNode &n, const char *k) {
  auto it = n.params.find(k);
  if (it == n.params.end()) return {};
  if (auto p = std::get_if<std::vector<int64_t>>(&it->second)) return *p;
  if (auto p = std::get_if<int64_t>(&it->second)) return {*p};
  return {};
}

NodeInfo parse_info(const GIRNode &n) {
  NodeInfo I;
  I.op = parse_op(n);
  I.has_scalar = has(n, "scalar");
  I.scalar = pdouble(n, "scalar", 0);
  I.eps = pdouble(n, "eps", 1e-5);
  I.exponent = pdouble(n, "exponent", 2);
  I.lo = pdouble(n, "min", -INFINITY);
  I.hi = pdouble(n, "max", INFINITY);
  I.scale = pdouble(n, "scale", 1);
  I.fill = pdouble(n, "value", 0);
  if (has(n, "fill")) {
    auto it = n.params.find("fill");
    if (auto p = std::get_if<std::string>(&it->second)) I.fill = *p == "-inf" ? -INFINITY : std::stod(*p);
    else I.fill = pdouble(n, "fill", 0);
  }
  I.has_dim = has(n, "dim");
  I.dim = pint(n, "dim", 0);
  I.index = pint(n, "index", -1);
  I.has_start = has(n, "start"), I.has_end = has(n, "end");
  I.start = pint(n, "start", 0);
  I.end = pint(n, "end", std::numeric_limits<int64_t>::max());
  I.has_dims = has(n, "dims");
  I.dims = pints(n, "dims");
  I.keepdim = false;
  if (auto it = n.params.find("keepdim"); it != n.params.end())
    if (auto p = std::get_if<bool>(&it->second)) I.keepdim = *p;
  I.shape = pints(n, "shape");
  for (auto d : I.shape) I.shape_has_negative |= d < 0;
  I.axes = pints(n, "axes");
  return I;
}

std::string shape_str(const std::vector<int64_t> &s) {
  std::ostringstream o;
  o << "[";
  for (size_t i = 0; i < s.size(); i++) o << (i ? ", " : "") << s[i];
  o << "]";
  return o.str();
}

int64_t numel(const std::vector<int64_t> &s) {
  int64_t n = 1;
  for (auto x : s) n *= x;
  return n;
}

std::array<int64_t, 4> pad4(const std::vector<int64_t> &s) {
  if (s.size() > 4) throw std::runtime_error("fp64 executor: rank > 4 " + shape_str(s));
  std::array<int64_t, 4> r{1, 1, 1, 1};
  for (size_t i = 0; i < s.size(); i++) r[4 - s.size() + i] = s[i];
  return r;
}

std::vector<int64_t> bshape(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
  const size_t r = std::max(a.size(), b.size());
  std::vector<int64_t> o(r);
  for (size_t i = 0; i < r; i++) {
    const int64_t x = i < r - a.size() ? 1 : a[i - (r - a.size())];
    const int64_t y = i < r - b.size() ? 1 : b[i - (r - b.size())];
    const int64_t m = std::max(x, y);
    if (m % x || m % y) throw std::runtime_error("fp64 executor: shapes " + shape_str(a) + " and " + shape_str(b) +
                                                 " do not broadcast");
    o[i] = m;
  }
  return o;
}

// operand index map into a broadcast output (right-aligned; size 1 broadcasts, a divisor tiles)
struct BIdx {
  std::array<int64_t, 4> n, st;
  BIdx(const std::vector<int64_t> &s) {
    n = pad4(s);
    st[3] = 1;
    for (int d = 2; d >= 0; d--) st[d] = st[d + 1] * n[d + 1];
  }
  int64_t at(int d, int64_t i) const { return (n[d] == 1 ? 0 : (i % n[d])) * st[d]; }
};

bool par_ok(int64_t n) { return n > 32768; }

// f(o, ia, ib, ic) over the output; operands may be broadcast
template <class F>
void bcast3(const std::vector<int64_t> &os, const std::vector<int64_t> &as, const std::vector<int64_t> &bs,
            const std::vector<int64_t> &cs, F f, bool parallel = true) {
  const auto on = pad4(os);
  BIdx A(as), B(bs), C(cs);
  const int64_t N = numel(os);
  const int64_t inner = on[3];
#pragma omp parallel for collapse(3) schedule(static) if (parallel && par_ok(N))
  for (int64_t i0 = 0; i0 < on[0]; i0++)
    for (int64_t i1 = 0; i1 < on[1]; i1++)
      for (int64_t i2 = 0; i2 < on[2]; i2++) {
        const int64_t o = ((i0 * on[1] + i1) * on[2] + i2) * inner;
        const int64_t a0 = A.at(0, i0) + A.at(1, i1) + A.at(2, i2);
        const int64_t b0 = B.at(0, i0) + B.at(1, i1) + B.at(2, i2);
        const int64_t c0 = C.at(0, i0) + C.at(1, i1) + C.at(2, i2);
        for (int64_t i3 = 0; i3 < inner; i3++) f(o + i3, a0 + A.at(3, i3), b0 + B.at(3, i3), c0 + C.at(3, i3));
      }
}

// sum a broadcast output-shaped gradient back onto an operand of shape s (serial over the output: the targets
// collide; parallel over the operand's own elements when it is not broadcast)
template <class T>
void reduce_to(const std::vector<int64_t> &os, const std::vector<int64_t> &s, const T *g, T *out,
               const std::vector<int64_t> &dummy = {}) {
  (void)dummy;
  if (os == s || numel(os) == numel(s)) {
    const int64_t N = numel(s);
#pragma omp parallel for schedule(static) if (par_ok(N))
    for (int64_t i = 0; i < N; i++) out[i] += g[i];
    return;
  }
  bcast3(os, s, s, s, [&](int64_t o, int64_t ia, int64_t, int64_t) { out[ia] += g[o]; }, false);
}

// C[m, n] (+)= op(A) op(B), row-major; ta: A is stored [k, m]; tb: B is stored [n, k]
template <class T>
void gemm(int64_t m, int64_t n, int64_t k, const T *A, bool ta, const T *B, bool tb, T *C, bool parallel) {
  if (!ta && !tb) {
#pragma omp parallel for schedule(static) if (parallel && m * n * k > 65536)
    for (int64_t i = 0; i < m; i++) {
      T *c = C + i * n;
      for (int64_t p = 0; p < k; p++) {
        const T a = A[i * k + p];
        const T *b = B + p * n;
        for (int64_t j = 0; j < n; j++) c[j] += a * b[j];
      }
    }
  } else if (!ta && tb) {
#pragma omp parallel for schedule(static) if (parallel && m * n * k > 65536)
    for (int64_t i = 0; i < m; i++) {
      const T *a = A + i * k;
      for (int64_t j = 0; j < n; j++) {
        const T *b = B + j * k;
        T s = T(0.0);
        for (int64_t p = 0; p < k; p++) s += a[p] * b[p];
        C[i * n + j] += s;
      }
    }
  } else if (ta && !tb) {
#pragma omp parallel for schedule(static) if (parallel && m * n * k > 65536)
    for (int64_t i = 0; i < m; i++) {
      T *c = C + i * n;
      for (int64_t p = 0; p < k; p++) {
        const T a = A[p * m + i];
        const T *b = B + p * n;
        for (int64_t j = 0; j < n; j++) c[j] += a * b[j];
      }
    }
  } else {
#pragma omp parallel for schedule(static) if (parallel && m * n * k > 65536)
    for (int64_t i = 0; i < m; i++)
      for (int64_t j = 0; j < n; j++) {
        T s = T(0.0);
        for (int64_t p = 0; p < k; p++) s += A[p * m + i] * B[j * k + p];
        C[i * n + j] += s;
      }
  }
}

// batched torch matmul shapes: a [.., m, k] (1-D: [k] as [1, k]), b [.., k, n] (1-D: [k] as [k, 1])
struct MM {
  std::vector<int64_t> ab, bb, ob; // batch shapes
  int64_t m, k, n;
  std::vector<int64_t> out;
  bool a1, b1;
  MM(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
    a1 = a.size() == 1, b1 = b.size() == 1;
    std::vector<int64_t> A = a1 ? std::vector<int64_t>{1, a[0]} : a, B = b1 ? std::vector<int64_t>{b[0], 1} : b;
    m = A[A.size() - 2], k = A.back(), n = B.back();
    if (B[B.size() - 2] != k) throw std::runtime_error("fp64 executor: MATMUL " + shape_str(a) + " @ " + shape_str(b));
    ab.assign(A.begin(), A.end() - 2), bb.assign(B.begin(), B.end() - 2);
    ob = bshape(ab, bb);
    out = ob;
    if (!a1) out.push_back(m);
    if (!b1) out.push_back(n);
  }
};

template <class T> void scatter_slices(int64_t outer, int64_t D, int64_t inner, int64_t off, int64_t len, int64_t Din,
                                       const T *src, T *dst, bool add) {
  // dst [outer, Din, inner] slice [off, off + len) <- src [outer, len, inner]
  for (int64_t o = 0; o < outer; o++)
    for (int64_t d = 0; d < len; d++) {
      const T *s = src + (o * len + d) * inner;
      T *t = dst + (o * Din + off + d) * inner;
      if (add)
        for (int64_t i = 0; i < inner; i++) t[i] += s[i];
      else
        std::copy(s, s + inner, t);
    }
  (void)D;
}

void split_dim(const std::vector<int64_t> &s, int64_t d, int64_t &outer, int64_t &D, int64_t &inner) {
  outer = 1, inner = 1;
  for (int64_t i = 0; i < d; i++) outer *= s[i];
  D = s[d];
  for (size_t i = d + 1; i < s.size(); i++) inner *= s[i];
}

int64_t norm_dim(int64_t d, size_t rank) { return d < 0 ? d + int64_t(rank) : d; }

} // namespace

template <class T> Executor<T>::Executor(const GIRGraph &g) : g_(g) {
  info_.reserve(g.nodes.size());
  for (size_t k = 0; k < g.nodes.size(); k++) {
    info_.push_back(parse_info(g.nodes[k]));
    index_[g.nodes[k].id] = int(k);
  }
  if (!g.outputs.empty()) {
    const std::string &r = g.outputs[0].node_ref;
    if (r.rfind("node:", 0) != 0) throw std::runtime_error("fp64 executor: output is not a node");
    out_index_ = index_.at(std::stoi(r.substr(5)));
  } else {
    out_index_ = int(g.nodes.size()) - 1;
  }
}

template <class T> Executor<T>::~Executor() = default;

template <class T> std::vector<int64_t> Executor<T>::resolve(const std::vector<int64_t> &shape) const {
  std::vector<int64_t> r;
  auto get = [&](const char *n) -> int64_t {
    auto it = dims_.find(n);
    if (it == dims_.end()) throw std::runtime_error(std::string("fp64 executor: dimension ") + n + " not set");
    return it->second;
  };
  for (int64_t d : shape) {
    if (d == DIM_N_ATOMS) r.push_back(get("n_atoms"));
    else if (d == DIM_MAX_NEIGHBORS) r.push_back(get("max_neighbors"));
    else if (d == DIM_SEQ_LEN) r.push_back(get("n_atoms") * (get("max_neighbors") + 1));
    else if (d == DIM_N_EDGES) r.push_back(get("n_atoms") * get("max_neighbors"));
    else if (d == DIM_MN_PLUS_ONE) r.push_back(get("max_neighbors") + 1);
    else if (const std::string *name = named_dim(d)) {
      int64_t v = 1;
      size_t a = 0;
      while (v > 0) {
        const size_t b = name->find('*', a);
        const std::string f = name->substr(a, b == std::string::npos ? std::string::npos : b - a);
        auto it = dims_.find(f);
        if (it != dims_.end()) v *= it->second;
        else if (!f.empty() && f.find_first_not_of("0123456789") == std::string::npos) v *= std::stoll(f);
        else v = -1;
        if (b == std::string::npos) break;
        a = b + 1;
      }
      r.push_back(v);
    } else r.push_back(d);
  }
  return r;
}

template <class T> struct Executor<T>::Ref {
  char kind; // n(ode) i(nput) w(eight) c(onst)
  int idx = -1;
  std::string name;
  double c = 0;
};

template <class T> typename Executor<T>::Ref Executor<T>::ref(const std::string &r) const {
  Ref R;
  const auto p = r.find(':');
  if (p == std::string::npos) throw std::runtime_error("fp64 executor: reference " + r);
  const std::string t = r.substr(0, p), v = r.substr(p + 1);
  if (t == "node") R.kind = 'n', R.idx = index_.at(std::stoi(v));
  else if (t == "input") R.kind = 'i', R.name = v;
  else if (t == "weight") R.kind = 'w', R.name = v;
  else if (t == "const") R.kind = 'c', R.c = std::stod(v);
  else throw std::runtime_error("fp64 executor: reference " + r);
  return R;
}

template <class T> const Tensor<T> &Executor<T>::value(const std::string &r) const {
  const Ref R = ref(r);
  if (R.kind == 'n') {
    if (!vals_[R.idx].data) throw std::runtime_error("fp64 executor: node value freed or missing: " + r);
    return vals_[R.idx];
  }
  if (R.kind == 'i') {
    auto it = inputs_.find(R.name);
    if (it == inputs_.end()) throw std::runtime_error("fp64 executor: input not set: " + R.name);
    return it->second;
  }
  if (R.kind == 'w') {
    auto it = weights_.find(R.name);
    if (it == weights_.end()) throw std::runtime_error("fp64 executor: weight not set: " + R.name);
    return it->second;
  }
  throw std::runtime_error("fp64 executor: const reference as a tensor: " + r);
}

template <class T> const Tensor<T> &Executor<T>::output() const { return vals_.at(out_index_); }
template <class T> const Tensor<T> *Executor<T>::node_value(int id) const {
  auto it = index_.find(id);
  if (it == index_.end() || !vals_[it->second].data) return nullptr;
  return &vals_[it->second];
}

namespace {
template <class T> T sigm(T x) { return T(1.0) / (T(1.0) + exp(-x)); }
} // namespace

template <class T> Tensor<T> Executor<T>::eval(size_t k) {
  const GIRNode &n = g_.nodes[k];
  const NodeInfo &I = info_[k];
  auto in = [&](size_t i) -> const Tensor<T> & {
    if (i >= n.inputs.size()) throw std::runtime_error("missing operand " + std::to_string(i));
    return value(n.inputs[i]);
  };
  auto unary = [&](auto f) {
    const Tensor<T> &a = in(0);
    Tensor<T> o = Tensor<T>::zeros(a.shape);
    const int64_t N = a.numel();
    const T *x = a.ptr();
    T *y = o.ptr();
#pragma omp parallel for schedule(static) if (par_ok(N))
    for (int64_t i = 0; i < N; i++) y[i] = f(x[i]);
    return o;
  };
  auto binary = [&](const Tensor<T> &a, const Tensor<T> &b, auto f) {
    Tensor<T> o = Tensor<T>::zeros(bshape(a.shape, b.shape));
    const T *x = a.ptr(), *y = b.ptr();
    T *z = o.ptr();
    const int64_t N = o.numel();
    if (a.shape == b.shape) {
#pragma omp parallel for schedule(static) if (par_ok(N))
      for (int64_t i = 0; i < N; i++) z[i] = f(x[i], y[i]);
    } else if (b.numel() == 1 && a.numel() == N) {
      const T s = y[0];
#pragma omp parallel for schedule(static) if (par_ok(N))
      for (int64_t i = 0; i < N; i++) z[i] = f(x[i], s);
    } else {
      bcast3(o.shape, a.shape, b.shape, b.shape, [&](int64_t i, int64_t ia, int64_t ib, int64_t) { z[i] = f(x[ia], y[ib]); });
    }
    return o;
  };
  auto scalar_binary = [&](auto f) { // tensor op scalar param
    const T s = T(I.scalar);
    return unary([&](T x) { return f(x, s); });
  };
  switch (I.op) {
  case Op::IDENTITY:
  case Op::CONT:
  case Op::SPLIT:
  case Op::CHUNK:
    return in(0);
  case Op::ADD:
    if (n.inputs.size() == 1) return I.has_scalar ? scalar_binary([](T a, T b) { return a + b; }) : in(0);
    return binary(in(0), in(1), [](T a, T b) { return a + b; });
  case Op::SUB:
    if (n.inputs.size() == 1) {
      if (!I.has_scalar) throw std::runtime_error("SUB with one input and no scalar");
      return scalar_binary([](T a, T b) { return a - b; });
    }
    return binary(in(0), in(1), [](T a, T b) { return a - b; });
  case Op::MUL:
    if (n.inputs.size() == 1) {
      if (!I.has_scalar) throw std::runtime_error("MUL with one input and no scalar");
      return scalar_binary([](T a, T b) { return a * b; });
    }
    return binary(in(0), in(1), [](T a, T b) { return a * b; });
  case Op::DIV:
    if (n.inputs.size() == 1) {
      if (!I.has_scalar) throw std::runtime_error("DIV with one input and no scalar");
      return scalar_binary([](T a, T b) { return a / b; });
    }
    return binary(in(0), in(1), [](T a, T b) { return a / b; });
  case Op::SCALE: {
    const T s = T(I.scale);
    return unary([&](T x) { return x * s; });
  }
  case Op::SQR:
    return unary([](T x) { return x * x; });
  case Op::SQRT:
    return unary([](T x) { return sqrt(x); });
  case Op::LOG:
    return unary([](T x) { return log(x); });
  case Op::EXP:
    return unary([](T x) { return exp(x); });
  case Op::NEG:
    return unary([](T x) { return -x; });
  case Op::ABS:
    return unary([](T x) { return val(x) < 0 ? -x : x; });
  case Op::RELU:
    return unary([](T x) { return val(x) > 0 ? x : T(0.0); });
  case Op::SIGMOID:
    return unary([](T x) { return sigm(x); });
  case Op::SILU:
    return unary([](T x) { return x * sigm(x); });
  case Op::TANH:
    return unary([](T x) { return tanh(x); });
  case Op::ERF:
    return unary([](T x) { return erf(x); });
  case Op::COS:
    return unary([](T x) { return cos(x); });
  case Op::SIN:
    return unary([](T x) { return sin(x); });
  case Op::POW: {
    const double e = I.exponent;
    if (e == 2.0) return unary([](T x) { return x * x; });
    if (e == 0.5) return unary([](T x) { return sqrt(x); });
    return unary([e](T x) { return pow(x, e); });
  }
  case Op::CLAMP: {
    const double lo = I.lo, hi = I.hi;
    return unary([lo, hi](T x) { return val(x) < lo ? T(lo) : val(x) > hi ? T(hi) : x; });
  }
  case Op::BITWISE_NOT:
    return unary([](T x) { return T(val(x) != 0 ? 0.0 : 1.0); });
  case Op::LE:
  case Op::GE: {
    const bool le = I.op == Op::LE;
    if (n.inputs.size() > 1)
      return binary(in(0), in(1), [le](T a, T b) { return T((le ? val(a) <= val(b) : val(a) >= val(b)) ? 1.0 : 0.0); });
    const double s = I.scalar;
    return unary([le, s](T a) { return T((le ? val(a) <= s : val(a) >= s) ? 1.0 : 0.0); });
  }
  case Op::WHERE:
  case Op::INDEX_PUT:
  case Op::MASKED_FILL: {
    // WHERE(c, x, y) = c ? x : y; INDEX_PUT(src, mask, v) = mask ? v : src; MASKED_FILL(src, mask) = mask ? fill : src
    const Tensor<T> *c, *x, *y;
    Tensor<T> fillt;
    if (I.op == Op::WHERE) c = &in(0), x = &in(1), y = &in(2);
    else if (I.op == Op::INDEX_PUT) c = &in(1), x = &in(2), y = &in(0);
    else {
      fillt = Tensor<T>::from({}, {T(I.fill)});
      c = &in(1), x = &fillt, y = &in(0);
    }
    Tensor<T> o = Tensor<T>::zeros(bshape(bshape(c->shape, x->shape), y->shape));
    const T *cp = c->ptr(), *xp = x->ptr(), *yp = y->ptr();
    T *z = o.ptr();
    bcast3(o.shape, c->shape, x->shape, y->shape,
           [&](int64_t i, int64_t ic, int64_t ix, int64_t iy) { z[i] = val(cp[ic]) != 0 ? xp[ix] : yp[iy]; });
    return o;
  }
  case Op::NEW_ZEROS:
  case Op::NEW_ONES: {
    std::vector<int64_t> s = I.shape.empty() ? n.output_shape : I.shape;
    if (s.empty() && I.shape.empty() && n.output_shape.empty()) throw std::runtime_error("no shape");
    Tensor<T> o = Tensor<T>::zeros(resolve(s));
    if (I.op == Op::NEW_ONES) std::fill(o.data->begin(), o.data->end(), T(1.0));
    return o;
  }
  case Op::RESHAPE:
  case Op::VIEW: {
    const Tensor<T> &a = in(0);
    std::vector<int64_t> s = I.shape;
    if (I.op == Op::RESHAPE) {
      if (s.empty() || (!n.output_shape.empty() && n.output_shape.size() > s.size())) s = n.output_shape;
      if (s.empty()) throw std::runtime_error("RESHAPE: no shape");
    } else if (I.shape_has_negative || s.empty()) {
      if (!n.output_shape.empty()) s = n.output_shape;
    }
    if (s.empty()) return a; // getitem pass-through
    s = resolve(s);
    if (I.op == Op::VIEW && I.index >= 0) {
      // ggml view of a chunk (graph_interpreter.cpp build_view): offset index * ne0, the input's own strides
      const std::vector<int64_t> &as = a.shape;
      std::vector<int64_t> ane(as.rbegin(), as.rend()), one(s.rbegin(), s.rend());
      while (ane.size() < 4) ane.push_back(1);
      while (one.size() < 4) one.push_back(1);
      int64_t nb[4];
      nb[0] = 1;
      for (int j = 1; j < 4; j++) nb[j] = nb[j - 1] * ane[j - 1];
      const int64_t off = I.index * one[0];
      Tensor<T> o = Tensor<T>::zeros(s);
      T *z = o.ptr();
      const T *x = a.ptr();
      int64_t q = 0;
      for (int64_t i3 = 0; i3 < one[3]; i3++)
        for (int64_t i2 = 0; i2 < one[2]; i2++)
          for (int64_t i1 = 0; i1 < one[1]; i1++)
            for (int64_t i0 = 0; i0 < one[0]; i0++) {
              const int64_t src = off + i0 + i1 * nb[1] + i2 * nb[2] + i3 * nb[3];
              if (src >= a.numel()) throw std::runtime_error("VIEW index: out of range");
              z[q++] = x[src];
            }
      return o;
    }
    int64_t known = 1, neg = -1;
    for (size_t i = 0; i < s.size(); i++) {
      if (s[i] < 0) {
        if (neg >= 0) throw std::runtime_error("reshape with two unknown dims");
        neg = int64_t(i);
      } else known *= s[i];
    }
    if (neg >= 0) s[neg] = a.numel() / known;
    if (numel(s) != a.numel())
      throw std::runtime_error("reshape " + shape_str(a.shape) + " -> " + shape_str(s));
    Tensor<T> o = a;
    o.shape = s;
    return o;
  }
  case Op::SELECT: {
    const Tensor<T> &a = in(0);
    const int64_t d = norm_dim(I.has_dim ? I.dim : 1, a.shape.size());
    int64_t outer, D, inner;
    split_dim(a.shape, d, outer, D, inner);
    int64_t idx = I.index < 0 ? I.index + D : I.index;
    if (I.index == -1 && !n.params.count("index")) idx = 0;
    if (idx < 0 || idx >= D) throw std::runtime_error("SELECT index out of range");
    std::vector<int64_t> s = a.shape;
    s.erase(s.begin() + d);
    Tensor<T> o = Tensor<T>::zeros(s);
    for (int64_t q = 0; q < outer; q++)
      std::copy(a.ptr() + (q * D + idx) * inner, a.ptr() + (q * D + idx + 1) * inner, o.ptr() + q * inner);
    return o;
  }
  case Op::SLICE: {
    const Tensor<T> &a = in(0);
    if (I.has_dim && (I.has_start || I.has_end)) {
      const int64_t d = norm_dim(I.dim, n.output_shape.empty() ? a.shape.size() : n.output_shape.size());
      int64_t outer, D, inner;
      split_dim(a.shape, d, outer, D, inner);
      int64_t st = I.start, en = I.end;
      if (st < 0) st += D;
      if (en < 0) en += D;
      st = std::clamp<int64_t>(st, 0, D), en = std::clamp<int64_t>(en, 0, D);
      if (en < st) en = st;
      if (st == 0 && en == D) return a;
      std::vector<int64_t> s = a.shape;
      s[d] = en - st;
      Tensor<T> o = Tensor<T>::zeros(s);
      for (int64_t q = 0; q < outer; q++)
        std::copy(a.ptr() + (q * D + st) * inner, a.ptr() + (q * D + en) * inner, o.ptr() + q * (en - st) * inner);
      return o;
    }
    if (n.output_shape.empty()) return a;
    const std::vector<int64_t> s = resolve(n.output_shape);
    if (s == a.shape) return a;
    if (s.size() != a.shape.size()) throw std::runtime_error("SLICE (legacy): rank");
    int dd = -1;
    for (size_t i = 0; i < s.size(); i++)
      if (s[i] != a.shape[i]) {
        if (dd >= 0) throw std::runtime_error("SLICE (legacy): several dims differ");
        dd = int(i);
      }
    int64_t outer, D, inner;
    split_dim(a.shape, dd, outer, D, inner);
    Tensor<T> o = Tensor<T>::zeros(s);
    for (int64_t q = 0; q < outer; q++)
      std::copy(a.ptr() + q * D * inner, a.ptr() + (q * D + s[dd]) * inner, o.ptr() + q * s[dd] * inner);
    return o;
  }
  case Op::PERMUTE:
  case Op::TRANSPOSE: {
    const Tensor<T> &a = in(0);
    const size_t r = a.shape.size();
    std::vector<int64_t> perm(r);
    std::iota(perm.begin(), perm.end(), 0);
    if (I.op == Op::PERMUTE) {
      if (I.axes.size() > r) throw std::runtime_error("PERMUTE: more axes than dims");
      for (size_t i = 0; i < I.axes.size(); i++) perm[i] = norm_dim(I.axes[i], r);
    } else {
      if (I.dims.size() == 2) {
        const size_t rr = n.output_shape.empty() ? r : n.output_shape.size();
        std::swap(perm[norm_dim(I.dims[0], rr)], perm[norm_dim(I.dims[1], rr)]);
      } else {
        if (r < 2) throw std::runtime_error("TRANSPOSE of rank < 2");
        std::swap(perm[r - 1], perm[r - 2]); // ggml_transpose: the two innermost
      }
    }
    std::vector<int64_t> s(r), ist(r, 1);
    for (int i = int(r) - 2; i >= 0; i--) ist[i] = ist[i + 1] * a.shape[i + 1];
    for (size_t i = 0; i < r; i++) s[i] = a.shape[perm[i]];
    Tensor<T> o = Tensor<T>::zeros(s);
    const auto on = pad4(s);
    std::array<int64_t, 4> st{0, 0, 0, 0};
    for (size_t i = 0; i < r; i++) st[4 - r + i] = ist[perm[i]];
    T *z = o.ptr();
    const T *x = a.ptr();
#pragma omp parallel for collapse(2) schedule(static) if (par_ok(o.numel()))
    for (int64_t i0 = 0; i0 < on[0]; i0++)
      for (int64_t i1 = 0; i1 < on[1]; i1++)
        for (int64_t i2 = 0; i2 < on[2]; i2++)
          for (int64_t i3 = 0; i3 < on[3]; i3++)
            z[((i0 * on[1] + i1) * on[2] + i2) * on[3] + i3] = x[i0 * st[0] + i1 * st[1] + i2 * st[2] + i3 * st[3]];
    return o;
  }
  case Op::CONCAT: {
    std::vector<const Tensor<T> *> ts;
    for (size_t i = 0; i < n.inputs.size(); i++) ts.push_back(&in(i));
    const size_t r = ts[0]->shape.size();
    const int64_t d = norm_dim(I.dim, n.output_shape.empty() ? r : n.output_shape.size());
    std::vector<int64_t> s = ts[0]->shape;
    int64_t tot = 0;
    for (auto *t : ts) {
      if (t->shape.size() != r) throw std::runtime_error("CONCAT: ranks differ");
      for (size_t i = 0; i < r; i++)
        if (int64_t(i) != d && t->shape[i] != s[i]) throw std::runtime_error("CONCAT: shapes differ");
      tot += t->shape[d];
    }
    s[d] = tot;
    Tensor<T> o = Tensor<T>::zeros(s);
    int64_t outer, D, inner, off = 0;
    split_dim(s, d, outer, D, inner);
    for (auto *t : ts) {
      scatter_slices(outer, D, inner, off, t->shape[d], D, t->ptr(), o.ptr(), false);
      off += t->shape[d];
    }
    return o;
  }
  case Op::SUM_ROWS: {
    const Tensor<T> &a = in(0);
    int64_t d;
    if (I.has_dims) {
      if (I.dims.size() != 1) throw std::runtime_error("SUM_ROWS over several dims");
      d = norm_dim(I.dims[0], a.shape.size());
    } else if (n.output_shape.empty() && a.shape.size() > 1) {
      throw std::runtime_error("SUM_ROWS without dims to a scalar from rank > 1");
    } else d = int64_t(a.shape.size()) - 1;
    if (a.shape.empty()) return a;
    int64_t outer, D, inner;
    split_dim(a.shape, d, outer, D, inner);
    std::vector<int64_t> s = a.shape;
    if (I.keepdim) s[d] = 1;
    else s.erase(s.begin() + d);
    Tensor<T> o = Tensor<T>::zeros(s);
    const T *x = a.ptr();
    T *z = o.ptr();
#pragma omp parallel for schedule(static) if (par_ok(a.numel()))
    for (int64_t q = 0; q < outer; q++)
      for (int64_t i = 0; i < inner; i++) {
        T acc = T(0.0);
        for (int64_t j = 0; j < D; j++) acc += x[(q * D + j) * inner + i];
        z[q * inner + i] = acc;
      }
    return o;
  }
  case Op::REPEAT: {
    const Tensor<T> &a = in(0);
    bool shape_sources = n.inputs.size() >= 2 && !n.output_shape.empty();
    for (size_t i = 1; shape_sources && i < n.inputs.size(); i++) shape_sources = n.inputs[i].rfind("input:", 0) == 0;
    std::vector<int64_t> s = (n.inputs.size() < 2 || shape_sources) ? resolve(n.output_shape) : in(1).shape;
    Tensor<T> o = Tensor<T>::zeros(s);
    (void)bshape(a.shape, s);
    const T *x = a.ptr();
    T *z = o.ptr();
    bcast3(s, a.shape, a.shape, a.shape, [&](int64_t i, int64_t ia, int64_t, int64_t) { z[i] = x[ia]; });
    return o;
  }
  case Op::SOFT_MAX: {
    if (n.inputs.size() > 1 && n.inputs[1] != "null") throw std::runtime_error("SOFT_MAX with a mask");
    const Tensor<T> &a = in(0);
    const int64_t d = norm_dim(I.has_dim ? I.dim : -1, a.shape.size());
    int64_t outer, D, inner;
    split_dim(a.shape, d, outer, D, inner);
    Tensor<T> o = Tensor<T>::zeros(a.shape);
    const T sc = T(I.scale);
    const T *x = a.ptr();
    T *z = o.ptr();
#pragma omp parallel for collapse(2) schedule(static) if (par_ok(a.numel()))
    for (int64_t q = 0; q < outer; q++)
      for (int64_t i = 0; i < inner; i++) {
        double mx = -INFINITY;
        for (int64_t j = 0; j < D; j++) mx = std::max(mx, val(x[(q * D + j) * inner + i] * sc));
        T sum = T(0.0);
        for (int64_t j = 0; j < D; j++) {
          const int64_t p = (q * D + j) * inner + i;
          z[p] = exp(x[p] * sc - T(mx));
          sum += z[p];
        }
        for (int64_t j = 0; j < D; j++) z[(q * D + j) * inner + i] = z[(q * D + j) * inner + i] / sum;
      }
    return o;
  }
  case Op::LAYER_NORM:
  case Op::RMS_NORM: {
    const Tensor<T> &a = in(0);
    const bool ln = I.op == Op::LAYER_NORM;
    const Tensor<T> *w = nullptr, *b = nullptr;
    if (ln) {
      if (n.inputs.size() < 3) throw std::runtime_error("LAYER_NORM needs weight and bias");
      w = n.inputs.size() == 3 ? &in(1) : &in(2);
      b = n.inputs.size() == 3 ? &in(2) : &in(3);
    } else {
      if (n.inputs.size() < 2) throw std::runtime_error("RMS_NORM needs a weight");
      w = &in(1);
    }
    double eps = I.eps;
    if (!ln && eps == kF32Eps) eps = rms_eps_f32_replacement;
    const int64_t D = a.shape.back(), R = a.numel() / D;
    if (w->numel() != D || (b && b->numel() != D)) throw std::runtime_error("norm: weight size");
    Tensor<T> o = Tensor<T>::zeros(a.shape);
    const T *x = a.ptr(), *wp = w->ptr(), *bp = b ? b->ptr() : nullptr;
    T *z = o.ptr();
#pragma omp parallel for schedule(static) if (par_ok(a.numel()))
    for (int64_t r = 0; r < R; r++) {
      const T *xr = x + r * D;
      T *zr = z + r * D;
      T mean = T(0.0);
      if (ln) {
        for (int64_t j = 0; j < D; j++) mean += xr[j];
        mean = mean / T(double(D));
      }
      T var = T(0.0);
      for (int64_t j = 0; j < D; j++) {
        const T c = xr[j] - mean;
        var += c * c;
      }
      const T inv = T(1.0) / sqrt(var / T(double(D)) + T(eps));
      for (int64_t j = 0; j < D; j++) zr[j] = (xr[j] - mean) * inv * wp[j] + (bp ? bp[j] : T(0.0));
    }
    return o;
  }
  case Op::GET_ROWS:
  case Op::INDEX: {
    const Tensor<T> &a = in(0), &idx = in(1);
    if (a.shape.empty()) throw std::runtime_error("gather from a scalar");
    const int64_t N = a.shape[0], row = a.numel() / N;
    std::vector<int64_t> s = idx.shape;
    s.insert(s.end(), a.shape.begin() + 1, a.shape.end());
    Tensor<T> o = Tensor<T>::zeros(s);
    const int64_t M = idx.numel();
    for (int64_t i = 0; i < M; i++) {
      const int64_t j = int64_t(val(idx.ptr()[i]));
      if (j < 0 || j >= N) throw std::runtime_error("gather index out of range");
      std::copy(a.ptr() + j * row, a.ptr() + (j + 1) * row, o.ptr() + i * row);
    }
    return o;
  }
  case Op::LINEAR: {
    const Tensor<T> &x = in(0), &w = in(1);
    if (w.shape.size() != 2) throw std::runtime_error("LINEAR weight rank");
    const int64_t O = w.shape[0], K = w.shape[1];
    if (x.shape.back() != K) throw std::runtime_error("LINEAR: " + shape_str(x.shape) + " x " + shape_str(w.shape));
    const int64_t R = x.numel() / K;
    std::vector<int64_t> s = x.shape;
    s.back() = O;
    Tensor<T> o = Tensor<T>::zeros(s);
    if (n.inputs.size() > 2) {
      const Tensor<T> &b = in(2);
      if (b.numel() != O) throw std::runtime_error("LINEAR bias size");
      for (int64_t r = 0; r < R; r++) std::copy(b.ptr(), b.ptr() + O, o.ptr() + r * O);
    }
    gemm<T>(R, O, K, x.ptr(), false, w.ptr(), true, o.ptr(), true);
    return o;
  }
  case Op::MATMUL: {
    const Tensor<T> &a = in(0), &b = in(1);
    MM mm(a.shape, b.shape);
    Tensor<T> o = Tensor<T>::zeros(mm.out);
    const int64_t nb = numel(mm.ob);
    BIdx A(mm.ab), B(mm.bb);
    const auto obn = pad4(mm.ob);
    const int64_t sa = mm.m * mm.k, sb = mm.k * mm.n, so = mm.m * mm.n;
    const bool batch_par = nb >= 8;
#pragma omp parallel for schedule(static) if (batch_par)
    for (int64_t q = 0; q < nb; q++) {
      int64_t r = q, ia = 0, ib = 0;
      for (int d = 3; d >= 0; d--) {
        const int64_t i = r % obn[d];
        r /= obn[d];
        ia += A.at(d, i), ib += B.at(d, i);
      }
      gemm<T>(mm.m, mm.n, mm.k, a.ptr() + ia * sa, false, b.ptr() + ib * sb, false, o.ptr() + q * so, !batch_par);
    }
    return o;
  }
  }
  throw std::runtime_error("fp64 executor: unhandled op");
}

template <class T> void Executor<T>::forward(bool keep_all) {
  vals_.assign(g_.nodes.size(), Tensor<T>{});
  std::vector<int> last_use(g_.nodes.size(), -1);
  if (!keep_all) {
    for (size_t k = 0; k < g_.nodes.size(); k++)
      for (const auto &r : g_.nodes[k].inputs)
        if (r.rfind("node:", 0) == 0) last_use[index_.at(std::stoi(r.substr(5)))] = int(k);
    last_use[out_index_] = int(g_.nodes.size());
  }
  static const bool trace = std::getenv("MLIPCPP_FP64_TRACE") != nullptr;
  for (size_t k = 0; k < g_.nodes.size(); k++) {
    const GIRNode &n = g_.nodes[k];
    try {
      Tensor<T> v = eval(k);
      // the result's shape must be the traced one (a size-1 rank difference of a pass-through is a reshape)
      if (!n.output_shape.empty()) {
        const std::vector<int64_t> s = resolve(n.output_shape);
        bool resolved = true;
        for (auto d : s) resolved &= d >= 0;
        if (resolved && s != v.shape) {
          if (numel(s) == v.numel() && (info_[k].op == Op::SUM_ROWS || info_[k].op == Op::IDENTITY))
            v.shape = s;
          else
            throw std::runtime_error("result shape " + shape_str(v.shape) + " != traced " + shape_str(s));
        }
      }
      if (trace) std::fprintf(stderr, "fp64 node %d %s %s\n", n.id, n.op.c_str(), shape_str(v.shape).c_str());
      vals_[k] = std::move(v);
    } catch (const std::exception &e) {
      throw std::runtime_error("fp64 executor: node " + std::to_string(n.id) + " (" + n.op + " \"" + n.name +
                               "\"): " + e.what());
    }
    if (!keep_all)
      for (const auto &r : n.inputs)
        if (r.rfind("node:", 0) == 0) {
          const int j = index_.at(std::stoi(r.substr(5)));
          if (last_use[j] == int(k)) vals_[j] = Tensor<T>{};
        }
  }
}

// ----------------------------------------------------------------------------------------------------------------
// reverse mode
// ----------------------------------------------------------------------------------------------------------------

template <class T> void Executor<T>::backward(const std::vector<T> &cot, bool want_w) {
  const size_t N = g_.nodes.size();
  if (int64_t(cot.size()) != vals_.at(out_index_).numel())
    throw std::runtime_error("fp64 executor backward: cotangent size");
  in_grad_.clear(), w_grad_.clear();
  node_grads_.assign(keep_node_grads ? N : 0, {});
  // which nodes depend on a differentiable leaf (float inputs; weights when asked)
  std::vector<char> dep(N, 0);
  std::map<std::string, bool> float_in;
  for (const auto &i : g_.inputs) float_in[i.name] = i.dtype == GIRDtype::F32;
  auto leaf_dep = [&](const std::string &r) {
    if (r.rfind("node:", 0) == 0) return bool(dep[index_.at(std::stoi(r.substr(5)))]);
    if (r.rfind("input:", 0) == 0) {
      auto it = float_in.find(r.substr(6));
      return it != float_in.end() && it->second;
    }
    if (r.rfind("weight:", 0) == 0) return want_w;
    return false;
  };
  for (size_t k = 0; k < N; k++) {
    const Op op = info_[k].op;
    if (op == Op::LE || op == Op::GE || op == Op::BITWISE_NOT || op == Op::NEW_ZEROS || op == Op::NEW_ONES) continue;
    const auto &ins = g_.nodes[k].inputs;
    // gather indices / select conditions are not differentiated
    size_t lim = ins.size();
    if (op == Op::GET_ROWS || op == Op::INDEX) lim = 1;
    for (size_t i = 0; i < lim && !dep[k]; i++) {
      if (op == Op::WHERE && i == 0) continue;
      if ((op == Op::INDEX_PUT || op == Op::MASKED_FILL) && i == 1) continue;
      if ((op == Op::VIEW || op == Op::RESHAPE || op == Op::SPLIT || op == Op::REPEAT) && i > 0) continue;
      dep[k] = leaf_dep(ins[i]);
    }
  }
  std::vector<std::vector<T>> adj(N);
  adj[out_index_] = cot;
  auto acc_ref = [&](const std::string &r, int64_t n) -> T * {
    if (!leaf_dep(r)) return nullptr;
    if (r.rfind("node:", 0) == 0) {
      auto &v = adj[index_.at(std::stoi(r.substr(5)))];
      if (v.empty()) v.assign(size_t(n), T(0.0));
      return v.data();
    }
    auto &m = r.rfind("input:", 0) == 0 ? in_grad_ : w_grad_;
    auto &v = m[r.substr(r.find(':') + 1)];
    if (v.empty()) v.assign(size_t(n), T(0.0));
    return v.data();
  };
  for (size_t kk = N; kk-- > 0;) {
    if (adj[kk].empty()) continue;
    const GIRNode &n = g_.nodes[kk];
    const NodeInfo &I = info_[kk];
    std::vector<T> gv = std::move(adj[kk]);
    adj[kk].clear();
    adj[kk].shrink_to_fit();
    if (keep_node_grads) node_grads_[kk] = gv;
    if (!dep[kk]) continue;
    const T *g = gv.data();
    const Tensor<T> &y = vals_[kk];
    const int64_t NY = y.numel();
    auto in = [&](size_t i) -> const Tensor<T> & { return value(n.inputs[i]); };
    auto grad_of = [&](size_t i) -> T * { return i < n.inputs.size() ? acc_ref(n.inputs[i], in(i).numel()) : nullptr; };
    try {
      auto unary_back = [&](auto df) { // dx += g * df(x, y)
        T *ga = grad_of(0);
        if (!ga) return;
        const T *x = in(0).ptr(), *yy = y.ptr();
#pragma omp parallel for schedule(static) if (par_ok(NY))
        for (int64_t i = 0; i < NY; i++) ga[i] += g[i] * df(x[i], yy[i]);
      };
      auto binary_back = [&](auto da, auto db) { // da(x, y, out), db(x, y, out): local partials
        const Tensor<T> &a = in(0), &b = in(1);
        T *ga = grad_of(0), *gb = grad_of(1);
        const T *x = a.ptr(), *w = b.ptr(), *o = y.ptr();
        if (a.shape == b.shape) {
          if (ga) {
#pragma omp parallel for schedule(static) if (par_ok(NY))
            for (int64_t i = 0; i < NY; i++) ga[i] += g[i] * da(x[i], w[i], o[i]);
          }
          if (gb) {
#pragma omp parallel for schedule(static) if (par_ok(NY))
            for (int64_t i = 0; i < NY; i++) gb[i] += g[i] * db(x[i], w[i], o[i]);
          }
          return;
        }
        if (ga) {
          if (a.numel() == NY) {
            bcast3(y.shape, a.shape, b.shape, b.shape,
                   [&](int64_t i, int64_t ia, int64_t ib, int64_t) { ga[ia] += g[i] * da(x[ia], w[ib], o[i]); });
          } else {
            bcast3(y.shape, a.shape, b.shape, b.shape,
                   [&](int64_t i, int64_t ia, int64_t ib, int64_t) { ga[ia] += g[i] * da(x[ia], w[ib], o[i]); }, false);
          }
        }
        if (gb) {
          if (b.numel() == NY) {
            bcast3(y.shape, a.shape, b.shape, b.shape,
                   [&](int64_t i, int64_t ia, int64_t ib, int64_t) { gb[ib] += g[i] * db(x[ia], w[ib], o[i]); });
          } else {
            bcast3(y.shape, a.shape, b.shape, b.shape,
                   [&](int64_t i, int64_t ia, int64_t ib, int64_t) { gb[ib] += g[i] * db(x[ia], w[ib], o[i]); }, false);
          }
        }
      };
      auto pass = [&]() { // same elements, any shape
        if (T *ga = grad_of(0)) {
#pragma omp parallel for schedule(static) if (par_ok(NY))
          for (int64_t i = 0; i < NY; i++) ga[i] += g[i];
        }
      };
      const T one = T(1.0);
      switch (I.op) {
      case Op::IDENTITY:
      case Op::CONT:
      case Op::SPLIT:
      case Op::CHUNK:
        pass();
        break;
      case Op::ADD:
        if (n.inputs.size() == 1) pass();
        else binary_back([&](T, T, T) { return one; }, [&](T, T, T) { return one; });
        break;
      case Op::SUB:
        if (n.inputs.size() == 1) pass();
        else binary_back([&](T, T, T) { return one; }, [&](T, T, T) { return -one; });
        break;
      case Op::MUL:
        if (n.inputs.size() == 1) {
          const T s = T(I.scalar);
          unary_back([&](T, T) { return s; });
        } else binary_back([](T, T b, T) { return b; }, [](T a, T, T) { return a; });
        break;
      case Op::DIV:
        if (n.inputs.size() == 1) {
          const T s = T(I.scalar);
          unary_back([&](T, T) { return one / s; });
        } else binary_back([&](T, T b, T) { return one / b; }, [](T, T b, T o) { return -o / b; });
        break;
      case Op::SCALE: {
        const T s = T(I.scale);
        unary_back([&](T, T) { return s; });
        break;
      }
      case Op::SQR:
        unary_back([](T x, T) { return x + x; });
        break;
      case Op::SQRT:
        unary_back([](T, T o) { return T(0.5) / o; });
        break;
      case Op::LOG:
        unary_back([&](T x, T) { return one / x; });
        break;
      case Op::EXP:
        unary_back([](T, T o) { return o; });
        break;
      case Op::NEG:
        unary_back([&](T, T) { return -one; });
        break;
      case Op::ABS:
        unary_back([](T x, T) { return T(val(x) > 0 ? 1.0 : val(x) < 0 ? -1.0 : 0.0); });
        break;
      case Op::RELU:
        unary_back([](T x, T) { return T(val(x) > 0 ? 1.0 : 0.0); });
        break;
      case Op::SIGMOID:
        unary_back([&](T, T o) { return o * (one - o); });
        break;
      case Op::SILU:
        unary_back([&](T x, T) {
          const T s = sigm(x);
          return s * (one + x * (one - s));
        });
        break;
      case Op::TANH:
        unary_back([&](T, T o) { return one - o * o; });
        break;
      case Op::ERF:
        unary_back([](T x, T) { return T(1.1283791670955126) * exp(-x * x); });
        break;
      case Op::COS:
        unary_back([](T x, T) { return -sin(x); });
        break;
      case Op::SIN:
        unary_back([](T x, T) { return cos(x); });
        break;
      case Op::POW: {
        const double e = I.exponent;
        if (e == 2.0) unary_back([](T x, T) { return x + x; });
        else if (e == 0.5) unary_back([](T, T o) { return T(0.5) / o; });
        else unary_back([e](T x, T) { return T(e) * pow(x, e - 1); });
        break;
      }
      case Op::CLAMP: {
        const double lo = I.lo, hi = I.hi;
        unary_back([lo, hi](T x, T) { return T(val(x) >= lo && val(x) <= hi ? 1.0 : 0.0); });
        break;
      }
      case Op::LE:
      case Op::GE:
      case Op::BITWISE_NOT:
      case Op::NEW_ZEROS:
      case Op::NEW_ONES:
        break;
      case Op::WHERE:
      case Op::INDEX_PUT:
      case Op::MASKED_FILL: {
        size_t ic, ix, iy;
        if (I.op == Op::WHERE) ic = 0, ix = 1, iy = 2;
        else if (I.op == Op::INDEX_PUT) ic = 1, ix = 2, iy = 0;
        else ic = 1, ix = 99, iy = 0;
        const Tensor<T> &c = in(ic);
        static const Tensor<T> scalar = Tensor<T>::from({}, {T(0.0)});
        const Tensor<T> &x = ix < 99 ? in(ix) : scalar, &yy = in(iy);
        T *gx = ix < 99 ? grad_of(ix) : nullptr, *gy = grad_of(iy);
        const T *cp = c.ptr();
        const bool px = gx && x.numel() == NY, py = gy && yy.numel() == NY;
        bcast3(y.shape, c.shape, x.shape, yy.shape, [&](int64_t i, int64_t icc, int64_t ixx, int64_t iyy) {
          if (val(cp[icc]) != 0) {
            if (gx) gx[ixx] += g[i];
          } else if (gy) gy[iyy] += g[i];
        }, px && (py || !gy) && (!gx || px));
        break;
      }
      case Op::RESHAPE:
      case Op::VIEW: {
        if (I.op == Op::VIEW && I.index >= 0) {
          T *ga = grad_of(0);
          if (!ga) break;
          const Tensor<T> &a = in(0);
          std::vector<int64_t> ane(a.shape.rbegin(), a.shape.rend()), one4(y.shape.rbegin(), y.shape.rend());
          while (ane.size() < 4) ane.push_back(1);
          while (one4.size() < 4) one4.push_back(1);
          int64_t nb[4];
          nb[0] = 1;
          for (int j = 1; j < 4; j++) nb[j] = nb[j - 1] * ane[j - 1];
          const int64_t off = I.index * one4[0];
          int64_t q = 0;
          for (int64_t i3 = 0; i3 < one4[3]; i3++)
            for (int64_t i2 = 0; i2 < one4[2]; i2++)
              for (int64_t i1 = 0; i1 < one4[1]; i1++)
                for (int64_t i0 = 0; i0 < one4[0]; i0++) ga[off + i0 + i1 * nb[1] + i2 * nb[2] + i3 * nb[3]] += g[q++];
        } else pass();
        break;
      }
      case Op::SELECT: {
        T *ga = grad_of(0);
        if (!ga) break;
        const Tensor<T> &a = in(0);
        const int64_t d = norm_dim(I.has_dim ? I.dim : 1, a.shape.size());
        int64_t outer, D, inner;
        split_dim(a.shape, d, outer, D, inner);
        int64_t idx = I.index < 0 ? I.index + D : I.index;
        if (I.index == -1 && !n.params.count("index")) idx = 0;
        for (int64_t q = 0; q < outer; q++)
          for (int64_t i = 0; i < inner; i++) ga[(q * D + idx) * inner + i] += g[q * inner + i];
        break;
      }
      case Op::SLICE: {
        T *ga = grad_of(0);
        if (!ga) break;
        const Tensor<T> &a = in(0);
        if (a.shape == y.shape) {
          pass();
          break;
        }
        int dd = -1;
        for (size_t i = 0; i < a.shape.size(); i++)
          if (a.shape[i] != y.shape[i]) dd = int(i);
        int64_t st = 0;
        if (I.has_dim && (I.has_start || I.has_end)) {
          st = I.start;
          if (st < 0) st += a.shape[dd];
          st = std::clamp<int64_t>(st, 0, a.shape[dd]);
        }
        int64_t outer, D, inner;
        split_dim(a.shape, dd, outer, D, inner);
        scatter_slices(outer, D, inner, st, y.shape[dd], D, g, ga, true);
        break;
      }
      case Op::PERMUTE:
      case Op::TRANSPOSE: {
        T *ga = grad_of(0);
        if (!ga) break;
        const Tensor<T> &a = in(0);
        const size_t r = a.shape.size();
        std::vector<int64_t> perm(r);
        std::iota(perm.begin(), perm.end(), 0);
        if (I.op == Op::PERMUTE) {
          for (size_t i = 0; i < I.axes.size(); i++) perm[i] = norm_dim(I.axes[i], r);
        } else if (I.dims.size() == 2) {
          const size_t rr = n.output_shape.empty() ? r : n.output_shape.size();
          std::swap(perm[norm_dim(I.dims[0], rr)], perm[norm_dim(I.dims[1], rr)]);
        } else std::swap(perm[r - 1], perm[r - 2]);
        std::vector<int64_t> ist(r, 1);
        for (int i = int(r) - 2; i >= 0; i--) ist[i] = ist[i + 1] * a.shape[i + 1];
        const auto on = pad4(y.shape);
        std::array<int64_t, 4> st{0, 0, 0, 0};
        for (size_t i = 0; i < r; i++) st[4 - r + i] = ist[perm[i]];
#pragma omp parallel for collapse(2) schedule(static) if (par_ok(NY))
        for (int64_t i0 = 0; i0 < on[0]; i0++)
          for (int64_t i1 = 0; i1 < on[1]; i1++)
            for (int64_t i2 = 0; i2 < on[2]; i2++)
              for (int64_t i3 = 0; i3 < on[3]; i3++)
                ga[i0 * st[0] + i1 * st[1] + i2 * st[2] + i3 * st[3]] += g[((i0 * on[1] + i1) * on[2] + i2) * on[3] + i3];
        break;
      }
      case Op::CONCAT: {
        const size_t r = y.shape.size();
        const int64_t d = norm_dim(I.dim, n.output_shape.empty() ? r : n.output_shape.size());
        int64_t outer, D, inner, off = 0;
        split_dim(y.shape, d, outer, D, inner);
        for (size_t t = 0; t < n.inputs.size(); t++) {
          const int64_t len = in(t).shape[d];
          if (T *ga = grad_of(t))
            for (int64_t q = 0; q < outer; q++)
              for (int64_t j = 0; j < len; j++)
                for (int64_t i = 0; i < inner; i++) ga[(q * len + j) * inner + i] += g[(q * D + off + j) * inner + i];
          off += len;
        }
        break;
      }
      case Op::SUM_ROWS: {
        T *ga = grad_of(0);
        if (!ga) break;
        const Tensor<T> &a = in(0);
        if (a.shape.empty()) {
          ga[0] += g[0];
          break;
        }
        const int64_t d = I.has_dims ? norm_dim(I.dims[0], a.shape.size()) : int64_t(a.shape.size()) - 1;
        int64_t outer, D, inner;
        split_dim(a.shape, d, outer, D, inner);
#pragma omp parallel for schedule(static) if (par_ok(a.numel()))
        for (int64_t q = 0; q < outer; q++)
          for (int64_t j = 0; j < D; j++)
            for (int64_t i = 0; i < inner; i++) ga[(q * D + j) * inner + i] += g[q * inner + i];
        break;
      }
      case Op::REPEAT: {
        T *ga = grad_of(0);
        if (!ga) break;
        reduce_to(y.shape, in(0).shape, g, ga);
        break;
      }
      case Op::SOFT_MAX: {
        T *ga = grad_of(0);
        if (!ga) break;
        const Tensor<T> &a = in(0);
        const int64_t d = norm_dim(I.has_dim ? I.dim : -1, a.shape.size());
        int64_t outer, D, inner;
        split_dim(a.shape, d, outer, D, inner);
        const T sc = T(I.scale);
        const T *p = y.ptr();
#pragma omp parallel for collapse(2) schedule(static) if (par_ok(NY))
        for (int64_t q = 0; q < outer; q++)
          for (int64_t i = 0; i < inner; i++) {
            T s = T(0.0);
            for (int64_t j = 0; j < D; j++) {
              const int64_t k = (q * D + j) * inner + i;
              s += g[k] * p[k];
            }
            for (int64_t j = 0; j < D; j++) {
              const int64_t k = (q * D + j) * inner + i;
              ga[k] += sc * p[k] * (g[k] - s);
            }
          }
        break;
      }
      case Op::LAYER_NORM:
      case Op::RMS_NORM: {
        const bool ln = I.op == Op::LAYER_NORM;
        const size_t iw = ln ? (n.inputs.size() == 3 ? 1 : 2) : 1, ib = ln ? iw + 1 : 99;
        const Tensor<T> &a = in(0), &w = in(iw);
        T *ga = grad_of(0), *gw = grad_of(iw), *gb = ib < 99 ? grad_of(ib) : nullptr;
        double eps = I.eps;
        if (!ln && eps == kF32Eps) eps = rms_eps_f32_replacement;
        const int64_t D = a.shape.back(), R = a.numel() / D;
        const T *x = a.ptr(), *wp = w.ptr();
        const T dd = T(double(D));
        std::vector<T> gwacc(gw ? size_t(D) : 0, T(0.0)), gbacc(gb ? size_t(D) : 0, T(0.0));
#pragma omp parallel if (par_ok(a.numel()))
        {
          std::vector<T> lw(gw ? size_t(D) : 0, T(0.0)), lb(gb ? size_t(D) : 0, T(0.0)), xh(static_cast<size_t>(D)), gh(static_cast<size_t>(D));
#pragma omp for schedule(static)
          for (int64_t r = 0; r < R; r++) {
            const T *xr = x + r * D, *gr = g + r * D;
            T mean = T(0.0);
            if (ln) {
              for (int64_t j = 0; j < D; j++) mean += xr[j];
              mean = mean / dd;
            }
            T var = T(0.0);
            for (int64_t j = 0; j < D; j++) {
              const T c = xr[j] - mean;
              var += c * c;
            }
            const T inv = T(1.0) / sqrt(var / dd + T(eps));
            T m1 = T(0.0), m2 = T(0.0);
            for (int64_t j = 0; j < D; j++) {
              xh[j] = (xr[j] - mean) * inv;
              gh[j] = gr[j] * wp[j];
              m1 += gh[j];
              m2 += gh[j] * xh[j];
              if (gw) lw[j] += gr[j] * xh[j];
              if (gb) lb[j] += gr[j];
            }
            m1 = m1 / dd, m2 = m2 / dd;
            if (ga)
              for (int64_t j = 0; j < D; j++) ga[r * D + j] += inv * (gh[j] - (ln ? m1 : T(0.0)) - xh[j] * m2);
          }
#pragma omp critical
          {
            for (int64_t j = 0; gw && j < D; j++) gwacc[j] += lw[j];
            for (int64_t j = 0; gb && j < D; j++) gbacc[j] += lb[j];
          }
        }
        for (int64_t j = 0; gw && j < D; j++) gw[j] += gwacc[j];
        for (int64_t j = 0; gb && j < D; j++) gb[j] += gbacc[j];
        break;
      }
      case Op::GET_ROWS:
      case Op::INDEX: {
        T *ga = grad_of(0);
        if (!ga) break;
        const Tensor<T> &a = in(0), &idx = in(1);
        const int64_t row = a.numel() / a.shape[0], M = idx.numel();
        for (int64_t i = 0; i < M; i++) {
          const int64_t j = int64_t(val(idx.ptr()[i]));
          for (int64_t c = 0; c < row; c++) ga[j * row + c] += g[i * row + c];
        }
        break;
      }
      case Op::LINEAR: {
        const Tensor<T> &x = in(0), &w = in(1);
        const int64_t O = w.shape[0], K = w.shape[1], R = x.numel() / K;
        if (T *gx = grad_of(0)) gemm<T>(R, K, O, g, false, w.ptr(), false, gx, true);
        if (T *gw = grad_of(1)) gemm<T>(O, K, R, g, true, x.ptr(), false, gw, true);
        if (n.inputs.size() > 2)
          if (T *gb = grad_of(2))
            for (int64_t r = 0; r < R; r++)
              for (int64_t o = 0; o < O; o++) gb[o] += g[r * O + o];
        break;
      }
      case Op::MATMUL: {
        const Tensor<T> &a = in(0), &b = in(1);
        MM mm(a.shape, b.shape);
        T *ga = grad_of(0), *gb = grad_of(1);
        const int64_t nb = numel(mm.ob);
        BIdx A(mm.ab), B(mm.bb);
        const auto obn = pad4(mm.ob);
        const int64_t sa = mm.m * mm.k, sb = mm.k * mm.n, so = mm.m * mm.n;
        const bool a_own = numel(mm.ab) == nb, b_own = numel(mm.bb) == nb; // no batch broadcast: distinct targets
        auto batch = [&](int64_t q, int64_t &ia, int64_t &ib) {
          int64_t r = q;
          ia = ib = 0;
          for (int d = 3; d >= 0; d--) {
            const int64_t i = r % obn[d];
            r /= obn[d];
            ia += A.at(d, i), ib += B.at(d, i);
          }
        };
        // ga[m, k] += g[m, n] b[k, n]^T ; gb[k, n] += a[m, k]^T g[m, n]
        if (ga) {
          const bool bp = a_own && nb >= 8;
#pragma omp parallel for schedule(static) if (bp)
          for (int64_t q = 0; q < nb; q++) {
            int64_t ia, ib;
            batch(q, ia, ib);
            gemm<T>(mm.m, mm.k, mm.n, g + q * so, false, b.ptr() + ib * sb, true, ga + ia * sa, !bp);
          }
        }
        if (gb) {
          const bool bp = b_own && nb >= 8;
#pragma omp parallel for schedule(static) if (bp)
          for (int64_t q = 0; q < nb; q++) {
            int64_t ia, ib;
            batch(q, ia, ib);
            gemm<T>(mm.k, mm.n, mm.m, a.ptr() + ia * sa, true, g + q * so, false, gb + ib * sb, !bp);
          }
        }
        break;
      }
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("fp64 executor backward: node " + std::to_string(n.id) + " (" + n.op + "): " +
                               e.what());
    }
  }
}

template <class T> const std::vector<T> &Executor<T>::input_grad(const std::string &name) const {
  static const std::vector<T> none;
  auto it = in_grad_.find(name);
  return it == in_grad_.end() ? none : it->second;
}
template <class T> const std::vector<T> &Executor<T>::weight_grad(const std::string &name) const {
  static const std::vector<T> none;
  auto it = w_grad_.find(name);
  return it == w_grad_.end() ? none : it->second;
}
template <class T> const std::vector<T> *Executor<T>::node_grad(int id) const {
  auto it = index_.find(id);
  if (it == index_.end() || size_t(it->second) >= node_grads_.size() || node_grads_[it->second].empty()) return nullptr;
  return &node_grads_[it->second];
}

template class Executor<double>;
template class Executor<Dual>;

} // namespace mlipcpp::runtime::fp64
