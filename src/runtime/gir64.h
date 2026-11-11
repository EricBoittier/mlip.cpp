#pragma once
// fp64 executor of a traced GIR graph (PARITY.md "fp64 mode", option A): a double-precision host interpreter of
// the ops the iris PET-LR graphs use, in torch semantics (row-major tensors with torch shapes), not through ggml.
//   * forward:  Executor<double>::forward
//   * reverse:  Executor<double>::backward (an arbitrary cotangent on the output; adjoints of every float input and,
//               on request, of the weights; the forward values are kept, so several sweeps reuse one forward)
//   * forward mode (JVP): Executor<Dual> (value + tangent): seed tangents on inputs / weights, run forward; its
//               backward is forward-over-reverse (the mixed derivatives the trainer's GraphJVP + backward give)
// fp32-trace artefacts undone here: RMS_NORM eps = finfo(float32).eps -> finfo(float64).eps, scalar params kept
// in double (the ggml interpreter casts them to float).
// Semantics differences from the ggml lowering (deliberate): WHERE is a select (no c x + (1 - c) y), DIV by a
// scalar divides, POW is std::pow, no attention / softmax / chunk-linear fusions.

#include "graph_ir.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlipcpp::runtime::fp64 {

/// forward-mode dual number
struct Dual {
  double v = 0, d = 0;
  Dual() = default;
  Dual(double x) : v(x), d(0) {} // NOLINT (implicit: constants)
  Dual(double x, double t) : v(x), d(t) {}
};
inline Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
inline Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
inline Dual operator-(Dual a) { return {-a.v, -a.d}; }
inline Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
inline Dual operator/(Dual a, Dual b) { const double q = a.v / b.v; return {q, (a.d - q * b.d) / b.v}; }
inline Dual &operator+=(Dual &a, Dual b) { a.v += b.v, a.d += b.d; return a; }
inline Dual &operator-=(Dual &a, Dual b) { a.v -= b.v, a.d -= b.d; return a; }
inline Dual &operator*=(Dual &a, Dual b) { a = a * b; return a; }
inline Dual exp(Dual a) { const double e = std::exp(a.v); return {e, e * a.d}; }
inline Dual log(Dual a) { return {std::log(a.v), a.d / a.v}; }
inline Dual sqrt(Dual a) { const double s = std::sqrt(a.v); return {s, a.d / (2 * s)}; }
inline Dual sin(Dual a) { return {std::sin(a.v), std::cos(a.v) * a.d}; }
inline Dual cos(Dual a) { return {std::cos(a.v), -std::sin(a.v) * a.d}; }
inline Dual tanh(Dual a) { const double t = std::tanh(a.v); return {t, (1 - t * t) * a.d}; }
inline Dual erf(Dual a) { return {std::erf(a.v), 1.1283791670955126 * std::exp(-a.v * a.v) * a.d}; }
inline Dual pow(Dual a, double n) { return {std::pow(a.v, n), n * std::pow(a.v, n - 1) * a.d}; }
inline double val(double x) { return x; }
inline double val(const Dual &x) { return x.v; }

template <class T> struct Tensor {
  std::vector<int64_t> shape; // torch order; {} = scalar
  std::shared_ptr<std::vector<T>> data;
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
  }
  T *ptr() { return data->data(); }
  const T *ptr() const { return data->data(); }
  static Tensor zeros(std::vector<int64_t> s) {
    Tensor t;
    t.shape = std::move(s);
    t.data = std::make_shared<std::vector<T>>(size_t(t.numel()), T(0.0));
    return t;
  }
  static Tensor from(std::vector<int64_t> s, std::vector<T> v) {
    Tensor t;
    t.shape = std::move(s);
    t.data = std::make_shared<std::vector<T>>(std::move(v));
    return t;
  }
};

/// finfo(float32).eps, the RMSNorm eps a float32 trace bakes in
constexpr double kF32Eps = 1.1920928955078125e-07;

struct NodeInfo; // parsed op + params (gir64.cpp)

template <class T> class Executor {
public:
  explicit Executor(const GIRGraph &g);
  ~Executor();
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;

  void set_dimension(const std::string &name, int64_t v) { dims_[name] = v; }
  void set_weight(const std::string &name, Tensor<T> t) { weights_[name] = std::move(t); }
  void set_input(const std::string &name, Tensor<T> t) { inputs_[name] = std::move(t); }
  bool has_input(const std::string &name) const { return inputs_.count(name) > 0; }
  const Tensor<T> &input(const std::string &name) const { return inputs_.at(name); }
  /// RMS_NORM eps: finfo(float32).eps -> this (default finfo(float64).eps; kF32Eps keeps the trace's value)
  double rms_eps_f32_replacement = 2.220446049250313e-16;

  /// every node in order; keep_all: keep every value (needed for backward), else free a value after its last use
  void forward(bool keep_all = true);
  const Tensor<T> &output() const;
  const Tensor<T> *node_value(int id) const; // nullptr if freed / not computed
  const GIRGraph &graph() const { return g_; }

  /// reverse sweep of <cotangent, output>; adjoints of the float inputs (all of them) and, with weights = true, of
  /// the weights. Several sweeps may follow one forward (different cotangents).
  void backward(const std::vector<T> &out_cotangent, bool weights = false);
  /// adjoint of an input / weight after backward (empty if it does not influence the output)
  const std::vector<T> &input_grad(const std::string &name) const;
  const std::vector<T> &weight_grad(const std::string &name) const;
  /// d(output . cotangent)/d(node) of the last backward for every node (diagnostics), when keep_node_grads was set
  bool keep_node_grads = false;
  const std::vector<T> *node_grad(int id) const;

private:
  const GIRGraph &g_;
  std::vector<NodeInfo> info_;
  std::map<std::string, int64_t> dims_;
  std::map<std::string, Tensor<T>> weights_, inputs_;
  std::vector<Tensor<T>> vals_;             // by node index
  std::map<int, int> index_;                // node id -> index
  int out_index_ = -1;
  std::map<std::string, std::vector<T>> in_grad_, w_grad_;
  std::vector<std::vector<T>> node_grads_;
  std::vector<int64_t> resolve(const std::vector<int64_t> &shape) const;
  struct Ref;
  Ref ref(const std::string &r) const;
  const Tensor<T> &value(const std::string &r) const;
  Tensor<T> eval(size_t k);
};

extern template class Executor<double>;
extern template class Executor<Dual>;

/// the shortest decimal that rounds to float32 x (the value the exporter meant: 0.1f -> 0.1), as a double
double f32_intent(float x);

} // namespace mlipcpp::runtime::fp64
