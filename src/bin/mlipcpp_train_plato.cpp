// Training Plato-PET-LR (plato-pet-lr GGUFs) with conservative forces.
//
//   mlipcpp-train-plato --init model.gguf --train train.bin --val val.bin -o out.gguf
//       [--epochs 3] [--batch-size 16] [--lr 1e-4] [--schedule cosine|linear] [--warmup 0.01]
//       [--decay-after 10] [--min-lr 1e-6] [--energy-weight 1] [--force-weight 1] [--grad-clip 1]
//       [--seed 0] [--no-augment] [--fd-eps 5e-3] [--gpus 1] [--save-every 100] [--log FILE.jsonl]
//       [--resume STATE] [--state STATE] [--state-every 10] [--max-epochs-this-run N] [--max-hours H] [--backend cuda]
//   With --val, OUT.best.gguf holds the epoch with the highest R^2(E) + R^2(F) on it (LOREM's criterion).
//   --charge-weight W (datasets with molecules, plato_dataset.py --molecules): adds
//   W * mean_structures 1/2 sum_m (Q_m - Q'_m)^2, Q_m the latent charge (e, first channel) summed
//   over molecule m and Q'_m its formal charge, centred as the latent charges are
//   (Q'_m = Q_m^formal - Q_total n_m / n). Its parameter gradient inside the central difference
//   as the force term would lose it to float32 cancellation, so it is one more plain backward pass
//   with charge weights c_i = dL/dq_i on the graph's total and energy weights 0 (exact).
//   mlipcpp-train-plato --init model.gguf --train data.bin --gradcheck H [--batch-size 8] [weights]: the loss
//       along the trainer's gradient direction, (L(p + h g/|g|) - L(p - h g/|g|)) / 2h against |g|
//   mlipcpp-train-plato --init model.gguf --train data.bin --check REF   (REF.json / REF.f64 of
//       scripts/plato/plato_grad_reference.py: loss and parameter gradients vs PyTorch)
//
// The loss is metatrain's (plato_pet_lr trainer): MSE of the energies per atom and of
// the position gradients, targets with the composition removed and divided by the
// energy scale. The force term's parameter gradient is the derivative of the energy
// along the force residual u = dL/d(dE/dx): with T(+-) = sum_b (1 +- eps a_b) E_b(x +- eps u),
// a_b = dL/dE_b, the central difference (dT(+)/dtheta - dT(-)/dtheta) / (2 eps) is
// dL/dtheta up to O(eps^2). Three evaluations of one graph per micro-batch; the
// parameter gradient is accumulated on the device inside that graph.
//
// Speed: the structures of a micro-batch in one graph (packed tokens, one mesh with a
// channel block per structure, the Ewald sums per structure, one mesh filter and one
// g(k) for rotated copies of one cell); micro-batches padded to shape buckets (an extra
// structure of weight 0 holds the padding atoms and edges), so a few graphs are built
// once and replayed (CUDA graphs); batches prepared on a host thread ahead of the GPU;
// gradient-norm clipping and Adam on the device; several GPUs of a node as data
// parallel replicas with an NCCL all-reduce of the gradient (MLIPCPP_NCCL builds).
// Schedules: metatrain's (warmup + cosine) or LOREM's (constant for --decay-after
// epochs, then linear to --min-lr). Data: scripts/plato/plato_dataset.py.
#include "models/plato/plato.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>
#include <nlohmann/json.hpp>

#ifdef MLIPCPP_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace mlipcpp;
using runtime::PlatoInputs;

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

struct Structure {
  int n = 0;
  std::vector<int32_t> Z;
  std::vector<double> x, F; // [n][3]
  double cell[9] = {}, energy = 0, stress[9] = {};
  bool pbc = true, has_forces = false, has_stress = false;
  bool has_molecules = false; // two molecules: atoms [0, split) and [split, n), formal charges QA, QB
  int split = 0;
  double QA = 0, QB = 0;
};

std::vector<Structure> read_dataset(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "PLATOD1", 7) != 0) throw std::runtime_error(path + ": not a plato dataset");
  int64_t count = 0;
  f.read(reinterpret_cast<char *>(&count), 8);
  std::vector<Structure> out(count);
  for (auto &s : out) {
    f.read(reinterpret_cast<char *>(&s.n), 4);
    s.Z.resize(s.n), s.x.resize(3 * s.n);
    f.read(reinterpret_cast<char *>(s.Z.data()), 4 * s.n);
    f.read(reinterpret_cast<char *>(s.x.data()), 8 * 3 * s.n);
    f.read(reinterpret_cast<char *>(s.cell), 72);
    uint8_t fl[4];
    f.read(reinterpret_cast<char *>(fl), 4);
    s.pbc = fl[0], s.has_forces = fl[1], s.has_stress = fl[2], s.has_molecules = fl[3];
    f.read(reinterpret_cast<char *>(&s.energy), 8);
    if (s.has_forces) s.F.resize(3 * s.n), f.read(reinterpret_cast<char *>(s.F.data()), 8 * 3 * s.n);
    if (s.has_stress) f.read(reinterpret_cast<char *>(s.stress), 72);
    if (s.has_molecules) {
      f.read(reinterpret_cast<char *>(&s.split), 4), f.read(reinterpret_cast<char *>(&s.QA), 8), f.read(reinterpret_cast<char *>(&s.QB), 8);
      if (s.split <= 0 || s.split >= s.n) throw std::runtime_error(path + ": bad molecule split");
    }
  }
  return out;
}

std::string arg(int argc, char **argv, const char *name, const std::string &fallback) {
  for (int i = 1; i + 1 < argc; i++)
    if (!std::strcmp(argv[i], name)) return argv[i + 1];
  return fallback;
}
bool flag(int argc, char **argv, const char *name) {
  for (int i = 1; i < argc; i++)
    if (!std::strcmp(argv[i], name)) return true;
  return false;
}

using Mat3 = std::array<double, 9>;
constexpr Mat3 IDENTITY = {1, 0, 0, 0, 1, 0, 0, 0, 1};

// random O(3): Haar rotation (normalised quaternion) times +-1, as metatrain's O3Augmenter
Mat3 random_o3(std::mt19937_64 &rng) {
  std::normal_distribution<double> N(0, 1);
  double q[4], s = 0;
  for (double &v : q) v = N(rng), s += v * v;
  s = std::sqrt(s);
  const double w = q[0] / s, x = q[1] / s, y = q[2] / s, z = q[3] / s;
  const double sg = std::uniform_int_distribution<int>(0, 1)(rng) ? -1.0 : 1.0;
  return {sg * (1 - 2 * (y * y + z * z)), sg * 2 * (x * y - w * z),       sg * 2 * (x * z + w * y),
          sg * 2 * (x * y + w * z),       sg * (1 - 2 * (x * x + z * z)), sg * 2 * (y * z - w * x),
          sg * 2 * (x * z - w * y),       sg * 2 * (y * z + w * x),       sg * (1 - 2 * (x * x + y * y))};
}

void inverse3(const double *A, double *I, double &det) {
  det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  const double id = 1.0 / det;
  I[0] = (A[4] * A[8] - A[5] * A[7]) * id, I[1] = (A[2] * A[7] - A[1] * A[8]) * id, I[2] = (A[1] * A[5] - A[2] * A[4]) * id;
  I[3] = (A[5] * A[6] - A[3] * A[8]) * id, I[4] = (A[0] * A[8] - A[2] * A[6]) * id, I[5] = (A[2] * A[3] - A[0] * A[5]) * id;
  I[6] = (A[3] * A[7] - A[4] * A[6]) * id, I[7] = (A[1] * A[6] - A[0] * A[7]) * id, I[8] = (A[0] * A[4] - A[1] * A[3]) * id;
}

int64_t round_up(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// ---------------------------------------------------------------------------
// one micro-batch on the host, padded to its shape bucket

struct Prepared {
  int Greal = 0, G = 0, n_real = 0, n = 0, L = 0;
  int64_t E_real = 0, E = 0, nk = 0;
  int mesh[3] = {0, 0, 0};
  double norm_G = 1, norm_N = 1; // structures and atoms of the whole step (the loss means)
  std::vector<const Structure *> structs;
  std::vector<Mat3> rot;
  std::vector<int> a0;               // [Greal + 1]
  std::vector<int32_t> species, sidx, ecs, nbs, cen, ej, rev, off;
  std::vector<float> ev0, pos0, aict, onehot, invc, invv, mk, mpad, mdec, lk, ln, taper;
  std::vector<double> Et, Ft;        // scaled targets (rotated frame)
  std::tuple<int, int64_t, int, int, int64_t, int, int, int> key() const { return {n, E, L, G, nk, mesh[0], mesh[1], mesh[2]}; }
};

struct Context {
  const runtime::PlatoSpec &spec;
  double scale, spacing;
  const std::map<int, double> &composition;
  const std::map<int, int> &species_index;
  bool shared_filter = true;
};

Prepared prepare(const Context &cx, const std::vector<const Structure *> &batch, const std::vector<Mat3> &rot, void *&vesin) {
  Prepared P;
  const auto &spec = cx.spec;
  const int Gr = static_cast<int>(batch.size());
  P.Greal = Gr, P.G = Gr + 1, P.structs = batch, P.rot = rot;
  std::vector<std::vector<float>> X(Gr);
  std::vector<plato::Edges> ed(Gr);
  P.a0.assign(Gr + 1, 0);
  int L = 1;
  const Structure &s0 = *batch[0];
  for (int g = 0; g < Gr; g++) {
    const Structure &s = *batch[g];
    if (!s.pbc) throw std::runtime_error("training batches of periodic structures only");
    if (std::memcmp(s.cell, s0.cell, sizeof(s.cell)) != 0) throw std::runtime_error("a micro-batch needs one cell (rotations aside)");
    const double *R = rot[g].data();
    X[g].resize(3 * s.n);
    for (int i = 0; i < s.n; i++)
      for (int c = 0; c < 3; c++) X[g][3 * i + c] = float(R[3 * c] * s.x[3 * i] + R[3 * c + 1] * s.x[3 * i + 1] + R[3 * c + 2] * s.x[3 * i + 2]);
    float Lc[3][3];
    for (int a = 0; a < 3; a++)
      for (int c = 0; c < 3; c++) Lc[a][c] = float(R[3 * c] * s.cell[3 * a] + R[3 * c + 1] * s.cell[3 * a + 1] + R[3 * c + 2] * s.cell[3 * a + 2]);
    Cell cell(Lc, true, true, true);
    AtomicSystem sys(s.n, X[g].data(), s.Z.data(), &cell);
    ed[g] = plato::neighbor_edges(sys, spec.cutoff, vesin);
    for (int i = 0; i < s.n; i++) L = std::max<int>(L, 1 + int(ed[g].start[i + 1] - ed[g].start[i]));
    P.a0[g + 1] = P.a0[g] + s.n;
  }
  // the mesh from the unrotated cell: a rotation must not change it (docs/plato.md)
  for (int a = 0; a < 3; a++) {
    const float row[3] = {float(s0.cell[3 * a]), float(s0.cell[3 * a + 1]), float(s0.cell[3 * a + 2])};
    P.mesh[a] = plato::mesh_points(row, cx.spacing);
  }
  P.n_real = P.a0[Gr];
  std::vector<int64_t> e0(Gr + 1, 0);
  for (int g = 0; g < Gr; g++) e0[g + 1] = e0[g] + int64_t(ed[g].j.size());
  P.E_real = e0[Gr];
  // bucket: padding edges on padding atoms (each at most L - 1), one more structure
  P.L = int(std::max<int64_t>(8, round_up(L, 4)));
  P.E = round_up(std::max<int64_t>(P.E_real, 1), 1024);
  const int64_t pad_e = P.E - P.E_real, need = 1 + (pad_e + P.L - 2) / (P.L - 1);
  P.n = int(round_up(P.n_real + need, 128));
  const int n = P.n, G = P.G;
  const int64_t E = P.E, T = n + E;

  double Ci0[9], det0;
  inverse3(s0.cell, Ci0, det0);
  P.species.assign(n, 0), P.sidx.assign(n, Gr), P.ecs.resize(E), P.nbs.resize(E), P.cen.resize(E), P.ej.resize(E);
  P.rev.resize(T), P.off.resize(n + 1);
  P.ev0.resize(3 * E), P.pos0.assign(3 * size_t(n), 0.0f), P.aict.resize(9 * size_t(n)), P.onehot.assign(size_t(n) * G, 0.0f);
  P.invc.resize(G), P.invv.assign(1, float(1.0 / std::abs(det0)));
  for (int g = 0; g < Gr; g++) {
    const Structure &s = *batch[g];
    double Ci[9], det;
    double cr[9];
    const double *R = rot[g].data();
    for (int a = 0; a < 3; a++)
      for (int c = 0; c < 3; c++) cr[3 * a + c] = double(float(R[3 * c] * s.cell[3 * a] + R[3 * c + 1] * s.cell[3 * a + 1] + R[3 * c + 2] * s.cell[3 * a + 2]));
    inverse3(cr, Ci, det);
    P.invc[g] = 1.0f / float(s.n);
    for (int i = 0; i < s.n; i++) {
      const int A = P.a0[g] + i;
      auto it = cx.species_index.find(s.Z[i]);
      if (it == cx.species_index.end()) throw std::runtime_error("unknown species");
      P.species[A] = it->second, P.sidx[A] = g, P.rev[A] = A;
      P.onehot[size_t(g) * n + A] = 1.0f;
      for (int c = 0; c < 3; c++) P.pos0[3 * A + c] = X[g][3 * i + c];
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) P.aict[9 * A + j * 3 + k] = float(Ci[k * 3 + j]);
      P.off[A] = int32_t(e0[g] + ed[g].start[i]);
    }
    for (int i = 0; i < s.n; i++)
      for (int64_t e = ed[g].start[i]; e < ed[g].start[i + 1]; e++) {
        const int64_t ge = e0[g] + e;
        P.ecs[ge] = P.species[P.a0[g] + i], P.cen[ge] = P.a0[g] + i, P.ej[ge] = P.a0[g] + ed[g].j[e];
        P.nbs[ge] = P.species[P.ej[ge]];
        P.rev[n + ge] = int32_t(n + e0[g] + ed[g].rev[e]);
        for (int c = 0; c < 3; c++) P.ev0[3 * ge + c] = ed[g].v[3 * e + c];
      }
  }
  // padding atoms (structure Gr, weight 0): in the box centre, their edges to themselves
  const int n_pad = n - P.n_real;
  P.invc[Gr] = 1.0f / float(n_pad);
  int64_t e = P.E_real;
  for (int A = P.n_real; A < n; A++) {
    P.rev[A] = A, P.onehot[size_t(Gr) * n + A] = 1.0f;
    for (int c = 0; c < 3; c++) P.pos0[3 * A + c] = float(0.5 * (s0.cell[c] + s0.cell[3 + c] + s0.cell[6 + c]) + 0.01 * (A - P.n_real));
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) P.aict[9 * A + j * 3 + k] = float(Ci0[k * 3 + j]);
    P.off[A] = int32_t(e);
    for (int q = 0; q < P.L - 1 && e < E; q++, e++) {
      P.ecs[e] = 0, P.nbs[e] = 0, P.cen[e] = A, P.ej[e] = A, P.rev[n + e] = int32_t(n + e);
      P.ev0[3 * e] = 0.5f * spec.cutoff, P.ev0[3 * e + 1] = 0.01f * q, P.ev0[3 * e + 2] = 0.0f;
    }
  }
  P.off[n] = int32_t(E);
  if (e != E) throw std::runtime_error("padding edges");
  // shared long-range constants: from the unrotated cell (only |k| enters)
  const int M0 = P.mesh[0], M1 = P.mesh[1], M2 = P.mesh[2], Mh2 = M2 / 2 + 1;
  const int64_t Mh = int64_t(M0) * M1 * Mh2;
  auto freq = [](int i, int M) { return i <= (M - 1) / 2 ? i : i - M; };
  if (spec.mesh) {
    P.mk.resize(3 * Mh), P.mpad.assign(Mh, 0.0f), P.mdec.resize(Mh);
    for (int i0 = 0; i0 < M0; i0++)
      for (int i1 = 0; i1 < M1; i1++)
        for (int i2 = 0; i2 < Mh2; i2++) {
          const int64_t q = (int64_t(i0) * M1 + i1) * Mh2 + i2;
          const int m[3] = {freq(i0, M0), freq(i1, M1), i2};
          double dc = 1;
          for (int a = 0; a < 3; a++) {
            const double f = (4 + 2 * std::cos(2 * M_PI * m[a] / P.mesh[a])) / 6;
            dc *= f * f;
          }
          for (int b = 0; b < 3; b++) P.mk[3 * q + b] = float(2 * M_PI * (m[0] * Ci0[3 * b] + m[1] * Ci0[3 * b + 1] + m[2] * Ci0[3 * b + 2]));
          P.mpad[q] = q == 0 ? 1.0f : 0.0f, P.mdec[q] = q == 0 ? 0.0f : float(1.0 / dc);
        }
  }
  if (spec.les) {
    const double kc = spec.les_kcut / spec.les_sigma, wd = 1.0 / spec.les_sigma;
    int nmax[3];
    for (int a = 0; a < 3; a++)
      nmax[a] = int(std::floor(kc * std::sqrt(s0.cell[3 * a] * s0.cell[3 * a] + s0.cell[3 * a + 1] * s0.cell[3 * a + 1] + s0.cell[3 * a + 2] * s0.cell[3 * a + 2]) / (2 * M_PI)));
    for (int a = -nmax[0]; a <= nmax[0]; a++)
      for (int b = -nmax[1]; b <= nmax[1]; b++)
        for (int c = -nmax[2]; c <= nmax[2]; c++) {
          if (!(a > 0 || (a == 0 && b > 0) || (a == 0 && b == 0 && c > 0))) continue;
          double k[3], k2 = 0;
          for (int q = 0; q < 3; q++) k[q] = 2 * M_PI * (a * Ci0[3 * q] + b * Ci0[3 * q + 1] + c * Ci0[3 * q + 2]), k2 += k[q] * k[q];
          if (!(k2 < kc * kc)) continue;
          for (int q = 0; q < 3; q++) P.lk.push_back(float(k[q]));
          P.ln.push_back(float(2 * M_PI * a)), P.ln.push_back(float(2 * M_PI * b)), P.ln.push_back(float(2 * M_PI * c));
          P.taper.push_back(std::sqrt(k2) >= kc - wd ? 1.0f : 0.0f);
        }
    P.nk = int64_t(P.taper.size());
  }
  // targets (scaled, composition removed) in the rotated frame
  P.Et.resize(Gr), P.Ft.resize(3 * size_t(P.n_real));
  for (int g = 0; g < Gr; g++) {
    const Structure &s = *batch[g];
    double c = 0;
    for (int z : s.Z)
      if (auto it = cx.composition.find(z); it != cx.composition.end()) c += it->second;
    P.Et[g] = (s.energy - c) / cx.scale;
    const double *R = rot[g].data();
    for (int i = 0; i < s.n; i++)
      for (int a = 0; a < 3; a++)
        P.Ft[3 * (P.a0[g] + i) + a] = s.has_forces ? (R[3 * a] * s.F[3 * i] + R[3 * a + 1] * s.F[3 * i + 1] + R[3 * a + 2] * s.F[3 * i + 2]) / cx.scale : 0.0;
  }
  return P;
}

// ---------------------------------------------------------------------------
// a graph of one bucket: forward, backward, and acc_k += s * dT/dtheta_k

struct BatchGraph {
  ggml_context *ctx = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_backend_buffer_t ibuf = nullptr;
  ggml_gallocr_t galloc = nullptr;
  PlatoInputs in;
  runtime::PlatoOutputs out;
  ggml_tensor *g_ev = nullptr, *g_pos = nullptr, *loss_grad = nullptr;
  bool constants_set = false;
  ~BatchGraph() {
    if (galloc) ggml_gallocr_free(galloc);
    if (ibuf) ggml_backend_buffer_free(ibuf);
    if (ctx) ggml_free(ctx);
  }
};

struct Worker {
  int device = 0;
  std::unique_ptr<plato::PlatoModel> model;
  std::vector<std::string> names;
  std::vector<ggml_tensor *> params, acc, m, v;
  std::vector<int64_t> offsets;
  int64_t n_values = 0;
  ggml_context *ctx_opt = nullptr, *ctx_step = nullptr;
  ggml_backend_buffer_t acc_buf = nullptr, mv_buf = nullptr, step_inputs = nullptr;
  ggml_cgraph *step_graph = nullptr;
  ggml_gallocr_t step_galloc = nullptr;
  ggml_tensor *adamw = nullptr, *clip_value = nullptr, *sumsq = nullptr;
  std::list<std::pair<std::tuple<int, int64_t, int, int, int64_t, int, int, int>, std::unique_ptr<BatchGraph>>> cache;
  size_t cache_limit = 12;
  double fd_eps = 5e-3, energy_weight = 1, force_weight = 1, charge_weight = 0;
  double t_build = 0, t_compute = 0, t_host = 0; // ms
  int64_t builds = 0;

  void init(const std::string &gguf, const plato::PlatoOptions &o, const std::vector<std::string> &param_names) {
    model = std::make_unique<plato::PlatoModel>(o);
    model->load_from_gguf(gguf);
    for (const auto &nm : param_names) {
      auto it = model->weights().find(nm);
      if (it == model->weights().end()) throw std::runtime_error("no weight " + nm);
      params.push_back(it->second), names.push_back(nm), offsets.push_back(n_values);
      n_values += ggml_nelements(it->second);
      ggml_set_param(it->second);
    }
    // accumulators (one buffer: the all-reduce), Adam moments
    ctx_opt = ggml_init({ggml_tensor_overhead() * (3 * params.size() + 4), nullptr, true});
    for (ggml_tensor *p : params) acc.push_back(ggml_dup_tensor(ctx_opt, p));
    ggml_context *ctx_mv = ctx_opt;
    acc_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_opt, model->backend()->buffer_type());
    ggml_context *c2 = ggml_init({ggml_tensor_overhead() * (2 * params.size() + 4), nullptr, true});
    for (ggml_tensor *p : params) m.push_back(ggml_dup_tensor(c2, p)), v.push_back(ggml_dup_tensor(c2, p));
    mv_buf = ggml_backend_alloc_ctx_tensors_from_buft(c2, model->backend()->buffer_type());
    ggml_backend_buffer_clear(mv_buf, 0), ggml_backend_buffer_clear(acc_buf, 0);
    (void)ctx_mv;
    ctx_mv_ = c2;
    // the step graph: clip by the global norm, Adam
    const size_t GS = 4 * params.size() + 64;
    ctx_step = ggml_init({ggml_tensor_overhead() * GS * 4 + ggml_graph_overhead_custom(GS * 4, false), nullptr, true});
    adamw = ggml_new_tensor_1d(ctx_step, GGML_TYPE_F32, 7), ggml_set_input(adamw);
    clip_value = ggml_new_tensor_1d(ctx_step, GGML_TYPE_F32, 1), ggml_set_input(clip_value);
    step_graph = ggml_new_graph_custom(ctx_step, GS * 4, false);
    for (size_t k = 0; k < params.size(); k++) {
      ggml_tensor *s2 = ggml_sum(ctx_step, ggml_sqr(ctx_step, acc[k]));
      sumsq = sumsq ? ggml_add(ctx_step, sumsq, s2) : s2;
    }
    ggml_set_output(sumsq);
    ggml_tensor *f = ggml_clamp(ctx_step, ggml_div(ctx_step, clip_value, ggml_scale_bias(ctx_step, ggml_sqrt(ctx_step, sumsq), 1.0f, 1e-6f)), 0.0f, 1.0f);
    ggml_build_forward_expand(step_graph, sumsq);
    for (size_t k = 0; k < params.size(); k++)
      ggml_build_forward_expand(step_graph, ggml_opt_step_adamw(ctx_step, params[k], ggml_mul(ctx_step, acc[k], f), m[k], v[k], adamw));
    ggml_context *ci = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
    (void)ci;
    ggml_free(ci);
    step_inputs = ggml_backend_buft_alloc_buffer(model->backend()->buffer_type(), 1024);
    char *base = static_cast<char *>(ggml_backend_buffer_get_base(step_inputs));
    const size_t al = ggml_backend_buft_get_alignment(model->backend()->buffer_type());
    ggml_backend_tensor_alloc(step_inputs, adamw, base), ggml_backend_tensor_alloc(step_inputs, clip_value, base + GGML_PAD(64, al));
    step_galloc = ggml_gallocr_new(model->backend()->buffer_type());
    if (!ggml_gallocr_alloc_graph(step_galloc, step_graph)) throw std::runtime_error("cannot allocate the step graph");
  }
  ggml_context *ctx_mv_ = nullptr;
  ~Worker() {
    cache.clear();
    if (step_galloc) ggml_gallocr_free(step_galloc);
    if (step_inputs) ggml_backend_buffer_free(step_inputs);
    if (acc_buf) ggml_backend_buffer_free(acc_buf);
    if (mv_buf) ggml_backend_buffer_free(mv_buf);
    if (ctx_step) ggml_free(ctx_step);
    if (ctx_opt) ggml_free(ctx_opt);
    if (ctx_mv_) ggml_free(ctx_mv_);
  }

  std::vector<float> get_params() const {
    std::vector<float> out(n_values);
    for (size_t k = 0; k < params.size(); k++) ggml_backend_tensor_get(params[k], out.data() + offsets[k], 0, ggml_nbytes(params[k]));
    return out;
  }
  void set_params(const std::vector<float> &x) {
    for (size_t k = 0; k < params.size(); k++) ggml_backend_tensor_set(params[k], x.data() + offsets[k], 0, ggml_nbytes(params[k]));
  }
  std::vector<float> get_moments(bool second) const {
    const auto &t = second ? v : m;
    std::vector<float> out(n_values);
    for (size_t k = 0; k < t.size(); k++) ggml_backend_tensor_get(t[k], out.data() + offsets[k], 0, ggml_nbytes(t[k]));
    return out;
  }
  void set_moments(const std::vector<float> &x, bool second) {
    auto &t = second ? v : m;
    for (size_t k = 0; k < t.size(); k++) ggml_backend_tensor_set(t[k], x.data() + offsets[k], 0, ggml_nbytes(t[k]));
  }
  std::vector<float> get_acc() const {
    std::vector<float> out(n_values);
    for (size_t k = 0; k < acc.size(); k++) ggml_backend_tensor_get(acc[k], out.data() + offsets[k], 0, ggml_nbytes(acc[k]));
    return out;
  }
  void zero_acc() { ggml_backend_buffer_clear(acc_buf, 0); }

  BatchGraph &graph_for(const Prepared &P) {
    const auto key = P.key();
    for (auto it = cache.begin(); it != cache.end(); ++it)
      if (it->first == key) {
        cache.splice(cache.begin(), cache, it);
        return *cache.front().second;
      }
    const auto t0 = clk::now();
    if (cache.size() >= cache_limit) cache.pop_back();
    auto bg = std::make_unique<BatchGraph>();
    const auto &spec = model->spec();
    const int n = P.n, G = P.G, M0 = P.mesh[0], M1 = P.mesh[1], M2 = P.mesh[2];
    const int64_t E = P.E, T = n + E, nk = P.nk, Mh = int64_t(M0) * M1 * (M2 / 2 + 1);
    const size_t GS = 65536;
    bg->ctx = ggml_init({ggml_tensor_overhead() * GS * 2 + ggml_graph_overhead_custom(GS, true), nullptr, true});
    ggml_context *ctx = bg->ctx;
    std::vector<ggml_tensor *> inputs;
    auto input = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, const char *nm) {
      ggml_tensor *t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
      ggml_set_name(t, nm), ggml_set_input(t), inputs.push_back(t);
      return t;
    };
    const int64_t hdF = int64_t(spec.head_dim() / 2) * spec.frames;
    PlatoInputs &in = bg->in;
    in.n_struct = G, in.shared_filter = true;
    in.species = input(GGML_TYPE_I32, n, 1, 1, "species");
    in.edge_center_species = input(GGML_TYPE_I32, E, 1, 1, "edge_center_species");
    in.neighbor_species = input(GGML_TYPE_I32, E, 1, 1, "neighbor_species");
    in.edge_vectors = input(GGML_TYPE_F32, 3, E, 1, "edge_vectors");
    in.reverse = input(GGML_TYPE_I32, T, 1, 1, "reverse");
    in.offsets = input(GGML_TYPE_I32, n + 1, 1, 1, "offsets");
    in.centers = input(GGML_TYPE_I32, E, 1, 1, "centers");
    in.zero_ang = input(GGML_TYPE_F32, hdF, n, 1, "zero_ang");
    in.rbf0 = input(GGML_TYPE_F32, spec.n_rbf, 1, 1, "rbf0");
    in.rel_identity = input(GGML_TYPE_F32, 9, 1, 1, "rel_identity");
    in.struct_index = input(GGML_TYPE_I32, n, 1, 1, "struct_index");
    in.energy_weights = input(GGML_TYPE_F32, 1, G, 1, "energy_weights");
    if (spec.les) in.charge_weights = input(GGML_TYPE_F32, n, 1, 1, "charge_weights");
    in.positions = input(GGML_TYPE_F32, 3, n, 1, "positions");
    in.strain = input(GGML_TYPE_F32, 3, 3, 1, "strain");
    in.atom_inv_cell_t = input(GGML_TYPE_F32, 3, 3, n, "atom_inv_cell_t");
    in.inv_volume = input(GGML_TYPE_F32, 1, 1, 1, "inv_volume");
    in.onehot = input(GGML_TYPE_F32, n, G, 1, "onehot");
    in.inv_count = input(GGML_TYPE_F32, 1, G, 1, "inv_count");
    for (int a = 0; a < 3; a++) in.mesh_size[a] = P.mesh[a];
    if (spec.mesh) {
      in.mesh_k = input(GGML_TYPE_F32, 3, Mh, 1, "mesh_k");
      in.mesh_k0_pad = input(GGML_TYPE_F32, 1, Mh, 1, "mesh_k0_pad");
      in.mesh_deconv = input(GGML_TYPE_F32, 1, Mh, 1, "mesh_deconv");
    }
    if (spec.les) {
      in.les_k = input(GGML_TYPE_F32, 3, nk, 1, "les_k");
      in.les_n = input(GGML_TYPE_F32, 3, nk, 1, "les_n");
      in.les_taper = input(GGML_TYPE_F32, 1, nk, 1, "les_taper");
    }
    ggml_set_param(in.edge_vectors), ggml_set_param(in.positions);
    bg->out = runtime::build_plato(ctx, spec, model->weights(), in, n, E, P.L, true);
    ggml_set_output(bg->out.energies);
    if (bg->out.charges) ggml_set_output(bg->out.charges);
    bg->graph = ggml_new_graph_custom(ctx, GS, true);
    ggml_build_forward_expand(bg->graph, bg->out.energies);
    ggml_build_forward_expand(bg->graph, bg->out.total);
    ggml_set_loss(bg->out.total);
    ggml_build_backward_expand(ctx, bg->graph, nullptr);
    bg->g_ev = ggml_graph_get_grad(bg->graph, in.edge_vectors), bg->g_pos = ggml_graph_get_grad(bg->graph, in.positions);
    for (ggml_tensor *t = bg->g_ev; t; t = t->view_src) ggml_set_output(t);
    for (ggml_tensor *t = bg->g_pos; t; t = t->view_src) ggml_set_output(t);
    bg->loss_grad = ggml_graph_get_grad_acc(bg->graph, bg->out.total);
    ggml_set_input(bg->loss_grad), inputs.push_back(bg->loss_grad);
    for (size_t k = 0; k < params.size(); k++) {
      ggml_tensor *g = ggml_graph_get_grad(bg->graph, params[k]);
      if (!g) continue;
      // the loss seed carries the scale (+-1 / 2 eps): acc_k += dT/dtheta_k
      ggml_build_forward_expand(bg->graph, ggml_add_inplace(ctx, acc[k], ggml_reshape(ctx, g, acc[k])));
    }
    ggml_backend_buffer_type_t buft = model->backend()->buffer_type();
    const size_t al = ggml_backend_buft_get_alignment(buft);
    size_t size = 0;
    std::vector<size_t> offs;
    for (ggml_tensor *t : inputs) offs.push_back(size), size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), al);
    bg->ibuf = ggml_backend_buft_alloc_buffer(buft, size);
    char *base = static_cast<char *>(ggml_backend_buffer_get_base(bg->ibuf));
    for (size_t k = 0; k < inputs.size(); k++) ggml_backend_tensor_alloc(bg->ibuf, inputs[k], base + offs[k]);
    bg->galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(bg->galloc, bg->graph)) throw std::runtime_error("cannot allocate the training graph");
    // inputs that are the same for every batch of the bucket
    std::vector<float> zeros(ggml_nelements(in.zero_ang), 0.0f);
    ggml_backend_tensor_set(in.zero_ang, zeros.data(), 0, ggml_nbytes(in.zero_ang));
    std::vector<float> rbf0(spec.n_rbf);
    const double w = spec.cutoff / spec.n_rbf;
    for (int k = 0; k < spec.n_rbf; k++) {
      const double mu = spec.cutoff * double(float(k) / float(spec.n_rbf - 1));
      rbf0[k] = float(std::exp(-std::pow((1e-6 - mu) / w, 2)));
    }
    ggml_backend_tensor_set(in.rbf0, rbf0.data(), 0, ggml_nbytes(in.rbf0));
    const float I9[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, Z9[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, one = 1.0f;
    ggml_backend_tensor_set(in.rel_identity, I9, 0, sizeof(I9));
    ggml_backend_tensor_set(in.strain, Z9, 0, sizeof(Z9));
    ggml_backend_tensor_set(bg->loss_grad, &one, 0, sizeof(one));
    cache.emplace_front(key, std::move(bg));
    t_build += ms_since(t0), builds++;
    return *cache.front().second;
  }

  struct Result {
    double loss_e = 0, loss_f = 0;   // sums over the micro-batch, normalised by the step's norm_G / norm_N
    std::vector<double> E;           // physical energies
    std::vector<std::vector<double>> F;
    double e_abs = 0, f_abs = 0;     // sum |dE| / n per structure, sum |dF| (physical)
    int64_t f_count = 0;
    // for R^2 as LOREM (marathon) computes it: per-atom energies after the composition baseline,
    // force components pooled
    double e_sq = 0, f_sq = 0, f_sum = 0, f_sum2 = 0;
    std::vector<double> e_target; // (E - composition) / n per structure
    // molecule charges: loss (normalised by norm_G), sum |Q_m - Q'_m| over molecules, and the
    // sign of Q_m against the formal charge of the charged molecules
    double loss_q = 0, q_abs = 0;
    int64_t q_count = 0, q_sign_ok = 0, q_charged = 0;
  };

  // eval0 for the forces; with `train` the two displaced evaluations accumulate
  // dL/dtheta (this micro-batch's share) into acc
  Result evaluate(const Prepared &P, bool train) {
    BatchGraph &bg = graph_for(P);
    const auto t0 = clk::now();
    const PlatoInputs &in = bg.in;
    auto set = [](ggml_tensor *t, const void *d) { ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t)); };
    const int n = P.n, G = P.G, Gr = P.Greal;
    const int64_t E = P.E;
    set(in.species, P.species.data()), set(in.struct_index, P.sidx.data()), set(in.edge_center_species, P.ecs.data());
    set(in.neighbor_species, P.nbs.data()), set(in.centers, P.cen.data()), set(in.reverse, P.rev.data()), set(in.offsets, P.off.data());
    set(in.atom_inv_cell_t, P.aict.data()), set(in.onehot, P.onehot.data()), set(in.inv_count, P.invc.data());
    set(in.inv_volume, P.invv.data());
    if (!bg.constants_set) { // the long-range constants of the (one) cell
      if (in.mesh_k) set(in.mesh_k, P.mk.data()), set(in.mesh_k0_pad, P.mpad.data()), set(in.mesh_deconv, P.mdec.data());
      if (in.les_k) set(in.les_k, P.lk.data()), set(in.les_n, P.ln.data()), set(in.les_taper, P.taper.data());
      bg.constants_set = true;
    }
    auto run = [&](const std::vector<float> &ev, const std::vector<float> &pos, const std::vector<float> &w, const std::vector<float> &cw,
                   float s) {
      set(in.edge_vectors, ev.data()), set(in.positions, pos.data()), set(in.energy_weights, w.data()), set(bg.loss_grad, &s);
      if (in.charge_weights) set(in.charge_weights, cw.data());
      const auto a = clk::now();
      if (model->backend()->compute(bg.graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
      t_compute += ms_since(a);
    };
    std::vector<float> w1(G, 1.0f);
    w1[Gr] = 0.0f;
    t_host += ms_since(t0);
    const std::vector<float> cw0(n, 0.0f);
    run(P.ev0, P.pos0, w1, cw0, 1.0f);
    const auto t1 = clk::now();
    std::vector<float> Eraw(G), gev(3 * E), gpos(3 * size_t(n)), q(bg.out.charges ? n : 0);
    ggml_backend_tensor_get(bg.out.energies, Eraw.data(), 0, 4 * G);
    if (bg.out.charges) ggml_backend_tensor_get(bg.out.charges, q.data(), 0, 4 * size_t(n)); // first channel
    ggml_backend_tensor_get(bg.g_ev, gev.data(), 0, gev.size() * 4);
    ggml_backend_tensor_get(bg.g_pos, gpos.data(), 0, gpos.size() * 4);
    zero_acc(); // eval0 accumulated dT/dtheta with seed 1
    std::vector<double> dEdx(3 * size_t(P.n_real), 0.0);
    for (int A = 0; A < P.n_real; A++)
      for (int c = 0; c < 3; c++) dEdx[3 * A + c] = gpos[3 * A + c];
    for (int64_t e = 0; e < P.E_real; e++)
      for (int c = 0; c < 3; c++) dEdx[3 * P.cen[e] + c] -= gev[3 * e + c], dEdx[3 * P.ej[e] + c] += gev[3 * e + c];
    Result r;
    std::vector<double> a(G, 0.0), u(3 * size_t(n), 0.0);
    for (int g = 0; g < Gr; g++) {
      const double nb = P.structs[g]->n, de = (Eraw[g] - P.Et[g]) / nb;
      r.loss_e += de * de / P.norm_G;
      a[g] = energy_weight * 2 * de / (P.norm_G * nb);
    }
    for (int A = 0; A < P.n_real; A++)
      for (int c = 0; c < 3; c++) {
        const double d = dEdx[3 * A + c] + P.Ft[3 * A + c]; // dE/dx - (-F / scale)
        r.loss_f += d * d / (3.0 * P.norm_N);
        u[3 * A + c] = force_weight * 2 * d / (3.0 * P.norm_N);
      }
    const double scale = model->energy_scale();
    std::vector<double> cq(n, 0.0); // dL/dq_i of the molecule-charge term
    if (!q.empty()) {
      const double unit = std::sqrt(scale); // latent charge in e = sqrt(energy scale) q
      for (int g = 0; g < Gr; g++) {
        const Structure &s = *P.structs[g];
        if (!s.has_molecules) continue;
        const double Qt = s.QA + s.QB;
        const int lo[2] = {0, s.split}, hi[2] = {s.split, s.n};
        const double Qf[2] = {s.QA, s.QB};
        for (int m = 0; m < 2; m++) {
          double S = 0;
          for (int i = lo[m]; i < hi[m]; i++) S += unit * q[P.a0[g] + i];
          const double target = Qf[m] - Qt * double(hi[m] - lo[m]) / s.n, res = S - target;
          r.loss_q += 0.5 * res * res / P.norm_G;
          r.q_abs += std::abs(res), r.q_count++;
          if (Qf[m] != 0) r.q_charged++, r.q_sign_ok += (S > 0) == (Qf[m] > 0);
          for (int i = lo[m]; i < hi[m]; i++) cq[P.a0[g] + i] = charge_weight * res * unit / P.norm_G;
        }
      }
    }
    r.E.resize(Gr), r.F.resize(Gr);
    for (int g = 0; g < Gr; g++) {
      const Structure &s = *P.structs[g];
      r.E[g] = scale * Eraw[g] + (scale * P.Et[g] - (s.energy - 0.0)) * -1.0; // = scale E_raw + composition
      r.e_abs += std::abs(scale * (Eraw[g] - P.Et[g])) / s.n;
      r.e_sq += std::pow(scale * (Eraw[g] - P.Et[g]) / s.n, 2);
      r.e_target.push_back(scale * P.Et[g] / s.n);
      r.F[g].resize(3 * s.n);
      const double *R = P.rot[g].data(); // back to the dataset frame: R^T
      for (int i = 0; i < s.n; i++)
        for (int a2 = 0; a2 < 3; a2++) {
          double vv = 0;
          for (int b = 0; b < 3; b++) vv += R[3 * b + a2] * (-scale * dEdx[3 * (P.a0[g] + i) + b]);
          r.F[g][3 * i + a2] = vv;
          if (s.has_forces) {
            const double y = s.F[3 * i + a2];
            r.f_abs += std::abs(vv - y), r.f_sq += (vv - y) * (vv - y), r.f_sum += y, r.f_sum2 += y * y, r.f_count++;
          }
        }
    }
    if (train) {
      // eps: displacements of at most fd_eps A and energy weights within 1 +- fd_eps
      double umax = 0;
      for (double x : u) umax = std::max(umax, std::abs(x));
      for (double x : a) umax = std::max(umax, std::abs(x));
      const double eps = umax > 0 ? fd_eps / umax : fd_eps;
      std::vector<float> ev(3 * E), pos(3 * size_t(n)), w(G, 0.0f), cw(n, 0.0f);
      for (int sgn : {+1, -1}) {
        for (int A = 0; A < n; A++)
          for (int c = 0; c < 3; c++) pos[3 * A + c] = float(P.pos0[3 * A + c] + sgn * eps * u[3 * A + c]);
        for (int64_t e = 0; e < E; e++)
          for (int c = 0; c < 3; c++) ev[3 * e + c] = float(P.ev0[3 * e + c] + sgn * eps * (u[3 * P.ej[e] + c] - u[3 * P.cen[e] + c]));
        for (int g = 0; g < Gr; g++) w[g] = float(1.0 + sgn * eps * a[g]);
        t_host += ms_since(t1);
        run(ev, pos, w, cw0, float(sgn / (2 * eps)));
      }
      // the charge term is linear in q: its gradient is one plain backward pass of sum_i cq_i q_i
      // (energy weights 0), exact; inside the central difference it would be lost to the
      // float32 cancellation of the weight-1 energies
      if (charge_weight > 0 && in.charge_weights) {
        std::vector<float> w0s(G, 0.0f);
        for (int i = 0; i < n; i++) cw[i] = float(cq[i]);
        run(P.ev0, P.pos0, w0s, cw, 1.0f);
      }
    } else {
      t_host += ms_since(t1);
    }
    return r;
  }

  // clip and Adam step on the device; returns the gradient norm
  double step(double lr, int64_t t, double clip) {
    const float p[7] = {float(lr), 0.9f, 0.999f, 1e-8f, 0.0f, float(1.0 / (1.0 - std::pow(0.9, double(t)))),
                        float(1.0 / (1.0 - std::pow(0.999, double(t))))};
    const float c = float(clip);
    ggml_backend_tensor_set(adamw, p, 0, sizeof(p));
    ggml_backend_tensor_set(clip_value, &c, 0, sizeof(c));
    const auto a = clk::now();
    if (model->backend()->compute(step_graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("step failed");
    t_compute += ms_since(a);
    float s2 = 0;
    ggml_backend_tensor_get(sumsq, &s2, 0, sizeof(s2));
    return std::sqrt(double(s2));
  }
};

#ifdef MLIPCPP_NCCL
void nccl_check(ncclResult_t r, const char *what) {
  if (r != ncclSuccess) throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(r));
}
struct Nccl {
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
  std::vector<int> devices;
  explicit Nccl(int W) {
    comms.resize(W), streams.resize(W);
    for (int d = 0; d < W; d++) devices.push_back(d);
    nccl_check(ncclCommInitAll(comms.data(), W, devices.data()), "ncclCommInitAll");
    for (int d = 0; d < W; d++) cudaSetDevice(d), cudaStreamCreate(&streams[d]);
  }
  ~Nccl() {
    for (size_t d = 0; d < comms.size(); d++) ncclCommDestroy(comms[d]), cudaSetDevice(int(d)), cudaStreamDestroy(streams[d]);
  }
  void allreduce_sum(std::vector<std::unique_ptr<Worker>> &ws) {
    nccl_check(ncclGroupStart(), "ncclGroupStart");
    for (size_t w = 0; w < ws.size(); w++) {
      void *ptr = ggml_backend_buffer_get_base(ws[w]->acc_buf);
      const size_t count = ggml_backend_buffer_get_size(ws[w]->acc_buf) / sizeof(float);
      nccl_check(ncclAllReduce(ptr, ptr, count, ncclFloat, ncclSum, comms[w], streams[w]), "ncclAllReduce");
    }
    nccl_check(ncclGroupEnd(), "ncclGroupEnd");
    for (size_t w = 0; w < ws.size(); w++) cudaSetDevice(int(w)), cudaStreamSynchronize(streams[w]);
  }
};
#endif

void save_gguf(const std::string &source, const std::string &path, const Worker &w, const std::vector<float> &values,
               const nlohmann::json &info) {
  ggml_context *data = nullptr;
  gguf_context *in = gguf_init_from_file(source.c_str(), {false, &data});
  if (!in) throw std::runtime_error("cannot read " + source);
  for (size_t k = 0; k < w.params.size(); k++) {
    ggml_tensor *t = ggml_get_tensor(data, w.names[k].c_str());
    std::memcpy(t->data, values.data() + w.offsets[k], ggml_nbytes(t));
  }
  gguf_context *out = gguf_init_empty();
  gguf_set_kv(out, in);
  for (int64_t i = 0; i < gguf_get_n_tensors(in); ++i) gguf_add_tensor(out, ggml_get_tensor(data, gguf_get_tensor_name(in, i)));
  gguf_set_val_str(out, "train.mlipcpp_plato", info.dump().c_str());
  const std::string tmp = path + ".tmp";
  const bool ok = gguf_write_to_file(out, tmp.c_str(), false);
  gguf_free(out), gguf_free(in), ggml_free(data);
  if (!ok) throw std::runtime_error("cannot write " + path);
  std::filesystem::rename(tmp, path);
}

std::vector<std::string> param_names(const std::string &path) {
  gguf_context *g = gguf_init_from_file(path.c_str(), {true, nullptr});
  const int64_t k = gguf_find_key(g, "train.param_names");
  if (k < 0) throw std::runtime_error("no train.param_names");
  auto names = nlohmann::json::parse(gguf_get_val_str(g, k)).get<std::vector<std::string>>();
  gguf_free(g);
  return names;
}

// the trainer state of a chain of jobs: parameters (as the GGUF), Adam moments, counters, RNG
struct State {
  int64_t step = 0;
  int epoch = 0;
  std::string rng;
  double best = 1e300;
  std::vector<float> m, v;
};
void save_state(const std::string &path, const State &s) {
  std::ofstream f(path + ".tmp", std::ios::binary);
  const nlohmann::json h = {{"step", s.step}, {"epoch", s.epoch}, {"rng", s.rng}, {"best", s.best}, {"n", s.m.size()}};
  const std::string hs = h.dump();
  const uint64_t hl = hs.size();
  f.write(reinterpret_cast<const char *>(&hl), 8), f.write(hs.data(), hl);
  f.write(reinterpret_cast<const char *>(s.m.data()), 4 * s.m.size()), f.write(reinterpret_cast<const char *>(s.v.data()), 4 * s.v.size());
  f.close();
  std::filesystem::rename(path + ".tmp", path);
}
State load_state(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  uint64_t hl = 0;
  f.read(reinterpret_cast<char *>(&hl), 8);
  std::string hs(hl, '\0');
  f.read(hs.data(), hl);
  const auto h = nlohmann::json::parse(hs);
  State s;
  s.step = h["step"], s.epoch = h["epoch"], s.rng = h["rng"], s.best = h["best"];
  const size_t n = h["n"];
  s.m.resize(n), s.v.resize(n);
  f.read(reinterpret_cast<char *>(s.m.data()), 4 * n), f.read(reinterpret_cast<char *>(s.v.data()), 4 * n);
  return s;
}

// batches prepared on host threads, in step order
struct Prefetcher {
  std::mutex mu;
  std::condition_variable cv;
  std::map<int64_t, std::vector<Prepared>> ready; // step -> micro-batches (one per worker)
  std::vector<std::thread> threads;
  std::atomic<int64_t> next{0};
  int64_t end = 0, depth = 4, consumed = 0;
  bool stop = false;
  std::function<std::vector<Prepared>(int64_t, void *&)> make;
  void start(int64_t first, int64_t last, int n_threads) {
    next = first, end = last, consumed = first;
    for (int t = 0; t < n_threads; t++)
      threads.emplace_back([this] {
        void *vesin = nullptr;
        while (true) {
          int64_t s;
          {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return stop || next.load() < std::min(end, consumed + depth); });
            if (stop || next.load() >= end) break;
            s = next++;
          }
          auto mb = make(s, vesin);
          {
            std::lock_guard<std::mutex> lk(mu);
            ready[s] = std::move(mb);
          }
          cv.notify_all();
        }
        plato::free_vesin(vesin);
      });
  }
  std::vector<Prepared> get(int64_t s) {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&] { return ready.count(s) > 0; });
    auto mb = std::move(ready[s]);
    ready.erase(s);
    consumed = s + 1;
    cv.notify_all();
    return mb;
  }
  void finish() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv.notify_all();
    for (auto &t : threads) t.join();
    threads.clear();
  }
};

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string init = arg(argc, argv, "--init", ""), train_path = arg(argc, argv, "--train", "");
    if (init.empty() || train_path.empty()) {
      std::fprintf(stderr, "usage: mlipcpp-train-plato --init M.gguf --train T.bin [--val V.bin -o OUT.gguf | --check REF]\n");
      return 2;
    }
    const int W = std::stoi(arg(argc, argv, "--gpus", "1"));
    plato::PlatoOptions o;
    o.backend = parse_backend_preference(arg(argc, argv, "--backend", "auto"));
    o.fold_in_graph = true, o.decomposed_layer_norm = true;
    const auto names = param_names(init);
    std::vector<std::unique_ptr<Worker>> workers;
    for (int d = 0; d < W; d++) {
      auto w = std::make_unique<Worker>();
      o.device = d;
      w->init(init, o, names);
      w->fd_eps = std::stod(arg(argc, argv, "--fd-eps", "5e-3"));
      w->energy_weight = std::stod(arg(argc, argv, "--energy-weight", "1"));
      w->force_weight = std::stod(arg(argc, argv, "--force-weight", "1"));
      w->charge_weight = std::stod(arg(argc, argv, "--charge-weight", "0"));
      w->cache_limit = std::stoul(arg(argc, argv, "--graph-cache", "12"));
      workers.push_back(std::move(w));
    }
    Worker &w0 = *workers[0];
    std::printf("backend %s x %d\n", w0.model->backend_name(), W);
    const auto train = read_dataset(train_path);
    std::printf("%zu training structures, %lld parameters\n", train.size(), (long long)w0.n_values);
    const Context cx{w0.model->spec(), w0.model->energy_scale(), w0.model->mesh_spacing(), w0.model->composition(),
                     w0.model->species_index()};

    const std::string check = arg(argc, argv, "--check", "");
    if (!check.empty()) {
      std::ifstream jf(check + ".json");
      const auto ref = nlohmann::json::parse(jf);
      w0.energy_weight = ref["energy_weight"], w0.force_weight = ref["force_weight"];
      std::vector<const Structure *> b;
      for (int i : ref["indices"].get<std::vector<int>>()) b.push_back(&train.at(i));
      void *vesin = nullptr;
      Prepared P = prepare(cx, b, std::vector<Mat3>(b.size(), IDENTITY), vesin);
      plato::free_vesin(vesin);
      P.norm_G = P.Greal, P.norm_N = P.n_real;
      w0.zero_acc();
      const auto r = w0.evaluate(P, true);
      const std::vector<float> acc = w0.get_acc();
      std::vector<double> g(w0.n_values);
      std::ifstream gf(check + ".f64", std::ios::binary);
      gf.read(reinterpret_cast<char *>(g.data()), 8 * g.size());
      const double loss = w0.energy_weight * r.loss_e + w0.force_weight * r.loss_f;
      std::printf("loss  mlip.cpp %.10g (E %.6g F %.6g)  torch %.10g (E %.6g F %.6g)  rel %.2e\n", loss, r.loss_e, r.loss_f,
                  ref["loss"].get<double>(), ref["loss_energy"].get<double>(), ref["loss_forces"].get<double>(),
                  std::abs(loss - ref["loss"].get<double>()) / std::abs(ref["loss"].get<double>()));
      double num = 0, den = 0, dot = 0, na = 0;
      std::vector<std::pair<double, std::string>> worst;
      for (const auto &p : ref["params"]) {
        const std::string name = p["name"];
        const int64_t ro = p["offset"];
        auto it = std::find(w0.names.begin(), w0.names.end(), name);
        if (it == w0.names.end()) throw std::runtime_error("no parameter " + name);
        const int64_t mo = w0.offsets[it - w0.names.begin()], cnt = ggml_nelements(w0.params[it - w0.names.begin()]);
        double n2 = 0, d2 = 0;
        for (int64_t q = 0; q < cnt; q++) {
          const double x = acc[mo + q], y = g[ro + q];
          n2 += (x - y) * (x - y), d2 += y * y, dot += x * y, na += x * x;
        }
        num += n2, den += d2;
        worst.emplace_back(n2, name);
      }
      for (auto &x : worst) x.first = std::sqrt(x.first / den);
      std::sort(worst.rbegin(), worst.rend());
      std::printf("gradient (padded to %d atoms, %lld edges): |g - g_torch| / |g_torch| = %.3e, cosine %.8f, %zu tensors\n", P.n,
                  (long long)P.E, std::sqrt(num / den), dot / std::sqrt(na * den), worst.size());
      for (size_t k = 0; k < std::min<size_t>(4, worst.size()); k++)
        std::printf("  largest share of the error %-36s %.3e\n", worst[k].second.c_str(), worst[k].first);
      return 0;
    }

    const std::string gradcheck = arg(argc, argv, "--gradcheck", "");
    if (!gradcheck.empty()) { // directional derivative of the loss along the trainer's gradient vs |g|
      const double h = std::stod(gradcheck);
      const int nb = std::min<int>(std::stoi(arg(argc, argv, "--batch-size", "8")), int(train.size()));
      std::vector<const Structure *> b; // half of them with charged molecules, when there are any
      for (const auto &s : train)
        if (int(b.size()) < nb / 2 && s.has_molecules && (s.QA != 0 || s.QB != 0)) b.push_back(&s);
      for (const auto &s : train)
        if (int(b.size()) < nb && std::find(b.begin(), b.end(), &s) == b.end()) b.push_back(&s);
      void *vesin = nullptr;
      Prepared P = prepare(cx, b, std::vector<Mat3>(b.size(), IDENTITY), vesin);
      plato::free_vesin(vesin);
      P.norm_G = P.Greal, P.norm_N = P.n_real;
      auto loss = [&](const Worker::Result &r) { return w0.energy_weight * r.loss_e + w0.force_weight * r.loss_f + w0.charge_weight * r.loss_q; };
      w0.zero_acc();
      const auto r0 = w0.evaluate(P, true);
      const std::vector<float> g = w0.get_acc(), p0 = w0.get_params();
      double gn = 0;
      for (float x : g) gn += double(x) * x;
      gn = std::sqrt(gn);
      double L[2];
      for (int k = 0; k < 2; k++) {
        std::vector<float> p = p0;
        for (size_t i = 0; i < p.size(); i++) p[i] += float((k ? -h : h) * g[i] / gn);
        w0.set_params(p);
        L[k] = loss(w0.evaluate(P, false));
      }
      w0.set_params(p0);
      const double dd = (L[0] - L[1]) / (2 * h);
      std::printf("gradcheck (%d structures, weights E %g F %g Q %g): loss %.8g (E %.4g F %.4g Q %.4g), |g| %.6e, "
                  "dL/dh along g %.6e, rel %.2e; molecule charge MAE %.4f e over %lld molecules\n",
                  nb, w0.energy_weight, w0.force_weight, w0.charge_weight, loss(r0), r0.loss_e, r0.loss_f, r0.loss_q, gn, dd,
                  std::abs(dd - gn) / gn, r0.q_abs / std::max<int64_t>(r0.q_count, 1), (long long)r0.q_count);
      return 0;
    }
    const auto val = read_dataset(arg(argc, argv, "--val", ""));
    const std::string output = arg(argc, argv, "-o", "plato-trained.gguf");
    const int epochs = std::stoi(arg(argc, argv, "--epochs", "3")), B = std::stoi(arg(argc, argv, "--batch-size", "16"));
    const double lr0 = std::stod(arg(argc, argv, "--lr", "1e-4")), warm = std::stod(arg(argc, argv, "--warmup", "0.01"));
    const double clip = std::stod(arg(argc, argv, "--grad-clip", "1.0")), min_lr = std::stod(arg(argc, argv, "--min-lr", "1e-6"));
    const int decay_after = std::stoi(arg(argc, argv, "--decay-after", "10"));
    const std::string schedule = arg(argc, argv, "--schedule", "cosine");
    const int64_t max_steps = std::stoll(arg(argc, argv, "--max-steps", "-1"));
    const int save_every = std::stoi(arg(argc, argv, "--save-every", "100")), val_every = std::stoi(arg(argc, argv, "--val-every", "1"));
    const int state_every = std::stoi(arg(argc, argv, "--state-every", "10"));
    const int max_epochs_run = std::stoi(arg(argc, argv, "--max-epochs-this-run", "1000000"));
    const double max_hours = std::stod(arg(argc, argv, "--max-hours", "1e9"));
    const bool augment = !flag(argc, argv, "--no-augment"), no_val = flag(argc, argv, "--no-val");
    const std::string log_path = arg(argc, argv, "--log", output + ".log.jsonl"), state_path = arg(argc, argv, "--state", output + ".state");
    const int prep_threads = std::stoi(arg(argc, argv, "--prep-threads", "4"));
    const int64_t spe = (int64_t(train.size()) + B - 1) / B, total = int64_t(epochs) * spe;
    auto lr_at = [&](int64_t step) {
      if (schedule == "linear") { // LOREM: constant for decay_after epochs, then linear to min_lr at the end
        const int64_t s0 = int64_t(decay_after) * spe;
        if (step < s0) return lr0;
        const double p = double(step - s0) / double(std::max<int64_t>(1, total - s0));
        return lr0 + (min_lr - lr0) * std::min(1.0, p);
      }
      const int64_t warm_steps = int64_t(warm * total);
      if (step < warm_steps) return lr0 * double(step) / double(std::max<int64_t>(1, warm_steps));
      const double p = double(step - warm_steps) / double(std::max<int64_t>(1, total - warm_steps));
      return lr0 * 0.5 * (1 + std::cos(M_PI * p));
    };

    State st;
    std::mt19937_64 rng(std::stoull(arg(argc, argv, "--seed", "0")));
    const std::string resume = arg(argc, argv, "--resume", "");
    if (!resume.empty()) {
      st = load_state(resume);
      std::istringstream(st.rng) >> rng;
      for (auto &w : workers) w->set_moments(st.m, false), w->set_moments(st.v, true);
      std::printf("resumed at epoch %d, step %lld\n", st.epoch, (long long)st.step);
    }
#ifdef MLIPCPP_NCCL
    std::unique_ptr<Nccl> nccl;
    if (W > 1) nccl = std::make_unique<Nccl>(W);
#else
    if (W > 1) throw std::runtime_error("--gpus > 1 needs a build with NCCL");
#endif
    if (W > 1) { // the same parameters everywhere
      const auto p = w0.get_params();
      for (int d = 1; d < W; d++) workers[d]->set_params(p);
    }

    // E MAE, F MAE, loss, R^2 of E and F in % (LOREM's checkpoint metric is their sum)
    auto validate = [&]() {
      double e = 0, f = 0, loss = 0, e_sq = 0, f_sq = 0, f_sum = 0, f_sum2 = 0, q_abs = 0;
      int64_t fc = 0, ns = 0, qn = 0, q_ok = 0, q_ch = 0;
      std::vector<double> et;
      void *vesin = nullptr;
      for (size_t s = 0; s < val.size(); s += B) {
        std::vector<const Structure *> b;
        for (size_t k = s; k < std::min(val.size(), s + B); k++) b.push_back(&val[k]);
        Prepared P = prepare(cx, b, std::vector<Mat3>(b.size(), IDENTITY), vesin);
        P.norm_G = P.Greal, P.norm_N = P.n_real;
        auto r = w0.evaluate(P, false);
        e += r.e_abs, f += r.f_abs, fc += r.f_count, ns += P.Greal;
        loss += (w0.energy_weight * r.loss_e + w0.force_weight * r.loss_f + w0.charge_weight * r.loss_q) * P.Greal;
        q_abs += r.q_abs, qn += r.q_count, q_ok += r.q_sign_ok, q_ch += r.q_charged;
        e_sq += r.e_sq, f_sq += r.f_sq, f_sum += r.f_sum, f_sum2 += r.f_sum2;
        et.insert(et.end(), r.e_target.begin(), r.e_target.end());
      }
      plato::free_vesin(vesin);
      const double em = std::accumulate(et.begin(), et.end(), 0.0) / std::max<size_t>(et.size(), 1);
      double e_sst = 0;
      for (double y : et) e_sst += (y - em) * (y - em);
      const double f_sst = f_sum2 - f_sum * f_sum / std::max<int64_t>(fc, 1);
      // E MAE, F MAE, loss, R2 E, R2 F, molecule charge MAE (e), charged molecules with the formal sign (%), molecules
      return std::array<double, 8>{e / ns, f / std::max<int64_t>(fc, 1), loss / ns, 100 * (1 - e_sq / e_sst), 100 * (1 - f_sq / f_sst),
                                   q_abs / std::max<int64_t>(qn, 1), 100.0 * q_ok / std::max<int64_t>(q_ch, 1), double(qn)};
    };
    auto log_line = [&](const nlohmann::json &j) {
      std::ofstream(log_path, std::ios::app) << j.dump() << "\n";
    };
    if (st.epoch == 0 && !no_val) {
      const auto m = validate();
      std::printf("epoch 0 val: E MAE %.4f meV/atom, F MAE %.4f meV/A, loss %.6g, R2 E %.4f%% F %.4f%%\n", 1e3 * m[0], 1e3 * m[1],
                  m[2], m[3], m[4]);
      nlohmann::json l0 = {{"epoch", 0}, {"val_e_mae_mev_atom", 1e3 * m[0]}, {"val_f_mae_mev_a", 1e3 * m[1]}, {"val_loss", m[2]},
                           {"val_r2_e", m[3]}, {"val_r2_f", m[4]}};
      if (m[7] > 0) l0.update({{"val_q_mae_e", m[5]}, {"val_q_sign_pct", m[6]}});
      log_line(l0);
      std::fflush(stdout);
    }
    const auto run_start = clk::now();
    int epochs_this_run = 0;
    std::string stop_reason = "done";
    for (int ep = st.epoch + 1; ep <= epochs; ep++) {
      // the epoch's order and rotations, drawn up front (reproducible on resume)
      std::vector<size_t> order(train.size());
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), rng);
      std::vector<Mat3> rots(train.size(), IDENTITY);
      if (augment)
        for (auto &R : rots) R = random_o3(rng);
      Prefetcher pf;
      pf.make = [&](int64_t s, void *&vesin) {
        const size_t lo = size_t(s) * B, hi = std::min(order.size(), lo + B);
        const size_t per = (hi - lo + W - 1) / W;
        std::vector<Prepared> mb;
        double norm_G = double(hi - lo), norm_N = 0;
        for (size_t k = lo; k < hi; k++) norm_N += train[order[k]].n;
        for (int d = 0; d < W; d++) {
          const size_t a = std::min(hi, lo + d * per), b2 = std::min(hi, a + per);
          if (a >= b2) continue;
          std::vector<const Structure *> b;
          std::vector<Mat3> r;
          for (size_t k = a; k < b2; k++) b.push_back(&train[order[k]]), r.push_back(rots[order[k]]);
          Prepared P = prepare(cx, b, r, vesin);
          P.norm_G = norm_G, P.norm_N = norm_N;
          mb.push_back(std::move(P));
        }
        return mb;
      };
      pf.start(0, spe, prep_threads);
      const auto t0 = clk::now();
      double le = 0, lf = 0, lq = 0, gn = 0, wait_ms = 0;
      int64_t nb = 0;
      for (auto &w : workers) w->t_build = w->t_compute = w->t_host = 0, w->builds = 0;
      for (int64_t s = 0; s < spe; s++) {
        const auto tw = clk::now();
        auto mb = pf.get(s);
        wait_ms += ms_since(tw);
        std::vector<std::future<Worker::Result>> fut;
        for (size_t d = 0; d < mb.size(); d++) {
          workers[d]->zero_acc();
          fut.push_back(std::async(std::launch::async, [&, d] { return workers[d]->evaluate(mb[d], true); }));
        }
        for (size_t d = mb.size(); d < workers.size(); d++) workers[d]->zero_acc();
        for (auto &f : fut) {
          auto r = f.get();
          le += r.loss_e, lf += r.loss_f, lq += r.loss_q;
        }
#ifdef MLIPCPP_NCCL
        if (nccl) nccl->allreduce_sum(workers);
#endif
        st.step++;
        std::vector<std::future<double>> sf;
        for (auto &w : workers) sf.push_back(std::async(std::launch::async, [&, wp = w.get()] { return wp->step(lr_at(st.step - 1), st.step, clip); }));
        for (auto &f : sf) gn = f.get();
        nb++;
        if (st.step % 100 == 0) {
          std::printf("  step %lld loss %.6g (E %.4g F %.4g) |g| %.3g lr %.3g\n", (long long)st.step,
                      (w0.energy_weight * le + w0.force_weight * lf + w0.charge_weight * lq) / nb, le / nb, lf / nb, gn, lr_at(st.step));
          std::fflush(stdout);
        }
        if (max_steps > 0 && st.step >= max_steps) break;
      }
      pf.finish();
      const double wall = std::chrono::duration<double>(clk::now() - t0).count();
      st.epoch = ep, epochs_this_run++;
      std::array<double, 8> m = {0, 0, 0, 0, 0, 0, 0, 0};
      const bool do_val = !no_val && (ep % val_every == 0 || ep == epochs);
      const auto tv = clk::now();
      if (do_val) m = validate();
      const double val_s = std::chrono::duration<double>(clk::now() - tv).count();
      std::printf("epoch %d: train loss %.6g (E %.4g F %.4g), %.1f s (host wait %.1f, build %.1f, inputs %.1f, compute %.1f s; %lld graphs built), lr %.3g",
                  ep, (w0.energy_weight * le + w0.force_weight * lf + w0.charge_weight * lq) / nb, le / nb, lf / nb, wall, wait_ms / 1e3, w0.t_build / 1e3,
                  w0.t_host / 1e3, w0.t_compute / 1e3, (long long)w0.builds, lr_at(st.step));
      if (do_val)
        std::printf(", val E MAE %.4f meV/atom, F MAE %.4f meV/A, loss %.6g, R2 E %.4f%% F %.4f%% (%.1f s)", 1e3 * m[0], 1e3 * m[1],
                    m[2], m[3], m[4], val_s);
      if (do_val && m[7] > 0) std::printf(", molecule charge MAE %.4f e, formal sign %.1f%%", m[5], m[6]);
      std::printf("\n");
      std::fflush(stdout);
      nlohmann::json line = {{"epoch", ep}, {"step", st.step}, {"train_loss", (w0.energy_weight * le + w0.force_weight * lf + w0.charge_weight * lq) / nb},
                             {"train_loss_e", le / nb}, {"train_loss_f", lf / nb}, {"train_loss_q", lq / nb}, {"wall_s", wall}, {"lr", lr_at(st.step)}};
      // the best checkpoint by LOREM's criterion: highest R^2(E) + R^2(F) on the validation set
      const bool best = do_val && -(m[3] + m[4]) < st.best;
      if (do_val) line.update({{"val_e_mae_mev_atom", 1e3 * m[0]}, {"val_f_mae_mev_a", 1e3 * m[1]}, {"val_loss", m[2]},
                               {"val_r2_e", m[3]}, {"val_r2_f", m[4]}, {"best", best}});
      if (do_val && m[7] > 0) line.update({{"val_q_mae_e", m[5]}, {"val_q_sign_pct", m[6]}});
      log_line(line);
      const bool last = ep == epochs || epochs_this_run >= max_epochs_run || (max_steps > 0 && st.step >= max_steps) ||
                        std::chrono::duration<double>(clk::now() - run_start).count() / 3600.0 > max_hours;
      const nlohmann::json info = {{"epoch", ep}, {"step", st.step}, {"init", init}, {"lr", lr0}, {"schedule", schedule},
                                   {"epochs", epochs}, {"batch_size", B}, {"fd_eps", w0.fd_eps}, {"gpus", W},
                                   {"charge_weight", w0.charge_weight}};
      if (best) {
        st.best = -(m[3] + m[4]);
        nlohmann::json bi = info;
        bi.update({{"val_r2_e", m[3]}, {"val_r2_f", m[4]}});
        save_gguf(init, output + ".best.gguf", w0, w0.get_params(), bi);
      }
      if (ep % save_every == 0 || ep % state_every == 0 || last || best) {
        const auto p = w0.get_params();
        save_gguf(init, output, w0, p, info);
        if (ep % save_every == 0) {
          char tag[32];
          std::snprintf(tag, sizeof(tag), ".epoch%05d.gguf", ep);
          save_gguf(init, output + tag, w0, p, info);
        }
        std::ostringstream rs;
        rs << rng;
        st.rng = rs.str(), st.m = w0.get_moments(false), st.v = w0.get_moments(true);
        save_state(state_path, st);
      }
      if (last) {
        if (ep < epochs) stop_reason = "paused";
        break;
      }
    }
    std::printf("%s at epoch %d: wrote %s (state %s)\n", stop_reason.c_str(), st.epoch, output.c_str(), state_path.c_str());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
