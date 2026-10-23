// Training Plato-PET-LR (plato-pet-lr GGUFs) with conservative forces.
//
//   mlipcpp-train-plato --init model.gguf --train train.bin --val val.bin -o out.gguf
//       [--epochs 3] [--batch-size 16] [--lr 1e-4] [--warmup 0.01] [--energy-weight 1]
//       [--force-weight 1] [--seed 0] [--no-augment] [--fd-eps 1e-3] [--backend cuda]
//   mlipcpp-train-plato --init model.gguf --train data.bin --check REF   (REF.json / REF.f64 of
//       scripts/plato/plato_grad_reference.py: loss and parameter gradients vs PyTorch)
//
// The loss is metatrain's (plato_pet_lr trainer): MSE of the energies per atom and of
// the position gradients, targets with the composition removed and divided by the
// energy scale. The force term's parameter gradient is the derivative of the energy
// along the force residual u = dL/d(dE/dx): with T(+-) = sum_b (1 +- eps a_b) E_b(x +- eps u),
// a_b = dL/dE_b, the central difference (dT(+)/dtheta - dT(-)/dtheta) / (2 eps) is
// dL/dtheta up to O(eps^2). Three graph evaluations per batch (the unperturbed one for
// the forces); every structure of a batch in one graph (packed tokens, the mesh and the
// Ewald sum batched over structures). Adam, gradient-norm clipping and the warmup +
// cosine schedule of metatrain. Data: scripts/plato/plato_dataset.py.
#include "models/plato/plato.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlipcpp;
using runtime::PlatoInputs;

namespace {

struct Structure {
  int n = 0;
  std::vector<int32_t> Z;
  std::vector<double> x, F; // [n][3]
  double cell[9] = {}, energy = 0, stress[9] = {};
  bool pbc = true, has_forces = false, has_stress = false;
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
    s.pbc = fl[0], s.has_forces = fl[1], s.has_stress = fl[2];
    f.read(reinterpret_cast<char *>(&s.energy), 8);
    if (s.has_forces) s.F.resize(3 * s.n), f.read(reinterpret_cast<char *>(s.F.data()), 8 * 3 * s.n);
    if (s.has_stress) f.read(reinterpret_cast<char *>(s.stress), 72);
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

// random O(3): Haar rotation (normalised quaternion) times +-1, as metatrain's O3Augmenter
void random_o3(std::mt19937_64 &rng, double R[9]) {
  std::normal_distribution<double> N(0, 1);
  double q[4], s = 0;
  for (double &v : q) v = N(rng), s += v * v;
  s = std::sqrt(s);
  const double w = q[0] / s, x = q[1] / s, y = q[2] / s, z = q[3] / s;
  const double m[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
                       2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                       2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
  const double sg = std::uniform_int_distribution<int>(0, 1)(rng) ? -1.0 : 1.0;
  for (int k = 0; k < 9; k++) R[k] = sg * m[k];
}

void inverse3(const double *A, double *I, double &det) {
  det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  const double id = 1.0 / det;
  I[0] = (A[4] * A[8] - A[5] * A[7]) * id, I[1] = (A[2] * A[7] - A[1] * A[8]) * id, I[2] = (A[1] * A[5] - A[2] * A[4]) * id;
  I[3] = (A[5] * A[6] - A[3] * A[8]) * id, I[4] = (A[0] * A[8] - A[2] * A[6]) * id, I[5] = (A[2] * A[3] - A[0] * A[5]) * id;
  I[6] = (A[3] * A[7] - A[4] * A[6]) * id, I[7] = (A[1] * A[6] - A[0] * A[7]) * id, I[8] = (A[0] * A[4] - A[1] * A[3]) * id;
}

struct Trainer {
  plato::PlatoModel &model;
  const runtime::PlatoSpec &spec;
  std::vector<std::string> names;      // trainable, in GGUF train.param_names order
  std::vector<ggml_tensor *> params;
  std::vector<int64_t> offsets;
  int64_t n_values = 0;
  ggml_gallocr_t galloc = nullptr;
  void *vesin = nullptr;
  double fd_eps = 1e-3, energy_weight = 1, force_weight = 1;
  bool plain = false; // the gradient of sum_b E_b (checks)
  double t_build = 0, t_compute = 0, t_host = 0; // ms, accumulated

  Trainer(plato::PlatoModel &m, const std::vector<std::string> &param_names) : model(m), spec(m.spec()) {
    for (const auto &n : param_names) {
      auto it = m.weights().find(n);
      if (it == m.weights().end()) throw std::runtime_error("no weight " + n);
      params.push_back(it->second), names.push_back(n), offsets.push_back(n_values);
      n_values += ggml_nelements(it->second);
      ggml_set_param(it->second);
    }
  }
  ~Trainer() {
    if (galloc) ggml_gallocr_free(galloc);
    plato::free_vesin(vesin);
  }

  std::vector<float> get_params() const {
    std::vector<float> v(n_values);
    for (size_t k = 0; k < params.size(); k++) ggml_backend_tensor_get(params[k], v.data() + offsets[k], 0, ggml_nbytes(params[k]));
    return v;
  }
  void set_params(const std::vector<float> &v) {
    for (size_t k = 0; k < params.size(); k++) ggml_backend_tensor_set(params[k], v.data() + offsets[k], 0, ggml_nbytes(params[k]));
  }

  struct Result {
    double loss = 0, loss_e = 0, loss_f = 0;
    std::vector<double> grad;          // dL/dtheta
    std::vector<double> E;             // physical energies (eV)
    std::vector<std::vector<double>> F; // physical forces
  };

  // one batch: loss, and with `train` its parameter gradient
  Result evaluate(const std::vector<const Structure *> &batch, const std::vector<std::array<double, 9>> &rot, bool train) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto t0 = clk::now();
    const int G = static_cast<int>(batch.size());
    const double scale = model.energy_scale();
    // structures, rotated
    std::vector<std::vector<float>> X(G);
    std::vector<std::array<double, 9>> cellR(G);
    std::vector<plato::Edges> ed(G);
    std::vector<int> a0(G + 1, 0);
    int max_len = 1;
    int mesh[3] = {0, 0, 0};
    for (int g = 0; g < G; g++) {
      const Structure &s = *batch[g];
      if (!s.pbc) throw std::runtime_error("training batches of periodic structures only");
      const double *R = rot[g].data();
      X[g].resize(3 * s.n);
      for (int i = 0; i < s.n; i++)
        for (int c = 0; c < 3; c++) X[g][3 * i + c] = float(R[3 * c] * s.x[3 * i] + R[3 * c + 1] * s.x[3 * i + 1] + R[3 * c + 2] * s.x[3 * i + 2]);
      float L[3][3];
      for (int a = 0; a < 3; a++)
        for (int c = 0; c < 3; c++) {
          const double v = R[3 * c] * s.cell[3 * a] + R[3 * c + 1] * s.cell[3 * a + 1] + R[3 * c + 2] * s.cell[3 * a + 2];
          L[a][c] = float(v), cellR[g][3 * a + c] = double(L[a][c]);
        }
      // the mesh from the unrotated cell (a rotation must not change it; see docs/plato.md)
      for (int a = 0; a < 3; a++) {
        const float row[3] = {float(s.cell[3 * a]), float(s.cell[3 * a + 1]), float(s.cell[3 * a + 2])};
        const int m = plato::mesh_points(row, model.mesh_spacing());
        if (g == 0) mesh[a] = m;
        else if (m != mesh[a]) throw std::runtime_error("a batch needs one mesh size");
      }
      Cell cell(L, true, true, true);
      AtomicSystem sys(s.n, X[g].data(), s.Z.data(), &cell);
      ed[g] = plato::neighbor_edges(sys, spec.cutoff, vesin);
      for (int i = 0; i < s.n; i++) max_len = std::max<int>(max_len, 1 + int(ed[g].start[i + 1] - ed[g].start[i]));
      a0[g + 1] = a0[g] + s.n;
    }
    const int n = a0[G];
    int64_t E = 0;
    std::vector<int64_t> e0(G + 1, 0);
    for (int g = 0; g < G; g++) e0[g + 1] = e0[g] + int64_t(ed[g].j.size());
    E = e0[G];
    // the k sets: of the unrotated cell (the same integer set for every structure; checked)
    const double kc = spec.les_kcut / spec.les_sigma, wd = 1.0 / spec.les_sigma;
    std::vector<int> ns;
    std::vector<float> les_taper;
    {
      const Structure &s = *batch[0];
      double Ci[9], det;
      inverse3(s.cell, Ci, det);
      int nmax[3];
      for (int a = 0; a < 3; a++)
        nmax[a] = int(std::floor(kc * std::sqrt(s.cell[3 * a] * s.cell[3 * a] + s.cell[3 * a + 1] * s.cell[3 * a + 1] + s.cell[3 * a + 2] * s.cell[3 * a + 2]) / (2 * M_PI)));
      for (int a = -nmax[0]; a <= nmax[0]; a++)
        for (int b = -nmax[1]; b <= nmax[1]; b++)
          for (int c = -nmax[2]; c <= nmax[2]; c++) {
            if (!(a > 0 || (a == 0 && b > 0) || (a == 0 && b == 0 && c > 0))) continue;
            double k2 = 0;
            for (int q = 0; q < 3; q++) {
              const double k = 2 * M_PI * (a * Ci[3 * q] + b * Ci[3 * q + 1] + c * Ci[3 * q + 2]);
              k2 += k * k;
            }
            if (!(k2 < kc * kc)) continue;
            ns.push_back(a), ns.push_back(b), ns.push_back(c);
            les_taper.push_back(std::sqrt(k2) >= kc - wd ? 1.0f : 0.0f);
          }
      for (int g = 1; g < G; g++)
        if (std::memcmp(batch[g]->cell, s.cell, sizeof(s.cell)) != 0) throw std::runtime_error("a batch needs one cell shape");
    }
    const int64_t nk = int64_t(les_taper.size());
    const int M0 = mesh[0], M1 = mesh[1], M2 = mesh[2], Mh2 = M2 / 2 + 1;
    const int64_t Mh = int64_t(M0) * M1 * Mh2;

    // graph
    const size_t GS = 65536;
    ggml_context *ctx = ggml_init({ggml_tensor_overhead() * GS * 2 + ggml_graph_overhead_custom(GS, true), nullptr, true});
    std::vector<ggml_tensor *> inputs;
    auto input = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, const char *name) {
      ggml_tensor *t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
      ggml_set_name(t, name), ggml_set_input(t), inputs.push_back(t);
      return t;
    };
    const int64_t T = n + E, hdF = int64_t(spec.head_dim() / 2) * spec.frames;
    PlatoInputs in;
    in.n_struct = G;
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
    in.positions = input(GGML_TYPE_F32, 3, n, 1, "positions");
    in.strain = input(GGML_TYPE_F32, 3, 3, G, "strain");
    in.atom_inv_cell_t = input(GGML_TYPE_F32, 3, 3, n, "atom_inv_cell_t");
    in.inv_volume = input(GGML_TYPE_F32, 1, G, 1, "inv_volume");
    in.onehot = input(GGML_TYPE_F32, n, G, 1, "onehot");
    in.onehot_t = input(GGML_TYPE_F32, G, n, 1, "onehot_t");
    in.inv_count = input(GGML_TYPE_F32, 1, G, 1, "inv_count");
    for (int a = 0; a < 3; a++) in.mesh_size[a] = mesh[a];
    if (spec.mesh) {
      in.mesh_k = input(GGML_TYPE_F32, 3, Mh, G, "mesh_k");
      in.mesh_k0_pad = input(GGML_TYPE_F32, 1, Mh, 1, "mesh_k0_pad");
      in.mesh_deconv = input(GGML_TYPE_F32, 1, Mh, 1, "mesh_deconv");
    }
    if (spec.les) {
      in.les_k = input(GGML_TYPE_F32, 3, nk, G, "les_k");
      in.les_n = input(GGML_TYPE_F32, 3, nk, 1, "les_n");
      in.les_taper = input(GGML_TYPE_F32, 1, nk, 1, "les_taper");
    }
    ggml_set_param(in.edge_vectors), ggml_set_param(in.positions), ggml_set_param(in.strain);
    runtime::PlatoOutputs out = runtime::build_plato(ctx, spec, model.weights(), in, n, E, max_len, true);
    ggml_set_output(out.energies);
    ggml_cgraph *graph = ggml_new_graph_custom(ctx, GS, true);
    ggml_build_forward_expand(graph, out.energies);
    ggml_build_forward_expand(graph, out.total);
    ggml_set_loss(out.total);
    ggml_build_backward_expand(ctx, graph, nullptr);
    ggml_tensor *g_ev = ggml_graph_get_grad(graph, in.edge_vectors), *g_pos = ggml_graph_get_grad(graph, in.positions);
    ggml_tensor *loss_grad = ggml_graph_get_grad_acc(graph, out.total);
    ggml_set_input(loss_grad), inputs.push_back(loss_grad);
    std::vector<ggml_tensor *> grads(params.size(), nullptr);
    for (size_t k = 0; k < params.size(); k++) {
      grads[k] = ggml_graph_get_grad(graph, params[k]);
      // a gradient that is a view (a RESHAPE of an OUT_PROD, ...): its source must stay
      // allocated too, the allocator frees it otherwise
      for (ggml_tensor *t = grads[k]; t; t = t->view_src) ggml_set_output(t);
    }
    for (ggml_tensor *t = g_ev; t; t = t->view_src) ggml_set_output(t);
    for (ggml_tensor *t = g_pos; t; t = t->view_src) ggml_set_output(t);
    ggml_backend_buffer_type_t buft = model.backend()->buffer_type();
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t size = 0;
    std::vector<size_t> offs;
    for (ggml_tensor *t : inputs) offs.push_back(size), size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
    ggml_backend_buffer_t ibuf = ggml_backend_buft_alloc_buffer(buft, size);
    char *base = static_cast<char *>(ggml_backend_buffer_get_base(ibuf));
    for (size_t k = 0; k < inputs.size(); k++) ggml_backend_tensor_alloc(ibuf, inputs[k], base + offs[k]);
    if (!galloc) galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, graph)) throw std::runtime_error("cannot allocate the training graph");
    const auto t1 = clk::now();

    // fixed inputs
    auto set = [](ggml_tensor *t, const void *d) { ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t)); };
    std::vector<int32_t> species(n), sidx(n), ecs(E), nbs(E), cen(E), rev(T), off(n + 1);
    std::vector<float> ev0(3 * E), pos0(3 * size_t(n)), aict(9 * size_t(n)), onehot(size_t(n) * G, 0.0f), onehot_t(size_t(G) * n, 0.0f);
    std::vector<float> invv(G), invc(G), mk, lk, ln(3 * nk);
    for (int g = 0; g < G; g++) {
      const Structure &s = *batch[g];
      double Ci[9], det;
      inverse3(cellR[g].data(), Ci, det);
      invv[g] = float(1.0 / std::abs(det)), invc[g] = 1.0f / float(s.n);
      for (int i = 0; i < s.n; i++) {
        const int A = a0[g] + i;
        auto it = model.species_index().find(s.Z[i]);
        if (it == model.species_index().end()) throw std::runtime_error("unknown species");
        species[A] = it->second, sidx[A] = g, rev[A] = A;
        onehot[size_t(g) * n + A] = 1.0f, onehot_t[size_t(A) * G + g] = 1.0f;
        for (int c = 0; c < 3; c++) pos0[3 * A + c] = X[g][3 * i + c];
        for (int j = 0; j < 3; j++)
          for (int k = 0; k < 3; k++) aict[9 * A + j * 3 + k] = float(Ci[k * 3 + j]);
        off[A] = int32_t(e0[g] + ed[g].start[i]);
      }
      for (int i = 0; i < s.n; i++)
        for (int64_t e = ed[g].start[i]; e < ed[g].start[i + 1]; e++) {
          const int64_t ge = e0[g] + e;
          ecs[ge] = species[a0[g] + i], cen[ge] = a0[g] + i, nbs[ge] = species[a0[g] + ed[g].j[e]];
          rev[n + ge] = int32_t(n + e0[g] + ed[g].rev[e]);
          for (int c = 0; c < 3; c++) ev0[3 * ge + c] = ed[g].v[3 * e + c];
        }
      if (spec.mesh) {
        auto freq = [](int i, int M) { return i <= (M - 1) / 2 ? i : i - M; };
        for (int i0 = 0; i0 < M0; i0++)
          for (int i1 = 0; i1 < M1; i1++)
            for (int i2 = 0; i2 < Mh2; i2++) {
              const double m[3] = {double(freq(i0, M0)), double(freq(i1, M1)), double(i2)};
              for (int b = 0; b < 3; b++) mk.push_back(float(2 * M_PI * (m[0] * Ci[3 * b] + m[1] * Ci[3 * b + 1] + m[2] * Ci[3 * b + 2])));
            }
      }
      if (spec.les)
        for (int64_t q = 0; q < nk; q++)
          for (int b = 0; b < 3; b++)
            lk.push_back(float(2 * M_PI * (ns[3 * q] * Ci[3 * b] + ns[3 * q + 1] * Ci[3 * b + 1] + ns[3 * q + 2] * Ci[3 * b + 2])));
    }
    off[n] = int32_t(E);
    for (int64_t q = 0; q < nk; q++)
      for (int b = 0; b < 3; b++) ln[3 * q + b] = float(2 * M_PI * ns[3 * q + b]);
    set(in.species, species.data()), set(in.struct_index, sidx.data()), set(in.edge_center_species, ecs.data());
    set(in.neighbor_species, nbs.data()), set(in.centers, cen.data()), set(in.reverse, rev.data()), set(in.offsets, off.data());
    std::vector<float> zeros(ggml_nelements(in.zero_ang), 0.0f);
    set(in.zero_ang, zeros.data());
    {
      std::vector<float> rbf0(spec.n_rbf);
      const double w = spec.cutoff / spec.n_rbf;
      for (int k = 0; k < spec.n_rbf; k++) {
        const double mu = spec.cutoff * double(float(k) / float(spec.n_rbf - 1));
        rbf0[k] = float(std::exp(-std::pow((1e-6 - mu) / w, 2)));
      }
      set(in.rbf0, rbf0.data());
      const float I9[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      set(in.rel_identity, I9);
    }
    set(in.atom_inv_cell_t, aict.data()), set(in.inv_volume, invv.data()), set(in.onehot, onehot.data());
    set(in.onehot_t, onehot_t.data()), set(in.inv_count, invc.data());
    if (spec.mesh) {
      std::vector<float> pad(Mh, 0.0f), dec(Mh);
      auto freq = [](int i, int M) { return i <= (M - 1) / 2 ? i : i - M; };
      for (int i0 = 0; i0 < M0; i0++)
        for (int i1 = 0; i1 < M1; i1++)
          for (int i2 = 0; i2 < Mh2; i2++) {
            const int64_t e = (int64_t(i0) * M1 + i1) * Mh2 + i2;
            const int m[3] = {freq(i0, M0), freq(i1, M1), i2};
            double dc = 1;
            for (int a = 0; a < 3; a++) {
              const double f = (4 + 2 * std::cos(2 * M_PI * m[a] / mesh[a])) / 6;
              dc *= f * f;
            }
            const bool zero = e == 0;
            pad[e] = zero ? 1.0f : 0.0f, dec[e] = zero ? 0.0f : float(1.0 / dc);
          }
      set(in.mesh_k, mk.data()), set(in.mesh_k0_pad, pad.data()), set(in.mesh_deconv, dec.data());
    }
    if (spec.les) set(in.les_k, lk.data()), set(in.les_n, ln.data()), set(in.les_taper, les_taper.data());
    const float one = 1.0f;
    set(loss_grad, &one);
    std::vector<float> zstrain(9 * size_t(G), 0.0f);
    set(in.strain, zstrain.data());

    // targets (scaled, composition removed) in the rotated frame
    std::vector<double> Et(G), Ft(3 * size_t(n));
    for (int g = 0; g < G; g++) {
      const Structure &s = *batch[g];
      double c = 0;
      for (int z : s.Z)
        if (auto it = model.composition().find(z); it != model.composition().end()) c += it->second;
      Et[g] = (s.energy - c) / scale;
      const double *R = rot[g].data();
      for (int i = 0; i < s.n; i++)
        for (int a = 0; a < 3; a++)
          Ft[3 * (a0[g] + i) + a] = (R[3 * a] * s.F[3 * i] + R[3 * a + 1] * s.F[3 * i + 1] + R[3 * a + 2] * s.F[3 * i + 2]) / scale;
    }

    auto run = [&](const std::vector<float> &ev, const std::vector<float> &pos, const std::vector<float> &w) {
      set(in.edge_vectors, ev.data()), set(in.positions, pos.data()), set(in.energy_weights, w.data());
      const auto a = clk::now();
      if (model.backend()->compute(graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
      t_compute += ms(a, clk::now());
    };
    std::vector<float> w1(G, 1.0f);
    run(ev0, pos0, w1);
    std::vector<float> Eraw(G), gev(3 * E), gpos(3 * size_t(n));
    ggml_backend_tensor_get(out.energies, Eraw.data(), 0, 4 * G);
    if (g_ev) ggml_backend_tensor_get(g_ev, gev.data(), 0, gev.size() * 4);
    if (g_pos) ggml_backend_tensor_get(g_pos, gpos.data(), 0, gpos.size() * 4);
    std::vector<double> dEdx(3 * size_t(n), 0.0);
    for (int A = 0; A < n; A++)
      for (int c = 0; c < 3; c++) dEdx[3 * A + c] = gpos[3 * A + c];
    for (int64_t e = 0; e < E; e++) {
      const int i = cen[e];
      const int g = sidx[i];
      const int jl = ed[g].j[e - e0[g]], J = a0[g] + jl;
      for (int c = 0; c < 3; c++) dEdx[3 * i + c] -= gev[3 * e + c], dEdx[3 * J + c] += gev[3 * e + c];
    }
    Result r;
    std::vector<double> a(G), u(3 * size_t(n));
    for (int g = 0; g < G; g++) {
      const double nb = batch[g]->n, de = (Eraw[g] - Et[g]) / nb;
      r.loss_e += de * de / G;
      a[g] = energy_weight * 2 * de / (G * nb);
    }
    for (int A = 0; A < n; A++)
      for (int c = 0; c < 3; c++) {
        const double d = dEdx[3 * A + c] + Ft[3 * A + c]; // dE/dx - (-F / scale)
        r.loss_f += d * d / (3.0 * n);
        u[3 * A + c] = force_weight * 2 * d / (3.0 * n);
      }
    r.loss = energy_weight * r.loss_e + force_weight * r.loss_f;
    r.E.resize(G), r.F.resize(G);
    for (int g = 0; g < G; g++) {
      const Structure &s = *batch[g];
      double c = 0;
      for (int z : s.Z)
        if (auto it = model.composition().find(z); it != model.composition().end()) c += it->second;
      r.E[g] = scale * Eraw[g] + c;
      r.F[g].resize(3 * s.n);
      const double *R = rot[g].data(); // back to the dataset frame: R^T
      for (int i = 0; i < s.n; i++)
        for (int a2 = 0; a2 < 3; a2++) {
          double v = 0;
          for (int b = 0; b < 3; b++) v += R[3 * b + a2] * (-scale * dEdx[3 * (a0[g] + i) + b]);
          r.F[g][3 * i + a2] = v;
        }
    }
    if (train && plain) {
      r.grad.assign(n_values, 0.0);
      std::vector<float> gbuf;
      for (size_t k = 0; k < params.size(); k++) {
        if (!grads[k]) continue;
        gbuf.resize(ggml_nelements(params[k]));
        ggml_backend_tensor_get(grads[k], gbuf.data(), 0, ggml_nbytes(params[k]));
        for (size_t q = 0; q < gbuf.size(); q++) r.grad[offsets[k] + q] = gbuf[q];
      }
    } else if (train) {
      // eps: displacements of at most fd_eps A and energy weights within 1 +- fd_eps
      double umax = 0;
      for (double v : u) umax = std::max(umax, std::abs(v));
      for (double v : a) umax = std::max(umax, std::abs(v));
      const double eps = umax > 0 ? fd_eps / umax : fd_eps;
      r.grad.assign(n_values, 0.0);
      std::vector<float> ev(3 * E), pos(3 * size_t(n)), w(G), gbuf;
      for (int sgn : {+1, -1}) {
        for (int A = 0; A < n; A++)
          for (int c = 0; c < 3; c++) pos[3 * A + c] = float(pos0[3 * A + c] + sgn * eps * u[3 * A + c]);
        for (int64_t e = 0; e < E; e++) {
          const int i = cen[e], g = sidx[i], J = a0[g] + ed[g].j[e - e0[g]];
          for (int c = 0; c < 3; c++) ev[3 * e + c] = float(ev0[3 * e + c] + sgn * eps * (u[3 * J + c] - u[3 * i + c]));
        }
        for (int g = 0; g < G; g++) w[g] = float(1.0 + sgn * eps * a[g]);
        run(ev, pos, w);
        const auto h0 = clk::now();
        for (size_t k = 0; k < params.size(); k++) {
          if (!grads[k]) continue;
          gbuf.resize(ggml_nelements(params[k]));
          ggml_backend_tensor_get(grads[k], gbuf.data(), 0, ggml_nbytes(params[k]));
          double *dst = r.grad.data() + offsets[k];
          const double f = sgn / (2 * eps);
          for (size_t q = 0; q < gbuf.size(); q++) dst[q] += f * gbuf[q];
        }
        t_host += ms(h0, clk::now());
      }
    }
    ggml_backend_buffer_free(ibuf);
    ggml_free(ctx);
    t_build += ms(t0, t1);
    return r;
  }
};

// torch.optim.Adam (no weight decay) on the host, in double
struct Adam {
  std::vector<double> m, v;
  double b1 = 0.9, b2 = 0.999, eps = 1e-8;
  int64_t t = 0;
  void step(std::vector<float> &p, const std::vector<double> &g, double lr) {
    if (m.empty()) m.assign(p.size(), 0.0), v.assign(p.size(), 0.0);
    t++;
    const double c1 = 1 - std::pow(b1, double(t)), c2 = 1 - std::pow(b2, double(t));
    for (size_t k = 0; k < p.size(); k++) {
      m[k] = b1 * m[k] + (1 - b1) * g[k];
      v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k];
      const double denom = std::sqrt(v[k]) / std::sqrt(c2) + eps;
      p[k] = float(p[k] - lr / c1 * m[k] / denom);
    }
  }
};

void save_gguf(const std::string &source, const std::string &path, const Trainer &tr, const std::vector<float> &values,
               const nlohmann::json &info) {
  ggml_context *data = nullptr;
  gguf_context *in = gguf_init_from_file(source.c_str(), {false, &data});
  if (!in) throw std::runtime_error("cannot read " + source);
  for (size_t k = 0; k < tr.params.size(); k++) {
    ggml_tensor *t = ggml_get_tensor(data, tr.names[k].c_str());
    std::memcpy(t->data, values.data() + tr.offsets[k], ggml_nbytes(t));
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

struct Metrics {
  double e_mae = 0, f_mae = 0, loss = 0;
  int64_t n_atoms = 0, n_struct = 0;
};

Metrics validate(Trainer &tr, const std::vector<Structure> &val, int B) {
  Metrics m;
  std::array<double, 9> I = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double fsum = 0;
  int64_t fcount = 0;
  for (size_t s = 0; s < val.size(); s += B) {
    std::vector<const Structure *> b;
    for (size_t k = s; k < std::min(val.size(), s + B); k++) b.push_back(&val[k]);
    auto r = tr.evaluate(b, std::vector<std::array<double, 9>>(b.size(), I), false);
    m.loss += r.loss * b.size();
    for (size_t g = 0; g < b.size(); g++) {
      m.e_mae += std::abs(r.E[g] - b[g]->energy) / b[g]->n;
      for (int q = 0; q < 3 * b[g]->n; q++) fsum += std::abs(r.F[g][q] - b[g]->F[q]), fcount++;
    }
    m.n_struct += b.size();
  }
  m.e_mae /= m.n_struct, m.f_mae = fsum / fcount, m.loss /= m.n_struct;
  return m;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string init = arg(argc, argv, "--init", ""), train_path = arg(argc, argv, "--train", "");
    if (init.empty() || train_path.empty()) {
      std::fprintf(stderr, "usage: mlipcpp-train-plato --init M.gguf --train T.bin [--val V.bin -o OUT.gguf | --check REF]\n");
      return 2;
    }
    plato::PlatoOptions o;
    o.backend = parse_backend_preference(arg(argc, argv, "--backend", "auto"));
    o.fold_in_graph = true, o.decomposed_layer_norm = true;
    plato::PlatoModel model(o);
    model.load_from_gguf(init);
    std::printf("backend %s\n", model.backend_name());
    Trainer tr(model, param_names(init));
    tr.fd_eps = std::stod(arg(argc, argv, "--fd-eps", "1e-3"));
    tr.energy_weight = std::stod(arg(argc, argv, "--energy-weight", "1"));
    tr.force_weight = std::stod(arg(argc, argv, "--force-weight", "1"));
    const auto train = read_dataset(train_path);
    std::printf("%zu training structures, %lld parameters\n", train.size(), (long long)tr.n_values);

    const std::string check = arg(argc, argv, "--check", "");
    if (!check.empty()) {
      std::ifstream jf(check + ".json");
      const auto ref = nlohmann::json::parse(jf);
      tr.energy_weight = ref["energy_weight"], tr.force_weight = ref["force_weight"];
      tr.plain = ref.value("plain", false);
      std::vector<const Structure *> b;
      for (int i : ref["indices"].get<std::vector<int>>()) b.push_back(&train.at(i));
      std::array<double, 9> I = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      const auto r = tr.evaluate(b, std::vector<std::array<double, 9>>(b.size(), I), true);
      std::vector<double> g(tr.n_values);
      std::ifstream gf(check + ".f64", std::ios::binary);
      gf.read(reinterpret_cast<char *>(g.data()), 8 * g.size());
      std::printf("loss  mlip.cpp %.10g (E %.6g F %.6g)  torch %.10g (E %.6g F %.6g)  rel %.2e\n", r.loss, r.loss_e, r.loss_f,
                  ref["loss"].get<double>(), ref["loss_energy"].get<double>(), ref["loss_forces"].get<double>(),
                  std::abs(r.loss - ref["loss"].get<double>()) / std::abs(ref["loss"].get<double>()));
      // per tensor, in the reference's (named_parameters) order
      double num = 0, den = 0, dot = 0, na = 0;
      std::vector<std::pair<double, std::string>> worst;
      for (const auto &p : ref["params"]) {
        const std::string name = p["name"];
        const int64_t ro = p["offset"];
        auto it = std::find(tr.names.begin(), tr.names.end(), name);
        if (it == tr.names.end()) throw std::runtime_error("no parameter " + name);
        const int64_t mo = tr.offsets[it - tr.names.begin()], cnt = ggml_nelements(tr.params[it - tr.names.begin()]);
        double n2 = 0, d2 = 0;
        for (int64_t q = 0; q < cnt; q++) {
          const double x = r.grad[mo + q], y = g[ro + q];
          n2 += (x - y) * (x - y), d2 += y * y, dot += x * y, na += x * x;
        }
        num += n2, den += d2;
        // relative to the tensor's norm, or to the whole gradient's scale for (near) zero tensors
        worst.emplace_back(std::sqrt(n2 / std::max(d2, 1e-12 * std::max(den, 1e-300))), name);
        if (std::getenv("PLATO_CHECK_PRINT") && name.find(std::getenv("PLATO_CHECK_PRINT")) != std::string::npos)
          for (int64_t q = 0; q < std::min<int64_t>(cnt, 4); q++)
            std::printf("    %s[%lld] mlip.cpp %.6e torch %.6e\n", name.c_str(), (long long)q, r.grad[mo + q], g[ro + q]);
      }
      std::sort(worst.rbegin(), worst.rend());
      std::printf("gradient: |g - g_torch| / |g_torch| = %.3e, cosine %.8f, %zu tensors\n", std::sqrt(num / den),
                  dot / std::sqrt(na * den), worst.size());
      for (size_t k = 0; k < std::min<size_t>(6, worst.size()); k++)
        std::printf("  worst %-40s %.3e\n", worst[k].second.c_str(), worst[k].first);
      std::vector<double> rels;
      for (auto &w : worst) rels.push_back(w.first);
      std::printf("  median tensor rel %.3e\n", rels[rels.size() / 2]);
      return 0;
    }

    const auto val = read_dataset(arg(argc, argv, "--val", ""));
    const std::string output = arg(argc, argv, "-o", "plato-trained.gguf");
    const int epochs = std::stoi(arg(argc, argv, "--epochs", "3")), B = std::stoi(arg(argc, argv, "--batch-size", "16"));
    const double lr0 = std::stod(arg(argc, argv, "--lr", "1e-4")), warm = std::stod(arg(argc, argv, "--warmup", "0.01"));
    const double clip = std::stod(arg(argc, argv, "--grad-clip", "1.0"));
    const int64_t max_steps = std::stoll(arg(argc, argv, "--max-steps", "-1"));
    const bool augment = !flag(argc, argv, "--no-augment");
    std::mt19937_64 rng(std::stoull(arg(argc, argv, "--seed", "0")));
    const int64_t spe = (int64_t(train.size()) + B - 1) / B, total = int64_t(epochs) * spe, warm_steps = int64_t(warm * total);
    auto lr_at = [&](int64_t step) {
      if (step < warm_steps) return lr0 * double(step) / double(std::max<int64_t>(1, warm_steps));
      const double p = double(step - warm_steps) / double(std::max<int64_t>(1, total - warm_steps));
      return lr0 * 0.5 * (1 + std::cos(M_PI * p));
    };
    auto m0 = validate(tr, val, B);
    std::printf("epoch 0 val: E MAE %.4f meV/atom, F MAE %.4f meV/A, loss %.6g\n", 1e3 * m0.e_mae, 1e3 * m0.f_mae, m0.loss);
    std::fflush(stdout);
    Adam adam;
    std::vector<float> p = tr.get_params();
    int64_t step = 0;
    nlohmann::json log = nlohmann::json::array();
    for (int ep = 1; ep <= epochs; ep++) {
      std::vector<size_t> order(train.size());
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), rng);
      const auto t0 = std::chrono::steady_clock::now();
      double loss_sum = 0;
      int64_t nb = 0;
      tr.t_build = tr.t_compute = tr.t_host = 0;
      for (size_t s = 0; s < order.size(); s += B) {
        std::vector<const Structure *> b;
        std::vector<std::array<double, 9>> rot;
        for (size_t k = s; k < std::min(order.size(), s + B); k++) {
          b.push_back(&train[order[k]]);
          std::array<double, 9> R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
          if (augment) random_o3(rng, R.data());
          rot.push_back(R);
        }
        auto r = tr.evaluate(b, rot, true);
        double gn = 0;
        for (double v : r.grad) gn += v * v;
        gn = std::sqrt(gn);
        if (gn > clip) {
          const double f = clip / (gn + 1e-6);
          for (double &v : r.grad) v *= f;
        }
        adam.step(p, r.grad, lr_at(step));
        tr.set_params(p);
        loss_sum += r.loss, nb++, step++;
        if (step % 100 == 0) {
          std::printf("  step %lld loss %.6g |g| %.3g lr %.3g\n", (long long)step, loss_sum / nb, gn, lr_at(step));
          std::fflush(stdout);
        }
        if (max_steps > 0 && step >= max_steps) break;
      }
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      auto m = validate(tr, val, B);
      std::printf("epoch %d: train loss %.6g, %.1f s (graph build+inputs %.1f s, compute %.1f s, host grads %.1f s), val E MAE %.4f meV/atom, F MAE %.4f meV/A, loss %.6g\n",
                  ep, loss_sum / nb, wall, tr.t_build / 1e3, tr.t_compute / 1e3, tr.t_host / 1e3, 1e3 * m.e_mae, 1e3 * m.f_mae, m.loss);
      std::fflush(stdout);
      log.push_back({{"epoch", ep}, {"train_loss", loss_sum / nb}, {"wall_s", wall}, {"val_e_mae_mev_atom", 1e3 * m.e_mae},
                     {"val_f_mae_mev_a", 1e3 * m.f_mae}, {"val_loss", m.loss}});
      save_gguf(init, output, tr, p, {{"log", log}, {"init", init}, {"lr", lr0}, {"batch_size", B}, {"fd_eps", tr.fd_eps}});
      if (max_steps > 0 && step >= max_steps) break;
    }
    std::printf("wrote %s\n", output.c_str());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
