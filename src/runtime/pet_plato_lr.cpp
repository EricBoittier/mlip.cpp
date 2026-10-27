#include "pet_plato_lr.h"

#include "core/gguf_loader.h"
#include "graph_ir.h"
#include "models/plato/plato.h"

#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>

namespace mlipcpp::runtime {

namespace {

constexpr double KE = 14.399645; // eV A / e^2, metatrain plato_pet_lr modules/ewald.py

// full FP32 products for the geometry (as runtime/plato_graph.cpp)
ggml_tensor *mm_f32(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b) {
  ggml_tensor *y = ggml_mul_mat(ctx, a, b);
  ggml_mul_mat_set_prec(y, GGML_PREC_F32);
  ggml_mul_mat_set_grad_prec(y, GGML_PREC_F32);
  return y;
}

ggml_tensor *cont(ggml_context *ctx, ggml_tensor *a) { return ggml_is_contiguous(a) ? a : ggml_cont(ctx, a); }

double param_num(const GIRNode &node, const std::string &key, double dflt) {
  auto it = node.params.find(key);
  if (it == node.params.end()) return dflt;
  if (auto *d = std::get_if<double>(&it->second)) return *d;
  if (auto *i = std::get_if<int64_t>(&it->second)) return static_cast<double>(*i);
  if (auto *b = std::get_if<bool>(&it->second)) return *b ? 1.0 : 0.0;
  return dflt;
}

void inverse3(const double A[3][3], double I[3][3], double &det) {
  det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  if (!(std::abs(det) > 1e-12)) throw std::runtime_error("Plato-PET-LR: singular cell");
  const double id = 1.0 / det;
  I[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * id;
  I[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * id;
  I[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * id;
  I[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * id;
  I[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * id;
  I[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * id;
  I[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * id;
  I[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * id;
  I[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * id;
}

} // namespace

PetPlatoLRSpec PetPlatoLRSpec::from_gguf(const GGUFLoader &L) {
  PetPlatoLRSpec s;
  s.enabled = L.get_int32("pet.plato_lr", 0) != 0;
  if (!s.enabled) return s;
  s.frames = L.get_string("pet.plato_lr.frames", "identity");
  s.n_frames = L.get_int32("pet.plato_lr.n_frames", 1);
  s.rope = L.get_int32("pet.plato_lr.rope", 0) != 0;
  s.edge_group_sum = L.get_int32("pet.plato_lr.edge_group_sum", 0) != 0;
  s.mesh = L.get_int32("pet.plato_lr.mesh_coupling", 0) != 0;
  s.couple_at = L.get_int32("pet.plato_lr.couple_at", -1);
  s.mesh_channels = L.get_int32("pet.plato_lr.mesh_channels", 0);
  s.mesh_spacing = L.get_float32("pet.plato_lr.mesh_spacing", 1.0f);
  s.mesh_sigma = L.get_float32("pet.plato_lr.mesh_sigma", 1.5f);
  s.mesh_realspace = L.get_string("pet.plato_lr.mesh_open_boundary", "realspace") == "realspace";
  s.n_quad = L.get_int32("pet.plato_lr.n_quad", 0);
  s.les = L.get_int32("pet.plato_lr.les", 0) != 0;
  s.num_charges = L.get_int32("pet.plato_lr.num_charges", 0);
  s.les_sigma = L.get_float32("pet.plato_lr.les_sigma", 1.0f);
  s.les_kcut = L.get_float32("pet.plato_lr.les_kcut", 6.0f);
  s.total_charge_constraint = L.get_int32("pet.plato_lr.total_charge_constraint", 0) != 0;
  s.les_method = L.get_string("pet.plato_lr.les_method", "auto");
  s.les_pme_atoms = L.get_int32("pet.plato_lr.les_pme_atoms", 128);
  if (const char *e = std::getenv("MLIPCPP_PLATO_LES_METHOD"); e && *e) s.les_method = e;
  if (s.les_method != "ewald" && s.les_method != "pme" && s.les_method != "auto")
    throw std::runtime_error("Plato-PET-LR: les_method must be ewald, pme or auto, not " + s.les_method);
  return s;
}

void PetPlatoLR::create(ggml_context *ctx, const PetPlatoLRSpec &spec, const std::vector<PetPlatoLRStructure> &structures,
                        int n_atoms, bool grad) {
  spec_ = spec, st_ = structures, n_ = n_atoms;
  all_.clear();
  frac_ = pair_r_ = nullptr;
  const int G = static_cast<int>(st_.size());
  if (G < 1) throw std::runtime_error("Plato-PET-LR: no structure");
  periodic_ = st_[0].periodic;
  for (const auto &s : st_)
    if (s.periodic != periodic_) throw std::runtime_error("Plato-PET-LR: a batch mixes periodic and open structures");
  auto input = [&](ggml_type t, int64_t a, int64_t b = 1, int64_t c = 1, const char *name = "") {
    ggml_tensor *x = ggml_new_tensor_3d(ctx, t, a, b, c);
    ggml_set_name(x, name);
    ggml_set_input(x);
    all_.push_back(x);
    return x;
  };
  positions = input(GGML_TYPE_F32, 3, n_, 1, "plato_lr.positions");
  if (grad) ggml_set_param(positions);
  // structure bookkeeping
  h_sidx_.assign(n_, 0), h_onehot_.assign(size_t(n_) * G, 0.0f), h_invc_.assign(G, 0.0f), h_qpa_.assign(n_, 0.0f);
  for (int g = 0; g < G; g++) {
    const auto &s = st_[g];
    const int na = s.a1 - s.a0;
    if (na <= 0 || s.a0 < 0 || s.a1 > n_) throw std::runtime_error("Plato-PET-LR: bad structure atom range");
    h_invc_[g] = 1.0f / na;
    for (int i = s.a0; i < s.a1; i++)
      h_sidx_[i] = g, h_onehot_[size_t(g) * n_ + i] = 1.0f, h_qpa_[i] = static_cast<float>(s.charge / na);
  }
  struct_index_ = input(GGML_TYPE_I32, n_, 1, 1, "plato_lr.struct_index");
  onehot_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, n_, G, 1, "plato_lr.onehot"), n_, G);
  inv_count_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, G, 1, "plato_lr.inv_count"), 1, G);
  charge_per_atom_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, n_, 1, "plato_lr.charge_per_atom"), 1, n_);
  mesh_[0] = mesh_[1] = mesh_[2] = 0, les_mesh_[0] = les_mesh_[1] = les_mesh_[2] = 0, nk_ = 0, n_pairs_ = 0;
  strain = nullptr;
  if (!spec_.long_range()) return;

  if (!periodic_) { // off-diagonal pairs within each structure
    h_pi_.clear(), h_pj_.clear();
    for (const auto &s : st_)
      for (int i = s.a0; i < s.a1; i++)
        for (int j = s.a0; j < s.a1; j++)
          if (i != j) h_pi_.push_back(i), h_pj_.push_back(j);
    n_pairs_ = static_cast<int64_t>(h_pi_.size());
    if (n_pairs_ > 0) {
      pair_i_ = input(GGML_TYPE_I32, n_pairs_, 1, 1, "plato_lr.pair_i");
      pair_j_ = input(GGML_TYPE_I32, n_pairs_, 1, 1, "plato_lr.pair_j");
    }
    return;
  }

  // periodic: inverse cells, volumes, mesh and k vectors per structure
  strain = input(GGML_TYPE_F32, 3, 3, G, "plato_lr.strain");
  if (grad) ggml_set_param(strain);
  std::vector<std::array<std::array<double, 3>, 3>> inv(G);
  h_invv_.assign(G, 0.0f), h_aict_.assign(9 * size_t(n_), 0.0f);
  for (int g = 0; g < G; g++) {
    double Ci[3][3], det;
    inverse3(st_[g].cell, Ci, det);
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) inv[g][a][b] = Ci[a][b];
    h_invv_[g] = static_cast<float>(1.0 / std::abs(det));
    for (int i = st_[g].a0; i < st_[g].a1; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) h_aict_[9 * size_t(i) + j * 3 + k] = static_cast<float>(Ci[k][j]);
  }
  atom_inv_cell_t_ = input(GGML_TYPE_F32, 3, 3, n_, "plato_lr.atom_inv_cell_t");
  inv_volume_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, G, 1, "plato_lr.inv_volume"), 1, G);
  auto row = [&](int g, int a) {
    static thread_local float r[3];
    for (int b = 0; b < 3; b++) r[b] = static_cast<float>(st_[g].cell[a][b]);
    return r;
  };
  auto rnorm = [&](int g, int a) {
    double s = 0;
    for (int b = 0; b < 3; b++) s += st_[g].cell[a][b] * st_[g].cell[a][b];
    return std::sqrt(s);
  };
  if (spec_.mesh) {
    for (int a = 0; a < 3; a++) mesh_[a] = plato::mesh_points(row(0, a), spec_.mesh_spacing);
    for (int g = 1; g < G; g++)
      for (int a = 0; a < 3; a++)
        if (plato::mesh_points(row(g, a), spec_.mesh_spacing) != mesh_[a])
          throw std::runtime_error("Plato-PET-LR: the periodic structures of a batch need the same mesh");
    const int M0 = mesh_[0], M1 = mesh_[1], M2 = mesh_[2], Mh2 = M2 / 2 + 1;
    const int64_t Mh = int64_t(M0) * M1 * Mh2;
    h_mk_.assign(3 * size_t(Mh) * G, 0.0f), h_mpad_.assign(Mh, 0.0f), h_mdec_.assign(Mh, 0.0f);
    auto freq = [](int i, int M) { return i <= (M - 1) / 2 ? i : i - M; };
    for (int i0 = 0; i0 < M0; i0++)
      for (int i1 = 0; i1 < M1; i1++)
        for (int i2 = 0; i2 < Mh2; i2++) {
          const int64_t e = (int64_t(i0) * M1 + i1) * Mh2 + i2;
          const double m[3] = {double(freq(i0, M0)), double(freq(i1, M1)), double(i2)};
          for (int g = 0; g < G; g++)
            for (int b = 0; b < 3; b++)
              h_mk_[3 * (size_t(g) * Mh + e) + b] =
                  static_cast<float>(2 * M_PI * (m[0] * inv[g][b][0] + m[1] * inv[g][b][1] + m[2] * inv[g][b][2]));
          double dc = 1;
          for (int a = 0; a < 3; a++) {
            const double f = (4 + 2 * std::cos(2 * M_PI * m[a] / mesh_[a])) / 6;
            dc *= f * f;
          }
          const bool zero = i0 == 0 && i1 == 0 && i2 == 0;
          h_mpad_[e] = zero ? 1.0f : 0.0f;
          h_mdec_[e] = zero ? 0.0f : static_cast<float>(1.0 / dc);
        }
    mesh_k_ = input(GGML_TYPE_F32, 3, Mh, G, "plato_lr.mesh_k");
    mesh_k0_pad_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, Mh, 1, "plato_lr.mesh_k0_pad"), 1, Mh);
    mesh_deconv_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, Mh, 1, "plato_lr.mesh_deconv"), 1, Mh);
  }
  if (spec_.les) {
    int max_atoms = 0;
    for (const auto &s : st_) max_atoms = std::max(max_atoms, s.a1 - s.a0);
    const bool pme = spec_.les_method == "pme" || (spec_.les_method == "auto" && max_atoms >= spec_.les_pme_atoms);
    if (pme) {
      if (spec_.num_charges != 1) throw std::runtime_error("Plato-PET-LR: LES PME needs one charge channel");
      const double h = spec_.les_pme_spacing * spec_.les_sigma;
      for (int a = 0; a < 3; a++) les_mesh_[a] = plato::pme_mesh_points(row(0, a), h, spec_.les_pme_order);
      for (int g = 1; g < G; g++)
        for (int a = 0; a < 3; a++)
          if (plato::pme_mesh_points(row(g, a), h, spec_.les_pme_order) != les_mesh_[a])
            throw std::runtime_error("Plato-PET-LR: the periodic structures of a batch need the same LES mesh");
      h_tab_ = plato::pme_axis_table(les_mesh_, spec_.les_pme_order);
      h_lK_.assign(9 * size_t(G), 0.0f);
      for (int g = 0; g < G; g++)
        for (int b = 0; b < 3; b++)
          for (int q = 0; q < 3; q++) h_lK_[9 * size_t(g) + 3 * b + q] = static_cast<float>(2 * M_PI * inv[g][q][b]);
      les_K_ = input(GGML_TYPE_F32, 3, 3, G, "plato_lr.les_K");
      les_tab_ = input(GGML_TYPE_F32, static_cast<int64_t>(h_tab_.size()), 1, 1, "plato_lr.les_tab");
    } else { // the union over the batch of the integer k sets (each structure's window drops the rest)
      const double kc = spec_.les_kcut / spec_.les_sigma;
      int nmax[3] = {0, 0, 0};
      for (int g = 0; g < G; g++)
        for (int a = 0; a < 3; a++) nmax[a] = std::max(nmax[a], static_cast<int>(std::floor(kc * rnorm(g, a) / (2 * M_PI))));
      std::vector<std::array<int, 3>> ns;
      for (int a = -nmax[0]; a <= nmax[0]; a++)
        for (int b = -nmax[1]; b <= nmax[1]; b++)
          for (int c = -nmax[2]; c <= nmax[2]; c++) {
            if (!(a > 0 || (a == 0 && b > 0) || (a == 0 && b == 0 && c > 0))) continue;
            bool any = false;
            for (int g = 0; g < G && !any; g++) {
              double k2 = 0;
              for (int q = 0; q < 3; q++) {
                const double k = 2 * M_PI * (a * inv[g][q][0] + b * inv[g][q][1] + c * inv[g][q][2]);
                k2 += k * k;
              }
              any = k2 < kc * kc;
            }
            if (any) ns.push_back({a, b, c});
          }
      nk_ = static_cast<int64_t>(ns.size());
      if (nk_ == 0) ns.push_back({1, 0, 0}), nk_ = 1; // a tiny cell: one k, outside the window (weight 0)
      h_ln_.clear(), h_lk_.assign(3 * size_t(nk_) * G, 0.0f);
      for (int64_t e = 0; e < nk_; e++) {
        for (int q = 0; q < 3; q++) h_ln_.push_back(static_cast<float>(2 * M_PI * ns[e][q]));
        for (int g = 0; g < G; g++)
          for (int q = 0; q < 3; q++)
            h_lk_[3 * (size_t(g) * nk_ + e) + q] = static_cast<float>(
                2 * M_PI * (ns[e][0] * inv[g][q][0] + ns[e][1] * inv[g][q][1] + ns[e][2] * inv[g][q][2]));
      }
      les_n_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 3, nk_, 1, "plato_lr.les_n"), 3, nk_);
      les_k_ = input(GGML_TYPE_F32, 3, nk_, G, "plato_lr.les_k");
    }
  }
}

void PetPlatoLR::upload(const float *pos) const {
  auto set = [](ggml_tensor *t, const void *d) {
    if (t) ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t));
  };
  set(positions, pos);
  set(struct_index_, h_sidx_.data());
  set(onehot_, h_onehot_.data()), set(inv_count_, h_invc_.data()), set(charge_per_atom_, h_qpa_.data());
  if (pair_i_) set(pair_i_, h_pi_.data()), set(pair_j_, h_pj_.data());
  if (strain) {
    std::vector<float> z(ggml_nelements(strain), 0.0f);
    set(strain, z.data());
  }
  if (atom_inv_cell_t_) set(atom_inv_cell_t_, h_aict_.data()), set(inv_volume_, h_invv_.data());
  if (mesh_k_) set(mesh_k_, h_mk_.data()), set(mesh_k0_pad_, h_mpad_.data()), set(mesh_deconv_, h_mdec_.data());
  if (les_n_) set(les_n_, h_ln_.data()), set(les_k_, h_lk_.data());
  if (les_K_) set(les_K_, h_lK_.data()), set(les_tab_, h_tab_.data());
}

ggml_tensor *PetPlatoLR::build(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in) {
  if (node.op == "MLIPCPP_ROPE_FRAME_ATTN") return rope_attn(ctx, node, in);
  if (node.op == "MLIPCPP_EDGE_GROUP_SUM") return edge_sum(ctx, in);
  if (node.op == "MLIPCPP_PLATO_MESH") return mesh_op(ctx, node, in);
  if (node.op == "MLIPCPP_PLATO_LES") return les_op(ctx, node, in);
  throw std::runtime_error("Plato-PET-LR: unknown graph op " + node.op);
}

ggml_tensor *PetPlatoLR::frac(ggml_context *ctx) {
  if (!frac_) // [3, n]: each atom with the inverse cell of its structure
    frac_ = ggml_reshape_2d(ctx, mm_f32(ctx, atom_inv_cell_t_, ggml_reshape_3d(ctx, positions, 3, 1, n_)), 3, n_);
  return frac_;
}

ggml_tensor *PetPlatoLR::pair_r(ggml_context *ctx) {
  if (!pair_r_) {
    ggml_tensor *d = ggml_sub(ctx, ggml_get_rows(ctx, positions, pair_i_), ggml_get_rows(ctx, positions, pair_j_)); // [3, P]
    pair_r_ = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, d)));                                               // [1, P]
  }
  return pair_r_;
}

// q, k, v [Ch, L, H, N] (views), bias [L, L, 1, N], pos [3, L, N], omega [3, Ch/2], frames [3, 3, F]
ggml_tensor *PetPlatoLR::rope_attn(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in) {
  if (in.size() != 7) throw std::runtime_error("ROPE_FRAME_ATTN: 7 inputs");
  ggml_tensor *q = cont(ctx, in[0]), *k = cont(ctx, in[1]), *v = in[2], *bias = cont(ctx, in[3]);
  ggml_tensor *omega = in[5], *frames = in[6];
  const int64_t Ch = q->ne[0], L = q->ne[1], H = q->ne[2], N = q->ne[3], C2 = Ch / 2, D = Ch * H;
  const int F = static_cast<int>(frames->ne[2]);
  const float scale = static_cast<float>(param_num(node, "scale", 1.0 / std::sqrt(double(Ch))));
  ggml_tensor *pos = ggml_reshape_2d(ctx, cont(ctx, in[4]), 3, L * N);
  // [Ch, L, H, N] -> [D, 1, L, N] (heads of a token together, as ggml_pet_attn reads them)
  auto tokens = [&](ggml_tensor *x) { return ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)), D, 1, L, N); };
  ggml_tensor *vt = tokens(v);
  auto half = [&](ggml_tensor *x, int s) {
    return ggml_view_4d(ctx, x, C2, L, H, N, x->nb[1], x->nb[2], x->nb[3], s * C2 * ggml_element_size(x));
  };
  ggml_tensor *acc = nullptr;
  for (int f = 0; f < F; f++) {
    ggml_tensor *R = ggml_view_2d(ctx, frames, 3, 3, frames->nb[1], f * frames->nb[2]);
    ggml_tensor *Rw = mm_f32(ctx, R, omega);                                     // [3, C2]: R_f omega_c
    ggml_tensor *ang = ggml_reshape_4d(ctx, mm_f32(ctx, Rw, pos), C2, L, 1, N);  // pos . R_f omega_c
    ggml_tensor *c = ggml_cos(ctx, ang), *s = ggml_sin(ctx, ang);
    auto rope = [&](ggml_tensor *x) {
      ggml_tensor *x1 = half(x, 0), *x2 = half(x, 1);
      return ggml_concat(ctx, ggml_sub(ctx, ggml_mul(ctx, x1, c), ggml_mul(ctx, x2, s)),
                         ggml_add(ctx, ggml_mul(ctx, x1, s), ggml_mul(ctx, x2, c)), 0);
    };
    ggml_tensor *xt = ggml_concat(ctx, ggml_concat(ctx, tokens(rope(q)), tokens(rope(k)), 1), vt, 1); // [D, 3, L, N]
    ggml_tensor *r = ggml_pet_attn(ctx, xt, bias, static_cast<int>(H), scale);                         // [D + H, L, N]
    ggml_tensor *o = ggml_cont(ctx, ggml_view_3d(ctx, r, D, L, N, r->nb[1], r->nb[2], 0));
    acc = acc ? ggml_add(ctx, acc, o) : o;
  }
  if (F > 1) acc = ggml_scale(ctx, acc, 1.0f / F);
  return ggml_permute(ctx, ggml_reshape_4d(ctx, acc, Ch, H, L, N), 0, 2, 1, 3); // [Ch, L, H, N]
}

// x [4, M, n] ([r, |r|]), W [3, d], frames [3, 3, F] -> [d, M, n]
ggml_tensor *PetPlatoLR::edge_sum(ggml_context *ctx, const std::vector<ggml_tensor *> &in) {
  if (in.size() != 3) throw std::runtime_error("EDGE_GROUP_SUM: 3 inputs");
  ggml_tensor *x = in[0], *W = in[1], *frames = in[2];
  const int64_t M = x->ne[1], n = x->ne[2], E = M * n, d = W->ne[1];
  const int F = static_cast<int>(frames->ne[2]);
  ggml_tensor *r = ggml_cont(ctx, ggml_view_2d(ctx, cont(ctx, x), 3, E, x->ne[0] * ggml_element_size(x), 0)); // [3, E]
  ggml_tensor *acc = nullptr;
  for (int f = 0; f < F; f++) {
    ggml_tensor *R = ggml_view_2d(ctx, frames, 3, 3, frames->nb[1], f * frames->nb[2]);
    ggml_tensor *rf = mm_f32(ctx, ggml_cont(ctx, ggml_transpose(ctx, R)), r); // R_f^T r
    ggml_tensor *t = ggml_silu(ctx, ggml_mul_mat(ctx, W, rf));                 // [d, E]
    acc = acc ? ggml_add(ctx, acc, t) : t;
  }
  if (F > 1) acc = ggml_scale(ctx, acc, 1.0f / F);
  return ggml_reshape_3d(ctx, acc, d, M, n);
}

// v [C, n], filt.0 (w [1, h], b [h]), filt.2 (w [h, C], b [C]), k_quad [nq], w_quad [nq] -> [C, n]
ggml_tensor *PetPlatoLR::mesh_op(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in) {
  if (in.size() != 7) throw std::runtime_error("PLATO_MESH: 7 inputs");
  ggml_tensor *v = cont(ctx, in[0]);
  const int64_t C = v->ne[0], n = v->ne[1];
  const int G = n_struct();
  auto filt = [&](ggml_tensor *kk) { // [1, m] -> [C, m]
    return ggml_add(ctx, ggml_mul_mat(ctx, in[3], ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, in[1], kk), in[2]))), in[4]);
  };
  const float sigma = static_cast<float>(param_num(node, "sigma", spec_.mesh_sigma));
  if (!periodic_) {
    if (param_num(node, "open_boundary", 1) == 0) return ggml_scale(ctx, v, 0.0f);
    ggml_tensor *kq = in[5], *wq = in[6];
    const int64_t nq = kq->ne[0];
    ggml_tensor *wf = ggml_mul(ctx, filt(ggml_reshape_2d(ctx, kq, 1, nq)), ggml_reshape_2d(ctx, wq, 1, nq)); // [C, nq]
    ggml_tensor *A = ggml_cont(ctx, ggml_transpose(ctx, wf));                                                 // [nq, C]
    // i = j: K(0) = sqrt(2 / pi) / sigma + sum_n w_n f(k_n)
    const float k0 = static_cast<float>(std::sqrt(2.0 / M_PI) / sigma);
    ggml_tensor *diag = ggml_scale_bias(ctx, ggml_reshape_2d(ctx, ggml_sum_rows(ctx, A), C, 1), 1.0f, k0); // [C, 1]
    ggml_tensor *m = ggml_mul(ctx, v, diag);
    if (n_pairs_ > 0) {
      ggml_tensor *r = pair_r(ctx);                                                              // [1, P]
      const int64_t P = n_pairs_;
      ggml_tensor *kr = ggml_mul(ctx, ggml_repeat(ctx, r, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nq, P)),
                                 ggml_reshape_2d(ctx, kq, nq, 1));                                  // [nq, P]
      ggml_tensor *j0 = ggml_div(ctx, ggml_sin(ctx, kr), kr);
      ggml_tensor *K = ggml_add(ctx, mm_f32(ctx, A, j0),
                                ggml_div(ctx, ggml_plato_erf(ctx, ggml_scale(ctx, r, 1.0f / (std::sqrt(2.0f) * sigma))), r)); // [C, P]
      ggml_tensor *t = ggml_mul(ctx, K, ggml_get_rows(ctx, v, pair_j_));
      m = ggml_add(ctx, m, ggml_get_rows_back(ctx, t, pair_i_, v));
    }
    return m;
  }
  // periodic (runtime/plato_graph.cpp's mesh coupling), times the mesh points
  const int M0 = mesh_[0], M1 = mesh_[1], M2 = mesh_[2];
  const int64_t Mh = int64_t(M0) * M1 * (M2 / 2 + 1);
  ggml_tensor *fr = frac(ctx);
  ggml_tensor *rho = G > 1 ? ggml_plato_spread_blocked(ctx, v, fr, struct_index_, G, M0, M1, M2)
                           : ggml_plato_spread(ctx, v, fr, M0, M1, M2);
  ggml_tensor *rk = ggml_plato_fft3(ctx, rho, 0, M0, M1, M2);
  ggml_tensor *k = ggml_sub(ctx, mesh_k_, mm_f32(ctx, strain, mesh_k_)); // [3, Mh, G]
  ggml_tensor *k2 = ggml_sum_rows(ctx, ggml_sqr(ctx, k));                // [1, Mh, G]
  ggml_tensor *k2s = ggml_add(ctx, k2, mesh_k0_pad_);
  ggml_tensor *base = ggml_div(ctx, ggml_exp(ctx, ggml_scale(ctx, k2, -0.5f * sigma * sigma)), k2s);
  auto e = [&](int q) { return ggml_cont(ctx, ggml_view_2d(ctx, strain, 1, G, strain->nb[2], q * sizeof(float))); };
  ggml_tensor *tr = ggml_add(ctx, ggml_add(ctx, e(0), e(4)), e(8));
  ggml_tensor *ivol = ggml_mul(ctx, ggml_scale_bias(ctx, tr, -1.0f, 1.0f), inv_volume_); // [1, G]
  const float npts = static_cast<float>(double(M0) * M1 * M2);
  base = ggml_mul(ctx, ggml_mul(ctx, base, mesh_deconv_), ggml_reshape_3d(ctx, ggml_scale(ctx, ivol, 4.0f * M_PI * npts), 1, 1, G));
  ggml_tensor *g = ggml_mul(ctx, ggml_scale_bias(ctx, filt(ggml_reshape_2d(ctx, ggml_sqrt(ctx, k2s), 1, Mh * G)), 1.0f, 1.0f),
                            ggml_reshape_2d(ctx, base, 1, Mh * G)); // [C, Mh G]
  g = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, g, C, Mh, G), 1, 0, 2, 3)), 1, Mh, C, G);
  rk = ggml_reshape_4d(ctx, ggml_mul(ctx, ggml_reshape_4d(ctx, rk, 2, Mh, C, G), g), rk->ne[0], M1, M0, C * G);
  ggml_tensor *phi = ggml_plato_fft3(ctx, rk, 1, M0, M1, M2);
  return G > 1 ? ggml_plato_gather_blocked(ctx, phi, fr, struct_index_, static_cast<int>(C), false)
               : ggml_plato_gather(ctx, phi, fr, false); // [C, n]
}

// q [Q, n], charge_unit [1] -> [n]
ggml_tensor *PetPlatoLR::les_op(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in) {
  if (in.size() != 2) throw std::runtime_error("PLATO_LES: 2 inputs");
  ggml_tensor *q = cont(ctx, in[0]);
  if (std::getenv("MLIPCPP_PLATO_LR_DEBUG"))
    std::fprintf(stderr, "PLATO_LES q [%lld %lld %lld %lld]\n", (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2], (long long)q->ne[3]);
  const int64_t Q = q->ne[0], n = q->ne[1];
  const int G = n_struct();
  const float sigma = static_cast<float>(param_num(node, "sigma", spec_.les_sigma));
  const float kcut = static_cast<float>(param_num(node, "kcut", spec_.les_kcut));
  const bool tcc = param_num(node, "total_charge_constraint", spec_.total_charge_constraint) != 0;
  // neutral per structure (or the structure's total charge / charge_unit)
  ggml_tensor *qt = ggml_cont(ctx, ggml_transpose(ctx, q));                                           // [n, Q]
  ggml_tensor *mean = ggml_mul(ctx, ggml_cont(ctx, ggml_transpose(ctx, mm_f32(ctx, onehot_, qt))), inv_count_); // [Q, G]
  q = ggml_sub(ctx, q, ggml_get_rows(ctx, mean, struct_index_));
  if (tcc) q = ggml_add(ctx, q, ggml_div(ctx, charge_per_atom_, ggml_repeat(ctx, in[1], charge_per_atom_)));
  auto per_atom = [&](ggml_tensor *Eg) { // [1, G] -> [n], E_g / N_g on each atom of g
    return ggml_reshape_1d(ctx, ggml_get_rows(ctx, ggml_mul(ctx, Eg, inv_count_), struct_index_), n);
  };
  if (!periodic_) {
    if (n_pairs_ == 0) return ggml_reshape_1d(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, q), 0.0f), n);
    ggml_tensor *r = pair_r(ctx);
    ggml_tensor *pair = ggml_div(ctx, ggml_plato_erf(ctx, ggml_scale(ctx, r, 1.0f / (std::sqrt(2.0f) * sigma))), r); // [1, P]
    ggml_tensor *s = ggml_get_rows_back(ctx, ggml_mul(ctx, ggml_get_rows(ctx, q, pair_j_), pair), pair_i_, q);     // [Q, n]
    return ggml_reshape_1d(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, q, s)), static_cast<float>(0.5 * KE)), n);
  }
  qt = ggml_cont(ctx, ggml_transpose(ctx, q));                                                   // [n, Q], centred
  ggml_tensor *fr = frac(ctx);
  auto e = [&](int k) { return ggml_cont(ctx, ggml_view_2d(ctx, strain, 1, G, strain->nb[2], k * sizeof(float))); };
  ggml_tensor *ivol = ggml_mul(ctx, ggml_scale_bias(ctx, ggml_add(ctx, ggml_add(ctx, e(0), e(4)), e(8)), -1.0f, 1.0f), inv_volume_);
  // sum_i q_i^2 per structure: [1, n] . onehot [n, G]
  ggml_tensor *q2 = ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, q)))); // [n, 1]
  ggml_tensor *selfE = ggml_scale(ctx, ggml_reshape_2d(ctx, mm_f32(ctx, q2, onehot_), 1, G), 1.0f / (std::sqrt(2.0f * M_PI) * sigma));
  ggml_tensor *rec;
  if (les_mesh_[0] > 0) {
    const int M0 = les_mesh_[0], M1 = les_mesh_[1], M2 = les_mesh_[2];
    ggml_tensor *rho = ggml_plato_spread_order(ctx, ggml_reshape_2d(ctx, q, 1, n), fr, G > 1 ? struct_index_ : nullptr, G, M0,
                                               M1, M2, spec_.les_pme_order);
    ggml_tensor *rk = ggml_plato_fft3(ctx, rho, 0, M0, M1, M2);
    ggml_tensor *K = ggml_sub(ctx, les_K_, mm_f32(ctx, strain, les_K_));
    ggml_tensor *A = ggml_reshape_2d(ctx, mm_f32(ctx, K, K), 9, G);
    const float kc = kcut / sigma, wd = 1.0f / sigma;
    ggml_tensor *ep = ggml_plato_pme(ctx, rk, A, les_tab_, M0, M1, M2, sigma, kc, wd); // [M0, G]
    rec = ggml_mul(ctx, ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ep), 1, G), ggml_scale(ctx, ivol, 2.0f * M_PI));
  } else {
    ggml_tensor *sk; // [nk, G Q]
    if (Q == 1) {
      ggml_tensor *r = ggml_plato_ewald(ctx, fr, ggml_reshape_1d(ctx, q, n), G == 1 ? nullptr : struct_index_, les_n_, G);
      sk = ggml_view_2d(ctx, r, nk_, G, r->nb[1], 2 * r->nb[2]);
    } else {
      if (G != 1) throw std::runtime_error("Plato-PET-LR: batched periodic LES needs one charge channel");
      ggml_tensor *ph = mm_f32(ctx, fr, les_n_); // [n, nk]
      ggml_tensor *Cq = mm_f32(ctx, ggml_cos(ctx, ph), qt), *Sq = mm_f32(ctx, ggml_sin(ctx, ph), qt); // [nk, Q]
      sk = ggml_add(ctx, ggml_sqr(ctx, Cq), ggml_sqr(ctx, Sq));
    }
    ggml_tensor *k = ggml_sub(ctx, les_k_, mm_f32(ctx, strain, les_k_));
    ggml_tensor *k2 = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, k)), nk_, G);
    const float kc = kcut / sigma, wd = 1.0f / sigma;
    ggml_tensor *xw = ggml_clamp(ctx, ggml_scale_bias(ctx, ggml_sqrt(ctx, k2), 1.0f / wd, -(kc - wd) / wd), 0.0f, 1.0f);
    ggml_tensor *win = ggml_scale_bias(ctx, ggml_cos(ctx, ggml_scale(ctx, xw, static_cast<float>(M_PI))), 0.5f, 0.5f);
    ggml_tensor *gk = ggml_div(ctx, ggml_mul(ctx, win, ggml_exp(ctx, ggml_scale(ctx, k2, -0.5f * sigma * sigma))), k2); // [nk, G]
    rec = Q == 1 ? ggml_sum_rows(ctx, ggml_mul(ctx, sk, gk)) // [1, G]
                 : ggml_reshape_2d(ctx, ggml_sum(ctx, ggml_mul(ctx, sk, gk)), 1, 1);
    rec = ggml_mul(ctx, rec, ggml_scale(ctx, ivol, 4.0f * M_PI));
  }
  return per_atom(ggml_scale(ctx, ggml_sub(ctx, rec, selfE), static_cast<float>(KE)));
}

} // namespace mlipcpp::runtime
