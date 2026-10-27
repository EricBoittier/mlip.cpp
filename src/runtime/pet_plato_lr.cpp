#include "pet_plato_lr.h"

#include "core/gguf_loader.h"
#include "graph_ir.h"
#include "models/plato/plato.h"

#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
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
  all_.clear(), groups_.clear();
  frac_ = pair_r_ = charges_ = key_log_keep_ = nullptr;
  strain = nullptr, pair_i_ = pair_j_ = nullptr, atom_inv_cell_t_ = nullptr, open_mask_ = nullptr;
  n_pairs_ = 0, any_open_ = false;
  const int G = static_cast<int>(st_.size());
  if (G < 1) throw std::runtime_error("Plato-PET-LR: no structure");
  auto input = [&](ggml_type t, int64_t a, int64_t b = 1, int64_t c = 1, const char *name = "") {
    ggml_tensor *x = ggml_new_tensor_3d(ctx, t, a, b, c);
    ggml_set_name(x, name);
    ggml_set_input(x);
    all_.push_back(x);
    return x;
  };
  positions = input(GGML_TYPE_F32, 3, n_, 1, "plato_lr.positions");
  if (grad) ggml_set_param(positions);
  h_sidx_.assign(n_, 0), h_onehot_.assign(size_t(n_) * G, 0.0f), h_invc_.assign(G, 0.0f), h_qpa_.assign(n_, 0.0f);
  h_omask_.assign(n_, 0.0f);
  for (int g = 0; g < G; g++) {
    const auto &s = st_[g];
    const int na = s.a1 - s.a0;
    if (na <= 0 || s.a0 < 0 || s.a1 > n_) throw std::runtime_error("Plato-PET-LR: bad structure atom range");
    h_invc_[g] = 1.0f / na;
    for (int i = s.a0; i < s.a1; i++) {
      h_sidx_[i] = g, h_onehot_[size_t(g) * n_ + i] = 1.0f, h_qpa_[i] = static_cast<float>(s.charge / na);
      if (!s.periodic) h_omask_[i] = 1.0f;
    }
    any_open_ |= !s.periodic;
  }
  struct_index_ = input(GGML_TYPE_I32, n_, 1, 1, "plato_lr.struct_index");
  onehot_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, n_, G, 1, "plato_lr.onehot"), n_, G);
  inv_count_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, G, 1, "plato_lr.inv_count"), 1, G);
  charge_per_atom_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, n_, 1, "plato_lr.charge_per_atom"), 1, n_);
  if (!spec_.long_range()) return;

  if (any_open_) { // off-diagonal pairs within each open structure
    h_pi_.clear(), h_pj_.clear();
    for (const auto &s : st_)
      if (!s.periodic)
        for (int i = s.a0; i < s.a1; i++)
          for (int j = s.a0; j < s.a1; j++)
            if (i != j) h_pi_.push_back(i), h_pj_.push_back(j);
    n_pairs_ = static_cast<int64_t>(h_pi_.size());
    if (n_pairs_ > 0) {
      pair_i_ = input(GGML_TYPE_I32, n_pairs_, 1, 1, "plato_lr.pair_i");
      pair_j_ = input(GGML_TYPE_I32, n_pairs_, 1, 1, "plato_lr.pair_j");
    }
    open_mask_ = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, n_, 1, "plato_lr.open_mask"), 1, n_);
  }
  bool any_periodic = false;
  for (const auto &s : st_) any_periodic |= s.periodic;
  if (!any_periodic) return;

  // periodic structures: inverse cells, then groups of equal meshes
  std::vector<std::array<std::array<double, 3>, 3>> inv(G);
  std::vector<double> vol(G, 0.0);
  h_aict_.assign(9 * size_t(n_), 0.0f);
  auto row = [&](int g, int a, float *r) {
    for (int b = 0; b < 3; b++) r[b] = static_cast<float>(st_[g].cell[a][b]);
    return r;
  };
  auto rnorm = [&](int g, int a) {
    double x = 0;
    for (int b = 0; b < 3; b++) x += st_[g].cell[a][b] * st_[g].cell[a][b];
    return std::sqrt(x);
  };
  int max_atoms = 0;
  for (const auto &s : st_) max_atoms = std::max(max_atoms, s.a1 - s.a0);
  const bool pme = spec_.les && (spec_.les_method == "pme" || (spec_.les_method == "auto" && max_atoms >= spec_.les_pme_atoms));
  if (pme && spec_.num_charges != 1) throw std::runtime_error("Plato-PET-LR: LES PME needs one charge channel");
  const double h_pme = spec_.les_pme_spacing * spec_.les_sigma;
  for (int g = 0; g < G; g++) {
    if (!st_[g].periodic) continue;
    double Ci[3][3], det;
    inverse3(st_[g].cell, Ci, det);
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) inv[g][a][b] = Ci[a][b];
    vol[g] = std::abs(det);
    for (int i = st_[g].a0; i < st_[g].a1; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) h_aict_[9 * size_t(i) + j * 3 + k] = static_cast<float>(Ci[k][j]);
    int mesh[3] = {0, 0, 0}, lmesh[3] = {0, 0, 0};
    float r[3];
    for (int a = 0; a < 3; a++) {
      if (spec_.mesh) mesh[a] = plato::mesh_points(row(g, a, r), spec_.mesh_spacing);
      if (pme) lmesh[a] = plato::pme_mesh_points(row(g, a, r), h_pme, spec_.les_pme_order);
    }
    Group *grp = nullptr;
    for (auto &q : groups_)
      if (std::equal(mesh, mesh + 3, q.mesh) && std::equal(lmesh, lmesh + 3, q.les_mesh)) grp = &q;
    if (!grp) {
      groups_.emplace_back();
      grp = &groups_.back();
      std::copy(mesh, mesh + 3, grp->mesh), std::copy(lmesh, lmesh + 3, grp->les_mesh);
    }
    grp->structs.push_back(g);
  }
  atom_inv_cell_t_ = input(GGML_TYPE_F32, 3, 3, n_, "plato_lr.atom_inv_cell_t");
  auto freq = [](int i, int M) { return i <= (M - 1) / 2 ? i : i - M; };
  for (Group &q : groups_) {
    const int Gq = q.size();
    q.h_bidx.assign(n_, 0), q.h_mask.assign(n_, 0.0f), q.h_invc.resize(Gq), q.h_invv.resize(Gq);
    for (int b = 0; b < Gq; b++) {
      const auto &s = st_[q.structs[b]];
      q.h_invc[b] = 1.0f / (s.a1 - s.a0), q.h_invv[b] = static_cast<float>(1.0 / vol[q.structs[b]]);
      for (int i = s.a0; i < s.a1; i++) q.h_bidx[i] = b, q.h_mask[i] = 1.0f;
    }
    q.bidx = input(GGML_TYPE_I32, n_, 1, 1, "plato_lr.group_index");
    q.mask = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, n_, 1, "plato_lr.group_mask"), 1, n_);
    q.inv_count = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, Gq, 1, "plato_lr.group_inv_count"), 1, Gq);
    q.inv_volume = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, Gq, 1, "plato_lr.inv_volume"), 1, Gq);
    q.strain = input(GGML_TYPE_F32, 3, 3, Gq, "plato_lr.strain");
    if (grad) ggml_set_param(q.strain);
    if (spec_.mesh) {
      const int M0 = q.mesh[0], M1 = q.mesh[1], M2 = q.mesh[2], Mh2 = M2 / 2 + 1;
      const int64_t Mh = int64_t(M0) * M1 * Mh2;
      q.h_mk.assign(3 * size_t(Mh) * Gq, 0.0f), q.h_mpad.assign(Mh, 0.0f), q.h_mdec.assign(Mh, 0.0f);
      for (int i0 = 0; i0 < M0; i0++)
        for (int i1 = 0; i1 < M1; i1++)
          for (int i2 = 0; i2 < Mh2; i2++) {
            const int64_t e = (int64_t(i0) * M1 + i1) * Mh2 + i2;
            const double m[3] = {double(freq(i0, M0)), double(freq(i1, M1)), double(i2)};
            for (int b = 0; b < Gq; b++) {
              const auto &I = inv[q.structs[b]];
              for (int c = 0; c < 3; c++)
                q.h_mk[3 * (size_t(b) * Mh + e) + c] = static_cast<float>(2 * M_PI * (m[0] * I[c][0] + m[1] * I[c][1] + m[2] * I[c][2]));
            }
            double dc = 1;
            for (int a = 0; a < 3; a++) {
              const double f = (4 + 2 * std::cos(2 * M_PI * m[a] / q.mesh[a])) / 6;
              dc *= f * f;
            }
            const bool zero = i0 == 0 && i1 == 0 && i2 == 0;
            q.h_mpad[e] = zero ? 1.0f : 0.0f;
            q.h_mdec[e] = zero ? 0.0f : static_cast<float>(1.0 / dc);
          }
      q.mesh_k = input(GGML_TYPE_F32, 3, Mh, Gq, "plato_lr.mesh_k");
      q.mesh_k0_pad = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, Mh, 1, "plato_lr.mesh_k0_pad"), 1, Mh);
      q.mesh_deconv = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 1, Mh, 1, "plato_lr.mesh_deconv"), 1, Mh);
    }
    if (spec_.les && pme) {
      q.h_tab = plato::pme_axis_table(q.les_mesh, spec_.les_pme_order);
      q.h_lK.assign(9 * size_t(Gq), 0.0f);
      for (int b = 0; b < Gq; b++)
        for (int c = 0; c < 3; c++)
          for (int k = 0; k < 3; k++) q.h_lK[9 * size_t(b) + 3 * c + k] = static_cast<float>(2 * M_PI * inv[q.structs[b]][k][c]);
      q.les_K = input(GGML_TYPE_F32, 3, 3, Gq, "plato_lr.les_K");
      q.les_tab = input(GGML_TYPE_F32, static_cast<int64_t>(q.h_tab.size()), 1, 1, "plato_lr.les_tab");
    } else if (spec_.les) { // the union over the group of the integer k sets (each structure's window drops the rest)
      const double kc = spec_.les_kcut / spec_.les_sigma;
      int nmax[3] = {0, 0, 0};
      for (int g : q.structs)
        for (int a = 0; a < 3; a++) nmax[a] = std::max(nmax[a], static_cast<int>(std::floor(kc * rnorm(g, a) / (2 * M_PI))));
      std::vector<std::array<int, 3>> ns;
      for (int a = -nmax[0]; a <= nmax[0]; a++)
        for (int b = -nmax[1]; b <= nmax[1]; b++)
          for (int c = -nmax[2]; c <= nmax[2]; c++) {
            if (!(a > 0 || (a == 0 && b > 0) || (a == 0 && b == 0 && c > 0))) continue;
            bool any = false;
            for (size_t t = 0; t < q.structs.size() && !any; t++) {
              const auto &I = inv[q.structs[t]];
              double k2 = 0;
              for (int d = 0; d < 3; d++) {
                const double k = 2 * M_PI * (a * I[d][0] + b * I[d][1] + c * I[d][2]);
                k2 += k * k;
              }
              any = k2 < kc * kc;
            }
            if (any) ns.push_back({a, b, c});
          }
      if (ns.empty()) ns.push_back({1, 0, 0}); // a tiny cell: one k, outside the window (weight 0)
      q.nk = static_cast<int64_t>(ns.size());
      q.h_ln.clear(), q.h_lk.assign(3 * size_t(q.nk) * Gq, 0.0f);
      for (int64_t e = 0; e < q.nk; e++) {
        for (int d = 0; d < 3; d++) q.h_ln.push_back(static_cast<float>(2 * M_PI * ns[e][d]));
        for (int b = 0; b < Gq; b++) {
          const auto &I = inv[q.structs[b]];
          for (int d = 0; d < 3; d++)
            q.h_lk[3 * (size_t(b) * q.nk + e) + d] =
                static_cast<float>(2 * M_PI * (ns[e][0] * I[d][0] + ns[e][1] * I[d][1] + ns[e][2] * I[d][2]));
        }
      }
      q.les_n = ggml_reshape_2d(ctx, input(GGML_TYPE_F32, 3, q.nk, 1, "plato_lr.les_n"), 3, q.nk);
      q.les_k = input(GGML_TYPE_F32, 3, q.nk, Gq, "plato_lr.les_k");
    }
  }
  strain = groups_[0].strain;
}

bool PetPlatoLR::refresh(const std::vector<PetPlatoLRStructure> &structures) {
  PetPlatoLR t;
  ggml_context *c = ggml_init({ggml_tensor_overhead() * 512, nullptr, true});
  try {
    t.create(c, spec_, structures, n_, false);
  } catch (...) {
    ggml_free(c);
    throw;
  }
  ggml_free(c);
  bool same = t.n_pairs_ == n_pairs_ && t.any_open_ == any_open_ && t.groups_.size() == groups_.size() &&
              t.st_.size() == st_.size() && t.h_pi_ == h_pi_ && t.h_sidx_ == h_sidx_;
  for (size_t k = 0; same && k < groups_.size(); k++) {
    const Group &a = groups_[k], &b = t.groups_[k];
    same = a.structs == b.structs && a.nk == b.nk && std::equal(a.mesh, a.mesh + 3, b.mesh) &&
           std::equal(a.les_mesh, a.les_mesh + 3, b.les_mesh) && a.h_bidx == b.h_bidx;
  }
  if (!same) return false;
  st_ = t.st_;
  h_onehot_ = t.h_onehot_, h_invc_ = t.h_invc_, h_qpa_ = t.h_qpa_, h_aict_ = t.h_aict_, h_omask_ = t.h_omask_;
  for (size_t k = 0; k < groups_.size(); k++) {
    Group &a = groups_[k];
    const Group &b = t.groups_[k];
    a.h_mask = b.h_mask, a.h_invc = b.h_invc, a.h_invv = b.h_invv, a.h_mk = b.h_mk, a.h_mpad = b.h_mpad;
    a.h_mdec = b.h_mdec, a.h_lk = b.h_lk, a.h_ln = b.h_ln, a.h_lK = b.h_lK, a.h_tab = b.h_tab;
  }
  return true;
}

std::vector<ggml_tensor *> PetPlatoLR::strains() const {
  std::vector<ggml_tensor *> out;
  for (const auto &q : groups_) out.push_back(q.strain);
  return out;
}

void PetPlatoLR::upload(const float *pos) const {
  auto set = [](ggml_tensor *t, const void *d) { // (reshapes: through their source, which a graph may not use)
    if (t && t->view_src && ggml_nbytes(t) == ggml_nbytes(t->view_src)) t = t->view_src;
    if (t && t->buffer) ggml_backend_tensor_set(t, d, 0, ggml_nbytes(t)); // (unused by the graph: not allocated)
  };
  set(positions, pos);
  set(struct_index_, h_sidx_.data());
  set(onehot_, h_onehot_.data()), set(inv_count_, h_invc_.data()), set(charge_per_atom_, h_qpa_.data());
  if (pair_i_) set(pair_i_, h_pi_.data()), set(pair_j_, h_pj_.data());
  if (open_mask_) set(open_mask_, h_omask_.data());
  if (atom_inv_cell_t_) set(atom_inv_cell_t_, h_aict_.data());
  for (const Group &q : groups_) {
    std::vector<float> z(ggml_nelements(q.strain), 0.0f);
    set(q.strain, z.data());
    set(q.bidx, q.h_bidx.data()), set(q.mask, q.h_mask.data()), set(q.inv_count, q.h_invc.data());
    set(q.inv_volume, q.h_invv.data());
    if (q.mesh_k) set(q.mesh_k, q.h_mk.data()), set(q.mesh_k0_pad, q.h_mpad.data()), set(q.mesh_deconv, q.h_mdec.data());
    if (q.les_n) set(q.les_n, q.h_ln.data()), set(q.les_k, q.h_lk.data());
    if (q.les_K) set(q.les_K, q.h_lK.data()), set(q.les_tab, q.h_tab.data());
  }
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
  if (padding_mask_ && padding_mask_->ne[0] + 1 == L && padding_mask_->ne[1] == N &&
      !std::getenv("MLIPCPP_PLATO_FRAME_PAD_FINITE")) {
    // log(1 - padded) per key, the centre kept: -inf on the padded keys (no gradient)
    if (!key_log_keep_) {
      ggml_tensor *center = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, N), 1.0f);
      ggml_tensor *keep = ggml_concat(ctx, center, ggml_scale_bias(ctx, padding_mask_, -1.0f, 1.0f), 0); // [L, N]
      key_log_keep_ = ggml_reshape_4d(ctx, ggml_log(ctx, keep), L, 1, 1, N);
    }
    bias = ggml_add(ctx, bias, key_log_keep_);
  }
  // [Ch, L, H, N] -> [D, 1, L, N] (heads of a token together, as ggml_pet_attn reads them)
  auto tokens = [&](ggml_tensor *x) { return ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)), D, 1, L, N); };
  ggml_tensor *vt = tokens(v);
  auto half = [&](ggml_tensor *x, int s) {
    return ggml_view_4d(ctx, x, C2, L, H, N, x->nb[1], x->nb[2], x->nb[3], s * C2 * ggml_element_size(x));
  };
  static const bool loop = std::getenv("MLIPCPP_PLATO_FRAME_LOOP") != nullptr;
  if (!loop) {
    // every frame at once: the frames as an outer block of the sequence axis, N F sequences in
    // one fused attention (fewer, larger kernels than one attention per frame)
    const int64_t NF = N * F;
    ggml_tensor *Rw = mm_f32(ctx, ggml_reshape_2d(ctx, cont(ctx, frames), 3, 3 * F), omega); // [3F, C2]: (R_f omega_c)_a
    Rw = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Rw, 3, F, C2), 0, 2, 1, 3)), 3, C2 * F);
    ggml_tensor *ang = mm_f32(ctx, Rw, pos);                                                   // [C2 F, L N]
    ang = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, ang, C2, F, L * N), 0, 2, 1, 3)), C2, 1, L, NF);
    ggml_tensor *c = ggml_cos(ctx, ang), *s = ggml_sin(ctx, ang);
    auto heads = [&](ggml_tensor *x) { // [Ch, L, H, N] -> [Ch, H, L, N F], the frames' copies
      ggml_tensor *t = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
      return ggml_repeat(ctx, t, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Ch, H, L, NF));
    };
    auto rope = [&](ggml_tensor *x) { // [Ch, H, L, NF] -> [D, 1, L, NF]
      ggml_tensor *x1 = ggml_view_4d(ctx, x, C2, H, L, NF, x->nb[1], x->nb[2], x->nb[3], 0);
      ggml_tensor *x2 = ggml_view_4d(ctx, x, C2, H, L, NF, x->nb[1], x->nb[2], x->nb[3], C2 * ggml_element_size(x));
      ggml_tensor *y = ggml_concat(ctx, ggml_sub(ctx, ggml_mul(ctx, x1, c), ggml_mul(ctx, x2, s)),
                                   ggml_add(ctx, ggml_mul(ctx, x1, s), ggml_mul(ctx, x2, c)), 0);
      return ggml_reshape_4d(ctx, y, D, 1, L, NF);
    };
    ggml_tensor *vF = ggml_repeat(ctx, vt, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, L, NF));
    ggml_tensor *xt = ggml_concat(ctx, ggml_concat(ctx, rope(heads(q)), rope(heads(k)), 1), vF, 1); // [D, 3, L, NF]
    ggml_tensor *bF = ggml_repeat(ctx, bias, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, L, L, 1, NF));
    ggml_tensor *r = ggml_pet_attn(ctx, xt, bF, static_cast<int>(H), scale);                            // [D + H, L, NF]
    ggml_tensor *o = nullptr;
    for (int f = 0; f < F; f++) {
      ggml_tensor *of = ggml_view_3d(ctx, r, D, L, N, r->nb[1], r->nb[2], f * N * r->nb[2]);
      o = o ? ggml_add(ctx, o, of) : ggml_cont(ctx, of);
    }
    if (F > 1) o = ggml_scale(ctx, o, 1.0f / F);
    return ggml_permute(ctx, ggml_reshape_4d(ctx, o, Ch, H, L, N), 0, 2, 1, 3); // [Ch, L, H, N]
  }
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
  static const bool loop = std::getenv("MLIPCPP_PLATO_FRAME_LOOP") != nullptr;
  if (!loop) { // every frame in one product: A [3, 3F], A[j, i + 3f] = R_f[j][i]
    ggml_tensor *A = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, frames, 1, 0, 2, 3)), 3, 3 * F);
    ggml_tensor *rf = ggml_reshape_2d(ctx, mm_f32(ctx, A, r), 3, F * E);       // R_f^T r, frame-fastest
    ggml_tensor *t = ggml_reshape_3d(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, W, rf)), d, F, E);
    ggml_tensor *o = nullptr;
    for (int f = 0; f < F; f++) {
      ggml_tensor *tf = ggml_view_2d(ctx, t, d, E, t->nb[2], f * t->nb[1]);
      o = o ? ggml_add(ctx, o, tf) : ggml_cont(ctx, tf);
    }
    if (F > 1) o = ggml_scale(ctx, o, 1.0f / F);
    return ggml_reshape_3d(ctx, o, d, M, n);
  }
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

ggml_tensor *PetPlatoLR::pearson(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b) const {
  const int G = n_struct();
  auto smean = [&](ggml_tensor *x) { // per-structure mean of x [1, n], on every atom: [1, n]
    ggml_tensor *m = ggml_mul(ctx, ggml_reshape_2d(ctx, mm_f32(ctx, onehot_, ggml_cont(ctx, ggml_transpose(ctx, x))), 1, G), inv_count_);
    return ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_2d(ctx, m, 1, G), struct_index_), 1, n_);
  };
  auto ssum = [&](ggml_tensor *x) { // per-structure sum of x [1, n]: [1, G]
    return ggml_reshape_2d(ctx, mm_f32(ctx, onehot_, ggml_cont(ctx, ggml_transpose(ctx, x))), 1, G);
  };
  ggml_tensor *ac = ggml_sub(ctx, a, smean(a)), *bc = ggml_sub(ctx, b, smean(b));
  ggml_tensor *var = ggml_mul(ctx, ssum(ggml_sqr(ctx, ac)), ssum(ggml_sqr(ctx, bc)));
  return ggml_div(ctx, ssum(ggml_mul(ctx, ac, bc)), ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0f, 1e-30f)));
}

ggml_tensor *PetPlatoLR::inv_volume(ggml_context *ctx, const Group &g) {
  const int Gq = g.size();
  auto e = [&](int k) { return ggml_cont(ctx, ggml_view_2d(ctx, g.strain, 1, Gq, g.strain->nb[2], k * sizeof(float))); };
  ggml_tensor *tr = ggml_add(ctx, ggml_add(ctx, e(0), e(4)), e(8));
  return ggml_mul(ctx, ggml_scale_bias(ctx, tr, -1.0f, 1.0f), g.inv_volume);
}

// v [C, n], filt.0 (w [1, h], b [h]), filt.2 (w [h, C], b [C]), k_quad [nq], w_quad [nq] -> [C, n]
ggml_tensor *PetPlatoLR::mesh_op(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in) {
  if (in.size() != 7) throw std::runtime_error("PLATO_MESH: 7 inputs");
  ggml_tensor *v = cont(ctx, in[0]);
  const int64_t C = v->ne[0];
  auto filt = [&](ggml_tensor *kk) { // [1, m] -> [C, m]
    return ggml_add(ctx, ggml_mul_mat(ctx, in[3], ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, in[1], kk), in[2]))), in[4]);
  };
  const float sigma = static_cast<float>(param_num(node, "sigma", spec_.mesh_sigma));
  ggml_tensor *m = nullptr;
  auto acc = [&](ggml_tensor *t) { m = m ? ggml_add(ctx, m, t) : t; };
  if (any_open_ && param_num(node, "open_boundary", 1) != 0) {
    ggml_tensor *kq = in[5], *wq = in[6];
    const int64_t nq = kq->ne[0];
    ggml_tensor *wf = ggml_mul(ctx, filt(ggml_reshape_2d(ctx, kq, 1, nq)), ggml_reshape_2d(ctx, wq, 1, nq)); // [C, nq]
    ggml_tensor *A = ggml_cont(ctx, ggml_transpose(ctx, wf));                                                 // [nq, C]
    // i = j: K(0) = sqrt(2 / pi) / sigma + sum_n w_n f(k_n)
    const float k0 = static_cast<float>(std::sqrt(2.0 / M_PI) / sigma);
    ggml_tensor *diag = ggml_scale_bias(ctx, ggml_reshape_2d(ctx, ggml_sum_rows(ctx, A), C, 1), 1.0f, k0); // [C, 1]
    acc(ggml_mul(ctx, ggml_mul(ctx, v, diag), open_mask_));
    if (n_pairs_ > 0) {
      ggml_tensor *r = pair_r(ctx); // [1, P]
      ggml_tensor *kr = ggml_mul(ctx, ggml_repeat(ctx, r, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nq, n_pairs_)),
                                 ggml_reshape_2d(ctx, kq, nq, 1)); // [nq, P]
      ggml_tensor *j0 = ggml_div(ctx, ggml_sin(ctx, kr), kr);
      ggml_tensor *K = ggml_add(ctx, mm_f32(ctx, A, j0),
                                ggml_div(ctx, ggml_plato_erf(ctx, ggml_scale(ctx, r, 1.0f / (std::sqrt(2.0f) * sigma))), r)); // [C, P]
      acc(ggml_get_rows_back(ctx, ggml_mul(ctx, K, ggml_get_rows(ctx, v, pair_j_)), pair_i_, v));
    }
  }
  // periodic groups (runtime/plato_graph.cpp's mesh coupling), times the mesh points
  for (const Group &g : groups_) {
    const int M0 = g.mesh[0], M1 = g.mesh[1], M2 = g.mesh[2], Gq = g.size();
    const int64_t Mh = int64_t(M0) * M1 * (M2 / 2 + 1);
    ggml_tensor *fr = frac(ctx);
    ggml_tensor *vg = ggml_mul(ctx, v, g.mask); // the group's atoms only
    ggml_tensor *rho = ggml_plato_spread_blocked(ctx, vg, fr, g.bidx, Gq, M0, M1, M2);
    ggml_tensor *rk = ggml_plato_fft3(ctx, rho, 0, M0, M1, M2);
    ggml_tensor *k = ggml_sub(ctx, g.mesh_k, mm_f32(ctx, g.strain, g.mesh_k)); // [3, Mh, Gq]
    ggml_tensor *k2 = ggml_sum_rows(ctx, ggml_sqr(ctx, k));                    // [1, Mh, Gq]
    ggml_tensor *k2s = ggml_add(ctx, k2, g.mesh_k0_pad);
    ggml_tensor *base = ggml_div(ctx, ggml_exp(ctx, ggml_scale(ctx, k2, -0.5f * sigma * sigma)), k2s);
    const float npts = static_cast<float>(double(M0) * M1 * M2);
    base = ggml_mul(ctx, ggml_mul(ctx, base, g.mesh_deconv),
                    ggml_reshape_3d(ctx, ggml_scale(ctx, inv_volume(ctx, g), 4.0f * M_PI * npts), 1, 1, Gq));
    ggml_tensor *gk = ggml_mul(ctx, ggml_scale_bias(ctx, filt(ggml_reshape_2d(ctx, ggml_sqrt(ctx, k2s), 1, Mh * Gq)), 1.0f, 1.0f),
                               ggml_reshape_2d(ctx, base, 1, Mh * Gq)); // [C, Mh Gq]
    gk = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, gk, C, Mh, Gq), 1, 0, 2, 3)), 1, Mh, C, Gq);
    rk = ggml_reshape_4d(ctx, ggml_mul(ctx, ggml_reshape_4d(ctx, rk, 2, Mh, C, Gq), gk), rk->ne[0], M1, M0, C * Gq);
    ggml_tensor *phi = ggml_plato_fft3(ctx, rk, 1, M0, M1, M2);
    acc(ggml_mul(ctx, ggml_plato_gather_blocked(ctx, phi, fr, g.bidx, static_cast<int>(C), false), g.mask)); // [C, n]
  }
  return m ? m : ggml_scale(ctx, v, 0.0f);
}

// q [Q, n], charge_unit [1] -> [n]
ggml_tensor *PetPlatoLR::les_op(ggml_context *ctx, const GIRNode &node, const std::vector<ggml_tensor *> &in) {
  if (in.size() != 2) throw std::runtime_error("PLATO_LES: 2 inputs");
  ggml_tensor *q = cont(ctx, in[0]);
  const int64_t Q = q->ne[0], n = q->ne[1];
  const float sigma = static_cast<float>(param_num(node, "sigma", spec_.les_sigma));
  const float kcut = static_cast<float>(param_num(node, "kcut", spec_.les_kcut));
  const bool tcc = param_num(node, "total_charge_constraint", spec_.total_charge_constraint) != 0;
  // neutral per structure (or the structure's total charge / charge_unit)
  ggml_tensor *qt = ggml_cont(ctx, ggml_transpose(ctx, q));                                                    // [n, Q]
  ggml_tensor *mean = ggml_mul(ctx, ggml_cont(ctx, ggml_transpose(ctx, mm_f32(ctx, onehot_, qt))), inv_count_); // [Q, G]
  q = ggml_sub(ctx, q, ggml_get_rows(ctx, mean, struct_index_));
  if (tcc) q = ggml_add(ctx, q, ggml_div(ctx, charge_per_atom_, ggml_repeat(ctx, in[1], charge_per_atom_)));
  charges_ = q;
  ggml_tensor *e = nullptr; // [1, n]
  auto acc = [&](ggml_tensor *t) { e = e ? ggml_add(ctx, e, t) : t; };
  if (n_pairs_ > 0) { // open structures: the erf pair sum, per atom
    ggml_tensor *r = pair_r(ctx);
    ggml_tensor *pair = ggml_div(ctx, ggml_plato_erf(ctx, ggml_scale(ctx, r, 1.0f / (std::sqrt(2.0f) * sigma))), r); // [1, P]
    ggml_tensor *s = ggml_get_rows_back(ctx, ggml_mul(ctx, ggml_get_rows(ctx, q, pair_j_), pair), pair_i_, q);     // [Q, n]
    acc(ggml_scale(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, q, s)), static_cast<float>(0.5 * KE)));
  }
  if (!groups_.empty()) {
    ggml_tensor *fr = frac(ctx);
    // the self energy of the periodic atoms, per atom: q^2 / (sqrt(2 pi) sigma)
    ggml_tensor *pmask = nullptr;
    for (const Group &g : groups_) pmask = pmask ? ggml_add(ctx, pmask, g.mask) : g.mask;
    ggml_tensor *self = ggml_scale(ctx, ggml_mul(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, q)), pmask),
                                   static_cast<float>(KE / (std::sqrt(2.0 * M_PI) * sigma))); // [1, n]
    acc(ggml_scale(ctx, self, -1.0f));
    for (const Group &g : groups_) {
      const int Gq = g.size();
      ggml_tensor *qg = ggml_mul(ctx, q, g.mask); // [Q, n]
      ggml_tensor *ivol = inv_volume(ctx, g);
      ggml_tensor *rec; // [1, Gq]
      if (g.les_mesh[0] > 0) {
        const int M0 = g.les_mesh[0], M1 = g.les_mesh[1], M2 = g.les_mesh[2];
        ggml_tensor *rho = ggml_plato_spread_order(ctx, ggml_reshape_2d(ctx, qg, 1, n), fr, g.bidx, Gq, M0, M1, M2,
                                                   spec_.les_pme_order);
        ggml_tensor *rk = ggml_plato_fft3(ctx, rho, 0, M0, M1, M2);
        ggml_tensor *K = ggml_sub(ctx, g.les_K, mm_f32(ctx, g.strain, g.les_K));
        ggml_tensor *A = ggml_reshape_2d(ctx, mm_f32(ctx, K, K), 9, Gq);
        ggml_tensor *ep = ggml_plato_pme(ctx, rk, A, g.les_tab, M0, M1, M2, sigma, kcut / sigma, 1.0f / sigma); // [M0, Gq]
        rec = ggml_mul(ctx, ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ep), 1, Gq), ggml_scale(ctx, ivol, 2.0f * M_PI));
      } else {
        const int64_t nk = g.nk;
        ggml_tensor *sk; // [nk, Gq] (Q = 1) or [nk, Q] (one structure)
        if (Q == 1) {
          ggml_tensor *r = ggml_plato_ewald(ctx, fr, ggml_reshape_1d(ctx, qg, n), g.bidx, g.les_n, Gq);
          sk = ggml_view_2d(ctx, r, nk, Gq, r->nb[1], 2 * r->nb[2]);
        } else {
          if (Gq != 1) throw std::runtime_error("Plato-PET-LR: batched periodic LES needs one charge channel");
          ggml_tensor *qgt = ggml_cont(ctx, ggml_transpose(ctx, qg)); // [n, Q]
          ggml_tensor *ph = mm_f32(ctx, fr, g.les_n);                 // [n, nk]
          ggml_tensor *Cq = mm_f32(ctx, ggml_cos(ctx, ph), qgt), *Sq = mm_f32(ctx, ggml_sin(ctx, ph), qgt); // [nk, Q]
          sk = ggml_add(ctx, ggml_sqr(ctx, Cq), ggml_sqr(ctx, Sq));
        }
        ggml_tensor *k = ggml_sub(ctx, g.les_k, mm_f32(ctx, g.strain, g.les_k));
        ggml_tensor *k2 = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, k)), nk, Gq);
        const float kc = kcut / sigma, wd = 1.0f / sigma;
        ggml_tensor *xw = ggml_clamp(ctx, ggml_scale_bias(ctx, ggml_sqrt(ctx, k2), 1.0f / wd, -(kc - wd) / wd), 0.0f, 1.0f);
        ggml_tensor *win = ggml_scale_bias(ctx, ggml_cos(ctx, ggml_scale(ctx, xw, static_cast<float>(M_PI))), 0.5f, 0.5f);
        ggml_tensor *gk = ggml_div(ctx, ggml_mul(ctx, win, ggml_exp(ctx, ggml_scale(ctx, k2, -0.5f * sigma * sigma))), k2); // [nk, Gq]
        rec = Q == 1 ? ggml_sum_rows(ctx, ggml_mul(ctx, sk, gk)) : ggml_reshape_2d(ctx, ggml_sum(ctx, ggml_mul(ctx, sk, gk)), 1, 1);
        rec = ggml_mul(ctx, rec, ggml_scale(ctx, ivol, 4.0f * M_PI));
      }
      // E_rec of structure b spread evenly over its atoms
      ggml_tensor *per = ggml_get_rows(ctx, ggml_reshape_2d(ctx, ggml_mul(ctx, rec, g.inv_count), 1, Gq), g.bidx); // [1, n]
      acc(ggml_scale(ctx, ggml_mul(ctx, ggml_reshape_2d(ctx, per, 1, n), g.mask), static_cast<float>(KE)));
    }
  }
  if (!e) return ggml_reshape_1d(ctx, ggml_scale(ctx, ggml_sum_rows(ctx, q), 0.0f), n);
  return ggml_reshape_1d(ctx, e, n);
}

} // namespace mlipcpp::runtime
