#pragma once
// CUDA side of the MD engine: the neighbour list, the model inputs and the
// integrator all stay on the device; only a few scalars come back per step.
//
// Layouts: atom i's candidates (Verlet list at cutoff + skin) are
// [i*K, i*K + count[i]); graph inputs are [n_atoms, M] slots (ggml ne = (M, n)),
// edge vectors [n_atoms, M, 3].

#include <cstdint>

namespace mlipcpp::md::gpu {

struct Shift {
  int8_t s[3];
};

/// Device state owned by the engine (all pointers are device memory).
struct State {
  int n = 0, K = 0, M = 0;
  double *x = nullptr, *v = nullptr, *x_ref = nullptr; // [n, 3]
  double *inv_mass = nullptr;                          // [n], 1 / amu
  int32_t *species = nullptr;                          // [n], model species index
  float *nc_scale = nullptr;                           // [n], per-species force scale
  double cell[9] = {0}; // rows are lattice vectors
  int pbc[3] = {0, 0, 0}, images[3] = {0, 0, 0};

  // Verlet list
  int32_t *cand_j = nullptr, *cand_rev = nullptr, *count = nullptr; // [n*K], [n*K], [n]
  Shift *cand_s = nullptr;                                          // [n*K]
  // per step
  double *cand_v = nullptr, *cand_d = nullptr; // [n*K, 3], [n*K]
  int32_t *cand_slot = nullptr, *nkept = nullptr;
  double *r = nullptr, *root = nullptr, *dn_root = nullptr; // [n] adaptive radii
  int8_t *clamped = nullptr;
  double *dEdr = nullptr;  // [n] dE / d atomic cutoff
  double *chain = nullptr; // [n*K, 3]
  double *f_cons = nullptr, *f_nc = nullptr; // [n, 3]
  int32_t *scalars = nullptr; // [0] overflow (list), [1] max kept, [2] max displacement^2 (float bits)
};

/// Model inputs and outputs: device pointers into the ggml tensors.
struct Graph {
  int32_t *species = nullptr, *neighbor_species = nullptr, *reverse = nullptr;
  float *edge_vectors = nullptr, *padding_mask = nullptr, *cutoff_values = nullptr;
  const float *energy = nullptr, *nc = nullptr;        // [n], [n, 3] (unscaled)
  const float *grad_ev = nullptr, *grad_cv = nullptr;  // dE/d edge_vectors, dE/d cutoff_values
};

struct Cutoff {
  double cutoff = 0, skin = 0;
  double n_target = 0, width = 1; // adaptive (n_target 0: fixed cutoff)
};

void build_list(State &s, const Cutoff &c);          // sets scalars[0] on overflow
void prepare_inputs(State &s, const Cutoff &c, const Graph &g); // edges, radii, pack; scalars[1] = max kept
void conservative_forces(State &s, const Cutoff &c, const Graph &g, double energy_scale);
void nc_forces(State &s, const Graph &g);
void kick(State &s, const double *f1, double a1, const double *f2, double a2, double dt); // v += dt (a1 f1 + a2 f2) / m
void drift(State &s, double dt);                     // x += dt v; scalars[2] = max |x - x_ref|^2
void reset_scalars(State &s);
void synchronize();

} // namespace mlipcpp::md::gpu
