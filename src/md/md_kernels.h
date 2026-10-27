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

struct CellList; // build_list's binning buffers (md_kernels.cu)
struct GridScratch; // the "grid" adaptive cutoff's per-atom probes and per-edge d r / d d (md_kernels.cu)

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
  int32_t *offsets = nullptr; // [n + 1] ragged layout: exclusive scan of nkept (count_kept)
  double *r = nullptr, *root = nullptr, *dn_root = nullptr; // [n] adaptive radii
  int8_t *clamped = nullptr;
  double *dEdr = nullptr;  // [n] dE / d atomic cutoff
  double *chain = nullptr; // [n*K, 3]
  double *f_cons = nullptr, *f_nc = nullptr; // [n, 3]
  int32_t *scalars = nullptr; // [0] overflow (list), [1] max kept, [2] max displacement^2 (float bits)
  CellList *cells = nullptr;  // made by build_list when it bins the atoms; free_cells
  GridScratch *grid = nullptr; // made by the "grid" radii; free_cells
  // the CUDA stream (cudaStream_t) every function below runs on; nullptr: the
  // legacy default stream. Downloads (overflow, bounding box, list_valid)
  // synchronise this stream only.
  void *stream = nullptr;
};

/// Model inputs and outputs: device pointers into the ggml tensors.
struct Graph {
  int32_t *species = nullptr, *neighbor_species = nullptr, *reverse = nullptr;
  float *edge_vectors = nullptr, *padding_mask = nullptr, *cutoff_values = nullptr;
  const float *energy = nullptr, *nc = nullptr;        // [n], [n, 3] (unscaled)
  const float *grad_ev = nullptr, *grad_cv = nullptr;  // dE/d edge_vectors, dE/d cutoff_values
  // ragged layout (offsets set): edge rows [offsets[i], offsets[i + 1]) of atom i, then
  // padding rows up to e_cap; centers [e_cap] is each row's atom
  int32_t *offsets = nullptr, *centers = nullptr;
  int64_t e_cap = 0;
};

struct Cutoff {
  double cutoff = 0, skin = 0;
  double n_target = 0, width = 1; // adaptive (n_target 0: fixed cutoff)
  bool grid = false;              // metatrain's "grid" method (else "solver"), core/adaptive_cutoff.h
};

/// Candidates within cutoff + skin of each atom, in (j, shift) order; sets
/// scalars[0] on overflow. Atoms are binned into cells of at least that size
/// when the cell is periodic in all three directions (each at least three
/// cells across) or in none, and n >= 256; otherwise every pair and image is
/// tested. MLIPCPP_GPU_NL=cells|pairs forces one of the two (cells when
/// possible).
void build_list(State &s, const Cutoff &c);
/// True when the list of the last build_list still holds for s.x: wraps s.x into the
/// cell as build_list did, and no atom moved more than half the skin since.
bool list_valid(State &s, double skin);
void free_cells(State &s); // and the grid scratch
void prepare_inputs(State &s, const Cutoff &c, const Graph &g); // edges, radii, pack; scalars[1] = max kept
/// Ragged layout, in two steps around sizing the graph: count_kept sets nkept, s.offsets
/// (offsets[n] = total kept edges) and scalars[1] = max kept; pack_ragged fills the graph.
void count_kept(State &s, const Cutoff &c);
void pack_ragged(State &s, const Cutoff &c, const Graph &g);
/// pack_ragged without knowing on the host whether the graph is large enough:
/// flags[0] = 1 when the kept edges (s.offsets[n]) pass g.e_cap or a sequence
/// (1 + scalars[1]) passes max_len; the graph then gets empty sequences (g.offsets
/// all 0) and the packing is skipped, so nothing is written out of bounds.
void pack_ragged_guarded(State &s, const Cutoff &c, const Graph &g, int max_len, int32_t *flags);
/// after prepare_inputs: flags[0] = 1 when an atom keeps more than s.M edges
void slots_guard(State &s, int32_t *flags);
/// skip (device, may be null): when *skip is set, the kernels that would read
/// past the graph's rows return at once (their outputs are then garbage)
void conservative_forces(State &s, const Cutoff &c, const Graph &g, double energy_scale,
                         const int32_t *skip = nullptr);
/// Per-atom virials sum over edges of v (x) dE/dv, symmetrised, [n, 6] as
/// xx yy zz yz xz xy, unscaled; w is device memory. After conservative_forces.
void virial(State &s, const Cutoff &c, const Graph &g, double *w, const int32_t *skip = nullptr);

// ---- an engine's own device arrays (LAMMPS's KOKKOS package) ----
/// s.x[i] = x(i) (+ shift[i], if given), with x(i, c) at x[i * xs0 + c * xs1];
/// s.species[i] = type_species[types[i]], and *bad = 1 for a type outside
/// [0, n_types] or without a species (type_species -1).
void load_positions(State &s, const double *x, int64_t xs0, int64_t xs1, const double *shift,
                    const int32_t *types, const int32_t *type_species, int n_types, int32_t *bad);
/// shift[i] = s.x[i] - x(i): what build_list's wrap added to each atom
void store_shift(State &s, const double *x, int64_t xs0, int64_t xs1, double *shift);
/// scalars[2] = max |x - x_ref|^2 (float bits), on the device only
void displacement(State &s);
/// Totals in one block: out[0] = energy_scale sum e + sum type_energy[types],
/// out[1..6] = -energy_scale sum w (if w), out[7] = flags[0], out[8] = total
/// (*total, or -1), out[9] = scalars[1], out[10] = scalars[2] as a float,
/// out[11] = flags[1]; flags[2] = 1 when flags[0] and flags[1] are 0 and
/// scalars[2] <= max_d2 (the step is good).
void totals(State &s, const float *e, const int32_t *types, const double *type_energy, int n_types,
            double energy_scale, const double *w, const int32_t *total, double max_d2, int32_t *flags, double *out);
/// f(i) += s.f_cons[i] when *ok, with f(i, c) at f[i * fs0 + c * fs1]
void add_forces(State &s, double *f, int64_t fs0, int64_t fs1, const int32_t *ok);
void nc_forces(State &s, const Graph &g);
void kick(State &s, const double *f1, double a1, const double *f2, double a2, double dt); // v += dt (a1 f1 + a2 f2) / m
void drift(State &s, double dt);                     // x += dt v; scalars[2] = max |x - x_ref|^2
void reset_scalars(State &s);
void synchronize();

} // namespace mlipcpp::md::gpu
