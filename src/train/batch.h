#pragma once

#include "train/dataset.h"

#include <array>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace mlipcpp::train {

/// Physical-to-model target transform, fitted on the training set exactly as
/// metatrain does: energy E -> (E - Σ_i c_{Z_i}) / s_E, force F_i -> F_i / s_{Z_i}.
struct TargetScaling {
  std::map<int, double> composition; // Z -> eV, least-squares fit
  double energy_scale = 1.0;         // RMS of (E - comp) / N
  std::map<int, double> force_scale; // Z -> RMS of force components

  double comp(const int32_t *Z, int n) const;
  double fscale(int Z) const {
    auto it = force_scale.find(Z);
    return it == force_scale.end() ? 1.0 : it->second;
  }

  static TargetScaling fit(const MemmapSplit &train);
};

/// Uniformly random rotation, or improper rotation with probability 1/2
/// (metatomic.o3.random_transformations with add_inversions=True).
std::array<float, 9> random_o3(std::mt19937_64 &rng);

/// Several structures packed as one disjoint graph, in the layout of the
/// exported PET graph (forces mode) plus the loss inputs. Arrays are
/// row-major in PyTorch order, i.e. directly the bytes of the ggml tensors.
struct Batch {
  int n_atoms = 0, n_systems = 0, max_neighbors = 0; // padded shape (tiles)
  int n_real_systems = 0;                            // structures actually packed
  std::vector<int64_t> ids;                          // [n_real_systems] split index of each

  std::vector<int32_t> species;            // [n_atoms] species index
  std::vector<int32_t> neighbor_species;   // [n_atoms, max_nb]
  std::vector<float> edge_vectors;         // [n_atoms, max_nb, 3]
  std::vector<float> padding_mask;         // [n_atoms, max_nb], 1 = padded
  std::vector<int32_t> reverse_neighbor_index; // [n_atoms * max_nb]
  std::vector<float> cutoff_values;        // [n_atoms, max_nb]
  std::vector<int32_t> neighbor_atom;      // [n_atoms, max_nb]: the neighbour's batch index (-1: padded)
  // nl.taper: every structure's TaperEdges, in batch atom indices, vectors rotated
  std::vector<int32_t> taper_center, taper_neighbor;
  std::vector<float> taper_vector, taper_jacobian; // [E_t, 3], [E_t]

  std::vector<float> atom_weight;   // [n_systems, n_atoms]: 1/N_s if atom in s
  std::vector<float> energy_target; // [n_systems]: (E - comp) / (s_E N_s)
  std::vector<float> force_target;  // [n_atoms, 3]: R F / s_Z
  std::vector<float> force_scale;   // [n_atoms]: s_Z, for physical metrics (0: padding)
  std::vector<float> force_mask;    // [n_atoms]: 1 real atom, 0 padding atom
  std::vector<int32_t> n_per_system;
  // charge and spin multiplicity of each atom's structure (padding atoms: 0, 1),
  // the conditioned model's table rows are made from them at upload
  std::vector<int32_t> atom_charge, atom_spin; // [n_atoms]

  // Loss normalisation: 1/S and 1/(3N) of the *whole* step batch, so that
  // micro-batch losses (and gradients) sum to the full-batch ones.
  float energy_norm = 0.0f, force_norm = 0.0f;
};

/// Fixed-shape tiles: every micro-batch is padded up to a point of a
/// geometric grid (atoms x neighbours, systems fixed), so a training run only
/// meets a few dozen graph shapes, each built once and replayed (CUDA graphs).
/// Padding atoms have no neighbours, no system and no force loss.
struct Tiling {
  double nb_ratio = 0;   // neighbour grid ratio (0: no tiling)
  double atom_ratio = 2; // atom grid ratio
  int systems = 0;       // systems per tile (the per-GPU batch size)

  bool enabled() const { return nb_ratio > 1; }
  /// Smallest grid point >= x: lo, then x*ratio rounded up to a multiple of 8.
  static int bucket(int x, int lo, double ratio);
};

/// Pack `ids` of `split`. `rotations` is empty (no augmentation) or holds one
/// 3x3 row-major matrix per structure, applied to edge vectors and forces.
/// Shapes are padded up to `pad_atoms` / `pad_neighbors` / `pad_systems` (0:
/// exact).
Batch make_batch(const MemmapSplit &split, const NeighborCache &nl,
                 const std::vector<int64_t> &ids,
                 const std::vector<std::array<float, 9>> &rotations,
                 const std::map<int, int> &species_to_index,
                 const TargetScaling &scaling, float cutoff, int pad_atoms = 0,
                 int pad_neighbors = 0, int pad_systems = 0);

/// The batch `ids` as micro-batches of structures with similar neighbour
/// counts (all atoms of a micro-batch are padded to its largest count). A
/// micro-batch closes when its padded neighbour count would grow by more than
/// `growth` or its attention size n_atoms*(max_nb+1)^2 would exceed `budget`.
/// Losses are normalised by the whole batch, so the summed gradient equals
/// the full-batch gradient.
std::vector<Batch> make_microbatches(const MemmapSplit &split, const NeighborCache &nl,
                                     const std::vector<int64_t> &ids,
                                     const std::vector<std::array<float, 9>> &rotations,
                                     const std::map<int, int> &species_to_index,
                                     const TargetScaling &scaling, float cutoff,
                                     double budget = 5e7, double growth = 1.5,
                                     const Tiling &tiling = {});

/// Modelled compute time (ms) of a micro-batch of n atoms padded to m
/// neighbours: per micro-batch overhead, attention ~ n (m+1)^2, edges ~ n (m+1).
struct CostModel {
  std::array<double, 3> c = {2.0, 2e-6, 2e-4};
  static std::array<double, 3> features(double n, double m) {
    return {1.0, n * (m + 1) * (m + 1), n * (m + 1)};
  }
  double operator()(double n, double m) const {
    const auto f = features(n, m);
    return c[0] * f[0] + c[1] * f[1] + c[2] * f[2];
  }
};

/// All structures `ids` of a step (n_workers x batch) packed together:
/// sorted by neighbour count and cut into contiguous micro-batches that
/// minimise the modelled total cost (dynamic programming; none above
/// `budget` = n (m+1)^2), then dealt to the workers longest first. The cost
/// cap per micro-batch is the one (of a few) whose deal ends first. No
/// worker is preloaded with the optimizer: it runs after all of them, so a
/// preload only made the others straggle. Loss normalisation is the mean over the whole step:
/// summing every worker's gradient and dividing by n_workers gives the
/// full-batch gradient. A worker may get no micro-batch.
std::vector<std::vector<Batch>> make_step_batches(const MemmapSplit &split, const NeighborCache &nl,
                                                  const std::vector<int64_t> &ids,
                                                  const std::vector<std::array<float, 9>> &rotations,
                                                  const std::map<int, int> &species_to_index,
                                                  const TargetScaling &scaling, float cutoff,
                                                  int n_workers, const CostModel &cost,
                                                  double budget = 5e7,
                                                  const Tiling &tiling = {});

} // namespace mlipcpp::train
