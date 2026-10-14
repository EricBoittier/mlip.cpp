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

  std::vector<int32_t> species;            // [n_atoms] species index
  std::vector<int32_t> neighbor_species;   // [n_atoms, max_nb]
  std::vector<float> edge_vectors;         // [n_atoms, max_nb, 3]
  std::vector<float> padding_mask;         // [n_atoms, max_nb], 1 = padded
  std::vector<int32_t> reverse_neighbor_index; // [n_atoms * max_nb]
  std::vector<float> cutoff_values;        // [n_atoms, max_nb]

  std::vector<float> atom_weight;   // [n_systems, n_atoms]: 1/N_s if atom in s
  std::vector<float> energy_target; // [n_systems]: (E - comp) / (s_E N_s)
  std::vector<float> force_target;  // [n_atoms, 3]: R F / s_Z
  std::vector<float> force_scale;   // [n_atoms]: s_Z, for physical metrics (0: padding)
  std::vector<float> force_mask;    // [n_atoms]: 1 real atom, 0 padding atom
  std::vector<int32_t> n_per_system;

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

} // namespace mlipcpp::train
