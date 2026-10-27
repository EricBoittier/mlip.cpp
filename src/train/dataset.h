#pragma once

#include "mlipcpp/system.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace mlipcpp::train {

/// Read-only memory map of a raw C-order array file.
template <typename T> class Mapped {
public:
  Mapped() = default;
  explicit Mapped(const std::string &path);
  Mapped(Mapped &&o) noexcept { *this = std::move(o); }
  Mapped &operator=(Mapped &&o) noexcept;
  ~Mapped();

  const T *data() const { return data_; }
  size_t size() const { return size_; }
  const T &operator[](size_t i) const { return data_[i]; }

private:
  const T *data_ = nullptr;
  size_t size_ = 0;
};

/// int64 payload of a little-endian .npy file (any shape, '<i8' only).
std::vector<int64_t> read_npy_i64(const std::string &path);

/// One split (train/ val/ test/) of metatrain's MemmapDataset:
/// ns.npy (0-d int64), na.npy (ns+1 cumulative atom counts), x.bin (f32
/// [N,3]), a.bin (i32 [N]), c.bin (f32 [ns,3,3], rows are lattice vectors),
/// energy.bin (f32 [ns]) and non_conservative_force.bin (f32 [N,3]).
/// pbc[k] = (cell row k != 0), as in metatrain.
class MemmapSplit {
public:
  explicit MemmapSplit(const std::string &dir);

  int64_t size() const { return n_systems_; }
  int n_atoms(int64_t s) const { return static_cast<int>(na_[s + 1] - na_[s]); }
  int64_t first_atom(int64_t s) const { return na_[s]; }
  int64_t total_atoms() const { return na_.back(); }

  const float *positions(int64_t s) const { return x_.data() + 3 * na_[s]; }
  const int32_t *numbers(int64_t s) const { return a_.data() + na_[s]; }
  const float *cell(int64_t s) const { return c_.data() + 9 * s; }
  float energy(int64_t s) const { return energy_[s]; }
  const float *forces(int64_t s) const { return force_.data() + 3 * na_[s]; }
  /// total charge and spin multiplicity (charge.bin / spin_multiplicity.bin,
  /// float32 [ns, 1], metatrain's extra_data; absent: 0 and 1)
  int charge(int64_t s) const { return charge_.empty() ? 0 : charge_[s]; }
  int spin_multiplicity(int64_t s) const { return spin_.empty() ? 1 : spin_[s]; }
  bool has_conditioning_data() const { return !charge_.empty() || !spin_.empty(); }

  /// System s as an AtomicSystem. Zero cell rows become non-periodic vacuum
  /// vectors so the cell stays invertible for the neighbour-list grid.
  AtomicSystem system(int64_t s) const;

  const std::string &dir() const { return dir_; }

private:
  std::string dir_;
  int64_t n_systems_ = 0;
  std::vector<int64_t> na_;
  Mapped<float> x_, c_, energy_, force_;
  Mapped<int32_t> a_;
  std::vector<int32_t> charge_, spin_;
};

/// Neighbour lists of every structure in a split, built once and cached on
/// disk. A rotation changes neither the neighbour pairs nor the cell shifts,
/// so augmentation only rotates the stored edge vectors.
///
/// Edges of structure s are [edge_offset[s], edge_offset[s+1]), sorted by
/// centre; `slot` is the rank of an edge among its centre's edges and
/// `reverse` the local index of the edge j->i with the opposite shift
/// (or -1).
/// PET's adaptive cutoff (metatrain `num_neighbors_adaptive`, "solver"
/// method): per-atom radii r_i that make the smoothed neighbour count
/// `n_target`, pair cutoffs (r_i + r_j) / 2, edges beyond them dropped.
/// Positions are fixed in training, so this is part of the cached lists.
struct AdaptiveCutoff {
  double n_target = 0; // 0: the fixed cutoff
  double width = 1.0;  // cutoff_width_adaptive
  bool enabled() const { return n_target > 0; }
};

/// The edges of structure s's full (fixed-cutoff) list that move an adaptive
/// radius: d r_i / d d_ik != 0 (core/adaptive_cutoff.h), with that
/// derivative. -dE/dx has a term through the radii (pair cutoffs
/// (r_i + r_j) / 2), and these edges include ones the pair cutoffs dropped.
struct TaperEdges {
  std::vector<int32_t> center, neighbor; // local atom indices
  std::vector<std::array<float, 3>> vector;
  std::vector<float> jacobian; // d r_center / d |vector|
};

struct NeighborCache {
  float cutoff = 0.0f;
  AdaptiveCutoff adaptive;
  std::vector<int64_t> edge_offset; // [ns+1]
  std::vector<int32_t> max_neighbors; // [ns]
  std::vector<int32_t> center, neighbor, slot, reverse; // [E], local indices
  std::vector<std::array<float, 3>> vector; // [E], unrotated
  std::vector<float> pair_cutoff;            // [E] with an adaptive cutoff, else empty (= cutoff)
  /// conservative training with an adaptive cutoff: make_batch adds every
  /// structure's taper_edges (rebuilt from the positions, not cached)
  bool taper = false;

  int64_t n_edges() const { return edge_offset.back(); }

  /// Load `cache_path` if it matches the split and cutoff, else build with
  /// `n_threads` threads and write it there (empty path: no disk cache).
  static NeighborCache load_or_build(const MemmapSplit &split, float cutoff,
                                     const std::string &cache_path,
                                     int n_threads, const AdaptiveCutoff &adaptive = {});
};

TaperEdges taper_edges(const MemmapSplit &split, int64_t s, float cutoff, const AdaptiveCutoff &adaptive);

} // namespace mlipcpp::train
