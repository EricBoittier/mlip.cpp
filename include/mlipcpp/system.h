#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <array>
#include <vector>

namespace mlipcpp {

/// Represents a periodic cell with lattice vectors
struct Cell {
  float matrix[3][3];  // Row-major: matrix[i] is the i-th lattice vector
  float inverse[3][3]; // Cached inverse matrix
  bool periodic[3];    // Periodicity in each direction

  Cell();
  explicit Cell(const float lattice[3][3], bool pbc_x = true, bool pbc_y = true,
                bool pbc_z = true);

  /// Convert fractional coordinates to Cartesian
  void to_cartesian(const float frac[3], float cart[3]) const;

  /// Convert Cartesian coordinates to fractional
  void to_fractional(const float cart[3], float frac[3]) const;

  /// Apply minimum image convention for a displacement vector
  void minimum_image(float dr[3]) const;

private:
  void compute_inverse();
};

/// Atomic system representation
class AtomicSystem {
public:
  AtomicSystem() = default;
  AtomicSystem(int n_atoms, const float *positions,
               const int32_t *atomic_numbers, const Cell *cell = nullptr);

  int num_atoms() const { return n_atoms_; }
  const float *positions() const { return positions_.data(); }
  const int32_t *atomic_numbers() const { return atomic_numbers_.data(); }
  const Cell *cell() const { return cell_ ? &cell_.value() : nullptr; }

  bool is_periodic() const { return cell_.has_value(); }

  // Mutable access for construction
  std::vector<float> &positions_mut() { return positions_; }
  std::vector<int32_t> &atomic_numbers_mut() { return atomic_numbers_; }
  void set_cell(const Cell &cell) { cell_ = cell; }

  /// Total charge (e) and spin multiplicity 2S+1 of the system, for models with
  /// charge / spin conditioning (metatrain `system_conditioning`; extxyz
  /// info["charge"], info["spin_multiplicity"] or info["spin"]). Ignored by
  /// other models. Defaults: neutral singlet, as metatrain.
  int charge() const { return charge_; }
  int spin_multiplicity() const { return spin_multiplicity_; }
  void set_charge(int charge) { charge_ = charge; }
  void set_spin_multiplicity(int spin_multiplicity) { spin_multiplicity_ = spin_multiplicity; }

  /// Per-atom formal charges (e; extxyz per-atom array formal_charges), read by models with
  /// fragment-constrained charges (pet.plato_lr.frag_charges): only their sums over each bonded
  /// fragment matter. Empty: none (whole-structure conditioning). Ignored by other models.
  const std::vector<float> &formal_charges() const { return formal_charges_; }
  bool has_formal_charges() const { return !formal_charges_.empty(); }
  void set_formal_charges(std::vector<float> q) { formal_charges_ = std::move(q); }

  /// Origin of the dipole (A; extxyz info dipole_origin="x y z"), read by models that output a dipole
  /// (iris PET-LR, pet.iris_lr). Unset: the model's default (the centre of mass, ASE masses).
  bool has_dipole_origin() const { return has_dipole_origin_; }
  const std::array<double, 3> &dipole_origin() const { return dipole_origin_; }
  void set_dipole_origin(const std::array<double, 3> &r0) { dipole_origin_ = r0, has_dipole_origin_ = true; }

private:
  int n_atoms_ = 0;
  std::vector<float> positions_; // [n_atoms * 3], flattened (x,y,z) per atom
  std::vector<int32_t> atomic_numbers_;
  std::optional<Cell> cell_;
  int charge_ = 0, spin_multiplicity_ = 1;
  std::vector<float> formal_charges_;
  std::array<double, 3> dipole_origin_{0, 0, 0};
  bool has_dipole_origin_ = false;
};

} // namespace mlipcpp
