#pragma once
// Charge and spin-multiplicity conditioning (metatrain pet/modules/conditioning.py,
// SystemConditioningEmbedding): per-system lookup tables of the total charge, in
// [-max_charge, max_charge], and of the spin multiplicity 2S+1, in
// [1, max_spin_multiplicity], concatenated and projected (Linear, SiLU, Linear), the
// result added to every atom's node features after each GNN layer. The graph inputs
// are the table rows of each atom's structure: charge + max_charge and 2S+1 - 1.
// Values outside the tables are an error (never clamped), as in metatrain.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlipcpp {

struct Conditioning {
  bool enabled = false;
  int max_charge = 10, max_spin_multiplicity = 10;

  void check(int charge, int spin_multiplicity) const {
    if (charge < -max_charge || charge > max_charge)
      throw std::runtime_error("charge must be in [" + std::to_string(-max_charge) + ", " +
                               std::to_string(max_charge) + "], got " + std::to_string(charge) +
                               " (the model's max_charge)");
    if (spin_multiplicity < 1 || spin_multiplicity > max_spin_multiplicity)
      throw std::runtime_error("spin_multiplicity must be in [1, " + std::to_string(max_spin_multiplicity) +
                               "], got " + std::to_string(spin_multiplicity) +
                               " (the model's max_spin_multiplicity)");
  }
  /// (charge table row, spin table row), checked
  std::pair<int32_t, int32_t> rows(int charge, int spin_multiplicity) const {
    check(charge, spin_multiplicity);
    return {charge + max_charge, spin_multiplicity - 1};
  }
};

/// A real number that must be an integer (extxyz info, metatomic data): its value,
/// or an error naming `what`.
inline int integer_value(double x, const char *what) {
  const double r = static_cast<double>(static_cast<long long>(x >= 0 ? x + 0.5 : x - 0.5));
  if (!(x == r) || r > 1e9 || r < -1e9)
    throw std::runtime_error(std::string(what) + " must be an integer value, got " + std::to_string(x));
  return static_cast<int>(r);
}

} // namespace mlipcpp
