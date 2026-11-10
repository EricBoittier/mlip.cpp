#pragma once

// Physics-normalised long-range tail loss (mlipcpp-train --tail-data): energy differences within a pulled-apart scan,
// ΔE(d) = E(d) - E(d_far), against DFT, each residual divided by the scan point's physics scale
// s = max(|ΔE_ref|, floor), ΔE_ref the frozen-density electrostatics + induction difference (merge-aug set A).
// Per group (one scan): weight * mean over its pairs of Huber_delta((ΔE_model - ΔE_DFT) / s).
// Absolute energies never enter (set A sits 0.4-1.6 meV/atom off OMol25), nor forces.

#include <cstdint>
#include <string>
#include <vector>

namespace mlipcpp::train {

struct TailPair {
  int64_t k = 0, l = 0;   // structure indices (into the tail split, or local batch indices)
  double de_dft = 0;      // E_DFT(k) - E_DFT(l), eV
  double de_ref = 0;      // ΔE_ref(k) - ΔE_ref(l), eV (signed)
  double d = 0;           // closest-atom distance of k, Å
  int cls = -1;           // the pair's own class (TailSet::load with mixed classes; else the group's)
};

struct TailGroup {
  std::string name;
  int cls = 0;
  std::vector<int64_t> ids;     // distinct structures of the group (tail split indices)
  std::vector<TailPair> pairs;  // k, l: positions in ids
};

/// tail_pairs.tsv: a header line, then one pair per line:
/// group  class  k  l  dE_dft_eV  dE_ref_eV  d_A   (k, l: tail split indices)
struct TailSet {
  std::vector<std::string> classes;
  std::vector<TailGroup> groups;
  std::vector<std::vector<int>> by_class; // group indices per class
  /// mixed: a group may hold pairs of several classes (pair data: IP and EA of one geometry); the group's class,
  /// used to sample groups, is that of its first pair, and each pair keeps its own in TailPair::cls
  static TailSet load(const std::string &path, bool mixed = false);
  size_t n_pairs() const;
};

struct TailLoss {
  double weight = 0.0;
  double floor = 2.585e-3; // eV: 0.1 k_B T at 300 K
  double delta = 1.0;      // Huber delta on the normalised residual (1 = a whole missing tail)
  double scale(const TailPair &p) const;
  double residual(const TailPair &p, const std::vector<double> &E) const; // (ΔE_model - ΔE_DFT) / s
  /// weight * mean_p Huber(r_p); E: model energies (eV) per local index; grad (optional): dL/dE, resized to E
  double eval(const std::vector<TailPair> &pairs, const std::vector<double> &E, std::vector<double> *grad) const;
};

} // namespace mlipcpp::train
