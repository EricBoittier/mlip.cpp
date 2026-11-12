#pragma once

#include "graph_interpreter.h"

#include <map>
#include <string>
#include <vector>

namespace mlipcpp::runtime {

/// Several energy heads on one PET backbone, one per level of theory (multi-fidelity training).
/// GGUF: pet.energy_heads = H (absent: 1) and pet.energy_head_names (JSON list of H names). Head 0 is the graph's
/// own energy readout; head h >= 1 is a copy of every readout tensor under the name "head<h>." + name (the
/// per-atom energy MLPs and final layers, i.e. everything between the backbone features and the per-atom energy),
/// with its own composition baseline pet.head<h>.composition_{keys,values} (eV per atom of each element).
/// The backbone, the conditioning and the direct-force head are shared.
struct EnergyHeads {
  int n = 1;
  std::vector<std::string> names;                // [n]
  std::vector<std::string> weights;              // readout tensors that have a copy per head (names without prefix)
  std::vector<std::map<int, float>> composition; // [n]: head h's composition (h = 0: pet.composition_*)

  bool enabled() const { return n > 1; }
  static std::string prefix(int h) { return "head" + std::to_string(h) + "."; }
  /// head index of a name ("omol", ...) or a number; throws if unknown
  int index(const std::string &name_or_index) const;
};

/// GIR node id of the per-atom energy [n_atoms]: the graph output, or its CONCAT's first input
/// ((energy | direct force) exports), followed back through RESHAPE / CONT / VIEW.
int energy_node(const GIRGraph &g);

/// Training graphs: a post_node hook that replaces the per-atom energy e by sum_h mask[:, h] * e_h, e_h the energy
/// rebuilt with head h's tensors. `mask` is an F32 input {n_atoms, H}: the one-hot head of each atom's structure
/// (padding atoms: any). Chains to an already installed post_node.
void install_head_select(GraphInterpreter &interp, const EnergyHeads &heads, ggml_tensor *mask);

} // namespace mlipcpp::runtime
