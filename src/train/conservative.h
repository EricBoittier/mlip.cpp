#pragma once
#include "runtime/graph_model.h"
#include <memory>
namespace mlipcpp::train {
struct ConservativeResult {
  double loss = 0, energy_error = 0, force_rmse = 0;
  std::vector<float> gradient;
  std::vector<double> energies, forces;
};
// A bounded, reusable shape cache. Batches contain disjoint atomic graphs and
// independent P3M cells; no periodic interactions cross structure boundaries.
class ConservativeTrainer {
public:
  explicit ConservativeTrainer(runtime::GraphModel &model);
  ~ConservativeTrainer();
  ConservativeResult evaluate(const std::vector<AtomicSystem> &systems,
                              const std::vector<double> &energies,
                              const std::vector<float> &forces,
                              double energy_weight, double force_weight,
                              bool backward = true);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlipcpp::train
