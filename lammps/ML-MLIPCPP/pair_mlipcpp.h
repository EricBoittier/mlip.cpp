/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mlipcpp,PairMLIPCPP);
// clang-format on
#else

#ifndef LMP_PAIR_MLIPCPP_H
#define LMP_PAIR_MLIPCPP_H

#include "pair.h"

#include "mlipcpp/mlipcpp.h"

#include <cstdint>
#include <vector>

namespace LAMMPS_NS {

// mlip.cpp GGUF models (PET-MAD and friends) through its C API. mlip.cpp builds
// its own neighbour list (on the GPU, kept between steps with a Verlet skin), so
// the pair style hands it the owned atoms, the box and its periodicity each
// step, and takes back energy, forces and, for a periodic box, the stress.
class PairMLIPCPP : public Pair {
 public:
  PairMLIPCPP(class LAMMPS *);
  ~PairMLIPCPP() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  mlipcpp_model_t model = nullptr;
  double cutoff = 0.0;              // the model's
  std::vector<int> type_z;          // atomic number of each LAMMPS type
  std::vector<float> pos, frc;      // what crosses the C API, per step
  std::vector<int32_t> z;
  int charge = 0, spin_multiplicity = 1; // pair_style keywords charge / spin (conditioned models)

  void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif
