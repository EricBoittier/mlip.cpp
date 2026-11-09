/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mlipcpp/kk,PairMLIPCPPKokkos<LMPDeviceType>);
PairStyle(mlipcpp/kk/device,PairMLIPCPPKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_PAIR_MLIPCPP_KOKKOS_H
#define LMP_PAIR_MLIPCPP_KOKKOS_H

#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "pair_mlipcpp.h"

namespace LAMMPS_NS {

// pair_style mlipcpp on LAMMPS's KOKKOS package: mlip.cpp reads the owned
// atoms' positions and types from, and adds their forces to, LAMMPS's device
// arrays (mlipcpp_compute_device), on the Kokkos stream. Nothing crosses to the
// host but the energy and virial; mlip.cpp's own device neighbour list is
// rebuilt at LAMMPS's rebuilds, with LAMMPS's skin.
template <class DeviceType> class PairMLIPCPPKokkos : public PairMLIPCPP {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  PairMLIPCPPKokkos(class LAMMPS *);
  void compute(int, int) override;
  void init_style() override;

 private:
  std::vector<int32_t> type_table;  // type_z, as the C API takes it
};

}    // namespace LAMMPS_NS

#endif
#endif
