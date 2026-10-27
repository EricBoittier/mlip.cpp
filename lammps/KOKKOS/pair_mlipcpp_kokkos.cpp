/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

// pair_style mlipcpp/kk MODEL.gguf [backend ...]: pair_style mlipcpp (see
// ML-MLIPCPP/pair_mlipcpp.cpp) with LAMMPS's arrays left on the GPU.
//
// Each step mlip.cpp takes the owned atoms' x and type views and LAMMPS's f
// view as device pointers (with their strides) and the Kokkos execution
// space's CUDA stream, converts the positions and builds the model inputs on
// the device, and adds the forces into f with a kernel; the step's one
// synchronisation brings back the energy and virial (and mlip.cpp's check
// that the step held). LAMMPS's x and f are never synced to the host.
//
// mlip.cpp keeps its own Verlet list on the device (cutoff + skin, periodic
// images instead of ghosts). The pair style requests a device list from LAMMPS
// only so that LAMMPS's KOKKOS neighbor runs on the device (see init_style).
// mlip.cpp's list is rebuilt when LAMMPS rebuilds its own
// (neighbor->ago == 0: atoms may have been sorted or remapped into the box),
// with LAMMPS's skin, so between rebuilds no atom has moved half the skin;
// mlip.cpp checks that on the device anyway and computes the step again after
// a rebuild if not. One MPI rank; per-atom energy and virial not available.

#include "pair_mlipcpp_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cstring>
#include <type_traits>

using namespace LAMMPS_NS;

namespace {

const char *last_error()
{
  const char *e = mlipcpp_get_last_error();
  return e && *e ? e : "unknown error";
}

template <class DeviceType> constexpr bool on_cuda()
{
#ifdef KOKKOS_ENABLE_CUDA
  return std::is_same_v<typename DeviceType::execution_space, Kokkos::Cuda>;
#else
  return false;
#endif
}

// the CUDA stream LAMMPS's kernels on DeviceType run on
template <class DeviceType> void *stream_of()
{
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (on_cuda<DeviceType>()) return (void *) typename DeviceType::execution_space().cuda_stream();
#endif
  return nullptr;
}

}    // namespace

template <class DeviceType> PairMLIPCPPKokkos<DeviceType>::PairMLIPCPPKokkos(LAMMPS *lmp) : PairMLIPCPP(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

template <class DeviceType> void PairMLIPCPPKokkos<DeviceType>::init_style()
{
  PairMLIPCPP::init_style();
  // mlip.cpp takes CUDA device pointers and a CUDA stream
  if (!on_cuda<DeviceType>()) error->all(FLERR, "Pair style mlipcpp/kk needs LAMMPS's KOKKOS package on CUDA");
  if (!mlipcpp_model_supports_device(model))
    error->all(FLERR, "Pair style mlipcpp/kk: the model cannot run on device arrays (needs mlip.cpp on CUDA and a "
                      "pet-graph GGUF); use pair_style mlipcpp");
  // A full list on the device, which mlip.cpp does not read: with a device
  // request LAMMPS's KOKKOS neighbor checks the displacements and rebuilds on
  // the device (as for pet/kk); without one it does both on the host and
  // syncs x there every step.
  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_device(1);
  request->set_kokkos_host(0);
  if (comm->me == 0)
    utils::logmesg(lmp, "pair_style mlipcpp/kk: LAMMPS's device arrays on the Kokkos stream; mlip.cpp's list "
                        "rebuilt with LAMMPS's, skin {:.4g} A\n", neighbor->skin);
}

template <class DeviceType> void PairMLIPCPPKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  ev_init(eflag_in, vflag_in, 0);
  if (eflag_atom || vflag_atom)
    error->all(FLERR, "Pair style mlipcpp/kk does not compute per-atom energies or virials");

  const int n = atom->nlocal;
  if (n == 0) return;
  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  auto x = atomKK->k_x.template view<DeviceType>();
  auto f = atomKK->k_f.template view<DeviceType>();
  auto type = atomKK->k_type.template view<DeviceType>();
  static_assert(std::is_same_v<typename decltype(x)::non_const_value_type, double> &&
                    std::is_same_v<typename decltype(f)::non_const_value_type, double>,
                "pair_style mlipcpp/kk needs double positions and forces");
  static_assert(sizeof(typename decltype(type)::non_const_value_type) == sizeof(int32_t), "32-bit atom types");

  // LAMMPS's h is xx yy zz yz xz xy; mlip.cpp wants the lattice vectors as rows.
  const double *h = domain->h;
  const double cell[9] = {h[0], 0.0, 0.0, h[5], h[1], 0.0, h[4], h[3], h[2]};
  const bool pbc[3] = {domain->xperiodic != 0, domain->yperiodic != 0, domain->zperiodic != 0};
  const bool periodic = pbc[0] || pbc[1] || pbc[2];
  type_table.assign(type_z.begin(), type_z.end());

  mlipcpp_device_system_t s;
  mlipcpp_device_system_default(&s);
  s.n_atoms = n;
  s.positions = x.data();
  s.position_strides[0] = x.stride(0), s.position_strides[1] = x.stride(1);
  s.types = reinterpret_cast<const int32_t *>(type.data());
  s.n_types = atom->ntypes;
  s.type_atomic_numbers = type_table.data();
  s.cell = periodic ? cell : nullptr;
  for (int k = 0; k < 3; ++k) s.pbc[k] = pbc[k];
  s.forces = f.data();
  s.force_strides[0] = f.stride(0), s.force_strides[1] = f.stride(1);
  s.stream = stream_of<DeviceType>();
  s.rebuild_neighbors = neighbor->ago == 0;
  s.skin = neighbor->skin;
  s.compute_virial = vflag_global != 0;
  s.charge = charge;
  s.spin_multiplicity = spin_multiplicity;

  mlipcpp_device_result_t r;
  if (mlipcpp_compute_device(model, &s, &r) != MLIPCPP_OK)
    error->one(FLERR, "Pair style mlipcpp/kk: {}", last_error());

  if (eflag_global) eng_vdwl += r.energy;
  // LAMMPS's virial is xx yy zz xy xz yz; mlip.cpp's xx yy zz yz xz xy, the
  // same sign (-dE/de, for a periodic box -V sigma)
  if (vflag_global) {
    const int voigt[6] = {0, 1, 2, 5, 4, 3};
    for (int k = 0; k < 6; ++k) virial[k] += r.virial[voigt[k]];
  }
  atomKK->modified(execution_space, F_MASK);
}

namespace LAMMPS_NS {
template class PairMLIPCPPKokkos<LMPDeviceType>;
}
