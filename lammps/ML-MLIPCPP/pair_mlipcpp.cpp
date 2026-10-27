/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

// pair_style mlipcpp MODEL.gguf [backend auto|cpu|cuda|hip|metal|vulkan|sycl]
// pair_coeff * * ELEMENT...        (one element symbol per atom type)
//
// Units metal, one MPI rank. Every step the owned atoms go to mlip.cpp as they
// are, with LAMMPS's box as the cell and its periodicity as the pbc flags (no
// cell for a box periodic in no direction, as mlip.cpp's ASE calculator does);
// mlip.cpp finds the neighbours itself (on the GPU, reused between steps
// while no atom has moved half of MLIPCPP_NL_SKIN), so the pair style asks
// LAMMPS for no neighbour list and its ghosts carry no forces.
//
// The global virial is mlip.cpp's: -V sigma for a box periodic in any
// direction (sigma = dE/de / V, its stress), else sum_i x_i (x) F_i,
// symmetrised (the same thing without images). Per-atom energy and virial are
// not available through the C API.
//
// mlip.cpp's run-time switches (MLIPCPP_RAGGED, MLIPCPP_TF32, MLIPCPP_NL_SKIN,
// ...) are read from the environment, as for its other front ends.

#include "pair_mlipcpp.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <string>

using namespace LAMMPS_NS;

namespace {

const char *const ELEMENTS[] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl",
    "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
    "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
    "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

int element_number(const char *symbol)
{
  for (int z = 1; z <= int(sizeof(ELEMENTS) / sizeof(ELEMENTS[0])); ++z)
    if (std::strcmp(symbol, ELEMENTS[z - 1]) == 0) return z;
  return 0;
}

const char *last_error()
{
  const char *e = mlipcpp_get_last_error();
  return e && *e ? e : "unknown error";
}

}    // namespace

/* ---------------------------------------------------------------------- */

PairMLIPCPP::PairMLIPCPP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;    // mlip.cpp's virial, see compute()
  centroidstressflag = CENTROID_NOTAVAIL;
  // Under the KOKKOS package (a host pair style there), only what crosses
  // between LAMMPS's device and host copies each step.
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

PairMLIPCPP::~PairMLIPCPP()
{
  mlipcpp_model_free(model);
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairMLIPCPP::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (eflag_atom || vflag_atom)
    error->all(FLERR, "Pair style mlipcpp does not compute per-atom energies or virials");

  const int n = atom->nlocal;
  if (n == 0) return;
  double **x = atom->x, **f = atom->f;
  const int *type = atom->type;
  pos.resize(3 * std::size_t(n)), frc.resize(3 * std::size_t(n)), z.resize(n);
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < 3; ++c) pos[3 * i + c] = float(x[i][c]);
    z[i] = type_z[type[i]];
  }

  // LAMMPS's h is xx yy zz yz xz xy; mlip.cpp wants the lattice vectors as rows.
  const double *h = domain->h;
  const float cell[9] = {float(h[0]), 0.0f, 0.0f, float(h[5]), float(h[1]), 0.0f, float(h[4]), float(h[3]), float(h[2])};
  const bool pbc[3] = {domain->xperiodic != 0, domain->yperiodic != 0, domain->zperiodic != 0};
  const bool periodic = pbc[0] || pbc[1] || pbc[2];

  mlipcpp_predict_options_t opts;
  mlipcpp_predict_options_default(&opts);
  opts.compute_forces = true;
  opts.compute_stress = periodic && vflag_global;
  opts.charge = charge;
  opts.spin_multiplicity = spin_multiplicity;
  mlipcpp_result_t res = nullptr;
  if (mlipcpp_predict_ptr_with_options(model, n, pos.data(), z.data(), periodic ? cell : nullptr,
                                       periodic ? pbc : nullptr, &opts, &res) != MLIPCPP_OK)
    error->one(FLERR, "Pair style mlipcpp: {}", last_error());

  float energy = 0.0f, stress[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  bool ok = mlipcpp_result_get_energy(res, &energy) == MLIPCPP_OK &&
      mlipcpp_result_get_forces(res, frc.data(), n) == MLIPCPP_OK;
  if (ok && opts.compute_stress) ok = mlipcpp_result_get_stress(res, stress) == MLIPCPP_OK;
  mlipcpp_result_free(res);
  if (!ok) error->one(FLERR, "Pair style mlipcpp: {}", last_error());

  if (eflag_global) eng_vdwl += energy;
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) f[i][c] += frc[3 * i + c];

  // LAMMPS's virial is xx yy zz xy xz yz, -dE/de; mlip.cpp's stress is
  // xx yy zz yz xz xy, dE/de / V (with V of the cell it was given).
  if (vflag_global) {
    if (periodic) {
      const double volume = std::fabs(double(cell[0]) * cell[4] * cell[8]);
      const int voigt[6] = {0, 1, 2, 5, 4, 3};
      for (int k = 0; k < 6; ++k) virial[k] -= volume * stress[voigt[k]];
    } else {
      double w[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int i = 0; i < n; ++i) {
        const double *r = x[i], g[3] = {frc[3 * i], frc[3 * i + 1], frc[3 * i + 2]};
        w[0] += r[0] * g[0], w[1] += r[1] * g[1], w[2] += r[2] * g[2];
        w[3] += 0.5 * (r[0] * g[1] + r[1] * g[0]);
        w[4] += 0.5 * (r[0] * g[2] + r[2] * g[0]);
        w[5] += 0.5 * (r[1] * g[2] + r[2] * g[1]);
      }
      for (int k = 0; k < 6; ++k) virial[k] += w[k];
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMLIPCPP::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;
}

void PairMLIPCPP::settings(int narg, char **arg)
{
  if (narg < 1 || narg % 2 == 0)
    error->all(FLERR, "Illegal pair_style mlipcpp command: pair_style mlipcpp MODEL.gguf "
                      "[backend auto|cpu|cuda|hip|metal|vulkan|sycl] [charge Q] [spin 2S+1]");
  mlipcpp_model_options_t opts;
  mlipcpp_model_options_default(&opts);
  for (int k = 1; k < narg; k += 2) {
    const std::string key = arg[k], val = arg[k + 1];
    // total charge and spin multiplicity of the whole box, for models with charge /
    // spin conditioning (an error with other models, out of the model's range too)
    if (key == "charge") { charge = utils::inumeric(FLERR, val, false, lmp); continue; }
    if (key == "spin") { spin_multiplicity = utils::inumeric(FLERR, val, false, lmp); continue; }
    if (key != "backend") error->all(FLERR, "Illegal pair_style mlipcpp keyword: {}", key);
    if (val == "auto") opts.backend = MLIPCPP_BACKEND_AUTO;
    else if (val == "cpu") opts.backend = MLIPCPP_BACKEND_CPU;
    else if (val == "cuda") opts.backend = MLIPCPP_BACKEND_CUDA;
    else if (val == "hip") opts.backend = MLIPCPP_BACKEND_HIP;
    else if (val == "metal") opts.backend = MLIPCPP_BACKEND_METAL;
    else if (val == "vulkan") opts.backend = MLIPCPP_BACKEND_VULKAN;
    else if (val == "sycl") opts.backend = MLIPCPP_BACKEND_SYCL;
    else error->all(FLERR, "Illegal pair_style mlipcpp backend: {}", val);
  }

  mlipcpp_model_free(model);
  model = mlipcpp_model_create(&opts);
  if (!model || mlipcpp_model_load(model, arg[0]) != MLIPCPP_OK)
    error->all(FLERR, "Pair style mlipcpp: cannot load {}: {}", arg[0], last_error());
  float rc = 0.0f;
  if (mlipcpp_model_get_cutoff(model, &rc) != MLIPCPP_OK)
    error->all(FLERR, "Pair style mlipcpp: {}", last_error());
  cutoff = rc;
  bool conditioned = false;
  int32_t max_q = 0, max_s = 0;
  if (mlipcpp_model_get_conditioning(model, &conditioned, &max_q, &max_s) != MLIPCPP_OK)
    error->all(FLERR, "Pair style mlipcpp: {}", last_error());
  if (!conditioned && (charge != 0 || spin_multiplicity != 1))
    error->all(FLERR, "Pair style mlipcpp: charge / spin given, but {} is not conditioned on them", arg[0]);
  if (conditioned && (charge < -max_q || charge > max_q || spin_multiplicity < 1 || spin_multiplicity > max_s))
    error->all(FLERR, "Pair style mlipcpp: charge {} / spin {} outside the model's [-{}, {}] / [1, {}]", charge,
               spin_multiplicity, max_q, max_q, max_s);
  if (comm->me == 0)
    utils::logmesg(lmp, "pair_style mlipcpp: {} on {}, cutoff {:.4g} A, mlip.cpp {}\n", arg[0],
                   mlipcpp_get_backend_name(), cutoff, mlipcpp_version());
}

void PairMLIPCPP::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  const int n = atom->ntypes;
  if (narg != 2 + n)
    error->all(FLERR, "Incorrect args for pair coefficients: pair_coeff * * followed by {} elements", n);
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients: pair_style mlipcpp takes pair_coeff * *");
  type_z.assign(n + 1, 0);
  for (int t = 1; t <= n; ++t) {
    type_z[t] = element_number(arg[1 + t]);
    if (!type_z[t]) error->all(FLERR, "Pair style mlipcpp: unknown element {}", arg[1 + t]);
  }
  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 1;
}

void PairMLIPCPP::init_style()
{
  if (strcmp(update->unit_style, "metal") != 0) error->all(FLERR, "Pair style mlipcpp requires metal units");
  // The owned atoms are the whole system only on one rank: ghosts that belong
  // to other ranks would need their own exchange (as pair_style pet's).
  if (comm->nprocs > 1) error->all(FLERR, "Pair style mlipcpp runs on one MPI rank only");
  // mlip.cpp builds its own neighbour list: no request to LAMMPS.
}

double PairMLIPCPP::init_one(int, int)
{
  return cutoff;
}
