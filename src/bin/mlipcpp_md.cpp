// GPU molecular dynamics with a trained PET GGUF (md/md_engine.h).
//
//   mlipcpp-md check MODEL DATA_SPLIT [--n 200] [--fd 4] [--max-atoms 256] [--dump PREFIX.test.bin]
//       [--ids FILE] [--out PREFIX] [--cpu 1] [--gpu-ref 0]
//       Validates the MD engine on structures of a memmap split: against the
//       labels, against GraphModel::predict on the CPU (an independent
//       implementation of the inputs, adaptive cutoff and forces), against the
//       trainer's own per-structure errors (--dump), and the conservative
//       forces against finite differences of the energy.
//       --ids: the structures to check (whitespace-separated indices; no
//       --n / --max-atoms selection). --cpu 0 skips the CPU GraphModel,
//       --gpu-ref 1 adds GraphModel on CUDA. --out writes PREFIX.csv (one row
//       per structure: energies) and PREFIX.forces.bin (float32, 21 per atom:
//       label, engine direct, engine conservative, CPU direct, CPU cons,
//       GPU direct, GPU cons; NaN where not computed).
//
//   mlipcpp-md run MODEL (--xyz FILE | --data DATA_SPLIT --index I) [--repeat a,b,c]
//       [--integrator respa|conservative|direct] [--inner 4] [--dt 0.5] [--steps 1000]
//       [--temperature 300] [--seed 1] [--skin 1.0] [--log-every 10]
//       [--traj FILE.xyz --traj-every 100] [--thermo FILE.tsv]
//       NVE from Maxwell-Boltzmann velocities; dt in fs, --steps counts dt steps.

#include "core/adaptive_cutoff.h"
#include "md/md_engine.h"
#include "mlipcpp/io.h"
#include "mlipcpp/neighbor_list.h"
#include "train/dataset.h"
#include "train/trainer.h"

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mlipcpp;

namespace {

struct Args {
  std::vector<std::string> pos;
  std::map<std::string, std::string> opt;
  std::string get(const std::string &k, const std::string &def = "") const {
    auto it = opt.find(k);
    return it == opt.end() ? def : it->second;
  }
  double num(const std::string &k, double def) const { return opt.count(k) ? std::stod(opt.at(k)) : def; }
};

Args parse(int argc, char **argv) {
  Args a;
  for (int i = 2; i < argc; i++) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      a.opt[s.substr(2)] = argv[++i];
    } else a.pos.push_back(s);
  }
  return a;
}

const char *SYMBOLS[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
    "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
    "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

AtomicSystem repeat(const AtomicSystem &s, int ra, int rb, int rc) {
  if (ra * rb * rc == 1) return s;
  const Cell *c = s.cell();
  if (!c) throw std::runtime_error("--repeat needs a cell");
  const int n = s.num_atoms();
  std::vector<float> x;
  std::vector<int32_t> z;
  for (int a = 0; a < ra; a++)
    for (int b = 0; b < rb; b++)
      for (int k = 0; k < rc; k++)
        for (int i = 0; i < n; i++) {
          for (int d = 0; d < 3; d++)
            x.push_back(s.positions()[3 * i + d] + a * c->matrix[0][d] + b * c->matrix[1][d] + k * c->matrix[2][d]);
          z.push_back(s.atomic_numbers()[i]);
        }
  float m[3][3];
  const int r[3] = {ra, rb, rc};
  for (int a = 0; a < 3; a++)
    for (int d = 0; d < 3; d++) m[a][d] = c->matrix[a][d] * r[a];
  Cell big(m, c->periodic[0], c->periodic[1], c->periodic[2]);
  return AtomicSystem(n * ra * rb * rc, x.data(), z.data(), &big);
}

void write_frame(std::ofstream &out, const md::Engine &e, int64_t step, double time, double epot) {
  const auto x = e.positions(), v = e.velocities();
  const double *c = e.cell();
  const int *p = e.pbc();
  out << e.n_atoms() << "\n";
  out << fmt::format("Lattice=\"{} {} {} {} {} {} {} {} {}\" pbc=\"{} {} {}\" ", c[0], c[1], c[2], c[3], c[4], c[5],
                     c[6], c[7], c[8], p[0] ? "T" : "F", p[1] ? "T" : "F", p[2] ? "T" : "F")
      << fmt::format("Properties=species:S:1:pos:R:3:vel:R:3 step={} time={:.3f} energy={:.8f}\n", step, time, epot);
  for (int i = 0; i < e.n_atoms(); i++)
    out << fmt::format("{} {:.8f} {:.8f} {:.8f} {:.8e} {:.8e} {:.8e}\n", SYMBOLS[e.numbers()[i]], x[3 * i],
                       x[3 * i + 1], x[3 * i + 2], v[3 * i], v[3 * i + 1], v[3 * i + 2]);
}

struct Acc { // root mean square and mean absolute accumulator
  double se = 0, ae = 0;
  int64_t n = 0;
  void add(double d) { se += d * d, ae += std::abs(d), n++; }
  double rms() const { return n ? std::sqrt(se / n) : NAN; }
  double mae() const { return n ? ae / n : NAN; }
};

AtomicSystem load_system(const Args &a);

// one structure: engine vs CPU energy, and kept edges per atom vs the CPU
// neighbour list at the model cutoff (fixed-cutoff models: the same set)
int edges(const Args &a) {
  const AtomicSystem sys = load_system(a);
  md::Engine eng(a.pos[0], 0, a.num("skin", 1.0));
  eng.set_system(sys);
  const double E = eng.evaluate(true);
  runtime::GraphModel ref;
  ref.set_backend_preference(BackendPreference::CPU);
  ref.load_from_gguf(a.pos[0]);
  const ModelResult r = ref.predict(sys, true);
  const NeighborList nl = NeighborListBuilder(NeighborListOptions{ref.cutoff(), true, false}).build(sys);
  std::vector<int> cpu(sys.num_atoms(), 0);
  for (int e = 0; e < nl.num_pairs(); e++) cpu[nl.centers[e]]++;
  const auto gpu = eng.kept_per_atom();
  const Cell *c = sys.cell();
  if (c)
    for (int k = 0; k < 3; k++)
      fmt::print("cell {}: {:.4f} {:.4f} {:.4f} pbc {}\n", k, c->matrix[k][0], c->matrix[k][1], c->matrix[k][2], c->periodic[k]);
  fmt::print("E engine {:.6f}  cpu {:.6f}  diff/atom {:.3f} meV; edges engine {} cpu {}\n", E, r.energy,
             1e3 * (E - r.energy) / sys.num_atoms(), eng.n_edges(), nl.num_pairs());
  // GraphModel's adaptive radii (solver, double, from the full list) and the
  // edges it keeps: d <= mean of the two radii
  std::vector<std::vector<double>> dist(sys.num_atoms());
  for (int e = 0; e < nl.num_pairs(); e++) {
    const auto &v = nl.edge_vectors[e];
    dist[nl.centers[e]].push_back(std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]) + 1e-15);
  }
  std::vector<double> rad(sys.num_atoms(), ref.cutoff());
  if (ref.num_neighbors_adaptive() > 0)
    for (int i = 0; i < sys.num_atoms(); i++)
      rad[i] = adaptive::solve(static_cast<int>(dist[i].size()), [&](int k) { return dist[i][k]; }, ref.cutoff(),
                               ref.num_neighbors_adaptive(), ref.cutoff_width_adaptive()).r;
  std::vector<int> n_keep(sys.num_atoms(), 0);
  for (int e = 0; e < nl.num_pairs(); e++) {
    const auto &v = nl.edge_vectors[e];
    const double d = std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]) + 1e-15;
    n_keep[nl.centers[e]] += d <= 0.5 * (rad[nl.centers[e]] + rad[nl.neighbors[e]]);
  }
  for (int i = 0; i < sys.num_atoms(); i++)
    fmt::print("  atom {} Z {}: engine {} cpu {} kept {} radius {:.9f}{}\n", i, sys.atomic_numbers()[i], gpu[i], cpu[i],
               n_keep[i], rad[i], gpu[i] != n_keep[i] ? "  <--" : "");
  // translation invariance of the CPU model: at the engine's (wrapped)
  // positions, and with every atom moved by the same vector
  {
    const auto xw = eng.positions();
    std::vector<float> xf(xw.begin(), xw.end()), xt(sys.positions(), sys.positions() + 3 * sys.num_atoms());
    for (size_t k = 0; k < xt.size(); k++) xt[k] += (k % 3 == 0 ? 0.37f : k % 3 == 1 ? -1.21f : 0.58f);
    const AtomicSystem wrapped(sys.num_atoms(), xf.data(), sys.atomic_numbers(), sys.cell()),
        moved(sys.num_atoms(), xt.data(), sys.atomic_numbers(), sys.cell());
    const double ew = ref.predict(wrapped, true).energy, em = ref.predict(moved, true).energy;
    eng.set_positions(std::vector<double>(xt.begin(), xt.end()));
    const double eg = eng.evaluate(true);
    fmt::print("forward only: engine {:.6f}  cpu {:.6f}\n", eng.evaluate(false), ref.predict(sys, false).energy);
    runtime::GraphModel gref;
    gref.set_backend_preference(BackendPreference::CUDA);
    gref.load_from_gguf(a.pos[0]);
    fmt::print("GraphModel on CUDA: forces graph {:.6f}  forward {:.6f}\n", gref.predict(sys, true).energy,
               gref.predict(sys, false).energy);
    fmt::print("translation: cpu orig {:.6f} wrapped {:.6f} moved {:.6f}; engine orig {:.6f} moved {:.6f}\n", r.energy, ew,
               em, E, eg);
    eng.set_system(sys);
    eng.evaluate(true);
  }
  // the engine's inputs: every kept slot's reverse holds -v, the neighbour
  // species is the neighbour's, and the CPU list has an edge of that length
  const auto in = eng.inputs();
  const int M = in.M;
  int bad_rev = 0, bad_rev2 = 0, bad_len = 0, kept = 0;
  double worst = 0;
  for (int i = 0; i < sys.num_atoms(); i++)
    for (int q = 0; q < M; q++) {
      const int f = i * M + q;
      if (in.padding_mask[f] > 0.5f) continue;
      kept++;
      const int r = in.reverse[f];
      double dv = 0;
      for (int c = 0; c < 3; c++) dv = std::max(dv, double(std::abs(in.edge_vectors[3 * f + c] + in.edge_vectors[3 * r + c])));
      worst = std::max(worst, dv);
      bad_rev += dv > 1e-4 || in.padding_mask[r] > 0.5f || r / M == i && r == f;
      bad_rev2 += in.reverse[r] != f;
      const double len = std::sqrt(double(in.edge_vectors[3 * f]) * in.edge_vectors[3 * f] +
                                   double(in.edge_vectors[3 * f + 1]) * in.edge_vectors[3 * f + 1] +
                                   double(in.edge_vectors[3 * f + 2]) * in.edge_vectors[3 * f + 2]);
      bool found = false;
      for (int e = 0; e < nl.num_pairs() && !found; e++)
        found = nl.centers[e] == i && std::abs(nl.distances[e] - len) < 1e-4;
      bad_len += !found;
    }
  fmt::print("inputs: {} kept slots (M {}); reverse not -v: {} (max |v + v_rev| {:.2e}), reverse not an involution: {}, "
             "no CPU edge of that length: {}\n", kept, M, bad_rev, worst, bad_rev2, bad_len);
  return 0;
}

int check(const Args &a) {
  if (a.pos.size() < 2) throw std::runtime_error("usage: mlipcpp-md check MODEL DATA_SPLIT [options]");
  train::MemmapSplit split(a.pos[1]);
  const int want = static_cast<int>(a.num("n", 200)), n_fd = static_cast<int>(a.num("fd", 4));
  const int max_atoms = static_cast<int>(a.num("max-atoms", 256));
  md::Engine eng(a.pos[0], 0, a.num("skin", 1.0));
  const bool use_cpu = a.num("cpu", 1) != 0, use_gpu = a.num("gpu-ref", 0) != 0;
  runtime::GraphModel ref, gref;
  ref.set_backend_preference(BackendPreference::CPU);
  ref.load_from_gguf(a.pos[0]);
  if (use_gpu) gref.set_backend_preference(BackendPreference::CUDA), gref.load_from_gguf(a.pos[0]);
  fmt::print("model {}: cutoff {} A, adaptive {} ({}), direct forces {}\n", a.pos[0], ref.cutoff(),
             ref.num_neighbors_adaptive(), ref.adaptive_cutoff_method(), ref.nc_forces() ? "yes" : "no");
  std::vector<int64_t> ids;
  if (auto f = a.get("ids"); !f.empty()) {
    std::ifstream in(f);
    for (int64_t i; in >> i;) ids.push_back(i);
    fmt::print("{} structures from {}\n", ids.size(), f);
  }
  std::ofstream out_csv;
  std::unique_ptr<std::FILE, int (*)(std::FILE *)> out_f(nullptr, std::fclose);
  if (auto o = a.get("out"); !o.empty()) {
    out_csv.open(o + ".csv"), out_csv << "index,n_atoms,periodic,E_label,E_engine,E_cpu,E_gpu\n";
    out_f.reset(std::fopen((o + ".forces.bin").c_str(), "wb"));
    if (!out_f) throw std::runtime_error("cannot open " + o + ".forces.bin");
  }

  // the trainer's per-structure errors (train::SystemError records)
  std::map<int64_t, std::pair<float, float>> dump;
  if (auto path = a.get("dump"); !path.empty()) {
    std::ifstream in(path, std::ios::binary);
    train::SystemError r;
    while (in.read(reinterpret_cast<char *>(&r), sizeof r)) dump[r.id] = {r.de, r.fse};
    fmt::print("trainer dump {}: {} structures\n", path, dump.size());
  }

  std::vector<double> fd_steps;
  for (std::string t = a.get("fd-steps", "1e-2,3e-3,1e-3") + ","; !t.empty(); t.erase(0, t.find(',') + 1))
    if (t.find(',') > 0) fd_steps.push_back(std::stod(t.substr(0, t.find(','))));
  Acc e_lab, fc_lab, fn_lab, e_ref, fc_ref, fn_ref, de_dump, fse_dump, fd_mag, e_gpu, fn_gpu, fc_gpu;
  std::ofstream csv;
  if (auto c = a.get("csv"); !c.empty()) csv.open(c), csv << "index,n_atoms,periodic,edges,E_engine,E_cpu,E_label,E_trainer\n";
  std::vector<Acc> fd_err(fd_steps.size());
  double fc_ref_max = 0, fn_ref_max = 0, e_ref_max = 0, de_dump_max = 0, fse_dump_rel_max = 0;
  double e_gpu_max = 0, fn_gpu_max = 0, fc_gpu_max = 0;
  int done = 0, n_fd_done = 0;
  const int64_t stride = ids.empty() ? std::max<int64_t>(1, split.size() / want) : 1;
  const int64_t n_try = ids.empty() ? split.size() : static_cast<int64_t>(ids.size());
  for (int64_t t = 0; t < n_try && (!ids.empty() || done < want); t += stride) {
    const int64_t s = ids.empty() ? t : ids[t];
    const int n = split.n_atoms(s);
    if (n < 2 || (ids.empty() && n > max_atoms)) continue;
    const AtomicSystem sys = split.system(s);
    eng.set_system(sys);
    const double E = eng.evaluate(true);
    const auto Fc = eng.forces_conservative(), Fn = eng.forces_direct();
    const ModelResult r = use_cpu ? ref.predict(sys, true) : ModelResult{};
    const ModelResult g = use_gpu ? gref.predict(sys, true) : ModelResult{};

    e_lab.add((E - split.energy(s)) / n);
    if (use_cpu) e_ref.add((E - r.energy) / n), e_ref_max = std::max(e_ref_max, std::abs(E - r.energy) / n);
    if (use_gpu) e_gpu.add((E - g.energy) / n), e_gpu_max = std::max(e_gpu_max, std::abs(E - g.energy) / n);
    double fse = 0;
    std::vector<float> row(21 * static_cast<size_t>(n), NAN);
    for (int k = 0; k < 3 * n; k++) {
      const double lab = split.forces(s)[k];
      float *o = &row[(k / 3) * 21 + k % 3];
      o[0] = lab, o[6] = Fc[k];
      fc_lab.add(Fc[k] - lab);
      if (use_cpu) {
        o[12] = r.forces[k];
        fc_ref.add(Fc[k] - r.forces[k]);
        fc_ref_max = std::max(fc_ref_max, std::abs(Fc[k] - r.forces[k]));
      }
      if (use_gpu) {
        o[18] = g.forces[k];
        fc_gpu.add(Fc[k] - g.forces[k]);
        fc_gpu_max = std::max(fc_gpu_max, std::abs(Fc[k] - g.forces[k]));
      }
      if (eng.has_direct()) {
        o[3] = Fn[k];
        fn_lab.add(Fn[k] - lab);
        fse += (Fn[k] - lab) * (Fn[k] - lab);
        if (use_cpu) {
          o[9] = r.nc_forces[k];
          fn_ref.add(Fn[k] - r.nc_forces[k]);
          fn_ref_max = std::max(fn_ref_max, std::abs(Fn[k] - double(r.nc_forces[k])));
        }
        if (use_gpu) {
          o[15] = g.nc_forces[k];
          fn_gpu.add(Fn[k] - g.nc_forces[k]);
          fn_gpu_max = std::max(fn_gpu_max, std::abs(Fn[k] - double(g.nc_forces[k])));
        }
      }
    }
    if (out_f) {
      out_csv << fmt::format("{},{},{},{:.8f},{:.8f},{:.8f},{:.8f}\n", s, n, sys.cell() ? 1 : 0, split.energy(s), E,
                             use_cpu ? r.energy : NAN, use_gpu ? g.energy : NAN);
      if (std::fwrite(row.data(), sizeof(float), row.size(), out_f.get()) != row.size())
        throw std::runtime_error("cannot write the forces");
    }
    if (csv.is_open()) {
      auto it = dump.find(s);
      csv << fmt::format("{},{},{},{},{:.8f},{:.8f},{:.8f},{}\n", s, n, sys.cell() ? 1 : 0, eng.n_edges(), E,
                         use_cpu ? r.energy : NAN,
                         split.energy(s), it != dump.end() ? fmt::format("{:.8f}", split.energy(s) + it->second.first * n) : "");
    }
    if (auto it = dump.find(s); it != dump.end()) {
      const double de = (E - split.energy(s)) / n - it->second.first;
      de_dump.add(de);
      de_dump_max = std::max(de_dump_max, std::abs(de));
      fse_dump.add(std::sqrt(fse / n) - std::sqrt(it->second.second / n));
      if (it->second.second > 1e-6)
        fse_dump_rel_max = std::max(fse_dump_rel_max, std::abs(fse / it->second.second - 1.0));
    }
    // central differences of the energy (several steps h), the first atoms;
    // structures with max |F_cons| <= --fd-fmax only
    double fmax = 0;
    for (int k = 0; k < 3 * n; k++) fmax = std::max(fmax, std::abs(Fc[k]));
    if (n_fd_done < n_fd && n <= 64 && fmax <= a.num("fd-fmax", 5.0)) {
      const auto x0 = eng.positions();
      for (size_t hi = 0; hi < fd_steps.size(); hi++) {
        const double h = fd_steps[hi];
        for (int k = 0; k < std::min(3 * n, 9); k++) {
          auto x = x0;
          x[k] = x0[k] + h;
          eng.set_positions(x);
          const double ep = eng.evaluate(true);
          x[k] = x0[k] - h;
          eng.set_positions(x);
          const double em = eng.evaluate(true);
          fd_err[hi].add(-(ep - em) / (2 * h) - Fc[k]);
          if (hi == 0) fd_mag.add(Fc[k]);
        }
      }
      eng.set_positions(x0);
      n_fd_done++;
    }
    done++;
  }
  const auto meV = [](const Acc &x) { return 1e3 * x.rms(); };
  if (ids.empty()) fmt::print("\n{} structures (every {}th of {}, <= {} atoms)\n", done, stride, split.size(), max_atoms);
  else fmt::print("\n{} structures (of {} listed; 1-atom structures skipped)\n", done, ids.size());
  fmt::print("vs labels:      E {:.2f} meV/atom   F_direct {:.2f} meV/A   F_conservative {:.2f} meV/A\n",
             meV(e_lab), meV(fn_lab), meV(fc_lab));
  fmt::print("vs labels, MAE: E {:.2f} meV/atom   F_direct {:.2f} meV/A   F_conservative {:.2f} meV/A\n",
             1e3 * e_lab.mae(), 1e3 * fn_lab.mae(), 1e3 * fc_lab.mae());
  if (use_gpu)
    fmt::print("vs CUDA GraphModel: E rms {:.3g} max {:.3g} meV/atom   F_direct rms {:.3g} max {:.3g}   "
               "F_cons rms {:.3g} max {:.3g} meV/A\n",
               meV(e_gpu), 1e3 * e_gpu_max, meV(fn_gpu), 1e3 * fn_gpu_max, meV(fc_gpu), 1e3 * fc_gpu_max);
  if (use_cpu) fmt::print("vs CPU GraphModel: E rms {:.3g} max {:.3g} meV/atom   F_direct rms {:.3g} max {:.3g}   "
             "F_cons rms {:.3g} max {:.3g} meV/A\n",
             meV(e_ref), 1e3 * e_ref_max, meV(fn_ref), 1e3 * fn_ref_max, meV(fc_ref), 1e3 * fc_ref_max);
  if (de_dump.n)
    fmt::print("vs trainer dump ({} matched): dE/atom rms {:.3g} max {:.3g} meV; RMS-F per structure rms diff "
               "{:.3g} meV/A, max rel diff of sum |dF|^2 {:.3g}\n",
               de_dump.n, meV(de_dump), 1e3 * de_dump_max, meV(fse_dump), fse_dump_rel_max);
  for (size_t hi = 0; hi < fd_steps.size(); hi++)
    if (fd_err[hi].n)
      fmt::print("finite differences h={} A ({} components, {} structures, max |F| <= {}): |F_cons - FD| rms "
                 "{:.3g} meV/A (F_cons rms {:.3g} meV/A, ratio {:.3g})\n",
                 fd_steps[hi], fd_err[hi].n, n_fd_done, a.num("fd-fmax", 5.0), meV(fd_err[hi]), meV(fd_mag),
                 fd_err[hi].rms() / fd_mag.rms());
  return 0;
}

AtomicSystem load_system(const Args &a) {
  AtomicSystem sys;
  if (auto xyz = a.get("xyz"); !xyz.empty()) sys = io::read_xyz(xyz);
  else if (auto data = a.get("data"); !data.empty())
    sys = train::MemmapSplit(data).system(static_cast<int64_t>(a.num("index", 0)));
  else throw std::runtime_error("run: give --xyz or --data/--index");
  if (auto r = a.get("repeat"); !r.empty()) {
    int ra = 1, rb = 1, rc = 1;
    if (std::sscanf(r.c_str(), "%d,%d,%d", &ra, &rb, &rc) != 3) throw std::runtime_error("--repeat a,b,c");
    sys = repeat(sys, ra, rb, rc);
  }
  return sys;
}

// Work-energy check along a straight path: every atom moves along one random
// unit direction u (3n components), x(t) = x0 + t u; prints E(t), F_cons . u,
// the kept edge count, and the running -integral of F . u (trapezoid), which
// equals E(t) - E(0) for exact gradients of a continuous energy.
int path(const Args &a) {
  if (a.pos.empty()) throw std::runtime_error("usage: mlipcpp-md path MODEL (--xyz | --data --index) [--length 0.5] [--points 500]");
  const AtomicSystem sys = load_system(a);
  md::Engine eng(a.pos[0], 0, a.num("skin", 1.0));
  eng.set_system(sys);
  const int n = eng.n_atoms(), points = static_cast<int>(a.num("points", 500));
  const double length = a.num("length", 0.5);
  std::mt19937_64 rng(static_cast<uint64_t>(a.num("seed", 1)));
  std::normal_distribution<double> normal;
  std::vector<double> u(3 * n);
  double norm = 0;
  for (double &x : u) x = normal(rng), norm += x * x;
  for (double &x : u) x /= std::sqrt(norm);
  const auto x0 = eng.positions();
  double e0 = 0, work = 0, fu_prev = 0;
  fmt::print("t_A\tE_eV\tdE_eV\tFdotu_eV_A\tminus_work_eV\tedges\tmax_nb\n");
  for (int p = 0; p <= points; p++) {
    const double t = length * p / points;
    auto x = x0;
    for (int k = 0; k < 3 * n; k++) x[k] += t * u[k];
    eng.set_positions(x);
    const double E = eng.evaluate(true);
    const auto F = eng.forces_conservative();
    double fu = 0;
    for (int k = 0; k < 3 * n; k++) fu += F[k] * u[k];
    if (p == 0) e0 = E;
    else work += 0.5 * (fu + fu_prev) * (length / points);
    fu_prev = fu;
    fmt::print("{:.6f}\t{:.8f}\t{:.8f}\t{:.6f}\t{:.8f}\t{}\t{}\n", t, E, E - e0, fu, -work, eng.n_edges(),
               eng.max_neighbors());
  }
  return 0;
}

int run(const Args &a) {
  if (a.pos.empty()) throw std::runtime_error("usage: mlipcpp-md run MODEL (--xyz FILE | --data SPLIT --index I)");
  const AtomicSystem sys = load_system(a);
  const std::string integ_name = a.get("integrator", "respa");
  const auto integ = integ_name == "respa"          ? md::Engine::Integrator::Respa
                     : integ_name == "conservative" ? md::Engine::Integrator::Conservative
                     : integ_name == "direct"       ? md::Engine::Integrator::Direct
                                                    : throw std::runtime_error("--integrator respa|conservative|direct");
  const int inner = integ == md::Engine::Integrator::Respa ? static_cast<int>(a.num("inner", 4)) : 1;
  const double dt = a.num("dt", 0.5), T0 = a.num("temperature", 300);
  const int64_t steps = static_cast<int64_t>(a.num("steps", 1000));
  const int64_t log_every = std::max<int64_t>(inner, static_cast<int64_t>(a.num("log-every", 10)));
  const int64_t traj_every = static_cast<int64_t>(a.num("traj-every", 100));

  md::Engine eng(a.pos[0], 0, a.num("skin", 1.0));
  eng.set_system(sys);
  eng.maxwell_boltzmann(T0, static_cast<uint64_t>(a.num("seed", 1)));
  const double e0 = eng.evaluate(true);
  const double etot0 = e0 + eng.kinetic_energy();
  const int n = eng.n_atoms();
  fmt::print("{} atoms, pbc {}{}{}, integrator {} (inner {}), dt {} fs, {} steps, T0 {} K\n", n, eng.pbc()[0],
             eng.pbc()[1], eng.pbc()[2], integ_name, inner, dt, steps, T0);

  std::ofstream traj, thermo;
  if (auto t = a.get("traj"); !t.empty()) traj.open(t), write_frame(traj, eng, 0, 0, e0);
  if (auto t = a.get("thermo"); !t.empty()) thermo.open(t);
  const std::string head = "step\ttime_fs\tEpot_eV\tEkin_eV\tEtot_eV\tT_K\tdrift_meV_atom\tmax_nb\tM\tlists\tms_per_step";
  fmt::print("{}\n", head);
  if (thermo.is_open()) thermo << head << "\n";
  auto t0 = std::chrono::steady_clock::now();
  int64_t last = 0;
  for (int64_t s = inner; s <= steps; s += inner) {
    const bool log = s % log_every < inner || s + inner > steps;
    eng.step(integ, dt, inner, log);
    if (log) {
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
                        double(s - last);
      const double ek = eng.kinetic_energy(), et = eng.energy() + ek;
      const std::string line =
          fmt::format("{}\t{:.2f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.1f}\t{:.4f}\t{}\t{}\t{}\t{:.3f}", s, s * dt, eng.energy(),
                      ek, et, eng.temperature(), 1e3 * (et - etot0) / n, eng.max_neighbors(), eng.capacity_neighbors(),
                      eng.list_builds(), ms);
      fmt::print("{}\n", line);
      if (thermo.is_open()) thermo << line << "\n" << std::flush;
      if (!std::isfinite(et)) throw std::runtime_error("energy is not finite");
      t0 = std::chrono::steady_clock::now(), last = s;
    }
    if (traj.is_open() && traj_every > 0 && s % traj_every < inner) write_frame(traj, eng, s, s * dt, eng.energy());
  }
  fmt::print("graph builds {}, list builds {}\n", eng.graph_builds(), eng.list_builds());
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) throw std::runtime_error("usage: mlipcpp-md check|run ...");
    const std::string cmd = argv[1];
    const Args a = parse(argc, argv);
    if (cmd == "check") return check(a);
    if (cmd == "run") return run(a);
    if (cmd == "path") return path(a);
    if (cmd == "edges") return edges(a);
    throw std::runtime_error("unknown command " + cmd);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mlipcpp-md: %s\n", e.what());
    return 1;
  }
}
