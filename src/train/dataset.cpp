#include "train/dataset.h"

#include "core/adaptive_cutoff.h"

#include "core/log.h"
#include "mlipcpp/neighbor_list.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace mlipcpp::train {

template <typename T> Mapped<T>::Mapped(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  ::fstat(fd, &st);
  size_ = st.st_size / sizeof(T);
  if (size_ > 0) {
    void *p = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("cannot mmap " + path);
    }
    data_ = static_cast<const T *>(p);
  }
  ::close(fd);
}

template <typename T> Mapped<T> &Mapped<T>::operator=(Mapped &&o) noexcept {
  std::swap(data_, o.data_);
  std::swap(size_, o.size_);
  return *this;
}

template <typename T> Mapped<T>::~Mapped() {
  if (data_) ::munmap(const_cast<T *>(data_), size_ * sizeof(T));
}

template class Mapped<float>;
template class Mapped<int32_t>;

std::vector<int64_t> read_npy_i64(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  uint8_t version[2];
  f.read(magic, 6).read(reinterpret_cast<char *>(version), 2);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
    throw std::runtime_error(path + " is not a .npy file");
  uint32_t header_len = 0;
  if (version[0] == 1) {
    uint16_t h;
    f.read(reinterpret_cast<char *>(&h), 2);
    header_len = h;
  } else {
    f.read(reinterpret_cast<char *>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);
  if (header.find("'<i8'") == std::string::npos ||
      header.find("'fortran_order': False") == std::string::npos)
    throw std::runtime_error(path + ": expected C-order '<i8', got " + header);
  std::vector<int64_t> out;
  int64_t v;
  while (f.read(reinterpret_cast<char *>(&v), sizeof v)) out.push_back(v);
  return out;
}

MemmapSplit::MemmapSplit(const std::string &dir) : dir_(dir) {
  n_systems_ = read_npy_i64(dir + "/ns.npy").at(0);
  na_ = read_npy_i64(dir + "/na.npy");
  if (static_cast<int64_t>(na_.size()) != n_systems_ + 1)
    throw std::runtime_error(dir + ": na.npy has " + std::to_string(na_.size()) +
                             " entries for " + std::to_string(n_systems_) + " systems");
  x_ = Mapped<float>(dir + "/x.bin");
  a_ = Mapped<int32_t>(dir + "/a.bin");
  c_ = Mapped<float>(dir + "/c.bin");
  energy_ = Mapped<float>(dir + "/energy.bin");
  force_ = Mapped<float>(dir + "/non_conservative_force.bin");
  const size_t N = na_.back(), ns = n_systems_;
  if (x_.size() != 3 * N || a_.size() != N || c_.size() != 9 * ns ||
      energy_.size() != ns || force_.size() != 3 * N)
    throw std::runtime_error(dir + ": array sizes do not match ns.npy / na.npy");
}

AtomicSystem MemmapSplit::system(int64_t s) const {
  const float *c = cell(s);
  bool pbc[3];
  for (int k = 0; k < 3; k++)
    pbc[k] = c[3 * k] != 0.0f || c[3 * k + 1] != 0.0f || c[3 * k + 2] != 0.0f;
  if (!pbc[0] && !pbc[1] && !pbc[2])
    return AtomicSystem(n_atoms(s), positions(s), numbers(s));

  float lattice[3][3];
  std::memcpy(lattice, c, sizeof lattice);
  // Partially periodic (slabs, wires): fill each zero row with a long vector
  // orthogonal to the periodic ones; the direction stays non-periodic.
  for (int k = 0; k < 3; k++) {
    if (pbc[k]) continue;
    const float *u = lattice[(k + 1) % 3], *v = lattice[(k + 2) % 3];
    float w[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                  u[0] * v[1] - u[1] * v[0]};
    float norm = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (norm == 0.0f) { // a wire: cross the periodic row with an axis not parallel to it
      const float *p = pbc[(k + 1) % 3] ? u : v;
      const float pn = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      const float axis[3] = {std::fabs(p[0]) < 0.9f * pn ? 1.0f : 0.0f,
                             std::fabs(p[0]) < 0.9f * pn ? 0.0f : 1.0f, 0.0f};
      w[0] = p[1] * axis[2] - p[2] * axis[1];
      w[1] = p[2] * axis[0] - p[0] * axis[2];
      w[2] = p[0] * axis[1] - p[1] * axis[0];
      norm = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    }
    for (int d = 0; d < 3; d++) lattice[k][d] = 1000.0f * w[d] / norm;
  }
  Cell cell_obj(lattice, pbc[0], pbc[1], pbc[2]);
  return AtomicSystem(n_atoms(s), positions(s), numbers(s), &cell_obj);
}

namespace {

constexpr uint64_t CACHE_MAGIC = 0x314c4e5050494c4dULL;          // "MLIPPNL1"
constexpr uint64_t CACHE_MAGIC_ADAPTIVE = 0x324c4e5050494c4dULL; // "MLIPPNL2": + n_target, width, pair_cutoff

struct LocalEdges {
  std::vector<int32_t> center, neighbor, slot, reverse;
  std::vector<std::array<float, 3>> vector;
  std::vector<float> pair_cutoff;
  int32_t max_neighbors = 0;
};

// metatrain's "solver" adaptive cutoff (core/adaptive_cutoff.h), in double.
// Edges sorted by centre: atom i's are [start[i], start[i+1]).
std::vector<double> adaptive_radii(const std::vector<int32_t> &start, const std::vector<double> &dist,
                                   double cutoff, const AdaptiveCutoff &a) {
  const int n = static_cast<int>(start.size()) - 1;
  std::vector<double> radii(n);
  for (int i = 0; i < n; i++)
    radii[i] = adaptive::solve(start[i + 1] - start[i], [&](int k) { return dist[start[i] + k]; },
                               cutoff, a.n_target, a.width).r;
  return radii;
}

// keep the edges within their pair cutoff (one decision per edge pair, so
// the lists stay symmetric), then renumber slot / reverse
void apply_adaptive(LocalEdges &e, int n, double cutoff, const AdaptiveCutoff &a) {
  const int E = static_cast<int>(e.center.size());
  std::vector<int32_t> start(n + 1, 0);
  for (int k = 0; k < E; k++) start[e.center[k] + 1]++;
  for (int i = 0; i < n; i++) start[i + 1] += start[i];
  std::vector<double> dist(E);
  for (int k = 0; k < E; k++) {
    const auto &v = e.vector[k];
    dist[k] = std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]) + 1e-15;
  }
  const std::vector<double> radii = adaptive_radii(start, dist, cutoff, a);
  std::vector<char> keep(E);
  std::vector<double> pc(E);
  for (int k = 0; k < E; k++) pc[k] = 0.5 * (radii[e.center[k]] + radii[e.neighbor[k]]);
  for (int k = 0; k < E; k++) {
    const int r = e.reverse[k], first = r >= 0 ? std::min(k, r) : k;
    keep[k] = dist[first] <= pc[first];
  }
  std::vector<int32_t> new_index(E, -1), count(n, 0);
  LocalEdges out;
  for (int k = 0; k < E; k++) {
    if (!keep[k]) continue;
    new_index[k] = static_cast<int32_t>(out.center.size());
    out.center.push_back(e.center[k]);
    out.neighbor.push_back(e.neighbor[k]);
    out.vector.push_back(e.vector[k]);
    out.slot.push_back(count[e.center[k]]++);
    out.pair_cutoff.push_back(static_cast<float>(pc[k]));
  }
  out.reverse.resize(out.center.size());
  for (int k = 0; k < E; k++)
    if (keep[k]) out.reverse[new_index[k]] = e.reverse[k] >= 0 ? new_index[e.reverse[k]] : -1;
  out.max_neighbors = n ? *std::max_element(count.begin(), count.end()) : 0;
  e = std::move(out);
}

LocalEdges build_one(const MemmapSplit &split, int64_t s,
                     const NeighborListBuilder &builder, const AdaptiveCutoff &adaptive, double cutoff) {
  NeighborList nl = builder.build(split.system(s));
  const int E = nl.num_pairs(), n = split.n_atoms(s);
  std::vector<int> order(E);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return nl.centers[a] < nl.centers[b]; });

  LocalEdges out;
  out.center.resize(E), out.neighbor.resize(E), out.slot.resize(E);
  out.reverse.assign(E, -1), out.vector.resize(E);
  std::vector<int32_t> count(n, 0);
  const bool shifts = !nl.cell_shifts.empty();
  std::vector<std::array<int32_t, 3>> shift(E, {0, 0, 0});
  for (int k = 0; k < E; k++) {
    const int e = order[k];
    out.center[k] = nl.centers[e];
    out.neighbor[k] = nl.neighbors[e];
    out.vector[k] = nl.edge_vectors[e];
    out.slot[k] = count[nl.centers[e]]++;
    if (shifts) shift[k] = nl.cell_shifts[e];
  }
  out.max_neighbors = n ? *std::max_element(count.begin(), count.end()) : 0;

  // reverse edge: (j, i, -shift); edges of each centre are contiguous
  std::vector<int32_t> start(n + 1, 0);
  for (int i = 0; i < n; i++) start[i + 1] = start[i] + count[i];
  for (int k = 0; k < E; k++) {
    const int i = out.center[k], j = out.neighbor[k];
    for (int r = start[j]; r < start[j + 1]; r++) {
      if (out.neighbor[r] == i && shift[r][0] == -shift[k][0] &&
          shift[r][1] == -shift[k][1] && shift[r][2] == -shift[k][2]) {
        out.reverse[k] = r;
        break;
      }
    }
  }
  if (adaptive.enabled()) apply_adaptive(out, n, cutoff, adaptive);
  return out;
}

// the full list of build_one (before apply_adaptive), edges sorted by centre
TaperEdges taper_one(const MemmapSplit &split, int64_t s, double cutoff, const AdaptiveCutoff &a) {
  const NeighborList nl = NeighborListBuilder(NeighborListOptions{static_cast<float>(cutoff), true, false})
                              .build(split.system(s));
  const int E = nl.num_pairs(), n = split.n_atoms(s);
  std::vector<int> order(E);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return nl.centers[x] < nl.centers[y]; });
  std::vector<int32_t> start(n + 1, 0);
  std::vector<double> dist(E);
  for (int k = 0; k < E; k++) {
    const auto &v = nl.edge_vectors[order[k]];
    start[nl.centers[order[k]] + 1]++;
    dist[k] = std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]) + 1e-15;
  }
  for (int i = 0; i < n; i++) start[i + 1] += start[i];
  TaperEdges t;
  for (int i = 0; i < n; i++) {
    const adaptive::Radius r = adaptive::solve(start[i + 1] - start[i], [&](int k) { return dist[start[i] + k]; },
                                               cutoff, a.n_target, a.width);
    for (int k = start[i]; k < start[i + 1]; k++) {
      const double J = adaptive::ddist(dist[k], r, a.width);
      if (J == 0.0) continue;
      t.center.push_back(i);
      t.neighbor.push_back(nl.neighbors[order[k]]);
      t.vector.push_back(nl.edge_vectors[order[k]]);
      t.jacobian.push_back(static_cast<float>(J));
    }
  }
  return t;
}

template <typename V> void write_vec(std::ofstream &f, const std::vector<V> &v) {
  f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(V));
}
template <typename V> void read_vec(std::ifstream &f, std::vector<V> &v, size_t n) {
  v.resize(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(V));
}

} // namespace

NeighborCache NeighborCache::load_or_build(const MemmapSplit &split, float cutoff,
                                           const std::string &cache_path,
                                           int n_threads, const AdaptiveCutoff &adaptive) {
  const int64_t ns = split.size();
  const int64_t n_atoms_total = split.total_atoms();
  NeighborCache nc;
  nc.cutoff = cutoff;
  nc.adaptive = adaptive;
  const uint64_t magic_expected = adaptive.enabled() ? CACHE_MAGIC_ADAPTIVE : CACHE_MAGIC;

  if (!cache_path.empty() && std::filesystem::exists(cache_path)) {
    std::ifstream f(cache_path, std::ios::binary);
    uint64_t magic = 0;
    int64_t c_ns = 0, c_atoms = 0, c_edges = 0;
    float c_cut = 0;
    f.read(reinterpret_cast<char *>(&magic), 8);
    f.read(reinterpret_cast<char *>(&c_ns), 8);
    f.read(reinterpret_cast<char *>(&c_atoms), 8);
    f.read(reinterpret_cast<char *>(&c_cut), 4);
    f.read(reinterpret_cast<char *>(&c_edges), 8);
    double c_adaptive[2] = {0, 0};
    if (magic == CACHE_MAGIC_ADAPTIVE) f.read(reinterpret_cast<char *>(c_adaptive), sizeof c_adaptive);
    if (f && magic == magic_expected && c_ns == ns && c_atoms == n_atoms_total && c_cut == cutoff &&
        (!adaptive.enabled() || (c_adaptive[0] == adaptive.n_target && c_adaptive[1] == adaptive.width))) {
      read_vec(f, nc.edge_offset, ns + 1);
      read_vec(f, nc.max_neighbors, ns);
      read_vec(f, nc.center, c_edges);
      read_vec(f, nc.neighbor, c_edges);
      read_vec(f, nc.slot, c_edges);
      read_vec(f, nc.reverse, c_edges);
      read_vec(f, nc.vector, c_edges);
      if (adaptive.enabled()) read_vec(f, nc.pair_cutoff, c_edges);
      if (f) {
        log::info("neighbour lists: loaded {} edges from {}", c_edges, cache_path);
        return nc;
      }
    }
    log::warn("neighbour-list cache {} is stale or truncated; rebuilding", cache_path);
  }

  const NeighborListBuilder builder(NeighborListOptions{cutoff, true, false});
  auto parallel = [&](auto &&f) {
    std::atomic<int64_t> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < std::max(1, n_threads); t++)
      pool.emplace_back([&] {
        for (int64_t s; (s = next.fetch_add(1)) < ns;) f(s);
      });
    for (auto &t : pool) t.join();
  };
  // two passes, so that peak memory is the final arrays (one allocation)
  // plus per-thread scratch: 2^24 structures hold ~4e9 edges (~120 GB).
  // Pass 1: edge counts (neighbour search only; with an adaptive cutoff the
  // full build); pass 2: build every structure straight into its slice of
  // the final arrays.
  std::vector<int64_t> count(ns);
  parallel([&](int64_t s) {
    count[s] = adaptive.enabled() ? static_cast<int64_t>(build_one(split, s, builder, adaptive, cutoff).center.size())
                                  : builder.build(split.system(s)).num_pairs();
  });
  nc.edge_offset.assign(ns + 1, 0);
  for (int64_t s = 0; s < ns; s++) nc.edge_offset[s + 1] = nc.edge_offset[s] + count[s];
  const int64_t E = nc.edge_offset.back();
  nc.max_neighbors.resize(ns);
  nc.center.resize(E), nc.neighbor.resize(E), nc.slot.resize(E);
  nc.reverse.resize(E), nc.vector.resize(E);
  if (adaptive.enabled()) nc.pair_cutoff.resize(E);
  parallel([&](int64_t s) {
    LocalEdges p = build_one(split, s, builder, adaptive, cutoff);
    if (static_cast<int64_t>(p.center.size()) != count[s])
      throw std::runtime_error("neighbour lists: edge count changed between passes");
    const int64_t o = nc.edge_offset[s];
    std::copy(p.center.begin(), p.center.end(), nc.center.begin() + o);
    std::copy(p.neighbor.begin(), p.neighbor.end(), nc.neighbor.begin() + o);
    std::copy(p.slot.begin(), p.slot.end(), nc.slot.begin() + o);
    std::copy(p.reverse.begin(), p.reverse.end(), nc.reverse.begin() + o);
    std::copy(p.vector.begin(), p.vector.end(), nc.vector.begin() + o);
    std::copy(p.pair_cutoff.begin(), p.pair_cutoff.end(), nc.pair_cutoff.begin() + o);
    nc.max_neighbors[s] = p.max_neighbors;
  });
  log::info("neighbour lists: built {} edges for {} structures", E, ns);

  if (!cache_path.empty()) {
    std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path());
    const std::string tmp = cache_path + ".tmp." + std::to_string(::getpid());
    {
      std::ofstream f(tmp, std::ios::binary);
      f.write(reinterpret_cast<const char *>(&magic_expected), 8);
      f.write(reinterpret_cast<const char *>(&ns), 8);
      f.write(reinterpret_cast<const char *>(&n_atoms_total), 8);
      f.write(reinterpret_cast<const char *>(&cutoff), 4);
      f.write(reinterpret_cast<const char *>(&E), 8);
      if (adaptive.enabled()) {
        const double a[2] = {adaptive.n_target, adaptive.width};
        f.write(reinterpret_cast<const char *>(a), sizeof a);
      }
      write_vec(f, nc.edge_offset);
      write_vec(f, nc.max_neighbors);
      write_vec(f, nc.center);
      write_vec(f, nc.neighbor);
      write_vec(f, nc.slot);
      write_vec(f, nc.reverse);
      write_vec(f, nc.vector);
      if (adaptive.enabled()) write_vec(f, nc.pair_cutoff);
      f.close();
      if (!f) {
        // e.g. quota or disk full: never leave a large partial file behind
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
        log::warn("neighbour lists: could not write the cache {} (not cached)", cache_path);
        return nc;
      }
    }
    std::filesystem::rename(tmp, cache_path); // atomic: readers never see a partial file
    log::info("neighbour lists: cached to {}", cache_path);
  }
  return nc;
}

TaperEdges taper_edges(const MemmapSplit &split, int64_t s, float cutoff, const AdaptiveCutoff &adaptive) {
  return adaptive.enabled() ? taper_one(split, s, cutoff, adaptive) : TaperEdges{};
}

} // namespace mlipcpp::train
