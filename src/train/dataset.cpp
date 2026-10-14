#include "train/dataset.h"

#include "core/log.h"
#include "mlipcpp/neighbor_list.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
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

namespace {

/// int64 payload of a little-endian .npy file (any shape, '<i8' only).
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

} // namespace

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

constexpr uint64_t CACHE_MAGIC = 0x314c4e5050494c4dULL; // "MLIPPNL1"

struct LocalEdges {
  std::vector<int32_t> center, neighbor, slot, reverse;
  std::vector<std::array<float, 3>> vector;
  int32_t max_neighbors = 0;
};

LocalEdges build_one(const MemmapSplit &split, int64_t s,
                     const NeighborListBuilder &builder) {
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
  return out;
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
                                           int n_threads) {
  const int64_t ns = split.size();
  const int64_t n_atoms_total = split.total_atoms();
  NeighborCache nc;
  nc.cutoff = cutoff;

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
    if (f && magic == CACHE_MAGIC && c_ns == ns && c_atoms == n_atoms_total &&
        c_cut == cutoff) {
      read_vec(f, nc.edge_offset, ns + 1);
      read_vec(f, nc.max_neighbors, ns);
      read_vec(f, nc.center, c_edges);
      read_vec(f, nc.neighbor, c_edges);
      read_vec(f, nc.slot, c_edges);
      read_vec(f, nc.reverse, c_edges);
      read_vec(f, nc.vector, c_edges);
      if (f) {
        log::info("neighbour lists: loaded {} edges from {}", c_edges, cache_path);
        return nc;
      }
    }
    log::warn("neighbour-list cache {} is stale or truncated; rebuilding", cache_path);
  }

  const NeighborListBuilder builder(NeighborListOptions{cutoff, true, false});
  std::vector<LocalEdges> parts(ns);
  std::atomic<int64_t> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, n_threads); t++) {
    pool.emplace_back([&] {
      for (int64_t s; (s = next.fetch_add(1)) < ns;) parts[s] = build_one(split, s, builder);
    });
  }
  for (auto &t : pool) t.join();

  nc.edge_offset.assign(ns + 1, 0);
  nc.max_neighbors.resize(ns);
  for (int64_t s = 0; s < ns; s++) {
    nc.edge_offset[s + 1] = nc.edge_offset[s] + parts[s].center.size();
    nc.max_neighbors[s] = parts[s].max_neighbors;
  }
  const int64_t E = nc.edge_offset.back();
  nc.center.reserve(E), nc.neighbor.reserve(E), nc.slot.reserve(E);
  nc.reverse.reserve(E), nc.vector.reserve(E);
  for (auto &p : parts) {
    nc.center.insert(nc.center.end(), p.center.begin(), p.center.end());
    nc.neighbor.insert(nc.neighbor.end(), p.neighbor.begin(), p.neighbor.end());
    nc.slot.insert(nc.slot.end(), p.slot.begin(), p.slot.end());
    nc.reverse.insert(nc.reverse.end(), p.reverse.begin(), p.reverse.end());
    nc.vector.insert(nc.vector.end(), p.vector.begin(), p.vector.end());
    p = LocalEdges{};
  }
  log::info("neighbour lists: built {} edges for {} structures", E, ns);

  if (!cache_path.empty()) {
    std::filesystem::create_directories(std::filesystem::path(cache_path).parent_path());
    const std::string tmp = cache_path + ".tmp." + std::to_string(::getpid());
    {
      std::ofstream f(tmp, std::ios::binary);
      const uint64_t magic = CACHE_MAGIC;
      f.write(reinterpret_cast<const char *>(&magic), 8);
      f.write(reinterpret_cast<const char *>(&ns), 8);
      f.write(reinterpret_cast<const char *>(&n_atoms_total), 8);
      f.write(reinterpret_cast<const char *>(&cutoff), 4);
      f.write(reinterpret_cast<const char *>(&E), 8);
      write_vec(f, nc.edge_offset);
      write_vec(f, nc.max_neighbors);
      write_vec(f, nc.center);
      write_vec(f, nc.neighbor);
      write_vec(f, nc.slot);
      write_vec(f, nc.reverse);
      write_vec(f, nc.vector);
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

} // namespace mlipcpp::train
