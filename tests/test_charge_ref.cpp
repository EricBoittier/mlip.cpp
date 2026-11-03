// mlipcpp-train --energy-add-charge-ref B (mmml --energy-add-charge-ref): MemmapSplit's target
// energies are E + B Q, Q from charge.bin; the files are not changed, B = 0 is the stored value.
#include "train/dataset.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using mlipcpp::train::MemmapSplit;
namespace fs = std::filesystem;

namespace {
template <typename T> void write_bin(const fs::path &p, const std::vector<T> &v) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
}
// little-endian int64 .npy, C order (shape () for a scalar)
void write_npy(const fs::path &p, const std::vector<int64_t> &v, bool scalar) {
  std::string h = "{'descr': '<i8', 'fortran_order': False, 'shape': " +
                  (scalar ? std::string("()") : "(" + std::to_string(v.size()) + ",)") + ", }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::ofstream f(p, std::ios::binary);
  const uint16_t len = uint16_t(h.size());
  f.write("\x93NUMPY\x01\x00", 8).write(reinterpret_cast<const char *>(&len), 2).write(h.data(), std::streamsize(h.size()));
  f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * 8));
}
} // namespace

TEST_CASE("energy-add-charge-ref: E + B Q", "[dataset]") {
  const fs::path dir = fs::temp_directory_path() / ("mlipcpp_charge_ref_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  const std::vector<int> nat = {2, 3, 1, 2};
  const std::vector<float> E = {-12.25f, 3.5f, 0.125f, -1000.75f}, Q = {0, -1, 2, 1};
  int N = 0;
  std::vector<int64_t> na = {0};
  for (int n : nat) na.push_back(N += n);
  write_npy(dir / "ns.npy", {int64_t(nat.size())}, true);
  write_npy(dir / "na.npy", na, false);
  write_bin(dir / "x.bin", std::vector<float>(3 * N, 0.5f));
  write_bin(dir / "a.bin", std::vector<int32_t>(N, 1));
  write_bin(dir / "c.bin", std::vector<float>(9 * nat.size(), 0.0f));
  write_bin(dir / "energy.bin", E);
  write_bin(dir / "non_conservative_force.bin", std::vector<float>(3 * N, 0.0f));
  write_bin(dir / "charge.bin", Q);
  write_bin(dir / "spin_multiplicity.bin", std::vector<float>(nat.size(), 1.0f));
  const auto before = fs::file_size(dir / "energy.bin");

  MemmapSplit split(dir.string());
  for (size_t s = 0; s < E.size(); s++) CHECK(split.energy(s) == E[s]); // off: the stored value, bit for bit
  const double B = 5.661417917773716;                                   // OMol25 pilot200k b_Q (eV / e)
  split.add_charge_reference(B);
  for (size_t s = 0; s < E.size(); s++) {
    CHECK(split.charge(s) == int(Q[s]));
    CHECK(split.energy(s) == static_cast<float>(double(E[s]) + B * Q[s]));
  }
  CHECK(split.energy(0) == E[0]);                       // Q = 0: unchanged
  CHECK(split.energy(1) == static_cast<float>(3.5 - B)); // Q = -1
  MemmapSplit again(dir.string());                      // the files are not modified
  for (size_t s = 0; s < E.size(); s++) CHECK(again.energy(s) == E[s]);
  CHECK(fs::file_size(dir / "energy.bin") == before);
  fs::remove_all(dir);
}
