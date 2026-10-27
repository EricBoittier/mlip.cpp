// Charge / spin conditioning inputs: extended-XYZ keys, the range checks (an error, never
// clamped), and AtomicSystem's defaults. (Model parity: scripts/conditioning/parity_pet.py.)
#include "core/conditioning.h"
#include "mlipcpp/io.h"
#include "mlipcpp/system.h"

#include <catch2/catch_test_macros.hpp>

#include <sstream>

using namespace mlipcpp;

TEST_CASE("conditioning: extxyz charge and spin", "[conditioning]") {
  auto read = [](const std::string &comment) {
    std::istringstream in("2\n" + comment + "\nH 0 0 0\nH 0 0 0.74\n");
    return io::read_xyz(in);
  };
  AtomicSystem a = read("Properties=species:S:1:pos:R:3");
  CHECK(a.charge() == 0);
  CHECK(a.spin_multiplicity() == 1);
  a = read("charge=-1 spin_multiplicity=2 energy=1.0");
  CHECK(a.charge() == -1);
  CHECK(a.spin_multiplicity() == 2);
  a = read("charge=1.0 spin=3"); // OMol's key; integer-valued reals
  CHECK(a.charge() == 1);
  CHECK(a.spin_multiplicity() == 3);
  a = read("total_charge=5 charge=\"2\"");
  CHECK(a.charge() == 2);
  CHECK_THROWS(read("charge=0.5"));
}

TEST_CASE("conditioning: ranges", "[conditioning]") {
  const Conditioning c{true, 3, 4};
  CHECK(c.rows(0, 1) == std::make_pair(3, 0));
  CHECK(c.rows(-3, 4) == std::make_pair(0, 3));
  CHECK(c.rows(3, 1) == std::make_pair(6, 0));
  CHECK_THROWS(c.rows(4, 1));
  CHECK_THROWS(c.rows(-4, 1));
  CHECK_THROWS(c.rows(0, 0));
  CHECK_THROWS(c.rows(0, 5));
}
