#include "core/conditioning.h"
#include "mlipcpp/io.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mlipcpp {
namespace io {

// Element symbols indexed by atomic number (index 0 unused). Covers the full
// periodic table (Z 1-118) so any element can be read/written.
static const char *const ELEMENT_SYMBOLS[] = {
    "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
    "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
    "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

static constexpr int MAX_ATOMIC_NUMBER =
    static_cast<int>(sizeof(ELEMENT_SYMBOLS) / sizeof(ELEMENT_SYMBOLS[0])) - 1;

static int element_to_atomic_number(const std::string &symbol) {
  for (int Z = 1; Z <= MAX_ATOMIC_NUMBER; ++Z) {
    if (symbol == ELEMENT_SYMBOLS[Z]) {
      return Z;
    }
  }

  throw std::runtime_error("Unknown element symbol: " + symbol);
}

bool parse_exyz_lattice(const std::string &comment_line, double cell[3][3]) {
  // Look for: Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"
  size_t lattice_pos = comment_line.find("Lattice=\"");
  if (lattice_pos == std::string::npos) {
    return false;
  }

  size_t start = lattice_pos + 9; // Skip 'Lattice="'
  size_t end = comment_line.find("\"", start);
  if (end == std::string::npos) {
    return false;
  }

  std::string lattice_str = comment_line.substr(start, end - start);
  std::istringstream iss(lattice_str);

  // Parse 9 values for 3x3 cell matrix
  double values[9];
  for (int i = 0; i < 9; ++i) {
    if (!(iss >> values[i])) {
      return false;
    }
  }

  // Store in cell array (row-major)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cell[i][j] = values[i * 3 + j];
    }
  }

  return true;
}

// pbc="T T F" (ASE: T / F / True / False / 1 / 0); absent: periodic in every direction of the Lattice
static void parse_exyz_pbc(const std::string &comment_line, bool pbc[3]) {
  pbc[0] = pbc[1] = pbc[2] = true;
  for (size_t pos = comment_line.find("pbc="); pos != std::string::npos; pos = comment_line.find("pbc=", pos + 1)) {
    if (pos > 0 && !std::isspace(static_cast<unsigned char>(comment_line[pos - 1]))) continue;
    size_t start = pos + 4;
    const bool quoted = start < comment_line.size() && comment_line[start] == '"';
    if (quoted) start++;
    const size_t end = quoted ? comment_line.find('"', start) : comment_line.find_first_of(" \t\r", start);
    std::istringstream iss(comment_line.substr(start, end == std::string::npos ? std::string::npos : end - start));
    for (int k = 0; k < 3; k++) {
      std::string t;
      if (!(iss >> t)) throw std::runtime_error("XYZ: cannot parse pbc= in: " + comment_line);
      if (t == "T" || t == "True" || t == "true" || t == "1") pbc[k] = true;
      else if (t == "F" || t == "False" || t == "false" || t == "0") pbc[k] = false;
      else throw std::runtime_error("XYZ: cannot parse pbc= in: " + comment_line);
    }
    return;
  }
}

bool parse_exyz_number(const std::string &comment_line, const std::string &key, double &value) {
  // key=value or key="value", the key at the start of the line or after a space
  for (size_t pos = comment_line.find(key + "="); pos != std::string::npos;
       pos = comment_line.find(key + "=", pos + 1)) {
    if (pos > 0 && !std::isspace(static_cast<unsigned char>(comment_line[pos - 1]))) continue;
    size_t start = pos + key.size() + 1;
    if (start < comment_line.size() && comment_line[start] == '"') start++;
    std::istringstream iss(comment_line.substr(start));
    if (!(iss >> value)) throw std::runtime_error("XYZ: cannot parse " + key + "= in: " + comment_line);
    return true;
  }
  return false;
}

void read_exyz_conditioning(const std::string &comment_line, AtomicSystem &system) {
  double v = 0;
  if (parse_exyz_number(comment_line, "charge", v)) system.set_charge(integer_value(v, "charge"));
  // ASE / metatomic: spin_multiplicity, else spin (OMol's key, also 2S+1)
  if (parse_exyz_number(comment_line, "spin_multiplicity", v) || parse_exyz_number(comment_line, "spin", v))
    system.set_spin_multiplicity(integer_value(v, "spin_multiplicity"));
  // dipole_origin="x y z" (iris PET-LR calculators): three numbers, quoted
  for (size_t pos = comment_line.find("dipole_origin="); pos != std::string::npos;
       pos = comment_line.find("dipole_origin=", pos + 1)) {
    if (pos > 0 && !std::isspace(static_cast<unsigned char>(comment_line[pos - 1]))) continue;
    size_t start = pos + 14;
    if (start < comment_line.size() && comment_line[start] == '"') start++;
    std::istringstream iss(comment_line.substr(start));
    std::array<double, 3> r0{};
    if (!(iss >> r0[0] >> r0[1] >> r0[2])) throw std::runtime_error("XYZ: cannot parse dipole_origin= in: " + comment_line);
    system.set_dipole_origin(r0);
    break;
  }
  // field="x y z" (V/A; his iris `field` input, atoms.info["field"]): three numbers, quoted
  for (size_t pos = comment_line.find("field="); pos != std::string::npos; pos = comment_line.find("field=", pos + 1)) {
    if (pos > 0 && !std::isspace(static_cast<unsigned char>(comment_line[pos - 1]))) continue;
    size_t start = pos + 6;
    if (start < comment_line.size() && comment_line[start] == '"') start++;
    std::istringstream iss(comment_line.substr(start));
    std::array<double, 3> E{};
    if (!(iss >> E[0] >> E[1] >> E[2])) throw std::runtime_error("XYZ: cannot parse field= in: " + comment_line);
    system.set_field(E);
    break;
  }
}

// the token column of a per-atom extxyz array (Properties=name:type:count:...), -1 if absent
static int exyz_property_column(const std::string &comment_line, const std::string &name) {
  size_t pos = comment_line.find("Properties=");
  if (pos == std::string::npos || (pos > 0 && !std::isspace(static_cast<unsigned char>(comment_line[pos - 1])))) return -1;
  pos += 11;
  size_t end = comment_line.find_first_of(" \t\r", pos);
  std::string spec = comment_line.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
  std::vector<std::string> f;
  std::stringstream ss(spec);
  for (std::string x; std::getline(ss, x, ':');) f.push_back(x);
  int col = 0;
  for (size_t k = 0; k + 2 < f.size(); k += 3) {
    if (f[k] == name) return col;
    col += std::stoi(f[k + 2]);
  }
  return -1;
}

AtomicSystem read_xyz(std::istream &stream) {
  std::string line;

  // Line 1: number of atoms
  if (!std::getline(stream, line)) {
    throw std::runtime_error("XYZ: Failed to read number of atoms");
  }
  int n_atoms = std::stoi(line);

  // Line 2: comment (may contain lattice for EXYZ)
  if (!std::getline(stream, line)) {
    throw std::runtime_error("XYZ: Failed to read comment line");
  }
  std::string comment = line;

  // Try to parse lattice from comment (EXYZ format)
  double cell[3][3] = {{0}};
  bool has_cell = parse_exyz_lattice(comment, cell);

  // Read atoms
  std::vector<int> atomic_numbers;
  std::vector<double> positions;
  // per-atom formal charges (fragment-constrained charges), Properties=...:formal_charges:R:1 (or I)
  const int fq_col = exyz_property_column(comment, "formal_charges");
  std::vector<float> formal;

  atomic_numbers.reserve(n_atoms);
  positions.reserve(n_atoms * 3);

  for (int i = 0; i < n_atoms; ++i) {
    if (!std::getline(stream, line)) {
      throw std::runtime_error("XYZ: Unexpected end of file at atom " +
                               std::to_string(i));
    }

    std::istringstream iss(line);
    std::string element;
    double x, y, z;

    if (!(iss >> element >> x >> y >> z)) {
      throw std::runtime_error("XYZ: Failed to parse atom line: " + line);
    }

    atomic_numbers.push_back(element_to_atomic_number(element));
    if (fq_col >= 4) {
      std::string tok;
      for (int c = 4; c <= fq_col; c++)
        if (!(iss >> tok)) throw std::runtime_error("XYZ: no formal_charges column in: " + line);
      formal.push_back(std::stof(tok));
    }
    positions.push_back(x);
    positions.push_back(y);
    positions.push_back(z);
  }

  // Create atomic system
  if (has_cell) {
    // Convert double cell to float
    float cell_f[3][3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        cell_f[i][j] = static_cast<float>(cell[i][j]);
      }
    }
    bool pbc[3];
    parse_exyz_pbc(comment, pbc);
    Cell cell_obj(cell_f, pbc[0], pbc[1], pbc[2]);

    // Convert to float positions
    std::vector<float> positions_f(positions.begin(), positions.end());

    AtomicSystem system(n_atoms, positions_f.data(), atomic_numbers.data(), &cell_obj);
    read_exyz_conditioning(comment, system);
    if (!formal.empty()) system.set_formal_charges(std::move(formal));
    return system;
  } else {
    // No cell - non-periodic
    std::vector<float> positions_f(positions.begin(), positions.end());
    AtomicSystem system(n_atoms, positions_f.data(), atomic_numbers.data(), nullptr);
    read_exyz_conditioning(comment, system);
    if (!formal.empty()) system.set_formal_charges(std::move(formal));
    return system;
  }
}

AtomicSystem read_xyz(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("XYZ: Failed to open file: " + filename);
  }
  return read_xyz(file);
}

void write_xyz(std::ostream &stream, const AtomicSystem &system,
               const std::string &comment) {
  int n_atoms = system.num_atoms();
  const int32_t *atomic_nums = system.atomic_numbers();
  const float *pos = system.positions();

  // Write number of atoms
  stream << n_atoms << "\n";

  // Write comment
  stream << comment << "\n";

  // Write atoms
  for (int i = 0; i < n_atoms; ++i) {
    int Z = atomic_nums[i];
    if (Z <= 0 || Z > MAX_ATOMIC_NUMBER) {
      throw std::runtime_error("XYZ: Unsupported atomic number: " +
                               std::to_string(Z));
    }

    stream << ELEMENT_SYMBOLS[Z] << " " << pos[i * 3 + 0] << " "
           << pos[i * 3 + 1] << " " << pos[i * 3 + 2] << "\n";
  }
}

void write_xyz(const std::string &filename, const AtomicSystem &system,
               const std::string &comment) {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("XYZ: Failed to open file for writing: " +
                             filename);
  }
  write_xyz(file, system, comment);
}

} // namespace io
} // namespace mlipcpp
