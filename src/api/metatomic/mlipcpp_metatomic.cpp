/**
 * @file mlipcpp_metatomic.cpp
 * @brief metatomic plugin running mlipcpp GGUF models
 *
 * Built as a shared library that engines load with `mta_load_plugin`; any
 * `*.gguf` path given to `mta_load_model` is then served by a
 * `mlipcpp::Predictor`. Outputs:
 *
 *   - "energy" (per system), with "positions" and "strain" gradients
 *   - "non_conservative_force" (per atom), for models with a direct force head
 *
 * The model computes its own neighbor list, so it requests no pair lists.
 * Model options (JSON object of strings, all optional):
 *
 *   - "backend": auto, cpu, cuda, hip, metal, vulkan, sycl (default: auto)
 *   - "dtype": float64 or float32, the dtype of systems and outputs
 *     (default: float64; the computation itself is always float32)
 *   - "interaction_range": override of cutoff x number of message passing
 *     layers, in Angstrom
 */

#include <metatomic.hpp>
#include <metatomic/plugin.hpp>

#include <gguf.h>

#include "mlipcpp/mlipcpp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using metatomic::Gradients;
using metatomic::Quantity;
using metatomic::SampleKind;
using DType = metatomic::ModelCapabilities::DType;

const char *ENERGY = "energy";
const char *NC_FORCE = "non_conservative_force";

double parse_double(const std::string &value, const std::string &name) {
  std::istringstream in(value);
  in.imbue(std::locale::classic());
  double result = 0.0;
  in >> std::noskipws >> result;
  if (!in || in.get() != std::char_traits<char>::eof()) {
    throw metatomic::Error("mlipcpp option '" + name + "' must be a number");
  }
  return result;
}

mlipcpp::Backend parse_backend(const std::string &name) {
  static const std::map<std::string, mlipcpp::Backend> BACKENDS = {
      {"auto", mlipcpp::Backend::Auto},     {"cpu", mlipcpp::Backend::CPU},
      {"cuda", mlipcpp::Backend::CUDA},     {"hip", mlipcpp::Backend::HIP},
      {"metal", mlipcpp::Backend::Metal},   {"vulkan", mlipcpp::Backend::Vulkan},
      {"webgpu", mlipcpp::Backend::WebGPU}, {"sycl", mlipcpp::Backend::SYCL},
      {"cann", mlipcpp::Backend::CANN},
  };
  auto it = BACKENDS.find(name);
  if (it == BACKENDS.end()) {
    throw metatomic::Error("unknown mlipcpp backend '" + name + "'");
  }
  return it->second;
}

/// What the plugin needs from the GGUF header besides the Predictor
struct GGUFInfo {
  std::string name;
  int message_passing_layers = 1;
};

GGUFInfo read_gguf_info(const std::string &path) {
  gguf_init_params params = {/*no_alloc*/ true, /*ctx*/ nullptr};
  auto *ctx = gguf_init_from_file(path.c_str(), params);
  if (ctx == nullptr) {
    throw metatomic::Error("failed to read GGUF file at '" + path + "'");
  }

  auto info = GGUFInfo();
  auto key = gguf_find_key(ctx, "general.name");
  if (key >= 0 && gguf_get_kv_type(ctx, key) == GGUF_TYPE_STRING) {
    info.name = gguf_get_val_str(ctx, key);
  }

  // PET message passing layers are stored as `gnn_layers_<i>_...`
  auto layers = std::set<int>();
  for (int64_t i = 0; i < gguf_get_n_tensors(ctx); i++) {
    auto tensor = std::string(gguf_get_tensor_name(ctx, i));
    if (tensor.rfind("gnn_layers_", 0) == 0) {
      layers.insert(std::stoi(tensor.substr(11)));
    }
  }
  info.message_passing_layers = std::max<int>(1, static_cast<int>(layers.size()));

  gguf_free(ctx);
  return info;
}

/// Copy a CPU DLPack tensor (any strides, float/int/bool) into a vector
template <typename T> std::vector<T> dlpack_to_vector(const metatomic::DLPackTensor &tensor) {
  const auto &dl = tensor->dl_tensor;
  if (dl.device.device_type != kDLCPU) {
    throw metatomic::Error("the mlipcpp plugin only accepts systems on CPU");
  }

  size_t count = 1;
  for (int d = 0; d < dl.ndim; d++) {
    count *= static_cast<size_t>(dl.shape[d]);
  }

  auto read = [&](const char *ptr) -> T {
    switch (dl.dtype.code) {
    case kDLFloat:
      if (dl.dtype.bits == 64) return static_cast<T>(*reinterpret_cast<const double *>(ptr));
      if (dl.dtype.bits == 32) return static_cast<T>(*reinterpret_cast<const float *>(ptr));
      break;
    case kDLInt:
      if (dl.dtype.bits == 32) return static_cast<T>(*reinterpret_cast<const int32_t *>(ptr));
      if (dl.dtype.bits == 64) return static_cast<T>(*reinterpret_cast<const int64_t *>(ptr));
      break;
    case kDLBool:
    case kDLUInt:
      if (dl.dtype.bits == 8) return static_cast<T>(*reinterpret_cast<const uint8_t *>(ptr));
      break;
    default:
      break;
    }
    throw metatomic::Error("unsupported DLPack dtype in system data");
  };

  auto result = std::vector<T>(count);
  const auto *base = static_cast<const char *>(dl.data) + dl.byte_offset;
  auto element_size = static_cast<int64_t>((dl.dtype.bits * dl.dtype.lanes + 7) / 8);
  auto index = std::vector<int64_t>(static_cast<size_t>(dl.ndim), 0);
  for (size_t i = 0; i < count; i++) {
    int64_t offset = 0;
    int64_t stride = 1;
    for (int d = dl.ndim - 1; d >= 0; d--) {
      auto s = dl.strides != nullptr ? dl.strides[d] : stride;
      offset += index[static_cast<size_t>(d)] * s;
      stride *= dl.shape[d];
    }
    result[i] = read(base + offset * element_size);

    for (int d = dl.ndim - 1; d >= 0; d--) {
      if (++index[static_cast<size_t>(d)] < dl.shape[d]) break;
      index[static_cast<size_t>(d)] = 0;
    }
  }
  return result;
}

double determinant(const std::array<float, 9> &m) {
  return double(m[0]) * (double(m[4]) * m[8] - double(m[5]) * m[7]) -
         double(m[1]) * (double(m[3]) * m[8] - double(m[5]) * m[6]) +
         double(m[2]) * (double(m[3]) * m[7] - double(m[4]) * m[6]);
}

/// A metatomic system converted to what mlipcpp expects
struct MlipSystem {
  std::vector<int32_t> types;
  std::vector<float> positions;
  std::array<float, 9> cell = {};
  std::array<bool, 3> pbc = {false, false, false};

  bool periodic() const { return pbc[0] || pbc[1] || pbc[2]; }

  MlipSystem(const metatomic::System &system, double cutoff) {
    types = dlpack_to_vector<int32_t>(system.types());
    positions = dlpack_to_vector<float>(system.positions());
    auto cell_values = dlpack_to_vector<float>(system.cell());
    auto pbc_values = dlpack_to_vector<uint8_t>(system.pbc());
    std::copy(cell_values.begin(), cell_values.end(), cell.begin());
    for (size_t i = 0; i < 3; i++) {
      pbc[i] = pbc_values[i] != 0;
    }

    if (periodic()) {
      fill_non_periodic_vectors(cutoff);
    }
  }

  /// metatomic uses zero cell vectors along non-periodic directions, while
  /// mlipcpp needs an invertible cell. Complete the periodic vectors with
  /// orthogonal ones, long enough that no atom interacts across them.
  void fill_non_periodic_vectors(double cutoff) {
    using Vec = std::array<double, 3>;
    auto dot = [](const Vec &a, const Vec &b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };

    double extent = 0.0;
    for (auto x : positions) {
      extent = std::max(extent, std::abs(double(x)));
    }
    auto length = 2.0 * (extent + cutoff) + 10.0;

    // orthonormal basis of the directions already used
    auto basis = std::vector<Vec>();
    auto add_to_basis = [&](Vec v) {
      for (const auto &b : basis) {
        auto p = dot(v, b);
        for (size_t d = 0; d < 3; d++) v[d] -= p * b[d];
      }
      auto n = std::sqrt(dot(v, v));
      for (auto &x : v) x /= n;
      basis.push_back(v);
      return v;
    };
    for (size_t i = 0; i < 3; i++) {
      if (pbc[i]) add_to_basis({cell[3 * i], cell[3 * i + 1], cell[3 * i + 2]});
    }

    for (size_t i = 0; i < 3; i++) {
      if (pbc[i]) continue;
      // the Cartesian axis least covered by the current basis
      auto best = Vec{};
      double best_residual = -1.0;
      for (size_t axis = 0; axis < 3; axis++) {
        auto residual = 1.0;
        for (const auto &b : basis) residual -= b[axis] * b[axis];
        if (residual > best_residual) {
          best_residual = residual;
          best = Vec{};
          best[axis] = 1.0;
        }
      }
      auto direction = add_to_basis(best);
      for (size_t d = 0; d < 3; d++) {
        cell[3 * i + d] = static_cast<float>(length * direction[d]);
      }
    }
  }
};

metatensor::Labels make_labels(const std::vector<std::string> &names,
                               const std::vector<int32_t> &values) {
  return metatensor::Labels(names, values.data(), values.size() / names.size(),
                            metatensor::assume_unique{});
}

class MlipcppModel final : public metatomic::BaseModel {
public:
  MlipcppModel(const std::string &path, const std::map<std::string, std::string> &options)
      : path_(path) {
    auto model_options = mlipcpp::ModelOptions();
    for (const auto &[key, value] : options) {
      if (key == "backend") {
        model_options.backend = parse_backend(value);
      } else if (key == "dtype") {
        if (value == "float64") {
          dtype_ = DType::Float64;
        } else if (value == "float32") {
          dtype_ = DType::Float32;
        } else {
          throw metatomic::Error("mlipcpp option 'dtype' must be float64 or float32");
        }
      } else if (key == "interaction_range") {
        interaction_range_ = parse_double(value, key);
      } else {
        throw metatomic::Error("unknown mlipcpp option '" + key + "'");
      }
    }

    info_ = read_gguf_info(path);
    predictor_ = std::make_unique<mlipcpp::Predictor>(path, model_options);
    if (interaction_range_ <= 0.0) {
      interaction_range_ = predictor_->cutoff() * info_.message_passing_layers;
    }
  }

  metatomic::ModelCapabilities capabilities() const override {
    auto builder = metatomic::ModelCapabilities::builder();
    builder.add_output(Quantity::builder()
                           .name(ENERGY)
                           .unit("eV")
                           .sample_kind(SampleKind::System)
                           .add_gradient(Gradients::Positions)
                           .add_gradient(Gradients::Strain)
                           .build());
    if (predictor_->has_nc_forces()) {
      builder.add_output(Quantity::builder()
                             .name(NC_FORCE)
                             .unit("eV/Angstrom")
                             .sample_kind(SampleKind::Atom)
                             .build());
    }
    return builder.atomic_types(predictor_->atomic_types())
        .interaction_range(interaction_range_)
        .length_unit("Angstrom")
        .supported_devices({metatomic::ModelCapabilities::Device::CPU})
        .dtype(dtype_)
        .build();
  }

  metatomic::ModelMetadata metadata() const override {
    auto description = std::string(predictor_->model_type()) + " model from " + path_ +
                       " (mlipcpp " + mlipcpp::version() + ")";
    auto builder = metatomic::ModelMetadata::builder();
    builder.name(info_.name.empty() ? path_ : info_.name).description(description);
    builder.add_reference("implementation", "https://github.com/peterspackman/mlip.cpp");
    return builder.build();
  }

  std::vector<metatomic::PairListOptions>
  requested_pair_lists(const std::vector<Quantity> &) const override {
    return {};
  }

  std::vector<Quantity> requested_inputs(const std::vector<Quantity> &) const override {
    return {};
  }

  std::vector<metatensor::TensorMap> execute_inner(const std::vector<metatomic::System> &systems,
                                                   std::optional<metatensor::Labels> selected_atoms,
                                                   const std::vector<Quantity> &requested_outputs) override {
    auto converted = std::vector<MlipSystem>();
    converted.reserve(systems.size());
    for (const auto &system : systems) {
      converted.emplace_back(system, predictor_->cutoff());
    }

    auto results = std::vector<metatensor::TensorMap>();
    for (const auto &output : requested_outputs) {
      if (output.name() == ENERGY && output.sample_kind() == SampleKind::System) {
        check_all_atoms_selected(converted, selected_atoms);
        results.emplace_back(energy(converted, output));
      } else if (output.name() == NC_FORCE && predictor_->has_nc_forces()) {
        results.emplace_back(nc_forces(converted, selected_atoms));
      } else {
        throw metatomic::Error("the mlipcpp plugin can not compute '" + output.name() + "'");
      }
    }
    return results;
  }

private:
  static void check_all_atoms_selected(const std::vector<MlipSystem> &systems,
                                       const std::optional<metatensor::Labels> &selected_atoms) {
    if (!selected_atoms.has_value()) return;

    size_t total = 0;
    for (size_t s = 0; s < systems.size(); s++) {
      for (size_t a = 0; a < systems[s].types.size(); a++, total++) {
        if (!selected_atoms->position({static_cast<int32_t>(s), static_cast<int32_t>(a)})) {
          throw metatomic::Error("mlipcpp models compute the energy of whole systems, "
                                 "selected_atoms must contain all atoms");
        }
      }
    }
    if (selected_atoms->count() != total) {
      throw metatomic::Error("selected_atoms contains atoms that are not in the systems");
    }
  }

  mlipcpp::Result predict(const MlipSystem &system, const mlipcpp::PredictOptions &options) {
    auto n_atoms = static_cast<int32_t>(system.types.size());
    if (system.periodic()) {
      return predictor_->predict(n_atoms, system.positions.data(), system.types.data(),
                                 system.cell.data(), system.pbc.data(), options);
    }
    return predictor_->predict(n_atoms, system.positions.data(), system.types.data(),
                               nullptr, nullptr, options);
  }

  metatensor::TensorBlock make_block(std::vector<double> values, std::vector<uintptr_t> shape,
                                     metatensor::Labels samples,
                                     std::vector<metatensor::Labels> components,
                                     const std::string &property) const {
    auto properties = metatensor::Labels({property}, {{0}});
    if (dtype_ == DType::Float32) {
      auto data = std::vector<float>(values.begin(), values.end());
      return metatensor::TensorBlock(
          std::make_unique<metatensor::SimpleDataArray<float>>(shape, std::move(data)), samples,
          components, properties);
    }
    return metatensor::TensorBlock(
        std::make_unique<metatensor::SimpleDataArray<double>>(shape, std::move(values)), samples,
        components, properties);
  }

  static metatensor::TensorMap single_block(metatensor::TensorBlock block) {
    auto blocks = std::vector<metatensor::TensorBlock>();
    blocks.emplace_back(std::move(block));
    return metatensor::TensorMap(metatensor::Labels({"_"}, {{0}}), std::move(blocks));
  }

  metatensor::TensorMap energy(const std::vector<MlipSystem> &systems, const Quantity &request) {
    bool do_positions = false;
    bool do_strain = false;
    for (auto gradient : request.gradients()) {
      do_positions |= gradient == Gradients::Positions;
      do_strain |= gradient == Gradients::Strain;
    }

    auto options = mlipcpp::PredictOptions();
    options.compute_forces = do_positions || do_strain;

    auto energies = std::vector<double>();
    auto position_gradients = std::vector<double>();
    auto strain_gradients = std::vector<double>();
    auto system_samples = std::vector<int32_t>();
    auto position_samples = std::vector<int32_t>();

    for (size_t s = 0; s < systems.size(); s++) {
      const auto &system = systems[s];
      options.compute_stress = do_strain && system.periodic();
      auto result = predict(system, options);

      energies.push_back(result.energy);
      system_samples.push_back(static_cast<int32_t>(s));

      if (do_positions) {
        for (size_t a = 0; a < system.types.size(); a++) {
          position_samples.insert(position_samples.end(), {static_cast<int32_t>(s),
                                                           static_cast<int32_t>(s),
                                                           static_cast<int32_t>(a)});
          for (size_t d = 0; d < 3; d++) {
            position_gradients.push_back(-double(result.forces[3 * a + d]));
          }
        }
      }

      if (do_strain) {
        // dE/de_ab = sum_i x_ia dE/dx_ib = V sigma_ab
        auto virial = std::array<double, 9>{};
        if (system.periodic()) {
          if (!result.has_stress()) {
            throw metatomic::Error("mlipcpp did not return a stress for a periodic system");
          }
          auto volume = std::abs(determinant(system.cell));
          const auto &v = result.stress; // xx yy zz yz xz xy
          virial = {v[0], v[5], v[4], v[5], v[1], v[3], v[4], v[3], v[2]};
          for (auto &x : virial) x *= volume;
        } else {
          for (size_t a = 0; a < system.types.size(); a++) {
            for (size_t i = 0; i < 3; i++) {
              for (size_t j = 0; j < 3; j++) {
                virial[3 * i + j] -=
                    double(system.positions[3 * a + i]) * double(result.forces[3 * a + j]);
              }
            }
          }
          for (size_t i = 0; i < 3; i++) {
            for (size_t j = i + 1; j < 3; j++) {
              auto sym = 0.5 * (virial[3 * i + j] + virial[3 * j + i]);
              virial[3 * i + j] = virial[3 * j + i] = sym;
            }
          }
        }
        strain_gradients.insert(strain_gradients.end(), virial.begin(), virial.end());
      }
    }

    auto n_systems = energies.size();
    auto block = make_block(std::move(energies), {n_systems, 1},
                            make_labels({"system"}, system_samples), {}, ENERGY);

    auto xyz = [](const std::string &name) {
      return metatensor::Labels({name}, {{0}, {1}, {2}});
    };
    if (do_positions) {
      auto n_atoms = position_samples.size() / 3;
      block.add_gradient(
          "positions",
          make_block(std::move(position_gradients), {n_atoms, 3, 1},
                     make_labels({"sample", "system", "atom"}, position_samples), {xyz("xyz")},
                     ENERGY));
    }
    if (do_strain) {
      block.add_gradient("strain",
                         make_block(std::move(strain_gradients), {n_systems, 3, 3, 1},
                                    make_labels({"sample"}, system_samples),
                                    {xyz("xyz_1"), xyz("xyz_2")}, ENERGY));
    }
    return single_block(std::move(block));
  }

  metatensor::TensorMap nc_forces(const std::vector<MlipSystem> &systems,
                                  const std::optional<metatensor::Labels> &selected_atoms) {
    auto options = mlipcpp::PredictOptions();
    options.compute_forces = true;
    options.use_nc_forces = true;

    auto values = std::vector<double>();
    auto samples = std::vector<int32_t>();
    for (size_t s = 0; s < systems.size(); s++) {
      auto result = predict(systems[s], options);
      for (size_t a = 0; a < systems[s].types.size(); a++) {
        auto sample = std::array<int32_t, 2>{static_cast<int32_t>(s), static_cast<int32_t>(a)};
        if (selected_atoms.has_value() && !selected_atoms->position({sample[0], sample[1]})) {
          continue;
        }
        samples.insert(samples.end(), sample.begin(), sample.end());
        for (size_t d = 0; d < 3; d++) {
          values.push_back(result.forces[3 * a + d]);
        }
      }
    }

    auto n_atoms = samples.size() / 2;
    return single_block(make_block(std::move(values), {n_atoms, 3, 1},
                                   make_labels({"system", "atom"}, samples),
                                   {metatensor::Labels({"xyz"}, {{0}, {1}, {2}})}, NC_FORCE));
  }

  std::string path_;
  GGUFInfo info_;
  DType dtype_ = DType::Float64;
  double interaction_range_ = 0.0;
  std::unique_ptr<mlipcpp::Predictor> predictor_;
};

bool ends_with(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::unique_ptr<metatomic::BaseModel> load_model(const std::string &load_from,
                                                 const std::map<std::string, std::string> &options) {
  if (!ends_with(load_from, ".gguf")) {
    return nullptr;
  }
  mlipcpp::suppress_logging();
  return std::make_unique<MlipcppModel>(load_from, options);
}

} // namespace

MTA_REGISTER_CXX_PLUGIN("mlipcpp", load_model);
