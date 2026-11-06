// WASM/Emscripten bindings for mlipcpp
// Provides JavaScript API for machine learning interatomic potentials

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <mlipcpp/mlipcpp.hpp>
#include <mlipcpp/io.h>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <memory>
#include <fstream>

using namespace emscripten;

// Wrapper class for easier JS interaction with atomic systems
class AtomicSystemWrapper {
public:
    AtomicSystemWrapper() = default;

    // Create from arrays
    static AtomicSystemWrapper create(
        const val& positions,      // Float64Array [x0,y0,z0, x1,y1,z1, ...]
        const val& atomicNumbers,  // Int32Array [Z0, Z1, ...]
        const val& cell,           // Float64Array [3x3] or null for non-periodic
        bool periodic
    ) {
        AtomicSystemWrapper wrapper;

        const size_t n_atoms = atomicNumbers["length"].as<size_t>();
        wrapper.n_atoms_ = n_atoms;

        // Parse positions
        wrapper.positions_.resize(n_atoms * 3);
        for (size_t i = 0; i < n_atoms * 3; ++i) {
            wrapper.positions_[i] = positions[i].as<float>();
        }

        // Parse atomic numbers
        wrapper.atomic_numbers_.resize(n_atoms);
        for (size_t i = 0; i < n_atoms; ++i) {
            wrapper.atomic_numbers_[i] = atomicNumbers[i].as<int32_t>();
        }

        // Parse cell if periodic
        wrapper.periodic_ = periodic;
        if (periodic && !cell.isNull() && !cell.isUndefined()) {
            wrapper.cell_.resize(9);
            for (int i = 0; i < 9; ++i) {
                wrapper.cell_[i] = cell[i].as<float>();
            }
            wrapper.pbc_ = {true, true, true};
        }

        return wrapper;
    }

    // Create from XYZ string
    static AtomicSystemWrapper fromXyzString(const std::string& xyz_content) {
        AtomicSystemWrapper wrapper;

        std::istringstream stream(xyz_content);
        auto system = mlipcpp::io::read_xyz(stream);

        wrapper.n_atoms_ = system.num_atoms();
        wrapper.positions_.resize(wrapper.n_atoms_ * 3);
        const float* pos = system.positions();
        for (size_t i = 0; i < wrapper.n_atoms_ * 3; ++i) {
            wrapper.positions_[i] = pos[i];
        }

        const int32_t* nums = system.atomic_numbers();
        wrapper.atomic_numbers_.assign(nums, nums + wrapper.n_atoms_);
        wrapper.periodic_ = system.is_periodic();
        wrapper.charge_ = system.charge();
        wrapper.spin_multiplicity_ = system.spin_multiplicity();
        if (system.has_formal_charges()) wrapper.formal_charges_ = system.formal_charges();

        if (system.is_periodic() && system.cell()) {
            wrapper.cell_.resize(9);
            const auto* cell = system.cell();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    wrapper.cell_[i * 3 + j] = cell->matrix[i][j];
                }
            }
            wrapper.pbc_ = {true, true, true};
        }

        return wrapper;
    }

    size_t numAtoms() const { return n_atoms_; }
    bool isPeriodic() const { return periodic_; }

    // Total charge (e) and spin multiplicity (2S+1), read by conditioned models
    void setCharge(int q) { charge_ = q; }
    void setSpinMultiplicity(int m) { spin_multiplicity_ = m; }
    int charge() const { return charge_; }
    int spinMultiplicity() const { return spin_multiplicity_; }
    // Per-atom formal charges (e), read by fragment-constrained-charge models
    // (pet.plato_lr.frag_charges); an empty / null array clears them
    void setFormalCharges(const val& q) {
        formal_charges_.clear();
        if (q.isNull() || q.isUndefined()) return;
        formal_charges_ = convertJSArrayToNumberVector<float>(q);
        if (!formal_charges_.empty() && formal_charges_.size() != n_atoms_)
            throw std::runtime_error("formal charges: length != number of atoms");
    }
    const std::vector<float>& formalCharges() const { return formal_charges_; }

    val getPositions() const {
        val result = val::global("Float32Array").new_(positions_.size());
        for (size_t i = 0; i < positions_.size(); ++i) {
            result.set(i, positions_[i]);
        }
        return result;
    }

    val getAtomicNumbers() const {
        val result = val::global("Int32Array").new_(atomic_numbers_.size());
        for (size_t i = 0; i < atomic_numbers_.size(); ++i) {
            result.set(i, atomic_numbers_[i]);
        }
        return result;
    }

    val getCell() const {
        if (!periodic_ || cell_.empty()) {
            return val::null();
        }
        val result = val::global("Float32Array").new_(9);
        for (int i = 0; i < 9; ++i) {
            result.set(i, cell_[i]);
        }
        return result;
    }

    // Internal accessors for prediction
    const float* positionsPtr() const { return positions_.data(); }
    const int32_t* atomicNumbersPtr() const { return atomic_numbers_.data(); }
    const float* cellPtr() const { return periodic_ && !cell_.empty() ? cell_.data() : nullptr; }
    const bool* pbcPtr() const { return periodic_ ? pbc_.data() : nullptr; }

private:
    size_t n_atoms_ = 0;
    std::vector<float> positions_;
    std::vector<int32_t> atomic_numbers_;
    std::vector<float> cell_;
    std::array<bool, 3> pbc_ = {false, false, false};
    bool periodic_ = false;
    int charge_ = 0;
    int spin_multiplicity_ = 1;
    std::vector<float> formal_charges_;
};

// Wrapper class for Predictor - uses shared_ptr for Embind compatibility
class PredictorWrapper {
public:
    PredictorWrapper() = default;
    PredictorWrapper(std::shared_ptr<mlipcpp::Predictor> p) : predictor_(std::move(p)) {}

    // Map user-facing backend name strings to the public Backend enum.
    static mlipcpp::Backend parseBackend(const std::string& name) {
        if (name.empty() || name == "auto") return mlipcpp::Backend::Auto;
        if (name == "cpu")    return mlipcpp::Backend::CPU;
        if (name == "webgpu" || name == "wgpu") return mlipcpp::Backend::WebGPU;
        if (name == "metal" || name == "mtl")   return mlipcpp::Backend::Metal;
        if (name == "cuda")   return mlipcpp::Backend::CUDA;
        if (name == "hip" || name == "rocm")    return mlipcpp::Backend::HIP;
        if (name == "vulkan") return mlipcpp::Backend::Vulkan;
        if (name == "sycl")   return mlipcpp::Backend::SYCL;
        if (name == "cann")   return mlipcpp::Backend::CANN;
        return mlipcpp::Backend::Auto;
    }

    // Load model from file path (Emscripten VFS)
    static PredictorWrapper load(const std::string& path) {
        return PredictorWrapper(std::make_shared<mlipcpp::Predictor>(path));
    }

    // Load model from ArrayBuffer
    static PredictorWrapper loadFromBuffer(const val& buffer) {
        return loadFromBufferWithBackend(buffer, std::string("auto"));
    }

    static PredictorWrapper loadFromBufferWithBackend(const val& buffer,
                                                      const std::string& backend) {
        mlipcpp::set_backend(parseBackend(backend));
        // Get data from ArrayBuffer
        val uint8Array = val::global("Uint8Array").new_(buffer);
        const size_t length = uint8Array["length"].as<size_t>();

        // Copy data to C++ vector
        std::vector<uint8_t> data(length);
        for (size_t i = 0; i < length; ++i) {
            data[i] = uint8Array[i].as<uint8_t>();
        }

        // Write to temp file using C++ fstream (works with Emscripten VFS)
        const std::string temp_path = "/tmp/model.gguf";
        {
            std::ofstream ofs(temp_path, std::ios::binary);
            if (!ofs) {
                throw std::runtime_error("Failed to create temp file");
            }
            ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
        }

        // Load from temp file
        auto predictor = std::make_shared<mlipcpp::Predictor>(temp_path);

        // Clean up temp file
        std::remove(temp_path.c_str());

        return PredictorWrapper(std::move(predictor));
    }

    // Load a GGUF the caller wrote into the virtual FS (Module.FS.writeFile,
    // one bulk copy; loadFromBuffer copies through embind one byte at a time)
    static PredictorWrapper loadFileWithBackend(const std::string& path,
                                                const std::string& backend) {
        mlipcpp::set_backend(parseBackend(backend));
        return PredictorWrapper(std::make_shared<mlipcpp::Predictor>(path));
    }

    bool conditioned() const { return predictor_ && predictor_->conditioned(); }
    int maxCharge() const { return predictor_ ? predictor_->max_charge() : 0; }
    int maxSpinMultiplicity() const {
        return predictor_ ? predictor_->max_spin_multiplicity() : 0;
    }
    bool hasNcForces() const { return predictor_ && predictor_->has_nc_forces(); }

    std::string modelType() const {
        return predictor_ ? std::string(predictor_->model_type()) : "";
    }

    double cutoff() const {
        return predictor_ ? predictor_->cutoff() : 0.0;
    }

    // Predict energy only
    double predictEnergy(const AtomicSystemWrapper& system) {
        if (!predictor_) {
            throw std::runtime_error("Model not loaded");
        }

        auto result = predictor_->predict(
            system.numAtoms(),
            system.positionsPtr(),
            system.atomicNumbersPtr(),
            system.cellPtr(),
            system.pbcPtr(),
            false  // compute_forces = false
        );

        return result.energy;
    }

    // Predict energy and forces (conservative, via gradient)
    val predict(const AtomicSystemWrapper& system) {
        return predictWithOptions(system, false);
    }

    // Predict with options - useNCForces uses forward pass heads instead of gradients
    val predictWithOptions(const AtomicSystemWrapper& system, bool useNCForces) {
        if (!predictor_) {
            throw std::runtime_error("Model not loaded");
        }

        return predictFull(system, true, useNCForces);
    }

    // Energy (and forces unless computeForces is false) with the system's
    // charge, spin multiplicity and formal charges; latent charges if the
    // model has an LES head
    val predictFull(const AtomicSystemWrapper& system, bool computeForces,
                    bool useNCForces) {
        if (!predictor_) {
            throw std::runtime_error("Model not loaded");
        }

        mlipcpp::PredictOptions options;
        options.compute_forces = computeForces;
        options.use_nc_forces = useNCForces;
        options.charge = system.charge();
        options.spin_multiplicity = system.spinMultiplicity();
        options.formal_charges = system.formalCharges();

        auto result = predictor_->predict(
            system.numAtoms(),
            system.positionsPtr(),
            system.atomicNumbersPtr(),
            system.cellPtr(),
            system.pbcPtr(),
            options
        );

        val output = val::object();
        output.set("energy", static_cast<double>(result.energy));

        // Return forces as Float32Array (native precision — no double-widening copy)
        val forces = val::global("Float32Array").new_(result.forces.size());
        if (!result.forces.empty())
            forces.call<void>("set", val(typed_memory_view(result.forces.size(),
                                                           result.forces.data())));
        output.set("forces", forces);

        if (!result.latent_charges.empty()) {
            val q = val::global("Float32Array").new_(result.latent_charges.size());
            q.call<void>("set", val(typed_memory_view(result.latent_charges.size(),
                                                      result.latent_charges.data())));
            output.set("latentCharges", q);
        }

        if (result.has_stress()) {
            val stress = val::global("Float32Array").new_(6);
            for (int i = 0; i < 6; ++i) {
                stress.set(i, result.stress[i]);
            }
            output.set("stress", stress);
        }

        return output;
    }

    bool isLoaded() const { return predictor_ != nullptr; }

    // Predict energy and capture every intermediate tensor for visualisation.
    // Returns:
    //   {
    //     energy: number,
    //     activations: Map<nodeId, {
    //       name: string,
    //       dtype: 'f32' | 'i32' | 'i8' | 'unsupported',
    //       shape: number[],
    //       data: Float32Array | Int32Array | Int8Array | null
    //     }>
    //   }
    val predictWithActivations(const AtomicSystemWrapper& system) {
        if (!predictor_) {
            throw std::runtime_error("Model not loaded");
        }

        auto result = predictor_->predict_with_activations(
            system.numAtoms(),
            system.positionsPtr(),
            system.atomicNumbersPtr(),
            system.cellPtr(),
            system.pbcPtr()
        );

        val output = val::object();
        output.set("energy", static_cast<double>(result.prediction.energy));

        val activations = val::global("Map").new_();
        for (const auto& a : result.activations) {
            val entry = val::object();
            entry.set("name", a.name);
            entry.set("dtype", a.dtype);

            val shape = val::array();
            for (size_t i = 0; i < a.shape.size(); ++i) {
                shape.set(static_cast<int>(i), static_cast<double>(a.shape[i]));
            }
            entry.set("shape", shape);

            // Bulk-copy through Embind's typed_memory_view, which constructs
            // a TypedArray view over the WASM heap directly. This avoids
            // depending on Module.HEAPF32 etc. being exposed by the build.
            val data = val::null();
            if (!a.data.empty()) {
                if (a.dtype == "f32") {
                    const size_t n = a.data.size() / sizeof(float);
                    data = val::global("Float32Array").new_(n);
                    val src(typed_memory_view(n,
                        reinterpret_cast<const float*>(a.data.data())));
                    data.call<void>("set", src);
                } else if (a.dtype == "i32") {
                    const size_t n = a.data.size() / sizeof(int32_t);
                    data = val::global("Int32Array").new_(n);
                    val src(typed_memory_view(n,
                        reinterpret_cast<const int32_t*>(a.data.data())));
                    data.call<void>("set", src);
                } else if (a.dtype == "i8") {
                    const size_t n = a.data.size();
                    data = val::global("Int8Array").new_(n);
                    val src(typed_memory_view(n,
                        reinterpret_cast<const int8_t*>(a.data.data())));
                    data.call<void>("set", src);
                }
            }
            entry.set("data", data);

            activations.call<void>("set", a.node_id, entry);
        }
        output.set("activations", activations);
        return output;
    }

private:
    std::shared_ptr<mlipcpp::Predictor> predictor_;
};

// Utility functions
std::string getVersion() {
    return mlipcpp::version();
}

std::string getBackendName() {
    return std::string(mlipcpp::get_backend_name());
}

void setBackend(const std::string& name) {
    mlipcpp::set_backend(PredictorWrapper::parseBackend(name));
}

// Emscripten bindings
EMSCRIPTEN_BINDINGS(mlipcpp) {
    // AtomicSystem wrapper
    class_<AtomicSystemWrapper>("AtomicSystem")
        .constructor<>()
        .class_function("create", &AtomicSystemWrapper::create)
        .class_function("fromXyzString", &AtomicSystemWrapper::fromXyzString)
        .function("numAtoms", &AtomicSystemWrapper::numAtoms)
        .function("isPeriodic", &AtomicSystemWrapper::isPeriodic)
        .function("getPositions", &AtomicSystemWrapper::getPositions)
        .function("getAtomicNumbers", &AtomicSystemWrapper::getAtomicNumbers)
        .function("getCell", &AtomicSystemWrapper::getCell)
        .function("setCharge", &AtomicSystemWrapper::setCharge)
        .function("setSpinMultiplicity", &AtomicSystemWrapper::setSpinMultiplicity)
        .function("charge", &AtomicSystemWrapper::charge)
        .function("spinMultiplicity", &AtomicSystemWrapper::spinMultiplicity)
        .function("setFormalCharges", &AtomicSystemWrapper::setFormalCharges);

    // Predictor wrapper (named Model for JS API simplicity)
    class_<PredictorWrapper>("Model")
        .constructor<>()
        .class_function("load", &PredictorWrapper::load)
        .class_function("loadFromBuffer", &PredictorWrapper::loadFromBuffer)
        .class_function("loadFromBufferWithBackend", &PredictorWrapper::loadFromBufferWithBackend)
        .class_function("loadFileWithBackend", &PredictorWrapper::loadFileWithBackend)
        .function("conditioned", &PredictorWrapper::conditioned)
        .function("maxCharge", &PredictorWrapper::maxCharge)
        .function("maxSpinMultiplicity", &PredictorWrapper::maxSpinMultiplicity)
        .function("hasNcForces", &PredictorWrapper::hasNcForces)
        .function("predictFull", &PredictorWrapper::predictFull)
        .function("modelType", &PredictorWrapper::modelType)
        .function("cutoff", &PredictorWrapper::cutoff)
        .function("predictEnergy", &PredictorWrapper::predictEnergy)
        .function("predict", &PredictorWrapper::predict)
        .function("predictWithOptions", &PredictorWrapper::predictWithOptions)
        .function("predictWithActivations", &PredictorWrapper::predictWithActivations)
        .function("isLoaded", &PredictorWrapper::isLoaded);

    // Utility functions
    function("getVersion", &getVersion);
    function("getBackendName", &getBackendName);
    function("setBackend", &setBackend);
}
