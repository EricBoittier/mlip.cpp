/*
 * Run a mlipcpp GGUF model through the metatomic C API, as an engine would:
 * load the plugin, load the model, build systems and call mta_execute_model.
 *
 *   metatomic_plugin <mlipcpp-metatomic.so> <model.gguf> [options-json]
 *
 * Computes energy (with positions and strain gradients) and the
 * non-conservative forces for a silicon cluster, slab (periodic in x and y)
 * and bulk cell, and checks the gradients against finite differences of the
 * energy.
 */
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatensor.h>
#include <metatensor/dlpack/dlpack.h>
#include <metatomic.h>

static void check(mta_status_t status, const char *what) {
    if (status != MTA_SUCCESS) {
        const char *message = NULL;
        mta_last_error(&message, NULL, NULL);
        fprintf(stderr, "error: %s: %s\n", what, message);
        exit(EXIT_FAILURE);
    }
}

static void check_mts(mts_status_t status, const char *what) {
    if (status != MTS_SUCCESS) {
        const char *message = NULL;
        mts_last_error(&message, NULL, NULL);
        fprintf(stderr, "error: %s: %s\n", what, message);
        exit(EXIT_FAILURE);
    }
}

// DLPack tensors owning their data, freed by the system that takes them
typedef struct {
    int64_t shape[2];
    int64_t strides[2];
} Layout;

static void owned_deleter(DLManagedTensorVersioned *self) {
    free(self->dl_tensor.data);
    free(self->manager_ctx);
    free(self);
}

static DLManagedTensorVersioned *owned_tensor(const void *values, size_t size, int32_t ndim,
                                              const int64_t *shape, DLDataType dtype) {
    Layout *layout = calloc(1, sizeof(Layout));
    DLManagedTensorVersioned *tensor = calloc(1, sizeof(DLManagedTensorVersioned));
    void *data = malloc(size);
    memcpy(data, values, size);
    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        layout->shape[i] = shape[i];
        layout->strides[i] = stride;
        stride *= shape[i];
    }
    tensor->version = (DLPackVersion){DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    tensor->manager_ctx = layout;
    tensor->deleter = owned_deleter;
    tensor->dl_tensor = (DLTensor){
        .data = data,
        .device = {kDLCPU, 0},
        .ndim = ndim,
        .dtype = dtype,
        .shape = layout->shape,
        .strides = layout->strides,
        .byte_offset = 0,
    };
    return tensor;
}

#define MAX_ATOMS 64

typedef struct {
    const char *name;
    int n_atoms;
    int32_t types[MAX_ATOMS];
    double positions[3 * MAX_ATOMS];
    double cell[9];
    bool pbc[3];
} Structure;

static mta_system_t *make_system(const Structure *s) {
    mta_system_t *system = NULL;
    int64_t n = s->n_atoms;
    check(mta_system_create(
              "angstrom",
              owned_tensor(s->types, sizeof(int32_t) * n, 1, (int64_t[]){n}, (DLDataType){kDLInt, 32, 1}),
              owned_tensor(s->positions, sizeof(double) * 3 * n, 2, (int64_t[]){n, 3},
                           (DLDataType){kDLFloat, 64, 1}),
              owned_tensor(s->cell, sizeof(double) * 9, 2, (int64_t[]){3, 3}, (DLDataType){kDLFloat, 64, 1}),
              owned_tensor(s->pbc, sizeof(bool) * 3, 1, (int64_t[]){3}, (DLDataType){kDLBool, 8, 1}),
              &system),
          "mta_system_create");
    return system;
}

// Copy the float64 values of a block (or of one of its gradients) to `out`
static size_t block_values(mts_block_t *block, const char *gradient, double *out, size_t max) {
    if (gradient != NULL) {
        check_mts(mts_block_gradient(block, gradient, &block), "mts_block_gradient");
    }
    mts_array_t array = {0};
    check_mts(mts_block_data(block, &array), "mts_block_data");

    DLManagedTensorVersioned *view = NULL;
    DLPackVersion version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    check_mts(array.as_dlpack(array.ptr, &view, (DLDevice){kDLCPU, 0}, NULL, version), "as_dlpack");
    if (array.destroy != NULL) {
        array.destroy(array.ptr);
    }

    size_t count = 1;
    for (int32_t i = 0; i < view->dl_tensor.ndim; i++) {
        count *= (size_t)view->dl_tensor.shape[i];
    }
    if (count > max) {
        fprintf(stderr, "error: output too large\n");
        exit(EXIT_FAILURE);
    }
    // outputs are freshly allocated contiguous arrays
    memcpy(out, (char *)view->dl_tensor.data + view->dl_tensor.byte_offset, count * sizeof(double));
    view->deleter(view);
    return count;
}

static mts_block_t *single_block(mts_tensormap_t *tensor) {
    mts_block_t *block = NULL;
    check_mts(mts_tensormap_block_by_id(tensor, &block, 0), "mts_tensormap_block_by_id");
    return block;
}

static bool nc_forces = false;

typedef struct {
    double energy;
    double gradient[3 * MAX_ATOMS]; // dE/dx
    double strain[9];               // dE/de
    double nc_forces[3 * MAX_ATOMS];
} Prediction;

static void run(mta_model_t model, const Structure *structures, int count, Prediction *out,
                bool full) {
    mta_system_t *systems[8];
    for (int i = 0; i < count; i++) {
        systems[i] = make_system(&structures[i]);
    }

    const char *energy_request =
        "{\"type\": \"metatomic_quantity\", \"name\": \"energy\", \"unit\": \"eV\", "
        "\"sample_kind\": \"system\", \"gradients\": [%s]}";
    const char *nc_request =
        "{\"type\": \"metatomic_quantity\", \"name\": \"non_conservative_force\", "
        "\"unit\": \"eV/Angstrom\", \"sample_kind\": \"atom\", \"gradients\": []}";
    char energy_json[256], requests[512];
    snprintf(energy_json, sizeof(energy_json), energy_request, full ? "\"positions\", \"strain\"" : "");
    if (full && nc_forces) {
        snprintf(requests, sizeof(requests), "[%s, %s]", energy_json, nc_request);
    } else {
        snprintf(requests, sizeof(requests), "[%s]", energy_json);
    }

    mts_tensormap_t *outputs[2] = {NULL, NULL};
    uintptr_t n_outputs = full ? (nc_forces ? 2 : 1) : 1;
    check(mta_execute_model(model, (const mta_system_t *const *)systems, (uintptr_t)count, NULL,
                            requests,
                            /*check_consistency*/ true, outputs, n_outputs),
          "mta_execute_model");

    static double buffer[MAX_ATOMS * 3 * 8];
    mts_block_t *energy = single_block(outputs[0]);
    block_values(energy, NULL, buffer, 8);
    for (int i = 0; i < count; i++) {
        out[i].energy = buffer[i];
    }

    if (full) {
        // gradient samples are (sample, system, atom), in order
        block_values(energy, "positions", buffer, sizeof(buffer) / sizeof(double));
        double *g = buffer;
        for (int i = 0; i < count; i++) {
            memcpy(out[i].gradient, g, sizeof(double) * 3 * (size_t)structures[i].n_atoms);
            g += 3 * structures[i].n_atoms;
        }
        block_values(energy, "strain", buffer, sizeof(buffer) / sizeof(double));
        for (int i = 0; i < count; i++) {
            memcpy(out[i].strain, buffer + 9 * i, sizeof(double) * 9);
        }
        if (!nc_forces) {
            memset(buffer, 0, sizeof(buffer));
        } else {
            block_values(single_block(outputs[1]), NULL, buffer, sizeof(buffer) / sizeof(double));
        }
        g = buffer;
        for (int i = 0; i < count; i++) {
            memcpy(out[i].nc_forces, g, sizeof(double) * 3 * (size_t)structures[i].n_atoms);
            g += 3 * structures[i].n_atoms;
        }
    }

    for (uintptr_t i = 0; i < n_outputs; i++) {
        mts_tensormap_free(outputs[i]);
    }
    for (int i = 0; i < count; i++) {
        mta_system_free(systems[i]);
    }
}

static double energy_of(mta_model_t model, const Structure *s) {
    Prediction p;
    run(model, s, 1, &p, false);
    return p.energy;
}

// Positions and cell (where periodic) deformed by (1 + e)
static Structure strained(const Structure *s, int a, int b, double e) {
    Structure out = *s;
    for (int i = 0; i < s->n_atoms; i++) {
        out.positions[3 * i + b] += e * s->positions[3 * i + a];
    }
    for (int i = 0; i < 3; i++) {
        out.cell[3 * i + b] += e * s->cell[3 * i + a];
    }
    return out;
}

// Rattled 8-atom diamond silicon, periodic along the first `periodic` axes
static Structure silicon(const char *name, int periodic) {
    static const double BASIS[8][3] = {
        {0.00, 0.00, 0.00}, {0.00, 0.50, 0.50}, {0.50, 0.00, 0.50}, {0.50, 0.50, 0.00},
        {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75}, {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25},
    };
    const double a = 5.43;
    Structure s = {.name = name, .n_atoms = 8};
    srand(42);
    for (int i = 0; i < 8; i++) {
        s.types[i] = 14;
        for (int k = 0; k < 3; k++) {
            double noise = 0.05 * ((double)rand() / RAND_MAX - 0.5);
            s.positions[3 * i + k] = a * BASIS[i][k] + noise;
        }
    }
    for (int k = 0; k < periodic; k++) {
        s.pbc[k] = true;
        s.cell[4 * k] = a;
    }
    return s;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <mlipcpp-metatomic.so> <model.gguf> [options-json]\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE *gguf = fopen(argv[2], "rb");
    if (gguf == NULL) {
        fprintf(stderr, "skipping: no model at %s\n", argv[2]);
        return 77;
    }
    fclose(gguf);

    check(mta_load_plugin(argv[1]), "mta_load_plugin");
    mta_model_t model = {0};
    check(mta_load_model(argv[2], argc > 3 ? argv[3] : NULL, NULL, &model), "mta_load_model");

    mta_string_t metadata = NULL;
    check(model.metadata(model.data, &metadata), "metadata");
    mta_string_t printed = NULL;
    check(mta_format_metadata(mta_string_view(metadata), &printed), "mta_format_metadata");
    printf("%s\n", mta_string_view(printed));
    mta_string_free(printed);
    mta_string_free(metadata);

    mta_string_t capabilities = NULL;
    check(model.capabilities(model.data, &capabilities), "capabilities");
    nc_forces = strstr(mta_string_view(capabilities), "non_conservative_force") != NULL;
    mta_string_free(capabilities);

    Structure structures[3] = {
        silicon("Si8", 0),
        silicon("Si slab", 2),
        silicon("Si bulk", 3),
    };
    Prediction predictions[3];
    run(model, structures, 3, predictions, true);

    const double h = 1e-3;
    int failures = 0;
    for (int i = 0; i < 3; i++) {
        const Structure *s = &structures[i];
        const Prediction *p = &predictions[i];

        double fd_max = 0.0, grad_max = 0.0, nc_diff = 0.0;
        for (int j = 0; j < 3 * s->n_atoms; j++) {
            Structure plus = *s, minus = *s;
            plus.positions[j] += h;
            minus.positions[j] -= h;
            double fd = (energy_of(model, &plus) - energy_of(model, &minus)) / (2 * h);
            fd_max = fmax(fd_max, fabs(fd - p->gradient[j]));
            grad_max = fmax(grad_max, fabs(p->gradient[j]));
            nc_diff += pow(p->nc_forces[j] + p->gradient[j], 2);
        }

        double strain_max = 0.0, virial_max = 0.0;
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                Structure plus = strained(s, a, b, h), minus = strained(s, a, b, -h);
                double fd = (energy_of(model, &plus) - energy_of(model, &minus)) / (2 * h);
                strain_max = fmax(strain_max, fabs(fd - p->strain[3 * a + b]));
                virial_max = fmax(virial_max, fabs(p->strain[3 * a + b]));
            }
        }

        printf("%-8s E = %12.6f eV | max|dE/dx| %.4f, FD error %.2e | max|dE/de| %.4f, FD error %.2e"
               " | rms(F_nc - F) %.4f eV/A\n",
               s->name, p->energy, grad_max, fd_max, virial_max, strain_max,
               sqrt(nc_diff / (3 * s->n_atoms)));
        // float32 model evaluated with float32 finite differences. The strain
        // check is loose: PET models with an adaptive cutoff have a stress that
        // differs from finite differences of their energy by up to ~30% (the
        // same in PyTorch); this still catches sign and convention errors.
        failures += fd_max > 2e-2 + 2e-2 * grad_max;
        failures += strain_max > 1e-1 + 0.5 * virial_max;
    }

    check(model.unload(model.data), "unload");
    if (failures != 0) {
        printf("%d finite-difference checks FAILED\n", failures);
        return EXIT_FAILURE;
    }
    printf("all finite-difference checks passed\n");
    return EXIT_SUCCESS;
}
