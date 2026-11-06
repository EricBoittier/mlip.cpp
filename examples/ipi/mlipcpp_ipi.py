"""i-PI PES: mlip.cpp C API (libmlipcpp_c.so), periodic cell, optional total charge / spin multiplicity.

Works with stock i-PI >= 3.3 as a custom PES, no i-PI patching:
  <ffdirect name="mlip"><pes>custom</pes><pes_path>mlipcpp_ipi.py</pes_path>
    <parameters>{template: start.xyz, model: pet-omol-s.gguf, lib: /path/libmlipcpp_c.so, backend: cuda, charge: 0, spin: 1,
                 precision: fp32}</parameters>
  </ffdirect>
or as a socket client:
  i-pi-py_driver -m custom -P mlipcpp_ipi.py -u -a mlip -o template=start.xyz,model=pet-omol-s.gguf,backend=cuda

Requires a mlip.cpp build whose mlipcpp_predict_options_t has charge and spin_multiplicity (the EricBoittier fork
branches feat/pet-plato-lr and later, incl. perf/conditioned-ragged). For unconditioned models (PET-MOLS-s, PET-MAD)
leave charge=0, spin=1.

precision (CUDA matmuls of this model; needs feat/model-precision or later):
  default  full FP32, or TF32 when MLIPCPP_TF32=1 is set (the behaviour without this option)
  fp32     every product in full FP32, also when the process runs TF32
  tf32     TF32 tensor cores (the edge-geometry product stays in FP32)
TF32 is a process-wide CUDA setting read once at the first matmul. Two <ffdirect> force fields of one i-PI run can
differ (say a TF32 model on the full ring polymer and an FP32 one on the contracted beads): i-PI builds every
force field before the first step, which is what mlip.cpp needs (a TF32 model loaded after the first FP32 predict
is an error). In a process that runs TF32 an fp32 model loses the fast FP32-only kernels (cuBLAS full-FP32 GEMMs
instead), so prefer default for a process with FP32 models only.
"""
import ctypes
import os

import numpy as np

from ipi.pes.dummy import Dummy_driver
from ipi.utils.units import unit_to_internal, unit_to_user

__DRIVER_NAME__ = "mlipcpp_ipi"
__DRIVER_CLASS__ = "MLIPCppDriver"

_BACKENDS = {"auto": 0, "cpu": 1, "cuda": 2, "hip": 3, "metal": 4, "vulkan": 5}
# mlipcpp_precision_t: F32 = 0 (default), TF32 = 2, F32_EXACT = 3
_PRECISIONS = {"default": 0, "auto": 0, "fp32": 3, "f32": 3, "tf32": 2}


class _ModelOptions(ctypes.Structure):
    _fields_ = [("backend", ctypes.c_int), ("precision", ctypes.c_int), ("cutoff_override", ctypes.c_float)]


class _PredictOptions(ctypes.Structure):
    _fields_ = [("compute_forces", ctypes.c_bool), ("compute_stress", ctypes.c_bool),
                ("use_nc_forces", ctypes.c_bool), ("charge", ctypes.c_int32),
                ("spin_multiplicity", ctypes.c_int32)]


def _flag(v):
    return str(v).lower() in ("1", "true", "yes", "on")


class MLIPCppDriver(Dummy_driver):
    def __init__(self, template, model, lib=None, backend="cuda", charge=0, spin=1, pbc=True, stress=True,
                 precision="default", *args, **kwargs):
        self.template, self.model_path = str(template), str(model)
        self.lib_path = lib or os.environ.get("MLIPCPP_C_LIB")
        if not self.lib_path:
            raise ValueError("give lib=/path/to/libmlipcpp_c.so or set MLIPCPP_C_LIB")
        self.backend = _BACKENDS[str(backend).lower()]
        if str(precision).lower() not in _PRECISIONS:
            raise ValueError(f"precision={precision!r}: one of {sorted(_PRECISIONS)}")
        self.precision = _PRECISIONS[str(precision).lower()]
        self.charge, self.spin = int(charge), int(spin)
        self.use_pbc, self.want_stress = _flag(pbc), _flag(stress)
        super().__init__(*args, **kwargs)

    def check_parameters(self):
        import ase.io

        lib = ctypes.CDLL(self.lib_path)
        lib.mlipcpp_set_backend.argtypes = [ctypes.c_int]
        lib.mlipcpp_model_options_default.argtypes = [ctypes.POINTER(_ModelOptions)]
        lib.mlipcpp_model_create.argtypes = [ctypes.POINTER(_ModelOptions)]
        lib.mlipcpp_model_create.restype = ctypes.c_void_p
        lib.mlipcpp_model_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.mlipcpp_predict_ptr_with_options.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(_PredictOptions), ctypes.POINTER(ctypes.c_void_p)]
        lib.mlipcpp_result_get_energy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.mlipcpp_result_get_forces.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.mlipcpp_result_get_stress.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.mlipcpp_result_free.argtypes = [ctypes.c_void_p]
        lib.mlipcpp_get_last_error.restype = ctypes.c_char_p
        lib.mlipcpp_set_backend(self.backend)

        opts = _ModelOptions()
        lib.mlipcpp_model_options_default(ctypes.byref(opts))
        opts.backend = self.backend
        opts.precision = self.precision
        self.lib = lib
        self.model = lib.mlipcpp_model_create(ctypes.byref(opts))
        if not self.model or lib.mlipcpp_model_load(self.model, self.model_path.encode()) != 0:
            raise RuntimeError(f"mlip.cpp could not load {self.model_path}: {lib.mlipcpp_get_last_error()}")

        self.numbers = np.ascontiguousarray(ase.io.read(self.template).numbers, dtype=np.int32)
        self.n = len(self.numbers)
        self.forces = np.zeros((self.n, 3), dtype=np.float32)
        self.stress6 = np.zeros(6, dtype=np.float32)
        self.pbc = np.array([self.use_pbc] * 3, dtype=np.bool_)
        self.popts = _PredictOptions(True, self.want_stress and self.use_pbc, False, self.charge, self.spin)

    def compute_structure(self, cell, pos):
        x = np.ascontiguousarray(unit_to_user("length", "angstrom", pos).reshape(-1, 3), dtype=np.float32)
        h = unit_to_user("length", "angstrom", np.asarray(cell))  # i-PI: columns are lattice vectors
        c = np.ascontiguousarray(h.T, dtype=np.float32).reshape(9)  # mlip.cpp: rows are lattice vectors
        res = ctypes.c_void_p()
        st = self.lib.mlipcpp_predict_ptr_with_options(
            self.model, self.n, x.ctypes.data, self.numbers.ctypes.data,
            c.ctypes.data if self.use_pbc else None, self.pbc.ctypes.data if self.use_pbc else None,
            ctypes.byref(self.popts), ctypes.byref(res))
        if st != 0:
            raise RuntimeError(self.lib.mlipcpp_get_last_error().decode())
        e = ctypes.c_float()
        self.lib.mlipcpp_result_get_energy(res, ctypes.byref(e))
        self.lib.mlipcpp_result_get_forces(res, self.forces.ctypes.data, self.n)
        vir = np.zeros((3, 3))
        if self.popts.compute_stress:
            self.lib.mlipcpp_result_get_stress(res, self.stress6.ctypes.data)
            xx, yy, zz, yz, xz, xy = self.stress6.astype(np.float64)  # Voigt, eV/A^3
            sigma = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            vol = abs(np.linalg.det(h))
            vir = unit_to_internal("energy", "electronvolt", -sigma * vol)
        self.lib.mlipcpp_result_free(res)
        pot = unit_to_internal("energy", "electronvolt", float(e.value))
        frc = unit_to_internal("force", "ev/ang", self.forces.astype(np.float64)).reshape(np.shape(pos))
        return pot, frc, vir, ""
