"""Copy a pet-graph GGUF for energy *differences*: composition baselines set to zero and pet.forward_only = 1.

usage: python make_delta_gguf.py pet-mols-s.gguf pet-mols-s-delta.gguf

The C API returns per-atom values as float32(energy_scale * y_i + composition[Z_i]). With the composition of a
heavy element (~ -1e4 eV) the float32 rounding is ~1e-3 eV per atom, too coarse for the per-molecule potential
differences of the mass-scaling estimator. Composition energies cancel in any difference at fixed composition, so the
copy drops them (per-atom values become energy_scale * y_i, resolved to ~1e-7 eV) and skips the backward graph.
"""
import sys

import gguf
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
w = gguf.GGUFWriter(dst, arch=arch)
for f in r.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF.") or f.name == "pet.forward_only":
        continue
    vt = f.types[0]
    sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = f.contents()
    if f.name == "pet.composition_values":
        val = [0.0] * len(val)
    w.add_key_value(f.name, val, vt, sub_type=sub)
w.add_key_value("pet.forward_only", 1, gguf.GGUFValueType.INT32)
for t in r.tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in r.tensors:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print(dst, "arch", arch, "tensors", len(r.tensors))
