"""Writer of one metatrain MemmapDataset split as mlipcpp-train reads it (src/train/dataset.h), plus the per-structure
multi-head files: head.bin (energy head index) and energy_weight.bin (0 = forces only), both float32 [ns]."""
import os

import numpy as np


class SplitWriter:
    """Append structures (Z, R, F, E, charge, spin, head, energy_weight, cell=None), then close()."""

    FILES = ("x.bin", "a.bin", "c.bin", "energy.bin", "non_conservative_force.bin", "charge.bin",
             "spin_multiplicity.bin", "head.bin", "energy_weight.bin")

    def __init__(self, out_dir, heads=True, extra=()):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.heads = out_dir, heads
        self.f = {n: open(os.path.join(out_dir, n), "wb") for n in self.FILES if heads or n not in ("head.bin", "energy_weight.bin")}
        self.extra = {n: open(os.path.join(out_dir, n), "wb") for n in extra}  # name -> raw float32 per structure rows
        self.na = [0]

    def add(self, Z, R, F, E, charge=0, spin=1, head=0, energy_weight=1.0, cell=None, **extra):
        Z = np.asarray(Z, np.int32); n = len(Z)
        self.f["x.bin"].write(np.asarray(R, np.float32).reshape(n, 3).tobytes())
        self.f["a.bin"].write(Z.tobytes())
        self.f["c.bin"].write((np.zeros((3, 3), np.float32) if cell is None else np.asarray(cell, np.float32)).tobytes())
        self.f["energy.bin"].write(np.float32(E).tobytes())
        self.f["non_conservative_force.bin"].write(np.asarray(F, np.float32).reshape(n, 3).tobytes())
        self.f["charge.bin"].write(np.float32(charge).tobytes())
        self.f["spin_multiplicity.bin"].write(np.float32(spin).tobytes())
        if self.heads:
            self.f["head.bin"].write(np.float32(head).tobytes())
            self.f["energy_weight.bin"].write(np.float32(energy_weight).tobytes())
        for k, v in extra.items():
            self.extra[k].write(np.asarray(v, np.float32).tobytes())
        self.na.append(self.na[-1] + n)

    def add_block(self, na, Z, R, F, E, charge, spin, head, energy_weight, cells=None, **extra):
        """Many structures at once: na = per-structure atom counts, Z/R/F concatenated."""
        na = np.asarray(na, np.int64); ns = len(na)
        self.f["x.bin"].write(np.ascontiguousarray(R, np.float32).tobytes())
        self.f["a.bin"].write(np.ascontiguousarray(Z, np.int32).tobytes())
        self.f["c.bin"].write((np.zeros((ns, 3, 3), np.float32) if cells is None else np.ascontiguousarray(cells, np.float32)).tobytes())
        self.f["energy.bin"].write(np.ascontiguousarray(E, np.float32).tobytes())
        self.f["non_conservative_force.bin"].write(np.ascontiguousarray(F, np.float32).tobytes())
        self.f["charge.bin"].write(np.ascontiguousarray(charge, np.float32).tobytes())
        self.f["spin_multiplicity.bin"].write(np.ascontiguousarray(spin, np.float32).tobytes())
        if self.heads:
            self.f["head.bin"].write(np.ascontiguousarray(head, np.float32).tobytes())
            self.f["energy_weight.bin"].write(np.ascontiguousarray(energy_weight, np.float32).tobytes())
        for k, v in extra.items():
            self.extra[k].write(np.ascontiguousarray(v, np.float32).tobytes())
        c = np.cumsum(na) + self.na[-1]
        self.na.extend(c.tolist())

    def close(self):
        for fh in list(self.f.values()) + list(self.extra.values()):
            fh.close()
        np.save(os.path.join(self.dir, "ns.npy"), np.array(len(self.na) - 1, np.int64))
        np.save(os.path.join(self.dir, "na.npy"), np.array(self.na, np.int64))
