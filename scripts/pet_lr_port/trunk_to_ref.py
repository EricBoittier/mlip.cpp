#!/usr/bin/env python3
"""A ref_trunk.py dump (per-atom trunk energies `energy`, forces, virial, atom_ptr) in the lr_parity.py layout
(`atom/energy`, `forces`, `virial`, `atom_ptr`), so trunk-only models (e.g. his les-water backbones) get the same
per-output table, floor verdict and paired P as the LR models.

    python trunk_to_ref.py DUMP.npz OUT.ref.npz
"""
import sys

import numpy as np

z = np.load(sys.argv[1])
np.savez(sys.argv[2], atom_ptr=z["atom_ptr"], **{"atom/energy": z["energy"]}, forces=z["forces"], virial=z["virial"])
