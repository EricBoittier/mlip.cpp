"""Check a make_memmap.py split against its source arrays: python -I verify_memmap.py MEMMAP_DIR SRC.npz [ATOM_REFS.json]"""
import json
import sys

import numpy as np

m, src = sys.argv[1], sys.argv[2]
d = dict(np.load(src))  # npz members decode on every access: load once
ns = int(np.load(f"{m}/ns.npy")); na = np.load(f"{m}/na.npy"); N = int(na[-1])
assert ns == len(d["n"]) and N == int(d["n"].sum()) and (np.diff(na) == d["n"]).all()
for f, w, dt in (("x", 3, np.float32), ("a", 1, np.int32), ("c", 9 / N * ns, np.float32), ("non_conservative_force", 3, np.float32),
                 ("energy_forces", 3, np.float32), ("mulliken_charge", 1, np.float32), ("lowdin_charge", 1, np.float32),
                 ("nbo_charge", 1, np.float32)):
    try:
        sz = np.fromfile(f"{m}/{f}.bin", dt).size
    except FileNotFoundError:
        print("absent", f); continue
    assert sz == round(w * N), (f, sz, w * N)
for f in ("energy", "charge", "spin_multiplicity"):
    assert np.fromfile(f"{m}/{f}.bin", np.float32).size == ns, f
x = np.fromfile(f"{m}/x.bin", np.float32).reshape(-1, 3); assert np.abs(x - d["R"]).max() < 1e-3
a = np.fromfile(f"{m}/a.bin", np.int32); assert (a == d["Z"]).all()
F = np.fromfile(f"{m}/non_conservative_force.bin", np.float32).reshape(-1, 3); assert np.abs(F - d["F"]).max() < 1e-4
q = np.fromfile(f"{m}/charge.bin", np.float32); assert (q == d["charge"]).all()
s = np.fromfile(f"{m}/spin_multiplicity.bin", np.float32); assert (s == d["spin_multiplicity"]).all()
e = np.fromfile(f"{m}/energy.bin", np.float32)
if len(sys.argv) > 3:
    b = np.zeros(119)
    for z, v in json.load(open(sys.argv[3]))["composition"].items():
        b[int(z)] = v
    off = np.concatenate([[0], np.cumsum(d["n"])])
    ref = np.array([d["E"][k] - b[d["Z"][off[k]:off[k + 1]]].sum() for k in range(ns)])
    assert np.abs(e - ref).max() < 1e-2 * np.maximum(1, np.abs(ref)).max(), np.abs(e - ref).max()
    print("energy residual range %.2f .. %.2f eV, max float32 error %.2e eV" % (ref.min(), ref.max(), np.abs(e - ref).max()))
print("memmap verified", m, ns, "structures", N, "atoms")
