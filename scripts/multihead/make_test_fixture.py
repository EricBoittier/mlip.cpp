#!/usr/bin/env python3
"""Fixture of tests/test_energy_heads.cpp (MLIPCPP_TEST_MULTIHEAD_DIR): from a single-head PET GGUF,
  mh.gguf   two heads, head 1 a perturbed copy of head 0 (readout weights + noise, its own composition);
  ref1.gguf single head = mh.gguf's head 1 (tensors and composition);
  d1/{train,val,test}  six small molecules, all on head 1 (head.bin = 1), energy_weight 1 except structure 2 (0);
  d0/{train,val,test}  the same without head.bin / energy_weight.bin, except energy_weight.bin as in d1.
    python make_test_fixture.py PET.gguf OUTDIR"""
import os
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from gguf_raw import GGUF  # noqa: E402
from memmap import SplitWriter  # noqa: E402


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    subprocess.run([sys.executable, "-I", os.path.join(here, "add_energy_heads.py"), "add", src, f"{out}/mh0.gguf"], check=True)
    g = GGUF(f"{out}/mh0.gguf")
    rng = np.random.default_rng(7)
    ws = [t for t in g.tensors if t.startswith("head1.")]
    for w in ws:
        a = g.array(w)
        g.set_tensor(w, a + 0.03 * np.std(a) * rng.normal(size=a.shape).astype(np.float32))
    ck, cv = g.get_array("pet.head1.composition_keys"), g.get_array("pet.head1.composition_values")
    g.set_array("pet.head1.composition_values", [v + 0.05 * (k % 3) - 0.3 for k, v in zip(ck, cv)], 6)
    g.write(f"{out}/mh.gguf")
    subprocess.run([sys.executable, "-I", os.path.join(here, "add_energy_heads.py"), "select", f"{out}/mh.gguf",
                    f"{out}/ref1.gguf", "--head", "1"], check=True)
    # six molecules (H, C, N, O), jittered
    mols = {
        "H2O": ([8, 1, 1], [[0, 0, 0.12], [0, 0.76, -0.47], [0, -0.76, -0.47]]),
        "NH3": ([7, 1, 1, 1], [[0, 0, 0.12], [0, 0.94, -0.27], [0.81, -0.47, -0.27], [-0.81, -0.47, -0.27]]),
        "CH4": ([6, 1, 1, 1, 1], [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]),
        "HCOOH": ([6, 8, 8, 1, 1], [[0, 0.42, 0], [1.08, 0.94, 0], [-1.12, 1.07, 0], [0.04, -0.68, 0], [1.83, 0.30, 0]]),
        "CH3OH": ([6, 8, 1, 1, 1, 1], [[-0.05, 0.66, 0], [-0.05, -0.76, 0], [-1.09, 0.97, 0], [0.43, 1.08, 0.89], [0.43, 1.08, -0.89], [0.86, -1.04, 0]]),
        "HCN": ([1, 6, 7], [[0, 0, -1.6], [0, 0, -0.5], [0, 0, 0.65]]),
    }
    data = []
    for k, (Z, R) in enumerate(mols.values()):
        R = np.array(R) + 0.04 * rng.normal(size=(len(Z), 3))
        data.append((Z, R, 0.3 * rng.normal(size=(len(Z), 3)), -5.0 * len(Z) + rng.normal(), 0 if k % 3 else 1, 1 if k % 3 else 2))
    for d, heads in (("d1", True), ("d0", False)):
        for split in ("train", "val", "test"):
            w = SplitWriter(f"{out}/{d}/{split}", heads=heads)
            for k, (Z, R, F, E, q, s) in enumerate(data):
                w.add(Z, R, F, E, charge=q, spin=s, head=1, energy_weight=0.0 if k == 2 else 1.0)
            w.close()
            if not heads:  # the energy weights alone
                np.full(len(data), 1.0, np.float32).tofile(f"{out}/{d}/{split}/energy_weight.bin")
                ew = np.fromfile(f"{out}/{d}/{split}/energy_weight.bin", np.float32); ew[2] = 0; ew.tofile(f"{out}/{d}/{split}/energy_weight.bin")
    print("fixture in", out)


if __name__ == "__main__":
    main()
