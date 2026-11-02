#!/usr/bin/env python3
"""A GGUF written by mlipcpp-train -> mmml portable JSON -> mmml (JAX) energies and forces,
against mlip.cpp on the GGUF itself, on structures of a mlipcpp-train split.

    PYTHONPATH=$MMML:build/lib:python python scripts/mmml/roundtrip_check.py model.gguf \\
        --split DATA/val --n 64 [--json-out model_mmml.json] [--x64]

Prints max |dE|, |dE|/atom and |dF| (eV, eV/A) and the energies of the first structures.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mmml"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--x64", action="store_true")
    a = ap.parse_args()
    import convert_mmml as cm
    import mlipcpp
    from mmml_ref import JaxRef

    js = a.json_out or os.path.join(tempfile.mkdtemp(), "model.json")
    info = cm.to_json(a.gguf, js)
    ref = JaxRef(js, x64=a.x64)

    sp = Path(a.split)
    na = np.load(sp / "na.npy")
    X = np.fromfile(sp / "x.bin", np.float32).reshape(-1, 3)
    Z = np.fromfile(sp / "a.bin", np.int32)
    Q = np.fromfile(sp / "charge.bin", np.float32)
    S = np.fromfile(sp / "spin_multiplicity.bin", np.float32)
    n = min(a.n, len(na) - 1)
    mols = [dict(Z=Z[na[i]:na[i + 1]], R=X[na[i]:na[i + 1]].astype(np.float64), Q=int(Q[i]), S=int(S[i]))
            for i in range(n)]
    Ej, Fj = ref(mols)

    pred = mlipcpp.Predictor(a.gguf)
    Em, Fm = [], []
    for m in mols:
        r = pred.predict(np.ascontiguousarray(m["R"], np.float32), np.ascontiguousarray(m["Z"], np.int32),
                         charge=m["Q"], spin_multiplicity=m["S"])
        Em.append(float(r.energy)), Fm.append(np.asarray(r.forces, np.float64))
    Em = np.asarray(Em)
    N = np.array([len(m["Z"]) for m in mols])
    dE = np.abs(Em - Ej)
    dF = max(np.abs(f1 - f2).max() for f1, f2 in zip(Fm, Fj))
    out = dict(gguf=a.gguf, json=js, n_structures=n, max_dE=float(dE.max()), max_dE_per_atom=float((dE / N).max()),
               max_dF=float(dF), max_abs_E=float(np.abs(Ej).max()), x64=a.x64,
               first=[dict(N=int(N[i]), Q=mols[i]["Q"], S=mols[i]["S"], E_mlipcpp=float(Em[i]), E_jax=float(Ej[i]))
                      for i in range(min(4, n))])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
