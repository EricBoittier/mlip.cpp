#!/usr/bin/env python3
"""Parity of a PET-Plato-LR GGUF (mlip.cpp, float32) against metatrain (float64,
autograd): energy, forces and stress of open molecules and periodic cells, a few
(charge, spin) pairs (the LES total-charge constraint and the conditioning).

    python parity_pet_plato_lr.py --model m.pt --gguf m.gguf --eval build/bin/mlipcpp-eval
        [--backend cpu] [--out DIR] [--json table.json] [--set molecules|crystals|all]
        [--mlipcpp-json OUT/mlipcpp.json]   (mlipcpp-eval run elsewhere, e.g. a GPU node)

--model: make_pet_plato_lr.py's dump (or a checkpoint). Prints one row per structure and
exits non-zero above the tolerances (|dE|/atom 2e-4 eV, |dF| 2e-3 eV/A, |dS| 2e-4 eV/A^3).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import ase
import ase.build
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conditioning"))
from export_pytorch.export_pet_full import load_local_model  # noqa: E402
from parity_pet import metatrain_eval, write_frames  # noqa: E402

TOL_E, TOL_F, TOL_S = 2e-4, 2e-3, 2e-4


def molecules(rng):
    out = []
    for name in ("H2O", "CH4", "CH3CH2OH", "CH3NO2"):
        a = ase.build.molecule(name)
        a.positions += rng.normal(0, 0.05, a.positions.shape)
        out.append((name, a))
    # two molecules 6 A apart (long range beyond the cutoff), and a larger cluster
    d = ase.build.molecule("H2O") + ase.build.molecule("NH3")
    d.positions[3:] += [6.0, 0.5, 0.2]
    out.append(("dimer6A", d))
    c = ase.Atoms()
    for k, name in enumerate(("CH3OH", "H2O", "NH3", "CH4", "H2O", "HCOOH")):
        m = ase.build.molecule(name)
        m.positions += [3.2 * (k % 3), 3.4 * (k // 3), 0.6 * k]
        c += m
    c.positions += rng.normal(0, 0.05, c.positions.shape)
    out.append(("cluster", c))
    return out


def crystals(rng):
    out = []
    box = ase.Atoms()
    for k, name in enumerate(("H2O", "NH3", "CH4")):
        m = ase.build.molecule(name)
        m.positions += np.array([1.5 + 2.2 * k, 2.0 + 0.7 * k, 1.8 + 1.1 * k])
        box += m
    box.cell = [[6.5, 0, 0], [0.4, 6.2, 0], [-0.3, 0.5, 6.0]]
    box.pbc = True
    box.positions += rng.normal(0, 0.05, box.positions.shape)
    out.append(("box", box))
    # a molecular crystal-like cell: 4 water molecules in a skewed 7 A cell
    cr = ase.Atoms()
    for k in range(4):
        m = ase.build.molecule("H2O")
        m.rotate(37 * k, "z")
        m.positions += [1.0 + 3.4 * (k % 2), 1.2 + 3.3 * (k // 2), 1.5 + 1.7 * k]
        cr += m
    cr.cell = [[7.0, 0, 0], [0.8, 6.8, 0], [0.3, -0.4, 7.4]]
    cr.pbc = True
    cr.positions += rng.normal(0, 0.05, cr.positions.shape)
    out.append(("crystal", cr))
    big = cr.repeat((2, 1, 1))  # 24 atoms
    big.positions += rng.normal(0, 0.03, big.positions.shape)
    out.append(("crystal2x1", big))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--eval")
    ap.add_argument("--mlipcpp-json")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--out", default="parity-plato-lr")
    ap.add_argument("--json")
    ap.add_argument("--set", choices=["molecules", "crystals", "all"], default="all")
    ap.add_argument("--pairs", default="0,1;1,2;-1,2", help="(charge, 2S+1) pairs, ';'-separated")
    ap.add_argument("--exact-padding", action="store_true",
                    help="padded neighbour slots masked exactly on both sides (metatrain: attention epsilon "
                         "1e-300 in float64; mlip.cpp: MLIPCPP_PAD_EXACT=1): PET's finite -34.5 bias lets the "
                         "padded slots leak into the frame RoPE attention, so the outputs depend on their contents")
    ap.add_argument("--write-only", action="store_true", help="write OUT/frames.xyz and stop")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    structs = (molecules(rng) if args.set != "crystals" else []) + (crystals(rng) if args.set != "molecules" else [])
    for _, a in structs:
        a.positions = a.positions.astype(np.float32).astype(np.float64)
        if a.pbc.any():
            a.cell = a.cell.array.astype(np.float32).astype(np.float64)
    pairs = [tuple(int(x) for x in p.split(",")) for p in args.pairs.split(";")]
    model = load_local_model(args.model).to(torch.float64).eval()
    if args.exact_padding:
        from metatrain.pet.modules.transformer import AttentionBlock
        for mod in model.modules():
            if isinstance(mod, AttentionBlock):
                mod.epsilon = 1e-300
    if getattr(model, "system_conditioning", None) is None and getattr(model.backend, "les", None) is None:
        pairs = [(0, 1)]
    frames = [(n, a, q, s) for n, a in structs for q, s in pairs]
    write_frames(out / "frames.xyz", frames)
    if args.write_only:
        return
    if args.mlipcpp_json:
        got = json.load(open(args.mlipcpp_json))
        args.backend = got.get("backend", args.backend)
        got = got["frames"]
    else:
        import os
        env = dict(os.environ, **({"MLIPCPP_PAD_EXACT": "1"} if args.exact_padding else {}))
        run = subprocess.run([args.eval, args.gguf, str(out / "frames.xyz"), str(out / "mlipcpp.json"),
                              "--backend", args.backend], capture_output=True, text=True, env=env)
        if run.returncode != 0:
            print(run.stdout, run.stderr)
            raise SystemExit("mlipcpp-eval failed")
        got = json.load(open(out / "mlipcpp.json"))["frames"]
    rows, ok = [], True
    print(f"{'structure':11s} {'n':>3s} {'q':>3s} {'2S+1':>4s} {'E_mtt (eV)':>14s} {'|dE|/atom':>10s} "
          f"{'max|dF|':>10s} {'max|dS|':>10s}")
    for (name, a, q, s), g in zip(frames, got):
        if "error" in g:
            print(name, q, s, "mlip.cpp error:", g["error"])
            ok = False
            continue
        e, f, st = metatrain_eval(model, a, q, s, symmetric=True)
        de = abs(g["energy"] - e) / len(a)
        df = float(np.abs(np.array(g["forces"]).reshape(-1, 3) - f).max())
        ds = float(np.abs(np.array(g["stress"]) - np.array(st)).max()) if st is not None else float("nan")
        row = dict(structure=name, n_atoms=len(a), charge=q, spin=s, e_metatrain=e, e_mlipcpp=g["energy"],
                   de_per_atom=de, df_max=df, ds_max=ds, backend=args.backend,
                   f_rms=float(np.sqrt((f ** 2).mean())))
        ok &= de < TOL_E and df < TOL_F and (st is None or ds < TOL_S)
        rows.append(row)
        print(f"{name:11s} {len(a):3d} {q:3d} {s:4d} {e:14.6f} {de:10.2e} {df:10.2e} {ds:10.2e}")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
