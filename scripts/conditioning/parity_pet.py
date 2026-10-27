#!/usr/bin/env python3
"""Charge / spin conditioning parity of a pet-graph (or, --arch plato, plato-pet-lr) GGUF
(mlip.cpp) against metatrain
(float64, autograd) and metatomic (the exported AtomisticModel through its ASE
calculator, float32), on isolated and periodic structures for several (charge, spin
multiplicity) pairs, plus the out-of-range errors.

    python parity_pet.py --model pet-cond.pt --gguf pet-cond.gguf --eval build/bin/mlipcpp-eval
        [--mta pet-cond-mta.pt] [--backend cpu] [--out DIR] [--ckpt pet-cond-rt.ckpt]

--model: the make_pet_cond.py dump (or any {"model": PET} / metatrain checkpoint).
--ckpt: a second checkpoint (e.g. gguf_to_checkpoint.py's round trip) that must give
the same metatrain energies bit for bit. Prints one table row per structure and
(charge, spin); exits non-zero if a tolerance is exceeded (float32: |dE| < 1e-4 eV
per atom... see TOL below).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import ase
import ase.build
import ase.io
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_pytorch.export_pet_full import load_local_model  # noqa: E402

PAIRS = [(0, 1), (1, 2), (-1, 2), (2, 3), (-2, 1), (0, 3), (3, 4), (-3, 4)]
BAD = [(4, 1), (-4, 1), (0, 0), (0, 5)]  # with max_charge 3, max_spin_multiplicity 4
TOL_E, TOL_F = 2e-4, 2e-3  # eV/atom, eV/A (float32 mlip.cpp vs float64 metatrain)


def structures(seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for name in ("H2O", "CH4", "NH3", "CH3CH2OH", "CH3NO2"):
        a = ase.build.molecule(name)
        a.positions += rng.normal(0, 0.05, a.positions.shape)
        out.append((name, a))
    # periodic: a few molecules in a small box (every atom with neighbours through
    # the boundary), and a skewed cell
    box = ase.Atoms()
    for k, name in enumerate(("H2O", "NH3", "CH4")):
        m = ase.build.molecule(name)
        m.positions += np.array([1.5 + 2.2 * k, 2.0 + 0.7 * k, 1.8 + 1.1 * k])
        box += m
    box.cell = [[6.5, 0, 0], [0.4, 6.2, 0], [-0.3, 0.5, 6.0]]
    box.pbc = True
    box.positions += rng.normal(0, 0.05, box.positions.shape)
    out.append(("box", box))
    for _, a in out:
        a.positions = a.positions.astype(np.float32).astype(np.float64)
        if a.pbc.any():
            a.cell = a.cell.array.astype(np.float32).astype(np.float64)
    return out


def write_frames(path, frames):
    """extxyz with charge= / spin_multiplicity= (what mlipcpp-eval reads)."""
    with open(path, "w") as f:
        for _, a, q, s in frames:
            lines = [str(len(a))]
            info = f"charge={q} spin_multiplicity={s}"
            if a.pbc.any():
                cell = " ".join(f"{x:.9g}" for x in a.cell.array.reshape(-1))
                info = f'Lattice="{cell}" pbc="T T T" ' + info
            lines.append(info)
            for sym, p in zip(a.get_chemical_symbols(), a.positions):
                lines.append(f"{sym} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}")
            f.write("\n".join(lines) + "\n")


def scalar_tmap(value, name):
    from metatensor.torch import Labels, TensorBlock, TensorMap
    return TensorMap(Labels.single(), [TensorBlock(
        values=torch.tensor([[float(value)]], dtype=torch.float64),
        samples=Labels("system", torch.tensor([[0]])), components=[],
        properties=Labels(name, torch.tensor([[0]])))])


def metatrain_eval(model, a, q, s, symmetric=True):
    """(energy, forces, stress) of metatrain's model in float64 by autograd; the
    stress as 1/V dE/d(strain), of a symmetric strain (PET, as metatrain / mlip.cpp's
    pet-graph), or (symmetric=False) of a general one, its upper triangle (Plato, as
    scripts/plato/plato_reference.py: the frames make the model only approximately
    rotation invariant, so dE/d(strain) is not symmetric)."""
    from metatomic.torch import ModelOutput, System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    pos = torch.tensor(a.positions, dtype=torch.float64, requires_grad=True)
    periodic = bool(a.pbc.any())
    cell = torch.tensor(a.cell.array if periodic else np.zeros((3, 3)), dtype=torch.float64)
    strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    sym = 0.5 * (strain + strain.T) if symmetric else strain
    system = System(types=torch.tensor(a.numbers, dtype=torch.int32), positions=pos + pos @ sym,
                    cell=cell + cell @ sym, pbc=torch.tensor([periodic] * 3))
    system.add_data("charge", scalar_tmap(q, "charge"))
    system.add_data("spin_multiplicity", scalar_tmap(s, "spin_multiplicity"))
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    out = model([system], {"energy": ModelOutput(sample_kind="system")})
    e = out["energy"].block().values.sum()
    gp, gs = torch.autograd.grad(e, [pos, strain])
    stress = None
    if periodic:
        v = abs(np.linalg.det(a.cell.array))
        st = gs.detach().numpy() / v
        stress = [st[0, 0], st[1, 1], st[2, 2], st[1, 2], st[0, 2], st[0, 1]]
    return float(e), -gp.detach().numpy(), stress


def metatomic_eval(path, a, q, s):
    from metatomic_ase import MetatomicCalculator
    b = a.copy()
    b.info["charge"], b.info["spin_multiplicity"] = q, s
    b.calc = MetatomicCalculator(path, device="cpu")
    st = b.get_stress(voigt=True) if b.pbc.any() else None  # ase: 1/V dE/d(strain), xx yy zz yz xz xy
    return float(b.get_potential_energy()), b.get_forces(), None if st is None else list(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--eval", required=True, help="mlipcpp-eval binary")
    ap.add_argument("--mta", help="exported metatomic model (.pt)")
    ap.add_argument("--ckpt", help="another checkpoint that must match --model bit for bit")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--arch", choices=["pet", "plato"], default="pet")
    ap.add_argument("--out", default="parity-pet")
    ap.add_argument("--json", help="write the table here")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.arch == "plato":  # plato_pet_lr with scripts/plato/plato_conditioning.py
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plato"))
        from plato_conditioning import load_plato_model as load
    else:
        load = load_local_model
    model = load(args.model).to(torch.float64).eval()
    rt = load(args.ckpt).to(torch.float64).eval() if args.ckpt else None
    frames = [(n, a, q, s) for n, a in structures() for q, s in PAIRS]
    bad = [(n, a, q, s) for n, a in structures()[:1] for q, s in BAD]
    write_frames(out / "frames.xyz", frames + bad)
    run = subprocess.run([args.eval, args.gguf, str(out / "frames.xyz"), str(out / "mlipcpp.json"),
                          "--backend", args.backend], capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stdout, run.stderr)
        raise SystemExit("mlipcpp-eval failed")
    got = json.load(open(out / "mlipcpp.json"))["frames"]

    rows, ok = [], True
    base = {}
    print(f"{'structure':10s} {'q':>3s} {'2S+1':>4s} {'E_mtt (eV)':>14s} {'E(q,s)-E(0,1)':>14s} "
          f"{'|dE| mlipcpp':>13s} {'max|dF|':>10s} {'max|dS|':>10s} {'|dE| mta':>10s} {'max|dF| mta':>11s}")
    for (name, a, q, s), g in zip(frames, got):
        e, f, st = metatrain_eval(model, a, q, s, symmetric=args.arch == "pet")
        if (q, s) == (0, 1):
            base[name] = e
        de = abs(g["energy"] - e)
        df = float(np.abs(np.array(g["forces"]).reshape(-1, 3) - f).max())
        ds = float(np.abs(np.array(g["stress"]) - np.array(st)).max()) if st is not None else float("nan")
        row = dict(structure=name, charge=q, spin=s, n_atoms=len(a), e_metatrain=e, e_shift=e - base[name],
                   de_mlipcpp=de, df_mlipcpp=df, ds_mlipcpp=ds, backend=args.backend)
        if args.mta:
            em, fm, sm = metatomic_eval(args.mta, a, q, s)
            row["de_metatomic"] = abs(em - e)
            row["df_metatomic"] = float(np.abs(fm - f).max())
            row["de_mlipcpp_vs_metatomic"] = abs(g["energy"] - em)
        if rt is not None:
            e2, f2, _ = metatrain_eval(rt, a, q, s, symmetric=args.arch == "pet")
            row["roundtrip_bitwise"] = bool(e2 == e and np.array_equal(f2, f))
            ok &= row["roundtrip_bitwise"]
        ok &= de / len(a) < TOL_E and df < TOL_F
        rows.append(row)
        print(f"{name:10s} {q:3d} {s:4d} {e:14.6f} {e - base[name]:14.6f} {de:13.2e} {df:10.2e} {ds:10.2e} "
              + (f"{row['de_metatomic']:10.2e} {row['df_metatomic']:11.2e}" if args.mta else "")
              + ("  rt=" + str(row["roundtrip_bitwise"]) if rt is not None else ""))

    # out of range: an error on both sides
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists  # noqa: F401
    for (name, a, q, s), g in zip(bad, got[len(frames):]):
        try:
            metatrain_eval(model, a, q, s)
            mtt = "no error"
        except Exception as exc:  # noqa: BLE001
            mtt = type(exc).__name__
        mlip = g.get("error", "no error")
        print(f"out of range q={q} 2S+1={s}: metatrain {mtt}; mlip.cpp: {mlip}")
        ok &= mtt != "no error" and "error" in g
        rows.append(dict(structure=name, charge=q, spin=s, metatrain_error=mtt, mlipcpp_error=mlip))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
