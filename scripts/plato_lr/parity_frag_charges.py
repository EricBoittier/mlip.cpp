#!/usr/bin/env python3
"""Parity of fragment-constrained charges (metatrain plato_lr frag_charges, mlip.cpp pet.plato_lr.frag_charges)
on charged multi-fragment structures: Li+ PF6- at several separations (through the 7.5-8 A Coulomb switch),
a like-charge pair (Gdm+ Gdm+), a Zn(NH3)4 2+ complex with Cl- counter ions, a single unit with formal charges,
and frames without formal charges (which must equal the base model).

    # 1. frames + metatrain reference (float64, autograd)
    python parity_frag_charges.py write --model fa-frag.pt --base fa-base.pt --out DIR
    # 2. mlipcpp-eval DIR/frames.xyz with the frag GGUF and the base GGUF (any backend, e.g. on a GPU node)
    mlipcpp-eval fa-frag.gguf DIR/frames.xyz DIR/frag-cpu.json; mlipcpp-eval fa-base.gguf DIR/frames.xyz DIR/base-cpu.json
    # 3. compare
    python parity_frag_charges.py compare --out DIR --frag DIR/frag-cpu.json --base DIR/base-cpu.json [--json table.json]

Frames carry a per-atom formal_charges array (on one atom per fragment: only the unit sums matter) or none.
Checks: frag GGUF vs metatrain frag (|dE|/atom <= 2e-4 eV, |dF| <= 2e-3 eV/A, float32 vs float64); frames without
formal charges: frag GGUF vs base GGUF (bitwise expected) and metatrain frag vs base (<= 1e-10).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conditioning"))

TOL_E, TOL_F = 2e-4, 2e-3
IDX = "/home/boittier/data/petlr-lrd/lrdall_index.json"


def pair_at(p0, dmin):
    """Fragment B moved along the COM axis so that the closest A-B distance is dmin."""
    Z = np.array(p0["Z"]); P0 = np.array(p0["pos"]); nA = p0["nA"]
    axis = P0[nA:].mean(0) - P0[:nA].mean(0); axis /= np.linalg.norm(axis)
    lo, hi = -20.0, 40.0
    for _ in range(80):
        m = 0.5 * (lo + hi); P = P0.copy(); P[nA:] += m * axis
        d = np.linalg.norm(P[:nA, None] - P[None, nA:], axis=-1).min()
        lo, hi = (m, hi) if d < dmin else (lo, m)
    P = P0.copy(); P[nA:] += hi * axis
    return Z, P, nA


def zn_complex(cl):
    """Zn(NH3)4 2+ (tetrahedral, Zn-N 2.10 A) and Cl- at the given distances from Zn: the first towards the centre
    of the face opposite an NH3 (bonded to Zn below 1.3 x (1.22 + 1.02) = 2.91 A), the second beyond that NH3."""
    import ase.build
    d = 2.10
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    Z, P = [30], [[0.0, 0.0, 0.0]]
    for v in t:
        nh3 = ase.build.molecule("NH3")
        n = [i for i, s in enumerate(nh3.get_chemical_symbols()) if s == "N"][0]
        pos = nh3.positions - nh3.positions[n]
        # point the lone pair (opposite the H centroid) at Zn
        h = pos[[i for i in range(4) if i != n]].mean(0); h /= np.linalg.norm(h)
        R = rot_from_to(-h, -v)
        pos = pos @ R.T + d * v
        Z += list(nh3.numbers); P += pos.tolist()
    dirs = [np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])]  # a face centre, then beyond an NH3
    for k, r in enumerate(cl):
        u = dirs[k] / np.linalg.norm(dirs[k]); Z.append(17); P.append((r * u).tolist())
    form = np.zeros(len(Z)); form[0] = 2.0
    form[len(Z) - len(cl):] = -1.0
    return np.array(Z), np.array(P), form


def rot_from_to(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def frames():
    idx = json.load(open(IDX))
    pick = lambda name: sorted([x for x in idx if x["pair"] == name and x["orient"] == 0], key=lambda x: x["dmin"])[0]
    li, gdm = pick("Li_PF6"), pick("Gdm_Gdm")
    out = []  # (name, Z, P, charge, spin, formal or None)
    for dm in (3.0, 7.6, 7.8, 8.0, 10.0, 15.0):
        Z, P, nA = pair_at(li, dm); f = np.zeros(len(Z)); f[0] = li["qA"]; f[nA] = li["qB"]
        out.append((f"LiPF6_d{dm:g}", Z, P, 0, 1, f))
    Z, P, nA = pair_at(li, 8.0); out.append(("LiPF6_d8_noformal", Z, P, 0, 1, None))
    for dm in (4.0, 7.8, 12.0):
        Z, P, nA = pair_at(gdm, dm); f = np.zeros(len(Z)); f[0] = 1; f[nA] = 1
        out.append((f"GdmGdm_d{dm:g}", Z, P, 2, 1, f))
    Z, P, nA = pair_at(gdm, 12.0); out.append(("GdmGdm_d12_noformal", Z, P, 2, 1, None))
    out.append(("Gdm_single_formal", Z[:nA], P[:nA], 1, 1, np.r_[1.0, np.zeros(nA - 1)]))
    for cl in ((3.5, 9.0), (3.5, 10.6), (2.4, 9.0)):
        Z, P, f = zn_complex(cl); out.append((f"ZnNH3_4_Cl2_{cl[0]:g}_{cl[1]:g}", Z, P, 0, 1, f))
    Z, P, f = zn_complex((8.0,)); out.append(("ZnNH3_4_Cl_8", Z, P, 1, 2, f))
    Z, P, f = zn_complex((8.0,)); out.append(("ZnNH3_4_Cl_8_noformal", Z, P, 1, 2, None))
    res = []
    for name, Z, P, q, s, f in out:
        P = np.asarray(P, dtype=np.float32).astype(np.float64)  # what mlipcpp-eval reads
        res.append((name, np.asarray(Z), P, q, s, f))
    return res


def write_xyz(path, fr):
    with open(path, "w") as fh:
        from ase.data import chemical_symbols
        for name, Z, P, q, s, f in fr:
            props = "species:S:1:pos:R:3" + (":formal_charges:R:1" if f is not None else "")
            fh.write(f"{len(Z)}\nProperties={props} charge={q} spin_multiplicity={s} name={name} pbc=\"F F F\"\n")
            for k in range(len(Z)):
                line = f"{chemical_symbols[Z[k]]} {P[k, 0]:.9g} {P[k, 1]:.9g} {P[k, 2]:.9g}"
                if f is not None:
                    line += f" {f[k]:g}"
                fh.write(line + "\n")


def mt_eval(model, Z, P, q, s, formal):
    import torch
    from metatomic.torch import ModelOutput, System
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
    from parity_pet import scalar_tmap
    pos = torch.tensor(P, dtype=torch.float64, requires_grad=True)
    sy = System(types=torch.tensor(Z, dtype=torch.int32), positions=pos, cell=torch.zeros(3, 3, dtype=torch.float64),
                pbc=torch.tensor([False] * 3))
    sy.add_data("charge", scalar_tmap(q, "charge")); sy.add_data("spin_multiplicity", scalar_tmap(s, "spin_multiplicity"))
    if formal is not None:
        n = len(Z)
        sy.add_data("mtt::formal_charges", TensorMap(Labels.single(), [TensorBlock(
            values=torch.tensor(formal, dtype=torch.float64).reshape(n, 1),
            samples=Labels(["system", "atom"], torch.tensor([[0, i] for i in range(n)])), components=[],
            properties=Labels("formal_charges", torch.tensor([[0]])))]))
    sy = get_system_with_neighbor_lists(sy, model.requested_neighbor_lists())
    e = model([sy], {"energy": ModelOutput(sample_kind="system")})["energy"].block().values.sum()
    f = -torch.autograd.grad(e, pos)[0]
    return float(e), f.detach().numpy()


def frag_term(Z, P, formal, R=8.0, t=0.5, sigma=4.0):
    """The beyond-R Coulomb energy (eV) with metatrain's PETFragCharges (units + energy), float64."""
    import torch
    from metatrain.pet.modules.plato_lr import PETFragCharges
    m = PETFragCharges(R=R, t=t, sigma=sigma)
    pos = torch.tensor(P, dtype=torch.float64)
    lab, Qu, nu, C = m.units(torch.tensor(Z, dtype=torch.long), pos, torch.tensor(formal), torch.tensor(0.0))
    _, qa = m.charges(lab, Qu, nu, C)
    return float(m.energy(qa, pos).sum()), lab.tolist(), Qu.tolist()


def cmd_write(a):
    import torch
    from export_pytorch.export_pet_full import load_local_model
    torch.set_default_dtype(torch.float64)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fr = frames()
    write_xyz(out / "frames.xyz", fr)
    frag = load_local_model(a.model).to(torch.float64).eval()
    base = load_local_model(a.base).to(torch.float64).eval()
    ref = []
    for name, Z, P, q, s, f in fr:
        ef, ff = mt_eval(frag, Z, P, q, s, f)
        eb, fb = mt_eval(base, Z, P, q, s, None)
        r = {"name": name, "n": len(Z), "charge": q, "formal": f is not None, "e_frag": ef, "f_frag": ff.tolist(),
             "e_base": eb, "f_base": fb.tolist()}
        if f is not None:
            r["e_term"], r["units"], r["Qu"] = frag_term(Z, P, f)
        ref.append(r)
        print(f"{name:28s} n {len(Z):3d} E_frag {ef:.6f} E_base {eb:.6f} term {r.get('e_term', 0):.6f} units {r.get('Qu')}")
    json.dump(ref, open(out / "metatrain.json", "w"))


def cmd_compare(a):
    out = Path(a.out)
    ref = json.load(open(out / "metatrain.json"))
    fj = json.load(open(a.frag))["frames"]; bj = json.load(open(a.base))["frames"]
    rows, ok = [], True
    for r, f, b in zip(ref, fj, bj):
        n = r["n"]
        if "error" in f or "error" in b:
            print(r["name"], "ERROR", f.get("error"), b.get("error")); ok = False; continue
        Ff, Fb = np.array(f["forces"]).reshape(n, 3), np.array(b["forces"]).reshape(n, 3)
        row = {"name": r["name"], "n": n, "formal": r["formal"],
               "dE_mt_per_atom": abs(f["energy"] - r["e_frag"]) / n,
               "dF_mt": float(np.abs(Ff - np.array(r["f_frag"])).max()),
               "E_term_mt": r.get("e_term", 0.0),
               "dE_frag_vs_base_mlipcpp": f["energy"] - b["energy"],
               "dF_frag_vs_base_mlipcpp": float(np.abs(Ff - Fb).max()),
               "dE_frag_vs_base_mt": r["e_frag"] - r["e_base"]}
        good = row["dE_mt_per_atom"] <= TOL_E and row["dF_mt"] <= TOL_F
        if not r["formal"]:
            good &= row["dE_frag_vs_base_mlipcpp"] == 0 and row["dF_frag_vs_base_mlipcpp"] == 0
            good &= abs(row["dE_frag_vs_base_mt"]) <= 1e-10
        row["pass"] = bool(good); ok &= good
        rows.append(row)
        print(f"{r['name']:28s} {'formal' if r['formal'] else 'none  '} |dE|/at {row['dE_mt_per_atom']:.2e} |dF| {row['dF_mt']:.2e}"
              f"  frag-base: mlipcpp {row['dE_frag_vs_base_mlipcpp']:+.6f} eV ({row['dF_frag_vs_base_mlipcpp']:.2e}),"
              f" metatrain {row['dE_frag_vs_base_mt']:+.6f}  term {row['E_term_mt']:+.6f}  {'PASS' if good else 'FAIL'}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("--model", required=True); w.add_argument("--base", required=True)
    w.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--out", required=True); c.add_argument("--frag", required=True)
    c.add_argument("--base", required=True); c.add_argument("--json")
    a = ap.parse_args()
    cmd_write(a) if a.cmd == "write" else cmd_compare(a)


if __name__ == "__main__":
    main()
