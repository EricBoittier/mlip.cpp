#!/usr/bin/env python3
"""Parity of split-charge equilibration (metatrain plato_lr sqe, float64; mlip.cpp pet.plato_lr.sqe, host float64 term
plus the float32 backbone) on the frag parity frames (Li+ PF6- through the 5-6 A charge-transfer switch and beyond,
Gdm+ Gdm+, Zn(NH3)4 2+ with Cl-, single units, frames without formal charges).

    # 1. frames, a spec (tabulated start, perturbed within the bounds with --perturb SEED) and the metatrain reference
    python parity_sqe.py write --model fa-base.pt --out DIR [--perturb 7] [--q0 formal] [--backbone-charge neutral]
    # 2. python add_sqe_keys.py fa-base.gguf DIR/sqe.gguf --spec DIR/spec.json --q0 ... --backbone-charge ...
    #    mlipcpp-eval DIR/sqe.gguf DIR/frames.xyz DIR/sqe-cpu.json [--backend cuda]
    # 3. python parity_sqe.py compare --out DIR --eval DIR/sqe-cpu.json [--json table.json]
Checks: |dE|/atom <= 2e-4 eV, |dF| <= 2e-3 eV/A (float32 backbone vs float64), |dq| <= 1e-5 e.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conditioning"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_frag_charges import frames, write_xyz, mt_eval  # noqa: E402

TOL_E, TOL_F, TOL_Q = 2e-4, 2e-3, 1e-5


def cmd_write(a):
    import torch
    from export_pytorch.export_pet_full import load_local_model
    from metatrain.pet.modules.sqe import PETSQE, spec_json
    torch.set_default_dtype(torch.float64)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = load_local_model(a.model).to(torch.float64).eval()
    pet = model.module if hasattr(model, "module") else model
    bk = pet.backend
    zs = [z for z, i in sorted(((z, int(i)) for z, i in enumerate(bk.species_to_species_index.tolist()) if int(i) >= 0),
                               key=lambda t: t[1])]
    bk.sqe = PETSQE(zs, None, a.q0, a.backbone_charge)
    if a.perturb is not None:
        g = torch.Generator().manual_seed(a.perturb)
        with torch.no_grad():
            for p in bk.sqe.params.parameters():
                p.add_(0.5 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    json.dump(spec_json(bk.sqe.params), open(out / "spec.json", "w"))
    fr = frames()
    write_xyz(out / "frames.xyz", fr)
    ref = []
    for name, Z, P, q, s, f in fr:
        e, F = mt_eval(model, Z, P, q, s, f)
        qs = pet._sqe_charges.tolist()
        ref.append({"name": name, "n": len(Z), "formal": f is not None, "e": e, "f": F.tolist(), "q": qs})
        print(f"{name:28s} n {len(Z):3d} E {e:.6f}  q[0] {qs[0]:+.4f}")
    json.dump(ref, open(out / "metatrain.json", "w"))


def cmd_compare(a):
    ref = json.load(open(Path(a.out) / "metatrain.json")); ev = json.load(open(a.eval))["frames"]
    ok, rows = True, []
    for r, f in zip(ref, ev):
        n = r["n"]
        if "error" in f:
            print(r["name"], "ERROR", f["error"]); ok = False; continue
        row = {"name": r["name"], "n": n, "dE_per_atom": abs(f["energy"] - r["e"]) / n,
               "dF": float(np.abs(np.array(f["forces"]).reshape(n, 3) - np.array(r["f"])).max()),
               "dq": float(np.abs(np.array(f["sqe_charges"]) - np.array(r["q"])).max())}
        row["pass"] = bool(row["dE_per_atom"] <= TOL_E and row["dF"] <= TOL_F and row["dq"] <= TOL_Q); ok &= row["pass"]
        rows.append(row)
        print(f"{r['name']:28s} |dE|/at {row['dE_per_atom']:.2e} |dF| {row['dF']:.2e} |dq| {row['dq']:.2e} {'PASS' if row['pass'] else 'FAIL'}")
    m = {k: max(r[k] for r in rows) for k in ("dE_per_atom", "dF", "dq")}
    print("max", m); print("ALL PASS" if ok else "FAILURES")
    if a.json:
        json.dump({"rows": rows, "max": m, "pass": ok}, open(a.json, "w"), indent=1)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("--model", required=True); w.add_argument("--out", required=True)
    w.add_argument("--perturb", type=int); w.add_argument("--q0", default="uniform")
    w.add_argument("--backbone-charge", default="input")
    c = sub.add_parser("compare"); c.add_argument("--out", required=True); c.add_argument("--eval", required=True)
    c.add_argument("--json")
    a = ap.parse_args()
    cmd_write(a) if a.cmd == "write" else cmd_compare(a)


if __name__ == "__main__":
    main()
