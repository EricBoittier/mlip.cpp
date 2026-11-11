#!/usr/bin/env python3
"""Full PETLR parity (after M. Langer, ipsum/PET-LR): mlip.cpp (pet.iris_lr GGUF, mlipcpp-eval JSON) and each of his
fp32 runs, all against his fp64 (`ref_petlr.py --x64`), per output: energy (|dE| / sum|E_i|), forces
(max|dF| / max|F|), virial (max|dW| / max|W|, periodic 3-D), charges (max|dq|), mu (|dmu|), dipole (max|dD| / max|D|), metallicity (|d|). Max over frames,
the floor verdict against the max of his runs, and the paired per-frame P(mlip.cpp worse than his run) with its
z score (n frames, 0.5 = same distribution).

    python lr_parity.py CPP.json REF64.npz RUN32.npz [RUN32.npz ...] [--label NAME]
"""
import argparse
import json

import numpy as np

KEYS = ("E", "F", "W", "q", "q_rel", "mu", "dipole", "dipole_abs", "metallicity", "qeq_residual", "charge_residual",
        "dipole_flow")


def per_frame(get, z):
    ap = z["atom_ptr"]
    out = {k: [] for k in KEYS}
    for f in range(len(ap) - 1):
        A = slice(ap[f], ap[f + 1])
        g = get(f, A)
        if g is None:
            for k in KEYS:
                out[k].append(np.nan)
            continue
        e = z["atom/energy"][A]
        out["E"].append(abs(g["E"] - e.sum()) / np.abs(e).sum())
        out["F"].append(np.abs(g["F"] - z["forces"][A]).max() / np.abs(z["forces"][A]).max() if g["F"].size else np.nan)
        if "virial" in z.files and np.all(np.isfinite(z["virial"][f])) and g.get("W") is not None:
            out["W"].append(np.abs(g["W"] - z["virial"][f]).max() / np.abs(z["virial"][f]).max())
        periodic = "virial" in z.files and np.all(np.isfinite(z["virial"][f]))
        if "atom/charges" in z.files:
            out["q"].append(np.abs(g["q"] - z["atom/charges"][A]).max())
            out["q_rel"].append(out["q"][-1] / np.abs(z["atom/charges"][A]).max())
            D = z["struct/dipole"][f]  # (no dipole for a periodic cell: his calculator reports none)
            out["dipole"].append(np.nan if periodic else np.abs(g["dipole"] - D).max() / max(np.abs(D).max(), 1e-12))
            out["dipole_abs"].append(np.nan if periodic else np.abs(g["dipole"] - D).max())
        if "int/fermi_level" in z.files:
            out["mu"].append(abs(g["mu"] - z["int/fermi_level"][A][0]))
        if "struct/metallicity" in z.files:  # (ref_branch.py refs: metallicity without the Fermi level)
            out["metallicity"].append(abs(g["metallicity"] - z["struct/metallicity"][f]))
        if "struct/qeq_residual" in z.files:  # QEq: his |P (A q + chi)| (eV); mlip.cpp reports it as `metallicity`
            out["qeq_residual"].append(abs(g["qeq_residual"] - z["struct/qeq_residual"][f]))
        if "struct/dipole_flow" in z.files:  # conserving charges: the flow dipole [S, 3] (e A), max|d| / max|D|
            D = z["struct/dipole_flow"][f]
            out["dipole_flow"].append(np.abs(np.asarray(g["dipole_flow"]) - D).max() / max(np.abs(D).max(), 1e-12))
        if "struct/charge_residual" in z.files:  # threshold reference: his charge_residual [S] (signed, e)
            out["charge_residual"].append(abs(g["charge_residual"] - z["struct/charge_residual"][f]))
    return {k: np.asarray(v, float) for k, v in out.items() if len(v)}


def from_cpp(path, z):
    fr = json.load(open(path))["frames"]

    def get(f, A):
        if f >= len(fr):  # a shorter run (first frames only)
            return None
        x = fr[f]
        if "error" in x:
            return None
        return dict(E=x["energy"], F=np.asarray(x["forces"]).reshape(-1, 3), q=np.asarray(x.get("charges", [])),
                    W=np.asarray(x["virial"]).reshape(3, 3) if "virial" in x else None,
                    dipole=np.asarray(x.get("dipole", [np.nan] * 3)), mu=x.get("fermi_level", np.nan),
                    metallicity=x.get("metallicity", np.nan), qeq_residual=x.get("metallicity", np.nan),
                    charge_residual=x.get("charge_residual", np.nan), dipole_flow=x.get("dipole_flow", [np.nan] * 3))
    return per_frame(get, z)


def from_ref(path, z):
    r = np.load(path)

    def get(f, A):
        g = dict(E=r["atom/energy"][A].sum(), F=r["forces"][A], W=r["virial"][f] if "virial" in r.files else None)
        if "atom/charges" in r.files:
            g.update(q=r["atom/charges"][A], dipole=r["struct/dipole"][f])
        if "int/fermi_level" in r.files:
            g.update(mu=r["int/fermi_level"][A][0])
        if "struct/metallicity" in r.files:
            g.update(metallicity=r["struct/metallicity"][f])
        if "struct/qeq_residual" in r.files:
            g.update(qeq_residual=r["struct/qeq_residual"][f])
        if "struct/charge_residual" in r.files:
            g.update(charge_residual=r["struct/charge_residual"][f])
        if "struct/dipole_flow" in r.files:
            g.update(dipole_flow=r["struct/dipole_flow"][f])
        return g
    return per_frame(get, z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpp"); ap.add_argument("ref64"); ap.add_argument("runs", nargs="*")
    ap.add_argument("--label", default="")
    ap.add_argument("--overflow", nargs=3, metavar=("FRAMES", "CUTOFF", "MAX_NEIGHBORS"),
                    help="skip frames where an atom has more than MAX_NEIGHBORS neighbours within CUTOFF: his "
                         "calculator raises there (Calculator.calculate, overflow), his oracle (predict on the "
                         "calculator's batch) silently keeps the nearest MAX_NEIGHBORS; hard-cutoff models in dense cells")
    a = ap.parse_args()
    z = np.load(a.ref64)
    c = from_cpp(a.cpp, z)
    runs = {p.split("/")[-1].replace(".npz", ""): from_ref(p, z) for p in a.runs}
    if a.overflow:
        from ase.io import read
        from ase.neighborlist import neighbor_list
        fr = read(a.overflow[0], ":")
        over = [f for f in range(len(c["E"])) if f < len(fr) and
                np.bincount(neighbor_list("i", fr[f], float(a.overflow[1])), minlength=len(fr[f])).max()
                > int(a.overflow[2])]
        for k in c:
            c[k][over] = np.nan
        print(f"  (skipped {len(over)} overflow frames: {over})")
    for r in runs.values():  # the same frames for all (a shorter mlip.cpp run: his maxima over those frames only)
        for k in r:
            if k in c and len(c[k]) == len(r[k]):
                r[k] = np.where(np.isnan(c["E"]), np.nan, r[k])
    nerr = int(np.isnan(c["E"]).sum())
    print(f"== {a.label or a.cpp}: {len(c['E'])} frames ({nerr} failed); max over frames vs JAX fp64")
    print(f"  {'':22s} " + " ".join(f"{k:>11s}" for k in KEYS if k in c))
    print(f"  {'mlip.cpp':22s} " + " ".join(f"{np.nanmax(c[k]):11.2e}" for k in KEYS if k in c))
    for n, r in runs.items():
        print(f"  {'JAX ' + n:22s} " + " ".join(f"{np.nanmax(r[k]):11.2e}" if k in r else f"{'-':>11s}"
                                             for k in KEYS if k in c))
    for k in KEYS:
        if k not in c or np.all(np.isnan(c[k])):  # (e.g. the dipole on periodic cells: none reported)
            continue
        his = [np.nanmax(r[k]) for r in runs.values() if k in r and not np.all(np.isnan(r[k]))]
        if not his:
            continue
        mx = np.nanmax(c[k])
        verdict = "within" if mx <= max(his) else "ABOVE"
        ps = []
        for r in runs.values():
            ok = ~np.isnan(c[k]) & ~np.isnan(r[k])
            n = ok.sum()
            p = float((c[k][ok] > r[k][ok]).mean())
            ps.append(f"P {p:.2f} (z {(p - 0.5) / (0.5 / np.sqrt(n)):+.1f})")
        print(f"  {k:12s} his max {max(his):.2e}; mlip.cpp {mx:.2e} -> {verdict} ({mx / max(his):.2f}x); "
              f"median {np.nanmedian(c[k]):.2e} vs {np.nanmedian(list(runs.values())[0][k]):.2e}; " + "; ".join(ps))


if __name__ == "__main__":
    main()
