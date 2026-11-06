#!/usr/bin/env python3
"""Per-checkpoint normalised tail error monitor (ERM-with-priors frame, ~/data/lr-program/ERM_priors.md).

eps = (dE_model - dE_DFT) / dE_ref, dE = E_int(r) - E_int(r_far of the scan) (offset-free), dE_ref signed: exact
frozen-density electrostatics + induction where available, else the multipole + induction expansion; points with
|dE_ref| < 0.1 meV are dropped (scale undefined). eps = 0 exact, -1 a missing tail, flat in r when the functional form is
right (g lambda^2 - 1). Same definition and r bins as ~/data/figures/scripts/normalised.py.

Sets
  dimers  longrange-dimers valid scans (1,254 points x 3 frames AB, A, B: mlipcpp-eval on dim_all.xyz); classes ion pair,
          like charge, ion + polar, ion + nonpolar, neutral; d_min >= 8 A, bins 8-11 / 11-16 / 16-26 A
  pulled  merge-aug set A val (prep_tail.py: <data>/val.xyz + val/points.jsonl); d_far = 25 A; classes ion_pair,
          like_charge, ion_neutral; bins 4-7 (multipole reference, contact), 8-11, 11-16, 16-26 A

Per checkpoint x set x class: median eps per r bin (IQR, 95% bootstrap CI over scans), and over all points at r >= 8 A the
Theil-Sen slopes of eps vs r_COM (deps/dr, 1/A) and of ln|eps| vs ln r_COM (k in |eps| ~ r^k), bootstrap CIs over scans.

Charges (when the evals dump latent_charges): lam = local ion charge / formal (sum over the ion's atoms, after the
total-charge shift) and qN/qI = the neutral partner's net latent charge over the ion's formal charge (ion-neutral classes;
the shift q_i - mean(q) + Q/N puts (n_N/N)(1 - lam)Q on it, a spurious 1/R tail); medians at r >= 8 A with CIs.

Flag rule (stop/flag, evaluated along the checkpoint order given):
  drift  |median eps(r >= 8)| of a class grows for two checkpoints in a row (|m_t| > |m_t-1| > |m_t-2|, each step by more
         than --tol), i.e. eps moves away from 0;
  slope  for ion-neutral classes (ion + polar, ion + nonpolar, ion_neutral), |deps/dr| grows for two checkpoints in a row.
Flags go into the jsonl rows ("flags") and a summary row; --stop-exit returns 3 when any flag is raised.

    eps_monitor.py --out eps.jsonl [--eval-bin mlipcpp-eval --workdir W] NAME=GGUF ...       (evaluates missing jsons)
    eps_monitor.py --out eps.jsonl NAME=dimers.json[,pulled.json] ...                       (scores existing evals)
    NAME=A.json+B.json+...  averages the energies of several checkpoints (ensembles; normalised.py's 4-seed eps)"""
import argparse, json, os, subprocess, sys, collections, numpy as np
from scipy.stats import theilslopes

H = os.path.expanduser("~")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("ckpts", nargs="+", help="NAME=GGUF, NAME=dimers.json[,pulled.json], or NAME=a.json+b.json (ensemble)")
ap.add_argument("--out", required=True)
ap.add_argument("--dimers-jsonl", default=f"{H}/Documents/metawork-longrange-dimers/longrange-dimers/data/hf/dimers/dimers_valid.jsonl")
ap.add_argument("--dimers-index", default=f"{H}/data/figures/scripts/dim_all_index.json")
ap.add_argument("--dimers-xyz", default=f"{H}/data/figures/scripts/dim_all.xyz")
ap.add_argument("--pulled", default=f"{H}/data/petlr-tail/data", help="prep_tail.py output (val.xyz, val/points.jsonl)")
ap.add_argument("--eval-bin", default=None)
ap.add_argument("--eval-args", default="--backend cuda")
ap.add_argument("--workdir", default=".")
ap.add_argument("--boot", type=int, default=200)
ap.add_argument("--tol", type=float, default=0.02, help="minimum change counted as growth (eps units; 1/A for slopes)")
ap.add_argument("--stop-exit", action="store_true")
ap.add_argument("--les-gate", type=float, default=None, help="held LES gate g: also report the identity g lam^2 - 1")
a = ap.parse_args()
RNG = np.random.default_rng(5)
A0 = 0.529177

# ---- references -------------------------------------------------------------------------------------------------
GRP = {"ion_pair": "ion pair", "same_sign": "like charge", "ion_polar": "ion + polar", "ion_nonpolar": "ion + nonpolar",
       "neutral_polar_control": "neutral", "neutral_nonpolar_control": "neutral"}
ION_NEUTRAL = {"ion + polar", "ion + nonpolar", "ion_neutral"}


def dimer_points():
    hf = [json.loads(l) for l in open(a.dimers_jsonl)]
    ref = np.array([r["E_int"] for r in hf]); cls = np.array([r["cls"] for r in hf]); dmin = np.array([r["dmin"] for r in hf])
    scan = np.array([f'{r["pair"]}|{r["orient"]}' for r in hf]); rcom = np.array([r["r_com"] for r in hf])
    emp = np.array([np.nan if r.get("E_mp") is None else r["E_mp"] for r in hf])
    eex = np.array([np.nan if r.get("E_es_exact") is None else r["E_es_exact"] + (r.get("E_mp_ind") or 0.0) for r in hf])
    global DIMER_FRAG
    DIMER_FRAG = [(int(r["n_A"]), r["qA_formal"], r["qB_formal"]) for r in hf]
    j8 = np.where((dmin >= 8) & np.isin(cls, list(GRP)))[0]
    last = {s: j8[scan[j8] == s][np.argmax(dmin[j8][scan[j8] == s])] for s in set(scan[j8])}
    P = []
    for k in j8:
        l = last[scan[k]]
        if k == l: continue
        dref = (eex[k] - eex[l]) if np.isfinite(eex[k]) and np.isfinite(eex[l]) else (emp[k] - emp[l])
        if not np.isfinite(dref) or abs(dref) < 1e-4: continue
        P.append(dict(k=int(k), l=int(l), cls=GRP[cls[k]], d=float(dmin[k]), r=float(rcom[k]), scan=scan[k], ddft=ref[k] - ref[l], dref=dref))
    return P


def pulled_points():
    pts = [json.loads(l) for l in open(f"{a.pulled}/val/points.jsonl")]
    by = collections.defaultdict(list)
    for p in pts: by[p["group"]].append(p)
    P = []
    for g, ps in by.items():
        far = [p for p in ps if abs(p["d"] - 25) < 1e-6]
        if not far: continue
        f = far[0]
        for p in ps:
            if p is f: continue
            ex = p["E_ref_exact"] is not None and f["E_ref_exact"] is not None
            dref = p["E_ref_exact"] - f["E_ref_exact"] if ex else (p["E_mp"] - f["E_mp"] if p["E_mp"] is not None else np.nan)
            if not np.isfinite(dref) or abs(dref) < 1e-4: continue
            P.append(dict(k=p["k"], l=f["k"], cls=p["cls"], d=p["d"], r=p["r_com"], scan=g, ddft=p["E"] - f["E"], dref=dref,
                          ref="exact" if ex else "multipole", frag=p.get("frag"), QU=p["QU"], QR=p["QR"]))
    return P


SETS = {"dimers": (dimer_points, [(8, 11), (11, 16), (16, 26)]), "pulled": (pulled_points, [(4, 7), (8, 11), (11, 16), (16, 26)])}

# ---- model energies ---------------------------------------------------------------------------------------------


def charge_stats(paths, kind, P):
    """Latent charges (after the total-charge shift, as mlipcpp-eval dumps them) per point: lam = sum_ion q / Q_ion (local ion
    charge; ion pairs: both ions; ion-neutral: the ion) and qN = sum_neutral q / Q_ion (the neutral partner's spurious
    monopole, ion-neutral classes). None where the eval has no latent charges."""
    fr = [json.load(open(p))["frames"] for p in paths.split("+")]
    if "latent_charges" not in fr[0][0]: return None
    out = []
    for p in P:
        if kind == "dimers":
            nA, qA, qB = DIMER_FRAG[p["k"]]; q = np.mean([np.array(f[3 * p["k"]]["latent_charges"])[:, 0] if np.ndim(f[3 * p["k"]]["latent_charges"]) == 2
                                                        else np.array(f[3 * p["k"]]["latent_charges"]) for f in fr], axis=0)
            m = np.arange(len(q)) < nA; QA, QB = qA, qB
        else:
            q = np.mean([np.array(f[p["k"]]["latent_charges"]).reshape(len(p["frag"]), -1)[:, 0] for f in fr], axis=0)
            m = np.array(p["frag"]) == 1; QA, QB = p["QU"], p["QR"]
        sA, sB = q[m].sum(), q[~m].sum()
        lam = [s_ / Q for s_, Q in ((sA, QA), (sB, QB)) if Q != 0]
        ion = (sA, QA, sB) if QA != 0 else (sB, QB, sA)
        qN = ion[2] / ion[1] if (QA == 0) != (QB == 0) else np.nan
        out.append((float(np.mean(lam)) if lam else np.nan, float(qN)))
    return np.array(out)


def energies(path, kind):
    fr = json.load(open(path))["frames"]; e = np.array([f["energy"] for f in fr])
    return e.reshape(-1, 3)[:, 0] - e.reshape(-1, 3)[:, 1] - e.reshape(-1, 3)[:, 2] if kind == "dimers" else e


def evaluate(name, gguf):
    out = {}
    for kind, xyz in (("dimers", a.dimers_xyz), ("pulled", f"{a.pulled}/val.xyz")):
        js = os.path.join(a.workdir, f"{name}.{kind}.json")
        if not os.path.exists(js):
            if not a.eval_bin: sys.exit(f"{js} missing and no --eval-bin")
            subprocess.run([a.eval_bin, gguf, xyz, js] + a.eval_args.split(), check=True, stdout=subprocess.DEVNULL)
        out[kind] = js
    return out


def parse(spec):
    name, v = spec.split("=", 1)
    if v.endswith(".gguf") or ".gguf." in v or v.endswith("gguf"): return name, evaluate(name, v)
    parts = v.split(","); d = {"dimers": parts[0]}
    if len(parts) > 1 and parts[1]: d["pulled"] = parts[1]
    return name, d


def model_e(paths, kind):  # "a.json+b.json": the mean of the members' energies
    return np.mean([energies(p, kind) for p in paths.split("+")], axis=0)

# ---- statistics -------------------------------------------------------------------------------------------------


def boot(scans, f, n):
    """f over resamples of scans (cluster bootstrap); 2.5/97.5 percentiles."""
    us = sorted(set(scans)); idx = collections.defaultdict(list)
    for i, s in enumerate(scans): idx[s].append(i)
    vals = []
    for _ in range(n):
        take = np.concatenate([idx[s] for s in RNG.choice(us, len(us))])
        v = f(take)
        if np.isfinite(v): vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def slopes(r, eps, take=None):
    take = np.arange(len(r)) if take is None else take
    rr, ee = r[take], eps[take]
    lin = theilslopes(ee, rr)[0] if len(set(rr)) > 1 else np.nan
    m = np.abs(ee) > 1e-3
    pw = theilslopes(np.log(np.abs(ee[m])), np.log(rr[m]))[0] if m.sum() > 2 and len(set(rr[m])) > 1 else np.nan
    return lin, pw


def score(name, paths):
    rows = []
    for kind, (pf, bins) in SETS.items():
        if kind not in paths: continue
        P = POINTS[kind]; E = model_e(paths[kind], kind)
        eps = np.array([((E[p["k"]] - E[p["l"]]) - p["ddft"]) / p["dref"] for p in P])
        cls = np.array([p["cls"] for p in P], dtype=object); d = np.array([p["d"] for p in P]); r = np.array([p["r"] for p in P])
        scans = np.array([p["scan"] for p in P])
        Cq = charge_stats(paths[kind], kind, P)
        for c in sorted(set(cls)):
            row = dict(ckpt=name, set=kind, cls=c, bins=[])
            for lo, hi in bins:
                m = np.where((cls == c) & (d >= lo) & (d < hi))[0]
                if not len(m): continue
                e = eps[m]
                row["bins"].append(dict(rbin=f"{lo}-{hi}", n=int(len(m)), eps_median=float(np.median(e)),
                                        eps_q25=float(np.percentile(e, 25)), eps_q75=float(np.percentile(e, 75)),
                                        ci95=boot(scans[m], lambda t: np.median(eps[m][t]), a.boot)))
            m = np.where((cls == c) & (d >= 8))[0]
            if len(m):
                lin, pw = slopes(r, eps, m)
                row.update(n=int(len(m)), eps_median=float(np.median(eps[m])), ci95=boot(scans[m], lambda t: np.median(eps[m][t]), a.boot),
                           deps_dr=float(lin), deps_dr_ci95=boot(scans[m], lambda t: slopes(r[m], eps[m], t)[0], a.boot),
                           k_power=float(pw), k_power_ci95=boot(scans[m], lambda t: slopes(r[m], eps[m], t)[1], a.boot))
                if Cq is not None:
                    lam, qN = Cq[m, 0], Cq[m, 1]
                    if np.isfinite(lam).any():
                        row["lam_median"] = float(np.nanmedian(lam)); row["lam_ci95"] = boot(scans[m], lambda t: np.nanmedian(lam[t]), a.boot)
                        if a.les_gate is not None: row["identity_g_lam2_minus_1"] = float(a.les_gate * row["lam_median"] ** 2 - 1)
                    if np.isfinite(qN).any():
                        row["qN_over_qion_median"] = float(np.nanmedian(qN)); row["qN_ci95"] = boot(scans[m], lambda t: np.nanmedian(qN[t]), a.boot)
            rows.append(row)
    return rows


def flags(history, tol):
    """history: list of row lists in checkpoint order -> flags per (set, cls) at each checkpoint."""
    out = []
    series = collections.defaultdict(list)
    for t, rows in enumerate(history):
        fl = []
        for row in rows:
            if "eps_median" not in row: continue
            key = (row["set"], row["cls"]); series[key].append(row)
            s = series[key]
            if len(s) >= 3:
                m = [abs(x["eps_median"]) for x in s[-3:]]
                if m[2] > m[1] + tol and m[1] > m[0] + tol: fl.append(dict(rule="drift", set=key[0], cls=key[1], abs_eps=m))
                if key[1] in ION_NEUTRAL:
                    g = [abs(x["deps_dr"]) for x in s[-3:]]
                    if g[2] > g[1] + tol / 10 and g[1] > g[0] + tol / 10: fl.append(dict(rule="slope", set=key[0], cls=key[1], abs_deps_dr=g))
        out.append(fl)
    return out


POINTS = {k: f() for k, (f, _) in SETS.items() if k == "dimers" or os.path.exists(f"{a.pulled}/val/points.jsonl")}
history, names = [], []
for spec in a.ckpts:
    name, paths = parse(spec); names.append(name); history.append(score(name, paths))
fl = flags(history, a.tol)
with open(a.out, "w") as f:
    for name, rows, fs in zip(names, history, fl):
        for row in rows: f.write(json.dumps(dict(row, flags=[x for x in fs if (x["set"], x["cls"]) == (row["set"], row["cls"])])) + "\n")
    f.write(json.dumps(dict(summary=True, ckpts=names, flagged=[dict(ckpt=n, **x) for n, fs in zip(names, fl) for x in fs])) + "\n")
for name, rows in zip(names, history):
    for row in rows:
        if "eps_median" in row:
            b = " ".join(f'{x["rbin"]}:{x["eps_median"]:+.3f}' for x in row["bins"])
            print(f'{name:14s} {row["set"]:7s} {row["cls"]:15s} eps(>=8) {row["eps_median"]:+.3f} [{row["ci95"][0]:+.3f},{row["ci95"][1]:+.3f}] '
                  f'deps/dr {row["deps_dr"]:+.4f}/A k {row["k_power"]:+.2f}'
                  + (f' lam {row["lam_median"]:.3f}' if "lam_median" in row else "") + (f' qN/qI {row["qN_over_qion_median"]:+.3f}' if "qN_over_qion_median" in row else "") + f' | {b}')
nf = sum(len(x) for x in fl)
print(f"flags: {nf}" + "".join(f"\n  {n}: {x}" for n, fs in zip(names, fl) for x in fs))
sys.exit(3 if (a.stop_exit and nf) else 0)
