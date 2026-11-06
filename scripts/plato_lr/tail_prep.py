#!/usr/bin/env python3
"""Tail-loss data for mlipcpp-train --tail-data, from merge-aug set A (pulled-apart OMol25; HF EricBoi/omol-longrange-tests,
config pulled_apart, canonical schema).

Per scan (structure id): the preferred, label_ok records (DFT at 4-12 A on the merge subset, anchors at 10-25 A; split
ct_ambiguous never read). Pairs (d, d_far = 25 A) for d >= --dmin:
  dE_DFT = E(d) - E(25)                         (within one structure: the gpu4pyscf/ORCA per-atom offset cancels)
  dE_ref = [E_es_exact + E_ind](d) - [...](25)  (frozen-density electrostatics + CPHF induction, merge-aug anchor parts)
Points below 8 A have no exact electrostatics; their dE_ref is the COM multipole expansion (q, mu, Theta of the monomer
DFT densities, through quadrupole-quadrupole) + induction -1/2 F.alpha.F (longrange-dimers multipole.py), which is what the
dimer E_mp is. It is written to points.jsonl for every point; validated against the exact parts at 8-12 A.

Writes OUT/<split>/ (MemmapSplit: ns.npy na.npy x.bin a.bin c.bin energy.bin non_conservative_force.bin charge.bin
spin_multiplicity.bin; energies and forces are stored but never fitted), OUT/<split>/tail_pairs.tsv, OUT/<split>/points.jsonl
and OUT/<split>.xyz (extxyz frames in split order, for mlipcpp-eval).
    python prep_tail.py CANON_DIR MONOMERS_JSONL MULTIPOLE_PY_DIR OUT [--dmin=8]"""
import sys, os, json, collections, numpy as np

CANON, MONO, MPDIR, OUT = sys.argv[1:5]
opt = dict(a.split("=", 1) for a in sys.argv[5:] if "=" in a)
DMIN = float(opt.get("--dmin", 8))
sys.path.insert(0, MPDIR)
import multipole as MP
from ase.data import atomic_numbers, atomic_masses
HA, BOHR = 27.211386245988, 0.529177210903

mono = collections.defaultdict(dict)
for l in open(MONO):
    m = json.loads(l); mono[m["id"]][m["which"]] = m


def com(sym, x):
    w = np.array([atomic_masses[atomic_numbers[s]] for s in sym]); return (w[:, None] * x).sum(0) / w.sum()


def e_mp(rec):
    """COM multipole electrostatics + induction (eV) between R (fragment 0) and U (fragment 1) at this geometry."""
    m = mono.get(rec["group"])
    if not m or "U" not in m or "R" not in m: return None
    x = np.array(rec["positions"]); sym = rec["species"]; fid = np.array(rec["fragment_id"])
    QR, QU = rec["fragment_charges"]
    M = {w: dict(q=Q, mu=m[w]["mu_au"], theta=m[w]["theta_au"], alpha=m[w].get("alpha_au")) for w, Q in (("R", QR), ("U", QU))}
    Rv = (com([s for s, f in zip(sym, fid) if f == 1], x[fid == 1]) - com([s for s, f in zip(sym, fid) if f == 0], x[fid == 0])) / BOHR
    ind = MP.induction(M["R"], M["U"], Rv) if M["R"]["alpha"] is not None and M["U"]["alpha"] is not None else 0.0
    return (MP.electrostatic(M["R"], M["U"], Rv) + ind) * HA


def write(split):
    recs = [json.loads(l) for l in open(f"{CANON}/pulled_apart/{split}.jsonl")]
    recs = [r for r in recs if r["extra"]["label_ok"] and r["extra"]["preferred"]]
    by = collections.defaultdict(list)
    for r in recs: by[r["group"]].append(r)
    D = f"{OUT}/{split}"; os.makedirs(D, exist_ok=True)
    frames, pairs, points, skipped = [], [], [], collections.Counter()
    for g, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["distance"])
        far = [r for r in rs if abs(r["distance"] - 25) < 1e-6]
        if not far: skipped["no 25 A point"] += 1; continue
        far = far[0]
        for r in rs:
            ex = r["refs"].get("E_es_exact"); ind = r["refs"].get("E_ind")
            r["_ref"] = ex + (ind or 0.0) if ex is not None else None
            r["_mp"] = e_mp(r)
            r["_k"] = len(frames); frames.append(r)
            points.append(dict(k=r["_k"], id=r["id"], group=g, cls=r["cls"], d=r["distance"], r_com=r["extra"]["r_com"],
                               label_source=r["label_source"], E=r["energy"], E_ref_exact=r["_ref"], E_mp=r["_mp"],
                               n=len(r["species"]), frag=r["fragment_id"], QU=r["extra"]["QU"], QR=r["extra"]["QR"], formula_U=r["extra"]["formula_U"]))
        for r in rs:
            if r is far or r["distance"] < DMIN - 1e-6: continue
            ref = r["_ref"] - far["_ref"] if r["_ref"] is not None and far["_ref"] is not None else r["_mp"] - far["_mp"]
            pairs.append((g, r["cls"], r["_k"], far["_k"], r["energy"] - far["energy"], ref, r["distance"]))
    n = np.array([len(r["species"]) for r in frames]); na = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    np.save(f"{D}/ns.npy", np.array(len(frames), dtype=np.int64)); np.save(f"{D}/na.npy", na)
    np.concatenate([np.array(r["positions"], np.float32) for r in frames]).tofile(f"{D}/x.bin")
    np.concatenate([np.array([atomic_numbers[s] for s in r["species"]], np.int32) for r in frames]).tofile(f"{D}/a.bin")
    np.zeros((len(frames), 3, 3), np.float32).tofile(f"{D}/c.bin")
    np.array([r["energy"] for r in frames], np.float32).tofile(f"{D}/energy.bin")
    np.concatenate([np.array(r["forces"], np.float32) for r in frames]).tofile(f"{D}/non_conservative_force.bin")
    np.array([[r["charge"]] for r in frames], np.float32).tofile(f"{D}/charge.bin")
    np.array([[r["spin"]] for r in frames], np.float32).tofile(f"{D}/spin_multiplicity.bin")
    with open(f"{D}/tail_pairs.tsv", "w") as f:
        f.write("group\tclass\tk\tl\tdE_dft_eV\tdE_ref_eV\td_A\n")
        for p in pairs: f.write("%s\t%s\t%d\t%d\t%.12g\t%.12g\t%.6g\n" % p)
    with open(f"{D}/points.jsonl", "w") as f:
        for p in points: f.write(json.dumps(p) + "\n")
    with open(f"{OUT}/{split}.xyz", "w") as f:
        for r in frames:
            f.write(f"{len(r['species'])}\nProperties=species:S:1:pos:R:3 charge={r['charge']} spin_multiplicity={r['spin']} pbc=\"F F F\"\n")
            for s, x in zip(r["species"], r["positions"]): f.write(f"{s} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}\n")
    cnt = collections.Counter((p[1], round(p[6])) for p in pairs); gc = collections.Counter(p[1] for p in {(p[0], p[1]) for p in pairs})
    print(f"{split}: {len(frames)} structures, {len(by)} scans, {len(pairs)} pairs; groups per class {dict(gc)}; skipped {dict(skipped)}")
    print("  pairs per (class, d):", dict(sorted(cnt.items())))
    # the multipole reference against the exact parts where both exist (dE vs 25 A)
    rel = collections.defaultdict(list)
    for g, rs in by.items():
        far = [r for r in rs if abs(r["distance"] - 25) < 1e-6 and "_ref" in r]
        if not far or far[0]["_ref"] is None or far[0]["_mp"] is None: continue
        for r in rs:
            if "_ref" in r and r is not far[0] and r["_ref"] is not None and r["_mp"] is not None:
                ex, mp = r["_ref"] - far[0]["_ref"], r["_mp"] - far[0]["_mp"]
                if abs(ex) > 1e-4: rel[(r["cls"], round(r["distance"]))].append((mp - ex) / ex)
    print("  multipole dE_ref vs exact, median |rel| per (class, d):", {k: round(float(np.median(np.abs(v))), 4) for k, v in sorted(rel.items())})


for split in ("train", "val"): write(split)
