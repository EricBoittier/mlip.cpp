#!/usr/bin/env python3
"""Stage 1 of the Fukui-equilibration arm: the head_q0 charge head (fit_head.py of petlr-chargehead: MACE-POLAR-1-style
head on the frozen start backbone's last-layer features, charge pass at Q = 0) refitted jointly with one of the Fukui
modes of the lrc graph mode (mlip.cpp feat/plato-lr-fukui, LrcSpec::FukuiMode):
  uniform    f_i = 1                                        (U)
  element    f_i = softplus(phi_Z)                          (E, the head_q0 control)
  atom       f_i = softplus(phi_Z + a_i)                    (A; a_i = a head of the charge head's shape on the same
                                                             features, so mk_fgguf.py maps it onto a GGUF copy of the
                                                             start model exactly as mk_qgguf.py does for q_raw)
  atom_pm    f+ / f- (two such heads), w = (1 - s) f+/F+ + s f-/F-, s = smootherstep(r / d + 1/2), r = Q - sum q_raw  (A+-)
  atom_cluster  per-atom f, no global step: q_raw projected per 8 A single-link cluster (hard clusters offline; the
             runtime uses the soft lrc clusters) onto plateaued totals P (width 0.6), residual Q - sum P routed by the
             plateau slope (lrc_qproj's formula with the residual)  (AC)
    q_i = q_raw_i + w_i (Q - sum_j q_raw_j)   (global modes)
phi starts from the head_q0 fit (fukui_q0.json); the a heads start at PyTorch's default (non-zero) init: no gate.
Supervision as head_q0 (per-atom DFT Loewdin, unit / fragment sums x WF, monomer dipoles x 0.05) plus, with --fl W,
DFT Fukui labels from the ipea set (q(N-1) - q(N), q(N) - q(N+1) per atom, all states' charge passes at Q = 0 with the
state's spin) on the training parents of --fold k/K (parent split, seed 2026).
Writes OUT.pt ({"state", "ZS", "mode"}), OUT.json (scores) and OUT_pred.pkl (per-point / per-parent predictions).
    python fit_fukui.py MODE ITERS OUT [--fl 0] [--fold 0/5] [--threads 4] [--pm-width 0.05] [--seed 13]"""
import argparse, json, pickle, time
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["uniform", "element", "atom", "atom_pm", "atom_cluster"])
ap.add_argument("iters", type=int)
ap.add_argument("out")
ap.add_argument("--fl", type=float, default=0.0, help="Fukui-label loss weight (e^2 per atom, relative to Loewdin)")
ap.add_argument("--fold", default="0/5")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--pm-width", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--clip", type=float, default=-1.0, help="gradient-norm clip (default: 1 for atom_cluster, off otherwise)")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wf", type=float, default=10.0)
ap.add_argument("--wr", type=float, default=-1.0, help="raw cluster-sum anchor weight (e^2): (sum_c q_raw - sum_c q_DFT)^2 "
                "per 8 A cluster; default 1 for atom_cluster (the plateau leaves the raw cluster totals a flat direction "
                "that drifts across the plateau edge without it), 0 otherwise")
ap.add_argument("--fit", default="/home/boittier/data/petlr-chargehead/fit")
ap.add_argument("--ipea", default="/home/boittier/data/plato-lr-scratch/ipea")
ap.add_argument("--ipea-feats", default="/home/boittier/data/plato-lr-scratch/fukui/feats")
ap.add_argument("--phi0", default="/home/boittier/data/petlr-chargehead/models/fukui_q0.json")
a = ap.parse_args()
torch.manual_seed(a.seed)
torch.set_num_threads(a.threads)
F = a.fit
BOHR = 0.529177210903
DN, DE = 1024, 256
MODE = a.mode
WR = a.wr if a.wr >= 0 else (1.0 if a.mode == "atom_cluster" else 0.0)
ATOM = MODE in ("atom", "atom_pm", "atom_cluster")


def load_bins(path, k=20):
    per = []
    for c in range(k):
        b = open(f"{path}.{c:02d}.bin", "rb").read(); o = 0; L = []
        while o < len(b):
            n, d = np.frombuffer(b[o:o + 16], np.int64); o += 16
            L.append(np.frombuffer(b[o:o + 4 * n * d], np.float32).reshape(n, d)); o += 4 * n * d
        per.append(L)
    return [per[i % k][i // k] for i in range(sum(len(L) for L in per))]


def clusters(x, cut=8.0):
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import cdist
    return connected_components(cdist(x, x) < cut, directed=False)[1].astype(np.int64)


t0 = time.time()
om = dict(np.load(f"{F}/omol_fit.npz", allow_pickle=True))
Xo = load_bins(f"{F}/featsq0/omol_fit")
G = pickle.load(open("/home/boittier/data/lr-sqe-pol/fermi/geoms.pkl", "rb"))["points"]
Xg = load_bins(f"{F}/featsq0/geoms")
mono = json.load(open(f"{F}/monomers.json"))
Xm = load_bins(f"{F}/featsq0/monomers")
bad = set(l.strip() for l in open("/home/boittier/data/data-audit/keep/setA_bad_structures.txt") if l.strip())
from ase.data import atomic_numbers, atomic_masses
from ase.io import read
ip_frames = read(f"{a.ipea}/ipea.xyz", ":")
Xi = load_bins(f"{a.ipea_feats}/ipea")
assert len(Xo) == len(om["idx"]) and len(Xg) == len(G) and len(Xm) == len(mono) and len(Xi) == len(ip_frames)
print(f"loaded ({time.time() - t0:.0f} s)", flush=True)

ZS = sorted(set(int(z) for zz in om["z"] for z in zz) | set(int(z) for p in G for z in p["Z"])
            | set(atomic_numbers[s] for m in mono for s in m["sym"]) | set(int(z) for fr in ip_frames for z in fr.numbers))
zix = {z: i for i, z in enumerate(ZS)}

S = []
rng = np.random.default_rng(13)  # the head_q0 OMol25 split
for i in range(len(Xo)):
    z = np.asarray(om["z"][i]); u = np.asarray(om["unit"][i]); x = np.asarray(om["x"][i], float)
    S.append(dict(X=Xo[i], z=z, Q=float(om["Q"][i]), lw=np.asarray(om["lowdin"][i], float), grp=u if u.max() > 0 else None,
                  mu=None, split="val" if rng.random() < 0.1 else "train", tag="omol", cl=clusters(x)))
for i, p in enumerate(G):
    lw = np.asarray(p["lowdin"], float)
    ok = lw.shape[0] == len(p["Z"]) and np.all(np.isfinite(lw))
    split = {"train": "train", "val": "val", "dimer": "dimer", "ct_ambiguous": "ct"}[p["split"]]
    if split == "train" and (p["scan"].split("|")[1] in bad or not ok): split = "drop"
    S.append(dict(X=Xg[i], z=np.asarray(p["Z"]), Q=float(p["Q"]), lw=lw if ok else None, grp=np.asarray(p["frag"]),
                  mu=None, split=split, tag="geom", gi=i, cl=clusters(np.asarray(p["R"], float))))
for i, m in enumerate(mono):
    z = np.array([atomic_numbers[s] for s in m["sym"]]); x = np.array(m["R"], float); ms = atomic_masses[z]
    com = (ms[:, None] * x).sum(0) / ms.sum()
    S.append(dict(X=Xm[i], z=z, Q=float(m["Q"]), lw=np.asarray(m["lowdin"], float), grp=None, mu=np.array(m["mu"], float),
                  Xc=(x - com) / BOHR, split="val" if "-val-" in m["id"] else "train", tag="mono", cl=clusters(x)))
# ipea: parents with N and at least one of N-1 / N+1 (all states: the same geometry, the state's spin, pass at Q = 0)
P = [json.loads(l) for l in open(f"{a.ipea}/points.jsonl")]
P = [p for p in P if p.get("status") == "stable" and "k" in p]
byk = {p["k"]: p for p in P}
par = {}
for k, fr in enumerate(ip_frames):
    p = byk[k]; assert fr.info["id"] == p["id"], (k, fr.info["id"], p["id"])
    par.setdefault(p["frag"], {})[p["state"]] = k
parents = sorted(f for f, d in par.items() if "N" in d and ("Nm1" in d or "Np1" in d))
fk, fK = (int(v) for v in a.fold.split("/"))
perm = np.random.default_rng(2026).permutation(len(parents))
test_par = set(parents[i] for i in perm if i % fK == fk)
IP = {}
for k, fr in enumerate(ip_frames):
    IP[k] = dict(X=Xi[k], z=fr.numbers.copy(), Q=float(fr.info["charge_orig"]) if "charge_orig" in fr.info else float(byk[k]["charge"]),
                 lw=fr.arrays["lowdin"].astype(float).ravel(), cl=np.zeros(len(fr), np.int64), tag="ipea")
print({s: sum(1 for d in S if d["split"] == s) for s in ("train", "val", "dimer", "ct", "drop")},
      f"ipea parents {len(parents)} (test fold {fk}/{fK}: {len(test_par)})", flush=True)

tr = [d for d in S if d["split"] == "train"]
Xall = np.concatenate([d["X"] for d in tr])
mu_n, sd_n = Xall[:, :DN].mean(0), Xall[:, :DN].std(0) + 1e-3
mu_e, sd_e = Xall[:, DN:DN + DE].mean(0), Xall[:, DN:DN + DE].std(0) + 1e-3
del Xall
phi0 = json.load(open(a.phi0)); phi0 = dict(zip(phi0["Z"], phi0["phi"]))


def sstep(t):
    t = torch.clamp(t, 0.0, 1.0)
    return t ** 3 * (t * (t * 6 - 15) + 10)


class HeadNet(torch.nn.Module):  # the PET energy-head shape: node MLP 1024 -> 256 -> 256 -> 1 + linear edge term + bias x count
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(DN, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256), torch.nn.SiLU(),
                                       torch.nn.Linear(256, 1))
        self.we = torch.nn.Linear(DE, 1)
        self.be = torch.nn.Parameter(torch.zeros(1))

    def forward(self, hn, he, cnt):
        return self.mlp(hn)[:, 0] + self.we(he)[:, 0] + self.be * cnt


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = HeadNet()
        p = torch.tensor([phi0.get(z, 0.0) for z in ZS], dtype=torch.float32)
        self.phi = torch.nn.Parameter(p.clone())
        if ATOM:
            self.fa = HeadNet()
        if MODE == "atom_pm":
            self.phim = torch.nn.Parameter(p.clone())
            self.fam = HeadNet()
        for k, v in (("mn", mu_n), ("sn", sd_n), ("me", mu_e), ("se", sd_e)):
            self.register_buffer(k, torch.tensor(v))

    def weights(self, hn, he, cnt, e, minus=False):
        if MODE == "uniform":
            return torch.ones(len(e))
        u = (self.phim if minus else self.phi)[e]
        if ATOM:
            u = u + (self.fam if minus else self.fa)(hn, he, cnt)
        return torch.nn.functional.softplus(u) + 1e-6

    def forward(self, X, e, seg, Q, nseg, cl, ncl):
        hn = (X[:, :DN] - self.mn) / self.sn; he = (X[:, DN:DN + DE] - self.me) / self.se; cnt = X[:, -1]
        q = self.q(hn, he, cnt)
        f = self.weights(hn, he, cnt, e)
        sq = torch.zeros(nseg).index_add_(0, seg, q)
        sf = torch.zeros(nseg).index_add_(0, seg, f)
        r = Q - sq
        self.last_raw, self.last_f = q, f
        if MODE == "atom_cluster":
            Q0 = torch.zeros(ncl).index_add_(0, cl, q); Fc = torch.zeros(ncl).index_add_(0, cl, f)
            k = torch.floor(Q0.detach()); u = (Q0 - k - 0.5) / 0.6 + 0.5
            Pv = k + sstep(u)
            dP = torch.where((u > 0) & (u < 1), 30 * u * u * (u - 1) ** 2, torch.zeros_like(u)) / 0.6
            s = Pv - Q0; g = Fc * (1e-3 + dP)
            cseg = torch.zeros(ncl, dtype=torch.long).index_put_((cl,), seg)
            Ss = torch.zeros(nseg).index_add_(0, cseg, s); Gs = torch.zeros(nseg).index_add_(0, cseg, g)
            T = r - Ss
            t = s + T[cseg] * g / Gs[cseg]
            return q + f / Fc[cl] * t[cl]
        w = f / sf[seg]
        if MODE == "atom_pm":
            fm = self.weights(hn, he, cnt, e, True)
            sm = torch.zeros(nseg).index_add_(0, seg, fm)
            s = sstep(r / a.pm_width + 0.5) if a.pm_width > 0 else (r > 0).float()
            w = (1 - s[seg]) * w + s[seg] * fm / sm[seg]
        return q + w * r[seg]


def batch(ds):
    X = torch.tensor(np.concatenate([d["X"] for d in ds]))
    e = torch.tensor(np.concatenate([[zix[int(z)] for z in d["z"]] for d in ds]))
    seg = torch.tensor(np.concatenate([[k] * len(d["z"]) for k, d in enumerate(ds)]))
    off, cl = 0, []
    for d in ds:
        cl.append(d["cl"] + off); off += int(d["cl"].max()) + 1
    return X, e, seg, torch.tensor([d["Q"] for d in ds], dtype=torch.float32), torch.tensor(np.concatenate(cl)), off


def run(model, ds):
    X, e, seg, Q, cl, ncl = batch(ds)
    q = model(X, e, seg, Q, len(ds), cl, ncl)
    off = np.cumsum([0] + [len(d["z"]) for d in ds])
    return q, off


def loss_terms(model, ds):
    q, off = run(model, ds)
    raw = model.last_raw
    la, lf, lm, na, nf, nm = 0.0, 0.0, 0.0, 0, 0, 0
    lr, nr = 0.0, 0
    for k, d in enumerate(ds):
        qk = q[off[k]:off[k + 1]]
        if d["lw"] is not None:
            lw = torch.tensor(d["lw"], dtype=torch.float32)
            if WR > 0:
                c = torch.tensor(d["cl"]); K = int(c.max()) + 1
                lr = lr + ((torch.zeros(K).index_add_(0, c, raw[off[k]:off[k + 1]]) - torch.zeros(K).index_add_(0, c, lw)) ** 2).sum()
                nr += K
            la = la + ((qk - lw) ** 2).sum(); na += len(qk)
            if d.get("grp") is not None:
                g = torch.tensor(d["grp"]); U = int(g.max()) + 1
                lf = lf + ((torch.zeros(U).index_add_(0, g, qk) - torch.zeros(U).index_add_(0, g, lw)) ** 2).sum(); nf += U
        if d.get("mu") is not None:
            m = (qk[:, None] * torch.tensor(d["Xc"], dtype=torch.float32)).sum(0)
            lm = lm + ((m - torch.tensor(d["mu"], dtype=torch.float32)) ** 2).sum(); nm += 1
    model.last_lraw = lr / max(nr, 1)
    return la / max(na, 1), lf / max(nf, 1), lm / max(nm, 1), q, off


def fukui_pairs(pars):  # (parent, label kind, k_lo, k_hi): q(k_hi) - q(k_lo) for one more unit of total charge
    out = []
    for f in pars:
        d = par[f]
        if "Nm1" in d: out.append((f, "f-", d["N"], d["Nm1"]))
        if "Np1" in d: out.append((f, "f+", d["Np1"], d["N"]))
    return out


def fukui_loss(model, pairs):
    ks = sorted(set(k for p in pairs for k in p[2:]))
    q, off = run(model, [IP[k] for k in ks])
    pos = {k: i for i, k in enumerate(ks)}
    L, n, per = 0.0, 0, []
    for f, kind, lo, hi in pairs:
        dm = q[off[pos[hi]]:off[pos[hi] + 1]] - q[off[pos[lo]]:off[pos[lo] + 1]]
        dd = torch.tensor(IP[hi]["lw"] - IP[lo]["lw"], dtype=torch.float32)
        L = L + ((dm - dd) ** 2).sum(); n += len(dd)
        per.append((f, kind, dm.detach().numpy().astype(float), dd.numpy().astype(float)))
    return L / max(n, 1), per


model = Model()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
CLIP = a.clip if a.clip >= 0 else (1.0 if MODE == "atom_cluster" else 0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
WF, WM = a.wf, 0.05
fl_train = fukui_pairs([p for p in parents if p not in test_par]) if a.fl > 0 else []
t0 = time.time()
for it in range(a.iters):
    idx = rng.choice(len(tr), size=256, replace=False)
    la, lf, lm, _, _ = loss_terms(model, [tr[i] for i in idx])
    L = la + WF * lf + WM * lm + (WR * model.last_lraw if WR > 0 else 0.0)
    if a.fl > 0:
        sub = [fl_train[i] for i in rng.choice(len(fl_train), size=min(32, len(fl_train)), replace=False)]
        lfl, _ = fukui_loss(model, sub)
        L = L + a.fl * lfl
    opt.zero_grad(); L.backward()
    if CLIP > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step(); sched.step()
    if it % 500 == 0 or it == a.iters - 1:
        print(f"it {it} atom {float(la.detach()):.5f} frag {float(lf.detach() if torch.is_tensor(lf) else lf):.5f} dip {float(lm.detach() if torch.is_tensor(lm) else lm):.4f} ({time.time() - t0:.0f} s)", flush=True)


@torch.no_grad()
def predict(ds, bs=512):
    out = []
    for s0 in range(0, len(ds), bs):
        part = ds[s0:s0 + bs]
        q, off = run(model, part)
        raw, f = model.last_raw, model.last_f
        out += [(q[off[k]:off[k + 1]].numpy().astype(float), raw[off[k]:off[k + 1]].numpy().astype(float),
                 f[off[k]:off[k + 1]].numpy().astype(float)) for k in range(len(part))]
    return out


geo = [d for d in S if d["tag"] == "geom"]
pg = predict(geo)
points = []
for d, (q, raw, f) in zip(geo, pg):
    p = G[d["gi"]]
    if d["lw"] is None: continue
    g = d["grp"]
    points.append(dict(i=d["gi"], set=p["set"], split=p["split"], cls=p["cls"], d=float(p["d"]), scan=p["scan"], Q=float(p["Q"]),
                       q=[float(q[g == k].sum()) for k in (0, 1)], dft=[float(d["lw"][g == k].sum()) for k in (0, 1)],
                       raw=[float(raw[g == k].sum()) for k in (0, 1)], F=[float(f[g == k].sum()) for k in (0, 1)],
                       ncl=int(d["cl"].max()) + 1))
res = {"mode": MODE, "iters": a.iters, "fl": a.fl, "wr": WR, "fold": a.fold, "pm_width": a.pm_width}
for tag, sp in (("omol", "val"), ("mono", "val")):
    ds = [d for d in S if d["tag"] == tag and d["split"] == sp]
    qs = [v[0] for v in predict(ds)]
    r = {"atom_mae": float(np.mean(np.concatenate([np.abs(q - d["lw"]) for d, q in zip(ds, qs)])))}
    if tag == "mono":
        r["dipole_mae_au"] = float(np.mean([np.linalg.norm((q[:, None] * d["Xc"]).sum(0) - d["mu"]) for d, q in zip(ds, qs)]))
    res[f"{tag}_{sp}"] = r
with torch.no_grad():
    _, per_test = fukui_loss(model, fukui_pairs(sorted(test_par)))
    _, per_all = fukui_loss(model, fukui_pairs(parents))
res["fukui_test_mae"] = float(np.mean(np.concatenate([np.abs(m - d) for _, _, m, d in per_test])))
json.dump(res, open(a.out + ".json", "w"), indent=1)
pickle.dump({"points": points, "fukui_test": per_test, "fukui_all": per_all, "test_parents": sorted(test_par)},
            open(a.out + "_pred.pkl", "wb"))
torch.save({"state": model.state_dict(), "ZS": ZS, "mode": MODE, "pm_width": a.pm_width}, a.out + ".pt")
print(json.dumps(res), flush=True)
