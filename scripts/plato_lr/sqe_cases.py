"""Reference cases for the mlip.cpp host SQE (fp64 metatrain): E, q, F, and dL/dtheta for L = a E + sum w.F."""
import sys, json, torch
sys.path.insert(0, "/home/boittier/Documents/metawork/metatrain-pet-sqe/src")
from metatrain.pet.modules.sqe import SQEParams, sqe_system, q0_fragment_loss
LEARNED = len(sys.argv) > 1 and sys.argv[1] == "learned"
torch.set_default_dtype(torch.float64); torch.manual_seed(1)
idx = json.load(open("/home/boittier/data/petlr-lrd/lrdall_index.json"))
Zs = [1, 3, 6, 7, 8, 9, 11, 15, 17]
P = SQEParams(Zs)
import os
if os.environ.get("LR_R"): P.cfg["lr_R"] = float(os.environ["LR_R"])
with torch.no_grad():
    for p in P.parameters(): p.add_(0.4 * torch.randn_like(p))
P.learned_q0 = LEARNED
WS = 0.3  # supervision weight (learned only)
cases = []
pick = [x for x in idx if x["pair"] in ("Li_PF6", "Gdm_NO3", "NH4_Cl", "MeOH_MeOH") and x["orient"] == 0 and x["dmin"] in (2.5, 4.0, 5.5, 6.0, 10.0)]
names = [k for k, _ in P.named_parameters()]
for x in pick:
    for mode in (("learned",) if LEARNED else ("uniform", "formal")):
        Z = x["Z"]; pos = torch.tensor(x["pos"], requires_grad=True); n = len(Z); nA = x["nA"]
        e = torch.tensor([Zs.index(z) for z in Z]); Q = torch.tensor(float(x["charge"]))
        q0 = None
        if mode == "formal":
            q0 = torch.zeros(n); q0[0] = x["qA"]; q0[nA] = x["qB"]
        ea, q, aux = sqe_system(P, e, pos, Q, q0)
        E = ea.sum(); F = -torch.autograd.grad(E, pos, create_graph=True)[0]
        w = torch.randn(n, 3); a = 0.7
        L = a * E + (w * F).sum()
        unit = torch.tensor([0] * nA + [1] * (n - nA)); Qu = torch.tensor([float(x["qA"]), float(x["qB"])])
        if LEARNED:
            L = L + WS * q0_fragment_loss(_["q0"] if False else sqe_system(P, e, pos, Q)[2]["q0"], unit, Qu)
        gs = [g if g is not None else torch.zeros_like(p) for g, p in zip(torch.autograd.grad(L, list(P.parameters()), allow_unused=True), P.parameters())]
        flat = torch.cat([g.reshape(-1) for g in gs])
        cases.append(dict(name=f"{x['pair']}_{x['dmin']}_{mode}", Z=Z, pos=x["pos"], Q=float(x["charge"]),
                          q0=None if q0 is None else q0.tolist(), E=float(E), q=q.tolist(), F=F.detach().reshape(-1).tolist(),
                          a=a, w=w.reshape(-1).tolist(), grad=flat.tolist(), q0ref=aux["q0"].detach().tolist(),
                          unit=unit.tolist(), Qu=Qu.tolist(), ws=WS if LEARNED else 0.0))
spec = dict(Z=Zs, chi0=P.chi0.tolist(), J0=P.J0.tolist(), rcov=P.rcov.tolist(), sigma=P.sigma.tolist(),
            chi_a=P.chi_a.tolist(), chi_b=P.chi_b.tolist(), j_z=P.j_z.tolist(), kappa_z=P.kappa_z.tolist(), w_raw=float(P.w_raw),
            cfg=P.cfg, q0_lo=P.q0_lo.tolist(), q0_hi=P.q0_hi.tolist(), q0_a=P.q0_a.tolist(),
            q0_b=P.q0_b.reshape(-1).tolist(), learned=LEARNED)
json.dump(dict(spec=spec, cases=cases), open("/home/boittier/data/petlr-sqe/parity/cases%s%s.json" % ("_learned" if LEARNED else "", "_lr" if os.environ.get("LR_R") else ""), "w"))
print(len(cases), "cases")
