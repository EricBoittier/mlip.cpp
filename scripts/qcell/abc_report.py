"""Compact tables of the A/B/C evaluation: abc_report.py FT_EVAL_DIR"""
import json
import os
import sys

D = sys.argv[1]
KEY = ["separated (>=8 A, charged fragment)", "separated domain=biomolecules", "|charge|>=2", "|charge|=1", "charge=0", "all",
       "domain=biomolecules", "domain=electrolytes", "domain=metal_complexes", "domain=neutral_control",
       "spin_multiplicity=1", "spin_multiplicity=2", "spin_multiplicity=3", "domain=biomolecules |q|>=2",
       "domain=electrolytes |q|>=2", "domain=metal_complexes |q|>=2", "separated domain=metal_complexes",
       "separated domain=electrolytes", "multi-component sep<4", "atoms 0-30", "atoms 100-1000"]
M = {X: json.load(open(f"{D}/{X}/val_sel.metrics.json"))["strata"] for X in "ZABC"}
print("val_sel strata: E MAE meV/atom | F MAE meV/A  (Z A B C)")
for k in KEY:
    if k in M["Z"]:
        print("%-38s %5d  E %s | F %s" % (k, M["Z"][k]["structures"], " ".join("%6.2f" % M[X][k]["e_mae_mev_atom"] for X in "ZABC"),
                                          " ".join("%6.2f" % M[X][k]["f_mae_mev_a"] for X in "ZABC")))
for pair in ("B-minus-A", "C-minus-A", "B-minus-Z", "C-minus-Z", "A-minus-Z"):
    P = json.load(open(f"{D}/{pair}.val_sel.json"))["strata"]
    print(f"\n{pair} val_sel: diff [95% CI]")
    for k in KEY[:10]:
        if k in P:
            e, f = P[k]["E_mae_mev_atom"], P[k]["F_mae_mev_a"]
            print("%-38s E %+7.2f [%+6.2f,%+6.2f]  F %+7.2f [%+6.2f,%+6.2f]" % (k, e["diff"], *e["ci95"], f["diff"], *f["ci95"]))
print("\nQCell charged curves (PBE0+MBD refs, different functional): within-scan dE MAE meV")
C = {X: json.load(open(f"{D}/{X}/qcell_curves.metrics.json")) for X in "ZABC"}
for k in ("all", "dimer cc sep<4", "dimer cn sep<4", "dimer cn sep 4-8"):
    pass
print("all 20 scans:", " ".join(f"{X} {C[X]['curves']['dE_mae_mev']:.1f}" for X in "ZABC"))
for pair in ("B-minus-A", "C-minus-A", "B-minus-Z", "C-minus-Z", "A-minus-Z"):
    P = json.load(open(f"{D}/{pair}.qcell_curves.json"))["strata"]
    print(pair, "; ".join("%s %+.1f [%+.1f,%+.1f]" % (k, P[k]["scan_dE_mae_mev"]["diff"], *P[k]["scan_dE_mae_mev"]["ci95"])
                          for k in ("all", "dimer cc sep<4", "dimer cn sep<4", "dimer cn sep 4-8") if k in P and "scan_dE_mae_mev" in P[k]))
for X in "BC":
    p = f"{D}/{X}/latent_charge_corr.json"
    if not os.path.exists(p):
        print(X, "no latent charge result"); continue
    L = json.load(open(p))
    print(f"\nlatent charges {X}:")
    for sch in ("mulliken_charges", "lowdin_charges", "nbo_charges"):
        for g in ("all", "domain=biomolecules", "domain=electrolytes", "domain=metal_complexes", "domain=neutral_control", "|charge|>=2"):
            b = L[sch].get(g)
            if b:
                ps = b["per_structure_r"]
                print("  %-9s %-24s r %.3f slope %.3f sign %.3f per-str median %.3f frac<0 %.3f mol-sum r %.3f" % (
                    sch.split("_")[0], g, b["pooled_r"], b["slope_ref_on_latent"], b["sign_agree_|ref|>0.1"], ps["median"],
                    ps["frac_negative"], b["molecule_sum_r"]))
        el = L[sch]["all"]["per_element"]
        print("    per element r:", {e: round(v["r"], 2) for e, v in el.items() if e in ("H", "C", "N", "O", "F", "P", "S", "Cl", "Na", "K", "Li", "Br", "I")})
