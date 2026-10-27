"""Print eval_omol.py metrics JSONs as tables: show.py M.json ..."""
import json
import sys

for f in sys.argv[1:]:
    d = json.load(open(f))
    print("==", f, d.get("label"))
    for k, v in d["strata"].items():
        print("%-32s %6d  E %8.2f meV/at  Eoff %8.1f meV  F %7.2f meV/A  Frmse %7.2f" % (
            k, v["structures"], v["e_mae_mev_atom"], v["e_mae_offset_free_mev"], v["f_mae_mev_a"], v["f_rmse_mev_a"]))
    c = d.get("curves")
    if c:
        print("curves", c["scans"], "scans, dE MAE %.1f meV" % c["dE_mae_mev"])
        if c["scans"] <= 40:
            for s, v in c["per_scan"].items():
                print("  scan %s %s qA %+d qB %+d pts %d r %.2f-%.2f A, min_inter at r_max %.2f A, dE MAE %.1f meV" % (
                    s, v["class"], v["qA"], v["qB"], v["points"], v["r_min"], v["r_max"], v["min_inter_max"], v["dE_mae_mev"]))
