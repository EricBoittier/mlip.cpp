"""Learning curves (validation F and E MAE per epoch) of the QCell runs, and a summary table.

    python -I curves.py RUNS_DIR OUT_PREFIX

OUT_PREFIX.png (two panels, shared epoch axis, log y), OUT_PREFIX.json (per run: last epoch,
best val, test metrics of final / best weights if finish.sh wrote them).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = ["ft-muon", "ft-adam", "scratch-muon", "ft-muon-s1"]
COLORS = {"ft-muon": "#2a78d6", "ft-adam": "#eb6834", "scratch-muon": "#1baf7a", "ft-muon-s1": "#6250d6"}
STYLE = {"ft-muon-s1": (0, (4, 2))}


def main():
    d, out = sys.argv[1], sys.argv[2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    summary = {}
    for name in RUNS:
        p = f"{d}/{name}/log.jsonl"
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        rows = {r["epoch"]: r for r in rows if "val_f_mae_mev_a" in r}  # resumed jobs may repeat an epoch
        ep = sorted(rows)
        f = [rows[e]["val_f_mae_mev_a"] for e in ep]
        e = [rows[k]["val_e_mae_mev_atom"] for k in ep]
        for ax, y in zip(axes, (f, e)):
            ax.plot(ep, y, color=COLORS[name], lw=2, ls=STYLE.get(name, "-"), label=name)
            ax.annotate(name, (ep[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                        color="#52514e", va="center")
        s = {"epochs_logged": ep[-1], "val_f_mae_last": f[-1], "val_e_mae_last": e[-1],
             "val_f_mae_min": min(f), "val_f_mae_min_epoch": ep[f.index(min(f))]}
        for w in ("final", "best"):
            m = f"{d}/{name}/test.{w}.metrics.json"
            if os.path.exists(m):
                s[f"test_{w}"] = json.load(open(m))["all"]
        summary[name] = s
    for ax, t in zip(axes, ("validation force MAE (meV/Å)", "validation energy MAE (meV/atom)")):
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title(t, fontsize=10, loc="left", color="#0b0b0b")
        ax.grid(True, which="major", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("QCell, Plato-PET-LR in mlip.cpp: bio_dimers fine-tune vs scratch (epoch 0 = init)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=130, facecolor="#fcfcfb")
    json.dump(summary, open(out + ".json", "w"), indent=1)
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if not kk.startswith("test")}
                      for k, v in summary.items()}))


if __name__ == "__main__":
    main()
