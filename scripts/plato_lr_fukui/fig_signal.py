#!/usr/bin/env python3
"""Forest plot of the offline signal check: paired difference vs the E control (95 % bootstrap CI) per arm, for the
pre-registered primary metric (leak), the neutral-partner charge, guard rails and the DFT Fukui labels.
    python fig_signal.py signal.json OUT.png"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1]))["metrics"]
panels = [("leak", "Primary: |fragment-sum error|,\nQ≠0 held out, 8–25 Å (pre-projection)"),
          ("neutral", "Neutral-partner |q|,\nion + neutral, 8–25 Å"),
          ("ct", "CT-ambiguous points\n(guard rail)"),
          ("near", "Held out < 8 Å\n(guard rail)"),
          (("fukui", "all"), "DFT Fukui labels (ipea),\nper-parent atom MAE")]
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
arms = [a for a in next(iter(r.values())) if a != "E"] if "E" in r["leak"] else list(r["leak"])
arms = [a for a in r["leak"] if a != "E"]
fig, axs = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 0.45 * len(arms) + 1.6), sharey=True)
for ax, (key, title) in zip(axs, panels):
    d = r[key[0]][key[1]] if isinstance(key, tuple) else r[key]
    for k, a in enumerate(arms):
        v = d[a]; y = len(arms) - 1 - k
        ax.plot(v["d_ci"], [y, y], color=COL[k % len(COL)], lw=2, solid_capstyle="round")
        ax.plot([v["d_vs_E"]], [y], "o", ms=7, color=COL[k % len(COL)], mec="white", mew=1.5)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(f"Δ vs E (e); E = {d['E']['mean']:.3f}, n = {d['E']['n']}", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
axs[0].set_yticks(range(len(arms))); axs[0].set_yticklabels(arms[::-1], fontsize=9)
fig.suptitle("Fukui arms vs element-Fukui control (stage-1 refits on frozen E8 features; negative = better; "
             "95 % bootstrap CI over scans / parents)", fontsize=9, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=180)
print(sys.argv[2])
