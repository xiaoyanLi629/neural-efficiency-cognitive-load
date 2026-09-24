"""Camera-ready (BIBM 2026): precise-activation pattern (reviewer #1, points 1-2).

  - recomputes the per-ROI group effect sizes of Table II
  - label-permutation test of the 7/14 ROI count (keeps ROI/condition dependence)
  - sensitivity: drop the participant without 2-back behaviour
  - new Fig. 2: d_0bk -> d_2bk slope plot for all 14 ROIs
Outputs: camera_ready/cr03_results.json, camera_ready/figs/fig2.{pdf,png}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3 (IEEE PDF eXpress)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "camera_ready"
(OUT / "figs").mkdir(exist_ok=True)
ROIS = ["DLPFC_L", "DLPFC_R", "VLPFC_L", "VLPFC_R", "PPC_L", "PPC_R", "ACC_L", "ACC_R",
        "Premotor_L", "Premotor_R", "mPFC", "PCC", "Angular_L", "Angular_R"]

df = pd.read_csv(ROOT / "results/latest/efficiency/neural_efficiency.csv")
high = (df["efficiency_group"] == "High_Efficiency").values
A0 = df[[f"{r}_activation_0bk" for r in ROIS]].values
A2 = df[[f"{r}_activation_2bk" for r in ROIS]].values


def cohen_d(x, g):
    a, b = x[g], x[~g]
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / sp


def count(g, a0=A0, a2=A2):
    d0, d2 = cohen_d(a0, g), cohen_d(a2, g)
    return int(np.sum((d0 < 0) & (d2 >= 0))), d0, d2


obs, d0, d2 = count(high)
rng = np.random.default_rng(42)
null = np.array([count(rng.permutation(high))[0] for _ in range(10000)])
p_perm = float((np.sum(null >= obs) + 1) / (len(null) + 1))

keep = df["acc_2bk"].notna().values
obs_drop, _, _ = count(high[keep], A0[keep], A2[keep])

res = {"n_pattern": obs, "perm_p": p_perm, "null_mean": float(null.mean()), "null_sd": float(null.std()),
       "n_pattern_drop_missing": obs_drop,
       "d": {r: [round(float(a), 3), round(float(b), 3)] for r, a, b in zip(ROIS, d0, d2)}}
json.dump(res, open(OUT / "cr03_results.json", "w"), indent=2)
print(json.dumps(res, indent=1))

# ---- Fig. 2: slope plot
pattern = (d0 < 0) & (d2 >= 0)
fig, ax = plt.subplots(figsize=(3.4, 3.0))
for r, a, b, on in zip(ROIS, d0, d2, pattern):
    col, lw, z = ("#1a5276", 1.6, 3) if on else ("#b0b7bd", 1.0, 2)
    ax.plot([0, 1], [a, b], color=col, lw=lw, marker="o", ms=3.5, zorder=z)
# labels for highlighted ROIs, pushed apart so they do not overlap
lab = sorted([(b, r) for r, b, on in zip(ROIS, d2, pattern) if on], reverse=True)
ys = []
for b, _ in lab:
    ys.append(min(b, ys[-1] - 0.024) if ys else b)
for (b, r), y in zip(lab, ys):
    ax.annotate(r.replace("_", "-"), xy=(1, b), xytext=(1.06, y), fontsize=6, va="center",
                color="#1a5276", arrowprops=dict(arrowstyle="-", color="#1a5276", lw=0.4))
ax.axhline(0, color="#555", ls="--", lw=0.8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["0-back", "2-back"], fontsize=8)
ax.set_xlim(-0.15, 1.35)
ax.set_ylabel("Cohen's d (high $-$ low)", fontsize=8)
ax.tick_params(labelsize=7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "figs/fig2.pdf")
fig.savefig(OUT / "figs/fig2.png", dpi=300)
print("saved fig2")
