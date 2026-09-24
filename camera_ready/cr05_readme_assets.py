"""README animations for the GitHub repository (not used in the paper).

  assets/threshold_sweep.gif  modularity gap vs. |z| threshold (why H4 is not robust)
  assets/network_sweep.gif    group-mean 30-node FC graph as the threshold rises
  assets/roi_slopes.gif       per-ROI group effect sizes moving from 0-back to 2-back
  assets/leakage_audit.gif    accuracy by feature set (target-leakage audit)

Inputs: camera_ready/fc30.npz (cr01), results/latest/efficiency/neural_efficiency.csv,
        camera_ready/cr03_results.json
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)
from scripts.s03_connectivity_analysis import get_network_parcels, compute_modularity  # noqa: E402

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
HIGH, LOW, GREY, INK = "#1a5276", "#c0392b", "#b0b7bd", "#2c3e50"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#333", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

df = pd.read_csv(ROOT / "results/latest/efficiency/neural_efficiency.csv")
subs = df["subject"].astype(str).tolist()
high = (df["efficiency_group"] == "High_Efficiency").values
fc = np.load(ROOT / "camera_ready/fc30.npz")
nets = get_network_parcels()
part = [i for i, (_, ps) in enumerate(nets.items()) for _ in ps]
net_names = list(nets)


def q_run(thr):
    """Whole-run Q per subject: binary |z|>thr, a priori partition, averaged over runs (as s03)."""
    out = []
    for s in subs:
        qs = []
        for r in ("LR", "RL"):
            k = f"{s}|full|{r}"
            if k in fc:
                a = (np.abs(fc[k]) > thr).astype(float)
                np.fill_diagonal(a, 0)
                qs.append(compute_modularity(a, part))
        out.append(np.mean(qs))
    return np.array(out)


def cohen(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


# ------------------------------------------------------------------ 1. threshold sweep
thrs = np.round(np.arange(0.05, 0.401, 0.01), 2)
Q = {t: q_run(t) for t in thrs}
D = np.array([cohen(Q[t][high], Q[t][~high]) for t in thrs])
P = np.array([stats.ttest_ind(Q[t][high], Q[t][~high], equal_var=False).pvalue for t in thrs])
dens = []
for t in thrs:
    dd = [((np.abs(fc[f"{s}|full|LR"]) > t).sum() - 0) / (30 * 29) for s in subs if f"{s}|full|LR" in fc]
    dens.append(np.mean(dd))
dens = np.array(dens)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})


def draw_sweep(i):
    t = thrs[i]
    ax1.clear(); ax2.clear()
    bins = np.linspace(min(Q[t].min(), -0.06), max(Q[t].max(), 0.06), 30)
    ax1.hist(Q[t][high], bins=bins, color=HIGH, alpha=0.6, label="High efficiency")
    ax1.hist(Q[t][~high], bins=bins, color=LOW, alpha=0.6, label="Low efficiency")
    ax1.axvline(Q[t][high].mean(), color=HIGH, lw=2)
    ax1.axvline(Q[t][~high].mean(), color=LOW, lw=2)
    ax1.set_xlabel("Whole-run modularity Q")
    ax1.set_ylabel("Participants")
    ax1.set_title(f"|z| > {t:.2f}   (graph density {dens[i]:.2f})", fontsize=10, loc="left")
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.axhspan(-0.1, 0.5, color="white")
    sig = P < 0.05
    ax2.plot(thrs, D, color=GREY, lw=1.5)
    ax2.scatter(thrs[:i + 1][~sig[:i + 1]], D[:i + 1][~sig[:i + 1]], color=GREY, s=18, zorder=3, label="p ≥ 0.05")
    ax2.scatter(thrs[:i + 1][sig[:i + 1]], D[:i + 1][sig[:i + 1]], color=HIGH, s=30, zorder=4, label="p < 0.05")
    ax2.axvline(0.15, color="#999", ls=":", lw=1)
    ax2.text(0.152, D.max() + 0.03, "paper's threshold", fontsize=7, color="#777")
    ax2.axhline(0, color="#555", lw=0.8, ls="--")
    ax2.scatter([t], [D[i]], s=120, facecolor="none", edgecolor=INK, lw=1.5, zorder=5)
    ax2.set_xlim(thrs[0] - 0.01, thrs[-1] + 0.01)
    ax2.set_ylim(min(-0.05, D.min() - 0.05), D.max() + 0.1)
    ax2.set_xlabel("Binarization threshold |z|")
    ax2.set_ylabel("Cohen's d (high − low)")
    ax2.set_title("The modularity gap depends on the threshold", fontsize=10, loc="left")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()


frames = list(range(len(thrs))) + [len(thrs) - 1] * 6
FuncAnimation(fig, draw_sweep, frames=frames).save(ASSETS / "threshold_sweep.gif", writer=PillowWriter(fps=4), dpi=100)
plt.close(fig)
print("threshold_sweep.gif", dict(zip(thrs.tolist(), np.round(D, 2).tolist())))

# ------------------------------------------------------------------ 2. network sweep
mean_fc = np.mean([fc[f"{s}|full|{r}"] for s in subs for r in ("LR", "RL") if f"{s}|full|{r}" in fc], axis=0)
n = mean_fc.shape[0]
ang = 2 * np.pi * np.arange(n) / n
xy = np.c_[np.cos(ang), np.sin(ang)]
cols = plt.cm.tab10(np.array(part) % 10)
sweep = np.round(np.arange(0.05, 0.61, 0.025), 3)
fig, ax = plt.subplots(figsize=(4.6, 4.6))


def draw_net(i):
    t = sweep[i]
    ax.clear(); ax.set_aspect("equal"); ax.axis("off")
    iu = np.triu_indices(n, 1)
    keep = np.abs(mean_fc[iu]) > t
    for a, b, w in zip(iu[0][keep], iu[1][keep], mean_fc[iu][keep]):
        same = part[a] == part[b]
        ax.plot(*zip(xy[a], xy[b]), color=cols[a] if same else "#9aa5ad",
                lw=0.4 + 2.2 * min(abs(w), 1), alpha=0.75 if same else 0.25, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], c=cols, s=70, edgecolor="white", lw=1, zorder=3)
    for k, name in enumerate(net_names):
        idx = [j for j in range(n) if part[j] == k]
        c = xy[idx].mean(0) * 1.28
        ax.text(c[0], c[1], name, ha="center", va="center", fontsize=8, color=plt.cm.tab10(k % 10), weight="bold")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Group-mean FC, |z| > {t:.3f}\ndensity = {keep.mean():.2f}", fontsize=10)


frames = list(range(len(sweep))) + [len(sweep) - 1] * 5
FuncAnimation(fig, draw_net, frames=frames).save(ASSETS / "network_sweep.gif", writer=PillowWriter(fps=4), dpi=100)
plt.close(fig)
print("network_sweep.gif")

# ------------------------------------------------------------------ 3. ROI slopes
d = json.load(open(ROOT / "camera_ready/cr03_results.json"))["d"]
rois = list(d)
d0 = np.array([d[r][0] for r in rois]); d2 = np.array([d[r][1] for r in rois])
pat = (d0 < 0) & (d2 >= 0)
fig, ax = plt.subplots(figsize=(5, 3.8))
steps = np.linspace(0, 1, 20)


def draw_roi(i):
    f = steps[min(i, len(steps) - 1)]
    ax.clear()
    ax.axhline(0, color="#555", ls="--", lw=0.8)
    for a, b, on, r in zip(d0, d2, pat, rois):
        cur = a + (b - a) * f
        ax.plot([0, f], [a, cur], color=HIGH if on else GREY, lw=2 if on else 1.2, marker="o", ms=4)
        if f == 1 and on:
            pass
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.37, 0.14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0-back (low load)", "2-back (high load)"])
    ax.set_ylabel("Cohen's d (high − low)")
    ax.set_title(f"Precise activation: {int(pat.sum())}/14 ROIs cross zero\n(trend only: permutation p = 0.094)", fontsize=10, loc="left")
    fig.tight_layout()


FuncAnimation(fig, draw_roi, frames=len(steps) + 8).save(ASSETS / "roi_slopes.gif", writer=PillowWriter(fps=8), dpi=100)
plt.close(fig)
print("roi_slopes.gif")

# ------------------------------------------------------------------ 4. leakage audit
sets = ["Neural only\n(90 features)", "IES$_{2bk}$ alone\n(label proxy)", "All 98 features\n(incl. behavioral)"]
lr = [0.515, 0.830, 0.875]; gbm = [0.525, 0.810, 0.890]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
steps = np.linspace(0, 1, 15)


def draw_leak(i):
    f = steps[min(i, len(steps) - 1)]
    ax.clear()
    x = np.arange(3)
    ax.bar(x - 0.18, [0.5 + (v - 0.5) * f for v in lr], 0.34, color=HIGH, label="Logistic regression")
    ax.bar(x + 0.18, [0.5 + (v - 0.5) * f for v in gbm], 0.34, color="#5dade2", label="Gradient boosting")
    ax.axhline(0.5, color=LOW, ls="--", lw=1.2)
    ax.text(2.45, 0.505, "chance", color=LOW, fontsize=8, ha="right", va="bottom")
    if f == 1:
        for xi, a, b in zip(x, lr, gbm):
            ax.text(xi - 0.18, a + 0.01, f"{a:.0%}", ha="center", fontsize=8)
            ax.text(xi + 0.18, b + 0.01, f"{b:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sets, fontsize=8)
    ax.set_ylim(0.4, 0.97); ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Leakage audit: the accuracy comes from the label's own ingredients", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()


FuncAnimation(fig, draw_leak, frames=len(steps) + 10).save(ASSETS / "leakage_audit.gif", writer=PillowWriter(fps=8), dpi=100)
plt.close(fig)
print("leakage_audit.gif")
