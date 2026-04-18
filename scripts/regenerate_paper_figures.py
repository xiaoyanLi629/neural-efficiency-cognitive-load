#!/usr/bin/env python3
"""
Regenerate the 5 paper figures with fixes based on manuscript feedback.

Fixes applied:
- Fig 1 (group_comparison): remove main title; align panel titles
- Fig 2 (glass brain): plain black title, larger colorbar fonts
- Fig 3 (gap_reversal): remove panels C, F; remove main title; fix D title; fix B labels
- Fig 4 (radar): larger fonts; remove bottom hint
- Fig 5 (ML comparison): larger fonts

Saves PNGs directly to IEEE_manuscript/ as fig1-fig5.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
RUN_DIR = PROJECT_ROOT / 'results' / 'run_20260415_122619'
OUT_DIR = PROJECT_ROOT / 'IEEE_manuscript'
OUT_DIR.mkdir(exist_ok=True)

# Common style
HIGH_EFF = '#1a5276'
LOW_EFF = '#c0392b'
HIGH_EFF_LIGHT = '#5dade2'
LOW_EFF_LIGHT = '#f1948a'
TEXT = '#2c3e50'
LINE = '#95a5a6'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
})


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.tick_params(colors=TEXT)


def add_stats(ax, x1, x2, y, label='***'):
    from scipy import stats
    t, p = stats.ttest_ind(x1, x2)
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'
    mean_d = (np.mean(x1) - np.mean(x2)) / np.sqrt((np.var(x1) + np.var(x2)) / 2)
    ax.plot([0, 0, 1, 1], [y, y * 1.01, y * 1.01, y], 'k-', lw=1)
    ax.text(0.5, y * 1.02, f'{sig}\np = {p:.3f}\nd = {mean_d:.2f}',
            ha='center', va='bottom', fontsize=9, color=TEXT)


# =====================================================================
# Figure 1: Group Comparison (behavioral, 6 panels)
# =====================================================================
def make_fig1():
    print("Generating Figure 1 (group comparison)...")
    behav = pd.read_csv(RUN_DIR / 'behavioral' / 'behavioral_summary.csv')
    high = behav[behav['efficiency_group'] == 'High_Efficiency']
    low = behav[behav['efficiency_group'] == 'Low_Efficiency']
    n_high, n_low = len(high), len(low)

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.32, figure=fig)

    # Uniform title parameters (fix alignment)
    title_kw = dict(fontsize=13, fontweight='bold', color=TEXT, pad=14, loc='left')

    def raincloud(ax, h, l, ylabel, letter, ptitle):
        style_axis(ax)
        for pos, vals, col, col_light in [(0, h.values, HIGH_EFF, HIGH_EFF_LIGHT),
                                          (1, l.values, LOW_EFF, LOW_EFF_LIGHT)]:
            # Violin
            parts = ax.violinplot(vals, positions=[pos + 0.15], widths=0.35,
                                  showmeans=False, showmedians=False, showextrema=False)
            for b in parts['bodies']:
                b.set_facecolor(col_light); b.set_alpha(0.6); b.set_edgecolor(col); b.set_linewidth(1.3)
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            # Box
            bp = ax.boxplot(vals, positions=[pos], widths=0.18, patch_artist=True,
                            boxprops=dict(facecolor=col_light, edgecolor=col, linewidth=1.3),
                            medianprops=dict(color=col, linewidth=2),
                            whiskerprops=dict(color=col, linewidth=1),
                            capprops=dict(color=col, linewidth=1),
                            flierprops=dict(marker='', alpha=0))
            # Scatter
            np.random.seed(42 + pos)
            jitter = np.random.normal(0, 0.025, len(vals))
            ax.scatter(pos - 0.22 + jitter, vals, s=22, alpha=0.55,
                       c=col, edgecolor='white', linewidth=0.4, zorder=4)
        y_max = max(h.max(), l.max())
        y_min = min(h.min(), l.min())
        add_stats(ax, h.values, l.values, y_max + 0.08 * (y_max - y_min))
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'High Efficiency\n(n={n_high})', f'Low Efficiency\n(n={n_low})'],
                           fontsize=10, color=TEXT)
        ax.set_ylabel(ylabel, fontsize=11, color=TEXT)
        ax.set_title(letter, **title_kw)
        ax.set_xlim(-0.45, 1.5)

    # A: 2-back Accuracy
    ax_a = fig.add_subplot(gs[0, 0])
    raincloud(ax_a, high['acc_2bk'].dropna(), low['acc_2bk'].dropna(),
              'Accuracy (proportion)', 'A', '2-back Accuracy')

    # B: 2-back RT
    ax_b = fig.add_subplot(gs[0, 1])
    raincloud(ax_b, high['rt_2bk'].dropna(), low['rt_2bk'].dropna(),
              'Reaction Time (ms)', 'B', '2-back Reaction Time')

    # C: Load x Efficiency Interaction
    ax_c = fig.add_subplot(gs[0, 2])
    style_axis(ax_c)
    means_h = [high['acc_0bk'].mean(), high['acc_2bk'].mean()]
    means_l = [low['acc_0bk'].mean(), low['acc_2bk'].mean()]
    sems_h = [high['acc_0bk'].sem(), high['acc_2bk'].sem()]
    sems_l = [low['acc_0bk'].sem(), low['acc_2bk'].sem()]
    x = np.array([0, 1])
    ax_c.fill_between(x, np.array(means_h) - np.array(sems_h),
                      np.array(means_h) + np.array(sems_h), alpha=0.2, color=HIGH_EFF)
    ax_c.fill_between(x, np.array(means_l) - np.array(sems_l),
                      np.array(means_l) + np.array(sems_l), alpha=0.2, color=LOW_EFF)
    ax_c.errorbar(x, means_h, yerr=sems_h, fmt='o-', color=HIGH_EFF,
                  markersize=11, linewidth=2.5, capsize=4, label=f'High Efficiency (n={n_high})')
    ax_c.errorbar(x, means_l, yerr=sems_l, fmt='s-', color=LOW_EFF,
                  markersize=11, linewidth=2.5, capsize=4, label=f'Low Efficiency (n={n_low})')
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(['0-back', '2-back'], fontsize=11, color=TEXT)
    ax_c.set_ylabel('Accuracy (M ± SEM)', fontsize=11, color=TEXT)
    ax_c.set_title('C', **title_kw)
    ax_c.legend(loc='lower left', fontsize=9, framealpha=0.9, edgecolor='none')
    ax_c.set_xlim(-0.3, 1.3)

    # D: Accuracy Cost
    def cost_bar(ax, h, l, ylabel, letter, ptitle):
        style_axis(ax)
        means = [h.mean(), l.mean()]
        sems = [h.sem(), l.sem()]
        ax.bar([0, 1], means, width=0.55, yerr=sems, capsize=5,
               color=[HIGH_EFF_LIGHT, LOW_EFF_LIGHT],
               edgecolor=[HIGH_EFF, LOW_EFF], linewidth=2)
        np.random.seed(42)
        for data, pos, col in [(h.values, 0, HIGH_EFF), (l.values, 1, LOW_EFF)]:
            jitter = np.random.normal(0, 0.06, len(data))
            ax.scatter(pos + jitter, data, s=22, alpha=0.55, c=col,
                       edgecolor='white', linewidth=0.4, zorder=5)
        ax.axhline(y=0, color=LINE, linestyle='--', linewidth=1, alpha=0.6)
        y_max = max(h.max(), l.max()); y_min = min(h.min(), l.min())
        add_stats(ax, h.values, l.values, y_max + 0.12 * (y_max - y_min))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['High Efficiency', 'Low Efficiency'], fontsize=10, color=TEXT)
        ax.set_ylabel(ylabel, fontsize=11, color=TEXT)
        ax.set_title(letter, **title_kw)

    ax_d = fig.add_subplot(gs[1, 0])
    cost_bar(ax_d, high['acc_cost'].dropna(), low['acc_cost'].dropna(),
             'Accuracy Cost (Δ)', 'D', 'Cognitive Load Cost')

    ax_e = fig.add_subplot(gs[1, 1])
    cost_bar(ax_e, high['rt_cost'].dropna(), low['rt_cost'].dropna(),
             'RT Cost (Δ ms)', 'E', 'Reaction Time Cost')

    # F: Individual Trajectories
    ax_f = fig.add_subplot(gs[1, 2])
    style_axis(ax_f)
    for idx in high.index:
        if pd.notna(high.loc[idx, 'acc_0bk']) and pd.notna(high.loc[idx, 'acc_2bk']):
            ax_f.plot([0, 1], [high.loc[idx, 'acc_0bk'], high.loc[idx, 'acc_2bk']],
                      color=HIGH_EFF, alpha=0.2, linewidth=1)
    for idx in low.index:
        if pd.notna(low.loc[idx, 'acc_0bk']) and pd.notna(low.loc[idx, 'acc_2bk']):
            ax_f.plot([0, 1], [low.loc[idx, 'acc_0bk'], low.loc[idx, 'acc_2bk']],
                      color=LOW_EFF, alpha=0.2, linewidth=1)
    ax_f.plot([0, 1], [high['acc_0bk'].mean(), high['acc_2bk'].mean()],
              color=HIGH_EFF, linewidth=3.5, marker='o', markersize=12,
              markeredgecolor='white', markeredgewidth=2, label=f'High Efficiency (n={n_high})')
    ax_f.plot([0, 1], [low['acc_0bk'].mean(), low['acc_2bk'].mean()],
              color=LOW_EFF, linewidth=3.5, marker='s', markersize=12,
              markeredgecolor='white', markeredgewidth=2, label=f'Low Efficiency (n={n_low})')
    ax_f.set_xticks([0, 1])
    ax_f.set_xticklabels(['0-back\n(Low Load)', '2-back\n(High Load)'], fontsize=11, color=TEXT)
    ax_f.set_ylabel('Accuracy', fontsize=11, color=TEXT)
    ax_f.set_title('F', **title_kw)
    ax_f.legend(loc='lower left', fontsize=9, framealpha=0.9, edgecolor='none')
    ax_f.set_xlim(-0.2, 1.2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig1.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'fig1.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig1.png")


# =====================================================================
# Figure 2: Glass Brain (load effect) - plain title, bigger colorbar
# =====================================================================
def make_fig2():
    print("Generating Figure 2 (glass brain)...")
    try:
        from nilearn import plotting, image
        import nibabel as nib
    except ImportError:
        print("  nilearn not available, keeping existing fig2.png")
        return

    activation_csv = RUN_DIR / 'activation' / 'roi_activation.csv'
    if not activation_csv.exists():
        print("  activation data not found, keeping existing fig2.png")
        return

    # Build a simple load-effect NIfTI for glass brain
    # We'll synthesize a volume with MNI coordinates of known ROIs
    from nilearn import datasets
    mni = datasets.load_mni152_template(resolution=2)
    arr = np.zeros(mni.shape, dtype=np.float32)

    df = pd.read_csv(activation_csv)
    rois = [c.replace('_activation_0bk', '') for c in df.columns if c.endswith('_activation_0bk')]

    # ROI MNI coords (matching configs)
    coords = {
        'DLPFC_L': (-46, 45, 20), 'DLPFC_R': (46, 45, 20),
        'VLPFC_L': (-52, 30, 2), 'VLPFC_R': (52, 30, 2),
        'PPC_L': (-40, -55, 45), 'PPC_R': (40, -55, 45),
        'ACC_L': (-5, 25, 35), 'ACC_R': (5, 25, 35),
        'mPFC': (0, 52, 10), 'PCC': (0, -50, 25),
        'Angular_L': (-45, -67, 30), 'Angular_R': (45, -67, 30),
        'Premotor_L': (-25, -10, 50), 'Premotor_R': (25, -10, 50),
    }

    affine = mni.affine
    from scipy.ndimage import gaussian_filter
    # Use a wide radius (distribute delta over a sphere) so blobs are visible
    for roi in rois:
        if roi not in coords:
            continue
        c0 = f'{roi}_activation_0bk'
        c2 = f'{roi}_activation_2bk'
        if c0 not in df.columns or c2 not in df.columns:
            continue
        delta = df[c2].mean() - df[c0].mean()
        # MNI -> voxel
        x, y, z = coords[roi]
        v = np.linalg.inv(affine) @ np.array([x, y, z, 1])
        vi, vj, vk = int(round(v[0])), int(round(v[1])), int(round(v[2]))
        # Paint a sphere of radius 6 voxels around the center
        rad = 6
        for di in range(-rad, rad + 1):
            for dj in range(-rad, rad + 1):
                for dk in range(-rad, rad + 1):
                    if di*di + dj*dj + dk*dk > rad*rad:
                        continue
                    ii, jj, kk = vi + di, vj + dj, vk + dk
                    if 0 <= ii < arr.shape[0] and 0 <= jj < arr.shape[1] and 0 <= kk < arr.shape[2]:
                        arr[ii, jj, kk] = max(arr[ii, jj, kk], delta) if delta > 0 else min(arr[ii, jj, kk], delta)
    arr = gaussian_filter(arr, sigma=2.5)
    img = nib.Nifti1Image(arr, affine)

    vmax_img = max(np.abs(arr).max(), 0.1)
    fig = plt.figure(figsize=(11, 4.5), facecolor='white')
    display = plotting.plot_glass_brain(
        img, display_mode='lyrz', colorbar=True,
        threshold=vmax_img * 0.05,
        cmap='RdBu_r', vmax=vmax_img, symmetric_cbar=True, figure=fig,
    )
    # Increase tick font on all axes (including colorbar)
    for ax in fig.axes:
        ax.tick_params(labelsize=14)
    # Plain black title (no background)
    fig.suptitle('Load Effect (2-back − 0-back)', fontsize=16, color='black', y=1.02)
    fig.savefig(OUT_DIR / 'fig2.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'fig2.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig2.png")


# =====================================================================
# Figure 3: Gap Reversal (remove panels C, F; remove main title; fix others)
# =====================================================================
def make_fig3():
    print("Generating Figure 3 (gap reversal)...")
    delta_df = pd.read_csv(RUN_DIR / 'delta_efficiency' / 'delta_efficiency.csv')
    behav = pd.read_csv(RUN_DIR / 'behavioral' / 'behavioral_summary.csv')
    network_df = pd.read_csv(RUN_DIR / 'connectivity' / 'network_metrics.csv')

    # Normalize groups
    if 'efficiency_group' not in delta_df.columns:
        delta_df = delta_df.merge(behav[['subject', 'efficiency_group']], on='subject', how='left')
    delta_df['efficiency_group'] = delta_df['efficiency_group'].replace({
        'High_Efficiency': 'High', 'Low_Efficiency': 'Low'})
    if 'efficiency_group' not in network_df.columns:
        network_df = network_df.merge(
            delta_df[['subject', 'efficiency_group']], on='subject', how='left')
    else:
        network_df['efficiency_group'] = network_df['efficiency_group'].replace({
            'High_Efficiency': 'High', 'Low_Efficiency': 'Low'})

    # Ensure required deltas exist
    for metric in ['global_efficiency', 'local_efficiency', 'modularity', 'mean_connectivity']:
        dcol = f'delta_{metric}'
        if dcol not in network_df.columns and f'{metric}_0bk' in network_df.columns:
            network_df[dcol] = network_df[f'{metric}_2bk'] - network_df[f'{metric}_0bk']

    high_mask = network_df['efficiency_group'] == 'High'
    low_mask = network_df['efficiency_group'] == 'Low'

    # 3 panels in a single row: A (trajectory), B (baseline modularity), C (Δ modularity).
    # Panel D (effect size forest) removed — redundant with Table V in the manuscript.
    fig = plt.figure(figsize=(18, 5.2), facecolor='white')
    gs = gridspec.GridSpec(1, 3, hspace=0.3, wspace=0.32, figure=fig)
    title_kw = dict(fontsize=14, fontweight='bold', color=TEXT, pad=14, loc='left')

    # ------ A: Trajectory (Gap Reversal) ------
    ax_a = fig.add_subplot(gs[0, 0])
    style_axis(ax_a)
    x0, x2 = 0, 1
    mh0 = network_df.loc[high_mask, 'modularity_0bk'].values
    mh2 = network_df.loc[high_mask, 'modularity_2bk'].values
    ml0 = network_df.loc[low_mask, 'modularity_0bk'].values
    ml2 = network_df.loc[low_mask, 'modularity_2bk'].values
    for a, b in zip(mh0, mh2):
        ax_a.plot([x0, x2], [a, b], color=HIGH_EFF, alpha=0.15, linewidth=0.8)
    for a, b in zip(ml0, ml2):
        ax_a.plot([x0, x2], [a, b], color=LOW_EFF, alpha=0.15, linewidth=0.8)
    ax_a.plot([x0, x2], [mh0.mean(), mh2.mean()],
              color=HIGH_EFF, linewidth=4, marker='o', markersize=13,
              markeredgecolor='white', markeredgewidth=2, label='High Efficiency')
    ax_a.plot([x0, x2], [ml0.mean(), ml2.mean()],
              color=LOW_EFF, linewidth=4, marker='s', markersize=13,
              markeredgecolor='white', markeredgewidth=2, label='Low Efficiency')
    ax_a.set_xticks([x0, x2])
    ax_a.set_xticklabels(['0-back\n(Low Load)', '2-back\n(High Load)'], fontsize=12, color=TEXT)
    ax_a.set_ylabel('Modularity', fontsize=12, color=TEXT)
    ax_a.set_title('A', **title_kw)
    ax_a.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='none')

    # ------ B: Overall (avg) modularity distribution by group — THE key significant result ------
    ax_b = fig.add_subplot(gs[0, 1])
    style_axis(ax_b)
    q_high = network_df.loc[high_mask, 'modularity'].dropna().values
    q_low = network_df.loc[low_mask, 'modularity'].dropna().values
    parts = ax_b.violinplot([q_high, q_low], positions=[0, 1], widths=0.6,
                             showmeans=False, showmedians=False, showextrema=False)
    for b, col, cl in zip(parts['bodies'], [HIGH_EFF, LOW_EFF], [HIGH_EFF_LIGHT, LOW_EFF_LIGHT]):
        b.set_facecolor(cl); b.set_alpha(0.7); b.set_edgecolor(col); b.set_linewidth(1.5)
    ax_b.boxplot([q_high, q_low], positions=[0, 1], widths=0.18, patch_artist=True,
                 boxprops=dict(facecolor='white', edgecolor='black', linewidth=1),
                 medianprops=dict(color='black', linewidth=2),
                 whiskerprops=dict(color='black'), capprops=dict(color='black'),
                 flierprops=dict(marker='', alpha=0))
    np.random.seed(7)
    for data, pos, col in [(q_high, 0, HIGH_EFF), (q_low, 1, LOW_EFF)]:
        jitter = np.random.normal(0, 0.04, len(data))
        ax_b.scatter(pos + jitter, data, s=20, alpha=0.45, c=col,
                     edgecolor='white', linewidth=0.3, zorder=5)
    from scipy import stats as _sps
    t, p = _sps.ttest_ind(q_high, q_low)
    cohen_d = (np.mean(q_high) - np.mean(q_low)) / np.sqrt((np.var(q_high) + np.var(q_low)) / 2)
    y_top = max(q_high.max(), q_low.max()) + abs(max(q_high.max(), q_low.max())) * 0.18
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ax_b.plot([0, 0, 1, 1], [y_top, y_top * 1.02, y_top * 1.02, y_top], 'k-', lw=1)
    ax_b.text(0.5, y_top * 1.03, f'{sig}  t = {t:.2f}, p = {p:.3f}, d = {cohen_d:.2f}',
              ha='center', va='bottom', fontsize=12, color=TEXT)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(['High Eff.', 'Low Eff.'], fontsize=12, color=TEXT)
    ax_b.set_ylabel('Modularity (avg. across loads)', fontsize=12, color=TEXT)
    ax_b.set_title('B', **title_kw)

    # ------ C (was D): Δ Distribution, clean title ------
    ax_c = fig.add_subplot(gs[0, 2])
    style_axis(ax_c)
    dh = network_df.loc[high_mask, 'delta_modularity'].dropna().values
    dl = network_df.loc[low_mask, 'delta_modularity'].dropna().values
    parts = ax_c.violinplot([dh, dl], positions=[0, 1], widths=0.6,
                            showmeans=False, showmedians=False, showextrema=False)
    for b, col, col_light in zip(parts['bodies'], [HIGH_EFF, LOW_EFF],
                                  [HIGH_EFF_LIGHT, LOW_EFF_LIGHT]):
        b.set_facecolor(col_light); b.set_alpha(0.7); b.set_edgecolor(col); b.set_linewidth(1.5)
    ax_c.boxplot([dh, dl], positions=[0, 1], widths=0.18, patch_artist=True,
                 boxprops=dict(facecolor='white', edgecolor='black', linewidth=1),
                 medianprops=dict(color='black', linewidth=2),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 flierprops=dict(marker='', alpha=0))
    np.random.seed(42)
    for data, pos, col in [(dh, 0, HIGH_EFF), (dl, 1, LOW_EFF)]:
        jitter = np.random.normal(0, 0.04, len(data))
        ax_c.scatter(pos + jitter, data, s=20, alpha=0.45, c=col,
                     edgecolor='white', linewidth=0.3, zorder=5)
    from scipy import stats as sps
    t, p = sps.ttest_ind(dh, dl)
    cohen_d = (np.mean(dh) - np.mean(dl)) / np.sqrt((np.var(dh) + np.var(dl)) / 2)
    y_top = max(dh.max(), dl.max()) + abs(max(dh.max(), dl.max())) * 0.12
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    ax_c.plot([0, 0, 1, 1], [y_top, y_top * 1.05, y_top * 1.05, y_top], 'k-', lw=1)
    ax_c.text(0.5, y_top * 1.08, f'{sig}  t = {t:.2f}, p = {p:.3f}, d = {cohen_d:.2f}',
              ha='center', va='bottom', fontsize=12, color=TEXT)
    ax_c.axhline(0, color=LINE, linestyle='--', linewidth=1, alpha=0.6)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(['High Eff.', 'Low Eff.'], fontsize=12, color=TEXT)
    ax_c.set_ylabel('Δ Modularity', fontsize=12, color=TEXT)
    ax_c.set_title('C', **title_kw)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig3.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'fig3.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig3.png")


# =====================================================================
# Figure 4: Radar chart (larger fonts, remove bottom hint)
# =====================================================================
def make_fig4():
    print("Generating Figure 4 (radar chart)...")
    delta_df = pd.read_csv(RUN_DIR / 'delta_efficiency' / 'delta_efficiency.csv')
    behav = pd.read_csv(RUN_DIR / 'behavioral' / 'behavioral_summary.csv')
    network_df = pd.read_csv(RUN_DIR / 'connectivity' / 'network_metrics.csv')

    if 'efficiency_group' not in delta_df.columns:
        delta_df = delta_df.merge(behav[['subject', 'efficiency_group']], on='subject', how='left')
    delta_df['efficiency_group'] = delta_df['efficiency_group'].replace({
        'High_Efficiency': 'High', 'Low_Efficiency': 'Low'})
    if 'efficiency_group' not in network_df.columns:
        network_df = network_df.merge(delta_df[['subject', 'efficiency_group']], on='subject', how='left')
    else:
        network_df['efficiency_group'] = network_df['efficiency_group'].replace({
            'High_Efficiency': 'High', 'Low_Efficiency': 'Low'})

    for m in ['global_efficiency', 'local_efficiency', 'modularity',
              'mean_connectivity', 'clustering_coefficient']:
        dcol = f'delta_{m}'
        if dcol not in network_df.columns and f'{m}_0bk' in network_df.columns:
            network_df[dcol] = network_df[f'{m}_2bk'] - network_df[f'{m}_0bk']

    # metrics to show
    metrics = ['modularity', 'global_efficiency', 'local_efficiency',
               'mean_connectivity', 'clustering_coefficient']
    labels = ['|Δ| Modularity', '|Δ| Global Efficiency', '|Δ| Local Efficiency',
              '|Δ| Mean Connectivity', '|Δ| Clustering Coeff.']

    high_mask = network_df['efficiency_group'] == 'High'
    low_mask = network_df['efficiency_group'] == 'Low'

    high_vals, low_vals = [], []
    for m in metrics:
        c = f'delta_{m}'
        if c not in network_df.columns:
            high_vals.append(0); low_vals.append(0); continue
        # Use absolute mean (magnitude of change)
        hv = np.nanmean(np.abs(network_df.loc[high_mask, c].values))
        lv = np.nanmean(np.abs(network_df.loc[low_mask, c].values))
        high_vals.append(hv); low_vals.append(lv)

    # Normalize so low efficiency is always 100%
    max_vals = [max(h, l) for h, l in zip(high_vals, low_vals)]
    high_norm = [100 * h / m if m > 0 else 0 for h, m in zip(high_vals, max_vals)]
    low_norm = [100 * l / m if m > 0 else 0 for l, m in zip(low_vals, max_vals)]

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    high_norm += high_norm[:1]
    low_norm += low_norm[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='white')
    ax.plot(angles, high_norm, color=HIGH_EFF, linewidth=2.5, label='High Efficiency')
    ax.fill(angles, high_norm, color=HIGH_EFF, alpha=0.25)
    ax.plot(angles, low_norm, color=LOW_EFF, linewidth=2.5, label='Low Efficiency')
    ax.fill(angles, low_norm, color=LOW_EFF, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16, color=TEXT)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=14, color=TEXT)
    ax.tick_params(axis='x', pad=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.22, 1.05), fontsize=15, frameon=False)
    ax.set_ylim(0, 110)
    # No main title, no bottom hint per user request
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig4.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_DIR / 'fig4.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig4.png")


# =====================================================================
# Figure 5: ML model comparison (larger fonts)
# =====================================================================
def make_fig5():
    print("Generating Figure 5 (ML comparison)...")
    with open(RUN_DIR / 'efficiency' / 'ai_classification_results.json') as f:
        ai = json.load(f)

    trad = ai['traditional']
    # Order by accuracy desc
    items = sorted(trad.items(), key=lambda kv: -kv[1]['accuracy'])
    names = [k for k, _ in items]
    accs = [v['accuracy'] for _, v in items]
    aucs = [v.get('auc', np.nan) for _, v in items]
    # Add GNN
    if ai.get('gnn'):
        names.append('GNN (GAT)')
        accs.append(ai['gnn']['accuracy'])
        aucs.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor='white')
    x = np.arange(len(names))

    # Accuracy bars
    colors = ['#2980b9'] * (len(names) - 1) + ['#95a5a6']
    bars = ax1.bar(x, accs, color=colors, edgecolor='black', linewidth=1)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=16, color=TEXT)
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label='Chance (0.5)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha='right', fontsize=15, color=TEXT)
    ax1.set_ylabel('Accuracy', fontsize=17, color=TEXT)
    ax1.set_title('A  Classification Accuracy', fontsize=18, fontweight='bold',
                  color=TEXT, pad=14, loc='left')
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=14, loc='upper right')
    style_axis(ax1)
    ax1.tick_params(labelsize=15)

    # AUC bars (skip nan)
    ax2.bar(x, [a if not np.isnan(a) else 0 for a in aucs], color=colors,
            edgecolor='black', linewidth=1)
    for i, v in enumerate(aucs):
        if not np.isnan(v):
            ax2.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=16, color=TEXT)
        else:
            ax2.text(i, 0.02, 'n/a', ha='center', fontsize=15, color='gray')
    ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label='Chance (0.5)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha='right', fontsize=15, color=TEXT)
    ax2.set_ylabel('AUC', fontsize=17, color=TEXT)
    ax2.set_title('B  Area Under ROC Curve', fontsize=18, fontweight='bold',
                  color=TEXT, pad=14, loc='left')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=14, loc='upper right')
    style_axis(ax2)
    ax2.tick_params(labelsize=15)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig5.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved fig5.png")


if __name__ == '__main__':
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print("\nAll paper figures regenerated.")
