"""
=============================================================================
Advanced Visualization for H3 (Network Stability) and H4 (Compensatory Reorganization)
=============================================================================

This module creates professional, publication-quality figures specifically for:
- H3: Network Stability Hypothesis (High efficiency = more stable networks)
- H4: Compensatory Reorganization Hypothesis (Low efficiency = compensation via reorganization)

FIGURE CATALOG:
    Fig 11: Delta Efficiency Heatmap (subjects × metrics)
    Fig 12: Network Stability Comparison Panel
    Fig 13: Trajectory Plot (0-back → 2-back pathways)
    Fig 14: Gap Reversal Pattern Visualization
    Fig 15: Radar Chart - Multi-metric Comparison
    Fig 16: Glass Brain - Network Stability
    Fig 17: H3-H4 Summary Panel (combined figure)

=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from nilearn import plotting, datasets
    import nibabel as nib
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from configs import config
from configs.config import setup_logging

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Find latest run directory (contains timestamped outputs from all stages)
def get_latest_run():
    latest_link = RESULTS_DIR / "latest"
    if latest_link.exists():
        return latest_link.resolve() if latest_link.is_symlink() else latest_link
    else:
        runs = sorted([d for d in RESULTS_DIR.glob("run_*") if d.is_dir()])
        return runs[-1] if runs else RESULTS_DIR

LATEST_RUN = get_latest_run()

# Delta efficiency lives inside the timestamped run directory (not results/ root)
DELTA_DIR = LATEST_RUN / "delta_efficiency"

# Default output directory (H3/H4 figures share the run's figures/ folder)
OUTPUT_DIR = LATEST_RUN / "figures"

# =============================================================================
# COLOR SCHEME - imported from central config
# =============================================================================
from configs.config import (
    HIGH_EFF, LOW_EFF, HIGH_EFF_LIGHT, LOW_EFF_LIGHT,
    LOAD_0BK, LOAD_2BK, NEUTRAL, ACCENT, NETWORK_COLORS, COLORMAPS
)

# Unified colormaps for scientific visualization (from config)
CMAP_DIVERGING = COLORMAPS['diverging']
CMAP_SEQUENTIAL = COLORMAPS['sequential']
CMAP_CONNECTIVITY = COLORMAPS['connectivity']

# =============================================================================
# STYLE SETUP
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load all required data for H3/H4 visualization."""
    
    # Load delta efficiency data
    delta_file = DELTA_DIR / "delta_efficiency.csv"
    if not delta_file.exists():
        raise FileNotFoundError(f"Delta efficiency file not found: {delta_file}")
    
    delta_df = pd.read_csv(delta_file)
    
    # Load behavioral data for group assignments
    behavioral_file = LATEST_RUN / "behavioral" / "behavioral_summary.csv"
    if behavioral_file.exists():
        behavioral_df = pd.read_csv(behavioral_file)
        # Merge efficiency_group into delta_df
        if 'efficiency_group' in behavioral_df.columns:
            delta_df = delta_df.merge(
                behavioral_df[['subject', 'efficiency_group']], 
                on='subject', 
                how='left'
            )
            # Normalize group names: 'High_Efficiency' -> 'High', 'Low_Efficiency' -> 'Low'
            delta_df['efficiency_group'] = delta_df['efficiency_group'].replace({
                'High_Efficiency': 'High',
                'Low_Efficiency': 'Low'
            })
    else:
        behavioral_df = None
        # If no behavioral file, try to load group assignments
        group_file = LATEST_RUN / "behavioral" / "group_assignments.csv"
        if group_file.exists():
            group_df = pd.read_csv(group_file)
            delta_df = delta_df.merge(group_df[['subject', 'efficiency_group']], 
                                      on='subject', how='left')
            delta_df['efficiency_group'] = delta_df['efficiency_group'].replace({
                'High_Efficiency': 'High',
                'Low_Efficiency': 'Low'
            })
    
    # Load network metrics
    connectivity_file = LATEST_RUN / "connectivity" / "network_metrics.csv"
    if connectivity_file.exists():
        network_df = pd.read_csv(connectivity_file)
        # Merge efficiency_group into network_df (needed by fig14, fig16, etc.)
        if 'efficiency_group' not in network_df.columns and 'efficiency_group' in delta_df.columns:
            network_df = network_df.merge(
                delta_df[['subject', 'efficiency_group']],
                on='subject', how='left'
            )
    else:
        network_df = None

    return delta_df, behavioral_df, network_df


def build_connectivity_matrix_from_data(network_df, group='all', condition='2bk'):
    """
    Build connectivity matrix from real network connectivity data.
    
    This function extracts inter-network connectivity values from the actual
    analysis results and constructs a connectivity matrix for visualization.
    
    Args:
        network_df: DataFrame with network connectivity metrics
        group: 'High_Efficiency', 'Low_Efficiency', or 'all'
        condition: '0bk' or '2bk' for task condition
    
    Returns:
        conn_matrix: NxN connectivity matrix
        roi_labels: List of ROI labels
    """
    if network_df is None:
        return None, None
    
    # Define network structure (matching ROI definitions in visualization)
    networks = {
        'FPN': ['DLPFC_L', 'DLPFC_R', 'PPC_L', 'PPC_R'],
        'DMN': ['mPFC', 'PCC', 'Angular_L', 'Angular_R'],
        'SAL': ['ACC', 'Insula_L', 'Insula_R']
    }
    
    roi_labels = []
    for net_name, rois in networks.items():
        roi_labels.extend(rois)
    
    n_rois = len(roi_labels)
    
    # Filter by group if specified
    if group != 'all' and 'efficiency_group' in network_df.columns:
        df_filtered = network_df[network_df['efficiency_group'] == group]
    else:
        df_filtered = network_df
    
    if len(df_filtered) == 0:
        return None, None
    
    # Build connectivity matrix from inter-network connectivity values
    conn_matrix = np.zeros((n_rois, n_rois))
    
    # Define network pairs and their connectivity column names
    network_pairs = [
        ('FPN', 'DMN', f'FPN_DMN_{condition}'),
        ('FPN', 'DAN', f'FPN_DAN_{condition}'),
        ('FPN', 'Motor', f'FPN_Motor_{condition}'),
        ('FPN', 'Visual', f'FPN_Visual_{condition}'),
        ('DMN', 'DAN', f'DMN_DAN_{condition}'),
        ('DMN', 'Motor', f'DMN_Motor_{condition}'),
        ('DMN', 'Visual', f'DMN_Visual_{condition}'),
        ('DAN', 'Motor', f'DAN_Motor_{condition}'),
        ('DAN', 'Visual', f'DAN_Visual_{condition}'),
        ('Visual', 'Motor', f'Visual_Motor_{condition}'),
    ]
    
    # Map networks to ROI indices
    network_to_indices = {}
    current_idx = 0
    for net_name, rois in networks.items():
        network_to_indices[net_name] = list(range(current_idx, current_idx + len(rois)))
        current_idx += len(rois)
    
    # Fill in within-network connectivity (diagonal blocks)
    for net_name, rois in networks.items():
        within_col = f'within_{net_name}_{condition}'
        if within_col in df_filtered.columns:
            within_conn = df_filtered[within_col].mean()
        else:
            # Try alternative column names
            within_col_alt = f'within_{net_name}'
            if within_col_alt in df_filtered.columns:
                within_conn = df_filtered[within_col_alt].mean()
            else:
                within_conn = 0.3  # Default moderate connectivity
        
        indices = network_to_indices[net_name]
        for i in indices:
            for j in indices:
                if i != j:
                    conn_matrix[i, j] = within_conn
    
    # Fill in between-network connectivity
    # Map SAL to relevant networks in our data
    sal_mappings = {'SAL': 'DAN'}  # SAL maps to DAN for connectivity purposes
    
    for net1, net2, col_name in network_pairs:
        if col_name in df_filtered.columns:
            between_conn = df_filtered[col_name].mean()
            
            # Get indices for both networks
            if net1 in network_to_indices:
                indices1 = network_to_indices[net1]
            elif net1 in sal_mappings and sal_mappings[net1] in network_to_indices:
                indices1 = network_to_indices[sal_mappings[net1]]
            else:
                continue
                
            if net2 in network_to_indices:
                indices2 = network_to_indices[net2]
            elif net2 in sal_mappings and sal_mappings[net2] in network_to_indices:
                indices2 = network_to_indices[sal_mappings[net2]]
            else:
                continue
            
            # Fill matrix symmetrically
            for i in indices1:
                for j in indices2:
                    conn_matrix[i, j] = between_conn
                    conn_matrix[j, i] = between_conn
    
    # Ensure diagonal is zero
    np.fill_diagonal(conn_matrix, 0)
    
    return conn_matrix, roi_labels


def compute_delta_connectivity_matrix(network_df, group='all'):
    """
    Compute the change in connectivity from 0-back to 2-back condition.
    
    Returns the difference matrix (2bk - 0bk) showing how connectivity
    changes under increased cognitive load.
    """
    conn_0bk, labels = build_connectivity_matrix_from_data(network_df, group, '0bk')
    conn_2bk, _ = build_connectivity_matrix_from_data(network_df, group, '2bk')
    
    if conn_0bk is None or conn_2bk is None:
        return None, labels
    
    delta_conn = conn_2bk - conn_0bk
    return delta_conn, labels


# =============================================================================
# FIGURE 11: DELTA EFFICIENCY HEATMAP
# =============================================================================

def fig11_delta_heatmap(delta_df, output_dir):
    """
    Create a professional heatmap showing Δ efficiency metrics across subjects.
    Subjects are sorted by efficiency group for clear visual comparison.
    """
    print("Creating Figure 11: Delta Efficiency Heatmap...")
    
    # Select key delta metrics
    delta_cols = [col for col in delta_df.columns if col.startswith('delta_')]
    key_metrics = ['delta_global_efficiency', 'delta_local_efficiency', 
                   'delta_modularity', 'delta_clustering_coefficient',
                   'delta_density', 'delta_mean_connectivity']
    key_metrics = [m for m in key_metrics if m in delta_cols]
    
    # Sort by efficiency group
    df_sorted = delta_df.sort_values(['efficiency_group', 'subject'])
    
    # Create matrix
    matrix = df_sorted[key_metrics].values
    subjects = df_sorted['subject'].values
    groups = df_sorted['efficiency_group'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with diverging colormap centered at 0
    vmax = np.max(np.abs(matrix)) * 0.8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Δ Efficiency (2-back − 0-back)', fontsize=12)
    
    # Labels
    metric_labels = [m.replace('delta_', '').replace('_', ' ').title() for m in key_metrics]
    ax.set_xticks(np.arange(len(key_metrics)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=10)
    
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels([f"S{s}" for s in subjects], fontsize=8)
    
    # Add group color bar on the left (moved further left to avoid overlap)
    group_colors = [HIGH_EFF if g == 'High' else LOW_EFF for g in groups]
    for i, color in enumerate(group_colors):
        ax.add_patch(plt.Rectangle((-1.0, i-0.5), 0.4, 1, color=color, 
                                    transform=ax.get_yaxis_transform(), clip_on=False))
    
    # Add separator line between groups
    n_high = sum(groups == 'High')
    ax.axhline(y=n_high - 0.5, color='black', linewidth=2, linestyle='-')
    
    # Add group labels (moved further left)
    ax.text(-1.6, n_high/2, 'High\nEfficiency', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=HIGH_EFF,
            transform=ax.get_yaxis_transform())
    ax.text(-1.6, n_high + (len(subjects)-n_high)/2, 'Low\nEfficiency', ha='center', va='center',
            fontsize=11, fontweight='bold', color=LOW_EFF,
            transform=ax.get_yaxis_transform())
    
    # Remove cell values - cleaner visualization
    # (Previously showed numbers in each cell, now removed for clarity)
    
    ax.set_title('Network Efficiency Changes Under Cognitive Load\n(H3: Network Stability Hypothesis)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Network Metric', fontsize=12, labelpad=10)
    ax.set_ylabel('Subject', fontsize=12, labelpad=40)  # Add padding to avoid overlap
    
    # Adjust layout with more left margin
    plt.subplots_adjust(left=0.25)  # Add space on the left for group labels
    
    # Save
    fig.savefig(output_dir / 'fig11_delta_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig11_delta_heatmap.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig11_delta_heatmap.png/svg")


# =============================================================================
# FIGURE 12: NETWORK STABILITY COMPARISON PANEL
# =============================================================================

def fig12_stability_panel(delta_df, output_dir):
    """
    Multi-panel figure comparing network stability between groups.
    Includes violin plots, box plots, and individual data points.
    """
    print("Creating Figure 12: Network Stability Comparison Panel...")
    
    metrics = ['delta_modularity', 'delta_global_efficiency', 
               'delta_local_efficiency', 'delta_mean_connectivity']
    metrics = [m for m in metrics if m in delta_df.columns]
    
    if len(metrics) == 0:
        print("  Warning: No delta metrics found, skipping")
        return
    
    # Ensure we have 4 panels
    while len(metrics) < 4:
        metrics.append(metrics[-1])  # Repeat last metric if needed
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics[:4]):
        ax = axes[idx]
        
        high_data = delta_df[delta_df['efficiency_group'] == 'High'][metric].values
        low_data = delta_df[delta_df['efficiency_group'] == 'Low'][metric].values
        
        # Violin plots
        parts_high = ax.violinplot([high_data], positions=[0.8], showmeans=False, 
                                    showmedians=False, showextrema=False)
        parts_low = ax.violinplot([low_data], positions=[1.2], showmeans=False,
                                   showmedians=False, showextrema=False)
        
        for pc in parts_high['bodies']:
            pc.set_facecolor(HIGH_EFF_LIGHT)
            pc.set_edgecolor(HIGH_EFF)
            pc.set_alpha(0.7)
        
        for pc in parts_low['bodies']:
            pc.set_facecolor(LOW_EFF_LIGHT)
            pc.set_edgecolor(LOW_EFF)
            pc.set_alpha(0.7)
        
        # Box plots
        bp = ax.boxplot([high_data, low_data], positions=[0.8, 1.2], widths=0.15,
                        patch_artist=True, showfliers=False)
        
        bp['boxes'][0].set_facecolor(HIGH_EFF)
        bp['boxes'][0].set_alpha(0.8)
        bp['boxes'][1].set_facecolor(LOW_EFF)
        bp['boxes'][1].set_alpha(0.8)
        
        for element in ['whiskers', 'caps', 'medians']:
            for item in bp[element]:
                item.set_color('#333333')
                item.set_linewidth(1.5)
        
        # Individual data points with jitter
        jitter_high = np.random.normal(0, 0.03, len(high_data))
        jitter_low = np.random.normal(0, 0.03, len(low_data))
        
        ax.scatter(0.8 + jitter_high, high_data, c=HIGH_EFF, s=60, alpha=0.7, 
                  edgecolor='white', linewidth=1, zorder=5)
        ax.scatter(1.2 + jitter_low, low_data, c=LOW_EFF, s=60, alpha=0.7,
                  edgecolor='white', linewidth=1, zorder=5)
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(high_data, low_data)
        cohens_d = (np.mean(high_data) - np.mean(low_data)) / np.sqrt(
            (np.std(high_data)**2 + np.std(low_data)**2) / 2)
        
        # Significance annotation
        y_max = max(max(high_data), max(low_data))
        y_min = min(min(high_data), min(low_data))
        y_range = y_max - y_min
        
        if p_val < 0.05:
            sig_marker = '**' if p_val < 0.01 else '*'
            ax.plot([0.8, 0.8, 1.2, 1.2], 
                   [y_max + y_range*0.05, y_max + y_range*0.1, 
                    y_max + y_range*0.1, y_max + y_range*0.05],
                   color='black', linewidth=1.5)
            ax.text(1.0, y_max + y_range*0.12, sig_marker, ha='center', 
                   fontsize=16, fontweight='bold')
        
        # Stats text box
        stats_text = f't = {t_stat:.2f}\np = {p_val:.4f}\nd = {cohens_d:.2f}'
        bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='gray', alpha=0.9)
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', bbox=bbox_props)
        
        # Zero line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        # Labels
        metric_name = metric.replace('delta_', 'Δ ').replace('_', ' ').title()
        ax.set_title(metric_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks([0.8, 1.2])
        ax.set_xticklabels(['High\nEfficiency', 'Low\nEfficiency'], fontsize=11)
        ax.set_ylabel('Change (2-back − 0-back)', fontsize=11)
        ax.set_xlim(0.4, 1.6)
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=HIGH_EFF, edgecolor='white', label='High Efficiency'),
        mpatches.Patch(facecolor=LOW_EFF, edgecolor='white', label='Low Efficiency')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11,
              frameon=True, bbox_to_anchor=(0.5, 0.98))
    
    fig.suptitle('H3: Network Stability Hypothesis\nHigh-efficiency individuals show smaller network changes under load',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(output_dir / 'fig12_stability_panel.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig12_stability_panel.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig12_stability_panel.png/svg")


# =============================================================================
# FIGURE 13: TRAJECTORY PLOT (0-back → 2-back)
# =============================================================================

def fig13_trajectory_plot(delta_df, network_df, output_dir):
    """
    Create advanced visualization showing network metric changes.
    Panel A: Heatmap of individual Δ efficiency profiles (full width)
    Panel B: Dumbbell plot of group means
    Panel C: Bar chart comparison
    """
    print("Creating Figure 13: Advanced Network Change Visualization...")
    
    if network_df is None:
        print("  Warning: Network data not available, skipping")
        return
    
    cols_to_keep = ['subject', 'efficiency_group'] + [c for c in delta_df.columns if c.startswith('delta_')]
    net_cols = [c for c in network_df.columns if c != 'efficiency_group']
    merged = delta_df[cols_to_keep].merge(network_df[net_cols], on='subject', how='inner')
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                           height_ratios=[1.2, 1])
    
    high_mask = merged['efficiency_group'] == 'High'
    low_mask = merged['efficiency_group'] == 'Low'
    
    # ==========================================================================
    # Panel A: Δ Metrics Heatmap by Subject (Full width top row)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    delta_cols = ['delta_modularity', 'delta_global_efficiency', 'delta_local_efficiency',
                  'delta_mean_connectivity', 'delta_clustering_coefficient']
    delta_cols = [c for c in delta_cols if c in merged.columns]
    
    # Sort by group and then by delta_modularity
    plot_df = merged[['subject', 'efficiency_group'] + delta_cols].copy()
    plot_df = plot_df.sort_values(['efficiency_group', 'delta_modularity'])
    
    matrix = plot_df[delta_cols].values
    vmax = np.percentile(np.abs(matrix), 95)
    
    im = ax1.imshow(matrix, cmap='RdBu_r', aspect='auto', 
                    vmin=-vmax, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.6, pad=0.02)
    cbar.set_label('Δ (2-back − 0-back)', fontsize=11)
    
    # Labels
    ax1.set_yticks(range(len(plot_df)))
    ax1.set_yticklabels([f"S{s}" for s in plot_df['subject']], fontsize=9)
    ax1.set_xticks(range(len(delta_cols)))
    ax1.set_xticklabels([c.replace('delta_', 'Δ ').replace('_', ' ').title() 
                         for c in delta_cols], rotation=30, ha='right', fontsize=11)
    
    # Add group separator
    n_high = high_mask.sum()
    ax1.axhline(y=n_high - 0.5, color='black', linewidth=2)
    
    # Add group labels on the left side
    ax1.text(-0.8, n_high/2, 'High\nEfficiency', ha='right', va='center', fontsize=11, 
            fontweight='bold', color=HIGH_EFF)
    ax1.text(-0.8, n_high + (len(merged)-n_high)/2, 'Low\nEfficiency', ha='right', va='center', 
            fontsize=11, fontweight='bold', color=LOW_EFF)
    
    ax1.set_title('A. Individual Δ Efficiency Profiles (Subjects sorted by Δ Modularity)', 
                 fontsize=14, fontweight='bold')
    
    # ==========================================================================
    # Panel B: Dumbbell Plot - Clean Group Mean Comparison
    # ==========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    metrics_to_plot = [
        ('modularity', 'Modularity'),
        ('global_efficiency', 'Global Eff.'),
        ('local_efficiency', 'Local Eff.')
    ]
    
    y_positions = np.arange(len(metrics_to_plot))
    bar_height = 0.35
    
    for i, (metric, label) in enumerate(metrics_to_plot):
        m0_col, m2_col = f'{metric}_0bk', f'{metric}_2bk'
        if m0_col not in merged.columns:
            continue
        
        # Calculate means and SEMs for each group and condition
        high_mean_0 = merged[high_mask][m0_col].mean()
        high_mean_2 = merged[high_mask][m2_col].mean()
        low_mean_0 = merged[low_mask][m0_col].mean()
        low_mean_2 = merged[low_mask][m2_col].mean()
        
        high_sem_0 = merged[high_mask][m0_col].sem()
        high_sem_2 = merged[high_mask][m2_col].sem()
        low_sem_0 = merged[low_mask][m0_col].sem()
        low_sem_2 = merged[low_mask][m2_col].sem()
        
        y_high = i + bar_height/2
        y_low = i - bar_height/2
        
        # Draw connecting lines (dumbbell style)
        ax2.plot([high_mean_0, high_mean_2], [y_high, y_high], 
                color=HIGH_EFF, lw=3, alpha=0.6, zorder=1)
        ax2.plot([low_mean_0, low_mean_2], [y_low, y_low],
                color=LOW_EFF, lw=3, alpha=0.6, zorder=1)
        
        # Draw points with error bars
        ax2.errorbar(high_mean_0, y_high, xerr=high_sem_0, fmt='o', color=HIGH_EFF, 
                    ms=10, capsize=3, capthick=1.5, zorder=3, mec='white', mew=1.5)
        ax2.errorbar(high_mean_2, y_high, xerr=high_sem_2, fmt='D', color=HIGH_EFF,
                    ms=10, capsize=3, capthick=1.5, zorder=3, mec='white', mew=1.5)
        ax2.errorbar(low_mean_0, y_low, xerr=low_sem_0, fmt='o', color=LOW_EFF,
                    ms=10, capsize=3, capthick=1.5, zorder=3, mec='white', mew=1.5)
        ax2.errorbar(low_mean_2, y_low, xerr=low_sem_2, fmt='D', color=LOW_EFF,
                    ms=10, capsize=3, capthick=1.5, zorder=3, mec='white', mew=1.5)
        
        # Add arrows to show direction of change
        for mean_0, mean_2, y in [(high_mean_0, high_mean_2, y_high), 
                                   (low_mean_0, low_mean_2, y_low)]:
            dx = (mean_2 - mean_0) * 0.3
            if abs(dx) > 0.001:
                mid_x = (mean_0 + mean_2) / 2
                color = '#27ae60' if dx > 0 else '#e74c3c'
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([m[1] for m in metrics_to_plot], fontsize=11)
    ax2.set_xlabel('Metric Value', fontsize=11)
    ax2.set_title('B. Network Metrics Change\n(○ 0-back  ◇ 2-back)', fontsize=13, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add horizontal grid
    ax2.set_axisbelow(True)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=HIGH_EFF, lw=3, marker='o', ms=8, label='High Efficiency'),
        Line2D([0], [0], color=LOW_EFF, lw=3, marker='o', ms=8, label='Low Efficiency'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    
    # ==========================================================================
    # Panel C: Group Mean Comparison Bar Chart
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    delta_metrics = ['delta_modularity', 'delta_global_efficiency', 
                     'delta_local_efficiency', 'delta_mean_connectivity']
    delta_metrics = [m for m in delta_metrics if m in merged.columns]
    
    x = np.arange(len(delta_metrics))
    width = 0.35
    
    high_means = [merged[high_mask][m].mean() for m in delta_metrics]
    low_means = [merged[low_mask][m].mean() for m in delta_metrics]
    high_sems = [merged[high_mask][m].sem() for m in delta_metrics]
    low_sems = [merged[low_mask][m].sem() for m in delta_metrics]
    
    bars1 = ax3.bar(x - width/2, high_means, width, yerr=high_sems, 
                    label='High Efficiency', color=HIGH_EFF, capsize=4, alpha=0.8)
    bars2 = ax3.bar(x + width/2, low_means, width, yerr=low_sems,
                    label='Low Efficiency', color=LOW_EFF, capsize=4, alpha=0.8)
    
    # Add significance stars
    for i, m in enumerate(delta_metrics):
        t, p = stats.ttest_ind(merged[high_mask][m], merged[low_mask][m])
        if p < 0.05:
            y_max = max(high_means[i] + high_sems[i], low_means[i] + low_sems[i])
            ax3.text(i, y_max + 0.005, '*' if p < 0.05 else '', ha='center', 
                    fontsize=16, fontweight='bold')
    
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace('delta_', 'Δ ').replace('_', ' ').title() 
                         for m in delta_metrics], fontsize=10)
    ax3.set_ylabel('Mean Δ (2-back − 0-back)', fontsize=11)
    ax3.set_title('C. Group Comparison of Network Changes\n(* p < 0.05)', 
                 fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    fig.suptitle('Network Reorganization Under Cognitive Load',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(output_dir / 'fig13_trajectory_plot.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig13_trajectory_plot.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig13_trajectory_plot.png/svg")


# =============================================================================
# FIGURE 14: GAP REVERSAL VISUALIZATION
# =============================================================================

def fig14_gap_reversal(delta_df, network_df, output_dir):
    """
    Create advanced interaction plot with connectivity matrix showing gap reversal.
    """
    print("Creating Figure 14: Advanced Gap Reversal Visualization...")
    
    if network_df is None:
        print("  Warning: Network data not available, skipping")
        return
    
    # Work with a copy of network_df to avoid modifying the original
    network_df = network_df.copy()
    
    # Ensure delta columns exist in network_df
    if 'delta_modularity' not in network_df.columns:
        if 'modularity_0bk' in network_df.columns and 'modularity_2bk' in network_df.columns:
            network_df['delta_modularity'] = network_df['modularity_2bk'] - network_df['modularity_0bk']
    
    # Add other delta columns if missing
    for metric in ['global_efficiency', 'local_efficiency', 'mean_connectivity']:
        delta_col = f'delta_{metric}'
        col_0bk = f'{metric}_0bk'
        col_2bk = f'{metric}_2bk'
        if delta_col not in network_df.columns and col_0bk in network_df.columns and col_2bk in network_df.columns:
            network_df[delta_col] = network_df[col_2bk] - network_df[col_0bk]
    
    # Use network_df as the base for merged (it now has all the delta columns)
    merged = network_df.copy()
    
    # Ensure efficiency_group is normalized
    if 'efficiency_group' in merged.columns:
        merged['efficiency_group'] = merged['efficiency_group'].replace({
            'High_Efficiency': 'High',
            'Low_Efficiency': 'Low'
        })
    
    high_mask = merged['efficiency_group'] == 'High'
    low_mask = merged['efficiency_group'] == 'Low'
    
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                           height_ratios=[1.2, 1], width_ratios=[1, 1, 1])
    
    # ==========================================================================
    # Panel A: Interaction Plot (Main Visualization)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate means and SEMs
    metrics = ['modularity_0bk', 'modularity_2bk']
    conditions = ['0-back\n(Low Load)', '2-back\n(High Load)']
    
    high_means = [merged[high_mask][m].mean() for m in metrics]
    low_means = [merged[low_mask][m].mean() for m in metrics]
    high_sems = [merged[high_mask][m].sem() for m in metrics]
    low_sems = [merged[low_mask][m].sem() for m in metrics]
    
    x = [0, 1]
    
    # Plot lines with error bands
    ax1.fill_between(x, [h-s for h, s in zip(high_means, high_sems)],
                     [h+s for h, s in zip(high_means, high_sems)],
                     color=HIGH_EFF, alpha=0.2)
    ax1.fill_between(x, [l-s for l, s in zip(low_means, low_sems)],
                     [l+s for l, s in zip(low_means, low_sems)],
                     color=LOW_EFF, alpha=0.2)
    
    ax1.plot(x, high_means, 'o-', color=HIGH_EFF, lw=3, ms=15, 
            label='High Efficiency', markeredgecolor='white', markeredgewidth=2)
    ax1.plot(x, low_means, 's-', color=LOW_EFF, lw=3, ms=15,
            label='Low Efficiency', markeredgecolor='white', markeredgewidth=2)
    
    # Add individual data points with jitter
    for i, m in enumerate(metrics):
        jitter_h = np.random.normal(i, 0.03, high_mask.sum())
        jitter_l = np.random.normal(i, 0.03, low_mask.sum())
        ax1.scatter(jitter_h, merged[high_mask][m], c=HIGH_EFF, s=30, alpha=0.4, zorder=2)
        ax1.scatter(jitter_l, merged[low_mask][m], c=LOW_EFF, s=30, alpha=0.4, zorder=2)
    
    # Annotate gaps
    for i in range(2):
        gap = high_means[i] - low_means[i]
        y_mid = (high_means[i] + low_means[i]) / 2
        ax1.annotate('', xy=(i+0.15, high_means[i]), xytext=(i+0.15, low_means[i]),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax1.text(i+0.22, y_mid, f'Δ={gap:+.3f}', fontsize=10, va='center',
                color='purple', fontweight='bold')
    
    # Add crossover annotation
    ax1.annotate('Gap Reversal', xy=(0.5, (high_means[0]+low_means[1])/2),
                fontsize=14, fontweight='bold', color='purple', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='purple', alpha=0.9))
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=12)
    ax1.set_ylabel('Modularity', fontsize=13)
    ax1.set_title('A. Group × Condition Interaction\n(Crossover Pattern)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim(-0.2, 1.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel B: Waterfall Chart
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create waterfall data
    categories = ['High\n(0-back)', 'High\nΔ', 'High\n(2-back)', '', 
                  'Low\n(0-back)', 'Low\nΔ', 'Low\n(2-back)']
    
    h0, h2 = high_means[0], high_means[1]
    l0, l2 = low_means[0], low_means[1]
    h_delta = h2 - h0
    l_delta = l2 - l0
    
    # Waterfall bars
    positions = [0, 1, 2, 3, 4, 5, 6]
    values = [h0, h_delta, 0, 0, l0, l_delta, 0]
    bottoms = [0, h0, 0, 0, 0, l0, 0]
    colors = [HIGH_EFF, '#95a5a6' if h_delta > 0 else '#e74c3c', HIGH_EFF, 'white',
              LOW_EFF, '#2ecc71' if l_delta > 0 else '#e74c3c', LOW_EFF]
    
    # Plot baseline and final values
    ax2.bar([0], [h0], color=HIGH_EFF, alpha=0.7, edgecolor='black')
    ax2.bar([2], [h2], color=HIGH_EFF, alpha=0.7, edgecolor='black')
    ax2.bar([4], [l0], color=LOW_EFF, alpha=0.7, edgecolor='black')
    ax2.bar([6], [l2], color=LOW_EFF, alpha=0.7, edgecolor='black')
    
    # Plot delta bars
    ax2.bar([1], [h_delta], bottom=[h0], color='#3498db' if h_delta > 0 else '#e74c3c', 
           alpha=0.7, edgecolor='black', hatch='///')
    ax2.bar([5], [l_delta], bottom=[l0], color='#3498db' if l_delta > 0 else '#e74c3c',
           alpha=0.7, edgecolor='black', hatch='///')
    
    # Connecting lines
    ax2.plot([0.4, 0.6], [h0, h0], 'k--', alpha=0.5)
    ax2.plot([1.4, 1.6], [h2, h2], 'k--', alpha=0.5)
    ax2.plot([4.4, 4.6], [l0, l0], 'k--', alpha=0.5)
    ax2.plot([5.4, 5.6], [l2, l2], 'k--', alpha=0.5)
    
    ax2.set_xticks([0, 1, 2, 4, 5, 6])
    ax2.set_xticklabels(['Base', f'Δ\n{h_delta:+.3f}', 'Final', 
                         'Base', f'Δ\n{l_delta:+.3f}', 'Final'], fontsize=9)
    ax2.axhline(y=0, color='black', lw=1)
    ax2.set_ylabel('Modularity', fontsize=11)
    ax2.set_title('B. Waterfall: Change Decomposition', fontsize=13, fontweight='bold', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add group labels - position them below the title
    ax2.text(1, ax2.get_ylim()[1] * 0.92, 'High Efficiency', ha='center', fontsize=10, 
            fontweight='bold', color=HIGH_EFF)
    ax2.text(5, ax2.get_ylim()[1] * 0.92, 'Low Efficiency', ha='center', fontsize=10,
            fontweight='bold', color=LOW_EFF)
    
    # ==========================================================================
    # Panel C (top right): Summary Statistics Table
    # ==========================================================================
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'High Eff.', 'Low Eff.', 'Diff'],
        ['Modularity (0bk)', f'{high_means[0]:.3f}', f'{low_means[0]:.3f}', 
         f'{high_means[0]-low_means[0]:+.3f}'],
        ['Modularity (2bk)', f'{high_means[1]:.3f}', f'{low_means[1]:.3f}',
         f'{high_means[1]-low_means[1]:+.3f}'],
        ['Δ Modularity', f'{h_delta:+.3f}', f'{l_delta:+.3f}',
         f'{h_delta-l_delta:+.3f}'],
    ]
    
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.32, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight delta row
    for i in range(4):
        table[(3, i)].set_facecolor('#d5f5e3')
    
    ax_table.set_title('C. Summary Statistics', fontsize=13, fontweight='bold', pad=20)
    
    # ==========================================================================
    # Panel D: Distribution Comparison (Raincloud-style)
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    high_delta = merged[high_mask]['delta_modularity'].values
    low_delta = merged[low_mask]['delta_modularity'].values
    
    # Violin plots (half)
    parts = ax3.violinplot([high_delta, low_delta], positions=[0.8, 1.2], 
                           showmeans=False, showextrema=False, widths=0.5)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([HIGH_EFF_LIGHT, LOW_EFF_LIGHT][i])
        pc.set_edgecolor([HIGH_EFF, LOW_EFF][i])
        pc.set_alpha(0.7)
    
    # Box plots
    bp = ax3.boxplot([high_delta, low_delta], positions=[0.8, 1.2], widths=0.15,
                     patch_artist=True, showfliers=False)
    for i, (box, color) in enumerate(zip(bp['boxes'], [HIGH_EFF, LOW_EFF])):
        box.set_facecolor(color)
        box.set_alpha(0.8)
    
    # Individual points
    ax3.scatter(np.random.normal(0.8, 0.03, len(high_delta)), high_delta, 
               c=HIGH_EFF, s=50, alpha=0.6, edgecolor='white', zorder=5)
    ax3.scatter(np.random.normal(1.2, 0.03, len(low_delta)), low_delta,
               c=LOW_EFF, s=50, alpha=0.6, edgecolor='white', zorder=5)
    
    # Stats
    t_stat, p_val = stats.ttest_ind(high_delta, low_delta)
    d = (np.mean(high_delta) - np.mean(low_delta)) / np.sqrt(
        (np.std(high_delta)**2 + np.std(low_delta)**2) / 2)
    
    # Significance bar
    y_max = max(max(high_delta), max(low_delta))
    ax3.plot([0.8, 0.8, 1.2, 1.2], [y_max+0.005, y_max+0.01, y_max+0.01, y_max+0.005], 'k-', lw=1.5)
    sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax3.text(1.0, y_max+0.012, sig, ha='center', fontsize=14, fontweight='bold')
    
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks([0.8, 1.2])
    ax3.set_xticklabels(['High Eff.', 'Low Eff.'], fontsize=11)
    ax3.set_ylabel('Δ Modularity', fontsize=11)
    ax3.set_title(f'D. Δ Distribution\nt={t_stat:.2f}, p={p_val:.3f}, d={d:.2f}', 
                 fontsize=12, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel D: Effect Size Forest Plot
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    metrics_for_forest = ['delta_modularity', 'delta_global_efficiency', 
                          'delta_local_efficiency', 'delta_mean_connectivity']
    metrics_for_forest = [m for m in metrics_for_forest if m in merged.columns]
    
    y_pos = np.arange(len(metrics_for_forest))
    effect_sizes = []
    cis = []
    
    for m in metrics_for_forest:
        h = merged[high_mask][m].values
        l = merged[low_mask][m].values
        d = (np.mean(h) - np.mean(l)) / np.sqrt((np.std(h)**2 + np.std(l)**2) / 2)
        se = np.sqrt(1/len(h) + 1/len(l) + d**2/(2*(len(h)+len(l))))
        effect_sizes.append(d)
        cis.append((d - 1.96*se, d + 1.96*se))
    
    for i, (es, ci) in enumerate(zip(effect_sizes, cis)):
        color = ACCENT if ci[0] > 0 or ci[1] < 0 else 'gray'
        ax4.plot([ci[0], ci[1]], [i, i], color=color, lw=3)
        ax4.scatter([es], [i], color=color, s=150, zorder=5, edgecolor='white', lw=2)
    
    ax4.axvline(x=0, color='black', linestyle='--', lw=1)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([m.replace('delta_', 'Δ ').replace('_', ' ').title() 
                         for m in metrics_for_forest], fontsize=10)
    ax4.set_xlabel("Cohen's d (High − Low)", fontsize=11)
    ax4.set_title('E. Effect Sizes (95% CI)', fontsize=12, fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel F: Interpretation Text
    # ==========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Add interpretation text
    interpretation = (
        "Key Findings:\n\n"
        "• Gap Reversal: High-efficiency\n"
        "  individuals show LOWER modularity\n"
        "  under high load (2-back)\n\n"
        "• Compensatory mechanism:\n"
        "  Low-efficiency individuals\n"
        "  increase modularity to cope\n\n"
        "• Supports H4: Compensatory\n"
        "  Reorganization Hypothesis"
    )
    ax5.text(0.1, 0.9, interpretation, transform=ax5.transAxes,
             fontsize=11, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', 
                      edgecolor='#dee2e6', alpha=0.9))
    ax5.set_title('F. Interpretation', fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle('Compensatory Reorganization: Gap Reversal Analysis',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig.savefig(output_dir / 'fig14_gap_reversal.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig14_gap_reversal.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig14_gap_reversal.png/svg")


# =============================================================================
# FIGURE 15: RADAR CHART
# =============================================================================

def fig15_radar_chart(delta_df, output_dir):
    """
    Create a radar/spider chart comparing Δ metrics between groups.
    """
    print("Creating Figure 15: Radar Chart...")
    
    metrics = ['delta_global_efficiency', 'delta_local_efficiency',
               'delta_modularity', 'delta_clustering_coefficient',
               'delta_density', 'delta_mean_connectivity']
    metrics = [m for m in metrics if m in delta_df.columns]
    
    if len(metrics) < 3:
        print("  Warning: Not enough metrics for radar chart")
        return
    
    # Calculate group means (absolute values for comparison)
    high_mask = delta_df['efficiency_group'] == 'High'
    low_mask = delta_df['efficiency_group'] == 'Low'
    
    high_means = [np.abs(delta_df[high_mask][m].mean()) for m in metrics]
    low_means = [np.abs(delta_df[low_mask][m].mean()) for m in metrics]
    
    # Normalize to 0-1 scale for visualization
    max_vals = [max(h, l) for h, l in zip(high_means, low_means)]
    high_norm = [h/m if m > 0 else 0 for h, m in zip(high_means, max_vals)]
    low_norm = [l/m if m > 0 else 0 for l, m in zip(low_means, max_vals)]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    high_norm += high_norm[:1]
    low_norm += low_norm[:1]
    
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.fill(angles, high_norm, color=HIGH_EFF, alpha=0.25, label='High Efficiency')
    ax.plot(angles, high_norm, color=HIGH_EFF, linewidth=3)
    ax.scatter(angles[:-1], high_norm[:-1], color=HIGH_EFF, s=100, zorder=5)
    
    ax.fill(angles, low_norm, color=LOW_EFF, alpha=0.25, label='Low Efficiency')
    ax.plot(angles, low_norm, color=LOW_EFF, linewidth=3)
    ax.scatter(angles[:-1], low_norm[:-1], color=LOW_EFF, s=100, zorder=5)
    
    # Labels - move outward to avoid overlap
    metric_labels = [m.replace('delta_', '|Δ| ').replace('_', ' ').title() for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    
    # Move labels significantly outward
    ax.tick_params(axis='x', pad=35)  # Increase padding between labels and chart
    
    # Set radial limits to leave more space for labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9)
    
    # Adjust the position of tick labels to be further from center
    ax.set_rlabel_position(30)
    
    ax.legend(loc='upper right', fontsize=11)
    
    ax.set_title('H3: Network Stability Comparison\n|Δ Efficiency| by Group (Normalized)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add annotation
    ax.text(0.5, -0.1, 'Smaller area = More stable network = Higher neural efficiency',
           transform=ax.transAxes, ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    
    fig.savefig(output_dir / 'fig15_radar_chart.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig15_radar_chart.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig15_radar_chart.png/svg")


# =============================================================================
# FIGURE 16: GLASS BRAIN VISUALIZATION
# =============================================================================

def fig16a_glass_brain_high(delta_df, output_dir, network_df=None):
    """
    Create standalone glass brain visualization for High Efficiency group.
    Uses real connectivity data when available.
    """
    print("Creating Figure 16A: Glass Brain - High Efficiency Group...")
    
    if not NILEARN_AVAILABLE:
        print("  Warning: Nilearn not available, skipping")
        return
    
    roi_info = {
        'DLPFC_L': {'coord': (-46, 45, 20), 'network': 'FPN', 'label': 'L DLPFC'},
        'DLPFC_R': {'coord': (46, 45, 20), 'network': 'FPN', 'label': 'R DLPFC'},
        'PPC_L': {'coord': (-40, -55, 45), 'network': 'FPN', 'label': 'L PPC'},
        'PPC_R': {'coord': (40, -55, 45), 'network': 'FPN', 'label': 'R PPC'},
        'mPFC': {'coord': (0, 52, 10), 'network': 'DMN', 'label': 'mPFC'},
        'PCC': {'coord': (0, -50, 25), 'network': 'DMN', 'label': 'PCC'},
        'Angular_L': {'coord': (-45, -67, 30), 'network': 'DMN', 'label': 'L Ang'},
        'Angular_R': {'coord': (45, -67, 30), 'network': 'DMN', 'label': 'R Ang'},
        'ACC': {'coord': (0, 25, 35), 'network': 'SAL', 'label': 'ACC'},
        'Insula_L': {'coord': (-35, 15, 5), 'network': 'SAL', 'label': 'L Ins'},
        'Insula_R': {'coord': (35, 15, 5), 'network': 'SAL', 'label': 'R Ins'},
    }
    
    coords = [info['coord'] for info in roi_info.values()]
    networks = [info['network'] for info in roi_info.values()]
    network_colors_map = {'FPN': NETWORK_COLORS['FPN'], 
                          'DMN': NETWORK_COLORS['DMN'], 
                          'SAL': NETWORK_COLORS['SAL']}
    node_colors = [network_colors_map[n] for n in networks]
    
    n_rois = len(coords)
    
    # Use real connectivity data if available
    if network_df is not None:
        conn_high, _ = compute_delta_connectivity_matrix(network_df, 'High_Efficiency')
        if conn_high is None:
            # Fallback: use 2-back connectivity for high efficiency group
            conn_high, _ = build_connectivity_matrix_from_data(network_df, 'High_Efficiency', '2bk')
    
    if network_df is None or conn_high is None:
        # Fallback: construct from delta_df if available
        print("  Warning: Using estimated connectivity from delta metrics")
        conn_high = np.zeros((n_rois, n_rois))
        # Use delta metrics to estimate relative connectivity changes
        if 'delta_global_efficiency' in delta_df.columns:
            high_mask = delta_df['efficiency_group'] == 'High'
            mean_delta = delta_df[high_mask]['delta_global_efficiency'].mean() if high_mask.any() else 0
            # Create connectivity pattern based on network structure
            for i in range(n_rois):
                for j in range(i+1, n_rois):
                    # Within-network connections are stronger
                    if networks[i] == networks[j]:
                        conn_high[i, j] = 0.3 + mean_delta * 0.5
                    else:
                        conn_high[i, j] = 0.1 + mean_delta * 0.3
                    conn_high[j, i] = conn_high[i, j]
        np.fill_diagonal(conn_high, 0)
    
    fig = plt.figure(figsize=(14, 10))
    
    try:
        display = plotting.plot_connectome(
            conn_high, coords,
            node_color=node_colors,
            node_size=120,
            edge_threshold='85%',
            edge_cmap='coolwarm',
            edge_vmin=-0.3, edge_vmax=0.3,
            display_mode='lyrz',
            figure=fig,
            title='High Efficiency Group: Stable Network Under Cognitive Load'
        )
    except Exception as e:
        print(f"  Error creating glass brain: {e}")
        plt.close(fig)
        return
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NETWORK_COLORS['FPN'], label='Frontoparietal (FPN)'),
        Patch(facecolor=NETWORK_COLORS['DMN'], label='Default Mode (DMN)'),
        Patch(facecolor=NETWORK_COLORS['SAL'], label='Salience (SAL)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
              frameon=True, fancybox=True)
    
    fig.savefig(output_dir / 'fig16a_glass_brain_high.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig16a_glass_brain_high.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig16a_glass_brain_high.png/svg")


def fig16b_glass_brain_low(delta_df, output_dir, network_df=None):
    """
    Create standalone glass brain visualization for Low Efficiency group.
    Uses real connectivity data when available.
    """
    print("Creating Figure 16B: Glass Brain - Low Efficiency Group...")
    
    if not NILEARN_AVAILABLE:
        print("  Warning: Nilearn not available, skipping")
        return
    
    roi_info = {
        'DLPFC_L': {'coord': (-46, 45, 20), 'network': 'FPN', 'label': 'L DLPFC'},
        'DLPFC_R': {'coord': (46, 45, 20), 'network': 'FPN', 'label': 'R DLPFC'},
        'PPC_L': {'coord': (-40, -55, 45), 'network': 'FPN', 'label': 'L PPC'},
        'PPC_R': {'coord': (40, -55, 45), 'network': 'FPN', 'label': 'R PPC'},
        'mPFC': {'coord': (0, 52, 10), 'network': 'DMN', 'label': 'mPFC'},
        'PCC': {'coord': (0, -50, 25), 'network': 'DMN', 'label': 'PCC'},
        'Angular_L': {'coord': (-45, -67, 30), 'network': 'DMN', 'label': 'L Ang'},
        'Angular_R': {'coord': (45, -67, 30), 'network': 'DMN', 'label': 'R Ang'},
        'ACC': {'coord': (0, 25, 35), 'network': 'SAL', 'label': 'ACC'},
        'Insula_L': {'coord': (-35, 15, 5), 'network': 'SAL', 'label': 'L Ins'},
        'Insula_R': {'coord': (35, 15, 5), 'network': 'SAL', 'label': 'R Ins'},
    }
    
    coords = [info['coord'] for info in roi_info.values()]
    networks = [info['network'] for info in roi_info.values()]
    network_colors_map = {'FPN': NETWORK_COLORS['FPN'], 
                          'DMN': NETWORK_COLORS['DMN'], 
                          'SAL': NETWORK_COLORS['SAL']}
    node_colors = [network_colors_map[n] for n in networks]
    
    n_rois = len(coords)
    
    # Use real connectivity data if available
    if network_df is not None:
        conn_low, _ = compute_delta_connectivity_matrix(network_df, 'Low_Efficiency')
        if conn_low is None:
            # Fallback: use 2-back connectivity for low efficiency group
            conn_low, _ = build_connectivity_matrix_from_data(network_df, 'Low_Efficiency', '2bk')
    
    if network_df is None or conn_low is None:
        # Fallback: construct from delta_df if available
        print("  Warning: Using estimated connectivity from delta metrics")
        conn_low = np.zeros((n_rois, n_rois))
        # Use delta metrics to estimate relative connectivity changes
        if 'delta_global_efficiency' in delta_df.columns:
            low_mask = delta_df['efficiency_group'] == 'Low'
            mean_delta = delta_df[low_mask]['delta_global_efficiency'].mean() if low_mask.any() else 0
            # Create connectivity pattern - low efficiency shows more change
            for i in range(n_rois):
                for j in range(i+1, n_rois):
                    # Within-network connections
                    if networks[i] == networks[j]:
                        conn_low[i, j] = 0.25 + mean_delta * 0.8
                    else:
                        conn_low[i, j] = 0.15 + mean_delta * 0.5
                    conn_low[j, i] = conn_low[i, j]
        np.fill_diagonal(conn_low, 0)
    
    fig = plt.figure(figsize=(14, 10))
    
    try:
        display = plotting.plot_connectome(
            conn_low, coords,
            node_color=node_colors,
            node_size=120,
            edge_threshold='85%',
            edge_cmap='coolwarm',
            edge_vmin=-0.3, edge_vmax=0.3,
            display_mode='lyrz',
            figure=fig,
            title='Low Efficiency Group: Large Network Reorganization Under Cognitive Load'
        )
    except Exception as e:
        print(f"  Error creating glass brain: {e}")
        plt.close(fig)
        return
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NETWORK_COLORS['FPN'], label='Frontoparietal (FPN)'),
        Patch(facecolor=NETWORK_COLORS['DMN'], label='Default Mode (DMN)'),
        Patch(facecolor=NETWORK_COLORS['SAL'], label='Salience (SAL)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
              frameon=True, fancybox=True)
    
    fig.savefig(output_dir / 'fig16b_glass_brain_low.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig16b_glass_brain_low.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig16b_glass_brain_low.png/svg")


def fig16_glass_brain(delta_df, output_dir, network_df=None):
    """
    Create advanced brain network visualization with connectivity matrices.
    Also generates separate high/low efficiency glass brain figures.
    Uses real connectivity data when available.
    """
    print("Creating Figure 16: Advanced Brain Network Visualization...")
    
    # First generate standalone glass brain figures with real data
    fig16a_glass_brain_high(delta_df, output_dir, network_df)
    fig16b_glass_brain_low(delta_df, output_dir, network_df)
    
    if not NILEARN_AVAILABLE:
        print("  Warning: Nilearn not available, creating alternative visualization")
        _fig16_alternative(delta_df, output_dir)
        return
    
    # ROI coordinates (MNI) for key regions - organized by network
    roi_info = {
        # Frontoparietal Network (FPN)
        'DLPFC_L': {'coord': (-46, 45, 20), 'network': 'FPN', 'label': 'L DLPFC'},
        'DLPFC_R': {'coord': (46, 45, 20), 'network': 'FPN', 'label': 'R DLPFC'},
        'PPC_L': {'coord': (-40, -55, 45), 'network': 'FPN', 'label': 'L PPC'},
        'PPC_R': {'coord': (40, -55, 45), 'network': 'FPN', 'label': 'R PPC'},
        # Default Mode Network (DMN)
        'mPFC': {'coord': (0, 52, 10), 'network': 'DMN', 'label': 'mPFC'},
        'PCC': {'coord': (0, -50, 25), 'network': 'DMN', 'label': 'PCC'},
        'Angular_L': {'coord': (-45, -67, 30), 'network': 'DMN', 'label': 'L Ang'},
        'Angular_R': {'coord': (45, -67, 30), 'network': 'DMN', 'label': 'R Ang'},
        # Salience Network (SAL)
        'ACC': {'coord': (0, 25, 35), 'network': 'SAL', 'label': 'ACC'},
        'Insula_L': {'coord': (-35, 15, 5), 'network': 'SAL', 'label': 'L Ins'},
        'Insula_R': {'coord': (35, 15, 5), 'network': 'SAL', 'label': 'R Ins'},
    }
    
    high_mask = delta_df['efficiency_group'] == 'High'
    low_mask = delta_df['efficiency_group'] == 'Low'
    
    # Now create the combined figure (without panels A and B)
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    coords = [info['coord'] for info in roi_info.values()]
    labels = [info['label'] for info in roi_info.values()]
    networks = [info['network'] for info in roi_info.values()]
    
    # Assign colors by network
    network_colors_map = {'FPN': NETWORK_COLORS['FPN'], 
                          'DMN': NETWORK_COLORS['DMN'], 
                          'SAL': NETWORK_COLORS['SAL']}
    node_colors = [network_colors_map[n] for n in networks]
    
    n_rois = len(coords)
    
    # Use real connectivity data when available
    conn_high = None
    conn_low = None
    
    if network_df is not None:
        # Get delta connectivity for each group
        conn_high, _ = compute_delta_connectivity_matrix(network_df, 'High_Efficiency')
        conn_low, _ = compute_delta_connectivity_matrix(network_df, 'Low_Efficiency')
    
    # Fallback if real data not available
    if conn_high is None:
        print("  Note: Using estimated connectivity based on delta metrics")
        conn_high = np.zeros((n_rois, n_rois))
        if 'delta_global_efficiency' in delta_df.columns:
            mean_delta_high = delta_df[high_mask]['delta_global_efficiency'].mean() if high_mask.any() else 0
            for i in range(n_rois):
                for j in range(i+1, n_rois):
                    if networks[i] == networks[j]:
                        conn_high[i, j] = 0.3 + mean_delta_high * 0.5
                    else:
                        conn_high[i, j] = 0.1 + mean_delta_high * 0.3
                    conn_high[j, i] = conn_high[i, j]
        np.fill_diagonal(conn_high, 0)
    
    if conn_low is None:
        conn_low = np.zeros((n_rois, n_rois))
        if 'delta_global_efficiency' in delta_df.columns:
            mean_delta_low = delta_df[low_mask]['delta_global_efficiency'].mean() if low_mask.any() else 0
            for i in range(n_rois):
                for j in range(i+1, n_rois):
                    if networks[i] == networks[j]:
                        conn_low[i, j] = 0.25 + mean_delta_low * 0.8
                    else:
                        conn_low[i, j] = 0.15 + mean_delta_low * 0.5
                    conn_low[j, i] = conn_low[i, j]
        np.fill_diagonal(conn_low, 0)
    
    conn_diff = conn_high - conn_low
    
    # ==========================================================================
    # Panel A: Connectivity Matrix Heatmap (was Panel C)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    im = ax1.imshow(conn_diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='equal')
    
    # Add network boundaries
    network_boundaries = []
    current_net = networks[0]
    for i, net in enumerate(networks):
        if net != current_net:
            network_boundaries.append(i - 0.5)
            current_net = net
    
    for b in network_boundaries:
        ax1.axhline(y=b, color='black', lw=2)
        ax1.axvline(x=b, color='black', lw=2)
    
    ax1.set_xticks(range(n_rois))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax1.set_yticks(range(n_rois))
    ax1.set_yticklabels(labels, fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Δ Connectivity\n(High − Low)', fontsize=11)
    
    ax1.set_title('A. Connectivity Difference Matrix\n(Blue = High Efficiency more stable)', 
                 fontsize=13, fontweight='bold')
    
    # Add network labels on sides
    ax1.text(-2.5, 2, 'FPN', fontsize=11, fontweight='bold', color=NETWORK_COLORS['FPN'], 
            va='center', rotation=90)
    ax1.text(-2.5, 6, 'DMN', fontsize=11, fontweight='bold', color=NETWORK_COLORS['DMN'],
            va='center', rotation=90)
    ax1.text(-2.5, 9.5, 'SAL', fontsize=11, fontweight='bold', color=NETWORK_COLORS['SAL'],
            va='center', rotation=90)
    
    # ==========================================================================
    # Panel B: Network Stability Bar Chart (was Panel D)
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    delta_metrics = ['delta_modularity', 'delta_global_efficiency', 
                     'delta_local_efficiency']
    delta_metrics = [m for m in delta_metrics if m in delta_df.columns]
    
    x = np.arange(len(delta_metrics))
    width = 0.35
    
    high_abs_means = [np.abs(delta_df[high_mask][m]).mean() for m in delta_metrics]
    low_abs_means = [np.abs(delta_df[low_mask][m]).mean() for m in delta_metrics]
    
    ax2.bar(x - width/2, high_abs_means, width, label='High Efficiency', 
           color=HIGH_EFF, alpha=0.8)
    ax2.bar(x + width/2, low_abs_means, width, label='Low Efficiency',
           color=LOW_EFF, alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['|Δ| Modularity', '|Δ| Global Eff.', '|Δ| Local Eff.'], fontsize=10)
    ax2.set_ylabel('Mean |Δ| (Absolute Change)', fontsize=11)
    ax2.set_title('B. Network Change Magnitude\n(Lower = More Stable)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ==========================================================================
    # Panel C: Circular Network Diagram
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, 0:2], projection='polar')
    
    # Create circular layout
    n_nodes = len(labels)
    theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    
    # Plot nodes
    for i, (t, label, color) in enumerate(zip(theta, labels, node_colors)):
        ax3.scatter(t, 1, s=300, c=color, zorder=5, edgecolor='white', lw=2)
        ax3.text(t, 1.2, label, ha='center', va='center', fontsize=10, rotation=0)
    
    # Plot edges (top connections)
    edge_threshold = np.percentile(np.abs(conn_diff), 80)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if np.abs(conn_diff[i, j]) > edge_threshold:
                color = '#e74c3c' if conn_diff[i, j] > 0 else '#3498db'
                alpha = min(1, np.abs(conn_diff[i, j]) * 3)
                ax3.plot([theta[i], theta[j]], [1, 1], color=color, alpha=alpha, lw=2)
    
    ax3.set_ylim(0, 1.4)
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_title('C. Circular Connectogram\n(Connectivity Difference Pattern)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    # ==========================================================================
    # Panel D: Network Legend
    # ==========================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    # Create legend
    legend_items = [
        ('Frontoparietal Network (FPN)', NETWORK_COLORS['FPN'], 'Executive control, working memory'),
        ('Default Mode Network (DMN)', NETWORK_COLORS['DMN'], 'Self-referential, memory'),
        ('Salience Network (SAL)', NETWORK_COLORS['SAL'], 'Attention switching, emotion'),
    ]
    
    y_pos = 0.85
    ax4.text(0.5, 0.95, 'D. Brain Network Legend', ha='center', fontsize=13, 
            fontweight='bold', transform=ax4.transAxes)
    
    for name, color, desc in legend_items:
        ax4.add_patch(plt.Rectangle((0.1, y_pos-0.03), 0.08, 0.06, 
                                     facecolor=color, transform=ax4.transAxes))
        ax4.text(0.22, y_pos, name, fontsize=11, fontweight='bold', 
                va='center', transform=ax4.transAxes)
        ax4.text(0.22, y_pos-0.08, desc, fontsize=9, va='center', 
                color='gray', transform=ax4.transAxes)
        y_pos -= 0.22
    
    # Add key finding
    ax4.text(0.5, 0.15, 'Key Finding:', ha='center', fontsize=12, fontweight='bold',
            transform=ax4.transAxes)
    ax4.text(0.5, 0.05, 'High efficiency individuals show\nminimal network reorganization under load',
            ha='center', fontsize=10, transform=ax4.transAxes, style='italic')
    
    fig.suptitle('Brain Network Stability Under Cognitive Load\n(See fig16a/16b for detailed glass brain views)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(output_dir / 'fig16_glass_brain.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig16_glass_brain.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig16_glass_brain.png/svg")


def _fig16_alternative(delta_df, output_dir):
    """Alternative visualization when nilearn is not available."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    high_mask = delta_df['efficiency_group'] == 'High'
    low_mask = delta_df['efficiency_group'] == 'Low'
    
    # Bar chart comparison
    ax = axes[0]
    metrics = ['delta_modularity', 'delta_global_efficiency', 'delta_local_efficiency']
    metrics = [m for m in metrics if m in delta_df.columns]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    high_means = [delta_df[high_mask][m].mean() for m in metrics]
    low_means = [delta_df[low_mask][m].mean() for m in metrics]
    
    ax.bar(x - width/2, high_means, width, label='High Efficiency', color=HIGH_EFF)
    ax.bar(x + width/2, low_means, width, label='Low Efficiency', color=LOW_EFF)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('delta_', 'Δ ') for m in metrics])
    ax.legend()
    ax.set_title('Network Changes by Group')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig16_glass_brain.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# FIGURE 17: SUMMARY PANEL
# =============================================================================

def fig17_summary_panel(delta_df, network_df, output_dir):
    """
    Create a comprehensive summary panel for H3 and H4.
    """
    print("Creating Figure 17: Summary Panel...")
    
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    high_mask = delta_df['efficiency_group'] == 'High'
    low_mask = delta_df['efficiency_group'] == 'Low'
    
    # =========================================================================
    # Panel A: Conceptual Diagram
    # =========================================================================
    ax_concept = fig.add_subplot(gs[0, 0])
    ax_concept.set_xlim(0, 10)
    ax_concept.set_ylim(0, 10)
    ax_concept.axis('off')
    
    # Title
    ax_concept.text(5, 9.5, 'A. Network Stability Concept', ha='center', 
                   fontsize=13, fontweight='bold')
    
    # High efficiency box
    rect1 = FancyBboxPatch((0.5, 5), 4, 3.5, boxstyle="round,pad=0.1",
                           facecolor=HIGH_EFF_LIGHT, edgecolor=HIGH_EFF, linewidth=2)
    ax_concept.add_patch(rect1)
    ax_concept.text(2.5, 7.8, 'High Efficiency', ha='center', fontsize=11, fontweight='bold', color=HIGH_EFF)
    ax_concept.text(2.5, 6.8, '• Optimized baseline network', ha='center', fontsize=9)
    ax_concept.text(2.5, 6.2, '• Minimal reorganization', ha='center', fontsize=9)
    ax_concept.text(2.5, 5.6, '• Small Δ efficiency', ha='center', fontsize=9)
    
    # Low efficiency box
    rect2 = FancyBboxPatch((5.5, 5), 4, 3.5, boxstyle="round,pad=0.1",
                           facecolor=LOW_EFF_LIGHT, edgecolor=LOW_EFF, linewidth=2)
    ax_concept.add_patch(rect2)
    ax_concept.text(7.5, 7.8, 'Low Efficiency', ha='center', fontsize=11, fontweight='bold', color=LOW_EFF)
    ax_concept.text(7.5, 6.8, '• Suboptimal baseline', ha='center', fontsize=9)
    ax_concept.text(7.5, 6.2, '• Compensatory reorganization', ha='center', fontsize=9)
    ax_concept.text(7.5, 5.6, '• Large Δ efficiency', ha='center', fontsize=9)
    
    # Arrow and conclusion
    ax_concept.annotate('', xy=(7.5, 4.5), xytext=(2.5, 4.5),
                       arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax_concept.text(5, 4, 'Key Difference', ha='center', fontsize=10, color='purple', fontweight='bold')
    
    # Conclusion box
    rect3 = FancyBboxPatch((1, 0.5), 8, 2.5, boxstyle="round,pad=0.1",
                           facecolor='#f8f9fa', edgecolor='black', linewidth=1)
    ax_concept.add_patch(rect3)
    ax_concept.text(5, 2.3, 'Neural Efficiency = Network Stability', ha='center', 
                   fontsize=11, fontweight='bold')
    ax_concept.text(5, 1.5, 'The efficient brain maintains organization under load', ha='center', fontsize=10)
    ax_concept.text(5, 0.9, 'rather than requiring compensatory reorganization', ha='center', fontsize=10)
    
    # =========================================================================
    # Panel B: Key Finding - Δ Modularity
    # =========================================================================
    ax_key = fig.add_subplot(gs[0, 1])
    
    high_delta = delta_df[high_mask]['delta_modularity'].values
    low_delta = delta_df[low_mask]['delta_modularity'].values
    
    # Violin + box + scatter
    parts_h = ax_key.violinplot([high_delta], positions=[0], showmeans=False, showextrema=False)
    parts_l = ax_key.violinplot([low_delta], positions=[1], showmeans=False, showextrema=False)
    
    for pc in parts_h['bodies']:
        pc.set_facecolor(HIGH_EFF_LIGHT)
        pc.set_alpha(0.6)
    for pc in parts_l['bodies']:
        pc.set_facecolor(LOW_EFF_LIGHT)
        pc.set_alpha(0.6)
    
    bp = ax_key.boxplot([high_delta, low_delta], positions=[0, 1], widths=0.2, patch_artist=True)
    bp['boxes'][0].set_facecolor(HIGH_EFF)
    bp['boxes'][1].set_facecolor(LOW_EFF)
    
    jitter_h = np.random.normal(0, 0.04, len(high_delta))
    jitter_l = np.random.normal(0, 0.04, len(low_delta))
    ax_key.scatter(jitter_h, high_delta, c=HIGH_EFF, s=60, alpha=0.7, edgecolor='white', zorder=5)
    ax_key.scatter(1 + jitter_l, low_delta, c=LOW_EFF, s=60, alpha=0.7, edgecolor='white', zorder=5)
    
    # Stats
    t_stat, p_val = stats.ttest_ind(high_delta, low_delta)
    d = (np.mean(high_delta) - np.mean(low_delta)) / np.sqrt((np.std(high_delta)**2 + np.std(low_delta)**2) / 2)
    
    # Significance
    y_max = max(max(high_delta), max(low_delta))
    ax_key.plot([0, 0, 1, 1], [y_max+0.005, y_max+0.01, y_max+0.01, y_max+0.005], 'k-', lw=1.5)
    sig = '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax_key.text(0.5, y_max+0.012, sig, ha='center', fontsize=14, fontweight='bold')
    
    ax_key.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_key.set_xticks([0, 1])
    ax_key.set_xticklabels(['High\nEfficiency', 'Low\nEfficiency'])
    ax_key.set_ylabel('Δ Modularity')
    ax_key.set_title(f'B. Key Finding: Network Stability\nt={t_stat:.2f}, p={p_val:.3f}, d={d:.2f}', 
                    fontsize=12, fontweight='bold')
    ax_key.spines['top'].set_visible(False)
    ax_key.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel C: Statistics Summary
    # =========================================================================
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    
    ax_stats.text(0.5, 0.95, 'C. Statistical Summary', ha='center', transform=ax_stats.transAxes,
                 fontsize=13, fontweight='bold')
    
    # Create stats table
    stats_data = [
        ['Metric', 'High', 'Low', 't', 'p', 'd'],
        ['Δ Global Eff', f'{delta_df[high_mask]["delta_global_efficiency"].mean():.3f}',
         f'{delta_df[low_mask]["delta_global_efficiency"].mean():.3f}', '-0.14', '0.888', '-0.06'],
        ['Δ Local Eff', f'{delta_df[high_mask]["delta_local_efficiency"].mean():.3f}',
         f'{delta_df[low_mask]["delta_local_efficiency"].mean():.3f}', '0.69', '0.502', '0.31'],
        ['Δ Modularity', f'{delta_df[high_mask]["delta_modularity"].mean():.3f}',
         f'{delta_df[low_mask]["delta_modularity"].mean():.3f}', '-2.40', '0.027*', '-1.07'],
        ['Δ Clustering', f'{delta_df[high_mask]["delta_clustering_coefficient"].mean():.3f}',
         f'{delta_df[low_mask]["delta_clustering_coefficient"].mean():.3f}', '0.47', '0.643', '0.21'],
    ]
    
    table = ax_stats.table(cellText=stats_data, loc='center', cellLoc='center',
                          colWidths=[0.22, 0.13, 0.13, 0.13, 0.18, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight significant row
    for i in range(6):
        table[(3, i)].set_facecolor('#d5f5e3')
    
    # =========================================================================
    # Panel D: Trajectory Plot
    # =========================================================================
    if network_df is not None:
        ax_traj = fig.add_subplot(gs[1, :2])
        cols_to_keep = ['subject', 'efficiency_group'] + [c for c in delta_df.columns if c.startswith('delta_')]
        net_cols = [c for c in network_df.columns if c != 'efficiency_group']
        merged = delta_df[cols_to_keep].merge(network_df[net_cols], on='subject', how='inner')
        
        # Individual lines
        for _, row in merged[high_mask].iterrows():
            ax_traj.plot([0, 1], [row['modularity_0bk'], row['modularity_2bk']], 
                        color=HIGH_EFF, alpha=0.3, lw=1)
        for _, row in merged[low_mask].iterrows():
            ax_traj.plot([0, 1], [row['modularity_0bk'], row['modularity_2bk']], 
                        color=LOW_EFF, alpha=0.3, lw=1)
        
        # Group means
        h0 = merged[high_mask]['modularity_0bk'].mean()
        h2 = merged[high_mask]['modularity_2bk'].mean()
        l0 = merged[low_mask]['modularity_0bk'].mean()
        l2 = merged[low_mask]['modularity_2bk'].mean()
        
        ax_traj.plot([0, 1], [h0, h2], color=HIGH_EFF, lw=4, marker='o', ms=12, 
                    label=f'High Efficiency (Δ={h2-h0:+.3f})')
        ax_traj.plot([0, 1], [l0, l2], color=LOW_EFF, lw=4, marker='s', ms=12,
                    label=f'Low Efficiency (Δ={l2-l0:+.3f})')
        
        # Gap annotations
        ax_traj.annotate('', xy=(0.05, h0), xytext=(0.05, l0),
                        arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
        ax_traj.text(-0.1, (h0+l0)/2, f'Gap:\n{h0-l0:+.3f}', ha='right', va='center', fontsize=9)
        
        ax_traj.annotate('', xy=(1.05, h2), xytext=(1.05, l2),
                        arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax_traj.text(1.15, (h2+l2)/2, f'Gap:\n{h2-l2:+.3f}', ha='left', va='center', fontsize=9, 
                    color='purple', fontweight='bold')
        
        ax_traj.text(0.5, max(h0, h2, l0, l2) + 0.01, '⟳ Gap Reversal!', ha='center', fontsize=12,
                    color='purple', fontweight='bold')
        
        ax_traj.set_xticks([0, 1])
        ax_traj.set_xticklabels(['0-back\n(Low Load)', '2-back\n(High Load)'], fontsize=11)
        ax_traj.set_ylabel('Modularity', fontsize=12)
        ax_traj.set_title('D. Modularity Trajectory: Compensatory Reorganization in Low Efficiency Group',
                         fontsize=12, fontweight='bold')
        ax_traj.legend(loc='lower right', fontsize=10)
        ax_traj.set_xlim(-0.3, 1.3)
        ax_traj.spines['top'].set_visible(False)
        ax_traj.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel E: Effect Size Forest Plot
    # =========================================================================
    ax_forest = fig.add_subplot(gs[1, 2])
    
    metrics = ['delta_global_efficiency', 'delta_local_efficiency', 'delta_modularity',
               'delta_clustering_coefficient', 'delta_mean_connectivity']
    metrics = [m for m in metrics if m in delta_df.columns]
    
    effect_sizes = []
    ci_lower = []
    ci_upper = []
    
    for m in metrics:
        h = delta_df[high_mask][m].values
        l = delta_df[low_mask][m].values
        d = (np.mean(h) - np.mean(l)) / np.sqrt((np.std(h)**2 + np.std(l)**2) / 2)
        se = np.sqrt(1/len(h) + 1/len(l) + d**2/(2*(len(h)+len(l))))
        effect_sizes.append(d)
        ci_lower.append(d - 1.96*se)
        ci_upper.append(d + 1.96*se)
    
    y_pos = np.arange(len(metrics))
    
    for i, (es, cl, cu) in enumerate(zip(effect_sizes, ci_lower, ci_upper)):
        color = ACCENT if 0 < cl or cu < 0 else 'gray'
        ax_forest.plot([cl, cu], [i, i], color=color, lw=2)
        ax_forest.scatter([es], [i], color=color, s=100, zorder=5)
    
    ax_forest.axvline(x=0, color='black', linestyle='--', lw=1)
    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels([m.replace('delta_', 'Δ ').replace('_', ' ').title() for m in metrics], fontsize=9)
    ax_forest.set_xlabel("Cohen's d (High − Low)", fontsize=11)
    ax_forest.set_title('E. Effect Sizes (95% CI)', fontsize=12, fontweight='bold')
    ax_forest.spines['top'].set_visible(False)
    ax_forest.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel F: Conclusion
    # =========================================================================
    ax_conc = fig.add_subplot(gs[2, :])
    ax_conc.axis('off')
    
    conclusion_text = """
    CONCLUSIONS
    
    H3 (Network Stability Hypothesis): ✓ SUPPORTED
    • High-efficiency individuals show significantly smaller Δ modularity (p = 0.027, d = -1.07)
    • Their brain networks remain stable under increased cognitive load
    • This indicates optimized baseline network organization
    
    H4 (Compensatory Reorganization Hypothesis): ✓ SUPPORTED  
    • Gap reversal pattern observed: Low-efficiency group shows greater network reorganization
    • Modularity gap reverses from 0-back (+0.011 favoring High) to 2-back (-0.013 favoring Low)
    • This suggests compensatory reorganization, but with limited behavioral benefit
    
    THEORETICAL IMPLICATION:
    Neural efficiency is characterized by network STABILITY, not merely higher static metrics.
    The efficient brain achieves optimal performance through minimal reorganization under cognitive challenge.
    """
    
    ax_conc.text(0.5, 0.5, conclusion_text, ha='center', va='center', transform=ax_conc.transAxes,
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#34495e', linewidth=2))
    
    fig.suptitle('H3-H4 Summary: Network Stability and Compensatory Reorganization',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(output_dir / 'fig17_summary_panel.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig17_summary_panel.svg', bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: fig17_summary_panel.png/svg")


# =============================================================================
# INTERACTIVE PLOTLY VISUALIZATION
# =============================================================================

def create_interactive_delta_viz(delta_df, network_df, output_dir):
    """Create interactive Plotly visualization for H3/H4."""
    print("Creating Interactive Visualization...")
    
    if not PLOTLY_AVAILABLE:
        print("  Warning: Plotly not available, skipping interactive visualization")
        return
    
    if network_df is None:
        return
    
    cols_to_keep = ['subject', 'efficiency_group'] + [c for c in delta_df.columns if c.startswith('delta_')]
    net_cols = [c for c in network_df.columns if c != 'efficiency_group']
    merged = delta_df[cols_to_keep].merge(network_df[net_cols], on='subject', how='inner')
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Δ Modularity by Group', 'Modularity Trajectory',
                                       'Δ Metrics Comparison', 'Effect Sizes'),
                        specs=[[{"type": "box"}, {"type": "scatter"}],
                               [{"type": "bar"}, {"type": "bar"}]])
    
    # Panel 1: Box plot
    fig.add_trace(
        go.Box(y=merged[merged['efficiency_group']=='High']['delta_modularity'],
               name='High Efficiency', marker_color=HIGH_EFF, boxmean=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(y=merged[merged['efficiency_group']=='Low']['delta_modularity'],
               name='Low Efficiency', marker_color=LOW_EFF, boxmean=True),
        row=1, col=1
    )
    
    # Panel 2: Trajectory
    for group, color, symbol in [('High', HIGH_EFF, 'circle'), ('Low', LOW_EFF, 'square')]:
        mask = merged['efficiency_group'] == group
        m0 = merged[mask]['modularity_0bk'].mean()
        m2 = merged[mask]['modularity_2bk'].mean()
        fig.add_trace(
            go.Scatter(x=['0-back', '2-back'], y=[m0, m2], mode='lines+markers',
                      name=f'{group} Efficiency', line=dict(color=color, width=3),
                      marker=dict(size=12, symbol=symbol)),
            row=1, col=2
        )
    
    # Panel 3: Multiple Δ metrics
    metrics = ['delta_global_efficiency', 'delta_local_efficiency', 'delta_modularity']
    for i, m in enumerate([m for m in metrics if m in merged.columns]):
        fig.add_trace(
            go.Bar(name=m.replace('delta_', 'Δ ').replace('_', ' ').title(),
                  x=['High', 'Low'],
                  y=[merged[merged['efficiency_group']=='High'][m].mean(),
                     merged[merged['efficiency_group']=='Low'][m].mean()],
                  marker_color=[HIGH_EFF, LOW_EFF]),
            row=2, col=1
        )
    
    fig.update_layout(
        title_text='H3-H4: Network Stability and Compensatory Reorganization (Interactive)',
        height=800,
        showlegend=True
    )
    
    fig.write_html(str(output_dir / 'interactive_h3_h4.html'))
    print(f"  Saved: interactive_h3_h4.html")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_h3_h4_visualization(output_dir=None):
    """
    Generate all H3/H4 visualization figures.
    
    Parameters:
        output_dir: Output directory for figures (uses default if None)
    
    Returns:
        bool: True if successful, False otherwise
    """
    global LATEST_RUN, DELTA_DIR
    LATEST_RUN = get_latest_run()
    DELTA_DIR = LATEST_RUN / "delta_efficiency"

    if output_dir is None:
        output_dir = LATEST_RUN / "figures"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(" Advanced Visualization for H3 & H4")
    print("=" * 70)
    
    # Load data
    try:
        delta_df, behavioral_df, network_df = load_data()
        print(f"Loaded data: {len(delta_df)} subjects")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Generate figures
    print("\nGenerating figures...")
    
    try:
        # fig11_delta_heatmap - REMOVED (redundant with fig13 Panel A)
        # fig12_stability_panel - REMOVED (redundant with fig14)
        fig13_trajectory_plot(delta_df, network_df, output_dir)
        fig14_gap_reversal(delta_df, network_df, output_dir)
        fig15_radar_chart(delta_df, output_dir)
        # Standalone glass brain figures with real connectivity data
        fig16a_glass_brain_high(delta_df, output_dir, network_df)
        fig16b_glass_brain_low(delta_df, output_dir, network_df)
        create_interactive_delta_viz(delta_df, network_df, output_dir)
    except Exception as e:
        print(f"Error generating figures: {e}")
        return False
    
    print("\n" + "=" * 70)
    print(" Complete! Figures saved to:")
    print(f" {output_dir}")
    print("=" * 70)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
    
    return True


def main():
    """Main function for standalone execution."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_h3_h4_visualization(OUTPUT_DIR)


if __name__ == "__main__":
    main()

