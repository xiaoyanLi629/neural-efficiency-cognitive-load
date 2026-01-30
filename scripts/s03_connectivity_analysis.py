"""
=============================================================================
Step 3: Functional Connectivity & Network Efficiency Analysis
=============================================================================

PURPOSE:
    Analyze functional connectivity patterns and network topology during the
    N-back task to characterize:
    1. Within-network integration (FPN coherence)
    2. Between-network segregation (FPN-DMN anticorrelation)
    3. Graph-theoretic efficiency metrics

RESEARCH QUESTIONS ADDRESSED:
    Q3.1: Does cognitive load alter functional connectivity patterns?
    Q3.2: Do efficient individuals show better network organization?
    Q3.3: Is network efficiency a better predictor of performance than
          regional activation?

THEORETICAL FRAMEWORK:
    Network efficiency reflects optimal information processing:
    - High global efficiency = efficient long-range communication
    - High local efficiency = efficient local processing/redundancy
    - High modularity = specialized processing within networks
    - Strong FPN-DMN anticorrelation = effective task focus

OUTPUT:
    - connectivity_matrices/: Subject-level FC matrices
    - network_metrics.csv: Graph theory metrics per subject
    - connectivity_stats.json: Statistical results

=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, pearsonr
from scipy.signal import detrend
import nibabel as nib
import json
import warnings
warnings.filterwarnings('ignore')

from configs import config
from configs.config import (
    DATA_ROOT, SUBJECTS,
    WM_TASK, NETWORKS, CONNECTIVITY_PARAMS, GRAPH_PARAMS,
    get_wm_fmri_path, get_wm_evs_path, setup_logging
)
# Use config.config.CONNECTIVITY_DIR, config.config.BEHAVIORAL_DIR for timestamped paths

logger = setup_logging('connectivity_analysis')

# =============================================================================
# PARCELLATION AND ROI DEFINITION
# =============================================================================

def get_network_parcels():
    """
    Define network parcels based on approximate HCP grayordinate locations.
    
    Returns dict with network name -> list of (parcel_name, indices).
    """
    n_left = 29696
    
    network_parcels = {
        'FPN': {
            'DLPFC_L': list(range(8000, 10000)),
            'DLPFC_R': list(range(n_left + 8000, n_left + 10000)),
            'VLPFC_L': list(range(6000, 7500)),
            'VLPFC_R': list(range(n_left + 6000, n_left + 7500)),
            'PPC_L': list(range(15000, 17000)),
            'PPC_R': list(range(n_left + 15000, n_left + 17000)),
            'ACC': list(range(3000, 4500)) + list(range(n_left + 3000, n_left + 4500)),
        },
        'DMN': {
            'mPFC': list(range(1000, 2500)) + list(range(n_left + 1000, n_left + 2500)),
            'PCC': list(range(20000, 21500)) + list(range(n_left + 20000, n_left + 21500)),
            'Angular_L': list(range(18000, 19000)),
            'Angular_R': list(range(n_left + 18000, n_left + 19000)),
            'MTG_L': list(range(22000, 23000)),
            'MTG_R': list(range(n_left + 22000, n_left + 23000)),
        },
        'DAN': {
            'FEF_L': list(range(5000, 5800)),
            'FEF_R': list(range(n_left + 5000, n_left + 5800)),
            'IPS_L': list(range(16500, 17500)),
            'IPS_R': list(range(n_left + 16500, n_left + 17500)),
        },
        'Visual': {
            'V1_L': list(range(24000, 26000)),
            'V1_R': list(range(n_left + 24000, n_left + 26000)),
            'V4_L': list(range(26000, 27000)),
            'V4_R': list(range(n_left + 26000, n_left + 27000)),
        },
        'Motor': {
            'M1_L': list(range(4000, 5000)),
            'M1_R': list(range(n_left + 4000, n_left + 5000)),
            'SMA': list(range(4500, 5200)) + list(range(n_left + 4500, n_left + 5200)),
        }
    }
    
    return network_parcels


def extract_parcel_timeseries(cifti_data, parcel_indices):
    """
    Extract mean time series from parcel.
    """
    valid_indices = [i for i in parcel_indices if i < cifti_data.shape[1]]
    if len(valid_indices) == 0:
        return np.zeros(cifti_data.shape[0])
    
    ts = np.nanmean(cifti_data[:, valid_indices], axis=1)
    return ts


# =============================================================================
# SIGNAL PREPROCESSING FOR CONNECTIVITY
# =============================================================================

def preprocess_timeseries(ts, tr=0.72):
    """
    Preprocess time series for connectivity analysis.
    
    Steps:
    1. Detrend (remove linear drift)
    2. Z-score normalization
    3. Bandpass filter (optional)
    """
    # Remove NaN
    ts = np.nan_to_num(ts)
    
    # Detrend
    ts = detrend(ts)
    
    # Z-score
    if np.std(ts) > 0:
        ts = (ts - np.mean(ts)) / np.std(ts)
    
    return ts


def extract_task_blocks(timeseries, events, condition, tr, buffer=2):
    """
    Extract time series segments during specific task blocks.
    
    Parameters:
        timeseries: Full run time series
        events: Dict with (onsets, durations)
        condition: '0bk' or '2bk'
        tr: Repetition time
        buffer: HRF delay buffer in TRs
    
    Returns:
        block_ts: Concatenated time series from relevant blocks
    """
    block_ts = []
    
    # Get all events for this condition
    relevant_events = []
    for key, (onsets, durations) in events.items():
        if key.startswith(condition) and not key.endswith(('_cor', '_err')):
            for onset, dur in zip(onsets, durations):
                relevant_events.append((onset, dur))
    
    # Sort by onset
    relevant_events.sort(key=lambda x: x[0])
    
    # Extract blocks
    for onset, dur in relevant_events:
        start_tr = int(onset / tr) + buffer
        end_tr = int((onset + dur) / tr) + buffer + 3  # Extra TRs for HRF delay
        
        if start_tr < len(timeseries) and end_tr <= len(timeseries):
            block_ts.append(timeseries[start_tr:end_tr])
    
    if block_ts:
        return np.concatenate(block_ts)
    else:
        return np.array([])


# =============================================================================
# CONNECTIVITY COMPUTATION
# =============================================================================

def compute_connectivity_matrix(parcel_timeseries, method='correlation'):
    """
    Compute functional connectivity matrix.
    
    Parameters:
        parcel_timeseries: Dict with parcel_name -> time series
        method: 'correlation' or 'partial_correlation'
    
    Returns:
        fc_matrix: Connectivity matrix (n_parcels x n_parcels)
        parcel_names: List of parcel names
    """
    parcel_names = list(parcel_timeseries.keys())
    n_parcels = len(parcel_names)
    
    # Stack into matrix
    data_matrix = np.column_stack([parcel_timeseries[name] for name in parcel_names])
    
    if method == 'correlation':
        # Pearson correlation
        fc_matrix = np.corrcoef(data_matrix.T)
    else:
        # Could add partial correlation here
        fc_matrix = np.corrcoef(data_matrix.T)
    
    # Set diagonal to 0
    np.fill_diagonal(fc_matrix, 0)
    
    # Fisher Z transform
    if CONNECTIVITY_PARAMS['fisher_z']:
        fc_matrix = np.arctanh(np.clip(fc_matrix, -0.999, 0.999))
    
    return fc_matrix, parcel_names


def compute_network_connectivity(fc_matrix, parcel_names, network_parcels):
    """
    Compute within-network and between-network connectivity.
    
    Returns dict with:
    - within_FPN: Mean connectivity within FPN
    - within_DMN: Mean connectivity within DMN
    - FPN_DMN: Mean connectivity between FPN and DMN
    - etc.
    """
    results = {}
    
    # Get indices for each network
    network_indices = {}
    for network_name, parcels in network_parcels.items():
        indices = []
        for parcel_name in parcels.keys():
            if parcel_name in parcel_names:
                indices.append(parcel_names.index(parcel_name))
        network_indices[network_name] = indices
    
    # Within-network connectivity
    for network_name, indices in network_indices.items():
        if len(indices) >= 2:
            submatrix = fc_matrix[np.ix_(indices, indices)]
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(submatrix, dtype=bool), k=1)
            within_conn = submatrix[mask].mean()
            results[f'within_{network_name}'] = within_conn
    
    # Between-network connectivity
    network_names = list(network_indices.keys())
    for i, net1 in enumerate(network_names):
        for j, net2 in enumerate(network_names):
            if i < j:
                idx1 = network_indices[net1]
                idx2 = network_indices[net2]
                if len(idx1) > 0 and len(idx2) > 0:
                    between_conn = fc_matrix[np.ix_(idx1, idx2)].mean()
                    results[f'{net1}_{net2}'] = between_conn
    
    return results


# =============================================================================
# GRAPH THEORY METRICS
# =============================================================================

def threshold_matrix(fc_matrix, threshold):
    """
    Threshold connectivity matrix to create binary adjacency matrix.
    """
    adj_matrix = (np.abs(fc_matrix) > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix


def compute_global_efficiency(adj_matrix):
    """
    Compute global efficiency of network.
    
    Global efficiency = average inverse shortest path length
    High efficiency = efficient information transfer
    """
    n = adj_matrix.shape[0]
    
    # Compute shortest paths using Floyd-Warshall
    # Convert to distance matrix (1/weight for weighted, 1 for binary)
    dist = np.where(adj_matrix > 0, 1, np.inf)
    np.fill_diagonal(dist, 0)
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    # Global efficiency = mean of 1/d for all pairs
    with np.errstate(divide='ignore'):
        inv_dist = 1 / dist
    inv_dist[np.isinf(inv_dist)] = 0
    np.fill_diagonal(inv_dist, 0)
    
    global_eff = inv_dist.sum() / (n * (n - 1))
    
    return global_eff


def compute_local_efficiency(adj_matrix):
    """
    Compute local efficiency (average of nodal efficiencies).
    
    Local efficiency = efficiency of local subgraphs
    High local efficiency = fault tolerance
    """
    n = adj_matrix.shape[0]
    local_effs = []
    
    for i in range(n):
        # Get neighbors of node i
        neighbors = np.where(adj_matrix[i, :] > 0)[0]
        
        if len(neighbors) < 2:
            local_effs.append(0)
            continue
        
        # Extract subgraph
        subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
        
        # Compute efficiency of subgraph
        sub_eff = compute_global_efficiency(subgraph)
        local_effs.append(sub_eff)
    
    return np.mean(local_effs)


def compute_modularity(adj_matrix, community_assignment):
    """
    Compute modularity Q for given community assignment.
    
    Q measures how well network is divided into communities.
    """
    n = adj_matrix.shape[0]
    m = adj_matrix.sum() / 2  # Total edges
    
    if m == 0:
        return 0
    
    k = adj_matrix.sum(axis=1)  # Node degrees
    
    Q = 0
    for i in range(n):
        for j in range(n):
            if community_assignment[i] == community_assignment[j]:
                Q += adj_matrix[i, j] - k[i] * k[j] / (2 * m)
    
    Q /= (2 * m)
    return Q


def compute_clustering_coefficient(adj_matrix):
    """
    Compute average clustering coefficient.
    """
    n = adj_matrix.shape[0]
    clustering = []
    
    for i in range(n):
        neighbors = np.where(adj_matrix[i, :] > 0)[0]
        k = len(neighbors)
        
        if k < 2:
            clustering.append(0)
            continue
        
        # Count triangles
        subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
        triangles = subgraph.sum() / 2
        
        # Maximum possible triangles
        max_triangles = k * (k - 1) / 2
        
        clustering.append(triangles / max_triangles if max_triangles > 0 else 0)
    
    return np.mean(clustering)


def compute_graph_metrics(fc_matrix, threshold=0.15):
    """
    Compute all graph theory metrics.
    """
    # Threshold matrix
    adj = threshold_matrix(fc_matrix, threshold)
    
    # Compute metrics
    metrics = {
        'global_efficiency': compute_global_efficiency(adj),
        'local_efficiency': compute_local_efficiency(adj),
        'clustering_coefficient': compute_clustering_coefficient(adj),
        'density': adj.sum() / (adj.shape[0] * (adj.shape[0] - 1)),
        'mean_connectivity': np.abs(fc_matrix[np.triu_indices_from(fc_matrix, k=1)]).mean(),
    }
    
    # Simple community assignment based on network membership
    network_parcels = get_network_parcels()
    all_parcels = []
    community = []
    comm_idx = 0
    for net_name, parcels in network_parcels.items():
        for parcel_name in parcels.keys():
            all_parcels.append(parcel_name)
            community.append(comm_idx)
        comm_idx += 1
    
    if len(community) == fc_matrix.shape[0]:
        metrics['modularity'] = compute_modularity(adj, community)
    else:
        metrics['modularity'] = 0
    
    return metrics


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_subject_connectivity(subject):
    """
    Analyze connectivity for a single subject.
    """
    logger.info(f"  Processing subject {subject}...")
    
    network_parcels = get_network_parcels()
    
    # Flatten parcels
    all_parcels = {}
    for net_name, parcels in network_parcels.items():
        for parcel_name, indices in parcels.items():
            all_parcels[parcel_name] = indices
    
    results = {'subject': subject}
    
    for run in ['LR', 'RL']:
        fmri_path = get_wm_fmri_path(subject, run)
        if not fmri_path.exists():
            continue
        
        try:
            # Load data
            cifti = nib.load(str(fmri_path))
            cifti_data = cifti.get_fdata()
            
            # Get events
            from scripts.s02_activation_analysis import get_task_events
            events = get_task_events(subject, run)
            
            # Extract parcel time series
            parcel_ts_full = {}
            for parcel_name, indices in all_parcels.items():
                ts = extract_parcel_timeseries(cifti_data, indices)
                ts = preprocess_timeseries(ts)
                parcel_ts_full[parcel_name] = ts
            
            # Compute full-run connectivity
            fc_full, parcel_names = compute_connectivity_matrix(parcel_ts_full)
            
            # Network-level connectivity
            net_conn = compute_network_connectivity(fc_full, parcel_names, network_parcels)
            for key, val in net_conn.items():
                results[f'{key}_{run}'] = val
            
            # Graph metrics
            graph_metrics = compute_graph_metrics(fc_full)
            for key, val in graph_metrics.items():
                results[f'{key}_{run}'] = val
            
            # Condition-specific connectivity (0-back vs 2-back)
            for condition in ['0bk', '2bk']:
                parcel_ts_cond = {}
                for parcel_name, ts in parcel_ts_full.items():
                    block_ts = extract_task_blocks(ts, events, condition, WM_TASK['tr'])
                    if len(block_ts) > 10:
                        parcel_ts_cond[parcel_name] = block_ts
                
                if len(parcel_ts_cond) == len(all_parcels):
                    fc_cond, _ = compute_connectivity_matrix(parcel_ts_cond)
                    graph_cond = compute_graph_metrics(fc_cond)
                    for key, val in graph_cond.items():
                        results[f'{key}_{condition}_{run}'] = val
                    
                    net_conn_cond = compute_network_connectivity(fc_cond, parcel_names, network_parcels)
                    for key, val in net_conn_cond.items():
                        results[f'{key}_{condition}_{run}'] = val
                        
        except Exception as e:
            logger.error(f"    Error processing {subject} run {run}: {e}")
            continue
    
    # Average across runs
    averaged = {'subject': subject}
    run_keys = [k for k in results.keys() if k != 'subject']
    
    for key in set([k.rsplit('_', 1)[0] for k in run_keys if k.endswith(('_LR', '_RL'))]):
        lr_key = f"{key}_LR"
        rl_key = f"{key}_RL"
        if lr_key in results and rl_key in results:
            averaged[key] = (results[lr_key] + results[rl_key]) / 2
        elif lr_key in results:
            averaged[key] = results[lr_key]
        elif rl_key in results:
            averaged[key] = results[rl_key]
    
    return averaged


def run_connectivity_analysis():
    """
    Main connectivity analysis pipeline.
    """
    logger.info("="*60)
    logger.info("Starting Connectivity Analysis Pipeline")
    logger.info("="*60)
    
    # Load group assignments
    group_file = config.BEHAVIORAL_DIR / 'group_assignments.csv'
    if group_file.exists():
        groups_df = pd.read_csv(group_file)
    else:
        groups_df = None
    
    # Analyze all subjects
    all_results = []
    for subject in SUBJECTS:
        result = analyze_subject_connectivity(subject)
        if result:
            all_results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Merge with groups
    if groups_df is not None:
        # Ensure subject columns are same type (string)
        results_df['subject'] = results_df['subject'].astype(str)
        groups_df['subject'] = groups_df['subject'].astype(str)
        results_df = results_df.merge(groups_df[['subject', 'efficiency_group']], 
                                       on='subject', how='left')
    
    # Statistical tests
    logger.info("\nRunning statistical tests...")
    stats_results = run_connectivity_statistics(results_df)
    
    # Save results
    logger.info("\nSaving results...")
    results_df.to_csv(config.CONNECTIVITY_DIR / 'network_metrics.csv', index=False)
    
    with open(config.CONNECTIVITY_DIR / 'connectivity_stats.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Connectivity Analysis Complete!")
    logger.info("="*60)
    
    return results_df, stats_results


def run_connectivity_statistics(df):
    """
    Statistical tests on connectivity data.
    """
    results = {}
    
    # 1. Load effect on network metrics
    for metric in ['global_efficiency', 'local_efficiency', 'modularity', 'within_FPN', 'FPN_DMN']:
        col_0bk = f'{metric}_0bk'
        col_2bk = f'{metric}_2bk'
        
        if col_0bk in df.columns and col_2bk in df.columns:
            valid = df[[col_0bk, col_2bk]].dropna()
            if len(valid) >= 5:
                t, p = ttest_rel(valid[col_2bk], valid[col_0bk])
                d = (valid[col_2bk].mean() - valid[col_0bk].mean()) / valid[[col_0bk, col_2bk]].stack().std()
                
                results[f'load_effect_{metric}'] = {
                    't_statistic': float(t),
                    'p_value': float(p),
                    'cohens_d': float(d),
                    'mean_0bk': float(valid[col_0bk].mean()),
                    'mean_2bk': float(valid[col_2bk].mean())
                }
    
    # 2. Group comparisons
    if 'efficiency_group' in df.columns:
        high_eff = df[df['efficiency_group'] == 'High_Efficiency']
        low_eff = df[df['efficiency_group'] == 'Low_Efficiency']
        
        for metric in ['global_efficiency', 'local_efficiency', 'within_FPN', 'FPN_DMN']:
            for cond in ['', '_0bk', '_2bk']:
                col = f'{metric}{cond}' if cond else metric
                
                if col in df.columns:
                    h = high_eff[col].dropna()
                    l = low_eff[col].dropna()
                    
                    if len(h) >= 3 and len(l) >= 3:
                        t, p = ttest_ind(h, l)
                        d = (h.mean() - l.mean()) / np.sqrt((h.var() + l.var()) / 2)
                        
                        results[f'group_diff_{col}'] = {
                            't_statistic': float(t),
                            'p_value': float(p),
                            'cohens_d': float(d),
                            'mean_high': float(h.mean()),
                            'mean_low': float(l.mean())
                        }
    
    return results


if __name__ == '__main__':
    results_df, stats = run_connectivity_analysis()
    
    print("\n" + "="*80)
    print("CONNECTIVITY RESULTS SUMMARY")
    print("="*80)
    print(results_df.head())

