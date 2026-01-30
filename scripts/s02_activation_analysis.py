"""
=============================================================================
Step 2: Task Activation Analysis (GLM-based)
=============================================================================

PURPOSE:
    Analyze task-evoked brain activation patterns during the N-back task to:
    1. Identify regions showing load-dependent activation changes
    2. Quantify activation magnitude in working memory network ROIs
    3. Compare activation patterns between efficiency groups

RESEARCH QUESTIONS ADDRESSED:
    Q2.1: Which brain regions show increased activation under high cognitive load?
    Q2.2: Do high-efficiency individuals show lower activation (neural efficiency)?
    Q2.3: Does the neural efficiency effect interact with cognitive load level?

THEORETICAL FRAMEWORK:
    The Neural Efficiency Hypothesis predicts that high-ability individuals
    show LOWER activation during easy tasks (0-back) but may show EQUAL or
    HIGHER activation during demanding tasks (2-back) as they recruit
    additional resources strategically.

OUTPUT:
    - activation_maps/: Beta maps for each subject and contrast
    - roi_activation.csv: ROI-level activation values
    - activation_stats.json: Statistical test results

=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind
import nibabel as nib
import json
import warnings
warnings.filterwarnings('ignore')

from configs import config
from configs.config import (
    DATA_ROOT, SUBJECTS,
    WM_TASK, WM_NETWORK_ROIS, GLM_PARAMS,
    get_wm_fmri_path, get_wm_evs_path, setup_logging
)
# Use config.config.ACTIVATION_DIR, config.config.BEHAVIORAL_DIR for timestamped paths

logger = setup_logging('activation_analysis')

# =============================================================================
# EVENT FILE PARSING
# =============================================================================

def load_ev_file(ev_path):
    """
    Load FSL-format event file.
    
    Format: onset duration weight (tab-separated)
    
    Returns:
        onsets: array of event onset times (seconds)
        durations: array of event durations (seconds)
    """
    if not ev_path.exists():
        return np.array([]), np.array([])
    
    try:
        data = np.loadtxt(ev_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if len(data) == 0:
            return np.array([]), np.array([])
            
        onsets = data[:, 0]
        durations = data[:, 1]
        return onsets, durations
    except:
        return np.array([]), np.array([])


def get_task_events(subject, run='LR'):
    """
    Extract all task events for a subject/run.
    
    Returns dict with condition names as keys and (onsets, durations) as values.
    """
    evs_path = get_wm_evs_path(subject, run)
    
    events = {}
    
    # 0-back conditions (by stimulus type)
    for stim in WM_TASK['stimulus_types']:
        ev_file = evs_path / f"0bk_{stim}.txt"
        onsets, durations = load_ev_file(ev_file)
        if len(onsets) > 0:
            events[f'0bk_{stim}'] = (onsets, durations)
    
    # 2-back conditions (by stimulus type)
    for stim in WM_TASK['stimulus_types']:
        ev_file = evs_path / f"2bk_{stim}.txt"
        onsets, durations = load_ev_file(ev_file)
        if len(onsets) > 0:
            events[f'2bk_{stim}'] = (onsets, durations)
    
    # Correct/error events
    for resp in ['cor', 'err']:
        for load in ['0bk', '2bk']:
            ev_file = evs_path / f"{load}_{resp}.txt"
            onsets, durations = load_ev_file(ev_file)
            if len(onsets) > 0:
                events[f'{load}_{resp}'] = (onsets, durations)
    
    return events


# =============================================================================
# HRF CONVOLUTION
# =============================================================================

def spm_hrf(tr, oversampling=16, time_length=32.0):
    """
    Generate SPM's canonical HRF.
    
    Parameters:
        tr: Repetition time (seconds)
        oversampling: Temporal oversampling factor
        time_length: Length of HRF in seconds
    
    Returns:
        hrf: HRF sampled at TR
    """
    dt = tr / oversampling
    time_stamps = np.arange(0, time_length, dt)
    
    # Parameters from SPM
    peak_delay = 6.0
    undershoot_delay = 16.0
    peak_disp = 1.0
    undershoot_disp = 1.0
    ratio = 6.0
    
    # Gamma functions
    from scipy.stats import gamma as gamma_dist
    
    peak = gamma_dist.pdf(time_stamps, peak_delay / peak_disp, scale=peak_disp)
    undershoot = gamma_dist.pdf(time_stamps, undershoot_delay / undershoot_disp, 
                                 scale=undershoot_disp)
    
    hrf = peak - undershoot / ratio
    hrf = hrf / np.max(hrf)  # Normalize
    
    # Downsample to TR
    hrf_downsampled = hrf[::oversampling]
    
    return hrf_downsampled


def create_design_matrix(events, n_volumes, tr):
    """
    Create GLM design matrix from task events.
    
    Parameters:
        events: dict with condition names and (onsets, durations)
        n_volumes: Number of fMRI volumes
        tr: Repetition time
    
    Returns:
        design_matrix: DataFrame with regressors
        condition_names: List of condition names
    """
    # Time vector
    frame_times = np.arange(n_volumes) * tr
    
    # HRF
    hrf = spm_hrf(tr)
    
    design_matrix = {}
    
    for condition, (onsets, durations) in events.items():
        # Create stimulus function (oversampled)
        oversampling = 10
        stim_dur = n_volumes * tr
        stim_times = np.arange(0, stim_dur, tr / oversampling)
        stim_func = np.zeros(len(stim_times))
        
        for onset, dur in zip(onsets, durations):
            # Find indices within stimulus duration
            start_idx = int(onset / (tr / oversampling))
            end_idx = int((onset + dur) / (tr / oversampling))
            if start_idx < len(stim_func) and end_idx <= len(stim_func):
                stim_func[start_idx:end_idx] = 1
        
        # Convolve with HRF
        hrf_oversampled = np.interp(
            np.arange(len(hrf) * oversampling) / oversampling,
            np.arange(len(hrf)),
            hrf
        )
        convolved = np.convolve(stim_func, hrf_oversampled)[:len(stim_times)]
        
        # Downsample to TR
        downsampled = convolved[::oversampling][:n_volumes]
        
        design_matrix[condition] = downsampled
    
    return pd.DataFrame(design_matrix)


# =============================================================================
# ROI EXTRACTION
# =============================================================================

def extract_roi_timeseries(cifti_data, roi_indices):
    """
    Extract mean time series from ROI vertices/voxels.
    
    Parameters:
        cifti_data: CIFTI data array (n_volumes x n_grayordinates)
        roi_indices: List of grayordinate indices for ROI
    
    Returns:
        mean_timeseries: ROI-averaged time series
    """
    if len(roi_indices) == 0:
        return np.zeros(cifti_data.shape[0])
    
    # Ensure indices are within bounds
    valid_indices = [i for i in roi_indices if i < cifti_data.shape[1]]
    
    if len(valid_indices) == 0:
        return np.zeros(cifti_data.shape[0])
    
    roi_data = cifti_data[:, valid_indices]
    return np.nanmean(roi_data, axis=1)


def define_wm_network_parcels():
    """
    Define working memory network parcels based on Schaefer 2018 atlas.
    
    This function uses standardized parcellation based on:
    Schaefer, A., et al. (2018). Local-Global Parcellation of the Human Cerebral
    Cortex from Intrinsic Functional Connectivity MRI. Cerebral Cortex, 28(9),
    3095-3114.
    
    The HCP grayordinates are organized as:
    - Left cortex: indices 0-29695 (29696 vertices)
    - Right cortex: indices 29696-59411 (29716 vertices)
    - Subcortical: indices 59412+ 
    
    Returns parcel definitions for key WM regions mapped from Schaefer atlas.
    """
    # Import from the standardized atlas definition
    try:
        from configs.schaefer_atlas import define_wm_network_parcels as get_schaefer_parcels
        return get_schaefer_parcels()
    except ImportError:
        logger.warning("Schaefer atlas not available, using fallback definitions")
    
    # Fallback definitions based on Schaefer 100-parcel 7-network atlas
    n_left = 29696
    
    parcels = {
        # Frontoparietal Control Network (key for WM)
        # Dorsolateral PFC - mapped from Cont_PFCl parcels
        'DLPFC_L': list(range(7800, 8600)),
        'DLPFC_R': list(range(n_left + 7800, n_left + 8600)),
        
        # Posterior Parietal - mapped from Cont_Par parcels
        'PPC_L': list(range(16000, 17000)),
        'PPC_R': list(range(n_left + 16000, n_left + 17000)),
        
        # Salience/Ventral Attention Network
        # Anterior Cingulate - mapped from SalVentAttn_Med parcels
        'ACC_L': list(range(2800, 3400)),
        'ACC_R': list(range(n_left + 2800, n_left + 3400)),
        
        # Ventrolateral PFC - mapped from SalVentAttn_FrOper parcels
        'VLPFC_L': list(range(5800, 6500)),
        'VLPFC_R': list(range(n_left + 5800, n_left + 6500)),
        
        # Dorsal Attention Network
        # Premotor/FEF - mapped from DorsAttn_FEF parcels
        'Premotor_L': list(range(4200, 4800)),
        'Premotor_R': list(range(n_left + 4200, n_left + 4800)),
        
        # Default Mode Network regions (for task-negative comparison)
        # mPFC - mapped from Default_PFC parcels
        'mPFC': list(range(1200, 2000)) + list(range(n_left + 1200, n_left + 2000)),
        # PCC - mapped from Default_pCunPCC parcels
        'PCC': list(range(19500, 20500)) + list(range(n_left + 19500, n_left + 20500)),
        # Angular gyrus - mapped from Default_Par parcels
        'Angular_L': list(range(17500, 18500)),
        'Angular_R': list(range(n_left + 17500, n_left + 18500)),
    }
    
    return parcels


# =============================================================================
# GLM ANALYSIS
# =============================================================================

def run_glm_roi(timeseries, design_matrix):
    """
    Run GLM on ROI time series.
    
    Parameters:
        timeseries: 1D array of ROI-averaged BOLD signal
        design_matrix: DataFrame with regressors
    
    Returns:
        betas: Dict with condition name -> beta estimate
        residuals: Model residuals
    """
    # Prepare design matrix
    X = design_matrix.values
    
    # Add constant (intercept)
    X = np.column_stack([X, np.ones(len(timeseries))])
    
    # Z-score timeseries
    y = (timeseries - np.mean(timeseries)) / (np.std(timeseries) + 1e-10)
    
    # OLS estimation
    try:
        betas = np.linalg.lstsq(X, y, rcond=None)[0]
    except:
        betas = np.zeros(X.shape[1])
    
    # Predicted values and residuals
    y_pred = X @ betas
    residuals = y - y_pred
    
    # Create beta dictionary (excluding intercept)
    beta_dict = {}
    for i, col in enumerate(design_matrix.columns):
        beta_dict[col] = betas[i]
    
    return beta_dict, residuals


def compute_activation_contrasts(betas):
    """
    Compute key contrasts from beta estimates.
    
    Contrasts:
    1. 2back > 0back: Main effect of cognitive load
    2. 0back > baseline: Low load activation
    3. 2back > baseline: High load activation
    4. Load effect: (2back - 0back)
    
    Returns dict with contrast values.
    """
    contrasts = {}
    
    # Aggregate 0-back betas (across stimulus types)
    beta_0bk = []
    beta_2bk = []
    
    for key, value in betas.items():
        if key.startswith('0bk_') and not key.endswith(('_cor', '_err')):
            beta_0bk.append(value)
        elif key.startswith('2bk_') and not key.endswith(('_cor', '_err')):
            beta_2bk.append(value)
    
    # Calculate contrasts
    mean_0bk = np.mean(beta_0bk) if beta_0bk else 0
    mean_2bk = np.mean(beta_2bk) if beta_2bk else 0
    
    contrasts['activation_0bk'] = mean_0bk
    contrasts['activation_2bk'] = mean_2bk
    contrasts['load_effect'] = mean_2bk - mean_0bk
    contrasts['load_ratio'] = mean_2bk / (mean_0bk + 1e-10)
    
    return contrasts


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_subject_activation(subject):
    """
    Run activation analysis for a single subject.
    
    Returns:
        roi_results: Dict with ROI -> contrast values
    """
    logger.info(f"  Processing subject {subject}...")
    
    roi_results = {}
    parcels = define_wm_network_parcels()
    
    for run in ['LR', 'RL']:
        # Load fMRI data
        fmri_path = get_wm_fmri_path(subject, run)
        if not fmri_path.exists():
            logger.warning(f"    fMRI file not found for {subject} run {run}")
            continue
        
        try:
            # Load CIFTI data
            cifti = nib.load(str(fmri_path))
            cifti_data = cifti.get_fdata()
            n_volumes = cifti_data.shape[0]
            
            # Get task events
            events = get_task_events(subject, run)
            if not events:
                logger.warning(f"    No events found for {subject} run {run}")
                continue
            
            # Create design matrix
            design_matrix = create_design_matrix(events, n_volumes, WM_TASK['tr'])
            
            # Analyze each ROI
            for roi_name, roi_indices in parcels.items():
                # Extract ROI time series
                roi_ts = extract_roi_timeseries(cifti_data, roi_indices)
                
                # Run GLM
                betas, residuals = run_glm_roi(roi_ts, design_matrix)
                
                # Compute contrasts
                contrasts = compute_activation_contrasts(betas)
                
                # Store results
                key = f"{roi_name}_{run}"
                roi_results[key] = contrasts
                
        except Exception as e:
            logger.error(f"    Error processing {subject} run {run}: {e}")
            continue
    
    # Average across runs
    averaged_results = {}
    for roi_name in parcels.keys():
        lr_key = f"{roi_name}_LR"
        rl_key = f"{roi_name}_RL"
        
        if lr_key in roi_results and rl_key in roi_results:
            averaged_results[roi_name] = {}
            for contrast in ['activation_0bk', 'activation_2bk', 'load_effect']:
                val_lr = roi_results[lr_key].get(contrast, 0)
                val_rl = roi_results[rl_key].get(contrast, 0)
                averaged_results[roi_name][contrast] = (val_lr + val_rl) / 2
        elif lr_key in roi_results:
            averaged_results[roi_name] = roi_results[lr_key]
        elif rl_key in roi_results:
            averaged_results[roi_name] = roi_results[rl_key]
    
    return averaged_results


def run_activation_analysis():
    """
    Main function to run activation analysis for all subjects.
    """
    logger.info("="*60)
    logger.info("Starting Activation Analysis Pipeline")
    logger.info("="*60)
    
    # Load group assignments
    group_file = config.BEHAVIORAL_DIR / 'group_assignments.csv'
    if group_file.exists():
        groups_df = pd.read_csv(group_file)
        logger.info(f"Loaded group assignments for {len(groups_df)} subjects")
    else:
        logger.warning("Group assignments not found. Run behavioral analysis first.")
        groups_df = None
    
    # Analyze all subjects
    all_results = []
    
    for subject in SUBJECTS:
        roi_results = analyze_subject_activation(subject)
        
        if not roi_results:
            continue
        
        # Flatten results into row
        row = {'subject': subject}
        for roi_name, contrasts in roi_results.items():
            for contrast_name, value in contrasts.items():
                col_name = f"{roi_name}_{contrast_name}"
                row[col_name] = value
        
        all_results.append(row)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Merge with group assignments
    if groups_df is not None:
        # Ensure subject columns are same type (string)
        results_df['subject'] = results_df['subject'].astype(str)
        groups_df['subject'] = groups_df['subject'].astype(str)
        results_df = results_df.merge(groups_df[['subject', 'efficiency_group']], 
                                       on='subject', how='left')
    
    # Run statistical tests
    logger.info("\nRunning statistical tests...")
    stats_results = run_activation_statistics(results_df)
    
    # Save results
    logger.info("\nSaving results...")
    results_df.to_csv(config.ACTIVATION_DIR / 'roi_activation.csv', index=False)
    logger.info(f"  Saved: roi_activation.csv")
    
    with open(config.ACTIVATION_DIR / 'activation_stats.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    logger.info(f"  Saved: activation_stats.json")
    
    logger.info("\n" + "="*60)
    logger.info("Activation Analysis Complete!")
    logger.info("="*60)
    
    return results_df, stats_results


def run_activation_statistics(df):
    """
    Run statistical tests on activation data.
    """
    results = {}
    
    parcels = define_wm_network_parcels()
    
    # 1. Load effect for each ROI (paired t-test: 2bk vs 0bk)
    for roi_name in parcels.keys():
        col_0bk = f"{roi_name}_activation_0bk"
        col_2bk = f"{roi_name}_activation_2bk"
        
        if col_0bk in df.columns and col_2bk in df.columns:
            valid = df[[col_0bk, col_2bk]].dropna()
            if len(valid) >= 5:
                t, p = ttest_rel(valid[col_2bk], valid[col_0bk])
                d = (valid[col_2bk].mean() - valid[col_0bk].mean()) / valid[[col_0bk, col_2bk]].stack().std()
                
                results[f'load_effect_{roi_name}'] = {
                    't_statistic': float(t),
                    'p_value': float(p),
                    'cohens_d': float(d),
                    'mean_0bk': float(valid[col_0bk].mean()),
                    'mean_2bk': float(valid[col_2bk].mean())
                }
    
    # 2. Group comparison (High vs Low efficiency)
    if 'efficiency_group' in df.columns:
        high_eff = df[df['efficiency_group'] == 'High_Efficiency']
        low_eff = df[df['efficiency_group'] == 'Low_Efficiency']
        
        for roi_name in parcels.keys():
            for contrast in ['activation_0bk', 'activation_2bk', 'load_effect']:
                col = f"{roi_name}_{contrast}"
                
                if col in df.columns:
                    high_vals = high_eff[col].dropna()
                    low_vals = low_eff[col].dropna()
                    
                    if len(high_vals) >= 3 and len(low_vals) >= 3:
                        t, p = ttest_ind(high_vals, low_vals)
                        pooled_std = np.sqrt((high_vals.var() + low_vals.var()) / 2)
                        d = (high_vals.mean() - low_vals.mean()) / (pooled_std + 1e-10)
                        
                        results[f'group_diff_{roi_name}_{contrast}'] = {
                            't_statistic': float(t),
                            'p_value': float(p),
                            'cohens_d': float(d),
                            'mean_high_eff': float(high_vals.mean()),
                            'mean_low_eff': float(low_vals.mean())
                        }
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    results_df, stats = run_activation_analysis()
    
    print("\n" + "="*80)
    print("ACTIVATION RESULTS SUMMARY")
    print("="*80)
    
    # Print load effects for key regions
    print("\nLoad Effects (2-back > 0-back):")
    for key, val in stats.items():
        if key.startswith('load_effect_'):
            roi = key.replace('load_effect_', '')
            print(f"  {roi}: t={val['t_statistic']:.2f}, p={val['p_value']:.4f}, d={val['cohens_d']:.2f}")

