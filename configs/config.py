"""
=============================================================================
Project Configuration: Neural Efficiency Under Cognitive Load
BIBM 2026 Submission
=============================================================================

This configuration file centralizes all paths, parameters, and constants
used throughout the analysis pipeline.

Path Resolution Order:
    1. Environment variables (HCP_DATA_ROOT, PROJECT_DIR)
    2. Default paths based on project location
"""

import os
from pathlib import Path
from datetime import datetime

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project directory is the parent of configs/
PROJECT_DIR = Path(__file__).parent.parent

# Data root: configurable via environment variable
# Default: project_dir/data (HCP subject directories)
DATA_ROOT = Path(os.environ.get("HCP_DATA_ROOT", PROJECT_DIR / "data"))

# =============================================================================
# TIMESTAMPED OUTPUT DIRECTORIES
# =============================================================================

# Global variable to hold the current run's timestamp
_RUN_TIMESTAMP = None

def get_run_timestamp():
    """Get or create the timestamp for current run"""
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _RUN_TIMESTAMP

def reset_run_timestamp():
    """Reset timestamp for a new run"""
    global _RUN_TIMESTAMP
    _RUN_TIMESTAMP = None

def init_output_dirs(use_timestamp=True):
    """
    Initialize output directories for the current run.

    Args:
        use_timestamp: If True, create timestamped subdirectory.
                      If False, use default 'latest' directory.

    Returns:
        dict: Dictionary containing all output directory paths
    """
    global RESULTS_DIR, BEHAVIORAL_DIR, ACTIVATION_DIR, CONNECTIVITY_DIR
    global EFFICIENCY_DIR, DELTA_EFFICIENCY_DIR, FIGURES_DIR, LOGS_DIR

    if use_timestamp:
        timestamp = get_run_timestamp()
        run_name = f"run_{timestamp}"
    else:
        run_name = "latest"

    # Create timestamped results directory
    RESULTS_DIR = PROJECT_DIR / "results" / run_name
    BEHAVIORAL_DIR = RESULTS_DIR / "behavioral"
    ACTIVATION_DIR = RESULTS_DIR / "activation"
    CONNECTIVITY_DIR = RESULTS_DIR / "connectivity"
    EFFICIENCY_DIR = RESULTS_DIR / "efficiency"
    DELTA_EFFICIENCY_DIR = RESULTS_DIR / "delta_efficiency"
    FIGURES_DIR = RESULTS_DIR / "figures"
    LOGS_DIR = PROJECT_DIR / "logs" / run_name
    
    # Create directories
    for d in [BEHAVIORAL_DIR, ACTIVATION_DIR, CONNECTIVITY_DIR,
              EFFICIENCY_DIR, DELTA_EFFICIENCY_DIR, FIGURES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create a symlink to the latest run
    latest_link = PROJECT_DIR / "results" / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        import shutil
        shutil.rmtree(latest_link)
    
    try:
        latest_link.symlink_to(RESULTS_DIR.name)
    except:
        pass  # Symlink may fail on some systems
    
    return {
        'results': RESULTS_DIR,
        'behavioral': BEHAVIORAL_DIR,
        'activation': ACTIVATION_DIR,
        'connectivity': CONNECTIVITY_DIR,
        'efficiency': EFFICIENCY_DIR,
        'delta_efficiency': DELTA_EFFICIENCY_DIR,
        'figures': FIGURES_DIR,
        'logs': LOGS_DIR,
        'run_name': run_name,
    }

# Default output directories (will be overwritten when init_output_dirs is called)
# These are placeholders - actual directories are created by init_output_dirs()
RESULTS_DIR = PROJECT_DIR / "results"
BEHAVIORAL_DIR = RESULTS_DIR / "behavioral"
ACTIVATION_DIR = RESULTS_DIR / "activation"
CONNECTIVITY_DIR = RESULTS_DIR / "connectivity"
EFFICIENCY_DIR = RESULTS_DIR / "efficiency"
DELTA_EFFICIENCY_DIR = RESULTS_DIR / "delta_efficiency"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = PROJECT_DIR / "logs"

# Note: Directories are NOT created at import time
# Call init_output_dirs() to create timestamped directories

# =============================================================================
# SUBJECT LIST
# =============================================================================

def get_subject_list():
    """Get list of all available subjects from data directory."""
    subjects = []
    if DATA_ROOT.exists():
        for item in DATA_ROOT.iterdir():
            if item.is_dir() and item.name.isdigit():
                subjects.append(item.name)
    return sorted(subjects)

SUBJECTS = get_subject_list()
N_SUBJECTS = len(SUBJECTS)

# =============================================================================
# TASK PARAMETERS
# =============================================================================

# Working Memory Task Parameters
WM_TASK = {
    'name': 'WM',
    'runs': ['LR', 'RL'],
    'conditions': {
        'low_load': '0bk',   # 0-back (low cognitive load)
        'high_load': '2bk',  # 2-back (high cognitive load)
    },
    'stimulus_types': ['body', 'faces', 'places', 'tools'],
    'response_types': ['cor', 'err', 'nlr'],  # correct, error, no response
    'tr': 0.72,  # HCP TR in seconds
    'n_volumes': 405,  # Number of volumes per run
}

# =============================================================================
# BRAIN PARCELLATION - Working Memory Network ROIs
# =============================================================================

# HCP-MMP1.0 parcellation labels for key regions
# Based on Glasser et al. (2016) Nature

WM_NETWORK_ROIS = {
    # Frontoparietal Network (FPN) - Executive Control
    'DLPFC_L': [6, 7, 8, 9, 10, 11, 12, 46],  # Left Dorsolateral PFC areas
    'DLPFC_R': [6, 7, 8, 9, 10, 11, 12, 46],  # Right Dorsolateral PFC areas
    'VLPFC_L': [44, 45, 47],  # Left Ventrolateral PFC
    'VLPFC_R': [44, 45, 47],  # Right Ventrolateral PFC
    'PPC_L': [7, 39, 40],     # Left Posterior Parietal Cortex
    'PPC_R': [7, 39, 40],     # Right Posterior Parietal Cortex
    'ACC': [24, 32, 33],      # Anterior Cingulate Cortex
    'preSMA': [6],            # Pre-Supplementary Motor Area
    
    # Default Mode Network (DMN) - Task-negative
    'mPFC': [10, 11, 32],     # Medial Prefrontal Cortex
    'PCC': [23, 31],          # Posterior Cingulate Cortex
    'Angular_L': [39],        # Left Angular Gyrus
    'Angular_R': [39],        # Right Angular Gyrus
    
    # Dorsal Attention Network (DAN)
    'FEF_L': [8],             # Left Frontal Eye Fields
    'FEF_R': [8],             # Right Frontal Eye Fields
    'IPS_L': [7, 40],         # Left Intraparietal Sulcus
    'IPS_R': [7, 40],         # Right Intraparietal Sulcus
}

# Network definitions for connectivity analysis
NETWORKS = {
    'FPN': ['DLPFC_L', 'DLPFC_R', 'VLPFC_L', 'VLPFC_R', 'PPC_L', 'PPC_R', 'ACC', 'preSMA'],
    'DMN': ['mPFC', 'PCC', 'Angular_L', 'Angular_R'],
    'DAN': ['FEF_L', 'FEF_R', 'IPS_L', 'IPS_R'],
}

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

# Behavioral analysis
BEHAVIORAL_PARAMS = {
    'efficiency_method': 'ies',  # Inverse Efficiency Score: RT / ACC
    'outlier_threshold': 3.0,    # Z-score threshold for outlier removal
    'min_accuracy': 0.5,         # Minimum accuracy to include subject
    'group_split': 'median',     # Method for group splitting
}

# GLM parameters
GLM_PARAMS = {
    'hrf_model': 'spm',          # HRF model type
    'high_pass': 128,            # High-pass filter cutoff (seconds)
    'ar_order': 1,               # AR model order for autocorrelation
    'smoothing_fwhm': 4,         # Spatial smoothing (mm)
}

# Connectivity parameters
CONNECTIVITY_PARAMS = {
    'method': 'correlation',     # Connectivity method
    'fisher_z': True,            # Apply Fisher Z transformation
    'threshold': 0.1,            # Threshold for graph construction
    'n_parcels': 360,            # HCP-MMP1.0 parcellation
}

# Graph theory parameters
GRAPH_PARAMS = {
    'threshold_range': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'metrics': ['global_efficiency', 'local_efficiency', 'modularity', 
                'clustering', 'path_length'],
}

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# =============================================================================
# UNIFIED COLOR SCHEME - Consistent across all figures
# =============================================================================

# Group colors (High vs Low Efficiency)
HIGH_EFF = '#1a5276'        # Deep Blue - High Efficiency
LOW_EFF = '#c0392b'         # Deep Red - Low Efficiency
HIGH_EFF_LIGHT = '#5dade2'  # Light Blue
LOW_EFF_LIGHT = '#f1948a'   # Light Red

# Load condition colors
LOAD_0BK = '#27ae60'        # Green - 0-back (easier)
LOAD_2BK = '#e74c3c'        # Red - 2-back (harder)

# Other colors
NEUTRAL = '#566573'         # Gray - Neutral
ACCENT = '#9b59b6'          # Purple - Accent

# Network colors (for brain visualizations)
NETWORK_COLORS = {
    'FPN': '#2980b9',       # Blue - Frontoparietal (Executive)
    'DMN': '#27ae60',       # Green - Default Mode
    'SAL': '#e67e22',       # Orange - Salience
    'DAN': '#2ca02c',       # Green - Dorsal Attention
    'Visual': '#9467bd',    # Purple
    'Motor': '#8c564b',     # Brown
}

# ColorBrewer colormaps (https://pratiman-91.github.io/colormaps/)
COLORMAPS = {
    'sequential': 'viridis',      # For sequential data (activation magnitude)
    'diverging': 'RdBu_r',        # For diverging data centered at zero
    'categorical': 'Set2',        # For categorical comparisons (groups)
    'correlation': 'RdBu_r',      # For correlation matrices
    'heatmap': 'viridis',         # For heatmaps (unified with sequential)
    'activation': 'plasma',       # For brain activation
    'connectivity': 'coolwarm',   # For connectivity matrices
}

# Figure parameters
FIGURE_PARAMS = {
    'dpi': 600,
    'formats': ['png', 'svg'],  # Output formats: PNG and SVG only
    'font_family': 'Arial',
    'font_size': {
        'title': 14,
        'label': 12,
        'tick': 10,
        'legend': 10,
    },
    'figsize': {
        'single': (6, 5),
        'double': (12, 5),
        'triple': (15, 5),
        'square': (8, 8),
        'large': (16, 12),
    },
}

# =============================================================================
# STATISTICAL PARAMETERS
# =============================================================================

STATS_PARAMS = {
    'alpha': 0.05,               # Significance level
    'correction': 'fdr_bh',      # Multiple comparison correction
    'n_permutations': 5000,      # For permutation tests
    'bootstrap_samples': 10000,  # For confidence intervals
    'effect_size': 'cohens_d',   # Effect size measure
}

# =============================================================================
# FILE NAMING CONVENTIONS
# =============================================================================

def get_wm_fmri_path(subject, run='LR'):
    """Get path to WM task fMRI data (S1200 release naming)"""
    return (DATA_ROOT / subject / "MNINonLinear" / "Results" /
            f"tfMRI_WM_{run}" / f"tfMRI_WM_{run}_Atlas_MSMAll.dtseries.nii")

def get_wm_evs_path(subject, run='LR'):
    """Get path to WM task EVs directory"""
    return DATA_ROOT / subject / "MNINonLinear" / "Results" / f"tfMRI_WM_{run}" / "EVs"

def get_wm_stats_path(subject, run='LR'):
    """Get path to WM behavioral stats"""
    return DATA_ROOT / subject / "MNINonLinear" / "Results" / f"tfMRI_WM_{run}" / "EVs" / "WM_Stats.csv"

def get_movement_path(subject, run='LR'):
    """Get path to movement parameters"""
    return (DATA_ROOT / subject / "MNINonLinear" / "Results" / 
            f"tfMRI_WM_{run}" / "Movement_RelativeRMS_mean.txt")

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

import logging

def setup_logging(name, level=logging.INFO):
    """Setup logging for a module"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(LOGS_DIR / f"{name}.log")
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

