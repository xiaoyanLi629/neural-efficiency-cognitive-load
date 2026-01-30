"""
=============================================================================
Step 1: Behavioral Data Analysis
=============================================================================

PURPOSE:
    Extract and analyze behavioral performance data (accuracy and RT) from the
    N-back working memory task to:
    1. Characterize the cognitive load effect (0-back vs 2-back)
    2. Calculate neural efficiency indices
    3. Split subjects into High-Efficiency vs Low-Efficiency groups

RESEARCH QUESTIONS ADDRESSED:
    Q1.1: Does cognitive load significantly impair behavioral performance?
    Q1.2: What is the individual variability in load-dependent performance cost?
    Q1.3: Can we identify distinct efficiency profiles among subjects?

OUTPUT:
    - behavioral_summary.csv: All behavioral metrics per subject
    - group_assignments.csv: Subject groupings (High/Low efficiency)
    - behavioral_stats.json: Statistical test results

=============================================================================
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, pearsonr, spearmanr
import json
import warnings
warnings.filterwarnings('ignore')

from configs import config
from configs.config import (
    DATA_ROOT, SUBJECTS, BEHAVIORAL_PARAMS,
    get_wm_stats_path, setup_logging
)
# Use config.BEHAVIORAL_DIR to get the current timestamped directory

logger = setup_logging('behavioral_analysis')

# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def load_wm_stats(subject, run='LR'):
    """
    Load WM task behavioral statistics for a subject.
    
    Returns DataFrame with columns: Value, ConditionName, Measure
    """
    stats_path = get_wm_stats_path(subject, run)
    if not stats_path.exists():
        logger.warning(f"Stats file not found for {subject} run {run}")
        return None
    
    try:
        df = pd.read_csv(stats_path)
        return df
    except Exception as e:
        logger.error(f"Error loading stats for {subject}: {e}")
        return None


def extract_behavioral_metrics(subject):
    """
    Extract key behavioral metrics for a subject across both runs.
    
    Metrics extracted:
    - Accuracy (ACC): Proportion correct responses
    - Reaction Time (RT): Median response time in ms
    - Inverse Efficiency Score (IES): RT / ACC (lower = more efficient)
    - Load Cost: Performance change from 0-back to 2-back
    
    Returns dict with all metrics.
    """
    metrics = {
        'subject': subject,
        # 0-back metrics
        'acc_0bk': [], 'rt_0bk': [],
        # 2-back metrics  
        'acc_2bk': [], 'rt_2bk': [],
    }
    
    for run in ['LR', 'RL']:
        df = load_wm_stats(subject, run)
        if df is None:
            continue
            
        # Extract 0-back accuracy (average across stimulus types)
        acc_0bk = df[(df['ConditionName'].str.startswith('0BK_')) & 
                     (df['Measure'] == 'ACC')]['Value'].values
        if len(acc_0bk) > 0:
            metrics['acc_0bk'].append(np.mean(acc_0bk))
        
        # Extract 2-back accuracy
        acc_2bk = df[(df['ConditionName'].str.startswith('2BK_')) & 
                     (df['Measure'] == 'ACC')]['Value'].values
        if len(acc_2bk) > 0:
            metrics['acc_2bk'].append(np.mean(acc_2bk))
        
        # Extract 0-back RT
        rt_0bk = df[(df['ConditionName'].str.startswith('0BK_')) & 
                    (df['Measure'] == 'MEDIAN_RT')]['Value'].values
        if len(rt_0bk) > 0:
            metrics['rt_0bk'].append(np.mean(rt_0bk))
        
        # Extract 2-back RT
        rt_2bk = df[(df['ConditionName'].str.startswith('2BK_')) & 
                    (df['Measure'] == 'MEDIAN_RT')]['Value'].values
        if len(rt_2bk) > 0:
            metrics['rt_2bk'].append(np.mean(rt_2bk))
    
    # Average across runs
    result = {'subject': subject}
    
    for key in ['acc_0bk', 'acc_2bk', 'rt_0bk', 'rt_2bk']:
        if len(metrics[key]) > 0:
            result[key] = np.mean(metrics[key])
        else:
            result[key] = np.nan
    
    return result


def compute_efficiency_metrics(df):
    """
    Compute derived efficiency metrics from raw behavioral data.
    
    Metrics computed:
    1. Inverse Efficiency Score (IES): RT / ACC
       - Lower values indicate better efficiency
       - Combines speed and accuracy into single metric
    
    2. Load Cost (Accuracy): ACC_0bk - ACC_2bk
       - Positive values indicate accuracy drop under load
    
    3. Load Cost (RT): RT_2bk - RT_0bk  
       - Positive values indicate RT slowing under load
    
    4. Efficiency Ratio: IES_0bk / IES_2bk
       - Values > 1 indicate efficiency maintained under load
    
    5. Composite Efficiency Score (z-scored)
       - Combines multiple metrics into single index
    """
    # Inverse Efficiency Scores
    df['ies_0bk'] = df['rt_0bk'] / df['acc_0bk']
    df['ies_2bk'] = df['rt_2bk'] / df['acc_2bk']
    
    # Load costs
    df['acc_cost'] = df['acc_0bk'] - df['acc_2bk']  # Positive = accuracy drop
    df['rt_cost'] = df['rt_2bk'] - df['rt_0bk']      # Positive = RT increase
    df['ies_cost'] = df['ies_2bk'] - df['ies_0bk']   # Positive = efficiency drop
    
    # Efficiency ratio (how well efficiency is maintained)
    df['efficiency_ratio'] = df['ies_0bk'] / df['ies_2bk']
    
    # Compute composite efficiency score (z-scored)
    # Higher = more efficient (better ACC, faster RT, lower costs)
    z_acc_2bk = stats.zscore(df['acc_2bk'].fillna(df['acc_2bk'].mean()))
    z_rt_2bk = -stats.zscore(df['rt_2bk'].fillna(df['rt_2bk'].mean()))  # Negative because lower is better
    z_acc_cost = -stats.zscore(df['acc_cost'].fillna(df['acc_cost'].mean()))  # Negative because lower cost is better
    z_rt_cost = -stats.zscore(df['rt_cost'].fillna(df['rt_cost'].mean()))
    
    df['composite_efficiency'] = (z_acc_2bk + z_rt_2bk + z_acc_cost + z_rt_cost) / 4
    
    return df


def assign_efficiency_groups(df, method='median'):
    """
    Assign subjects to High-Efficiency vs Low-Efficiency groups.
    
    Methods:
    - 'median': Split at median composite efficiency score
    - 'tercile': Use top and bottom terciles
    - 'kmeans': K-means clustering (k=2)
    
    Returns DataFrame with 'efficiency_group' column.
    """
    if method == 'median':
        threshold = df['composite_efficiency'].median()
        df['efficiency_group'] = np.where(
            df['composite_efficiency'] >= threshold,
            'High_Efficiency',
            'Low_Efficiency'
        )
    
    elif method == 'tercile':
        q33 = df['composite_efficiency'].quantile(0.33)
        q66 = df['composite_efficiency'].quantile(0.66)
        
        df['efficiency_group'] = 'Medium_Efficiency'
        df.loc[df['composite_efficiency'] >= q66, 'efficiency_group'] = 'High_Efficiency'
        df.loc[df['composite_efficiency'] <= q33, 'efficiency_group'] = 'Low_Efficiency'
    
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        
        X = df['composite_efficiency'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Assign labels based on cluster centers
        if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
            label_map = {0: 'High_Efficiency', 1: 'Low_Efficiency'}
        else:
            label_map = {1: 'High_Efficiency', 0: 'Low_Efficiency'}
        
        df['efficiency_group'] = [label_map[l] for l in labels]
    
    return df


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def run_statistical_tests(df):
    """
    Perform statistical tests on behavioral data.
    
    Tests performed:
    1. Paired t-test: 0-back vs 2-back (within-subject load effect)
    2. Effect sizes: Cohen's d for load effects
    3. Correlation: Load cost vs baseline efficiency
    4. Group comparison: High vs Low efficiency groups
    
    Returns dict with all statistical results.
    """
    results = {}
    
    # 1. Load effect on accuracy (paired t-test)
    # Get valid pairs (both conditions have data)
    acc_valid = df[['acc_0bk', 'acc_2bk']].dropna()
    t_acc, p_acc = ttest_rel(acc_valid['acc_0bk'], acc_valid['acc_2bk'])
    d_acc = (acc_valid['acc_0bk'].mean() - acc_valid['acc_2bk'].mean()) / acc_valid[['acc_0bk', 'acc_2bk']].stack().std()
    
    results['load_effect_accuracy'] = {
        'test': 'paired_t_test',
        't_statistic': float(t_acc),
        'p_value': float(p_acc),
        'cohens_d': float(d_acc),
        'mean_0bk': float(acc_valid['acc_0bk'].mean()),
        'mean_2bk': float(acc_valid['acc_2bk'].mean()),
        'n_pairs': len(acc_valid),
        'interpretation': 'significant' if p_acc < 0.05 else 'not_significant'
    }
    
    # 2. Load effect on RT
    rt_valid = df[['rt_0bk', 'rt_2bk']].dropna()
    t_rt, p_rt = ttest_rel(rt_valid['rt_0bk'], rt_valid['rt_2bk'])
    d_rt = (rt_valid['rt_2bk'].mean() - rt_valid['rt_0bk'].mean()) / rt_valid[['rt_0bk', 'rt_2bk']].stack().std()
    
    results['load_effect_rt'] = {
        'test': 'paired_t_test',
        't_statistic': float(t_rt),
        'p_value': float(p_rt),
        'cohens_d': float(d_rt),
        'mean_0bk': float(rt_valid['rt_0bk'].mean()),
        'mean_2bk': float(rt_valid['rt_2bk'].mean()),
        'n_pairs': len(rt_valid),
        'interpretation': 'significant' if p_rt < 0.05 else 'not_significant'
    }
    
    # 3. Correlation: Baseline efficiency vs load cost
    valid_idx = ~(df['ies_0bk'].isna() | df['ies_cost'].isna())
    if valid_idx.sum() >= 5:
        r, p = pearsonr(df.loc[valid_idx, 'ies_0bk'], df.loc[valid_idx, 'ies_cost'])
        results['baseline_loadcost_correlation'] = {
            'test': 'pearson_correlation',
            'r': float(r),
            'p_value': float(p),
            'interpretation': 'Higher baseline efficiency predicts lower load cost' if r < 0 else 'Higher baseline efficiency predicts higher load cost'
        }
    
    # 4. Group comparison
    high_eff = df[df['efficiency_group'] == 'High_Efficiency']
    low_eff = df[df['efficiency_group'] == 'Low_Efficiency']
    
    for metric in ['acc_2bk', 'rt_2bk', 'acc_cost', 'rt_cost']:
        t, p = ttest_ind(high_eff[metric].dropna(), low_eff[metric].dropna())
        pooled_std = np.sqrt((high_eff[metric].var() + low_eff[metric].var()) / 2)
        d = (high_eff[metric].mean() - low_eff[metric].mean()) / pooled_std
        
        results[f'group_comparison_{metric}'] = {
            'test': 'independent_t_test',
            't_statistic': float(t),
            'p_value': float(p),
            'cohens_d': float(d),
            'mean_high_eff': float(high_eff[metric].mean()),
            'mean_low_eff': float(low_eff[metric].mean())
        }
    
    return results


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def run_behavioral_analysis():
    """
    Main function to run complete behavioral analysis pipeline.
    
    Pipeline:
    1. Load behavioral data for all subjects
    2. Compute efficiency metrics
    3. Assign efficiency groups
    4. Run statistical tests
    5. Save results
    """
    logger.info("="*60)
    logger.info("Starting Behavioral Analysis Pipeline")
    logger.info("="*60)
    
    # Step 1: Extract behavioral metrics for all subjects
    logger.info(f"Extracting behavioral metrics for {len(SUBJECTS)} subjects...")
    
    all_metrics = []
    for subject in SUBJECTS:
        metrics = extract_behavioral_metrics(subject)
        all_metrics.append(metrics)
        logger.info(f"  {subject}: ACC_0bk={metrics.get('acc_0bk', 'NA'):.3f}, "
                   f"ACC_2bk={metrics.get('acc_2bk', 'NA'):.3f}")
    
    df = pd.DataFrame(all_metrics)
    
    # Step 2: Compute efficiency metrics
    logger.info("Computing efficiency metrics...")
    df = compute_efficiency_metrics(df)
    
    # Step 3: Assign efficiency groups
    logger.info("Assigning efficiency groups...")
    df = assign_efficiency_groups(df, method=BEHAVIORAL_PARAMS['group_split'])
    
    group_counts = df['efficiency_group'].value_counts()
    logger.info(f"  High Efficiency: {group_counts.get('High_Efficiency', 0)} subjects")
    logger.info(f"  Low Efficiency: {group_counts.get('Low_Efficiency', 0)} subjects")
    
    # Step 4: Run statistical tests
    logger.info("Running statistical tests...")
    stats_results = run_statistical_tests(df)
    
    # Log key findings
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS:")
    logger.info("="*60)
    
    load_acc = stats_results['load_effect_accuracy']
    logger.info(f"Load Effect (Accuracy): t={load_acc['t_statistic']:.2f}, "
               f"p={load_acc['p_value']:.4f}, d={load_acc['cohens_d']:.2f}")
    
    load_rt = stats_results['load_effect_rt']
    logger.info(f"Load Effect (RT): t={load_rt['t_statistic']:.2f}, "
               f"p={load_rt['p_value']:.4f}, d={load_rt['cohens_d']:.2f}")
    
    # Step 5: Save results
    logger.info("\nSaving results...")
    
    # Save behavioral summary
    df.to_csv(config.BEHAVIORAL_DIR / 'behavioral_summary.csv', index=False)
    logger.info(f"  Saved: behavioral_summary.csv")
    
    # Save group assignments
    group_df = df[['subject', 'composite_efficiency', 'efficiency_group']]
    group_df.to_csv(config.BEHAVIORAL_DIR / 'group_assignments.csv', index=False)
    logger.info(f"  Saved: group_assignments.csv")
    
    # Save statistical results
    with open(config.BEHAVIORAL_DIR / 'behavioral_stats.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    logger.info(f"  Saved: behavioral_stats.json")
    
    logger.info("\n" + "="*60)
    logger.info("Behavioral Analysis Complete!")
    logger.info("="*60)
    
    return df, stats_results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    df, stats = run_behavioral_analysis()
    
    # Print summary table
    print("\n" + "="*80)
    print("BEHAVIORAL SUMMARY TABLE")
    print("="*80)
    print(df[['subject', 'acc_0bk', 'acc_2bk', 'rt_0bk', 'rt_2bk', 
              'composite_efficiency', 'efficiency_group']].to_string(index=False))

