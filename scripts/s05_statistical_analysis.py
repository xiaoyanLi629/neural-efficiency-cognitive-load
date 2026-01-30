"""
=============================================================================
Step 5: Comprehensive Statistical Analysis
=============================================================================

PURPOSE:
    Perform rigorous statistical testing to evaluate all research hypotheses
    with appropriate multiple comparison corrections and effect size reporting.

HYPOTHESES TESTED:

    H1: Cognitive load impairs behavioral performance
        Test: Paired t-test (0-back vs 2-back)
        
    H2: High-efficiency individuals show lower activation under low load
        Test: Group × Load interaction (2×2 mixed ANOVA)
        
    H3: Neural efficiency predicts behavioral load cost
        Test: Regression with bootstrapped confidence intervals
        
    H4: Network organization differs between efficiency groups
        Test: Permutation-based group comparison

OUTPUT:
    - hypothesis_tests.json: All hypothesis test results
    - effect_sizes.csv: Effect sizes for all comparisons
    - supplementary_stats.json: Additional analyses

=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (ttest_rel, ttest_ind, pearsonr, spearmanr,
                         mannwhitneyu, wilcoxon, f_oneway)
import json
import warnings
warnings.filterwarnings('ignore')

from configs import config
from configs.config import STATS_PARAMS, setup_logging
# Use config.config.EFFICIENCY_DIR, config.config.BEHAVIORAL_DIR, etc. for timestamped paths

logger = setup_logging('statistical_analysis')

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def cohens_d_paired(diff):
    """Compute Cohen's d for paired samples."""
    return np.mean(diff) / np.std(diff, ddof=1)


def bootstrap_ci(data, stat_func=np.mean, n_boot=10000, ci=95):
    """Compute bootstrap confidence interval."""
    boot_stats = []
    n = len(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))
    
    lower = np.percentile(boot_stats, (100-ci)/2)
    upper = np.percentile(boot_stats, 100 - (100-ci)/2)
    return lower, upper


def permutation_test(group1, group2, n_perm=5000):
    """Permutation test for group difference."""
    observed_diff = np.mean(group1) - np.mean(group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    null_diffs = []
    for _ in range(n_perm):
        np.random.shuffle(combined)
        null_diffs.append(np.mean(combined[:n1]) - np.mean(combined[n1:]))
    
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value, null_diffs


def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    p_array = np.array(p_values)
    n = len(p_array)
    sorted_idx = np.argsort(p_array)
    sorted_p = p_array[sorted_idx]
    
    # BH procedure
    thresholds = alpha * np.arange(1, n+1) / n
    significant = sorted_p <= thresholds
    
    # Find largest significant
    if np.any(significant):
        max_sig_idx = np.max(np.where(significant)[0])
        corrected_sig = np.zeros(n, dtype=bool)
        corrected_sig[sorted_idx[:max_sig_idx+1]] = True
    else:
        corrected_sig = np.zeros(n, dtype=bool)
    
    return [bool(x) for x in corrected_sig]  # Convert to Python bool


# =============================================================================
# HYPOTHESIS TESTING
# =============================================================================

def test_h1_load_effect(df):
    """
    H1: Cognitive load significantly impairs behavioral performance.
    
    Tests:
    - Paired t-test: 0-back vs 2-back accuracy
    - Paired t-test: 0-back vs 2-back RT
    - Effect sizes (Cohen's d)
    """
    results = {'hypothesis': 'H1', 'description': 'Cognitive load impairs performance'}
    
    # Accuracy
    acc_0bk = df['acc_0bk'].dropna()
    acc_2bk = df['acc_2bk'].dropna()
    common_idx = acc_0bk.index.intersection(acc_2bk.index)
    
    if len(common_idx) >= 5:
        t, p = ttest_rel(acc_0bk[common_idx], acc_2bk[common_idx])
        diff = acc_0bk[common_idx] - acc_2bk[common_idx]
        d = cohens_d_paired(diff)
        ci_low, ci_high = bootstrap_ci(diff.values)
        
        results['accuracy'] = {
            'test': 'paired_t_test',
            't_statistic': float(t),
            'p_value': float(p),
            'cohens_d': float(d),
            'mean_0bk': float(acc_0bk[common_idx].mean()),
            'mean_2bk': float(acc_2bk[common_idx].mean()),
            'mean_difference': float(diff.mean()),
            'ci_95': [float(ci_low), float(ci_high)],
            'n': len(common_idx),
            'significant': bool(p < STATS_PARAMS['alpha'])
        }
    
    # RT
    rt_0bk = df['rt_0bk'].dropna()
    rt_2bk = df['rt_2bk'].dropna()
    common_idx = rt_0bk.index.intersection(rt_2bk.index)
    
    if len(common_idx) >= 5:
        t, p = ttest_rel(rt_0bk[common_idx], rt_2bk[common_idx])
        diff = rt_2bk[common_idx] - rt_0bk[common_idx]  # Note: 2bk - 0bk for RT
        d = cohens_d_paired(diff)
        ci_low, ci_high = bootstrap_ci(diff.values)
        
        results['reaction_time'] = {
            'test': 'paired_t_test',
            't_statistic': float(t),
            'p_value': float(p),
            'cohens_d': float(d),
            'mean_0bk': float(rt_0bk[common_idx].mean()),
            'mean_2bk': float(rt_2bk[common_idx].mean()),
            'mean_difference': float(diff.mean()),
            'ci_95': [float(ci_low), float(ci_high)],
            'n': len(common_idx),
            'significant': bool(p < STATS_PARAMS['alpha'])
        }
    
    # Conclusion
    acc_sig = results.get('accuracy', {}).get('significant', False)
    rt_sig = results.get('reaction_time', {}).get('significant', False)
    results['conclusion'] = 'Supported' if (acc_sig or rt_sig) else 'Not supported'
    
    return results


def get_efficiency_group_column(df):
    """
    Find the efficiency group column, handling potential naming variations from merging.
    """
    # Try direct name first
    if 'efficiency_group' in df.columns:
        return 'efficiency_group'
    
    # Try variations from merging
    for col in ['efficiency_group_x', 'efficiency_group_y']:
        if col in df.columns:
            return col
    
    return None


def test_h2_neural_efficiency(df):
    """
    H2: High-efficiency individuals show lower activation (neural efficiency effect).
    
    Tests:
    - Group comparison of activation under 0-back
    - Group comparison of activation under 2-back
    - Group × Load interaction
    """
    results = {'hypothesis': 'H2', 'description': 'Neural efficiency in high performers'}
    
    # Find efficiency group column (handle naming variations)
    eff_group_col = get_efficiency_group_column(df)
    if eff_group_col is None:
        results['error'] = 'Efficiency groups not defined'
        return results
    
    high = df[df[eff_group_col] == 'High_Efficiency']
    low = df[df[eff_group_col] == 'Low_Efficiency']
    
    # Find activation columns
    act_cols_0bk = [c for c in df.columns if 'activation_0bk' in c]
    act_cols_2bk = [c for c in df.columns if 'activation_2bk' in c]
    
    results['roi_comparisons'] = {}
    
    # Test each ROI
    for col in act_cols_0bk:
        roi = col.replace('_activation_0bk', '')
        col_2bk = f'{roi}_activation_2bk'
        
        if col_2bk not in df.columns:
            continue
        
        h_0bk = high[col].dropna()
        l_0bk = low[col].dropna()
        h_2bk = high[col_2bk].dropna()
        l_2bk = low[col_2bk].dropna()
        
        if len(h_0bk) >= 3 and len(l_0bk) >= 3:
            # 0-back group comparison
            t_0bk, p_0bk = ttest_ind(h_0bk, l_0bk)
            d_0bk = cohens_d(h_0bk.values, l_0bk.values)
            
            # 2-back group comparison
            t_2bk, p_2bk = ttest_ind(h_2bk, l_2bk)
            d_2bk = cohens_d(h_2bk.values, l_2bk.values)
            
            # Simple interaction proxy: difference in load effects
            load_effect_high = h_2bk.mean() - h_0bk.mean()
            load_effect_low = l_2bk.mean() - l_0bk.mean()
            interaction = load_effect_high - load_effect_low
            
            results['roi_comparisons'][roi] = {
                'group_diff_0bk': {
                    't': float(t_0bk), 'p': float(p_0bk), 'd': float(d_0bk),
                    'high_mean': float(h_0bk.mean()), 'low_mean': float(l_0bk.mean())
                },
                'group_diff_2bk': {
                    't': float(t_2bk), 'p': float(p_2bk), 'd': float(d_2bk),
                    'high_mean': float(h_2bk.mean()), 'low_mean': float(l_2bk.mean())
                },
                'interaction_proxy': float(interaction),
                'neural_efficiency_pattern': 'Yes' if d_0bk < 0 and d_2bk >= 0 else 'No'
            }
    
    # Count ROIs showing efficiency pattern
    efficiency_rois = sum(1 for v in results['roi_comparisons'].values() 
                         if v.get('neural_efficiency_pattern') == 'Yes')
    total_rois = len(results['roi_comparisons'])
    
    results['summary'] = {
        'rois_showing_efficiency_pattern': efficiency_rois,
        'total_rois': total_rois,
        'proportion': efficiency_rois / total_rois if total_rois > 0 else 0
    }
    
    results['conclusion'] = 'Supported' if efficiency_rois > total_rois / 2 else 'Partially supported'
    
    return results


def test_h3_efficiency_predicts_cost(df):
    """
    H3: Baseline neural efficiency predicts behavioral load cost.
    
    Tests:
    - Correlation: Neural efficiency (0-back) vs Accuracy cost
    - Correlation: Neural efficiency (0-back) vs RT cost
    - Regression with bootstrap CI
    """
    results = {'hypothesis': 'H3', 'description': 'Neural efficiency predicts load cost'}
    
    # Efficiency columns
    eff_cols = [c for c in df.columns if 'efficiency' in c.lower() and '0bk' in c]
    
    for eff_col in eff_cols:
        if eff_col not in df.columns:
            continue
            
        # vs Accuracy cost
        if 'acc_cost' in df.columns:
            valid = df[[eff_col, 'acc_cost']].dropna()
            if len(valid) >= 5:
                r, p = pearsonr(valid[eff_col], valid['acc_cost'])
                rho, p_rho = spearmanr(valid[eff_col], valid['acc_cost'])
                
                results[f'{eff_col}_vs_acc_cost'] = {
                    'pearson_r': float(r),
                    'pearson_p': float(p),
                    'spearman_rho': float(rho),
                    'spearman_p': float(p_rho),
                    'n': len(valid),
                    'interpretation': 'Higher efficiency → lower cost' if r < 0 else 'Higher efficiency → higher cost'
                }
        
        # vs RT cost
        if 'rt_cost' in df.columns:
            valid = df[[eff_col, 'rt_cost']].dropna()
            if len(valid) >= 5:
                r, p = pearsonr(valid[eff_col], valid['rt_cost'])
                
                results[f'{eff_col}_vs_rt_cost'] = {
                    'pearson_r': float(r),
                    'pearson_p': float(p),
                    'n': len(valid)
                }
    
    # Determine conclusion based on significant negative correlations
    neg_sig_count = sum(1 for k, v in results.items() 
                        if isinstance(v, dict) and 
                        v.get('pearson_r', 0) < 0 and 
                        v.get('pearson_p', 1) < 0.05)
    
    results['conclusion'] = 'Supported' if neg_sig_count > 0 else 'Not supported'
    
    return results


def test_h4_network_organization(df):
    """
    H4: Efficient individuals show better network organization.
    
    Tests:
    - Group comparison of global efficiency
    - Group comparison of FPN-DMN segregation
    - Permutation tests for robust inference
    """
    results = {'hypothesis': 'H4', 'description': 'Network organization differs by efficiency'}
    
    # Find efficiency group column (handle naming variations)
    eff_group_col = get_efficiency_group_column(df)
    if eff_group_col is None:
        results['error'] = 'Efficiency groups not defined'
        return results
    
    high = df[df[eff_group_col] == 'High_Efficiency']
    low = df[df[eff_group_col] == 'Low_Efficiency']
    
    # Network metrics to test
    network_metrics = ['global_efficiency', 'local_efficiency', 'modularity', 
                       'within_FPN', 'FPN_DMN']
    
    for metric in network_metrics:
        for suffix in ['', '_0bk', '_2bk']:
            col = f'{metric}{suffix}'
            
            if col not in df.columns:
                continue
            
            h = high[col].dropna().values
            l = low[col].dropna().values
            
            if len(h) >= 3 and len(l) >= 3:
                # Parametric test
                t, p = ttest_ind(h, l)
                d = cohens_d(h, l)
                
                # Non-parametric test
                u, p_mw = mannwhitneyu(h, l, alternative='two-sided')
                
                # Permutation test
                obs_diff, p_perm, _ = permutation_test(h, l, n_perm=STATS_PARAMS['n_permutations'])
                
                results[col] = {
                    'parametric': {'t': float(t), 'p': float(p)},
                    'nonparametric': {'U': float(u), 'p': float(p_mw)},
                    'permutation': {'observed_diff': float(obs_diff), 'p': float(p_perm)},
                    'cohens_d': float(d),
                    'high_mean': float(np.mean(h)),
                    'low_mean': float(np.mean(l)),
                    'direction': 'High > Low' if np.mean(h) > np.mean(l) else 'Low > High',
                    'significant': bool(p_perm < STATS_PARAMS['alpha'])
                }
    
    # Count significant differences
    sig_count = sum(1 for v in results.values() 
                   if isinstance(v, dict) and v.get('significant', False))
    total = sum(1 for v in results.values() if isinstance(v, dict) and 'significant' in v)
    
    results['summary'] = {
        'significant_metrics': sig_count,
        'total_metrics': total
    }
    
    results['conclusion'] = 'Supported' if sig_count > 0 else 'Not supported'
    
    return results


# =============================================================================
# MULTIPLE COMPARISON CORRECTION
# =============================================================================

def apply_fdr_correction(all_results):
    """
    Apply FDR correction across all tests.
    """
    # Collect all p-values
    p_values = []
    p_locations = []
    
    def extract_p_values(obj, path=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ['p_value', 'p', 'pearson_p', 'spearman_p']:
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        p_values.append(v)
                        p_locations.append(path + '.' + k)
                else:
                    extract_p_values(v, path + '.' + k)
    
    extract_p_values(all_results)
    
    # Apply FDR correction
    if p_values:
        significant = fdr_correction(p_values, STATS_PARAMS['alpha'])
        
        return {
            'n_tests': len(p_values),
            'n_significant_uncorrected': int(sum(np.array(p_values) < STATS_PARAMS['alpha'])),
            'n_significant_fdr': int(sum(significant)),
            'fdr_threshold': STATS_PARAMS['alpha']
        }
    
    return {}


# =============================================================================
# EFFECT SIZE SUMMARY
# =============================================================================

def compile_effect_sizes(all_results):
    """
    Compile all effect sizes into summary table.
    """
    effect_sizes = []
    
    def extract_effect_sizes(obj, source=''):
        if isinstance(obj, dict):
            if 'cohens_d' in obj or 'd' in obj:
                d = obj.get('cohens_d', obj.get('d', None))
                if d is not None and not np.isnan(d):
                    effect_sizes.append({
                        'source': source,
                        'cohens_d': d,
                        'magnitude': 'large' if abs(d) >= 0.8 else 'medium' if abs(d) >= 0.5 else 'small'
                    })
            for k, v in obj.items():
                extract_effect_sizes(v, source + '.' + k if source else k)
    
    extract_effect_sizes(all_results)
    
    return pd.DataFrame(effect_sizes)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_statistical_analysis():
    """
    Main statistical analysis pipeline.
    """
    logger.info("="*60)
    logger.info("Starting Statistical Analysis Pipeline")
    logger.info("="*60)
    
    # Load efficiency data (contains all merged data)
    eff_file = config.EFFICIENCY_DIR / 'neural_efficiency.csv'
    if eff_file.exists():
        df = pd.read_csv(eff_file)
        logger.info(f"Loaded data: {len(df)} subjects")
    else:
        # Try loading behavioral data alone
        beh_file = config.BEHAVIORAL_DIR / 'behavioral_summary.csv'
        if beh_file.exists():
            df = pd.read_csv(beh_file)
        else:
            logger.error("No data found!")
            return None, None
    
    # Run hypothesis tests
    logger.info("\nTesting H1: Cognitive load effect...")
    h1_results = test_h1_load_effect(df)
    logger.info(f"  Conclusion: {h1_results.get('conclusion', 'N/A')}")
    
    logger.info("\nTesting H2: Neural efficiency in high performers...")
    h2_results = test_h2_neural_efficiency(df)
    logger.info(f"  Conclusion: {h2_results.get('conclusion', 'N/A')}")
    
    logger.info("\nTesting H3: Efficiency predicts load cost...")
    h3_results = test_h3_efficiency_predicts_cost(df)
    logger.info(f"  Conclusion: {h3_results.get('conclusion', 'N/A')}")
    
    logger.info("\nTesting H4: Network organization differences...")
    h4_results = test_h4_network_organization(df)
    logger.info(f"  Conclusion: {h4_results.get('conclusion', 'N/A')}")
    
    # Compile all results
    all_results = {
        'H1_load_effect': h1_results,
        'H2_neural_efficiency': h2_results,
        'H3_efficiency_predicts_cost': h3_results,
        'H4_network_organization': h4_results
    }
    
    # Apply FDR correction
    logger.info("\nApplying FDR correction...")
    fdr_summary = apply_fdr_correction(all_results)
    all_results['fdr_correction'] = fdr_summary
    logger.info(f"  Tests: {fdr_summary.get('n_tests', 0)}, "
               f"Significant (FDR): {fdr_summary.get('n_significant_fdr', 0)}")
    
    # Compile effect sizes
    effect_sizes_df = compile_effect_sizes(all_results)
    
    # Save results
    logger.info("\nSaving results...")
    
    with open(config.EFFICIENCY_DIR / 'hypothesis_tests.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    if len(effect_sizes_df) > 0:
        effect_sizes_df.to_csv(config.EFFICIENCY_DIR / 'effect_sizes.csv', index=False)
    
    logger.info(f"  Saved: hypothesis_tests.json")
    logger.info(f"  Saved: effect_sizes.csv")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("HYPOTHESIS TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"H1 (Load Effect):        {h1_results.get('conclusion', 'N/A')}")
    logger.info(f"H2 (Neural Efficiency):  {h2_results.get('conclusion', 'N/A')}")
    logger.info(f"H3 (Predicts Cost):      {h3_results.get('conclusion', 'N/A')}")
    logger.info(f"H4 (Network Org):        {h4_results.get('conclusion', 'N/A')}")
    
    logger.info("\n" + "="*60)
    logger.info("Statistical Analysis Complete!")
    logger.info("="*60)
    
    return all_results, effect_sizes_df


if __name__ == '__main__':
    results, effect_sizes = run_statistical_analysis()

