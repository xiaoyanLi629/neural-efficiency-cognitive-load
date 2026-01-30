"""
=============================================================================
Step 4: Neural Efficiency Index Computation
=============================================================================

PURPOSE:
    Integrate behavioral and neural measures to compute comprehensive
    neural efficiency indices that capture the brain-behavior relationship
    under varying cognitive load conditions.

RESEARCH QUESTIONS ADDRESSED:
    Q4.1: Can we quantify neural efficiency as a ratio of behavioral
          output to neural cost?
    Q4.2: How does neural efficiency change with cognitive load?
    Q4.3: Do activation-based and connectivity-based efficiency
          metrics capture the same underlying construct?

THEORETICAL FRAMEWORK:
    Neural Efficiency = Behavioral Performance / Neural Resource Consumption
    
    Three complementary operationalizations:
    1. Activation Efficiency: Performance / Regional Activation
    2. Connectivity Efficiency: Performance / Network Disorganization  
    3. Composite Efficiency: Weighted combination of multiple metrics

OUTPUT:
    - neural_efficiency.csv: All efficiency metrics per subject
    - efficiency_correlations.json: Inter-metric correlations
    - brain_behavior_models.json: Regression model results

=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import json
import warnings
warnings.filterwarnings('ignore')

from configs import config
from configs.config import setup_logging
# Use config.config.BEHAVIORAL_DIR, config.config.ACTIVATION_DIR, config.config.CONNECTIVITY_DIR, 
# config.config.EFFICIENCY_DIR for timestamped paths

logger = setup_logging('efficiency_metrics')

# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """
    Load behavioral, activation, and connectivity data.
    """
    data = {}
    
    # Behavioral data
    beh_file = config.BEHAVIORAL_DIR / 'behavioral_summary.csv'
    if beh_file.exists():
        data['behavioral'] = pd.read_csv(beh_file)
        logger.info(f"Loaded behavioral data: {len(data['behavioral'])} subjects")
    else:
        logger.error("Behavioral data not found!")
        return None
    
    # Activation data
    act_file = config.ACTIVATION_DIR / 'roi_activation.csv'
    if act_file.exists():
        data['activation'] = pd.read_csv(act_file)
        logger.info(f"Loaded activation data: {len(data['activation'])} subjects")
    else:
        logger.warning("Activation data not found")
        data['activation'] = None
    
    # Connectivity data
    conn_file = config.CONNECTIVITY_DIR / 'network_metrics.csv'
    if conn_file.exists():
        data['connectivity'] = pd.read_csv(conn_file)
        logger.info(f"Loaded connectivity data: {len(data['connectivity'])} subjects")
    else:
        logger.warning("Connectivity data not found")
        data['connectivity'] = None
    
    return data


def merge_datasets(data):
    """
    Merge all datasets on subject ID.
    Preserves efficiency_group from behavioral data.
    """
    df = data['behavioral'].copy()
    
    # Store efficiency_group before merging (to avoid duplication issues)
    efficiency_group = df[['subject', 'efficiency_group']].copy() if 'efficiency_group' in df.columns else None
    
    if data['activation'] is not None:
        # Select key activation columns, exclude efficiency_group to avoid duplication
        act_cols = ['subject'] + [c for c in data['activation'].columns 
                                   if any(x in c for x in ['activation', 'load_effect'])
                                   and 'efficiency_group' not in c]
        df = df.merge(data['activation'][act_cols], on='subject', how='left')
    
    if data['connectivity'] is not None:
        # Select key connectivity columns, exclude efficiency_group to avoid duplication
        conn_cols = ['subject'] + [c for c in data['connectivity'].columns 
                                    if any(x in c for x in ['efficiency', 'within', 'FPN', 'DMN', 'modularity'])
                                    and 'efficiency_group' not in c]
        conn_cols = list(set(conn_cols))  # Remove duplicates
        available_cols = [c for c in conn_cols if c in data['connectivity'].columns]
        df = df.merge(data['connectivity'][available_cols], on='subject', how='left')
    
    # Clean up any duplicated efficiency_group columns from merging
    # Remove efficiency_group_x, efficiency_group_y if they exist
    cols_to_drop = [c for c in df.columns if c.startswith('efficiency_group_')]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Ensure efficiency_group is present from behavioral data
    if efficiency_group is not None and 'efficiency_group' not in df.columns:
        df = df.merge(efficiency_group, on='subject', how='left')
    
    return df


# =============================================================================
# NEURAL EFFICIENCY COMPUTATIONS
# =============================================================================

def compute_activation_efficiency(df):
    """
    Compute activation-based neural efficiency.
    
    Formula: Efficiency = Behavioral Performance / Brain Activation
    
    Higher values = more efficient (better performance with less activation)
    
    Operationalizations:
    1. Simple ratio: ACC / mean(ROI activation)
    2. Inverse Efficiency: ACC / (RT * activation)
    3. Load-specific efficiency for 0-back and 2-back
    """
    # Find activation columns
    act_cols_0bk = [c for c in df.columns if 'activation_0bk' in c]
    act_cols_2bk = [c for c in df.columns if 'activation_2bk' in c]
    
    # Mean activation across key ROIs
    if act_cols_0bk:
        df['mean_activation_0bk'] = df[act_cols_0bk].mean(axis=1)
    else:
        df['mean_activation_0bk'] = np.nan
    
    if act_cols_2bk:
        df['mean_activation_2bk'] = df[act_cols_2bk].mean(axis=1)
    else:
        df['mean_activation_2bk'] = np.nan
    
    # Activation efficiency (performance / activation)
    # Use absolute activation to avoid sign issues
    eps = 1e-6  # Small constant to avoid division by zero
    
    # 0-back efficiency
    df['activation_efficiency_0bk'] = df['acc_0bk'] / (np.abs(df['mean_activation_0bk']) + eps)
    
    # 2-back efficiency
    df['activation_efficiency_2bk'] = df['acc_2bk'] / (np.abs(df['mean_activation_2bk']) + eps)
    
    # Efficiency change with load
    df['activation_efficiency_change'] = df['activation_efficiency_2bk'] - df['activation_efficiency_0bk']
    
    # Efficiency maintenance ratio
    df['activation_efficiency_ratio'] = df['activation_efficiency_2bk'] / (df['activation_efficiency_0bk'] + eps)
    
    return df


def compute_connectivity_efficiency(df):
    """
    Compute connectivity-based neural efficiency.
    
    Efficient brains show:
    1. High global efficiency (short path lengths)
    2. High FPN integration (strong within-network connectivity)
    3. Strong FPN-DMN segregation (anticorrelation)
    
    Formula: Connectivity Efficiency = Performance * Network Organization
    """
    # Network efficiency (if available)
    if 'global_efficiency_0bk' in df.columns:
        df['net_efficiency_0bk'] = df['acc_0bk'] * df['global_efficiency_0bk']
    if 'global_efficiency_2bk' in df.columns:
        df['net_efficiency_2bk'] = df['acc_2bk'] * df['global_efficiency_2bk']
    
    # FPN integration efficiency
    if 'within_FPN_0bk' in df.columns:
        df['fpn_integration_0bk'] = df['acc_0bk'] * df['within_FPN_0bk']
    if 'within_FPN_2bk' in df.columns:
        df['fpn_integration_2bk'] = df['acc_2bk'] * df['within_FPN_2bk']
    
    # FPN-DMN segregation (anticorrelation is beneficial)
    # More negative FPN-DMN = better segregation = higher efficiency
    if 'FPN_DMN_0bk' in df.columns:
        df['segregation_efficiency_0bk'] = df['acc_0bk'] * (-df['FPN_DMN_0bk'])
    if 'FPN_DMN_2bk' in df.columns:
        df['segregation_efficiency_2bk'] = df['acc_2bk'] * (-df['FPN_DMN_2bk'])
    
    return df


def compute_composite_efficiency(df):
    """
    Compute composite neural efficiency index.
    
    Combines multiple efficiency metrics into single index using
    z-score standardization and equal weighting.
    """
    efficiency_cols = [c for c in df.columns if 'efficiency' in c.lower() 
                       and c not in ['efficiency_group', 'composite_efficiency']]
    
    # Standardize each efficiency metric
    scaler = StandardScaler()
    
    # 0-back composite
    cols_0bk = [c for c in efficiency_cols if '0bk' in c]
    if cols_0bk:
        valid_0bk = df[cols_0bk].dropna(axis=1, how='all')
        if len(valid_0bk.columns) > 0:
            z_0bk = pd.DataFrame(
                scaler.fit_transform(valid_0bk.fillna(valid_0bk.mean())),
                columns=valid_0bk.columns
            )
            df['neural_efficiency_0bk'] = z_0bk.mean(axis=1)
    
    # 2-back composite
    cols_2bk = [c for c in efficiency_cols if '2bk' in c]
    if cols_2bk:
        valid_2bk = df[cols_2bk].dropna(axis=1, how='all')
        if len(valid_2bk.columns) > 0:
            z_2bk = pd.DataFrame(
                scaler.fit_transform(valid_2bk.fillna(valid_2bk.mean())),
                columns=valid_2bk.columns
            )
            df['neural_efficiency_2bk'] = z_2bk.mean(axis=1)
    
    # Overall composite (combining behavioral and neural)
    composite_cols = ['composite_efficiency']  # From behavioral analysis
    if 'neural_efficiency_2bk' in df.columns:
        composite_cols.append('neural_efficiency_2bk')
    
    if len(composite_cols) > 1:
        valid_composite = df[composite_cols].dropna(axis=1, how='all')
        if len(valid_composite.columns) > 1:
            z_composite = pd.DataFrame(
                scaler.fit_transform(valid_composite.fillna(valid_composite.mean())),
                columns=valid_composite.columns
            )
            df['overall_efficiency'] = z_composite.mean(axis=1)
    
    return df


# =============================================================================
# BRAIN-BEHAVIOR RELATIONSHIP ANALYSIS
# =============================================================================

def analyze_brain_behavior_correlations(df):
    """
    Analyze correlations between brain measures and behavior.
    """
    results = {}
    
    # Neural predictors
    neural_cols = [c for c in df.columns if any(x in c for x in 
                   ['activation', 'efficiency', 'within', 'global', 'FPN', 'DMN'])]
    
    # Behavioral outcomes
    behavioral_cols = ['acc_0bk', 'acc_2bk', 'rt_0bk', 'rt_2bk', 
                       'acc_cost', 'rt_cost', 'ies_0bk', 'ies_2bk']
    
    for neural in neural_cols:
        for behav in behavioral_cols:
            if neural in df.columns and behav in df.columns:
                valid = df[[neural, behav]].dropna()
                # Ensure numeric types
                try:
                    x = pd.to_numeric(valid[neural], errors='coerce')
                    y = pd.to_numeric(valid[behav], errors='coerce')
                    mask = ~(x.isna() | y.isna())
                    x, y = x[mask], y[mask]
                    
                    if len(x) >= 5:
                        r, p = pearsonr(x, y)
                        rho, p_rho = spearmanr(x, y)
                    
                    results[f'{neural}_vs_{behav}'] = {
                        'pearson_r': float(r),
                        'pearson_p': float(p),
                        'spearman_rho': float(rho),
                        'spearman_p': float(p_rho),
                            'n': int(len(x))
                    }
                except Exception:
                    continue
    
    return results


def build_predictive_models(df):
    """
    Build regression models predicting behavior from neural measures.
    
    Models:
    1. Activation → Performance
    2. Connectivity → Performance
    3. Combined Neural → Performance
    4. Neural Efficiency → Load Cost
    """
    results = {}
    
    # Model 1: Activation predicting 2-back accuracy
    act_cols = [c for c in df.columns if 'activation_2bk' in c and 'efficiency' not in c]
    if act_cols and 'acc_2bk' in df.columns:
        X = df[act_cols].fillna(df[act_cols].mean())
        y = df['acc_2bk'].fillna(df['acc_2bk'].mean())
        
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=min(5, len(df)), scoring='r2')
        model.fit(X, y)
        
        results['activation_predicts_accuracy'] = {
            'cv_r2_mean': float(np.mean(scores)),
            'cv_r2_std': float(np.std(scores)),
            'full_r2': float(model.score(X, y)),
            'features': act_cols,
            'coefficients': {c: float(coef) for c, coef in zip(act_cols, model.coef_)}
        }
    
    # Model 2: Connectivity predicting performance
    conn_cols = [c for c in df.columns if any(x in c for x in ['global_efficiency', 'within_FPN', 'FPN_DMN']) 
                 and '2bk' in c]
    if conn_cols and 'acc_2bk' in df.columns:
        valid_conn = [c for c in conn_cols if c in df.columns]
        if valid_conn:
            X = df[valid_conn].fillna(df[valid_conn].mean())
            y = df['acc_2bk'].fillna(df['acc_2bk'].mean())
            
            model = LinearRegression()
            scores = cross_val_score(model, X, y, cv=min(5, len(df)), scoring='r2')
            model.fit(X, y)
            
            results['connectivity_predicts_accuracy'] = {
                'cv_r2_mean': float(np.mean(scores)),
                'cv_r2_std': float(np.std(scores)),
                'full_r2': float(model.score(X, y)),
                'features': valid_conn
            }
    
    # Model 3: Neural efficiency predicting load cost
    if 'neural_efficiency_0bk' in df.columns and 'acc_cost' in df.columns:
        X = df[['neural_efficiency_0bk']].fillna(df['neural_efficiency_0bk'].mean())
        y = df['acc_cost'].fillna(df['acc_cost'].mean())
        
        r, p = pearsonr(X.values.flatten(), y.values)
        
        results['baseline_efficiency_predicts_loadcost'] = {
            'r': float(r),
            'p': float(p),
            'interpretation': 'Higher baseline efficiency predicts smaller accuracy cost' if r < 0 else 'Higher baseline efficiency predicts larger accuracy cost'
        }
    
    return results


def analyze_efficiency_group_differences(df):
    """
    Compare neural metrics between efficiency groups.
    """
    results = {}
    
    if 'efficiency_group' not in df.columns:
        return results
    
    high = df[df['efficiency_group'] == 'High_Efficiency']
    low = df[df['efficiency_group'] == 'Low_Efficiency']
    
    # Compare neural metrics
    neural_cols = [c for c in df.columns if any(x in c for x in 
                   ['activation', 'efficiency', 'global', 'within', 'FPN', 'DMN']) 
                   and c != 'efficiency_group']
    
    for col in neural_cols:
        if col in df.columns:
            h = high[col].dropna()
            l = low[col].dropna()
            
            if len(h) >= 3 and len(l) >= 3:
                t, p = ttest_ind(h, l)
                d = (h.mean() - l.mean()) / np.sqrt((h.var() + l.var()) / 2)
                
                results[f'group_diff_{col}'] = {
                    't': float(t),
                    'p': float(p),
                    'd': float(d),
                    'mean_high': float(h.mean()),
                    'mean_low': float(l.mean()),
                    'direction': 'High > Low' if h.mean() > l.mean() else 'Low > High'
                }
    
    return results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_efficiency_analysis():
    """
    Main efficiency analysis pipeline.
    """
    logger.info("="*60)
    logger.info("Starting Neural Efficiency Analysis")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading all data...")
    data = load_all_data()
    if data is None:
        return None, None
    
    # Merge datasets
    logger.info("Merging datasets...")
    df = merge_datasets(data)
    logger.info(f"Merged data: {len(df)} subjects, {len(df.columns)} variables")
    
    # Compute efficiency metrics
    logger.info("Computing activation efficiency...")
    df = compute_activation_efficiency(df)
    
    logger.info("Computing connectivity efficiency...")
    df = compute_connectivity_efficiency(df)
    
    logger.info("Computing composite efficiency...")
    df = compute_composite_efficiency(df)
    
    # Analyze brain-behavior relationships
    logger.info("Analyzing brain-behavior correlations...")
    correlations = analyze_brain_behavior_correlations(df)
    
    logger.info("Building predictive models...")
    models = build_predictive_models(df)
    
    logger.info("Analyzing group differences...")
    group_diffs = analyze_efficiency_group_differences(df)
    
    # Compile all results
    all_results = {
        'correlations': correlations,
        'predictive_models': models,
        'group_differences': group_diffs
    }
    
    # Log key findings
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS:")
    logger.info("="*60)
    
    # Top correlations
    if correlations:
        top_corrs = sorted(correlations.items(), 
                          key=lambda x: abs(x[1].get('pearson_r', 0)), 
                          reverse=True)[:5]
        logger.info("\nTop Brain-Behavior Correlations:")
        for key, val in top_corrs:
            logger.info(f"  {key}: r={val['pearson_r']:.3f}, p={val['pearson_p']:.4f}")
    
    # Save results
    logger.info("\nSaving results...")
    df.to_csv(config.EFFICIENCY_DIR / 'neural_efficiency.csv', index=False)
    
    with open(config.EFFICIENCY_DIR / 'efficiency_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"  Saved: neural_efficiency.csv")
    logger.info(f"  Saved: efficiency_analysis.json")
    
    logger.info("\n" + "="*60)
    logger.info("Neural Efficiency Analysis Complete!")
    logger.info("="*60)
    
    return df, all_results


if __name__ == '__main__':
    df, results = run_efficiency_analysis()
    
    if df is not None:
        print("\n" + "="*80)
        print("EFFICIENCY METRICS SUMMARY")
        print("="*80)
        
        eff_cols = [c for c in df.columns if 'efficiency' in c.lower()]
        print(df[['subject'] + eff_cols[:10]].head(10).to_string())

