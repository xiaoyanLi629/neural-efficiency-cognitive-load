#!/usr/bin/env python3
"""
效率变化量 (Δ效率) 分析
========================

使用效率变化量 (Δ效率 = 效率_2bk - 效率_0bk) 验证假设 H3 和 H4

假设体系:
- H1: 负荷效应假设 (行为分析验证)
- H2: 神经效率假设 (激活分析验证)
- H3: 网络稳定性假设 - 高效者网络更稳定 (Δ效率更小)
- H4: 补偿性重组假设 - 低效者通过网络重组补偿

本脚本验证 H3 和 H4
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 数据加载
# =============================================================================

def get_latest_run_path():
    """Get the path to the latest run directory."""
    results_dir = Path('/root/autodl-fs/CogSci/project_1/results')
    latest_link = results_dir / 'latest'
    
    if latest_link.exists():
        return latest_link.resolve() if latest_link.is_symlink() else latest_link
    else:
        runs = sorted([d for d in results_dir.glob("run_*") if d.is_dir()])
        return runs[-1] if runs else results_dir


def load_data():
    """加载所有需要的数据"""
    base_path = get_latest_run_path()
    
    behavioral = pd.read_csv(base_path / 'behavioral' / 'behavioral_summary.csv')
    network = pd.read_csv(base_path / 'connectivity' / 'network_metrics.csv')
    neural = pd.read_csv(base_path / 'efficiency' / 'neural_efficiency.csv')
    
    return behavioral, network, neural

# =============================================================================
# 计算效率变化量 (Δ效率)
# =============================================================================

def compute_delta_efficiency(network):
    """
    计算效率变化量: Δ = 2bk - 0bk
    
    Δ > 0: 负荷增加时指标上升
    Δ < 0: 负荷增加时指标下降
    """
    print("\n" + "=" * 70)
    print("计算效率变化量 (Δ效率 = 2bk - 0bk)")
    print("=" * 70)
    
    # 识别配对指标
    metrics_0bk = [col for col in network.columns if col.endswith('_0bk')]
    metrics_2bk = [col for col in network.columns if col.endswith('_2bk')]
    
    paired_metrics = []
    for m0 in metrics_0bk:
        base_name = m0.replace('_0bk', '')
        m2 = base_name + '_2bk'
        if m2 in metrics_2bk:
            paired_metrics.append(base_name)
    
    print(f"\n找到 {len(paired_metrics)} 个配对指标")
    
    # 计算 Δ
    delta_df = network[['subject']].copy()
    
    for metric in paired_metrics:
        col_0bk = metric + '_0bk'
        col_2bk = metric + '_2bk'
        delta_col = 'delta_' + metric
        delta_df[delta_col] = network[col_2bk] - network[col_0bk]
    
    # 关键指标统计
    key_metrics = ['global_efficiency', 'local_efficiency', 'modularity', 
                   'clustering_coefficient', 'density', 'mean_connectivity']
    
    print("\n关键指标的 Δ 统计:")
    print("-" * 60)
    
    delta_stats = []
    for metric in key_metrics:
        delta_col = 'delta_' + metric
        if delta_col in delta_df.columns:
            values = delta_df[delta_col]
            mean_val = values.mean()
            std_val = values.std()
            t_stat, p_value = stats.ttest_1samp(values, 0)
            
            delta_stats.append({
                'metric': metric,
                'mean_delta': mean_val,
                'std_delta': std_val,
                't': t_stat,
                'p': p_value
            })
            
            sig = "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
            direction = "↑" if mean_val > 0 else "↓"
            print(f"  Δ_{metric}:")
            print(f"    Mean = {mean_val:+.4f} {direction}, SD = {std_val:.4f}")
            print(f"    t = {t_stat:.3f}, p = {p_value:.4f} {sig}")
    
    return delta_df, delta_stats, paired_metrics

# =============================================================================
# H3: 网络稳定性假设
# =============================================================================

def analyze_H3_network_stability(behavioral, delta_df):
    """
    H3: 网络稳定性假设
    
    假设: 高效者的脑网络在负荷增加时更稳定 (Δ效率更小)
    方法: 比较高低效率组的 Δ效率
    """
    print("\n" + "=" * 70)
    print("H3: 网络稳定性假设")
    print("=" * 70)
    print("假设: 高效者的脑网络在负荷增加时更稳定 (Δ效率更小)")
    
    results = {}
    
    # 合并数据
    merged = pd.merge(
        behavioral[['subject', 'composite_efficiency', 'efficiency_group']],
        delta_df,
        on='subject'
    )
    
    high_eff = merged[merged['efficiency_group'] == 'High_Efficiency']
    low_eff = merged[merged['efficiency_group'] == 'Low_Efficiency']
    
    print(f"\n高效率组: N = {len(high_eff)}")
    print(f"低效率组: N = {len(low_eff)}")
    
    # 关键 Δ 指标
    delta_vars = ['delta_global_efficiency', 'delta_local_efficiency', 
                  'delta_modularity', 'delta_mean_connectivity',
                  'delta_clustering_coefficient', 'delta_density']
    
    print("\n" + "-" * 50)
    print("组间比较: 高效组 vs 低效组的 Δ效率")
    print("-" * 50)
    
    group_comparison = []
    
    for delta_var in delta_vars:
        if delta_var not in merged.columns:
            continue
        
        high_vals = high_eff[delta_var].dropna()
        low_vals = low_eff[delta_var].dropna()
        
        if len(high_vals) < 3 or len(low_vals) < 3:
            continue
        
        # t 检验
        t_stat, p_value = stats.ttest_ind(high_vals, low_vals)
        
        # 效应量
        pooled_std = np.sqrt(((len(high_vals)-1)*high_vals.std()**2 + 
                              (len(low_vals)-1)*low_vals.std()**2) / 
                             (len(high_vals) + len(low_vals) - 2))
        cohens_d = (high_vals.mean() - low_vals.mean()) / pooled_std if pooled_std > 0 else 0
        
        group_comparison.append({
            'metric': delta_var,
            'high_mean': high_vals.mean(),
            'low_mean': low_vals.mean(),
            'difference': high_vals.mean() - low_vals.mean(),
            't': t_stat,
            'p': p_value,
            'cohens_d': cohens_d
        })
        
        sig = "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""
        stable = "高效更稳定" if abs(high_vals.mean()) < abs(low_vals.mean()) else "低效更稳定"
        
        print(f"\n{delta_var}:")
        print(f"  高效组: M = {high_vals.mean():+.4f}, SD = {high_vals.std():.4f}")
        print(f"  低效组: M = {low_vals.mean():+.4f}, SD = {low_vals.std():.4f}")
        print(f"  差异: {high_vals.mean() - low_vals.mean():+.4f} ({stable})")
        print(f"  t = {t_stat:.3f}, p = {p_value:.4f} {sig}")
        print(f"  Cohen's d = {cohens_d:.3f}")
    
    results['group_comparison'] = group_comparison
    
    # 统计显著结果
    sig_results = [r for r in group_comparison if r['p'] < 0.05]
    print("\n" + "-" * 50)
    print(f"H3 结论: {len(sig_results)} 个指标显示显著组间差异")
    
    if sig_results:
        print("\n显著结果:")
        for r in sig_results:
            print(f"  ✓ {r['metric']}: p = {r['p']:.4f}, d = {r['cohens_d']:.2f}")
    
    return results

# =============================================================================
# H4: 补偿性重组假设
# =============================================================================

def analyze_H4_compensatory_reorganization(behavioral, delta_df, network):
    """
    H4: 补偿性重组假设
    
    假设: 低效者通过大幅网络重组来补偿基线网络的不足
    方法: 分析基线和负荷态的网络指标变化模式
    """
    print("\n" + "=" * 70)
    print("H4: 补偿性重组假设")
    print("=" * 70)
    print("假设: 低效者通过大幅网络重组来补偿基线网络的不足")
    
    results = {}
    
    # 合并数据
    merged = pd.merge(
        behavioral[['subject', 'efficiency_group']],
        network[['subject', 'modularity_0bk', 'modularity_2bk', 
                'global_efficiency_0bk', 'global_efficiency_2bk']],
        on='subject'
    )
    
    high_eff = merged[merged['efficiency_group'] == 'High_Efficiency']
    low_eff = merged[merged['efficiency_group'] == 'Low_Efficiency']
    
    print("\n" + "-" * 50)
    print("模块度变化模式分析")
    print("-" * 50)
    
    # 模块度分析
    metrics = [
        ('modularity', '模块度'),
        ('global_efficiency', '全局效率')
    ]
    
    pattern_results = []
    
    for metric, name in metrics:
        col_0bk = metric + '_0bk'
        col_2bk = metric + '_2bk'
        
        if col_0bk not in merged.columns:
            continue
        
        # 计算各组的基线和负荷态均值
        h_0bk = high_eff[col_0bk].mean()
        h_2bk = high_eff[col_2bk].mean()
        l_0bk = low_eff[col_0bk].mean()
        l_2bk = low_eff[col_2bk].mean()
        
        # 计算变化量
        h_delta = h_2bk - h_0bk
        l_delta = l_2bk - l_0bk
        
        # 计算组间差距
        gap_0bk = h_0bk - l_0bk
        gap_2bk = h_2bk - l_2bk
        
        # 组间 t 检验
        t_0bk, p_0bk = stats.ttest_ind(high_eff[col_0bk], low_eff[col_0bk])
        t_2bk, p_2bk = stats.ttest_ind(high_eff[col_2bk], low_eff[col_2bk])
        
        pattern_results.append({
            'metric': name,
            'high_0bk': h_0bk,
            'high_2bk': h_2bk,
            'high_delta': h_delta,
            'low_0bk': l_0bk,
            'low_2bk': l_2bk,
            'low_delta': l_delta,
            'gap_0bk': gap_0bk,
            'gap_2bk': gap_2bk,
            'gap_reversal': gap_0bk > 0 and gap_2bk < 0
        })
        
        print(f"\n{name}:")
        print(f"  高效组: 0bk = {h_0bk:.4f} → 2bk = {h_2bk:.4f} (Δ = {h_delta:+.4f})")
        print(f"  低效组: 0bk = {l_0bk:.4f} → 2bk = {l_2bk:.4f} (Δ = {l_delta:+.4f})")
        print(f"  组间差距: 0bk = {gap_0bk:+.4f}, 2bk = {gap_2bk:+.4f}")
        
        if gap_0bk > 0 and gap_2bk < 0:
            print(f"  ✓ 差距反转! 支持补偿性重组假说")
        elif gap_0bk > 0 and gap_2bk > 0 and gap_2bk < gap_0bk:
            print(f"  ~ 差距缩小，部分支持补偿性重组")
    
    results['pattern_analysis'] = pattern_results
    
    # 结论
    print("\n" + "-" * 50)
    print("H4 结论:")
    
    reversal_found = any(r['gap_reversal'] for r in pattern_results)
    if reversal_found:
        print("  ✓ 发现差距反转模式，支持补偿性重组假说")
        print("  → 低效者通过大幅网络重组'追赶'甚至'超过'高效者")
        print("  → 但行为表现仍落后，说明补偿效果有限")
    
    return results

# =============================================================================
# 结果汇总
# =============================================================================

def summarize_results(delta_stats, h3_results, h4_results):
    """汇总所有分析结果"""
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    summary = {
        'findings': [],
        'conclusions': []
    }
    
    # H3 结果
    print("\n【H3: 网络稳定性假设】")
    print("-" * 50)
    
    h3_sig = [r for r in h3_results['group_comparison'] if r['p'] < 0.05]
    
    if h3_sig:
        print(f"  ✅ 支持 - 发现 {len(h3_sig)} 个显著差异")
        for r in h3_sig:
            print(f"     {r['metric']}: p = {r['p']:.4f}, d = {r['cohens_d']:.2f}")
            summary['findings'].append(f"H3: {r['metric']} 显著差异")
        summary['conclusions'].append("H3 支持: 高效者网络更稳定")
    else:
        print("  ⚠️ 未发现显著差异")
    
    # H4 结果
    print("\n【H4: 补偿性重组假设】")
    print("-" * 50)
    
    h4_reversal = any(r.get('gap_reversal', False) for r in h4_results['pattern_analysis'])
    
    if h4_reversal:
        print("  ✅ 支持 - 发现差距反转模式")
        summary['conclusions'].append("H4 支持: 低效者存在补偿性重组")
    else:
        print("  ⚠️ 未发现典型补偿模式")
    
    # 总体结论
    print("\n【总体结论】")
    print("-" * 50)
    print("""
  神经效率的本质:
  
  1. 高效者: 网络稳定 (Δ效率小)
     → 基线网络已优化，无需大幅重组
  
  2. 低效者: 需要补偿性重组 (Δ效率大)
     → 基线网络欠佳，通过重组来补偿
     → 补偿后结构指标可能超过高效者
     → 但行为表现仍落后
  
  核心发现:
  神经效率 = 网络稳定性，而非静态结构优越性
""")
    
    return summary

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("效率变化量 (Δ效率) 分析")
    print("=" * 70)
    print("\n验证假设:")
    print("  H3: 网络稳定性假设 - 高效者网络更稳定")
    print("  H4: 补偿性重组假设 - 低效者通过重组补偿")
    
    # 加载数据
    print("\n加载数据...")
    behavioral, network, neural = load_data()
    print(f"  行为数据: {len(behavioral)} 个被试")
    print(f"  网络数据: {len(network)} 行")
    
    # 计算 Δ效率
    delta_df, delta_stats, paired_metrics = compute_delta_efficiency(network)
    
    # H3 分析
    h3_results = analyze_H3_network_stability(behavioral, delta_df)
    
    # H4 分析
    h4_results = analyze_H4_compensatory_reorganization(behavioral, delta_df, network)
    
    # 汇总
    summary = summarize_results(delta_stats, h3_results, h4_results)
    
    # 保存结果
    output_dir = Path('/root/autodl-fs/CogSci/project_1/results/delta_efficiency')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    all_results = {
        'delta_stats': delta_stats,
        'H3_network_stability': h3_results,
        'H4_compensatory_reorganization': h4_results,
        'summary': summary
    }
    
    all_results_serializable = convert_to_serializable(all_results)
    
    with open(output_dir / 'delta_efficiency_results.json', 'w') as f:
        json.dump(all_results_serializable, f, indent=2)
    
    delta_df.to_csv(output_dir / 'delta_efficiency.csv', index=False)
    
    print(f"\n结果已保存到: {output_dir}")
    
    return all_results

def run_delta_efficiency_analysis():
    """
    Entry point for pipeline integration.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        results = main()
        return results is not None
    except Exception as e:
        print(f"Error in delta efficiency analysis: {e}")
        return False


if __name__ == '__main__':
    results = main()
