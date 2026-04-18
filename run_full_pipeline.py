#!/usr/bin/env python3
"""
=============================================================================
Neural Efficiency Under Cognitive Load: Full Analysis Pipeline
=============================================================================

BIBM 2026 Submission
Title: "Network Stability as the Hallmark of Neural Efficiency Under Cognitive Load"

This script runs the complete analysis pipeline for investigating how cognitive
load modulates neural efficiency during working memory tasks.

PIPELINE STAGES:
    1. Behavioral Analysis: Extract performance metrics, compute efficiency indices
    2. Activation Analysis: GLM-based analysis of task-evoked brain activity
    3. Connectivity Analysis: Functional connectivity and graph theory metrics
    4. Efficiency Metrics: Integrate brain and behavior into efficiency measures
    5. Statistical Analysis: Hypothesis testing with multiple comparison correction
    6. Visualization: Generate publication-quality figures (Fig 01-10)
    7. Delta Analysis: Δ efficiency analysis for H3 and H4 hypotheses
    8. H3/H4 Visualization: Advanced figures for network stability hypotheses (Fig 11-16)
    9. AI/ML Analysis: Advanced machine learning classification and interpretability

USAGE:
    python run_full_pipeline.py [--stage STAGE] [--skip-existing] [--verbose]
    
    --stage: Run specific stage only (1-8, or 'all')
    --skip-existing: Skip stages with existing output files
    --verbose: Enable verbose logging

OUTPUT:
    All results are saved in project_1/results/
    Figures are saved in project_1/results/figures/

=============================================================================
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config module
from configs import config
from configs.config import setup_logging, init_output_dirs, reset_run_timestamp

# Parse --run-dir early (before init) to allow reusing an existing run directory
import sys as _sys
_run_dir_arg = None
for i, arg in enumerate(_sys.argv):
    if arg == '--run-dir' and i + 1 < len(_sys.argv):
        _run_dir_arg = _sys.argv[i + 1]
        break

if _run_dir_arg:
    # Reuse existing run directory
    from pathlib import Path as _P
    run_name = _P(_run_dir_arg).name if '/' in _run_dir_arg else _run_dir_arg
    config._RUN_TIMESTAMP = run_name.replace('run_', '')
    output_dirs = init_output_dirs(use_timestamp=True)
else:
    # Fresh run
    reset_run_timestamp()
    output_dirs = init_output_dirs(use_timestamp=True)

# Now import the directory paths (they have been updated by init_output_dirs)
from configs.config import (
    RESULTS_DIR, FIGURES_DIR, LOGS_DIR,
    BEHAVIORAL_DIR, ACTIVATION_DIR, CONNECTIVITY_DIR, EFFICIENCY_DIR,
    DELTA_EFFICIENCY_DIR,
)

logger = setup_logging('pipeline')
logger.info(f"Output directory: {RESULTS_DIR}")

# =============================================================================
# PIPELINE STAGES
# =============================================================================

def run_stage1_behavioral():
    """Stage 1: Behavioral Analysis"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 1: BEHAVIORAL ANALYSIS")
    logger.info("="*70)
    
    from scripts.s01_behavioral_analysis import run_behavioral_analysis
    
    start_time = time.time()
    df, stats = run_behavioral_analysis()
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 1 completed in {elapsed:.1f} seconds")
    
    return df is not None


def run_stage2_activation():
    """Stage 2: Activation Analysis"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 2: ACTIVATION ANALYSIS")
    logger.info("="*70)
    
    from scripts.s02_activation_analysis import run_activation_analysis
    
    start_time = time.time()
    df, stats = run_activation_analysis()
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 2 completed in {elapsed:.1f} seconds")
    
    return df is not None


def run_stage3_connectivity():
    """Stage 3: Connectivity Analysis"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 3: CONNECTIVITY ANALYSIS")
    logger.info("="*70)
    
    from scripts.s03_connectivity_analysis import run_connectivity_analysis
    
    start_time = time.time()
    df, stats = run_connectivity_analysis()
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 3 completed in {elapsed:.1f} seconds")
    
    return df is not None


def run_stage4_efficiency():
    """Stage 4: Neural Efficiency Metrics"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 4: NEURAL EFFICIENCY METRICS")
    logger.info("="*70)
    
    from scripts.s04_efficiency_metrics import run_efficiency_analysis
    
    start_time = time.time()
    df, results = run_efficiency_analysis()
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 4 completed in {elapsed:.1f} seconds")
    
    return df is not None


def run_stage5_statistics():
    """Stage 5: Statistical Analysis"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 5: STATISTICAL ANALYSIS")
    logger.info("="*70)
    
    from scripts.s05_statistical_analysis import run_statistical_analysis
    
    start_time = time.time()
    results, effect_sizes = run_statistical_analysis()
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 5 completed in {elapsed:.1f} seconds")
    
    return results is not None


def run_stage6_visualization():
    """Stage 6: Visualization"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 6: VISUALIZATION")
    logger.info("="*70)
    
    from scripts.s06_unified_visualization import generate_all_figures
    
    start_time = time.time()
    figures = generate_all_figures()
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 6 completed in {elapsed:.1f} seconds")
    
    return figures is not None


def run_stage7_delta_analysis():
    """Stage 7: Delta Efficiency Analysis (H3, H4)"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 7: DELTA EFFICIENCY ANALYSIS (H3, H4)")
    logger.info("="*70)
    
    from scripts.s07_delta_efficiency_analysis import run_delta_efficiency_analysis
    
    start_time = time.time()
    success = run_delta_efficiency_analysis()
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 7 completed in {elapsed:.1f} seconds")
    
    return success


def run_stage8_h3h4_visualization():
    """Stage 8: H3/H4 Visualization"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 8: H3/H4 VISUALIZATION")
    logger.info("="*70)
    
    from scripts.s08_delta_efficiency_visualization import run_h3_h4_visualization
    
    start_time = time.time()
    # Save to the same figures directory as stage 6
    success = run_h3_h4_visualization(output_dir=FIGURES_DIR)
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 8 completed in {elapsed:.1f} seconds")
    
    return success


def run_stage9_ai_analysis():
    """Stage 9: AI/ML Analysis"""
    logger.info("\n" + "="*70)
    logger.info("STAGE 9: AI/ML ANALYSIS")
    logger.info("="*70)
    
    from scripts.s10_ai_analysis import run_ai_analysis
    
    start_time = time.time()
    results = run_ai_analysis()
    elapsed = time.time() - start_time
    
    logger.info(f"Stage 9 completed in {elapsed:.1f} seconds")
    
    return results is not None


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_full_pipeline(stages='all', skip_existing=False, verbose=False):
    """
    Run the complete analysis pipeline.
    
    Parameters:
        stages: Which stages to run ('all', or list of stage numbers)
        skip_existing: Skip stages with existing output
        verbose: Enable verbose output
    """
    pipeline_start = time.time()
    
    # Header
    print("\n" + "="*70)
    print(" NEURAL EFFICIENCY ANALYSIS PIPELINE ")
    print(" BIBM 2026: Network Stability as the Hallmark of Neural Efficiency")
    print("="*70)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Output:  {RESULTS_DIR}")
    print("="*70 + "\n")
    
    # Define stages
    stage_functions = {
        1: ('Behavioral Analysis', run_stage1_behavioral, BEHAVIORAL_DIR / 'behavioral_summary.csv'),
        2: ('Activation Analysis', run_stage2_activation, ACTIVATION_DIR / 'roi_activation.csv'),
        3: ('Connectivity Analysis', run_stage3_connectivity, CONNECTIVITY_DIR / 'network_metrics.csv'),
        4: ('Efficiency Metrics', run_stage4_efficiency, EFFICIENCY_DIR / 'neural_efficiency.csv'),
        5: ('Statistical Analysis', run_stage5_statistics, EFFICIENCY_DIR / 'hypothesis_tests.json'),
        6: ('Visualization', run_stage6_visualization, FIGURES_DIR / 'fig01_behavioral_load_effect.png'),
        7: ('Delta Analysis (H3,H4)', run_stage7_delta_analysis, DELTA_EFFICIENCY_DIR / 'delta_efficiency.csv'),
        8: ('H3/H4 Visualization', run_stage8_h3h4_visualization, FIGURES_DIR / 'fig11_delta_heatmap.png'),
        9: ('AI/ML Analysis', run_stage9_ai_analysis, EFFICIENCY_DIR / 'ai_classification_results.json'),
    }
    
    # Determine which stages to run
    if stages == 'all':
        stages_to_run = list(stage_functions.keys())
    else:
        stages_to_run = [int(s) for s in stages] if isinstance(stages, list) else [int(stages)]
    
    # Run stages
    results = {}
    
    for stage_num in stages_to_run:
        if stage_num not in stage_functions:
            logger.warning(f"Unknown stage: {stage_num}")
            continue
        
        name, func, output_file = stage_functions[stage_num]
        
        # Check if should skip
        if skip_existing and output_file.exists():
            logger.info(f"Skipping Stage {stage_num} ({name}): output exists")
            results[stage_num] = True
            continue
        
        # Run stage
        try:
            success = func()
            results[stage_num] = success
        except Exception as e:
            logger.error(f"Stage {stage_num} ({name}) failed: {e}")
            results[stage_num] = False
            
            # Continue or abort?
            if stage_num < 4:  # Early stages are critical
                logger.error("Critical stage failed. Aborting pipeline.")
                break
    
    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    
    print("\n" + "="*70)
    print(" PIPELINE SUMMARY")
    print("="*70)
    
    for stage_num, success in results.items():
        name = stage_functions[stage_num][0]
        status = "✓ COMPLETE" if success else "✗ FAILED"
        print(f"  Stage {stage_num}: {name:30s} [{status}]")
    
    print("-"*70)
    print(f"  Total time: {pipeline_elapsed:.1f} seconds ({pipeline_elapsed/60:.1f} minutes)")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Output locations
    print("\nOUTPUT FILES:")
    print(f"  Behavioral:   {BEHAVIORAL_DIR}")
    print(f"  Activation:   {ACTIVATION_DIR}")
    print(f"  Connectivity: {CONNECTIVITY_DIR}")
    print(f"  Efficiency:   {EFFICIENCY_DIR}")
    print(f"  Figures:      {FIGURES_DIR}")
    print(f"  Logs:         {LOGS_DIR}")
    print()
    
    return all(results.values())


# =============================================================================
# QUICK RUN FUNCTIONS
# =============================================================================

def quick_behavioral_only():
    """Run only behavioral analysis (fastest)."""
    return run_full_pipeline(stages=[1, 6])


def quick_brain_behavior():
    """Run behavioral + efficiency + visualization."""
    return run_full_pipeline(stages=[1, 4, 5, 6])


def full_analysis():
    """Run complete pipeline."""
    return run_full_pipeline(stages='all')


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Neural Efficiency Analysis Pipeline for BIBM 2026',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py                    # Run all stages
  python run_full_pipeline.py --stage 1          # Run only behavioral analysis
  python run_full_pipeline.py --stage 1 4 5 6    # Run specific stages
  python run_full_pipeline.py --skip-existing    # Skip completed stages
        """
    )
    
    parser.add_argument('--stage', nargs='+', default='all',
                       help='Stage(s) to run: 1-6 or "all" (default: all)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip stages with existing output files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--run-dir', type=str, default=None,
                       help='Reuse an existing run directory (e.g. run_20260415_122619)')
    parser.add_argument('--quick', choices=['behavioral', 'brain-behavior', 'full'],
                       help='Quick run presets')
    
    args = parser.parse_args()
    
    # Handle quick presets
    if args.quick == 'behavioral':
        success = quick_behavioral_only()
    elif args.quick == 'brain-behavior':
        success = quick_brain_behavior()
    elif args.quick == 'full':
        success = full_analysis()
    else:
        # Parse stages
        if args.stage == 'all' or args.stage == ['all']:
            stages = 'all'
        else:
            stages = args.stage
        
        success = run_full_pipeline(
            stages=stages,
            skip_existing=args.skip_existing,
            verbose=args.verbose
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

