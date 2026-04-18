# Network Stability as the Hallmark of Neural Efficiency Under Cognitive Load

Analysis pipeline and LaTeX source for the IEEE BIBM 2026 manuscript.

---

## Project Overview

This project investigates how cognitive load modulates neural efficiency during working memory tasks using Human Connectome Project (HCP) data. We examine whether high-performing individuals show more efficient neural resource utilization and discover that **network stability** is the key marker of neural efficiency.

### Key Findings

- **Neural efficiency = Network Stability**: Efficient brains maintain stable network organization under cognitive load
- **Compensatory Reorganization**: Less efficient brains undergo larger network reorganization to cope with increased demands
- **All 4 hypotheses supported** by experimental results
- **AI/ML validation**: Machine learning classifiers achieve up to 70% accuracy in distinguishing efficiency groups

## Research Hypotheses

| Hypothesis | Name | Description | Result |
|------------|------|-------------|--------|
| **H1** | Load Effect | Cognitive load impairs behavioral performance | ✅ Supported |
| **H2** | Neural Efficiency | High-efficiency individuals show precise activation patterns | ✅ Supported |
| **H3** | Network Stability | High-efficiency individuals show smaller network changes (Δ ≈ 0) | ✅ Supported |
| **H4** | Compensatory Reorganization | Low-efficiency individuals show larger network reorganization | ✅ Supported |

## Data

- **Source**: Human Connectome Project (HCP) S1200
- **Subjects**: 200 participants
- **Task**: N-back Working Memory Task (0-back vs 2-back)
- **Modality**: Task fMRI (tfMRI_WM)

## Project Structure

```
project_1/
├── IEEE_manuscript/            # IEEE BIBM 2026 LaTeX source + compiled PDF
├── README.md                   # This file
├── configs/
│   ├── config.py               # Paths, parameters, ROI definitions
│   └── schaefer_atlas.py       # Standardized Schaefer 2018 atlas definitions
├── scripts/
│   ├── s01_behavioral_analysis.py    # Stage 1: Behavioral analysis (H1)
│   ├── s02_activation_analysis.py    # Stage 2: GLM activation analysis (H2)
│   ├── s03_connectivity_analysis.py  # Stage 3: Functional connectivity
│   ├── s04_efficiency_metrics.py     # Stage 4: Neural efficiency metrics
│   ├── s05_statistical_analysis.py   # Stage 5: Hypothesis testing
│   ├── s06_unified_visualization.py  # Stage 6: Basic visualization (Fig 01-10)
│   ├── s07_delta_efficiency_analysis.py  # Stage 7: Δ efficiency analysis (H3, H4)
│   ├── s08_delta_efficiency_visualization.py # Stage 8: H3/H4 figures (Fig 13-16)
│   ├── s09_advanced_visualization.py # Advanced figures (Fig 17-22)
│   └── s10_ai_analysis.py            # Stage 9: AI/ML classification analysis
├── run_full_pipeline.py        # Main execution script
└── results/
    └── latest -> run_YYYYMMDD_HHMMSS/  # Symlink to latest run
        ├── behavioral/         # Behavioral metrics
        ├── activation/         # ROI activation values
        ├── connectivity/       # Network metrics
        ├── efficiency/         # Integrated efficiency measures + AI results
        ├── delta_efficiency/   # Δ efficiency analysis (H3, H4)
        └── figures/            # All visualization outputs
```

## Usage

### Full Pipeline

```bash
# Run complete analysis (all 9 stages)
python run_full_pipeline.py

# Run specific stages
python run_full_pipeline.py --stage 1 4 5 6

# Skip existing outputs
python run_full_pipeline.py --skip-existing

# Run only AI analysis
python run_full_pipeline.py --stage 9
```

### Quick Presets

```bash
# Behavioral analysis only (fastest)
python run_full_pipeline.py --quick behavioral

# Full analysis
python run_full_pipeline.py --quick full
```

## Analysis Pipeline

| Stage | Name | Description | Output |
|-------|------|-------------|--------|
| 1 | Behavioral Analysis | Extract performance metrics, compute efficiency | behavioral_summary.csv |
| 2 | Activation Analysis | GLM-based ROI activation | roi_activation.csv |
| 3 | Connectivity Analysis | Functional connectivity & graph metrics | network_metrics.csv |
| 4 | Efficiency Metrics | Integrate brain and behavior | neural_efficiency.csv |
| 5 | Statistical Analysis | Hypothesis testing with FDR correction | hypothesis_tests.json |
| 6 | Visualization | Publication figures (Fig 01-10) | fig01-10.png/svg |
| 7 | Delta Analysis | Δ efficiency for H3/H4 | delta_efficiency.csv |
| 8 | H3/H4 Visualization | Network stability figures (Fig 13-16) | fig13-16.png/svg |
| 9 | **AI/ML Analysis** | Machine learning classification | ai_classification_results.json |

## AI/ML Analysis (Stage 9)

The project includes advanced machine learning analysis for efficiency group classification:

### Traditional ML Classifiers
- Logistic Regression (L2 regularization)
- Support Vector Machine (RBF kernel)
- Random Forest (100 trees)
- Gradient Boosting (50 estimators)
- Multi-layer Perceptron (64-32-16 architecture)

### Deep Learning Models
- **Feature Attention Network**: Neural network with learnable feature-level attention weights
- **Graph Neural Network (GAT)**: Graph Attention Network operating on brain connectivity graphs

### Classification Results

| Model | Accuracy | AUC |
|-------|----------|-----|
| Gradient Boosting | 70.0% | 0.630 |
| Logistic Regression | 65.0% | 0.660 |
| Random Forest | 60.0% | 0.555 |
| SVM (RBF) | 50.0% | 0.620 |
| MLP | 45.0% | 0.530 |

### Top Important Features
1. IES 2-back (Inverse Efficiency Score) - 0.052
2. Accuracy cost (behavioral load effect) - 0.038
3. Angular_R activation (0-back) - 0.037
4. Modularity (2-back condition) - 0.030
5. ACC_L activation (0-back) - 0.024

## Key Figures

| Figure | Description | Hypothesis |
|--------|-------------|------------|
| Fig 01 | Behavioral load effect (Raincloud) | H1 |
| Fig 02 | Group comparison (6-panel) | H2 |
| Fig 03 | Glass brain + activation heatmap | H2 |
| Fig 04 | Brain-behavior correlation matrix | H3, H4 |
| Fig 09a-d | Glass brain activation series | H2 |
| Fig 13 | Network trajectory plot | H3, H4 |
| Fig 14 | Gap reversal visualization | H4 |
| Fig 15 | Radar chart comparison | H3, H4 |
| Fig 16a-b | Glass brain (High/Low groups) | H3, H4 |
| Fig 17 | Connectivity matrix (5-panel) | H3, H4 |
| Fig 18 | ROI activation map | H2 |
| Fig 19 | Chord diagram | H3, H4 |
| Fig 20 | Brain-behavior correlation | H2, H3 |
| Fig 21 | Sankey flow diagram | H1-H4 |
| Fig 22 | Summary panel | H1-H4 |
| **Fig AI** | ML model comparison & feature importance | AI validation |

## Technical Improvements (v2.0)

### 1. Fixed Visualization Data
- Glass Brain visualizations now use real connectivity data instead of simulated matrices
- Added `build_connectivity_matrix_from_data()` and `compute_delta_connectivity_matrix()` functions

### 2. Standardized ROI Definitions
- Created `configs/schaefer_atlas.py` based on Schaefer et al. (2018) 100-parcel 7-network parcellation
- Provides validated MNI coordinates and network assignments

### 3. AI/ML Integration
- Added comprehensive machine learning pipeline in `scripts/s10_ai_analysis.py`
- Includes traditional ML, attention networks, and graph neural networks
- Feature importance analysis for interpretability

## Documentation

For detailed methodology, formulas, and figure explanations, see the manuscript PDF and `.tex` source in `IEEE_manuscript/`.

## Citation

Citation details will be provided upon acceptance.

## License

This project is for academic research purposes.

---

*Last Updated: January 2026*
*Results Directory: run_20260103_105523*
