#!/usr/bin/env python3
"""
=============================================================================
Step 10: Advanced AI/ML Analysis for Neural Efficiency
=============================================================================

PURPOSE:
    Apply state-of-the-art AI and machine learning methods to analyze neural
    efficiency patterns, including:
    
    1. Deep Neural Network Classifier for efficiency group prediction
    2. Graph Neural Network (GNN) for brain connectivity analysis
    3. Attention-based model for interpretable feature importance
    4. Gradient-based feature attribution (Integrated Gradients)
    5. Cross-validated model evaluation with proper statistical testing

THEORETICAL CONTRIBUTION:
    These AI methods provide:
    - Nonlinear pattern recognition beyond traditional statistics
    - Interpretable attention weights showing which brain regions drive predictions
    - Graph-level representations capturing network topology
    - Rigorous cross-validation for generalization assessment

OUTPUT:
    - ai_classification_results.json: Model performance metrics
    - ai_feature_importance.csv: Feature importance rankings
    - ai_attention_weights.csv: Attention-based interpretability
    - figures/fig_ai_*.png: AI analysis visualizations

=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, cross_val_score, 
    cross_val_predict, permutation_test_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Try importing deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cuda':
        print(f"[AI] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[AI] CUDA unavailable, falling back to CPU")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("Warning: PyTorch not available. Deep learning models will be skipped.")

# Try importing PyTorch Geometric for GNN
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. GNN analysis will be skipped.")

from configs import config
from configs.config import (
    setup_logging, HIGH_EFF, LOW_EFF, HIGH_EFF_LIGHT, LOW_EFF_LIGHT,
    ACCENT, FIGURE_PARAMS
)

logger = setup_logging('ai_analysis')

# =============================================================================
# CONFIGURATION
# =============================================================================

AI_CONFIG = {
    'random_state': 42,
    'n_cv_folds': 5,  # Use 5-fold CV for small sample
    'n_permutations': 200,  # p resolution 0.005, keeps runtime manageable
    'n_jobs': -1,           # parallelize permutation_test_score across cores
    'mlp_hidden_layers': (64, 32, 16),
    'mlp_max_iter': 1000,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 8,
    'patience': 15,
}

# Features used to define the efficiency label (median split on ies_2bk).
# Including them as predictors causes target leakage, so the primary
# classification run excludes ALL behavioral variables. The IES-only and
# full-feature variants are kept only as leakage-sanity comparisons.
LEAKAGE_FEATURES = [
    'acc_0bk', 'acc_2bk', 'rt_0bk', 'rt_2bk',
    'ies_0bk', 'ies_2bk', 'acc_cost', 'rt_cost',
    'ies_cost', 'efficiency_ratio', 'composite_efficiency',
]

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data_for_ai():
    """Load and prepare data for AI analysis."""
    logger.info("Loading data for AI analysis...")
    
    # Load neural efficiency data
    eff_file = config.EFFICIENCY_DIR / 'neural_efficiency.csv'
    if not eff_file.exists():
        # Try latest run
        latest = config.PROJECT_DIR / 'results' / 'latest' / 'efficiency' / 'neural_efficiency.csv'
        if latest.exists():
            eff_file = latest
        else:
            raise FileNotFoundError("Neural efficiency data not found")
    
    df = pd.read_csv(eff_file)
    logger.info(f"Loaded {len(df)} subjects")
    
    # Load connectivity data for graph features
    conn_file = config.CONNECTIVITY_DIR / 'network_metrics.csv'
    if not conn_file.exists():
        conn_file = config.PROJECT_DIR / 'results' / 'latest' / 'connectivity' / 'network_metrics.csv'
    
    if conn_file.exists():
        conn_df = pd.read_csv(conn_file)
        # Merge connectivity features. neural_efficiency.csv already carries
        # many of these columns (added by s04), so the merge previously
        # produced dozens of `_conn`-suffixed exact duplicates that inflated
        # the feature count. Only bring over truly new columns.
        existing_cols = set(df.columns)
        new_conn_cols = [c for c in conn_df.columns
                         if c != 'subject'
                         and 'efficiency_group' not in c
                         and c not in existing_cols]
        if new_conn_cols:
            df = df.merge(conn_df[['subject'] + new_conn_cols], on='subject',
                          how='left')
        dropped = len(conn_df.columns) - 1 - len(new_conn_cols)  # -1 for 'subject'
        if dropped > 0:
            logger.info(
                f"  Skipped {dropped} connectivity columns already present in "
                f"neural_efficiency.csv (avoids duplicate features)"
            )

    return df


def prepare_features(df, feature_type='neural_only'):
    """
    Prepare feature matrix and labels for classification.

    Args:
        df: DataFrame with all features.
        feature_type:
            'neural_only' (default): activation + connectivity features, with
                ALL behavioral variables removed to avoid target leakage (the
                efficiency_group label is derived from ies_2bk).
            'ies_only': ies_2bk alone — used only as a leakage sanity check
                (should give near-perfect accuracy by construction).
            'all_with_leakage': full 152-feature set including behavioral
                variables — kept for reference comparison to show the
                leakage-inflated accuracy.
            'behavioral', 'activation', 'connectivity': single-family subsets.

    Returns:
        X: Feature matrix
        y: Labels (0=Low, 1=High efficiency)
        feature_names: List of feature names
    """
    # Define feature groups
    behavioral_cols = ['acc_0bk', 'acc_2bk', 'rt_0bk', 'rt_2bk',
                       'ies_0bk', 'ies_2bk', 'acc_cost', 'rt_cost']

    activation_cols = [c for c in df.columns if 'activation' in c and 'efficiency' not in c]

    connectivity_cols = [c for c in df.columns if any(x in c for x in
                        ['global_efficiency', 'local_efficiency', 'modularity',
                         'within_', 'FPN', 'DMN', 'clustering', 'density'])]
    connectivity_cols = [c for c in connectivity_cols if 'efficiency_group' not in c]
    # Remove any leakage-adjacent derived scores that slipped in via substring match
    connectivity_cols = [c for c in connectivity_cols if c not in LEAKAGE_FEATURES]

    # Select features based on type
    if feature_type == 'behavioral':
        feature_cols = behavioral_cols
    elif feature_type == 'activation':
        feature_cols = activation_cols
    elif feature_type == 'connectivity':
        feature_cols = connectivity_cols
    elif feature_type == 'neural_only':
        feature_cols = activation_cols + connectivity_cols
    elif feature_type == 'ies_only':
        feature_cols = ['ies_2bk']
    elif feature_type == 'all_with_leakage':
        feature_cols = behavioral_cols + activation_cols + connectivity_cols
    else:  # legacy 'all' alias → treat as leakage-inclusive
        feature_cols = behavioral_cols + activation_cols + connectivity_cols

    # Filter to available columns and drop any leakage features defensively
    feature_cols = [c for c in feature_cols if c in df.columns]
    if feature_type == 'neural_only':
        feature_cols = [c for c in feature_cols if c not in LEAKAGE_FEATURES]
    
    # Prepare X and y
    X = df[feature_cols].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['efficiency_group'].values)
    
    return X, y, feature_cols


# =============================================================================
# TRADITIONAL ML CLASSIFIERS
# =============================================================================

def train_traditional_classifiers(X, y, feature_names):
    """
    Train and evaluate traditional ML classifiers with cross-validation.
    """
    logger.info("Training traditional ML classifiers...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(
            random_state=AI_CONFIG['random_state'], max_iter=1000, C=0.1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', probability=True, random_state=AI_CONFIG['random_state'], C=1.0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=AI_CONFIG['random_state']
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=AI_CONFIG['random_state']
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=AI_CONFIG['mlp_hidden_layers'],
            max_iter=AI_CONFIG['mlp_max_iter'],
            random_state=AI_CONFIG['random_state'],
            early_stopping=True, validation_fraction=0.2
        ),
    }
    
    results = {}

    # 5-fold stratified CV (matches manuscript; LOOCV replaced for lower
    # variance and faster permutation testing)
    cv = StratifiedKFold(
        n_splits=AI_CONFIG['n_cv_folds'],
        shuffle=True,
        random_state=AI_CONFIG['random_state'],
    )

    for name, clf in classifiers.items():
        logger.info(f"  Training {name}...")
        
        # Cross-validated predictions
        y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
        y_prob = cross_val_predict(clf, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        # Handle AUC calculation for small samples
        try:
            auc = roc_auc_score(y, y_prob)
        except:
            auc = 0.5
        
        # Permutation test for statistical significance
        try:
            score, perm_scores, p_value = permutation_test_score(
                clf, X_scaled, y, cv=cv,
                n_permutations=AI_CONFIG['n_permutations'],
                scoring='accuracy',
                random_state=AI_CONFIG['random_state'],
                n_jobs=AI_CONFIG['n_jobs'],
            )
        except Exception:
            p_value = 1.0
            perm_scores = []
        
        results[name] = {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'p_value': float(p_value),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist(),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
        }
        
        logger.info(f"    Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, p={p_value:.4f}")
    
    # Get feature importance from Random Forest
    rf = classifiers['Random Forest']
    rf.fit(X_scaled, y)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Get coefficients from Logistic Regression
    lr = classifiers['Logistic Regression']
    lr.fit(X_scaled, y)
    lr_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    return results, feature_importance, lr_importance


# =============================================================================
# DEEP NEURAL NETWORK WITH ATTENTION
# =============================================================================

if TORCH_AVAILABLE:
    class AttentionClassifier(nn.Module):
        """
        Neural network classifier with self-attention mechanism for interpretability.
        
        The attention weights reveal which features are most important for classification.
        """
        def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout=0.3):
            super(AttentionClassifier, self).__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            
            # Feature embedding
            self.feature_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Multi-head self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=num_heads, 
                dropout=dropout,
                batch_first=True
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)
            )
            
            # Store attention weights for interpretability
            self.attention_weights = None
        
        def forward(self, x, return_attention=False):
            # x shape: (batch_size, input_dim)
            
            # Embed features
            embedded = self.feature_embedding(x)  # (batch, hidden_dim)
            
            # Reshape for attention: treat each feature dimension as a sequence element
            # (batch, 1, hidden_dim) -> self-attention over feature representation
            embedded = embedded.unsqueeze(1)  # (batch, 1, hidden_dim)
            
            # Self-attention
            attn_output, attn_weights = self.attention(
                embedded, embedded, embedded, 
                need_weights=True
            )
            
            self.attention_weights = attn_weights
            
            # Classification
            output = attn_output.squeeze(1)  # (batch, hidden_dim)
            logits = self.classifier(output)
            
            if return_attention:
                return logits, attn_weights
            return logits
    
    
    class FeatureAttentionNet(nn.Module):
        """
        Neural network with feature-level attention for interpretable predictions.
        Each input feature gets an attention weight.
        """
        def __init__(self, input_dim, hidden_dim=32, dropout=0.3):
            super(FeatureAttentionNet, self).__init__()
            
            self.input_dim = input_dim
            
            # Feature-wise attention
            self.attention_weights = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim),
                nn.Softmax(dim=1)
            )
            
            # Classification network
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)
            )
            
            self.last_attention = None
        
        def forward(self, x, return_attention=False):
            # Compute attention weights for each feature
            attention = self.attention_weights(x)  # (batch, input_dim)
            self.last_attention = attention
            
            # Apply attention (weighted features)
            weighted_x = x * attention
            
            # Classify
            logits = self.classifier(weighted_x)
            
            if return_attention:
                return logits, attention
            return logits


def train_attention_network(X, y, feature_names):
    """
    Train attention-based neural network and extract feature importance.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, skipping attention network")
        return None, None
    
    logger.info("Training attention-based neural network...")
    
    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
    y_tensor = torch.LongTensor(y).to(DEVICE)

    # Model
    model = FeatureAttentionNet(input_dim=X.shape[1], hidden_dim=32, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=AI_CONFIG['learning_rate'], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training with early stopping
    best_loss = float('inf')
    patience = AI_CONFIG['patience']
    patience_counter = 0
    
    model.train()
    for epoch in range(AI_CONFIG['epochs']):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break
        
        if epoch % 20 == 0:
            logger.info(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        _, attention = model(X_tensor, return_attention=True)
        attention_weights = attention.mean(dim=0).cpu().numpy()
    
    # Cross-validated evaluation
    cv = LeaveOneOut()
    y_pred_all = []
    
    for train_idx, test_idx in cv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train fresh model for each fold (match main-training regularization)
        fold_model = FeatureAttentionNet(input_dim=X.shape[1], hidden_dim=32, dropout=0.3).to(DEVICE)
        fold_optimizer = torch.optim.Adam(fold_model.parameters(),
                                          lr=AI_CONFIG['learning_rate'],
                                          weight_decay=0.01)

        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        y_train_t = torch.LongTensor(y_train).to(DEVICE)
        X_test_t = torch.FloatTensor(X_test).to(DEVICE)
        
        fold_model.train()
        best_fold_loss = float('inf')
        fold_patience = 0
        for _ in range(AI_CONFIG['epochs']):
            fold_optimizer.zero_grad()
            outputs = fold_model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            fold_optimizer.step()
            if loss.item() < best_fold_loss - 1e-4:
                best_fold_loss = loss.item()
                fold_patience = 0
            else:
                fold_patience += 1
                if fold_patience >= AI_CONFIG['patience']:
                    break
        
        fold_model.eval()
        with torch.no_grad():
            pred = fold_model(X_test_t).argmax(dim=1).item()
            y_pred_all.append(pred)
    
    accuracy = accuracy_score(y, y_pred_all)
    
    # Create attention importance DataFrame
    attention_df = pd.DataFrame({
        'feature': feature_names,
        'attention_weight': attention_weights
    }).sort_values('attention_weight', ascending=False)
    
    results = {
        'accuracy': float(accuracy),
        'attention_weights': attention_weights.tolist(),
    }
    
    logger.info(f"  Attention Network Accuracy: {accuracy:.3f}")
    
    return results, attention_df


# =============================================================================
# GRAPH NEURAL NETWORK FOR CONNECTIVITY
# =============================================================================

if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
    class BrainGNN(nn.Module):
        """
        Graph Neural Network for brain connectivity classification.
        
        Uses Graph Attention Networks (GAT) to learn from brain network structure.
        """
        def __init__(self, num_node_features, hidden_channels=32, num_classes=2, heads=4):
            super(BrainGNN, self).__init__()
            
            # Graph Attention layers
            self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.3)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.3)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_channels // 2, num_classes)
            )
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # Graph convolutions with attention
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            
            # Global pooling
            x = global_mean_pool(x, batch)
            
            # Classification
            return self.classifier(x)


def create_brain_graph(connectivity_matrix, node_features, threshold=0.1):
    """
    Create a PyTorch Geometric graph from connectivity matrix.
    
    Args:
        connectivity_matrix: NxN connectivity matrix
        node_features: Node feature matrix (N x F)
        threshold: Edge threshold for sparsification
    
    Returns:
        PyG Data object
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    n_nodes = connectivity_matrix.shape[0]
    
    # Create edges from connectivity matrix
    edges = []
    edge_weights = []
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if abs(connectivity_matrix[i, j]) > threshold:
                edges.append([i, j])
                edges.append([j, i])  # Undirected
                edge_weights.append(connectivity_matrix[i, j])
                edge_weights.append(connectivity_matrix[i, j])
    
    if len(edges) == 0:
        # Add self-loops if no edges
        edges = [[i, i] for i in range(n_nodes)]
        edge_weights = [1.0] * n_nodes
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


_CONN_CACHE = {}


def _load_subject_connectivity(subject, condition='2bk'):
    """Load saved connectivity matrix for a subject from s03 output.

    Checks the configured CONNECTIVITY_DIR first, then falls back to
    results/latest/connectivity/ so the GNN can access real FC matrices
    when the pipeline writes them into a timestamped run directory.
    """
    if condition in _CONN_CACHE:
        data = _CONN_CACHE[condition]
    else:
        candidates = [
            config.CONNECTIVITY_DIR / f'connectivity_matrices_{condition}.npz',
            config.PROJECT_DIR / 'results' / 'latest' / 'connectivity'
                / f'connectivity_matrices_{condition}.npz',
        ]
        data = None
        for mat_file in candidates:
            if mat_file.exists():
                data = np.load(mat_file, allow_pickle=True)
                break
        _CONN_CACHE[condition] = data

    if data is not None and subject in data.files:
        return data[subject]
    return None


def train_brain_gnn(df, feature_names):
    """
    Train Graph Neural Network on brain connectivity data.

    Uses real functional connectivity matrices (from s03) as graph structure,
    with multi-dimensional node features (activation values + load effect).
    """
    if not TORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
        logger.warning("PyTorch Geometric not available, skipping GNN analysis")
        return None

    logger.info("Training Brain Graph Neural Network...")

    # 14 WM-related ROIs — kept consistent with Table I of the manuscript and
    # with the activation feature set. Previously omitted Premotor_L/R are
    # restored here so the GNN operates on the same 14-node topology as the
    # rest of the pipeline.
    roi_names = ['DLPFC_L', 'DLPFC_R', 'VLPFC_L', 'VLPFC_R',
                 'PPC_L', 'PPC_R', 'ACC_L', 'ACC_R',
                 'Premotor_L', 'Premotor_R',
                 'mPFC', 'PCC', 'Angular_L', 'Angular_R']
    n_rois = len(roi_names)

    graphs = []
    labels = []
    graph_source = 'feature_correlation'

    for idx, row in df.iterrows():
        subj = str(row.get('subject', idx))

        # Node features: activation (0bk, 2bk) + load effect
        node_features = np.zeros((n_rois, 3))
        for i, roi in enumerate(roi_names):
            col_0bk = f"{roi}_activation_0bk"
            col_2bk = f"{roi}_activation_2bk"
            if col_0bk in df.columns:
                node_features[i, 0] = row.get(col_0bk, 0)
            if col_2bk in df.columns:
                node_features[i, 1] = row.get(col_2bk, 0)
            node_features[i, 2] = node_features[i, 1] - node_features[i, 0]

        # Try to load real connectivity matrix from s03 output
        conn_matrix = _load_subject_connectivity(subj, '2bk')
        if conn_matrix is not None and conn_matrix.shape[0] >= n_rois:
            conn_matrix = conn_matrix[:n_rois, :n_rois]
            graph_source = 'real_connectivity'
        else:
            # Build connectivity from pairwise feature columns in df
            conn_matrix = np.zeros((n_rois, n_rois))
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    ci = f"{roi_names[i]}_activation_2bk"
                    cj = f"{roi_names[j]}_activation_2bk"
                    if ci in df.columns and cj in df.columns:
                        conn_matrix[i, j] = 1.0 / (1.0 + abs(row.get(ci, 0) - row.get(cj, 0)))
                        conn_matrix[j, i] = conn_matrix[i, j]

        graph = create_brain_graph(conn_matrix, node_features, threshold=0.1)
        if graph is not None:
            graph.y = torch.tensor([1 if row['efficiency_group'] == 'High_Efficiency' else 0])
            graphs.append(graph)
            labels.append(graph.y.item())

    if len(graphs) < 10:
        logger.warning(f"Only {len(graphs)} graphs, insufficient for GNN training")
        return None

    # Stratified K-Fold cross-validation (more stable than LOOCV)
    n_splits = min(5, len(graphs) // 4)
    y_arr = np.array(labels)

    if n_splits >= 2:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_name = f'StratifiedKFold(k={n_splits})'
    else:
        cv = LeaveOneOut()
        cv_name = 'LOOCV'

    y_pred_all = []
    y_true_all = []

    for train_idx, test_idx in cv.split(graphs, y_arr):
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]

        model = BrainGNN(num_node_features=3, hidden_channels=16, num_classes=2, heads=2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        # Training with early stopping
        model.train()
        best_loss = float('inf')
        patience = 0

        for epoch in range(AI_CONFIG['epochs']):
            epoch_loss = 0.0
            np.random.shuffle(train_graphs)
            for graph in train_graphs:
                optimizer.zero_grad()
                batch = Batch.from_data_list([graph]).to(DEVICE)
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_graphs)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= AI_CONFIG['patience']:
                    break

        # Prediction
        model.eval()
        with torch.no_grad():
            for tg in test_graphs:
                batch = Batch.from_data_list([tg]).to(DEVICE)
                pred = model(batch).argmax(dim=1).item()
                y_pred_all.append(pred)
                y_true_all.append(tg.y.item())

    accuracy = accuracy_score(y_true_all, y_pred_all)

    results = {
        'accuracy': float(accuracy),
        'n_graphs': len(graphs),
        'n_nodes': n_rois,
        'n_node_features': 3,
        'cv_method': cv_name,
        'graph_source': graph_source,
    }

    logger.info(f"  GNN Accuracy: {accuracy:.3f} ({cv_name}, graph_source={graph_source})")

    return results


# =============================================================================
# INTEGRATED GRADIENTS FOR FEATURE ATTRIBUTION
# =============================================================================

def compute_integrated_gradients(model, X, baseline=None, steps=50):
    """
    Compute Integrated Gradients for feature attribution.
    
    This method provides theoretically grounded feature importance scores.
    """
    if not TORCH_AVAILABLE:
        return None
    
    if baseline is None:
        baseline = np.zeros_like(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).requires_grad_(True)
    baseline_tensor = torch.FloatTensor(baseline)
    
    # Interpolate between baseline and input
    alphas = np.linspace(0, 1, steps)
    gradients = []
    
    for alpha in alphas:
        interpolated = baseline_tensor + alpha * (X_tensor - baseline_tensor)
        interpolated = interpolated.requires_grad_(True)
        
        output = model(interpolated)
        
        # Get gradient for predicted class
        pred_class = output.argmax(dim=1)
        output_for_grad = output.gather(1, pred_class.unsqueeze(1)).sum()
        
        grad = torch.autograd.grad(output_for_grad, interpolated)[0]
        gradients.append(grad.detach().numpy())
    
    # Average gradients and multiply by (input - baseline)
    avg_gradients = np.mean(gradients, axis=0)
    integrated_gradients = (X - baseline) * avg_gradients
    
    return integrated_gradients


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_ai_results(results, feature_importance, attention_df, output_dir):
    """
    Create comprehensive visualization of AI analysis results.
    """
    logger.info("Creating AI analysis visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Model Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: Accuracy comparison
    ax = axes[0, 0]
    models = list(results['traditional'].keys())
    accuracies = [results['traditional'][m]['accuracy'] for m in models]
    colors = [ACCENT if acc > 0.6 else '#95a5a6' for acc in accuracies]
    
    bars = ax.barh(models, accuracies, color=colors, edgecolor='white', linewidth=1.5)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Chance level')
    ax.set_xlabel('5-fold Stratified CV Accuracy', fontsize=12)
    ax.set_title('A. Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend()
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 0.02, bar.get_y() + bar.get_height()/2, f'{acc:.2f}', 
                va='center', fontsize=10)
    
    # Panel B: Feature Importance (Random Forest)
    ax = axes[0, 1]
    top_features = feature_importance.head(10)
    colors = [HIGH_EFF if 'high' in f.lower() or '2bk' in f else LOW_EFF for f in top_features['feature']]
    ax.barh(range(len(top_features)), top_features['importance'].values, color=ACCENT)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in top_features['feature']], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('B. Top 10 Features (Random Forest)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Panel C: Attention Weights (if available)
    ax = axes[1, 0]
    if attention_df is not None and len(attention_df) > 0:
        top_attention = attention_df.head(10)
        ax.barh(range(len(top_attention)), top_attention['attention_weight'].values, color='#e74c3c')
        ax.set_yticks(range(len(top_attention)))
        ax.set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in top_attention['feature']], fontsize=9)
        ax.set_xlabel('Attention Weight', fontsize=12)
        ax.set_title('C. Top 10 Features (Neural Network Attention)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Attention analysis\nnot available', ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title('C. Neural Network Attention', fontsize=14, fontweight='bold')
    
    # Panel D: ROC Curves
    ax = axes[1, 1]
    # Reload the labels in the original subject order so each y_prob entry
    # can be paired with the correct y_true. The efficiency_group column in
    # neural_efficiency.csv is the same source used by prepare_features().
    y_true = None
    try:
        eff_file = config.EFFICIENCY_DIR / 'neural_efficiency.csv'
        if not eff_file.exists():
            eff_file = config.PROJECT_DIR / 'results' / 'latest' / 'efficiency' / 'neural_efficiency.csv'
        if eff_file.exists():
            label_df = pd.read_csv(eff_file, usecols=['efficiency_group'])
            y_true = LabelEncoder().fit_transform(label_df['efficiency_group'].values)
    except Exception as e:
        logger.warning(f"Could not reload labels for ROC plot: {e}")

    for model_name in ['Logistic Regression', 'SVM (RBF)', 'Random Forest',
                       'Gradient Boosting', 'MLP Neural Network']:
        if model_name not in results['traditional']:
            continue
        y_prob = results['traditional'][model_name].get('y_prob')
        if y_true is None or y_prob is None or len(y_prob) != len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = results['traditional'][model_name]['auc']
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.2f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('D. ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_ai_model_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_ai_model_comparison.svg', bbox_inches='tight')
    plt.close(fig)

    logger.info(f"  Saved: fig_ai_model_comparison.png/svg")

    # Compact single-panel ROC figure for the main manuscript. The 4-panel
    # figure above is retained for supplementary material; the manuscript
    # only needs the ROC panel to make the chance-level point.
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
    if y_true is not None:
        for model_name in ['Logistic Regression', 'SVM (RBF)', 'Random Forest',
                           'Gradient Boosting', 'MLP Neural Network']:
            if model_name not in results['traditional']:
                continue
            y_prob = results['traditional'][model_name].get('y_prob')
            if y_prob is None or len(y_prob) != len(y_true):
                continue
            fpr_, tpr_, _ = roc_curve(y_true, y_prob)
            auc_ = results['traditional'][model_name]['auc']
            ax2.plot(fpr_, tpr_, label=f'{model_name} (AUC={auc_:.2f})',
                     linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Chance')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    # Title removed per user request; panel legend conveys the classifier identities.
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig2.savefig(output_dir / 'fig_ai_roc_only.png', dpi=300,
                 bbox_inches='tight')
    fig2.savefig(output_dir / 'fig_ai_roc_only.svg', bbox_inches='tight')
    fig2.savefig(output_dir / 'fig_ai_roc_only.pdf', bbox_inches='tight')
    plt.close(fig2)
    logger.info("  Saved: fig_ai_roc_only.png/svg (single-panel ROC)")
    
    # Figure 2: Feature Importance Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Combine importance measures
    importance_df = feature_importance.copy()
    importance_df = importance_df.set_index('feature')
    
    if attention_df is not None:
        attention_df_indexed = attention_df.set_index('feature')
        importance_df = importance_df.join(attention_df_indexed, how='outer')
    
    importance_df = importance_df.fillna(0)
    
    # Normalize columns
    for col in importance_df.columns:
        if importance_df[col].max() > 0:
            importance_df[col] = importance_df[col] / importance_df[col].max()
    
    # Plot top 15 features
    top_features = importance_df.head(15)
    
    sns.heatmap(top_features, cmap='YlOrRd', annot=True, fmt='.2f', 
                cbar_kws={'label': 'Normalized Importance'}, ax=ax)
    ax.set_title('Feature Importance Across Methods', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_ai_feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_ai_feature_importance_heatmap.svg', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  Saved: fig_ai_feature_importance_heatmap.png/svg")
    
    # Figure 3: Summary Statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Create summary table
    summary_data = [
        ['Model', 'Accuracy', 'AUC', 'p-value'],
    ]
    
    for model_name, model_results in results['traditional'].items():
        summary_data.append([
            model_name,
            f"{model_results['accuracy']:.3f}",
            f"{model_results['auc']:.3f}",
            f"{model_results['p_value']:.4f}" if model_results['p_value'] < 1 else 'N/A'
        ])
    
    if 'attention_network' in results and results['attention_network']:
        summary_data.append([
            'Attention Network',
            f"{results['attention_network']['accuracy']:.3f}",
            'N/A',
            'N/A'
        ])
    
    if 'gnn' in results and results['gnn']:
        summary_data.append([
            'Graph Neural Network',
            f"{results['gnn']['accuracy']:.3f}",
            'N/A',
            'N/A'
        ])
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best model
    best_acc = max([r['accuracy'] for r in results['traditional'].values()])
    for i, (model_name, model_results) in enumerate(results['traditional'].items(), 1):
        if model_results['accuracy'] == best_acc:
            for j in range(4):
                table[(i, j)].set_facecolor('#d5f5e3')
    
    ax.set_title('AI Classification Results Summary\n(Best model highlighted)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'fig_ai_summary_table.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'fig_ai_summary_table.svg', bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"  Saved: fig_ai_summary_table.png/svg")


# =============================================================================
# LEAKAGE SANITY CHECK
# =============================================================================

def _run_leakage_check(df):
    """
    Quantify target leakage by comparing classifier accuracy across three
    feature sets:
        neural_only        — activation + connectivity (primary analysis)
        ies_only           — ies_2bk alone (label-defining variable)
        all_with_leakage   — full 152 features including behavioral

    Expected outcome: ies_only and all_with_leakage approach ceiling
    accuracy because ies_2bk effectively encodes the label (groups were
    defined by median split on ies_2bk). The gap between these and the
    neural_only result quantifies the leakage-induced inflation in the
    previously-reported 90.5% figure.
    """
    logger.info("Running leakage sanity check (LR + GBM on 3 feature sets)...")

    cv = StratifiedKFold(
        n_splits=AI_CONFIG['n_cv_folds'],
        shuffle=True,
        random_state=AI_CONFIG['random_state'],
    )
    check = {}

    for tag in ('neural_only', 'ies_only', 'all_with_leakage'):
        X_c, y_c, fnames = prepare_features(df, feature_type=tag)
        X_c = np.nan_to_num(X_c, nan=0.0)
        X_c = StandardScaler().fit_transform(X_c)

        lr = LogisticRegression(
            random_state=AI_CONFIG['random_state'], max_iter=1000, C=0.1,
        )
        gbm = GradientBoostingClassifier(
            n_estimators=50, max_depth=3,
            random_state=AI_CONFIG['random_state'],
        )
        lr_acc = cross_val_score(lr, X_c, y_c, cv=cv, scoring='accuracy').mean()
        gbm_acc = cross_val_score(gbm, X_c, y_c, cv=cv, scoring='accuracy').mean()

        check[tag] = {
            'n_features': int(X_c.shape[1]),
            'lr_accuracy': float(lr_acc),
            'gbm_accuracy': float(gbm_acc),
        }
        logger.info(
            f"  {tag:18s} (p={X_c.shape[1]:3d}): "
            f"LR={lr_acc:.3f}, GBM={gbm_acc:.3f}"
        )

    return check


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_ai_analysis(skip_traditional=False, skip_attention=False, skip_gnn=False):
    """
    Run complete AI analysis pipeline.

    Parameters:
        skip_traditional: Reuse existing traditional ML results from
            ai_classification_results.json instead of retraining.
        skip_attention: Same for Attention Network.
        skip_gnn: Same for GNN.

    Any skipped module reuses its previous results if present.
    """
    logger.info("="*70)
    logger.info("Starting Advanced AI Analysis Pipeline")
    logger.info("="*70)

    # Load data
    df = load_data_for_ai()

    # Primary feature set: NEURAL ONLY. Behavioral variables are excluded
    # because the efficiency_group label is derived by median split on
    # ies_2bk; including behavioral features would constitute target leakage.
    X, y, feature_names = prepare_features(df, feature_type='neural_only')
    logger.info(
        f"Prepared {X.shape[0]} samples with {X.shape[1]} neural features "
        f"(behavioral features excluded to avoid target leakage)"
    )

    # Load previous results if we need to skip parts
    prev_path = config.EFFICIENCY_DIR / 'ai_classification_results.json'
    prev_results = {}
    if any([skip_traditional, skip_attention, skip_gnn]) and prev_path.exists():
        with open(prev_path) as f:
            prev_results = json.load(f)
        logger.info(f"Loaded previous results from {prev_path.name}")

    # Initialize results
    all_results = {
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'feature_names': feature_names,
    }

    # 1. Traditional ML classifiers
    if skip_traditional and 'traditional' in prev_results:
        logger.info("Skipping traditional ML (reusing previous results)")
        all_results['traditional'] = prev_results['traditional']
        feature_importance = pd.read_csv(config.EFFICIENCY_DIR / 'ai_feature_importance.csv')
        lr_importance = pd.read_csv(config.EFFICIENCY_DIR / 'ai_logistic_coefficients.csv')
    else:
        trad_results, feature_importance, lr_importance = train_traditional_classifiers(X, y, feature_names)
        all_results['traditional'] = trad_results

    # 2. Attention-based neural network
    if skip_attention and 'attention_network' in prev_results:
        logger.info("Skipping Attention Network (reusing previous results)")
        all_results['attention_network'] = prev_results['attention_network']
        att_csv = config.EFFICIENCY_DIR / 'ai_attention_weights.csv'
        attention_df = pd.read_csv(att_csv) if att_csv.exists() else None
    else:
        attention_results, attention_df = train_attention_network(X, y, feature_names)
        all_results['attention_network'] = attention_results

    # 3. Graph Neural Network
    if skip_gnn and 'gnn' in prev_results:
        logger.info("Skipping GNN (reusing previous results)")
        all_results['gnn'] = prev_results['gnn']
        gnn_results = prev_results['gnn']
    else:
        gnn_results = train_brain_gnn(df, feature_names)
        all_results['gnn'] = gnn_results

    # 4. Leakage sanity check: run Logistic Regression and Gradient Boosting
    # on (a) IES_2bk alone and (b) the full 152-feature set that includes
    # behavioral variables, to quantify how much the previous 90%+ accuracy
    # was driven by label-defining features rather than neural signal.
    all_results['leakage_check'] = _run_leakage_check(df)

    # Determine output directory
    output_dir = config.FIGURES_DIR if hasattr(config, 'FIGURES_DIR') else config.PROJECT_DIR / 'results' / 'latest' / 'figures'
    output_dir = Path(output_dir)
    
    # Create visualizations
    plot_ai_results(all_results, feature_importance, attention_df, output_dir)
    
    # Save results
    results_dir = config.EFFICIENCY_DIR if hasattr(config, 'EFFICIENCY_DIR') else config.PROJECT_DIR / 'results' / 'latest' / 'efficiency'
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    all_results_serializable = convert_to_serializable(all_results)
    
    with open(results_dir / 'ai_classification_results.json', 'w') as f:
        json.dump(all_results_serializable, f, indent=2)
    
    feature_importance.to_csv(results_dir / 'ai_feature_importance.csv', index=False)
    
    if attention_df is not None:
        attention_df.to_csv(results_dir / 'ai_attention_weights.csv', index=False)
    
    lr_importance.to_csv(results_dir / 'ai_logistic_coefficients.csv', index=False)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("AI ANALYSIS SUMMARY")
    logger.info("="*70)
    
    logger.info("\nTraditional ML Results:")
    for model_name, model_results in all_results['traditional'].items():
        logger.info(f"  {model_name}: Accuracy={model_results['accuracy']:.3f}, AUC={model_results['auc']:.3f}")

    att = all_results.get('attention_network')
    if att:
        logger.info(f"\nAttention Network: Accuracy={att['accuracy']:.3f}")

    gnn = all_results.get('gnn')
    if gnn:
        logger.info(f"\nGraph Neural Network: Accuracy={gnn['accuracy']:.3f}")
    
    logger.info(f"\nTop 5 Important Features (Random Forest):")
    for _, row in feature_importance.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    lc = all_results.get('leakage_check')
    if lc:
        logger.info("\nLeakage Sanity Check (LR / GBM accuracy by feature set):")
        for tag, r in lc.items():
            logger.info(
                f"  {tag:18s} (p={r['n_features']:3d}): "
                f"LR={r['lr_accuracy']:.3f}, GBM={r['gbm_accuracy']:.3f}"
            )

    logger.info("\n" + "="*70)
    logger.info("AI Analysis Complete!")
    logger.info("="*70)
    
    return all_results


if __name__ == '__main__':
    results = run_ai_analysis()

