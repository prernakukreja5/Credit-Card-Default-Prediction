"""
PHASE 4: CROSS-VALIDATION (5-FOLD STRATIFIED WITH SMOTE)
Credit Card Default Prediction Project

This phase evaluates all models using 5-fold stratified cross-validation
with SMOTE applied to each fold's training data separately.
"""

import pandas as pd
import numpy as np
import json
import warnings
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, 
                              f1_score, accuracy_score)

warnings.filterwarnings('ignore')


class CrossValidationEvaluator:
    """
    Performs 5-fold stratified cross-validation with SMOTE
    """
    
    def __init__(self, n_splits=5, random_state=42):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        n_splits : int
            Number of CV folds
        random_state : int
            Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                   random_state=random_state)
        self.scaler = StandardScaler()
        
    def evaluate_model(self, model, X, y, model_name):
        """
        Evaluate a single model using cross-validation
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        model_name : str
            Name of model (for logging)
        
        Returns:
        --------
        fold_results : dict
            Results for each fold
        """
        
        fold_results = {}
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\n  Evaluating {model_name}...", end=" ")
        
        fold_num = 0
        for train_idx, val_idx in self.skf.split(X_scaled, y):
            fold_num += 1
            
            # Get fold data
            X_train_fold = X_scaled[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Apply SMOTE to training data ONLY
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_train_sm, y_train_sm = smote.fit_resample(X_train_fold, y_train_fold)
            
            # Train model
            model.fit(X_train_sm, y_train_sm)
            
            # Predict
            y_pred = model.predict(X_val_fold)
            y_proba = model.predict_proba(X_val_fold)[:, 1]
            
            # Evaluate
            fold_results[f'fold_{fold_num}'] = {
                'accuracy': accuracy_score(y_val_fold, y_pred),
                'precision': precision_score(y_val_fold, y_pred, zero_division=0),
                'recall': recall_score(y_val_fold, y_pred, zero_division=0),
                'f1': f1_score(y_val_fold, y_pred, zero_division=0),
                'auc': roc_auc_score(y_val_fold, y_proba),
            }
        
        print("✓")
        return fold_results
    
    def summarize_results(self, all_results):
        """
        Summarize cross-validation results across all folds
        
        Parameters:
        -----------
        all_results : dict
            Results from all models and folds
        
        Returns:
        --------
        summary : dict
            Mean and std for each model
        """
        
        summary = {}
        
        for model_name, fold_results in all_results.items():
            metrics = {
                'accuracy': [], 'precision': [], 'recall': [], 
                'f1': [], 'auc': []
            }
            
            for fold_name, fold_metrics in fold_results.items():
                for metric in metrics:
                    metrics[metric].append(fold_metrics[metric])
            
            summary[model_name] = {
                'accuracy': (np.mean(metrics['accuracy']), 
                            np.std(metrics['accuracy'])),
                'precision': (np.mean(metrics['precision']), 
                             np.std(metrics['precision'])),
                'recall': (np.mean(metrics['recall']), 
                          np.std(metrics['recall'])),
                'f1': (np.mean(metrics['f1']), 
                      np.std(metrics['f1'])),
                'auc': (np.mean(metrics['auc']), 
                       np.std(metrics['auc'])),
            }
        
        return summary


def run_cross_validation(X, y, output_path=None):
    """
    Run complete cross-validation pipeline
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    output_path : str, optional
        Path to save results
    
    Returns:
    --------
    summary : dict
        Cross-validation summary results
    """
    
    print("="*80)
    print("PHASE 4: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*80)
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Fold size: {len(X)//5:,} samples")
    print(f"  Target distribution: {(y==1).sum():,} defaults, {(y==0).sum():,} non-defaults")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                                n_jobs=-1, max_depth=15),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, 
                                                        random_state=42, max_depth=5),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, 
                               eval_metric='logloss', verbosity=0, max_depth=5),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, 
                                  verbose=-1, max_depth=5),
        'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, 
                                      verbose=0, depth=5),
    }
    
    # Initialize evaluator
    evaluator = CrossValidationEvaluator(n_splits=5, random_state=42)
    
    # Run CV for each model
    print("\n🔵 RUNNING CROSS-VALIDATION...\n")
    all_results = {}
    
    for model_name, model in models.items():
        all_results[model_name] = evaluator.evaluate_model(
            model, X, y, model_name
        )
    
    # Summarize results
    print("\n\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    summary = evaluator.summarize_results(all_results)
    
    # Display results
    print(f"\n{'Rank':<5} {'Model':<22} {'AUC (mean±std)':>20} "
          f"{'Recall (mean±std)':>22} {'F1 (mean±std)':>18}")
    print("-"*90)
    
    sorted_models = sorted(summary.items(), 
                          key=lambda x: x[1]['auc'][0], 
                          reverse=True)
    
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        auc_mean, auc_std = metrics['auc']
        recall_mean, recall_std = metrics['recall']
        f1_mean, f1_std = metrics['f1']
        
        print(f"{rank:<5} {name:<22} {auc_mean:.4f}±{auc_std:.4f}        "
              f"{recall_mean:.4f}±{recall_std:.4f}        "
              f"{f1_mean:.4f}±{f1_std:.4f}")
    
    # Identify top 3 models
    top_3 = [name for name, _ in sorted_models[:3]]
    print(f"\n🎯 Top 3 models: {top_3}")
    
    # Statistics
    avg_auc = np.mean([metrics['auc'][0] for metrics in summary.values()])
    print(f"📊 Average AUC across all models: {avg_auc:.4f}")
    
    # Save results
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump({'summary': summary, 'detailed': all_results}, f)
        print(f"\n✅ Results saved to {output_path}")
    
    return summary, sorted_models


if __name__ == "__main__":
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("data_engineered.csv")
    
    with open('final_features.json', 'r') as f:
        final_features = json.load(f)
    
    X = df[final_features].fillna(0)
    y = df['DEFAULT']
    
    # Run cross-validation
    summary, sorted_models = run_cross_validation(X, y, 'cv_results.pkl')
    
    print("\n" + "="*80)
    print("✅ PHASE 4 COMPLETE!")
    print("="*80)
    print("\nTop 3 models selected for Phase 5 (Hyperparameter Tuning):")
    for i, (name, _) in enumerate(sorted_models[:3], 1):
        print(f"  {i}. {name}")
