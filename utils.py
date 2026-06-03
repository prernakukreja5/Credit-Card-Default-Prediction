"""
UTILITY FUNCTIONS
Common utilities for the credit card default prediction project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Model name for title
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    tn, fp, fn, tp = cm.ravel()
    plt.text(0.5, -0.15, f'Sensitivity: {tp/(tp+fn):.3f} | Specificity: {tn/(tn+fp):.3f}',
            transform=plt.gca().transAxes, ha='center', fontsize=10)
    
    plt.tight_layout()
    return cm


def plot_roc_curve(y_true, y_proba, model_name="Model", ax=None):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Model name for legend
    ax : matplotlib axis
        Axis to plot on
    
    Returns:
    --------
    fpr : array
        False positive rates
    tpr : array
        True positive rates
    auc_score : float
        AUC score
    """
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)
    
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fpr, tpr, auc_score


def plot_precision_recall_curve(y_true, y_proba, model_name="Model", ax=None):
    """
    Plot Precision-Recall curve
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Model name for legend
    ax : matplotlib axis
        Axis to plot on
    
    Returns:
    --------
    precision : array
        Precision values
    recall : array
        Recall values
    ap_score : float
        Average precision score
    """
    
    from sklearn.metrics import average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)
    
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
    
    ax.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return precision, recall, ap_score


def plot_feature_importance(feature_importance_df, top_n=20):
    """
    Plot feature importance
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    """
    
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()


def compare_models_plots(results_dict, metric='auc'):
    """
    Plot model comparison
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics as values
    metric : str
        Metric to plot
    """
    
    models = list(results_dict.keys())
    values = [results_dict[m][metric] if isinstance(results_dict[m], dict) 
              else results_dict[m] for m in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(models)), values)
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}')
    
    # Color bars by value
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()


def create_comparison_table(models_results):
    """
    Create comprehensive comparison table
    
    Parameters:
    -----------
    models_results : dict
        Results from all models
    
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison table
    """
    
    data = []
    for model_name, metrics in models_results.items():
        row = {
            'Model': model_name,
            'AUC': metrics['auc'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
        }
        data.append(row)
    
    df = pd.DataFrame(data).sort_values('AUC', ascending=False)
    return df


def plot_threshold_analysis(thresholds, recalls, precisions, f1_scores):
    """
    Plot threshold optimization analysis
    
    Parameters:
    -----------
    thresholds : array-like
        Decision thresholds
    recalls : array-like
        Recall at each threshold
    precisions : array-like
        Precision at each threshold
    f1_scores : array-like
        F1 scores at each threshold
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Recall vs Threshold
    axes[0].plot(thresholds, recalls, 'b-', marker='o')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Recall')
    axes[0].set_title('Recall vs Threshold')
    axes[0].grid(True, alpha=0.3)
    
    # Precision vs Threshold
    axes[1].plot(thresholds, precisions, 'g-', marker='o')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision vs Threshold')
    axes[1].grid(True, alpha=0.3)
    
    # F1 vs Threshold
    axes[2].plot(thresholds, f1_scores, 'r-', marker='o')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score vs Threshold')
    axes[2].grid(True, alpha=0.3)
    
    # Mark optimal thresholds
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_recall_idx = np.argmax(recalls)
    
    axes[1].axvline(thresholds[optimal_f1_idx], color='red', linestyle='--', 
                   label=f'Optimal F1: {thresholds[optimal_f1_idx]:.2f}')
    axes[1].legend()
    
    plt.tight_layout()


def calculate_business_metrics(y_true, y_pred, y_proba, threshold=0.5):
    """
    Calculate comprehensive business metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like
        Predicted probabilities
    threshold : float
        Decision threshold
    
    Returns:
    --------
    metrics : dict
        Business metrics
    """
    
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_proba),
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'sensitivity': tp / (tp + fn),  # True Positive Rate
        'specificity': tn / (tn + fp),  # True Negative Rate
        'fpr': fp / (fp + tn),          # False Positive Rate
        'fnr': fn / (fn + tp),          # False Negative Rate
        'threshold': threshold,
    }
    
    return metrics


def print_model_report(y_true, y_pred, y_proba, model_name="Model"):
    """
    Print comprehensive model report
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Model name for report
    """
    
    metrics = calculate_business_metrics(y_true, y_pred, y_proba)
    
    print("\n" + "="*60)
    print(f"MODEL REPORT: {model_name}")
    print("="*60)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1 Score:       {metrics['f1']:.4f}")
    print(f"AUC-ROC:        {metrics['auc']:.4f}")
    print(f"\nSensitivity (TP Rate): {metrics['sensitivity']:.4f}")
    print(f"Specificity (TN Rate): {metrics['specificity']:.4f}")
    print(f"False Positive Rate:   {metrics['fpr']:.4f}")
    print(f"False Negative Rate:   {metrics['fnr']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print("="*60)


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
    print("Available functions:")
    print("  - plot_confusion_matrix()")
    print("  - plot_roc_curve()")
    print("  - plot_precision_recall_curve()")
    print("  - plot_feature_importance()")
    print("  - compare_models_plots()")
    print("  - create_comparison_table()")
    print("  - plot_threshold_analysis()")
    print("  - calculate_business_metrics()")
    print("  - print_model_report()")
