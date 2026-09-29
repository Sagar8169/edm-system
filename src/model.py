"""
model.py — Train and evaluate prediction models.

Trains a Logistic Regression baseline and XGBoost main model.
Evaluates both with accuracy, precision, recall, F1, confusion matrix, ROC-AUC.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import get_project_root


def get_models_dir() -> Path:
    """Return models directory, creating if needed."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_reports_dir() -> Path:
    """Return reports directory, creating if needed."""
    reports_dir = get_project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def prepare_data(features_df: pd.DataFrame, feature_cols: list[str],
                 test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train/test sets, stratified on target.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    X = features_df[feature_cols].values
    y = features_df["at_risk"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"📊 Data split:")
    print(f"  Train: {X_train.shape[0]} samples (at_risk: {y_train.sum()} = {y_train.mean()*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} samples (at_risk: {y_test.sum()} = {y_test.mean()*100:.1f}%)")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a trained model and return metrics dictionary.
    """
    y_pred = model.predict(X_test)

    # Get probabilities (handle different model interfaces)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = model.decision_function(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    print(f"\n📈 {model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"    FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")

    return metrics


def train_baseline(X_train, y_train, X_test, y_test) -> tuple:
    """Train Logistic Regression baseline."""
    print("\n" + "="*60)
    print("TRAINING: Logistic Regression (Baseline)")
    print("="*60)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    metrics = evaluate_model(lr, X_test, y_test, "Logistic Regression")

    return lr, metrics


def train_xgboost(X_train, y_train, X_test, y_test, tune: bool = True) -> tuple:
    """Train XGBoost model with optional hyperparameter tuning."""
    print("\n" + "="*60)
    print("TRAINING: XGBoost Classifier")
    print("="*60)

    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    if tune:
        print("  🔍 Hyperparameter tuning (GridSearch)...")
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "min_child_weight": [1, 3],
        }

        xgb_base = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )

        cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            xgb_base,
            param_grid,
            cv=cv_strategy,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        xgb = grid_search.best_estimator_

        print(f"\n  ✓ Best params: {grid_search.best_params_}")
        print(f"  ✓ Best CV F1: {grid_search.best_score_:.4f}")
    else:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)

    metrics = evaluate_model(xgb, X_test, y_test, "XGBoost")

    return xgb, metrics


def plot_evaluation(lr_metrics, xgb_metrics, lr_model, xgb_model,
                    X_test, y_test, feature_cols):
    """Generate evaluation plots: ROC curves, confusion matrices, comparison."""
    reports_dir = get_reports_dir() / "model"
    reports_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Model Evaluation Results", fontsize=16, fontweight="bold")

    # 1. ROC Curves
    ax = axes[0, 0]
    for model, name, color in [(lr_model, "Logistic Regression", "#e74c3c"),
                                (xgb_model, "XGBoost", "#2ecc71")]:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Confusion Matrix — XGBoost
    ax = axes[0, 1]
    cm = np.array(xgb_metrics["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not At Risk", "At Risk"],
                yticklabels=["Not At Risk", "At Risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost Confusion Matrix")

    # 3. Metrics Comparison Bar Chart
    ax = axes[1, 0]
    metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    lr_vals = [lr_metrics[m] for m in metrics_names]
    xgb_vals = [xgb_metrics[m] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x - width/2, lr_vals, width, label="Logistic Regression", color="#e74c3c", alpha=0.8)
    ax.bar(x + width/2, xgb_vals, width, label="XGBoost", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_names])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, (lr_v, xgb_v) in enumerate(zip(lr_vals, xgb_vals)):
        ax.text(i - width/2, lr_v + 0.02, f"{lr_v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, xgb_v + 0.02, f"{xgb_v:.3f}", ha="center", va="bottom", fontsize=8)

    # 4. Feature Importance (XGBoost)
    ax = axes[1, 1]
    if hasattr(xgb_model, "feature_importances_"):
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15
        ax.barh(range(len(indices)), importances[indices], color="#3498db", alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
        ax.set_title("XGBoost Feature Importance (Top 15)")
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(reports_dir / "model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Saved evaluation plots: {reports_dir / 'model_evaluation.png'}")


def export_error_analysis(model, X_test_scaled, X_test_raw, y_test, feature_cols):
    """Export false positives and false negatives for manual error analysis."""
    y_pred = model.predict(X_test_scaled)
    
    # Create DataFrame with raw features
    df = pd.DataFrame(X_test_raw, columns=feature_cols)
    df["actual_at_risk"] = y_test
    df["predicted_at_risk"] = y_pred
    
    # Isolate errors
    errors = df[df["actual_at_risk"] != df["predicted_at_risk"]].copy()
    errors["error_type"] = errors.apply(
        lambda row: "False Positive" if row["predicted_at_risk"] == 1 else "False Negative", axis=1
    )
    
    reports_dir = get_reports_dir()
    out_path = reports_dir / "error_analysis.csv"
    errors.to_csv(out_path, index=False)
    print(f"\n🔍 Saved {len(errors)} misclassified cases for manual review to {out_path}")


def run_modeling(features_df: pd.DataFrame, feature_cols: list[str],
                 tune: bool = True, save: bool = True) -> dict:
    """
    Full modeling pipeline: split → train baseline → train XGBoost → evaluate → save.

    Returns:
        Dictionary with models, metrics, scaler, and data splits.
    """
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_raw, X_test_raw = \
        prepare_data(features_df, feature_cols)

    # Train models
    lr_model, lr_metrics = train_baseline(X_train_scaled, y_train, X_test_scaled, y_test)
    xgb_model, xgb_metrics = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test, tune=tune)

    # Generate evaluation plots
    plot_evaluation(lr_metrics, xgb_metrics, lr_model, xgb_model,
                    X_test_scaled, y_test, feature_cols)

    # Export error analysis for XGBoost
    export_error_analysis(xgb_model, X_test_scaled, X_test_raw, y_test, feature_cols)

    # Comparison table
    comparison = pd.DataFrame([lr_metrics, xgb_metrics])
    comparison = comparison.drop(columns=["confusion_matrix"])
    print("\n📊 Model Comparison:")
    print(comparison.to_string(index=False))

    # Save
    if save:
        models_dir = get_models_dir()
        reports_dir = get_reports_dir()

        joblib.dump(lr_model, models_dir / "lr_baseline.joblib")
        joblib.dump(xgb_model, models_dir / "xgb_model.joblib")
        joblib.dump(scaler, models_dir / "scaler.joblib")
        joblib.dump(feature_cols, models_dir / "feature_cols.joblib")

        # Save metrics as JSON
        all_metrics = {"baseline": lr_metrics, "xgboost": xgb_metrics}
        with open(reports_dir / "model_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

        # Save test indices for SHAP
        joblib.dump({
            "X_test": X_test_scaled,
            "X_test_raw": X_test_raw,
            "y_test": y_test,
            "X_train": X_train_scaled,
            "y_train": y_train,
        }, models_dir / "data_splits.joblib")

        print(f"\n💾 Saved models to {models_dir}")
        print(f"💾 Saved metrics to {reports_dir}")

    return {
        "lr_model": lr_model,
        "xgb_model": xgb_model,
        "lr_metrics": lr_metrics,
        "xgb_metrics": xgb_metrics,
        "scaler": scaler,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }
