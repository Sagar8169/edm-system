"""
explainability.py — SHAP-based model explanations.

Generates global feature importance and per-student local explanations
using TreeExplainer for XGBoost.
"""

import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import get_project_root


def get_models_dir() -> Path:
    return get_project_root() / "models"


def get_reports_dir() -> Path:
    reports_dir = get_project_root() / "reports" / "shap"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def compute_shap_values(xgb_model, X_data: np.ndarray, feature_cols: list[str],
                        save: bool = True) -> tuple:
    """
    Compute SHAP values for the given data using TreeExplainer.

    Args:
        xgb_model: Trained XGBoost model.
        X_data: Feature matrix (typically test set).
        feature_cols: List of feature column names.
        save: Whether to cache SHAP values.

    Returns:
        Tuple of (shap_values array, explainer)
    """
    print("\n🔍 Computing SHAP values...")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_data)

    print(f"  ✓ SHAP values shape: {shap_values.shape}")
    print(f"  ✓ Expected value (base rate): {explainer.expected_value:.4f}")

    if save:
        models_dir = get_models_dir()
        np.save(models_dir / "shap_values.npy", shap_values)
        joblib.dump(explainer, models_dir / "shap_explainer.joblib")
        print(f"  💾 Cached SHAP values to {models_dir}")

    return shap_values, explainer


def plot_global_importance(shap_values: np.ndarray, X_data: np.ndarray,
                           feature_cols: list[str]) -> None:
    """Generate and save global SHAP feature importance plots."""
    reports_dir = get_reports_dir()

    # 1. Beeswarm / Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_data,
        feature_names=feature_cols,
        show=False,
        max_display=20
    )
    plt.title("SHAP Feature Importance (Global)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: shap_summary_beeswarm.png")

    # 2. Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_data,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        max_display=20
    )
    plt.title("Mean |SHAP Value| — Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: shap_importance_bar.png")


def get_student_explanation(student_idx: int, shap_values: np.ndarray,
                            X_data: np.ndarray, feature_cols: list[str],
                            expected_value: float) -> dict:
    """
    Generate a local explanation for a single student.

    Args:
        student_idx: Index of the student in the dataset.
        shap_values: Precomputed SHAP values array.
        X_data: Feature matrix.
        feature_cols: Feature column names.
        expected_value: SHAP base value.

    Returns:
        Dictionary with explanation data:
        {
            'student_idx': int,
            'base_value': float,
            'prediction': float,
            'features': [{'name': str, 'value': float, 'shap_value': float}, ...]
        }
    """
    sv = shap_values[student_idx]
    fv = X_data[student_idx]

    features = []
    for i, (name, val, shap_val) in enumerate(zip(feature_cols, fv, sv)):
        features.append({
            "name": name,
            "value": float(val),
            "shap_value": float(shap_val),
        })

    # Sort by absolute SHAP value (most impactful first)
    features.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    prediction = expected_value + sv.sum()

    return {
        "student_idx": student_idx,
        "base_value": float(expected_value),
        "prediction": float(prediction),
        "features": features,
    }


def plot_student_explanation(student_idx: int, shap_values: np.ndarray,
                             X_data: np.ndarray, feature_cols: list[str],
                             expected_value: float, save_path: Path = None) -> None:
    """Generate and save a waterfall plot for a single student."""
    sv = shap_values[student_idx]

    explanation = shap.Explanation(
        values=sv,
        base_values=expected_value,
        data=X_data[student_idx],
        feature_names=feature_cols
    )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP Explanation — Student Index {student_idx}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = get_reports_dir() / f"shap_student_{student_idx}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_explainability(model_results: dict, save: bool = True) -> dict:
    """
    Full explainability pipeline: compute SHAP → global plots → sample local plots.

    Args:
        model_results: Dictionary from run_modeling().

    Returns:
        Dictionary with SHAP values, explainer, and feature importances.
    """
    xgb_model = model_results["xgb_model"]
    X_test = model_results["X_test"]
    feature_cols = model_results["feature_cols"]

    # Compute SHAP values
    shap_values, explainer = compute_shap_values(xgb_model, X_test, feature_cols, save=save)

    # Global importance plots
    plot_global_importance(shap_values, X_test, feature_cols)

    # Sample local explanations (first 3 at-risk students)
    y_test = model_results["y_test"]
    at_risk_indices = np.where(y_test == 1)[0][:3]
    for idx in at_risk_indices:
        plot_student_explanation(idx, shap_values, X_test, feature_cols, explainer.expected_value)
        print(f"  📊 Saved student explanation for index {idx}")

    # Compute mean absolute SHAP importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    if save:
        importance_df.to_csv(get_reports_dir().parent / "shap_feature_importance.csv", index=False)

    return {
        "shap_values": shap_values,
        "explainer": explainer,
        "importance_df": importance_df,
    }
