"""
fairness.py — Algorithmic fairness analysis using Fairlearn.

Computes prediction accuracy/precision/recall across demographic groups
(gender, disability, age_band) and calculates Algorithmic Fairness Index.
"""

import pandas as pd
import numpy as np
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import get_project_root


def get_reports_dir() -> Path:
    reports_dir = get_project_root() / "reports" / "fairness"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              sensitive_features: pd.DataFrame) -> dict:
    """
    Compute per-group fairness metrics using Fairlearn's MetricFrame.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        sensitive_features: DataFrame with sensitive attribute columns
                           (e.g., gender_original, disability_original, age_band_original).

    Returns:
        Dictionary with MetricFrame objects and summary statistics per attribute.
    """
    print("\n⚖️  Computing fairness metrics...")

    metrics_dict = {
        "accuracy": accuracy_score,
        "precision": lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        "recall": lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
        "f1": lambda y_t, y_p: f1_score(y_t, y_p, zero_division=0),
    }

    results = {}

    for attr_col in sensitive_features.columns:
        attr_name = attr_col.replace("_original", "")
        sf = sensitive_features[attr_col].values

        print(f"\n  📊 Attribute: {attr_name}")

        mf = MetricFrame(
            metrics=metrics_dict,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sf,
        )

        # Per-group results
        by_group = mf.by_group
        print(f"    Groups: {list(by_group.index)}")

        # Compute gaps (max - min per metric)
        gaps = {}
        for metric_name in metrics_dict:
            values = by_group[metric_name]
            gap = values.max() - values.min()
            gaps[metric_name] = gap
            print(f"    {metric_name} gap: {gap:.4f} (range: {values.min():.4f} - {values.max():.4f})")

        # Overall metrics
        overall = mf.overall

        results[attr_name] = {
            "metric_frame": mf,
            "by_group": by_group,
            "overall": overall,
            "gaps": gaps,
        }

    return results


def compute_fairness_index(fairness_results: dict) -> float:
    """
    Compute Algorithmic Fairness Index (AFI).

    AFI = 1 - max(recall_gap across all sensitive attributes)

    Higher is better. 1.0 = perfectly fair. 0.0 = maximum disparity.
    """
    max_recall_gap = 0.0

    for attr_name, attr_results in fairness_results.items():
        recall_gap = attr_results["gaps"].get("recall", 0.0)
        max_recall_gap = max(max_recall_gap, recall_gap)

    afi = 1.0 - max_recall_gap
    afi = max(0.0, min(1.0, afi))  # Clamp to [0, 1]

    print(f"\n  🎯 Algorithmic Fairness Index (AFI): {afi:.4f}")
    print(f"     (1 - max recall gap of {max_recall_gap:.4f})")

    return afi


def plot_fairness(fairness_results: dict, afi: float) -> None:
    """Generate fairness visualization plots."""
    reports_dir = get_reports_dir()
    n_attrs = len(fairness_results)

    fig, axes = plt.subplots(1, n_attrs, figsize=(7 * n_attrs, 6))
    if n_attrs == 1:
        axes = [axes]

    for ax, (attr_name, attr_data) in zip(axes, fairness_results.items()):
        by_group = attr_data["by_group"]

        # Plot grouped bar chart
        metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(by_group.index))
        width = 0.2
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            values = by_group[metric].values
            ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.85)

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(by_group.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Fairness by {attr_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Algorithmic Fairness Analysis — AFI: {afi:.3f}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(reports_dir / "fairness_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Saved: {reports_dir / 'fairness_analysis.png'}")

    # Gap summary heatmap
    gap_data = {}
    for attr_name, attr_data in fairness_results.items():
        gap_data[attr_name] = attr_data["gaps"]

    gap_df = pd.DataFrame(gap_data).T
    gap_df.columns = [c.capitalize() for c in gap_df.columns]

    plt.figure(figsize=(8, 4))
    sns.heatmap(gap_df, annot=True, fmt=".4f", cmap="RdYlGn_r",
                vmin=0, vmax=0.3, linewidths=1)
    plt.title(f"Fairness Gaps by Attribute (lower = fairer)\nAFI = {afi:.3f}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / "fairness_gaps_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved: {reports_dir / 'fairness_gaps_heatmap.png'}")


def run_fairness_analysis(features_df: pd.DataFrame, y_true: np.ndarray,
                          y_pred: np.ndarray, test_indices: np.ndarray = None,
                          save: bool = True) -> dict:
    """
    Full fairness analysis pipeline.

    Args:
        features_df: Full feature DataFrame (to get sensitive attributes).
        y_true: Ground truth labels for the test set.
        y_pred: Predicted labels for the test set.
        test_indices: Indices of test set rows in features_df.
        save: Whether to save results.

    Returns:
        Dictionary with fairness results and AFI.
    """
    # Get sensitive features for the test set
    sensitive_cols_original = [c for c in features_df.columns if c.endswith("_original")
                               and any(s in c for s in ["gender", "disability", "age_band"])]

    if test_indices is not None:
        sensitive_df = features_df.iloc[test_indices][sensitive_cols_original].reset_index(drop=True)
    else:
        sensitive_df = features_df[sensitive_cols_original].reset_index(drop=True)

    if sensitive_df.empty:
        print("  ⚠ No sensitive attributes found for fairness analysis")
        return {"afi": 1.0, "results": {}}

    # Compute metrics
    fairness_results = compute_fairness_metrics(y_true, y_pred, sensitive_df)

    # Compute AFI
    afi = compute_fairness_index(fairness_results)

    # Plot
    plot_fairness(fairness_results, afi)

    # Save
    if save:
        reports_dir = get_reports_dir()

        # Save per-group metrics
        all_groups = []
        for attr_name, attr_data in fairness_results.items():
            by_group = attr_data["by_group"].copy()
            by_group["attribute"] = attr_name
            by_group["group"] = by_group.index
            all_groups.append(by_group)

        if all_groups:
            fairness_report = pd.concat(all_groups, ignore_index=True)
            fairness_report.to_csv(reports_dir / "fairness_report.csv", index=False)

        # Save AFI
        with open(reports_dir / "afi.txt", "w") as f:
            f.write(f"Algorithmic Fairness Index: {afi:.4f}\n")
            for attr_name, attr_data in fairness_results.items():
                f.write(f"\n{attr_name} gaps:\n")
                for metric, gap in attr_data["gaps"].items():
                    f.write(f"  {metric}: {gap:.4f}\n")

        print(f"  💾 Saved fairness report to {reports_dir}")

    return {
        "afi": afi,
        "results": fairness_results,
    }
