"""
run_pipeline.py — Master Orchestration Script.

Runs the complete EDM pipeline end-to-end:
    1. Download + load OULAD data
    2. Preprocess + merge
    3. Engineer features + EDA
    4. Train models (Logistic Regression baseline + XGBoost)
    5. Generate SHAP explanations
    6. Run personalization clustering
    7. Run fairness analysis
    8. Generate final report

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-download   # Skip data download (use existing)
    python run_pipeline.py --no-tune         # Skip hyperparameter tuning (faster)
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Force UTF-8 encoding for stdout on Windows to support emojis/checkmarks
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(features_df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Generate Exploratory Data Analysis plots."""
    reports_dir = PROJECT_ROOT / "reports" / "eda"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n📊 Generating EDA plots...")

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = features_df["at_risk"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(["Not At Risk (0)", "At Risk (1)"], [counts.get(0, 0), counts.get(1, 0)], color=colors)
    ax.set_title("Class Balance: At-Risk Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{count:,}\n({count/len(features_df)*100:.1f}%)",
                ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ class_balance.png")

    # 2. Correlation heatmap
    numeric_features = [c for c in feature_cols if c in features_df.columns
                       and features_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    if len(numeric_features) > 2:
        corr_cols = numeric_features[:15] + ["at_risk"]  # Top 15 + target
        corr_cols = [c for c in corr_cols if c in features_df.columns]
        corr_matrix = features_df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                    annot_kws={"size": 8})
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(reports_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ correlation_heatmap.png")

    # 3. Feature distributions by risk status
    plot_features = ["total_clicks", "avg_score", "days_active", "avg_clicks_per_day",
                     "submissions_missed", "score_std"]
    plot_features = [f for f in plot_features if f in features_df.columns]

    if plot_features:
        n_cols = min(3, len(plot_features))
        n_rows = (len(plot_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, feat in enumerate(plot_features):
            ax = axes[i]
            for risk, color, label in [(0, "#2ecc71", "Not At Risk"), (1, "#e74c3c", "At Risk")]:
                data = features_df[features_df["at_risk"] == risk][feat].dropna()
                ax.hist(data, bins=30, alpha=0.6, color=color, label=label, density=True)
            ax.set_title(feat, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.set_xlabel(feat)
            ax.set_ylabel("Density")

        # Hide unused axes
        for j in range(len(plot_features), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Distributions by At-Risk Status", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(reports_dir / "feature_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ feature_distributions.png")

    # 4. Boxplots
    if plot_features:
        fig, axes = plt.subplots(1, min(4, len(plot_features)), figsize=(5 * min(4, len(plot_features)), 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for i, feat in enumerate(plot_features[:4]):
            ax = axes[i]
            features_df.boxplot(column=feat, by="at_risk", ax=ax)
            ax.set_title(feat, fontsize=11, fontweight="bold")
            ax.set_xlabel("At Risk")
        plt.suptitle("Feature Boxplots by Risk Status", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(reports_dir / "feature_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ feature_boxplots.png")


def main():
    parser = argparse.ArgumentParser(description="EDM Pipeline — Full End-to-End Run")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  AI-BASED EDM PERFORMANCE RECOGNITION SYSTEM")
    print("  Full Pipeline Execution")
    print("=" * 70)

    # ============================================================
    # STEP 1: Load Data
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 1: Load OULAD Data")
    print("=" * 70)

    from src.data_loader import load_oulad, download_oulad

    if not args.skip_download:
        download_oulad()

    data = load_oulad()

    # ============================================================
    # STEP 2: Preprocess
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 2: Preprocess + Merge")
    print("=" * 70)

    from src.preprocessing import run_preprocessing
    student_master, label_encoders = run_preprocessing(data)

    # ============================================================
    # STEP 3: Feature Engineering
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 3: Feature Engineering")
    print("=" * 70)

    from src.feature_engineering import run_feature_engineering
    features_df, feature_cols = run_feature_engineering(student_master)

    # ============================================================
    # STEP 4: Exploratory Data Analysis
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 4: Exploratory Data Analysis")
    print("=" * 70)

    run_eda(features_df, feature_cols)

    # ============================================================
    # STEP 5: Model Training
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 5: Model Training")
    print("=" * 70)

    from src.model import run_modeling
    model_results = run_modeling(features_df, feature_cols, tune=not args.no_tune)

    # ============================================================
    # STEP 6: Explainability (SHAP)
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 6: SHAP Explainability")
    print("=" * 70)

    from src.explainability import run_explainability
    shap_results = run_explainability(model_results)

    # ============================================================
    # STEP 7: Personalization Clustering
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 7: Personalization Clustering")
    print("=" * 70)

    from src.clustering import run_clustering
    cluster_results = run_clustering(features_df, feature_cols)

    # ============================================================
    # STEP 8: Fairness Analysis
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 8: Fairness Analysis")
    print("=" * 70)

    from src.fairness import run_fairness_analysis

    # Get predictions for fairness analysis
    xgb_model = model_results["xgb_model"]
    X_test = model_results["X_test"]
    y_test = model_results["y_test"]
    y_pred_xgb = xgb_model.predict(X_test)

    # Get test indices to map back to features_df for sensitive attributes
    from sklearn.model_selection import train_test_split
    _, test_indices = train_test_split(
        np.arange(len(features_df)),
        test_size=0.2,
        random_state=42,
        stratify=features_df["at_risk"].values
    )

    fairness_results = run_fairness_analysis(
        features_df, y_test, y_pred_xgb, test_indices
    )

    # ============================================================
    # STEP 9: Report Generation
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  STEP 9: Report Generation")
    print("=" * 70)

    from src.report import generate_report
    import json

    # Load model metrics
    metrics_path = PROJECT_ROOT / "reports" / "model_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            model_metrics = json.load(f)
    else:
        model_metrics = {
            "baseline": model_results["lr_metrics"],
            "xgboost": model_results["xgb_metrics"],
        }

    # Get baseline predictions for hypothesis test
    lr_model = model_results["lr_model"]
    y_pred_baseline = lr_model.predict(X_test)

    report = generate_report(
        model_metrics=model_metrics,
        afi=fairness_results["afi"],
        y_true=y_test,
        y_pred_baseline=y_pred_baseline,
        y_pred_xgb=y_pred_xgb,
    )

    # ============================================================
    # DONE
    # ============================================================
    elapsed = time.time() - start_time

    print("\n\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETE")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    print(f"""
Generated outputs:
  📁 data/processed/student_master.csv
  📁 data/processed/features.csv
  📁 data/processed/student_clusters.csv
  📁 models/xgb_model.joblib
  📁 models/lr_baseline.joblib
  📁 models/shap_values.npy
  📁 reports/eda/*.png
  📁 reports/shap/*.png
  📁 reports/model/model_evaluation.png
  📁 reports/clustering/cluster_scatter.png
  📁 reports/fairness/fairness_analysis.png
  📁 reports/kpi_summary.csv
  📁 reports/full_report.txt

To launch the dashboard:
  cd {PROJECT_ROOT}
  python backend/app.py                             # Flask API + Web UI on http://localhost:5000
  streamlit run dashboard/Student_Development.py     # Streamlit Dashboard on http://localhost:8501
""")


if __name__ == "__main__":
    main()
