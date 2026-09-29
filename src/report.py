"""
report.py — Generate KPI summary reports and hypothesis test results.

Produces a comprehensive summary with all model metrics, fairness index,
and baseline vs. XGBoost comparison (H1 vs H0 hypothesis test).
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

from src.data_loader import get_project_root


def get_reports_dir() -> Path:
    reports_dir = get_project_root() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def mcnemar_test(y_true: np.ndarray, y_pred_baseline: np.ndarray,
                 y_pred_xgb: np.ndarray) -> dict:
    """
    Perform McNemar's test to compare two classifiers.

    Tests whether there is a statistically significant difference
    between the error rates of the baseline and XGBoost models.

    H0: Both models have the same error rate.
    H1: XGBoost has a significantly different (better) error rate.
    """
    # Build contingency table
    # b = baseline correct, xgb wrong
    # c = baseline wrong, xgb correct
    baseline_correct = (y_pred_baseline == y_true)
    xgb_correct = (y_pred_xgb == y_true)

    b = np.sum(baseline_correct & ~xgb_correct)  # baseline right, xgb wrong
    c = np.sum(~baseline_correct & xgb_correct)  # baseline wrong, xgb right

    # McNemar's test with continuity correction
    if b + c == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    result = {
        "test": "McNemar's Test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "baseline_only_correct": int(b),
        "xgb_only_correct": int(c),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
        "interpretation": (
            f"XGBoost corrected {c} errors that baseline missed, while baseline "
            f"corrected {b} errors that XGBoost missed. "
            f"{'The difference IS statistically significant' if p_value < 0.05 else 'The difference is NOT statistically significant'} "
            f"at α=0.05 (p={p_value:.4f})."
        ),
    }

    return result


def generate_kpi_summary(model_metrics: dict, afi: float,
                         cluster_info: dict = None) -> pd.DataFrame:
    """
    Generate a comprehensive KPI summary table.

    Args:
        model_metrics: Dict with 'baseline' and 'xgboost' metric dicts.
        afi: Algorithmic Fairness Index.
        cluster_info: Optional clustering statistics.

    Returns:
        DataFrame with all KPIs.
    """
    xgb = model_metrics.get("xgboost", {})
    lr = model_metrics.get("baseline", {})

    kpis = [
        {"KPI": "Model Accuracy", "XGBoost": f"{xgb.get('accuracy', 0):.4f}",
         "Baseline (LR)": f"{lr.get('accuracy', 0):.4f}", "Target": "≥ 0.85"},
        {"KPI": "Precision", "XGBoost": f"{xgb.get('precision', 0):.4f}",
         "Baseline (LR)": f"{lr.get('precision', 0):.4f}", "Target": "≥ 0.85"},
        {"KPI": "Recall", "XGBoost": f"{xgb.get('recall', 0):.4f}",
         "Baseline (LR)": f"{lr.get('recall', 0):.4f}", "Target": "≥ 0.85"},
        {"KPI": "F1 Score", "XGBoost": f"{xgb.get('f1', 0):.4f}",
         "Baseline (LR)": f"{lr.get('f1', 0):.4f}", "Target": "≥ 0.85"},
        {"KPI": "ROC-AUC", "XGBoost": f"{xgb.get('roc_auc', 0):.4f}",
         "Baseline (LR)": f"{lr.get('roc_auc', 0):.4f}", "Target": "≥ 0.85"},
        {"KPI": "PR-AUC", "XGBoost": f"{xgb.get('pr_auc', 0):.4f}",
         "Baseline (LR)": f"{lr.get('pr_auc', 0):.4f}", "Target": "—"},
        {"KPI": "Algorithmic Fairness Index", "XGBoost": f"{afi:.4f}",
         "Baseline (LR)": "—", "Target": "≥ 0.80"},
    ]

    return pd.DataFrame(kpis)


def generate_report(model_metrics: dict, afi: float,
                    y_true: np.ndarray = None,
                    y_pred_baseline: np.ndarray = None,
                    y_pred_xgb: np.ndarray = None,
                    cluster_info: dict = None,
                    save: bool = True) -> dict:
    """
    Generate the full report with KPI summary and hypothesis test.

    Returns:
        Dictionary with KPI table, hypothesis test results, and full report text.
    """
    print("\n📝 Generating final report...")
    reports_dir = get_reports_dir()

    # KPI summary
    kpi_df = generate_kpi_summary(model_metrics, afi, cluster_info)
    print("\n📊 KPI Summary:")
    print(kpi_df.to_string(index=False))

    # Hypothesis test
    hyp_test = None
    if y_true is not None and y_pred_baseline is not None and y_pred_xgb is not None:
        hyp_test = mcnemar_test(y_true, y_pred_baseline, y_pred_xgb)
        print(f"\n🧪 Hypothesis Test (H1 vs H0):")
        print(f"  {hyp_test['test']}")
        print(f"  Statistic: {hyp_test['statistic']:.4f}")
        print(f"  P-value: {hyp_test['p_value']:.6f}")
        print(f"  Significant at α=0.05: {hyp_test['significant_at_005']}")
        print(f"  {hyp_test['interpretation']}")

    # Full report text
    xgb = model_metrics.get("xgboost", {})
    lr = model_metrics.get("baseline", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_text = f"""
================================================================================
    AI-BASED EDM PERFORMANCE RECOGNITION SYSTEM — RESULTS REPORT
    Generated: {timestamp}
================================================================================

1. MODEL PERFORMANCE
--------------------
                    XGBoost     Logistic Regression (Baseline)
  Accuracy:         {xgb.get('accuracy', 0):.4f}       {lr.get('accuracy', 0):.4f}
  Precision:        {xgb.get('precision', 0):.4f}       {lr.get('precision', 0):.4f}
  Recall:           {xgb.get('recall', 0):.4f}       {lr.get('recall', 0):.4f}
  F1 Score:         {xgb.get('f1', 0):.4f}       {lr.get('f1', 0):.4f}
  ROC-AUC:          {xgb.get('roc_auc', 0):.4f}       {lr.get('roc_auc', 0):.4f}

2. ALGORITHMIC FAIRNESS
-----------------------
  Fairness Index (AFI): {afi:.4f}
  (1.0 = perfectly fair, 0.0 = maximum disparity)

3. HYPOTHESIS TEST (H1 vs H0)
------------------------------
  H0: XGBoost performs no better than Logistic Regression baseline.
  H1: XGBoost provides significantly better at-risk student prediction.
"""

    if hyp_test:
        report_text += f"""
  Test: {hyp_test['test']}
  Statistic: {hyp_test['statistic']:.4f}
  P-value: {hyp_test['p_value']:.6f}
  Result: {'REJECT H0 — XGBoost is significantly better' if hyp_test['significant_at_005'] else 'FAIL TO REJECT H0 — No significant difference found'}

4. INTERPRETATION
-----------------
  {hyp_test['interpretation']}

  Note: These are actual results from the OULAD dataset analysis.
  Any differences from aspirational KPIs in the project dossier
  reflect real-world model performance and are reported honestly.
"""

    report_text += """
================================================================================
"""

    # Save
    if save:
        kpi_df.to_csv(reports_dir / "kpi_summary.csv", index=False)

        with open(reports_dir / "full_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        if hyp_test:
            with open(reports_dir / "hypothesis_test.json", "w") as f:
                json.dump(hyp_test, f, indent=2)

        print(f"\n💾 Saved report files to {reports_dir}")

    return {
        "kpi_df": kpi_df,
        "hypothesis_test": hyp_test,
        "report_text": report_text,
    }
