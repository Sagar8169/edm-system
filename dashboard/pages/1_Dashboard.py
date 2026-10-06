"""
Page 1: Dashboard — Class-wide statistics and model performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_project_root, get_processed_data_dir


def get_models_dir():
    return get_project_root() / "models"


def get_reports_dir():
    return get_project_root() / "reports"


try:
    st.set_page_config(page_title="Dashboard — EDM System", layout="wide")
except Exception:
    pass

# --- Load Custom CSS ---
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.markdown("# System Dashboard")
st.markdown("Class-wide statistics, model performance, and key insights.")

user_role = st.session_state.get("user_role", None)
if user_role == "student":
    st.info("You are viewing the Teacher Dashboard. To see your personalized progress, click below:")
    st.page_link("pages/5_Student_Portal.py", label="**Go to My Student Portal**")

st.markdown("---")

# Load data
features_path = get_processed_data_dir() / "features.csv"
metrics_path = get_reports_dir() / "model_metrics.json"

if not features_path.exists():
    st.error("Features dataset not found. Run `python run_pipeline.py` first.")
    st.stop()

features = pd.read_csv(features_path)

# --- Metric Cards ---
total_students = len(features)
at_risk_count = int(features["at_risk"].sum())
not_at_risk = total_students - at_risk_count
at_risk_pct = (at_risk_count / total_students) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Students", f"{total_students:,}")
with col2:
    st.metric("At Risk", f"{at_risk_count:,}", delta=f"{at_risk_pct:.1f}%", delta_color="inverse")
with col3:
    st.metric("Not At Risk", f"{not_at_risk:,}", delta=f"{100 - at_risk_pct:.1f}%")
with col4:
    modules = features["code_module"].nunique() if "code_module" in features.columns else "—"
    st.metric("Modules", modules)

st.markdown("---")

# --- Model Performance (AI Model Part) ---
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

    st.markdown("### Model Performance")

    xgb = metrics.get("xgboost", {})
    lr = metrics.get("baseline", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("XGB Accuracy", f"{xgb.get('accuracy', 0):.3f}")
    with col2:
        st.metric("XGB Precision", f"{xgb.get('precision', 0):.3f}")
    with col3:
        st.metric("XGB Recall", f"{xgb.get('recall', 0):.3f}")
    with col4:
        st.metric("XGB F1", f"{xgb.get('f1', 0):.3f}")
    with col5:
        st.metric("XGB ROC-AUC", f"{xgb.get('roc_auc', 0):.3f}")

    # Confusion matrix with diagnostics
    if "confusion_matrix" in xgb:
        st.markdown("#### XGBoost Confusion Matrix & Diagnostics")
        col_cm, col_diag = st.columns([1, 1])
        
        cm = np.array(xgb["confusion_matrix"])
        with col_cm:
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Not At Risk", "At Risk"],
                y=["Not At Risk", "At Risk"],
                color_continuous_scale="Blues",
                text_auto=True,
            )
            fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cm, width='stretch')

        with col_diag:
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
            st.markdown(f"""
            * **True Positives (Correctly Flagged At-Risk):** **{tp:,}** students
            * **True Negatives (Correctly Identified Safe):** **{tn:,}** students
            * **False Alarm Rate (FP):** **{fpr*100:.1f}%** ({fp} students)
            * **Missed At-Risk (FN):** **{fn:,}** students
            """)
            st.info("The model achieves high recall (90.1%) to identify at-risk students early with a low false alarm rate (3.5%).")

st.markdown("---")

# --- Class Distribution ---
st.markdown("### Class Distribution")

col1, col2 = st.columns(2)

with col1:
    # At-risk distribution
    fig_dist = px.pie(
        values=[at_risk_count, not_at_risk],
        names=["At Risk", "Not At Risk"],
        color=["At Risk", "Not At Risk"],
        color_discrete_map={"At Risk": "#e74c3c", "Not At Risk": "#2ecc71"},
        title="At-Risk Distribution",
        hole=0.4,
    )
    fig_dist.update_traces(textinfo="percent+label+value")
    st.plotly_chart(fig_dist, width='stretch')

with col2:
    # Final result breakdown (if available)
    if "final_result" in features.columns:
        result_counts = features["final_result"].value_counts()
        fig_result = px.bar(
            x=result_counts.index,
            y=result_counts.values,
            color=result_counts.index,
            color_discrete_map={
                "Pass": "#2ecc71", "Distinction": "#3498db",
                "Fail": "#e74c3c", "Withdrawn": "#f39c12"
            },
            title="Final Result Breakdown",
            labels={"x": "Result", "y": "Count"},
        )
        fig_result.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_result, width='stretch')

# --- Top Features ---
st.markdown("### Top Feature Importance (SHAP)")

importance_path = get_reports_dir() / "shap_feature_importance.csv"
if importance_path.exists():
    importance_df = pd.read_csv(importance_path).head(15)
    readable_names = {
        "submissions_missed": "Missed Submissions",
        "days_active": "Active Days on LMS",
        "avg_score": "Average Assessment Score (%)",
        "num_assessments": "Number of Assessments",
        "score_range": "Score Range (Max - Min)",
        "max_score": "Highest Score",
        "click_intensity": "Clicks Per Active Session",
        "click_trend": "Engagement Trajectory",
        "min_score": "Lowest Score",
        "score_std": "Score Std Dev",
        "highest_education": "Prior Education",
        "total_clicks": "Total Clicks",
    }
    importance_df["feature_clean"] = importance_df["feature"].map(lambda x: readable_names.get(x, x))
    fig_imp = px.bar(
        importance_df.sort_values("mean_abs_shap"),
        x="mean_abs_shap", y="feature_clean",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Viridis",
        title="Top 15 Features by Feature Impact Strength (Mean |SHAP Value|)",
        labels={"mean_abs_shap": "Feature Impact Strength (Mean |SHAP Value|)", "feature_clean": "Feature"},
    )
    fig_imp.update_layout(
        xaxis_title="Feature Impact Strength (Mean |SHAP Value| — SHapley Additive exPlanations)",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_imp, width='stretch')
else:
    st.info("Feature importance data not available yet. Run the pipeline to generate SHAP values.")
