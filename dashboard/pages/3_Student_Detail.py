"""
Page 3: Student Detail — Individual student SHAP explanation + HITL override.
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_project_root, get_processed_data_dir
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_models_dir():
    return get_project_root() / "models"


try:
    st.set_page_config(page_title="Student Detail — EDM System", layout="wide")
except Exception:
    pass

# --- Load Custom CSS ---
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.markdown("# Student Detail")
st.markdown("View individual student predictions with SHAP explanations and provide teacher overrides.")

user_role = st.session_state.get("user_role", None)
if user_role == "student":
    st.info("You are viewing the Teacher Detail page. To see your student-focused coaching view, click below:")
    st.page_link("pages/5_Student_Portal.py", label="**Go to My Student Portal**")

st.markdown("---")

# Load data
features_path = get_processed_data_dir() / "features.csv"
clusters_path = get_processed_data_dir() / "student_clusters.csv"

if not features_path.exists():
    st.error("Features dataset not found. Run `python run_pipeline.py` first.")
    st.stop()

features = pd.read_csv(features_path)

# Merge clusters
if clusters_path.exists():
    clusters = pd.read_csv(clusters_path)
    merge_cols = ["id_student", "code_module", "code_presentation"]
    cols_to_merge = [c for c in clusters.columns if c in merge_cols or c not in features.columns]
    if len(cols_to_merge) > len(merge_cols):
        features = features.merge(clusters[cols_to_merge], on=merge_cols, how="left")

# Load model artifacts
model_path = get_models_dir() / "xgb_model.joblib"
scaler_path = get_models_dir() / "scaler.joblib"
feature_cols_path = get_models_dir() / "feature_cols.joblib"
shap_explainer_path = get_models_dir() / "shap_explainer.joblib"

model = joblib.load(model_path) if model_path.exists() else None
scaler = joblib.load(scaler_path) if scaler_path.exists() else None
feature_cols = joblib.load(feature_cols_path) if feature_cols_path.exists() else None
shap_explainer = joblib.load(shap_explainer_path) if shap_explainer_path.exists() else None

if model is None or scaler is None or feature_cols is None:
    st.error("Model artifacts not found. Run `python run_pipeline.py` first.")
    st.stop()

# --- Search & Select Student (On Page) ---
student_ids = sorted(features["id_student"].unique())

def update_student_selection(source_key):
    val = st.session_state.get(source_key)
    if val:
        try:
            val_int = int(val)
            if val_int in student_ids:
                st.session_state["student_id"] = val_int
                st.session_state["select_id_detail"] = val_int
                st.session_state["override_student_selector"] = val_int
        except Exception:
            pass

# Initialize student_id in session state if not set
if "student_id" not in st.session_state or st.session_state["student_id"] not in student_ids:
    st.session_state["student_id"] = student_ids[0]

# Ensure widget keys are set if missing
if "select_id_detail" not in st.session_state or st.session_state["select_id_detail"] not in student_ids:
    st.session_state["select_id_detail"] = st.session_state["student_id"]
if "override_student_selector" not in st.session_state or st.session_state["override_student_selector"] not in student_ids:
    st.session_state["override_student_selector"] = st.session_state["student_id"]

with st.container():
    st.markdown("### Search & Select Student")
    col_search, col_dropdown, col_mod = st.columns([2, 2, 1.5])

    with col_search:
        search_id_input = st.text_input("Search Student ID", value="", placeholder="Type student ID (e.g. 28400)...", key="search_id_detail")
        if search_id_input.strip():
            try:
                val = int(search_id_input.strip())
                if val in student_ids and val != st.session_state.get("student_id"):
                    st.session_state["student_id"] = val
                    st.session_state["select_id_detail"] = val
                    st.session_state["override_student_selector"] = val
                    st.rerun()
                elif val not in student_ids:
                    st.warning(f"Student ID {val} not found in database.")
            except ValueError:
                st.error("Please enter a numeric Student ID.")

    current_idx = student_ids.index(st.session_state["student_id"])

    with col_dropdown:
        selected_id = st.selectbox(
            "Select Student from Roster", 
            student_ids, 
            index=current_idx, 
            key="select_id_detail",
            on_change=update_student_selection,
            args=("select_id_detail",)
        )

    student_rows = features[features["id_student"] == selected_id]

    with col_mod:
        mod_col = "code_module_original" if "code_module_original" in student_rows.columns else "code_module"
        if len(student_rows) > 1 and mod_col in student_rows.columns:
            modules = student_rows[mod_col].unique()
            selected_module = st.selectbox("Select Module Enrollment", modules)
            student_row = student_rows[student_rows[mod_col] == selected_module].iloc[0]
        else:
            mod_val = student_rows.iloc[0].get(mod_col, "N/A")
            st.text_input("Enrolled Module", value=str(mod_val), disabled=True)
            student_row = student_rows.iloc[0]

# --- Student Profile ---
st.markdown(f"## Student: `{selected_id}`")

col1, col2, col3, col4 = st.columns(4)

# Get risk score
available_features = [c for c in feature_cols if c in features.columns]
if len(available_features) == len(feature_cols):
    X_student = student_row[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X_student)
    risk_score = model.predict_proba(X_scaled)[0][1]
    predicted_risk = int(risk_score >= 0.5)
else:
    risk_score = 0.5
    predicted_risk = student_row.get("at_risk", 0)

with col1:
    st.metric("Risk Score", f"{risk_score:.3f}")
with col2:
    st.metric("Predicted", "At Risk" if predicted_risk else "Not At Risk")
with col3:
    actual = student_row.get("final_result", "Unknown")
    st.metric("Actual Result", actual)
with col4:
    cluster_name = student_row.get("cluster_profile", "N/A")
    st.metric("Cluster", cluster_name)

st.markdown("---")

# --- Feature Details ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### Student Features")

    # Show key features in a nice table
    feature_display = []
    for col_name in feature_cols:
        if col_name in student_row.index:
            val = student_row[col_name]
            feature_display.append({"Feature": col_name, "Value": f"{val:.2f}" if isinstance(val, float) else str(val)})

    feature_table = pd.DataFrame(feature_display)
    st.dataframe(feature_table, width='stretch', height=400)

with col_right:
    st.markdown("### SHAP Explanation (Feature Impact)")
    st.caption("**Full Form:** **SHapley Additive exPlanations** — Measures how much each feature pushes this student's risk prediction up or down.")

    if shap_explainer is not None and len(available_features) == len(feature_cols):
        # Compute SHAP values for this student
        shap_vals = shap_explainer.shap_values(X_scaled)[0]

        # Create a horizontal bar chart of SHAP contributions
        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP Value": shap_vals,
            "Abs SHAP": np.abs(shap_vals),
        }).sort_values("Abs SHAP", ascending=False).head(15)

        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["SHAP Value"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=shap_df["SHAP Value"].values[::-1],
            y=shap_df["Feature"].values[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:+.3f}" for v in shap_df["SHAP Value"].values[::-1]],
            textposition="outside",
        ))

        fig.update_layout(
            title=f"Top 15 SHAP Contributions (Base: {shap_explainer.expected_value:.3f})",
            xaxis_title="SHAP Value (SHapley Additive exPlanations: -> increases risk, <- decreases risk)",
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=80),
        )
        st.plotly_chart(fig, width='stretch')

        # Text explanation
        top_risk = shap_df[shap_df["SHAP Value"] > 0].head(3)
        top_protect = shap_df[shap_df["SHAP Value"] < 0].head(3)

        st.markdown("**Key Risk Factors:**")
        for _, row in top_risk.iterrows():
            st.markdown(f"- **{row['Feature']}** pushes risk up by {row['SHAP Value']:+.3f}")

        st.markdown("**Protective Factors:**")
        for _, row in top_protect.iterrows():
            st.markdown(f"- **{row['Feature']}** reduces risk by {row['SHAP Value']:+.3f}")
    else:
        st.warning("SHAP explainer not available. Run the pipeline to generate explanations.")

# --- Intervention Recommendation ---
if "intervention" in student_row.index and pd.notna(student_row.get("intervention")):
    st.markdown("---")
    st.markdown("### Recommended Intervention")
    risk_level = student_row.get("risk_level", "Unknown")
    st.markdown(f"**Risk Level:** {risk_level}")
    st.info(student_row["intervention"])

# --- HITL Override ---
st.markdown("---")
st.markdown("### Teacher Override (Human-in-the-Loop)")
st.markdown("Review model predictions and submit teacher confirmation or manual override decision for any student:")

override_idx = student_ids.index(st.session_state["student_id"]) if st.session_state["student_id"] in student_ids else 0

o_sel_col1, o_sel_col2 = st.columns([2, 3])
with o_sel_col1:
    override_student_id = st.selectbox(
        "Select / Change Student ID to Override",
        options=student_ids,
        index=override_idx,
        key="override_student_selector",
        on_change=update_student_selection,
        args=("override_student_selector",)
    )

st.info(f"**Target Student:** `Student #{selected_id}` | **Current Model Risk Score:** `{risk_score:.3f}` | **Model Status:** `{'At Risk' if predicted_risk else 'Not At Risk'}`")

col_btn1, col_btn2, col_notes = st.columns([1, 1, 2])

with col_notes:
    teacher_notes = st.text_area(f"Teacher Notes for Student #{selected_id} (optional)", "", height=80)

feedback_log_path = get_project_root() / "feedback_log.csv"

with col_btn1:
    if st.button(f"Confirm Risk for #{selected_id}", type="primary", use_container_width=True):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "student_id": int(selected_id),
            "code_module": student_row.get("code_module", ""),
            "code_presentation": student_row.get("code_presentation", ""),
            "original_prediction": int(predicted_risk),
            "risk_score": f"{risk_score:.4f}",
            "teacher_decision": "confirmed_at_risk",
            "notes": teacher_notes,
        }

        # Append to CSV
        log = pd.read_csv(feedback_log_path) if feedback_log_path.exists() else pd.DataFrame()
        log = pd.concat([log, pd.DataFrame([entry])], ignore_index=True)
        log.to_csv(feedback_log_path, index=False)

        st.success(f"Risk assessment for Student #{selected_id} confirmed and logged!")

with col_btn2:
    if st.button(f"Override for #{selected_id}", type="secondary", use_container_width=True):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "student_id": int(selected_id),
            "code_module": student_row.get("code_module", ""),
            "code_presentation": student_row.get("code_presentation", ""),
            "original_prediction": int(predicted_risk),
            "risk_score": f"{risk_score:.4f}",
            "teacher_decision": "overridden_not_at_risk",
            "notes": teacher_notes,
        }

        log = pd.read_csv(feedback_log_path) if feedback_log_path.exists() else pd.DataFrame()
        log = pd.concat([log, pd.DataFrame([entry])], ignore_index=True)
        log.to_csv(feedback_log_path, index=False)

        st.success(f"Override logged for Student #{selected_id}! Model prediction has been corrected by teacher.")

# --- Feedback Log ---
st.markdown("---")
st.markdown("### Recent Feedback Log")
if feedback_log_path.exists():
    log = pd.read_csv(feedback_log_path)
    if len(log) > 0:
        st.dataframe(log.tail(20).sort_values("timestamp", ascending=False), width='stretch')
    else:
        st.info("No feedback entries yet. Use the buttons above to log teacher decisions.")
else:
    st.info("No feedback log yet. Feedback will be saved to `feedback_log.csv`.")
