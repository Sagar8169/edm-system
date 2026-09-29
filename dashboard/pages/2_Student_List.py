"""
Page 2: Student List — Searchable, sortable, filterable student table.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_project_root, get_processed_data_dir


def get_models_dir():
    return get_project_root() / "models"


try:
    st.set_page_config(page_title="Student List — EDM System", layout="wide")
except Exception:
    pass

# --- Load Custom CSS ---
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# Student List")
st.markdown("Searchable and filterable table of all students with risk predictions.")
st.markdown("---")

# Load data
features_path = get_processed_data_dir() / "features.csv"
clusters_path = get_processed_data_dir() / "student_clusters.csv"

if not features_path.exists():
    st.error("Features dataset not found. Run `python run_pipeline.py` first.")
    st.stop()

features = pd.read_csv(features_path)

# Merge cluster info if available
if clusters_path.exists():
    clusters = pd.read_csv(clusters_path)
    merge_cols = ["id_student", "code_module", "code_presentation"]
    cols_to_merge = [c for c in clusters.columns if c in merge_cols or c not in features.columns]
    if len(cols_to_merge) > len(merge_cols):
        features = features.merge(clusters[cols_to_merge], on=merge_cols, how="left")

# Generate risk scores using the model if available
model_path = get_models_dir() / "xgb_model.joblib"
scaler_path = get_models_dir() / "scaler.joblib"
feature_cols_path = get_models_dir() / "feature_cols.joblib"

if model_path.exists() and scaler_path.exists() and feature_cols_path.exists():
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(feature_cols_path)

        available_features = [c for c in feature_cols if c in features.columns]
        if len(available_features) == len(feature_cols):
            X = features[feature_cols].values
            X_scaled = scaler.transform(X)
            features["risk_score"] = model.predict_proba(X_scaled)[:, 1]
            features["predicted_risk"] = (features["risk_score"] >= 0.5).astype(int)
        else:
            features["risk_score"] = np.nan
            features["predicted_risk"] = features.get("at_risk", np.nan)
    except Exception:
        features["risk_score"] = np.nan
        features["predicted_risk"] = features.get("at_risk", np.nan)

# --- Search & Filter Controls (On Page) ---
with st.container():
    st.markdown("### Search & Filter Student Records")
    with st.form("filter_form", clear_on_submit=False):
        f_col1, f_col2, f_col3 = st.columns([2, 1.5, 1.5])

        with f_col1:
            search_id = st.text_input("Search Student ID", "", placeholder="Enter ID (e.g. 28400, 30268)...")

        with f_col2:
            risk_filter = st.multiselect(
                "Risk Level Filter",
                ["At Risk", "Not At Risk"],
                default=[],
                placeholder="All Students (no filter selected)",
                help="Leave empty to show all students, or select specific risk levels."
            )

        with f_col3:
            if "code_module" in features.columns:
                mod_col = "code_module_original" if "code_module_original" in features.columns else "code_module"
                modules = ["All Modules"] + sorted(features[mod_col].unique().astype(str).tolist())
                selected_module = st.selectbox("Course Module", modules)
            else:
                selected_module = "All Modules"

        f_col4, f_col5, f_col6 = st.columns([1.5, 1.5, 1.5])
        with f_col4:
            if "cluster" in features.columns:
                cluster_options = ["All Clusters"] + sorted(features["cluster"].dropna().unique().astype(int).astype(str).tolist())
                selected_cluster = st.selectbox("Cluster Group", cluster_options)
            else:
                selected_cluster = "All Clusters"

        with f_col5:
            risk_threshold = st.slider(
                "Min Risk Score Threshold",
                0.0, 1.0, 0.0, 0.05,
                help="Filter students with Risk Score >= this threshold (Set to 0.0 to show all)."
            )

        with f_col6:
            display_limit = st.selectbox(
                "Rows to Display in Table",
                options=["All (All students in CSV)", "500", "1,000", "5,000", "10,000"],
                index=0,
                help="Select how many records to show in the interactive table."
            )

        st.form_submit_button("Apply Filters / Search", type="primary", use_container_width=True)

# --- Apply Filters ---
display_df = features.copy()

if search_id.strip():
    display_df = display_df[display_df["id_student"].astype(str).str.contains(search_id.strip())]

# When no filter is selected (empty) or both are selected, include ALL students from CSV
if risk_filter and len(risk_filter) == 1:
    if "At Risk" in risk_filter:
        display_df = display_df[display_df["at_risk"] == 1]
    elif "Not At Risk" in risk_filter:
        display_df = display_df[display_df["at_risk"] == 0]

mod_col_name = "code_module_original" if "code_module_original" in features.columns else "code_module"
if mod_col_name in features.columns and selected_module not in ["All", "All Modules"]:
    display_df = display_df[display_df[mod_col_name].astype(str) == selected_module]

if "cluster" in features.columns and selected_cluster not in ["All", "All Clusters"]:
    display_df = display_df[display_df["cluster"].astype(int).astype(str) == selected_cluster]

# Apply Min Risk Threshold Filter
if "risk_score" in display_df.columns and risk_threshold > 0.0:
    display_df = display_df[display_df["risk_score"] >= risk_threshold]

# Check if any filter is active
is_filtered = bool(
    search_id.strip()
    or (risk_filter and len(risk_filter) == 1)
    or (selected_module not in ["All", "All Modules"])
    or (selected_cluster not in ["All", "All Clusters"])
    or risk_threshold > 0.0
)

if not is_filtered:
    st.info(f"**No filter selected** — Showing all **{len(features):,}** students loaded from CSV (`features.csv`).")

# --- Summary Stats ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    if not is_filtered:
        st.metric("Total Students (All in CSV)", f"{len(display_df):,}")
    else:
        st.metric("Filtered Students", f"{len(display_df):,} / {len(features):,}")
with col2:
    at_risk_shown = display_df["at_risk"].sum() if "at_risk" in display_df.columns else 0
    st.metric("At Risk", f"{int(at_risk_shown):,}")
with col3:
    if "risk_score" in display_df.columns:
        avg_risk = display_df["risk_score"].mean()
        st.metric("Avg Risk Score (Filtered)", f"{avg_risk:.3f}" if not np.isnan(avg_risk) else "—")
    else:
        st.metric("Avg Risk Score", "—")
with col4:
    if "risk_score" in features.columns:
        total_above_thresh = (features["risk_score"] >= risk_threshold).sum() if risk_threshold > 0.0 else len(features)
        st.metric(f"Students >= {risk_threshold:.2f}", f"{total_above_thresh:,}")
    else:
        st.metric("Risk Threshold Filter", f"{risk_threshold:.2f}")

st.markdown("---")

# --- Student Table ---
# Select display columns
display_columns = ["id_student"]
if "code_module" in display_df.columns:
    display_columns.append("code_module")
if "risk_score" in display_df.columns:
    display_columns.append("risk_score")
display_columns.append("at_risk")
if "cluster" in display_df.columns:
    display_columns.append("cluster")
if "cluster_profile" in display_df.columns:
    display_columns.append("cluster_profile")

# Add key feature columns
for col in ["total_clicks", "avg_score", "days_active", "avg_clicks_per_day"]:
    if col in display_df.columns:
        display_columns.append(col)

if "risk_level" in display_df.columns:
    display_columns.append("risk_level")

# Deduplicate columns
display_columns = list(dict.fromkeys(display_columns))
available_display = [c for c in display_columns if c in display_df.columns]
table_df = display_df[available_display].copy()

# Sort by risk score (highest first)
if "risk_score" in table_df.columns:
    table_df = table_df.sort_values("risk_score", ascending=False)

st.markdown("### Student Table")
if not is_filtered:
    st.markdown(f"*Showing all **{len(table_df):,}** students from CSV (sorted by risk score).*")
else:
    st.markdown(f"*Sorted by risk score (highest first). Showing **{len(table_df):,}** of **{len(features):,}** students.*")

if display_limit.startswith("All"):
    rows_to_show = table_df
else:
    rows_to_show = table_df.head(int(display_limit))

# Configure column formatting for fast, responsive rendering
column_config = {}
if "risk_score" in rows_to_show.columns:
    column_config["risk_score"] = st.column_config.ProgressColumn(
        "Risk Score",
        help="Model estimated dropout/failure probability (0.0 to 1.0)",
        format="%.3f",
        min_value=0.0,
        max_value=1.0,
    )
if "at_risk" in rows_to_show.columns:
    column_config["at_risk"] = st.column_config.NumberColumn(
        "At Risk (1=Yes, 0=No)",
        format="%d",
    )

st.dataframe(
    rows_to_show,
    column_config=column_config,
    width='stretch',
    height=600,
)

# --- CRUD Operations Section ---
st.markdown("---")
with st.expander("Manage Student Records (CRUD Operations)", expanded=False):
    crud_tab1, crud_tab2, crud_tab3 = st.tabs([
        "Add New Student (Create)",
        "Edit Student Record (Update)",
        "Delete Student (Delete)"
    ])

    def save_features_to_csv(df_to_save):
        df_to_save.to_csv(features_path, index=False)
        try:
            st.cache_data.clear()
        except Exception:
            pass

    # --- TAB 1: CREATE ---
    with crud_tab1:
        st.markdown("#### Add a New Student to Database")
        with st.form("create_student_form", clear_on_submit=True):
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                new_id = st.number_input("Student ID", min_value=1, value=999999, step=1)
                new_module = st.selectbox("Course Module", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"])
            with c_col2:
                new_score = st.number_input("Average Score (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
                new_clicks = st.number_input("Total Platform Clicks", min_value=0, value=300, step=10)
            with c_col3:
                new_days = st.number_input("Active Days on LMS", min_value=0, value=40, step=1)
                new_missed = st.number_input("Submissions Missed", min_value=0, value=0, step=1)
                new_at_risk = st.selectbox("At Risk Status", [0, 1], format_func=lambda x: "0 (Not At Risk)" if x == 0 else "1 (At Risk)")

            btn_create = st.form_submit_button("Save New Student", type="primary")
            if btn_create:
                if (features["id_student"] == new_id).any():
                    st.error(f"Student ID {new_id} already exists! Use Update tab to edit.")
                else:
                    new_row = {
                        "id_student": new_id,
                        "code_module": new_module,
                        "avg_score": new_score,
                        "total_clicks": new_clicks,
                        "days_active": new_days,
                        "submissions_missed": new_missed,
                        "at_risk": new_at_risk
                    }
                    for col in features.columns:
                        if col not in new_row:
                            new_row[col] = 0
                    
                    updated_df = pd.concat([features, pd.DataFrame([new_row])], ignore_index=True)
                    save_features_to_csv(updated_df)
                    st.success(f"Student ID {new_id} successfully added!")
                    st.rerun()

    # --- TAB 2: UPDATE ---
    with crud_tab2:
        st.markdown("#### Edit Existing Student Record")
        search_edit_id = st.number_input("Enter Student ID to Edit", min_value=1, step=1, value=28400)
        matching_student = features[features["id_student"] == search_edit_id]

        if matching_student.empty:
            st.warning(f"No student found with ID {search_edit_id}.")
        else:
            s_row = matching_student.iloc[0]
            with st.form("update_student_form"):
                u_col1, u_col2, u_col3 = st.columns(3)
                with u_col1:
                    u_score = st.number_input("Average Score (%)", min_value=0.0, max_value=100.0, value=float(s_row.get("avg_score", 0.0)))
                    u_clicks = st.number_input("Total Platform Clicks", min_value=0, value=int(s_row.get("total_clicks", 0)))
                with u_col2:
                    u_days = st.number_input("Active Days on LMS", min_value=0, value=int(s_row.get("days_active", 0)))
                    u_missed = st.number_input("Submissions Missed", min_value=0, value=int(s_row.get("submissions_missed", 0)))
                with u_col3:
                    u_at_risk = st.selectbox(
                        "At Risk Status", 
                        [0, 1], 
                        index=int(s_row.get("at_risk", 0)),
                        format_func=lambda x: "0 (Not At Risk)" if x == 0 else "1 (At Risk)"
                    )

                btn_update = st.form_submit_button("Update Record", type="primary")
                if btn_update:
                    idx = features[features["id_student"] == search_edit_id].index[0]
                    features.at[idx, "avg_score"] = u_score
                    features.at[idx, "total_clicks"] = u_clicks
                    features.at[idx, "days_active"] = u_days
                    features.at[idx, "submissions_missed"] = u_missed
                    features.at[idx, "at_risk"] = u_at_risk
                    save_features_to_csv(features)
                    st.success(f"Record for Student ID {search_edit_id} updated successfully!")
                    st.rerun()

    # --- TAB 3: DELETE ---
    with crud_tab3:
        st.markdown("#### Delete Student Record")
        del_id = st.number_input("Enter Student ID to Delete", min_value=1, step=1, key="del_id_input")
        matching_del = features[features["id_student"] == del_id]

        if matching_del.empty:
            st.warning(f"No student found with ID {del_id}.")
        else:
            st.error(f"Are you sure you want to delete Student ID **{del_id}**?")
            if st.button("Confirm Delete Student", type="primary"):
                updated_df = features[features["id_student"] != del_id]
                save_features_to_csv(updated_df)
                st.success(f"Student ID {del_id} deleted successfully!")
                st.rerun()

# --- Export ---
st.markdown("---")
st.markdown("### Export Data")

csv = table_df.to_csv(index=False)
st.download_button(
    label=f"Download Student List ({len(table_df):,} students as CSV)",
    data=csv,
    file_name="student_list_export.csv",
    mime="text/csv",
)
