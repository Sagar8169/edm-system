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
from src.student_names import generate_student_info, get_course_display_name


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
st.markdown("Searchable and filterable table of student records with risk predictions.")
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

# --- Search & Filter Controls ---
with st.container():
    st.markdown("### Search & Filter Student Records")
    with st.form("filter_form", clear_on_submit=False):
        f_col1, f_col2, f_col3 = st.columns([2, 1.5, 1.5])

        with f_col1:
            search_id = st.text_input("Search Student Name / ID", "", placeholder="Enter Name or ID (e.g. 28400)...")

        with f_col2:
            risk_filter = st.multiselect(
                "Risk Level Filter",
                ["At Risk", "Not At Risk"],
                default=[],
                placeholder="All Students",
            )

        with f_col3:
            mod_col = "code_module_original" if "code_module_original" in features.columns else "code_module"
            all_mods = sorted(features[mod_col].dropna().unique().astype(str).tolist())
            mod_options = ["All Modules"] + [get_course_display_name(m) for m in all_mods]
            selected_module_display = st.selectbox("Course Module", mod_options)

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
            )

        with f_col6:
            display_limit = st.selectbox(
                "Rows to Display in Table",
                options=["All (All students)", "100", "500", "1,000", "5,000"],
                index=0,
            )

        st.form_submit_button("Apply Filters / Search", type="primary", use_container_width=True)

# --- Compute Student Metadata (Name, CGPA, Attendance) ---
student_meta_list = []
for idx, row in features.iterrows():
    s_id = row["id_student"]
    score = row.get("avg_score", 75.0)
    days = row.get("days_active", 40.0)
    meta = generate_student_info(s_id, score, days)
    mod_raw = row.get("code_module_original", row.get("code_module", "AAA"))
    meta["course_module"] = get_course_display_name(mod_raw)
    student_meta_list.append(meta)

meta_df = pd.DataFrame(student_meta_list)
features["student_name"] = meta_df["student_name"]
features["cgpa"] = meta_df["cgpa"]
features["attendance_pct"] = meta_df["attendance_pct"]
features["course_display"] = meta_df["course_module"]

# --- Apply Filters ---
display_df = features.copy()

if search_id.strip():
    query = search_id.strip().lower()
    display_df = display_df[
        display_df["id_student"].astype(str).str.lower().str.contains(query) |
        display_df["student_name"].astype(str).str.lower().str.contains(query)
    ]

if risk_filter and len(risk_filter) == 1:
    if "At Risk" in risk_filter:
        display_df = display_df[display_df["at_risk"] == 1]
    elif "Not At Risk" in risk_filter:
        display_df = display_df[display_df["at_risk"] == 0]

if selected_module_display != "All Modules":
    display_df = display_df[display_df["course_display"] == selected_module_display]

if "cluster" in features.columns and selected_cluster not in ["All", "All Clusters"]:
    display_df = display_df[display_df["cluster"].astype(int).astype(str) == selected_cluster]

if "risk_score" in display_df.columns and risk_threshold > 0.0:
    display_df = display_df[display_df["risk_score"] >= risk_threshold]

# --- Summary Stats ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Students Shown", f"{len(display_df):,}")
with col2:
    at_risk_shown = display_df["at_risk"].sum() if "at_risk" in display_df.columns else 0
    st.metric("At Risk", f"{int(at_risk_shown):,}")
with col3:
    avg_cgpa = display_df["cgpa"].mean() if "cgpa" in display_df.columns else 0.0
    st.metric("Average CGPA", f"{avg_cgpa:.2f}")
with col4:
    avg_att = display_df["attendance_pct"].mean() if "attendance_pct" in display_df.columns else 0.0
    st.metric("Avg Attendance", f"{avg_att:.1f}%")

st.markdown("---")

# --- Student Table ---
# Sort by risk score
if "risk_score" in display_df.columns:
    display_df = display_df.sort_values("risk_score", ascending=False)

if not display_limit.startswith("All"):
    display_df = display_df.head(int(display_limit))

# Add proper 1-based sequential numbering
display_df = display_df.reset_index(drop=True)
display_df["S.No"] = range(1, len(display_df) + 1)

# Format Final Columns
table_cols = ["S.No", "id_student", "student_name", "course_display", "cgpa", "attendance_pct", "risk_score", "at_risk"]

if "cluster_profile" in display_df.columns:
    table_cols.append("cluster_profile")
for col in ["total_clicks", "avg_score", "days_active"]:
    if col in display_df.columns:
        table_cols.append(col)

available_table_cols = [c for c in table_cols if c in display_df.columns]
table_df = display_df[available_table_cols].copy()

# Rename headers for clean look
table_df = table_df.rename(columns={
    "id_student": "Student ID",
    "student_name": "Student Name",
    "course_display": "Course Module",
    "cgpa": "CGPA",
    "attendance_pct": "Overall Attendance (%)",
    "risk_score": "Risk Score",
    "at_risk": "At Risk",
    "total_clicks": "Total Clicks",
    "avg_score": "Avg Score (%)",
    "days_active": "Days Active"
})

st.markdown("### Student Roster Table")

column_config = {
    "S.No": st.column_config.NumberColumn("S.No", format="%d", width="small"),
    "Student ID": st.column_config.NumberColumn("Student ID", format="%d"),
    "CGPA": st.column_config.NumberColumn("CGPA", format="%.2f"),
    "Overall Attendance (%)": st.column_config.NumberColumn("Overall Attendance (%)", format="%.1f%%"),
    "Risk Score": st.column_config.ProgressColumn(
        "Risk Score",
        format="%.3f",
        min_value=0.0,
        max_value=1.0,
    ),
    "At Risk": st.column_config.NumberColumn("At Risk (1=Yes, 0=No)", format="%d")
}

st.dataframe(
    table_df,
    column_config=column_config,
    width='stretch',
    height=600,
)

# --- CRUD Operations Section ---
st.markdown("---")
with st.expander("Manage Student Records (Add / Update / Delete)", expanded=False):
    crud_tab1, crud_tab2, crud_tab3 = st.tabs([
        "Add New Student",
        "Edit Student Record",
        "Delete Student"
    ])

    def save_features_to_csv(df_to_save):
        cols_to_drop = ["S.No", "student_name", "cgpa", "attendance_pct", "course_display"]
        clean_save = df_to_save.drop(columns=[c for c in cols_to_drop if c in df_to_save.columns])
        clean_save.to_csv(features_path, index=False)
        try:
            st.cache_data.clear()
        except Exception:
            pass

    with crud_tab1:
        st.markdown("#### Add a New Student")
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
                new_at_risk = st.selectbox("At Risk Status", [0, 1], format_func=lambda x: "0 (Not At Risk)" if x == 0 else "1 (At Risk)")

            btn_create = st.form_submit_button("Save New Student", type="primary")
            if btn_create:
                if (features["id_student"] == new_id).any():
                    st.error(f"Student ID {new_id} already exists!")
                else:
                    new_row = {
                        "id_student": new_id,
                        "code_module": new_module,
                        "avg_score": new_score,
                        "total_clicks": new_clicks,
                        "days_active": new_days,
                        "at_risk": new_at_risk
                    }
                    for col in features.columns:
                        if col not in new_row:
                            new_row[col] = 0
                    
                    updated_df = pd.concat([features, pd.DataFrame([new_row])], ignore_index=True)
                    save_features_to_csv(updated_df)
                    st.success(f"Student ID {new_id} successfully added!")
                    st.rerun()

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
                    features.at[idx, "at_risk"] = u_at_risk
                    save_features_to_csv(features)
                    st.success(f"Record for Student ID {search_edit_id} updated successfully!")
                    st.rerun()

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
