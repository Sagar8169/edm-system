"""
Page 5: Student Portal — Personalized student dashboard with academic status and interactive tools.
"""

import html
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_project_root, get_processed_data_dir
from src.student_names import generate_student_info, get_course_display_name


def get_models_dir():
    return get_project_root() / "models"


try:
    st.set_page_config(page_title="My Student Portal — EDM System", layout="wide")
except Exception:
    pass

# --- Load Custom CSS ---
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Portal stylesheet
PORTAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap');

.pt-hello { padding: 1.2rem 0 1.5rem 0; }
.pt-hello h1 { font-size: clamp(2.0rem, 3.5vw, 2.8rem); margin: 0 0 0.5rem 0; color: #ffffff !important; }
.pt-hello .pt-meta { color: #94a3b8; font-size: 1rem; }
.pt-hello .pt-meta b { color: #f8fafc; font-weight: 600; }

.pt-status {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
}
.pt-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 4px 12px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600;
    border: 1px solid currentColor;
}
.pt-status h2 { font-size: 1.8rem; margin: 0.8rem 0 0.5rem 0; color: #ffffff !important; }
.pt-status p { color: #cbd5e1; font-size: 0.98rem; line-height: 1.5; margin: 0; }

.cmp { padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
.cmp-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.6rem; }
.cmp-label { color: #f8fafc; font-weight: 600; font-size: 0.95rem; }
.cmp-val { font-size: 1.6rem; color: #ffffff; font-weight: 700; }
.cmp-val small { font-size: 0.85rem; color: #94a3b8; margin-left: 4px; }
.track { position: relative; height: 8px; border-radius: 99px; background: rgba(15, 23, 42, 0.8); }
.fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; }
.avg { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #ffffff; opacity: 0.9; }

.tool-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 14px;
    padding: 18px;
    margin-top: 15px;
}
</style>
"""
st.markdown(PORTAL_CSS, unsafe_allow_html=True)


def num(value, default=0.0):
    try:
        v = float(value)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


# --- Check Login State ---
logged_in = st.session_state.get("logged_in", False)
user_role = st.session_state.get("user_role", None)
student_id = st.session_state.get("student_id", None)

features_path = get_processed_data_dir() / "features.csv"
if not features_path.exists():
    st.error("Dataset not found. Please run the pipeline first.")
    st.stop()

features = pd.read_csv(features_path)

if not student_id or user_role != "student":
    demo_ids = [28400, 30268, 11391, 65002, 31604, 8462]
    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        student_id_input = st.number_input("Enter Student ID", value=28400, step=1)
        student_id = int(student_id_input)
    with c_sel2:
        pick = st.selectbox("Sample Student ID", demo_ids, index=0)
        if st.button("Load Student Profile"):
            student_id = pick

student_rows = features[features["id_student"] == student_id]

if student_rows.empty:
    st.error(f"Student ID {student_id} not found.")
    st.stop()

student_row = student_rows.iloc[0]

# Generate student metadata
s_info = generate_student_info(student_id, student_row.get("avg_score", 75.0), student_row.get("days_active", 40.0))
student_name = s_info["student_name"]
student_cgpa = s_info["cgpa"]
student_att = s_info["attendance_pct"]

raw_mod = student_row.get("code_module_original", student_row.get("code_module", "AAA"))
course_name = get_course_display_name(raw_mod)

# Calculate Risk Score
model_path = get_models_dir() / "xgb_model.joblib"
scaler_path = get_models_dir() / "scaler.joblib"
feature_cols_path = get_models_dir() / "feature_cols.joblib"

risk_score = 0.2
if model_path.exists() and scaler_path.exists() and feature_cols_path.exists():
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(feature_cols_path)
        available_feats = [c for c in feature_cols if c in student_row.index]
        if len(available_feats) == len(feature_cols):
            x_vals = student_row[feature_cols].values.reshape(1, -1)
            x_scaled = scaler.transform(x_vals)
            risk_score = float(model.predict_proba(x_scaled)[0, 1])
    except Exception:
        risk_score = 0.5 if student_row.get("at_risk", 0) == 1 else 0.2

# Header
st.markdown(
    f"""
    <div class="pt-hello">
        <h1>Welcome back, {student_name}</h1>
        <div class="pt-meta">Student ID: <b>#{student_id}</b> &nbsp;·&nbsp; Enrolled Course: <b>{course_name}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status Cards
health_score = int(round((1.0 - risk_score) * 100))

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Academic Health Score", f"{health_score} / 100")
with col_m2:
    st.metric("Current CGPA", f"{student_cgpa:.2f}")
with col_m3:
    st.metric("Overall Attendance", f"{student_att:.1f}%")
with col_m4:
    avg_score = num(student_row.get("avg_score", 0.0))
    st.metric("Average Score", f"{avg_score:.1f}%")

st.markdown("---")

# Main Portal Tabs
tab_overview, tab_assignments, tab_planner, tab_cgpa = st.tabs([
    "Academic Overview",
    "Assignment Tracker",
    "Smart Weekly Study Planner",
    "CGPA & Target Score Estimator"
])

with tab_overview:
    col_status, col_gauge = st.columns([3, 2], gap="large")
    with col_status:
        if risk_score < 0.40:
            badge_color, badge_text = "#7FB59A", "On Track"
            head_txt = "You are performing very well!"
            body_txt = "Your assignment scores and engagement levels are consistently high. Keep up the great work!"
        elif risk_score < 0.70:
            badge_color, badge_text = "#E0B25A", "Needs Focus"
            head_txt = "Consistent effort recommended"
            body_txt = "Your activity is slightly below average. Regular logins and submitting pending tasks will boost your score."
        else:
            badge_color, badge_text = "#DB8B78", "Support Recommended"
            head_txt = "Academic assistance available"
            body_txt = "You have pending assignments or low attendance. Reach out to your course counselor or mentor for guidance."

        st.markdown(
            f"""
            <div class="pt-status">
                <span class="pt-pill" style="color:{badge_color}"><i></i>{badge_text}</span>
                <h2>{head_txt}</h2>
                <p>{body_txt}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_gauge:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={"text": "Performance Indicator", "font": {"size": 16, "color": "#cbd5e1"}},
            number={"font": {"size": 50, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38bdf8", "thickness": 0.3},
                "steps": [
                    {"range": [0, 40], "color": "rgba(219, 139, 120, 0.2)"},
                    {"range": [40, 70], "color": "rgba(224, 178, 90, 0.2)"},
                    {"range": [70, 100], "color": "rgba(127, 181, 154, 0.2)"},
                ],
            },
        ))
        fig_g.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, width="stretch")

    # Comparison metrics
    st.markdown("### Class Comparison")
    class_days_med = num(features["days_active"].median(), 45)
    class_score_med = num(features["avg_score"].median(), 75.0)

    my_days = num(student_row.get("days_active", 0))
    my_score = num(student_row.get("avg_score", 0.0))

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Days Active on LMS**: {int(my_days)} days (Class Avg: {int(class_days_med)} days)")
        st.progress(min(1.0, max(0.0, my_days / (class_days_med * 1.5))))
    with c2:
        st.write(f"**Average Assignment Grade**: {my_score:.1f}% (Class Avg: {class_score_med:.1f}%)")
        st.progress(min(1.0, max(0.0, my_score / 100.0)))

with tab_assignments:
    st.markdown("### Interactive Assignment & Quiz Tracker")
    st.markdown("Track your current semester submissions and marks.")
    
    # Mock assignments data customized for student
    assignments_data = [
        {"Assignment": "Module Quiz 1", "Due Date": "Week 3", "Weight": "15%", "Status": "Completed", "Score (%)": max(50.0, avg_score + 4.0)},
        {"Assignment": "Mid-Term Project", "Due Date": "Week 7", "Weight": "30%", "Status": "Completed", "Score (%)": max(45.0, avg_score - 2.0)},
        {"Assignment": "Lab Assessment", "Due Date": "Week 10", "Weight": "20%", "Status": "Completed" if num(student_row.get("submissions_missed", 0)) == 0 else "Pending", "Score (%)": max(40.0, avg_score) if num(student_row.get("submissions_missed", 0)) == 0 else 0.0},
        {"Assignment": "Final Capstone Submission", "Due Date": "Week 14", "Weight": "35%", "Status": "Upcoming", "Score (%)": "-"},
    ]
    df_ass = pd.DataFrame(assignments_data)
    st.dataframe(df_ass, width="stretch")

with tab_planner:
    st.markdown("### Smart Weekly Study Planner")
    st.markdown("Recommended study hours based on your target CGPA and subject difficulty.")
    
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        target_hours = st.slider("Weekly Dedicated Study Hours", 5, 30, 15, step=1)
        study_days = st.multiselect("Preferred Study Days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], default=["Monday", "Wednesday", "Friday", "Saturday"])
    
    with col_p2:
        if study_days:
            hrs_per_day = round(target_hours / len(study_days), 1)
            st.info(f"**Recommended Pace**: ~**{hrs_per_day} hours** on each selected day ({', '.join(study_days)}).")
        else:
            st.warning("Please select at least one preferred study day.")

    schedule_df = pd.DataFrame({
        "Subject / Activity": [f"{course_name} - Video Lectures", f"{course_name} - Practical Lab", "Quiz Practice & Review", "Doubts & Revision"],
        "Weekly Allocated Time": [f"{round(target_hours*0.35, 1)} hrs", f"{round(target_hours*0.35, 1)} hrs", f"{round(target_hours*0.2, 1)} hrs", f"{round(target_hours*0.1, 1)} hrs"],
        "Priority": ["High", "High", "Medium", "Normal"]
    })
    st.table(schedule_df)

with tab_cgpa:
    st.markdown("### CGPA & Target Score Estimator")
    st.markdown("Calculate what score you need in remaining assessments to achieve your target CGPA.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        current_gpa_input = st.number_input("Current CGPA", min_value=0.0, max_value=10.0, value=float(student_cgpa), step=0.1)
        target_gpa = st.number_input("Target CGPA", min_value=0.0, max_value=10.0, value=min(10.0, float(student_cgpa) + 0.5), step=0.1)
    
    with col_c2:
        remaining_weight = st.slider("Remaining Assessment Weight (%)", 10, 60, 35, step=5)
        
        # Simple CGPA projection math
        current_weight = 100 - remaining_weight
        needed_score = ((target_gpa * 10 - current_gpa_input * (current_weight / 10.0)) / (remaining_weight / 10.0)) * 10.0
        
        if needed_score <= 100.0:
            st.success(f"To reach **{target_gpa:.2f} CGPA**, you need an average of **{needed_score:.1f}%** in remaining exams/projects.")
        else:
            st.error(f"Reaching **{target_gpa:.2f} CGPA** requires **{needed_score:.1f}%** (exceeds 100%). Try aiming for a slightly lower target CGPA.")

# Teacher Notes Section
st.markdown("---")
st.markdown("### Notes from Teacher")

feedback_path = get_project_root() / "feedback_log.csv"
found_note = False

if feedback_path.exists():
    try:
        fb_df = pd.read_csv(feedback_path)
        if not fb_df.empty and "student_id" in fb_df.columns:
            student_fb = fb_df[fb_df["student_id"].astype(str) == str(student_id)]
            if not student_fb.empty and "notes" in student_fb.columns:
                valid_notes = student_fb[student_fb["notes"].fillna("").astype(str).str.strip() != ""]
                if not valid_notes.empty:
                    latest = valid_notes.iloc[-1]
                    note_text = html.escape(str(latest.get("notes", "")).strip())
                    st.info(f"**Teacher Feedback**: “{note_text}”")
                    found_note = True
    except Exception:
        pass

if not found_note:
    st.info("No specific alerts from your teacher. You are currently on track with your study plan.")

        )