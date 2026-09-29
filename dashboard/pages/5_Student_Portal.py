"""
Page 5: Student Portal — Personalized, encouraging student view with academic health and coaching recommendations.
"""

import html
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_project_root, get_processed_data_dir


def get_models_dir():
    return get_project_root() / "models"


try:
    st.set_page_config(page_title="My Student Portal — EDM System", layout="wide")
except Exception:
    pass

# --- Load Custom CSS (existing project stylesheet) ---
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Portal theme (matches the login page: navy + brass) ---
PORTAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #0E1726;
    --surface: #142033;
    --surface-raised: #1A2A42;
    --hairline: #26364F;
    --text: #F1EEE7;
    --text-muted: #9AA7BD;
    --brass: #C9A45C;
    --sage: #7FB59A;
    --amber: #E0B25A;
    --coral: #DB8B78;
}

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 88% -10%, rgba(201, 164, 92, 0.10), transparent 60%),
        var(--ink) !important;
    color: var(--text);
    font-family: 'Instrument Sans', system-ui, sans-serif;
}
[data-testid="stAppViewContainer"] .main .block-container { max-width: 1120px; padding-top: 1.5rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { font-family: 'Newsreader', Georgia, serif !important; color: var(--text) !important; font-weight: 500 !important; letter-spacing: -0.01em; }

/* Greeting */
.pt-hello { padding: 1.4rem 0 1.8rem 0; }
.pt-hello h1 { font-size: clamp(2.1rem, 4vw, 3.1rem); line-height: 1.1; margin: 0 0 0.7rem 0; }
.pt-hello .pt-meta { color: var(--text-muted); font-size: 1rem; }
.pt-hello .pt-meta b { color: var(--text); font-weight: 600; }

/* Section titles */
.pt-section { margin: 2.6rem 0 0.4rem 0; }
.pt-section h2 { font-size: 1.7rem; margin: 0 0 0.3rem 0; }
.pt-section p { color: var(--text-muted); margin: 0 0 0.6rem 0; font-size: 0.98rem; }

/* Status panel */
.pt-status {
    background: linear-gradient(180deg, var(--surface-raised), var(--surface));
    border: 1px solid var(--hairline);
    border-radius: 20px;
    padding: 1.9rem 2rem;
    height: 100%;
    box-shadow: 0 30px 60px -34px rgba(0, 0, 0, 0.75), inset 0 1px 0 rgba(255,255,255,0.04);
}
.pt-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 5px 13px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600;
    border: 1px solid currentColor;
}
.pt-pill i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }
.pt-status h2 { font-size: 2rem; line-height: 1.15; margin: 1.1rem 0 0.7rem 0; }
.pt-status p { color: var(--text-muted); font-size: 1.02rem; line-height: 1.65; max-width: 52ch; margin: 0; }

/* Comparison rows */
.cmp { padding: 1.2rem 0; border-bottom: 1px solid var(--hairline); }
.cmp:first-of-type { border-top: 1px solid var(--hairline); }
.cmp-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.8rem; }
.cmp-label { color: var(--text); font-weight: 600; font-size: 1rem; }
.cmp-val { font-family: 'Newsreader', Georgia, serif; font-size: 1.9rem; color: var(--text); }
.cmp-val small { font-family: 'Instrument Sans', sans-serif; font-size: 0.9rem; color: var(--text-muted); margin-left: 4px; }
.track { position: relative; height: 8px; border-radius: 99px; background: var(--surface-raised); }
.fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; }
.avg { position: absolute; top: -5px; bottom: -5px; width: 2px; background: var(--text); opacity: 0.85; border-radius: 2px; }
.cmp-foot { margin-top: 0.75rem; color: var(--text-muted); font-size: 0.9rem; }
.cmp-foot b { font-weight: 600; }
.good { color: var(--sage); }
.warn { color: var(--amber); }

/* Pending assignments */
.pending { display: flex; align-items: center; gap: 1.2rem; padding: 1.3rem 0; border-bottom: 1px solid var(--hairline); }
.pending .num { font-family: 'Newsreader', Georgia, serif; font-size: 2.6rem; line-height: 1; min-width: 2.2rem; }
.pending .txt strong { display: block; font-size: 1rem; color: var(--text); }
.pending .txt span { color: var(--text-muted); font-size: 0.92rem; }

/* Tips */
.tip { display: flex; gap: 14px; padding: 1.1rem 0; border-bottom: 1px solid var(--hairline); }
.tip:first-of-type { border-top: 1px solid var(--hairline); }
.tip .mark { flex: none; width: 9px; height: 9px; margin-top: 8px; border-radius: 50%; background: var(--brass); }
.tip h4 { margin: 0 0 0.3rem 0; font-family: 'Instrument Sans', sans-serif !important; font-weight: 600 !important; font-size: 1.02rem; letter-spacing: 0; }
.tip p { margin: 0; color: var(--text-muted); font-size: 0.95rem; line-height: 1.6; max-width: 58ch; }

/* Advisor note */
.note {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
}
.note .quote { font-family: 'Newsreader', Georgia, serif; font-size: 1.25rem; line-height: 1.5; color: var(--text); margin: 0 0 1rem 0; }
.note .by { color: var(--text-muted); font-size: 0.88rem; }
.note.calm .quote { font-size: 1.1rem; color: var(--text-muted); }

/* Native widgets */
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--hairline); background: var(--surface); }
[data-testid="stNumberInput"] input, [data-baseweb="select"] > div {
    background: var(--ink) !important; border-color: var(--hairline) !important; color: var(--text) !important; border-radius: 10px !important;
}
.stButton > button { border-radius: 10px; font-weight: 600; border: 1px solid var(--hairline); background: transparent; color: var(--text); }
.stButton > button:hover { border-color: var(--brass); color: var(--brass); }
button:focus-visible { outline: 2px solid var(--brass) !important; outline-offset: 2px; }
[data-testid="stHeader"] { background: rgba(14, 23, 38, 0.72) !important; backdrop-filter: blur(14px); border-bottom: 1px solid var(--hairline); }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { visibility: hidden; height: 0; }

@media (max-width: 800px) { .pt-status { padding: 1.4rem 1.3rem; } }
</style>
"""
st.markdown(PORTAL_CSS, unsafe_allow_html=True)


def block(s: str) -> str:
    """Collapse HTML to one line so Markdown never treats indentation as a code block."""
    return "".join(line.strip() for line in s.splitlines())


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


# Load data
features_path = get_processed_data_dir() / "features.csv"
if not features_path.exists():
    st.error("Dataset not found. Please run the pipeline first.")
    st.stop()

features = pd.read_csv(features_path)

# If teacher is testing the student portal or student is not selected, let them choose a student ID
if not student_id or user_role != "student":
    st.info("**Student ID lookup:** enter or pick a student ID below to preview their portal.")
    demo_ids = [28400, 30268, 11391, 65002, 31604, 8462]
    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        student_id_input = st.number_input("Enter Student ID", value=28400, step=1)
        student_id = int(student_id_input)
    with c_sel2:
        st.write("Or pick a sample student:")
        pick = st.selectbox("Sample Student", demo_ids, index=0)
        if st.button("Load Selected Sample"):
            student_id = pick

student_rows = features[features["id_student"] == student_id]

if student_rows.empty:
    st.error(f"Student ID **{student_id}** isn't in our records. Check the ID and try again.")
    st.stop()

# If multiple enrollments exist, let student pick module
mod_col = "code_module_original" if "code_module_original" in student_rows.columns else "code_module"
if len(student_rows) > 1:
    mod_list = student_rows[mod_col].tolist()
    sel_mod = st.selectbox("Select Course Module Enrollment", mod_list)
    student_row = student_rows[student_rows[mod_col] == sel_mod].iloc[0]
else:
    student_row = student_rows.iloc[0]

# Load model artifacts to calculate risk score
model_path = get_models_dir() / "xgb_model.joblib"
scaler_path = get_models_dir() / "scaler.joblib"
feature_cols_path = get_models_dir() / "feature_cols.joblib"

risk_score = 0.5
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

# --- Page Header ---
module_name = student_row.get("code_module_original", student_row.get("code_module", "N/A"))
semester_name = student_row.get("code_presentation_original", student_row.get("code_presentation", "Current Semester"))

st.markdown(
    block(f"""
    <div class="pt-hello">
        <h1>Welcome back, Student #{html.escape(str(student_id))}</h1>
        <div class="pt-meta">Course <b>{html.escape(str(module_name))}</b> &nbsp;·&nbsp; Semester <b>{html.escape(str(semester_name))}</b></div>
    </div>
    """),
    unsafe_allow_html=True,
)

# --- Section 1: Academic Status & Progress Score ---
if risk_score < 0.40:
    tone, badge = "var(--sage)", "On track"
    headline = "You're doing great."
    body = ("You're studying regularly, handing work in on time, and your grades show it. "
            "Keep this rhythm going through your final assignments.")
elif risk_score < 0.70:
    tone, badge = "var(--amber)", "Needs attention"
    headline = "A small push will help."
    body = ("Your recent activity or quiz scores are a little below your class average. "
            "A bit of extra study time and regular logins this week will move things in the right direction.")
else:
    tone, badge = "var(--coral)", "Support available"
    headline = "Let's get you extra support."
    body = ("Some assignments are pending, or you haven't logged in recently. That's fixable. "
            "Free 1-on-1 tutoring, assignment extensions, and advisor time are open to you.")

health_score = int(round((1.0 - risk_score) * 100))
gauge_hex = "#7FB59A" if health_score >= 60 else ("#E0B25A" if health_score >= 35 else "#DB8B78")

col_status, col_gauge = st.columns([3, 2], gap="large")

with col_status:
    st.markdown(
        block(f"""
        <div class="pt-status">
            <span class="pt-pill" style="color:{tone}"><i></i>{badge}</span>
            <h2>{headline}</h2>
            <p>{body}</p>
        </div>
        """),
        unsafe_allow_html=True,
    )

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={"text": "Overall progress", "font": {"size": 16, "color": "#9AA7BD", "family": "Instrument Sans, sans-serif"}},
        number={"font": {"size": 58, "color": "#F1EEE7", "family": "Newsreader, Georgia, serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)", "tickfont": {"color": "#64738C", "size": 11}},
            "bar": {"color": gauge_hex, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(219, 139, 120, 0.14)"},
                {"range": [35, 60], "color": "rgba(224, 178, 90, 0.14)"},
                {"range": [60, 100], "color": "rgba(127, 181, 154, 0.14)"},
            ],
        },
    ))
    fig_gauge.update_layout(
        height=270,
        margin=dict(l=24, r=24, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#F1EEE7"},
    )
    st.plotly_chart(fig_gauge, width="stretch")

# --- Section 2: Student Progress vs. Class Averages ---
class_days_med = num(features["days_active"].median(), 45) if "days_active" in features.columns else 45
class_score_med = num(features["avg_score"].median(), 75.0) if "avg_score" in features.columns else 75.0
class_clicks_med = num(features["total_clicks"].median(), 500) if "total_clicks" in features.columns else 500

my_days = num(student_row.get("days_active", 0))
my_score = num(student_row.get("avg_score", 0.0))
my_clicks = num(student_row.get("total_clicks", 0))
my_missed = num(student_row.get("submissions_missed", 0))

st.markdown(
    block("""
    <div class="pt-section">
        <h2>How you compare</h2>
        <p>Your numbers next to the class average. The white line marks the average.</p>
    </div>
    """),
    unsafe_allow_html=True,
)


def compare_row(label, mine, avg, mine_txt, unit, foot_diff_txt, scale_cap=None):
    scale = max(mine, avg * 1.6, 1.0)
    if scale_cap:
        scale = scale_cap
    fill_w = min(max(mine / scale * 100, 0), 100)
    avg_x = min(max(avg / scale * 100, 0), 100)
    ahead = mine >= avg
    color = "var(--sage)" if ahead else "var(--amber)"
    cls = "good" if ahead else "warn"
    return block(f"""
    <div class="cmp">
        <div class="cmp-head">
            <span class="cmp-label">{label}</span>
            <span class="cmp-val">{mine_txt}<small>{unit}</small></span>
        </div>
        <div class="track">
            <div class="fill" style="width:{fill_w:.1f}%; background:{color};"></div>
            <div class="avg" style="left:{avg_x:.1f}%;"></div>
        </div>
        <div class="cmp-foot"><b class="{cls}">{foot_diff_txt}</b></div>
    </div>
    """)


d_days = my_days - class_days_med
d_score = my_score - class_score_med
d_clicks = my_clicks - class_clicks_med

rows_html = ""
rows_html += compare_row(
    "Days logged in", my_days, class_days_med, f"{int(my_days)}", "days",
    f"{abs(int(d_days))} days {'ahead of' if d_days >= 0 else 'behind'} the class average of {int(class_days_med)}",
)
rows_html += compare_row(
    "Average assignment grade", my_score, class_score_med, f"{my_score:.1f}", "%",
    f"{abs(d_score):.1f} points {'above' if d_score >= 0 else 'below'} the class average of {class_score_med:.1f}%",
    scale_cap=100,
)
rows_html += compare_row(
    "Study activity", my_clicks, class_clicks_med, f"{int(my_clicks):,}", "clicks",
    f"{abs(int(d_clicks)):,} clicks {'more than' if d_clicks >= 0 else 'fewer than'} the class average of {int(class_clicks_med):,}",
)

if my_missed > 0:
    pend_color = "var(--amber)"
    pend_title = f"{int(my_missed)} assignment{'s' if int(my_missed) != 1 else ''} still waiting"
    pend_sub = "Submit them, or ask your teacher about handing them in late."
else:
    pend_color = "var(--sage)"
    pend_title = "You're all caught up"
    pend_sub = "No assignments are pending."

rows_html += block(f"""
<div class="pending">
    <div class="num" style="color:{pend_color}">{int(my_missed)}</div>
    <div class="txt"><strong>{pend_title}</strong><span>{pend_sub}</span></div>
</div>
""")

st.markdown(rows_html, unsafe_allow_html=True)

# --- Section 3: Personalized Study Tips & Next Steps ---
st.markdown(
    block("""
    <div class="pt-section">
        <h2>What to do next</h2>
        <p>Tips picked from your own numbers.</p>
    </div>
    """),
    unsafe_allow_html=True,
)

col_plan, col_msg = st.columns([3, 2], gap="large")

with col_plan:
    recommendations = []

    if my_missed > 0:
        recommendations.append({
            "title": f"Finish your {int(my_missed)} pending assignment{'s' if int(my_missed) != 1 else ''}",
            "tip": "Submit any overdue quizzes, or ask your teacher about turning in late work."
        })

    if my_days < class_days_med:
        recommendations.append({
            "title": "Log in more often",
            "tip": f"You've logged in {int(my_days)} days and the class average is {int(class_days_med)}. Aim for 3 to 4 days a week to go over the study notes."
        })

    if my_score < 70.0:
        recommendations.append({
            "title": "Join a free peer tutoring session",
            "tip": f"Study groups for {html.escape(str(module_name))} meet twice a week, and students who join tend to score better on exams."
        })

    if not recommendations:
        recommendations.append({
            "title": "Keep doing what you're doing",
            "tip": "You're ahead of the class on attendance, assignments, and grades. Hold on to that momentum."
        })

    tips_html = "".join(
        block(f"""
        <div class="tip">
            <span class="mark"></span>
            <div><h4>{item['title']}</h4><p>{item['tip']}</p></div>
        </div>
        """)
        for item in recommendations
    )
    st.markdown(tips_html, unsafe_allow_html=True)

with col_msg:
    st.markdown("#### Notes from your teacher")

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
                        timestamp_str = html.escape(str(latest.get("timestamp", ""))[:10])
                        st.markdown(
                            block(f"""
                            <div class="note">
                                <p class="quote">“{note_text}”</p>
                                <div class="by">Your teacher · {timestamp_str}</div>
                            </div>
                            """),
                            unsafe_allow_html=True,
                        )
                        found_note = True
        except Exception:
            pass

    if not found_note:
        st.markdown(
            block("""
            <div class="note calm">
                <p class="quote">Nothing flagged. Your teacher hasn't raised any concerns, so keep following your study plan.</p>
                <div class="by">Questions? Reach out any time.</div>
            </div>
            """),
            unsafe_allow_html=True,
        )