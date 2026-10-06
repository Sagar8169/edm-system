"""
Student_Development.py — EDM Performance Recognition System Authentication Gateway & Top Nav Bar.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_project_root, get_processed_data_dir


st.set_page_config(
    page_title="EDM Performance Recognition System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Load Custom CSS (existing project stylesheet) ---
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Premium theme layer (loaded after style.css so it takes precedence) ---
PREMIUM_CSS = """
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
    --brass-soft: rgba(201, 164, 92, 0.14);
    --sage: #7FB59A;
    --radius-lg: 18px;
    --radius-sm: 10px;
}

/* ---------- Base ---------- */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(900px 500px at 88% -10%, rgba(201, 164, 92, 0.10), transparent 60%),
        var(--ink) !important;
    color: var(--text);
    font-family: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}
h1, h2, h3 {
    font-family: 'Newsreader', Georgia, 'Times New Roman', serif !important;
    color: var(--text) !important;
    font-weight: 500 !important;
    letter-spacing: -0.015em;
}
p, label, span, li { font-family: 'Instrument Sans', system-ui, sans-serif; }

/* ---------- Header / top navigation ---------- */
[data-testid="stHeader"] {
    background: rgba(14, 23, 38, 0.72) !important;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--hairline);
}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { visibility: hidden; height: 0; }

/* ---------- Hero (login) ---------- */
.edm-hero { padding: 2rem 0 1rem 0; animation: edmRise 0.7s cubic-bezier(.2,.7,.2,1) both; }
.edm-hero h1 {
    font-size: clamp(2.1rem, 3.2vw, 3.0rem) !important;
    line-height: 1.15 !important;
    margin: 0 0 1.1rem 0 !important;
    max-width: 26ch !important;
}
.edm-hero .edm-lede {
    color: var(--text-muted);
    font-size: 1.05rem;
    line-height: 1.6;
    max-width: 48ch;
    margin: 0 0 2.0rem 0;
}
.edm-brand {
    display: flex; align-items: center; gap: 10px;
    font-weight: 600; font-size: 0.95rem; color: var(--text);
    margin-bottom: 2.0rem;
}
.edm-brand-mark {
    width: 30px; height: 30px; border-radius: 9px;
    background: linear-gradient(145deg, var(--brass), #8E7136);
    display: grid; place-items: center;
    color: #1B1406; font-family: 'Newsreader', serif; font-weight: 600; font-size: 1.05rem;
}
.edm-points { border-top: 1px solid var(--hairline); max-width: 480px; }
.edm-point { padding: 12px 0; border-bottom: 1px solid var(--hairline); }
.edm-point strong { display: block; color: var(--text); font-weight: 600; font-size: 0.95rem; }
.edm-point span { color: var(--text-muted); font-size: 0.88rem; line-height: 1.45; }

@keyframes edmRise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: none; }
}

/* ---------- Sign-in card (the form itself) ---------- */
[data-testid="stForm"] {
    background: linear-gradient(180deg, var(--surface-raised), var(--surface));
    border: 1px solid var(--hairline);
    border-radius: var(--radius-lg);
    padding: 1.8rem 1.7rem 1.5rem 1.7rem;
    box-shadow: 0 30px 60px -30px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

/* ---------- Role switch (segmented control) ---------- */
[data-testid="stRadio"] > div[role="radiogroup"] {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 999px;
    padding: 4px;
    gap: 6px !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: row !important;
}
[data-testid="stRadio"] label {
    flex: 1 1 0% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    white-space: nowrap !important;
    min-width: 100px !important;
    padding: 8px 16px !important;
    border-radius: 999px !important;
    cursor: pointer;
    transition: background .2s ease, color .2s ease;
    margin: 0 !important;
}
[data-testid="stRadio"] label > div:first-child { display: none !important; }
[data-testid="stRadio"] label:has(input:checked) {
    background: var(--brass-soft) !important;
    box-shadow: inset 0 0 0 1px rgba(201, 164, 92, 0.45) !important;
}
[data-testid="stRadio"] label:has(input:checked) p, [data-testid="stRadio"] label:has(input:checked) span {
    color: var(--brass) !important;
    font-weight: 600 !important;
}
[data-testid="stRadio"] label p, [data-testid="stRadio"] label span {
    color: var(--text-muted) !important;
    font-size: 0.93rem !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ---------- Inputs ---------- */
[data-testid="stTextInput"] label p { color: var(--text-muted); font-size: 0.86rem; font-weight: 500; }

/* Hide annoying "Press Enter to submit form" text overlapping eye icon */
[data-testid="stInputInstructions"], small[data-testid="stInputInstructions"] {
    display: none !important;
}

/* Style the input container wrapper */
[data-testid="stTextInput"] [data-baseweb="input"], [data-testid="stTextInputRootElement"] {
    background: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color .18s ease, box-shadow .18s ease;
}

[data-testid="stTextInput"] [data-baseweb="input"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within {
    border-color: var(--brass) !important;
    box-shadow: 0 0 0 3px var(--brass-soft) !important;
}

/* Inner raw input - remove duplicate borders */
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: var(--text) !important;
    padding: 0.65rem 0.85rem !important;
}

[data-testid="stTextInput"] input::placeholder { color: #64738C; }
[data-testid="stTextInput"] input:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stTextInput"] > div > div { background: transparent !important; border: none !important; }

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: var(--radius-sm);
    font-family: 'Instrument Sans', sans-serif;
    font-weight: 600;
    padding: 0.7rem 1.2rem;
    transition: transform .15s ease, box-shadow .2s ease, background .2s ease;
}
button[kind="primary"], button[kind="primaryFormSubmit"] {
    background: linear-gradient(180deg, #D5B36E, var(--brass)) !important;
    color: #1B1406 !important;
    border: none !important;
    box-shadow: 0 8px 22px -10px rgba(201, 164, 92, 0.65);
}
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 26px -10px rgba(201, 164, 92, 0.8);
}
button[kind="secondary"] {
    background: transparent !important;
    color: var(--text) !important;
    border: 1px solid var(--hairline) !important;
}
button[kind="secondary"]:hover { border-color: var(--brass) !important; color: var(--brass) !important; }
button:focus-visible { outline: 2px solid var(--brass) !important; outline-offset: 2px; }

/* ---------- Alerts ---------- */
[data-testid="stAlert"] { border-radius: var(--radius-sm); border: 1px solid var(--hairline); background: var(--surface); }

/* ---------- Demo credentials note ---------- */
.edm-demo {
    margin-top: 14px;
    padding: 12px 14px;
    border: 1px dashed var(--hairline);
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    font-size: 0.85rem;
    line-height: 1.6;
}
.edm-demo code {
    background: var(--brass-soft); color: var(--brass);
    padding: 1px 7px; border-radius: 6px; font-size: 0.82rem;
}

/* ---------- Signed-in status bar ---------- */
.edm-status { display: flex; align-items: center; gap: 12px; padding: 6px 0 14px 0; }
.edm-status .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--sage); box-shadow: 0 0 0 4px rgba(127, 181, 154, 0.15); }
.edm-status .who { color: var(--text); font-weight: 600; }
.edm-status .meta { color: var(--text-muted); font-size: 0.9rem; }

@media (prefers-reduced-motion: reduce) {
    .edm-hero { animation: none; }
    .stButton > button, [data-testid="stFormSubmitButton"] > button { transition: none; }
}
@media (max-width: 800px) {
    .edm-hero { padding-top: 1rem; }
    .edm-points { max-width: 100%; }
}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_features():
    path = get_processed_data_dir() / "features.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


features = load_features()

# --- Persistent Session State Syncing with Query Parameters ---
params = st.query_params

# Restore from query params if page refreshed
if params.get("user_role") == "teacher":
    st.session_state["logged_in"] = True
    st.session_state["user_role"] = "teacher"
    st.session_state["username"] = "admin11"
elif params.get("user_role") == "student" and params.get("student_id"):
    try:
        st.session_state["logged_in"] = True
        st.session_state["user_role"] = "student"
        st.session_state["student_id"] = int(params.get("student_id"))
    except ValueError:
        pass

logged_in = st.session_state.get("logged_in", False)
user_role = st.session_state.get("user_role", None)
student_id = st.session_state.get("student_id", None)

# Re-assert query params on every run to preserve login state across all page reloads & nav clicks
if logged_in:
    if user_role == "teacher":
        st.query_params["user_role"] = "teacher"
    elif user_role == "student" and student_id:
        st.query_params["user_role"] = "student"
        st.query_params["student_id"] = str(student_id)


def logout_user():
    st.session_state["logged_in"] = False
    st.session_state["user_role"] = None
    st.session_state["student_id"] = None
    st.query_params.clear()
    st.rerun()


def login_view():
    hero_col, form_col = st.columns([1.15, 1], gap="large")

    with hero_col:
        st.markdown(
            """
            <div class="edm-hero">
                <div class="edm-brand">
                    EDM Performance Recognition
                </div>
                <h1>Spot struggling students before the grades do.</h1>
                <p class="edm-lede">
                    Performance forecasts and early risk alerts, so teachers can step in sooner
                    and students can see where they stand.
                </p>
                <div class="edm-points">
                    <div class="edm-point"><strong>Forecast results</strong><span>See predicted outcomes for every student, updated with their activity.</span></div>
                    <div class="edm-point"><strong>Catch risk early</strong><span>Find students who are slipping while there is still time to help.</span></div>
                    <div class="edm-point"><strong>Plan interventions</strong><span>Review reports and decide who needs support first.</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        st.write("")
        st.write("")
        if not st.session_state.get("logged_in"):
            role_choice = st.radio(
                "Select Role",
                options=["Teacher", "Student"],
                horizontal=True,
                label_visibility="collapsed",
            )

            st.write("")

            if role_choice == "Teacher":
                st.markdown("### Teacher sign in")
                with st.form("teacher_form"):
                    user_id = st.text_input("User ID", placeholder="e.g. admin11")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    submit = st.form_submit_button("Sign in", type="primary", use_container_width=True)

                    if submit:
                        if user_id.strip() == "admin11" and password.strip() == "1122":
                            st.session_state["logged_in"] = True
                            st.session_state["user_role"] = "teacher"
                            st.session_state["username"] = "admin11"
                            st.query_params["user_role"] = "teacher"
                            st.rerun()
                        else:
                            st.error("User ID or password is incorrect. Check both and try again.")

                st.markdown(
                    """
                    <div class="edm-demo">
                        Demo access: User ID <code>admin11</code> · Password <code>1122</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.markdown("### Student sign in")
                with st.form("student_form"):
                    s_id_input = st.text_input("Student ID", placeholder="e.g. 28400")
                    s_password = st.text_input("Password", type="password", placeholder="Default password is 1234")
                    submit_student = st.form_submit_button("Sign in", type="primary", use_container_width=True)

                    if submit_student:
                        if not s_id_input.strip():
                            st.error("Enter your Student ID to continue.")
                        elif s_password.strip() != "1234":
                            st.error("Password is incorrect. The default password is 1234.")
                        else:
                            try:
                                s_id_num = int(s_id_input.strip())
                                if features is not None and s_id_num not in features["id_student"].values:
                                    st.error(f"Student ID {s_id_num} isn't in our records. Try 28400 or 30268.")
                                else:
                                    st.session_state["logged_in"] = True
                                    st.session_state["user_role"] = "student"
                                    st.session_state["student_id"] = s_id_num
                                    st.query_params["user_role"] = "student"
                                    st.query_params["student_id"] = str(s_id_num)
                                    st.rerun()
                            except ValueError:
                                st.error("Student ID must be a number, for example 28400.")

                st.markdown(
                    """
                    <div class="edm-demo">
                        Demo IDs: <code>28400</code> <code>30268</code> <code>11391</code> · Password <code>1234</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success(f"Signed in as {'Teacher (admin11)' if user_role == 'teacher' else f'Student #{student_id}'}")
            if st.button("Log out", type="primary", use_container_width=True):
                logout_user()


# --- Define Pages with explicit url_paths and defaults ---
login_page = st.Page(login_view, title="Login", url_path="login")
dashboard_page = st.Page("pages/1_Dashboard.py", title="Dashboard", url_path="Dashboard", default=True)
student_list_page = st.Page("pages/2_Student_List.py", title="Student List", url_path="Student_List")
student_detail_page = st.Page("pages/3_Student_Detail.py", title="Student Detail", url_path="Student_Detail")
report_page = st.Page("pages/4_Report.py", title="Report", url_path="Report")
student_portal_page = st.Page("pages/5_Student_Portal.py", title="Student Portal", url_path="Student_Portal", default=True)

# Dynamic Navigation
if not logged_in:
    nav = st.navigation([login_page], position="hidden")
elif user_role == "teacher":
    nav = st.navigation([dashboard_page, student_list_page, student_detail_page, report_page], position="hidden")
else:
    nav = st.navigation([student_portal_page], position="hidden")

# Global Top Header Bar with Navigation Tabs, User Account Info & Logout Button
if logged_in:
    header_container = st.container()
    with header_container:
        if user_role == "teacher":

            col_status, col_nav, col_logout = st.columns([1.8, 4.5, 1.2])
            with col_status:
                st.markdown(
                    '<div class="edm-status"><span class="dot"></span>'
                    '<span class="who">Teacher Access</span><span class="meta">admin11</span></div>',
                    unsafe_allow_html=True,
                )
            with col_nav:
                n1, n2, n3, n4 = st.columns(4)
                with n1:
                    st.page_link("pages/1_Dashboard.py", label="Dashboard")
                with n2:
                    st.page_link("pages/2_Student_List.py", label="Student List")
                with n3:
                    st.page_link("pages/3_Student_Detail.py", label="Student Detail")
                with n4:
                    st.page_link("pages/4_Report.py", label="Report")
            with col_logout:
                if st.button("Log out", key="global_top_logout", use_container_width=True):
                    logout_user()

        else:
            col_status, col_logout = st.columns([5, 1.2])
            with col_status:
                st.markdown(
                    f'<div class="edm-status"><span class="dot"></span>'
                    f'<span class="who">Student portal</span><span class="meta">Student #{student_id}</span></div>',
                    unsafe_allow_html=True,
                )
            with col_logout:
                if st.button("Log out", key="global_top_logout", use_container_width=True):
                    logout_user()

    st.markdown("<hr style='margin: 8px 0 20px 0; border: none; border-bottom: 1px solid var(--hairline);'>", unsafe_allow_html=True)

nav.run()