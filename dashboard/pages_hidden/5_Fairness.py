"""
Page 5: Fairness — Algorithmic fairness analysis across demographic groups.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_project_root


def get_reports_dir():
    return get_project_root() / "reports" / "fairness"


st.set_page_config(page_title="Fairness — EDM System", page_icon="⚖️", layout="wide")

# --- Load Custom CSS ---
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# ⚖️ Algorithmic Fairness Analysis")
st.markdown("Monitoring prediction equity across demographic groups to ensure unbiased at-risk identification.")
st.markdown("---")

reports_dir = get_reports_dir()

# Load AFI
afi_path = reports_dir / "afi.txt"
fairness_report_path = reports_dir / "fairness_report.csv"

if not fairness_report_path.exists():
    st.error("⚠️ Fairness report not found. Run `python run_pipeline.py` first.")
    st.stop()

# Read AFI
afi = 0.0
if afi_path.exists():
    with open(afi_path) as f:
        first_line = f.readline()
        try:
            afi = float(first_line.split(":")[1].strip())
        except (ValueError, IndexError):
            afi = 0.0

# --- AFI Gauge ---
st.markdown("### 🎯 Algorithmic Fairness Index (AFI)")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Create a gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=afi,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Algorithmic Fairness Index", "font": {"size": 20}},
        delta={"reference": 0.8, "increasing": {"color": "#2ecc71"}, "decreasing": {"color": "#e74c3c"}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 2},
            "bar": {"color": "#3498db"},
            "steps": [
                {"range": [0, 0.5], "color": "rgba(231, 76, 60, 0.3)"},
                {"range": [0.5, 0.7], "color": "rgba(243, 156, 18, 0.3)"},
                {"range": [0.7, 0.85], "color": "rgba(241, 196, 15, 0.3)"},
                {"range": [0.85, 1.0], "color": "rgba(46, 204, 113, 0.3)"},
            ],
            "threshold": {
                "line": {"color": "#e74c3c", "width": 3},
                "thickness": 0.8,
                "value": 0.8,
            },
        },
    ))
    fig_gauge.update_layout(height=350)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Interpretation
    if afi >= 0.85:
        st.success(f"✅ **Excellent fairness** (AFI = {afi:.3f}). The model shows minimal disparity across demographic groups.")
    elif afi >= 0.7:
        st.warning(f"⚠️ **Acceptable fairness** (AFI = {afi:.3f}). Some disparity exists but within reasonable bounds.")
    elif afi >= 0.5:
        st.warning(f"⚠️ **Moderate disparity** (AFI = {afi:.3f}). Consider investigating bias sources and applying mitigation.")
    else:
        st.error(f"🚨 **Significant disparity** (AFI = {afi:.3f}). Urgent review needed for potential algorithmic bias.")

st.markdown("---")

# --- Per-Group Metrics ---
st.markdown("### 📊 Metrics by Demographic Group")

fairness_df = pd.read_csv(fairness_report_path)

if "attribute" in fairness_df.columns:
    attributes = fairness_df["attribute"].unique()

    tabs = st.tabs([attr.replace("_original", "").replace("_", " ").title() for attr in attributes])

    for tab, attr in zip(tabs, attributes):
        with tab:
            attr_df = fairness_df[fairness_df["attribute"] == attr].copy()
            attr_name = attr.replace("_original", "").replace("_", " ").title()

            # Metrics table
            st.markdown(f"#### {attr_name} — Per-Group Performance")
            display_cols = [c for c in ["group", "accuracy", "precision", "recall", "f1"] if c in attr_df.columns]
            st.dataframe(
                attr_df[display_cols].style.format(
                    {c: "{:.4f}" for c in display_cols if c != "group"}
                ).background_gradient(cmap="RdYlGn", axis=0, subset=[c for c in display_cols if c != "group"]),
                use_container_width=True
            )

            # Bar chart
            if "group" in attr_df.columns:
                metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
                available_metrics = [m for m in metrics_to_plot if m in attr_df.columns]

                melted = attr_df.melt(
                    id_vars=["group"],
                    value_vars=available_metrics,
                    var_name="Metric",
                    value_name="Score"
                )

                fig = px.bar(
                    melted, x="group", y="Score", color="Metric",
                    barmode="group",
                    title=f"Model Performance by {attr_name}",
                    color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"],
                )
                fig.update_layout(
                    yaxis_range=[0, 1.05],
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=attr_name,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Compute and display gaps
                st.markdown(f"#### {attr_name} — Fairness Gaps")
                gaps = {}
                for metric in available_metrics:
                    vals = attr_df[metric].values
                    gap = vals.max() - vals.min()
                    gaps[metric] = gap

                gap_df = pd.DataFrame([gaps])
                gap_df.index = ["Gap (max - min)"]

                st.dataframe(
                    gap_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn_r", axis=1),
                    use_container_width=True
                )

st.markdown("---")

# --- Fairness Gaps Heatmap ---
st.markdown("### 🗺️ Fairness Gap Summary")

fairness_img_path = reports_dir / "fairness_gaps_heatmap.png"
if fairness_img_path.exists():
    st.image(str(fairness_img_path), caption="Fairness Gaps Heatmap (lower = fairer)", use_container_width=True)

# --- Methodology ---
st.markdown("---")
st.markdown("### 📖 Methodology")
st.markdown("""
**Algorithmic Fairness Index (AFI)** is computed as:

```
AFI = 1 - max(recall_gap across all sensitive attributes)
```

Where `recall_gap = max(group_recall) - min(group_recall)` for each sensitive attribute.

**Sensitive attributes analyzed:**
- **Gender**: Male vs. Female
- **Disability**: Yes vs. No
- **Age Band**: 0-35 vs. 35-55 vs. 55+

**Interpretation:**
- AFI ≥ 0.85: Excellent fairness
- AFI 0.70-0.85: Acceptable, minor disparities
- AFI 0.50-0.70: Moderate concern, investigate bias
- AFI < 0.50: Significant bias, requires mitigation

**Framework:** [Fairlearn](https://fairlearn.org/) MetricFrame
""")
