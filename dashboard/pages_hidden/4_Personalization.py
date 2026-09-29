"""
Page 4: Personalization — Cluster visualization and intervention recommendations.
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

from src.data_loader import get_project_root, get_processed_data_dir
from src.clustering import INTERVENTION_MAP


st.set_page_config(page_title="Personalization — EDM System", page_icon="🎯", layout="wide")

# --- Load Custom CSS ---
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("# 🎯 Personalization — Cluster-Based Interventions")
st.markdown("Students are grouped by engagement and performance patterns. Each cluster maps to a tailored intervention.")
st.markdown("---")

# Load data
features_path = get_processed_data_dir() / "features.csv"
clusters_path = get_processed_data_dir() / "student_clusters.csv"

if not features_path.exists() or not clusters_path.exists():
    st.error("⚠️ Data not found. Run `python run_pipeline.py` first.")
    st.stop()

features = pd.read_csv(features_path)
clusters = pd.read_csv(clusters_path)

# Merge
merge_cols = ["id_student", "code_module", "code_presentation"]
df = features.merge(clusters, on=merge_cols, how="left")

# --- Cluster Distribution ---
st.markdown("### 📊 Cluster Distribution")

cluster_counts = df["cluster"].value_counts().sort_index()

col1, col2 = st.columns([1, 1])

with col1:
    # Summary cards
    for cluster_id in sorted(df["cluster"].dropna().unique().astype(int)):
        info = INTERVENTION_MAP.get(cluster_id, INTERVENTION_MAP[3])
        count = int((df["cluster"] == cluster_id).sum())
        pct = (count / len(df)) * 100

        st.markdown(f"""
        <div style="border-left: 4px solid {info['color']}; padding: 10px 15px; margin: 10px 0;
                    background-color: {info['color']}10; border-radius: 0 8px 8px 0;">
            <h4 style="margin: 0; color: {info['color']};">Cluster {cluster_id}: {info['profile']}</h4>
            <p style="margin: 5px 0; font-size: 14px;"><strong>{count:,}</strong> students ({pct:.1f}%) | Risk: <strong>{info['risk_level']}</strong></p>
            <p style="margin: 5px 0; font-size: 13px; color: #666;">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Pie chart
    cluster_names = [INTERVENTION_MAP.get(int(c), {"profile": f"C{c}"})["profile"]
                     for c in cluster_counts.index]
    cluster_colors = [INTERVENTION_MAP.get(int(c), {"color": "#999"})["color"]
                      for c in cluster_counts.index]

    fig_pie = px.pie(
        values=cluster_counts.values,
        names=cluster_names,
        color_discrete_sequence=cluster_colors,
        title="Student Distribution by Cluster",
        hole=0.35,
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# --- PCA Scatter Plot ---
st.markdown("### 🗺️ Cluster Visualization (PCA)")

if "pca_1" in df.columns and "pca_2" in df.columns:
    fig_scatter = px.scatter(
        df.dropna(subset=["pca_1", "pca_2", "cluster"]),
        x="pca_1", y="pca_2",
        color="cluster",
        color_discrete_map={
            int(k): v["color"] for k, v in INTERVENTION_MAP.items()
        },
        hover_data=["id_student", "cluster_profile", "risk_level"],
        title="Student Clusters in PCA Space",
        labels={"pca_1": "PCA Component 1", "pca_2": "PCA Component 2"},
        opacity=0.6,
    )
    fig_scatter.update_layout(
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title="Cluster",
    )
    fig_scatter.update_traces(marker_size=5)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# --- Per-Cluster Metrics ---
st.markdown("### 📈 Cluster Metrics Comparison")

metric_cols = ["total_clicks", "avg_clicks_per_day", "avg_score", "days_active",
               "submissions_missed", "score_std"]
available_metrics = [c for c in metric_cols if c in df.columns]

if available_metrics:
    cluster_stats = df.groupby("cluster")[available_metrics].mean()

    # Rename index for readability
    cluster_stats.index = [
        f"C{int(c)}: {INTERVENTION_MAP.get(int(c), {'profile': '?'})['profile'][:30]}"
        for c in cluster_stats.index
    ]

    st.dataframe(
        cluster_stats.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=0),
        use_container_width=True
    )

    # Radar chart comparing clusters
    if len(available_metrics) >= 3:
        fig_radar = go.Figure()
        for cluster_id in sorted(df["cluster"].dropna().unique().astype(int)):
            info = INTERVENTION_MAP.get(cluster_id, {"profile": f"C{cluster_id}", "color": "#999"})
            vals = df[df["cluster"] == cluster_id][available_metrics].mean().values
            # Normalize to 0-1 range
            mins = df[available_metrics].min().values
            maxs = df[available_metrics].max().values
            ranges = maxs - mins
            ranges[ranges == 0] = 1
            norm_vals = (vals - mins) / ranges

            fig_radar.add_trace(go.Scatterpolar(
                r=list(norm_vals) + [norm_vals[0]],
                theta=available_metrics + [available_metrics[0]],
                fill="toself",
                name=f"C{cluster_id}: {info['profile'][:25]}",
                line_color=info["color"],
                opacity=0.6,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Cluster Profiles — Normalized Metrics",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# --- Intervention Details ---
st.markdown("---")
st.markdown("### 💡 Intervention Recommendations")

for cluster_id in sorted(INTERVENTION_MAP.keys()):
    info = INTERVENTION_MAP[cluster_id]
    count = int((df["cluster"] == cluster_id).sum()) if "cluster" in df.columns else 0

    with st.expander(f"Cluster {cluster_id}: {info['profile']} ({count} students)"):
        st.markdown(f"**Risk Level:** {info['risk_level']}")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Recommended Intervention:** {info['intervention']}")
