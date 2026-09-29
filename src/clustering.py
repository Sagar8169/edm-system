"""
clustering.py — KMeans personalization clustering.

Clusters students by engagement + performance features into 3-4 groups,
maps each cluster to a suggested intervention.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import get_project_root, get_processed_data_dir


def get_models_dir() -> Path:
    return get_project_root() / "models"


def get_reports_dir() -> Path:
    reports_dir = get_project_root() / "reports" / "clustering"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


# Predefined intervention mapping
INTERVENTION_MAP = {
    0: {
        "profile": "High Engagement, High Performance",
        "description": "Consistently active learners achieving strong scores across assessments.",
        "intervention": "Continue current learning path. Offer advanced or enrichment resources. "
                        "Consider peer mentoring roles to leverage their success.",
        "risk_level": "Low",
        "color": "#2ecc71",
    },
    1: {
        "profile": "Low Engagement, Declining Performance",
        "description": "Minimal VLE interaction with below-average and worsening assessment scores.",
        "intervention": "URGENT: Schedule immediate 1-on-1 tutorial support. Send personalized "
                        "outreach. Provide structured study plans with weekly milestones. "
                        "Consider accessibility barriers.",
        "risk_level": "Critical",
        "color": "#e74c3c",
    },
    2: {
        "profile": "Moderate Engagement, Inconsistent Scores",
        "description": "Regular but not intensive engagement with high variance in assessment performance.",
        "intervention": "Enroll in targeted study skills workshop. Set up peer mentoring pairing. "
                        "Provide formative assessment practice resources. Monitor progress bi-weekly.",
        "risk_level": "Medium",
        "color": "#f39c12",
    },
    3: {
        "profile": "High Engagement, Low Performance",
        "description": "Very active in VLE but struggling to translate effort into assessment scores.",
        "intervention": "Review assessment preparation strategies — possible content comprehension gap. "
                        "Offer alternative learning materials. Check for course-content mismatch. "
                        "Consider learning support services referral.",
        "risk_level": "High",
        "color": "#e67e22",
    },
}


def find_optimal_k(X_scaled: np.ndarray, max_k: int = 8) -> int:
    """
    Use elbow method to find optimal number of clusters.
    Returns the recommended k (defaults to 4 if elbow is ambiguous).
    """
    inertias = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Plot elbow curve
    reports_dir = get_reports_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_range), inertias, "bo-", linewidth=2)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-cluster sum of squares)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, alpha=0.3)
    plt.savefig(reports_dir / "elbow_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Simple elbow detection: find where rate of decrease slows most
    diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    if len(diffs) >= 2:
        ratios = [diffs[i+1] / max(diffs[i], 1e-10) for i in range(len(diffs)-1)]
        best_idx = np.argmin(ratios) + 2  # +2 because K_range starts at 2
    else:
        best_idx = 4

    best_k = min(max(best_idx, 3), 5)  # Clamp between 3 and 5
    print(f"  📊 Elbow analysis suggests k={best_k}")
    return best_k


def assign_cluster_labels(kmeans_model, X_scaled, cluster_features):
    """
    Reorder cluster labels so they match the predefined intervention map.
    Cluster 0 = highest engagement+score, Cluster 1 = lowest, etc.
    """
    centers = kmeans_model.cluster_centers_
    n_clusters = len(centers)

    # Score each center by a composite of engagement and performance
    # Assuming first features are engagement-related, then performance
    # Use mean of all features as a simple composite
    composite_scores = centers.mean(axis=1)

    # Sort: highest composite = cluster 0, lowest = cluster 1
    sorted_indices = np.argsort(-composite_scores)

    # Create mapping from old label to new label
    label_map = {}
    for new_label, old_label in enumerate(sorted_indices):
        label_map[old_label] = min(new_label, 3)  # Cap at 3 (our intervention map has 4 entries)

    return label_map


def run_clustering(features_df: pd.DataFrame, feature_cols: list[str],
                   n_clusters: int = 4, save: bool = True) -> dict:
    """
    Full clustering pipeline: select features → scale → KMeans → assign interventions.

    Args:
        features_df: Feature DataFrame with student data.
        feature_cols: All feature column names.
        n_clusters: Number of clusters (default 4).
        save: Whether to save results.

    Returns:
        Dictionary with cluster assignments, model, and visualization data.
    """
    print("\n🎯 Running personalization clustering...")

    # Select clustering features (engagement + performance subset)
    cluster_feature_names = []
    preferred = ["total_clicks", "avg_clicks_per_day", "click_trend", "days_active",
                 "avg_score", "score_std", "submissions_missed", "engagement_consistency"]

    for f in preferred:
        if f in feature_cols:
            cluster_feature_names.append(f)

    if len(cluster_feature_names) < 3:
        # Fallback: use first 8 numeric features
        cluster_feature_names = feature_cols[:8]

    print(f"  Clustering features ({len(cluster_feature_names)}): {cluster_feature_names}")

    X_cluster = features_df[cluster_feature_names].values

    # Scale
    cluster_scaler = StandardScaler()
    X_scaled = cluster_scaler.fit_transform(X_cluster)

    # Find optimal k (or use provided)
    optimal_k = find_optimal_k(X_scaled)
    k = n_clusters  # Force requested number of clusters (4)

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)

    # Assign labels and reorder
    label_map = assign_cluster_labels(kmeans, X_scaled, cluster_feature_names)
    raw_labels = kmeans.labels_
    mapped_labels = np.array([label_map.get(l, l) for l in raw_labels])

    features_df = features_df.copy()
    features_df["cluster"] = mapped_labels

    # Map interventions
    features_df["cluster_profile"] = features_df["cluster"].map(
        lambda c: INTERVENTION_MAP.get(c, INTERVENTION_MAP[3])["profile"]
    )
    features_df["intervention"] = features_df["cluster"].map(
        lambda c: INTERVENTION_MAP.get(c, INTERVENTION_MAP[3])["intervention"]
    )
    features_df["risk_level"] = features_df["cluster"].map(
        lambda c: INTERVENTION_MAP.get(c, INTERVENTION_MAP[3])["risk_level"]
    )

    # Print cluster distribution
    print(f"\n  Cluster distribution (k={k}):")
    for c in sorted(features_df["cluster"].unique()):
        count = (features_df["cluster"] == c).sum()
        info = INTERVENTION_MAP.get(c, {"profile": f"Cluster {c}", "risk_level": "?"})
        print(f"    Cluster {c}: {count:>6} students | {info['profile']} | Risk: {info['risk_level']}")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    features_df["pca_1"] = X_pca[:, 0]
    features_df["pca_2"] = X_pca[:, 1]

    # Plot clusters
    reports_dir = get_reports_dir()
    plt.figure(figsize=(10, 8))
    colors = [INTERVENTION_MAP.get(c, {"color": "#999"})["color"] for c in features_df["cluster"]]
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.5, s=15)

    # Add cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    for i, (cx, cy) in enumerate(centers_pca):
        mapped_i = label_map.get(i, i)
        info = INTERVENTION_MAP.get(mapped_i, {"profile": f"C{mapped_i}"})
        plt.scatter(cx, cy, marker="x", s=200, linewidths=3, color="black", zorder=5)
        plt.annotate(info["profile"], (cx, cy), fontsize=8, fontweight="bold",
                     xytext=(10, 10), textcoords="offset points",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("Student Clusters — Engagement & Performance", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / "cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Saved cluster scatter: {reports_dir / 'cluster_scatter.png'}")

    # Save
    if save:
        models_dir = get_models_dir()
        joblib.dump(kmeans, models_dir / "kmeans_model.joblib")
        joblib.dump(cluster_scaler, models_dir / "cluster_scaler.joblib")
        joblib.dump(cluster_feature_names, models_dir / "cluster_features.joblib")
        joblib.dump(pca, models_dir / "pca_model.joblib")

        cluster_output = features_df[["id_student", "code_module", "code_presentation",
                                       "cluster", "cluster_profile", "intervention", "risk_level",
                                       "pca_1", "pca_2"]].copy()
        cluster_output.to_csv(get_processed_data_dir() / "student_clusters.csv", index=False)
        print(f"  💾 Saved cluster assignments")

    return {
        "features_df": features_df,
        "kmeans": kmeans,
        "cluster_scaler": cluster_scaler,
        "pca": pca,
        "cluster_feature_names": cluster_feature_names,
    }
