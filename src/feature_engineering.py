"""
feature_engineering.py — Create ML-ready features from the student master table.

Generates engagement, assessment, and demographic features.
Creates binary target: at_risk = 1 if Withdrawn/Fail, else 0.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import get_processed_data_dir


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from the student master table.

    Args:
        df: Cleaned student_master DataFrame from preprocessing.

    Returns:
        DataFrame with engineered features and binary target.
    """
    print("\n⚙️  Engineering features...")
    feat = df.copy()

    # --- Binary target ---
    feat["at_risk"] = feat["final_result"].apply(
        lambda x: 1 if x in ["Withdrawn", "Fail"] else 0
    )
    print(f"  Target created: at_risk")
    print(f"    At risk (1): {feat['at_risk'].sum()} ({feat['at_risk'].mean()*100:.1f}%)")
    print(f"    Not at risk (0): {(1 - feat['at_risk']).sum()} ({(1 - feat['at_risk']).mean()*100:.1f}%)")

    # --- Engagement features ---
    # These were already aggregated in preprocessing; ensure they exist
    engagement_cols = ["total_clicks", "avg_clicks_per_day", "click_trend", "days_active", "unique_activities"]
    for col in engagement_cols:
        if col not in feat.columns:
            feat[col] = 0
            print(f"  ⚠ Created missing engagement col: {col} (filled with 0)")

    # Derived: click intensity (clicks per unique activity)
    feat["click_intensity"] = feat["total_clicks"] / feat["unique_activities"].replace(0, 1)

    # Derived: engagement consistency (days active / total possible days)
    # Approximate: use 270 days as a typical module length
    feat["engagement_consistency"] = feat["days_active"] / 270.0

    print(f"  Engagement features: {len(engagement_cols) + 2} cols")

    # --- Assessment features ---
    assessment_cols = ["avg_score", "score_std", "num_assessments", "submissions_missed", "max_score", "min_score"]
    for col in assessment_cols:
        if col not in feat.columns:
            feat[col] = 0
            print(f"  ⚠ Created missing assessment col: {col} (filled with 0)")

    # Derived: score range
    feat["score_range"] = feat["max_score"] - feat["min_score"]

    # Derived: score-to-clicks ratio (effort efficiency)
    feat["score_per_click"] = feat["avg_score"] / feat["total_clicks"].replace(0, 1)

    print(f"  Assessment features: {len(assessment_cols) + 2} cols")

    # --- Demographic features (already encoded in preprocessing) ---
    demo_cols = ["gender", "region", "highest_education", "imd_band", "age_band",
                 "disability", "num_of_prev_attempts", "studied_credits"]
    demo_present = [c for c in demo_cols if c in feat.columns]
    print(f"  Demographic features: {len(demo_present)} cols")

    # --- Registration features ---
    if "date_registration" in feat.columns:
        feat["registered_early"] = (feat["date_registration"] < 0).astype(int)
        feat["registration_gap"] = feat["date_registration"].fillna(0)
        print(f"  Registration features: 2 cols")

    # --- Select final feature columns ---
    feature_cols = (
        engagement_cols +
        ["click_intensity", "engagement_consistency"] +
        assessment_cols +
        ["score_range", "score_per_click"] +
        demo_present
    )

    # Add registration features if present
    for reg_col in ["registered_early", "registration_gap"]:
        if reg_col in feat.columns:
            feature_cols.append(reg_col)

    # Keep identifiers, originals, target
    id_cols = ["id_student", "code_module", "code_presentation"]
    original_cols = [c for c in feat.columns if c.endswith("_original")]
    keep_cols = id_cols + feature_cols + ["at_risk", "final_result"] + original_cols

    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in feat.columns]
    result = feat[keep_cols].copy()

    # Fill any remaining NaN
    for col in feature_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    print(f"\n  ✓ Final feature matrix: {result.shape[0]} rows × {len(feature_cols)} features + target")

    return result, feature_cols


def run_feature_engineering(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Full feature engineering pipeline.

    Args:
        df: Cleaned student_master DataFrame.
        save: Whether to save features.csv.

    Returns:
        Tuple of (feature DataFrame, list of feature column names)
    """
    features_df, feature_cols = create_features(df)

    if save:
        out_path = get_processed_data_dir() / "features.csv"
        features_df.to_csv(out_path, index=False)
        print(f"\n💾 Saved: {out_path}")

    return features_df, feature_cols


if __name__ == "__main__":
    # Test: load student_master and engineer features
    proc_dir = get_processed_data_dir()
    df = pd.read_csv(proc_dir / "student_master.csv")
    features_df, feature_cols = run_feature_engineering(df)
    print("\nFeature columns:", feature_cols)
