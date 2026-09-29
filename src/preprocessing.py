"""
preprocessing.py — Clean, merge, and encode the OULAD dataset.

Merges studentInfo + studentRegistration + aggregated assessment scores
+ aggregated VLE engagement into a single student_master DataFrame.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.data_loader import get_processed_data_dir


def aggregate_assessments(student_assessment: pd.DataFrame, assessments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate assessment data per student-module-presentation.

    Returns DataFrame with columns:
        id_student, code_module, code_presentation,
        avg_score, score_std, num_assessments, submissions_missed, max_score, min_score
    """
    # Merge to get module/presentation info
    merged = student_assessment.merge(assessments, on="id_assessment", how="left")

    # Calculate per-student aggregates
    agg = merged.groupby(["id_student", "code_module", "code_presentation"]).agg(
        avg_score=("score", "mean"),
        score_std=("score", "std"),
        num_assessments=("id_assessment", "nunique"),
        max_score=("score", "max"),
        min_score=("score", "min"),
        total_weight=("weight", "sum"),
    ).reset_index()

    # Fill NaN std (when only 1 assessment)
    agg["score_std"] = agg["score_std"].fillna(0)

    # Calculate submissions missed: assessments available - assessments submitted
    total_assessments = assessments.groupby(["code_module", "code_presentation"]).size().reset_index(name="total_available")
    agg = agg.merge(total_assessments, on=["code_module", "code_presentation"], how="left")
    agg["submissions_missed"] = agg["total_available"] - agg["num_assessments"]
    agg["submissions_missed"] = agg["submissions_missed"].clip(lower=0)

    return agg


def aggregate_vle(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VLE engagement data per student-module-presentation.

    Returns DataFrame with columns:
        id_student, code_module, code_presentation,
        total_clicks, avg_clicks_per_day, days_active, unique_activities,
        click_trend (slope of daily clicks over time)
    """
    # Merge to get activity type info
    merged = student_vle.merge(
        vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left"
    )

    # Basic aggregations per student
    agg = merged.groupby(["id_student", "code_module", "code_presentation"]).agg(
        total_clicks=("sum_click", "sum"),
        days_active=("date", "nunique"),
        unique_activities=("activity_type", "nunique"),
    ).reset_index()

    # Average clicks per active day
    agg["avg_clicks_per_day"] = agg["total_clicks"] / agg["days_active"].replace(0, 1)

    # Calculate click trend (slope) per student
    # Group daily clicks and compute linear trend
    daily = merged.groupby(
        ["id_student", "code_module", "code_presentation", "date"]
    )["sum_click"].sum().reset_index()

    def compute_slope(group):
        """Compute linear regression slope of clicks over days."""
        if len(group) < 2:
            return 0.0
        x = group["date"].values.astype(float)
        y = group["sum_click"].values.astype(float)
        # Simple OLS slope: cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        if denominator == 0:
            return 0.0
        return numerator / denominator

    trends = daily.groupby(
        ["id_student", "code_module", "code_presentation"]
    ).apply(compute_slope, include_groups=False).reset_index(name="click_trend")

    agg = agg.merge(trends, on=["id_student", "code_module", "code_presentation"], how="left")
    agg["click_trend"] = agg["click_trend"].fillna(0)

    return agg


def create_student_master(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all OULAD tables into a single student-level master table.

    Args:
        data: Dictionary of DataFrames from load_oulad()

    Returns:
        student_master DataFrame with one row per student-module-presentation
    """
    print("\n🔧 Creating student master table...")

    student_info = data["studentInfo"].copy()
    student_reg = data["studentRegistration"].copy()
    student_assessment = data["studentAssessment"].copy()
    assessments_table = data["assessments"].copy()
    student_vle = data["studentVle"].copy()
    vle_table = data["vle"].copy()

    # 1. Start with studentInfo as the base
    print(f"  Base: studentInfo → {len(student_info)} rows")

    # 2. Merge registration info
    reg_cols = ["id_student", "code_module", "code_presentation", "date_registration", "date_unregistration"]
    master = student_info.merge(
        student_reg[reg_cols],
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )
    print(f"  + studentRegistration → {len(master)} rows")

    # 3. Aggregate and merge assessment scores
    print("  Aggregating assessment scores...")
    agg_assess = aggregate_assessments(student_assessment, assessments_table)
    master = master.merge(
        agg_assess,
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )
    print(f"  + assessments → {len(master)} rows")

    # 4. Aggregate and merge VLE engagement
    print("  Aggregating VLE engagement (this may take a moment)...")
    agg_vle = aggregate_vle(student_vle, vle_table)
    master = master.merge(
        agg_vle,
        on=["id_student", "code_module", "code_presentation"],
        how="left"
    )
    print(f"  + VLE engagement → {len(master)} rows")

    return master


def clean_and_encode(master: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean missing values and encode categorical variables.

    Args:
        master: Raw merged student master DataFrame.

    Returns:
        Cleaned and encoded DataFrame.
    """
    print("\n🧹 Cleaning and encoding...")
    df = master.copy()

    # Drop rows with missing target
    initial_len = len(df)
    df = df.dropna(subset=["final_result"])
    print(f"  Dropped {initial_len - len(df)} rows with missing final_result")

    # --- Handle missing values ---

    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {col}: {null_count} nulls → median ({median_val:.2f})")

    # Categorical columns: fill with mode
    cat_cols = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]
    for col in cat_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  Imputed {col}: {null_count} nulls → mode ('{mode_val}')")

    # --- Encode categoricals ---
    label_encoders = {}
    encode_cols = ["gender", "region", "highest_education", "imd_band", "age_band",
                   "disability", "code_module", "code_presentation"]

    for col in encode_cols:
        if col in df.columns:
            # Keep original as _original for fairness analysis
            df[f"{col}_original"] = df[col].copy()
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded {col}: {len(le.classes_)} categories")

    print(f"\n  ✓ Final shape: {df.shape}")
    return df, label_encoders


def run_preprocessing(data: dict[str, pd.DataFrame], save: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline: merge → clean → encode → save.

    Args:
        data: Dictionary of DataFrames from load_oulad()
        save: Whether to save the result to CSV.

    Returns:
        Tuple of (cleaned DataFrame, label_encoders dict)
    """
    # Merge
    master = create_student_master(data)

    # Clean and encode
    df, encoders = clean_and_encode(master)

    # Save
    if save:
        out_path = get_processed_data_dir() / "student_master.csv"
        df.to_csv(out_path, index=False)
        print(f"\n💾 Saved: {out_path} ({len(df)} rows × {len(df.columns)} cols)")

    return df, encoders


if __name__ == "__main__":
    from src.data_loader import load_oulad
    data = load_oulad()
    df, encoders = run_preprocessing(data)
    print("\nColumns:", list(df.columns))
