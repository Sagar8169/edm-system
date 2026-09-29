import sys
import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.fairness import run_fairness_analysis
from src.report import generate_report

features_df = pd.read_csv("data/processed/features.csv")
splits = joblib.load("models/data_splits.joblib")

X_test = splits["X_test"]
y_test = splits["y_test"]

lr = joblib.load("models/lr_baseline.joblib")
xgb = joblib.load("models/xgb_model.joblib")

y_pred_baseline = lr.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

_, test_indices = train_test_split(
    np.arange(len(features_df)),
    test_size=0.2,
    random_state=42,
    stratify=features_df["at_risk"].values
)

fairness_results = run_fairness_analysis(
    features_df, y_test, y_pred_xgb, test_indices
)

with open("reports/model_metrics.json", encoding="utf-8") as f:
    model_metrics = json.load(f)

generate_report(
    model_metrics=model_metrics,
    afi=fairness_results["afi"],
    y_true=y_test,
    y_pred_baseline=y_pred_baseline,
    y_pred_xgb=y_pred_xgb,
)
print("Report generated successfully!")
