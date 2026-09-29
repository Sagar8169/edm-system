"""
verify_model.py — Verify that the trained EDM model is working correctly.

Loads the saved XGBoost model, runs predictions on the test set,
and compares against saved metrics to ensure everything is consistent.

Usage:
    python backend/verify_model.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import joblib
import json

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def verify():
    print("=" * 60)
    print("  EDM Model Verification")
    print("=" * 60)

    errors = []
    warnings = []

    # 1. Check model files exist
    print("\n1. Checking model files...")
    required_files = {
        "xgb_model.joblib": MODELS_DIR,
        "lr_baseline.joblib": MODELS_DIR,
        "scaler.joblib": MODELS_DIR,
        "feature_cols.joblib": MODELS_DIR,
        "data_splits.joblib": MODELS_DIR,
        "shap_values.npy": MODELS_DIR,
        "shap_explainer.joblib": MODELS_DIR,
        "kmeans_model.joblib": MODELS_DIR,
        "model_metrics.json": REPORTS_DIR,
        "hypothesis_test.json": REPORTS_DIR,
        "kpi_summary.csv": REPORTS_DIR,
        "shap_feature_importance.csv": REPORTS_DIR,
    }

    for fname, directory in required_files.items():
        path = directory / fname
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {fname} ({size:,} bytes)")
        else:
            errors.append(f"Missing: {fname}")
            print(f"  ✗ {fname} — NOT FOUND")

    # 2. Load and verify XGBoost model
    print("\n2. Loading XGBoost model...")
    try:
        xgb = joblib.load(MODELS_DIR / "xgb_model.joblib")
        print(f"  ✓ Model loaded: {type(xgb).__name__}")
        print(f"  ✓ N estimators: {xgb.n_estimators}")
        print(f"  ✓ Max depth: {xgb.max_depth}")
    except Exception as e:
        errors.append(f"Failed to load XGBoost: {e}")
        print(f"  ✗ Error: {e}")
        return errors, warnings

    # 3. Load test data
    print("\n3. Loading test data...")
    try:
        splits = joblib.load(MODELS_DIR / "data_splits.joblib")
        X_test = splits["X_test"]
        y_test = splits["y_test"]
        print(f"  ✓ Test set: {X_test.shape[0]} samples × {X_test.shape[1]} features")
        print(f"  ✓ At-risk in test: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
    except Exception as e:
        errors.append(f"Failed to load test data: {e}")
        print(f"  ✗ Error: {e}")
        return errors, warnings

    # 4. Run predictions
    print("\n4. Running predictions...")
    try:
        y_pred = xgb.predict(X_test)
        y_proba = xgb.predict_proba(X_test)[:, 1]
        print(f"  ✓ Predictions generated: {len(y_pred)} samples")
        print(f"  ✓ Predicted at-risk: {y_pred.sum()} ({y_pred.mean()*100:.1f}%)")
        print(f"  ✓ Risk score range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    except Exception as e:
        errors.append(f"Prediction failed: {e}")
        print(f"  ✗ Error: {e}")
        return errors, warnings

    # 5. Compare with saved metrics
    print("\n5. Comparing with saved metrics...")
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        computed = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        with open(REPORTS_DIR / "model_metrics.json") as f:
            saved = json.load(f)
        saved_xgb = saved.get("xgboost", {})

        for metric, value in computed.items():
            saved_val = saved_xgb.get(metric, 0)
            diff = abs(value - saved_val)
            status = "✓" if diff < 0.001 else "⚠"
            if diff >= 0.001:
                warnings.append(f"{metric}: computed={value:.4f} vs saved={saved_val:.4f}")
            print(f"  {status} {metric:12s}: computed={value:.4f}  saved={saved_val:.4f}  diff={diff:.6f}")

    except Exception as e:
        errors.append(f"Metric comparison failed: {e}")
        print(f"  ✗ Error: {e}")

    # 6. Verify SHAP explainer
    print("\n6. Verifying SHAP explainer...")
    try:
        explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")
        sample = X_test[:5]
        sv = explainer.shap_values(sample)
        print(f"  ✓ Explainer loaded, expected value: {explainer.expected_value:.4f}")
        print(f"  ✓ SHAP values shape: {sv.shape}")
        print(f"  ✓ SHAP range: [{sv.min():.4f}, {sv.max():.4f}]")
    except Exception as e:
        errors.append(f"SHAP verification failed: {e}")
        print(f"  ✗ Error: {e}")

    # 7. Verify baseline model
    print("\n7. Verifying baseline (Logistic Regression)...")
    try:
        lr = joblib.load(MODELS_DIR / "lr_baseline.joblib")
        lr_pred = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, lr_pred)
        print(f"  ✓ Baseline accuracy: {lr_acc:.4f}")
    except Exception as e:
        warnings.append(f"Baseline verification issue: {e}")
        print(f"  ⚠ {e}")

    # 8. Verify scaler
    print("\n8. Verifying scaler...")
    try:
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
        print(f"  ✓ Scaler loaded: {type(scaler).__name__}")
        print(f"  ✓ Feature columns: {len(feature_cols)} features")
        print(f"  ✓ Features: {', '.join(feature_cols[:5])}...")
    except Exception as e:
        warnings.append(f"Scaler check issue: {e}")
        print(f"  ⚠ {e}")

    # Summary
    print("\n" + "=" * 60)
    if not errors:
        print("  ✅ ALL CHECKS PASSED — Model is working correctly!")
    else:
        print(f"  ❌ {len(errors)} ERROR(S) FOUND:")
        for e in errors:
            print(f"     • {e}")

    if warnings:
        print(f"\n  ⚠ {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"     • {w}")

    print("=" * 60)
    return errors, warnings


if __name__ == "__main__":
    verify()
