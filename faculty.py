"""
EduPredict — Faculty dashboard utilities.

Provides batch prediction and cohort analytics used by the faculty page.
All data processed here is synthetic unless explicitly replaced with real data.

Real-data authentication requirements (documented, not implemented):
  - Role-based access control (RBAC): only faculty/admin roles may access
    cohort-level student data.
  - Session tokens with expiry (e.g., JWT or OAuth2 via an institution's SSO).
  - Data minimisation: expose only the indicators needed for the task.
  - Audit logging: every cohort export must be logged with user ID + timestamp.
  - Data-at-rest encryption for any CSV that leaves the system.
  - FERPA / local equivalent compliance before processing real student records.
  This project uses 100% synthetic data and has no authentication layer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib

from config import (
    FEATURE_COLS, TARGET_COL, PERFORMANCE_CLASSES,
    MODEL_PATH, PIPELINE_PATH, CLASS_COLORS, DATASET_PATH,
)
from explain import load_model, FEATURE_META


# ── Batch prediction ───────────────────────────────────────────────────────

def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on every row of df in one vectorised call.

    Returns a copy of df with extra columns:
      predicted_category  — str label
      prob_at_risk        — float probability
      prob_average        — float probability
      prob_high           — float probability
      prediction_correct  — bool  (only if TARGET_COL is present)
    """
    pipeline, le = load_model()

    X     = df[FEATURE_COLS].copy()
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)

    out = df.copy()
    classes = list(le.classes_)   # ['at-risk', 'average', 'high']
    out["predicted_category"] = le.inverse_transform(preds)
    out["prob_at_risk"]        = proba[:, classes.index("at-risk")]
    out["prob_average"]        = proba[:, classes.index("average")]
    out["prob_high"]           = proba[:, classes.index("high")]

    if TARGET_COL in df.columns:
        out["prediction_correct"] = (out["predicted_category"] == out[TARGET_COL])

    return out


def get_atrisk_table(predicted_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only predicted at-risk students, sorted by at-risk probability descending.
    """
    atrisk = predicted_df[predicted_df["predicted_category"] == "at-risk"].copy()
    atrisk = atrisk.sort_values("prob_at_risk", ascending=False)
    return atrisk


def cohort_summary(predicted_df: pd.DataFrame) -> dict:
    """
    Aggregate statistics for the current cohort slice.
    """
    total = len(predicted_df)
    cats  = predicted_df["predicted_category"].value_counts()
    return {
        "total":         total,
        "at_risk_n":     int(cats.get("at-risk",  0)),
        "average_n":     int(cats.get("average",  0)),
        "high_n":        int(cats.get("high",     0)),
        "at_risk_pct":   round(cats.get("at-risk", 0) / total * 100, 1) if total else 0,
        "avg_attendance": round(predicted_df["attendance_pct"].mean(), 1),
        "avg_marks":      round(predicted_df["internal_marks"].mean(), 1),
        "avg_gpa":        round(predicted_df["prev_sem_gpa"].mean(), 2),
        "avg_study_hrs":  round(predicted_df["study_hours_per_week"].mean(), 1),
    }


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of numeric features only."""
    return df[FEATURE_COLS].corr()


def export_csv(predicted_df: pd.DataFrame) -> bytes:
    """Serialise the filtered + predicted DataFrame to CSV bytes for download."""
    cols_order = (
        ["student_id", TARGET_COL, "predicted_category",
         "prob_at_risk", "prob_average", "prob_high"]
        + FEATURE_COLS
    )
    cols_present = [c for c in cols_order if c in predicted_df.columns]
    return predicted_df[cols_present].to_csv(index=False).encode("utf-8")
