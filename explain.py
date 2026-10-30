"""
EduPredict — Prediction, feature importance, attention flags, and indicator summary.

Root-cause note (multi_class AttributeError)
---------------------------------------------
sklearn < 1.5 stored 'multi_class' as a LogisticRegression instance attribute
and read it back inside predict_proba(). sklearn >= 1.5 removed that attribute.
Loading a model saved with new sklearn into old sklearn causes:
  AttributeError: 'LogisticRegression' object has no attribute 'multi_class'
Fix: ensure sklearn >= 1.5.0 in your environment, then retrain:
  pip install "scikit-learn>=1.5.0"
  python src/train.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
import sklearn

from config import (
    FEATURE_COLS, TARGET_COL, PERFORMANCE_CLASSES,
    MODEL_PATH, PIPELINE_PATH, MODEL_METADATA_PATH, CLASS_COLORS,
)

# ── Feature display metadata ───────────────────────────────────────────────
FEATURE_META = {
    "attendance_pct":            {"label": "Attendance %",            "min": 0,   "max": 100, "unit": "%"},
    "internal_marks":            {"label": "Internal Marks (/50)",    "min": 0,   "max": 50,  "unit": ""},
    "assignment_completion_pct": {"label": "Assignment Completion %", "min": 0,   "max": 100, "unit": "%"},
    "study_hours_per_week":      {"label": "Study Hours / Week",      "min": 0,   "max": 40,  "unit": "hrs"},
    "prev_sem_gpa":              {"label": "Previous Semester GPA",   "min": 0.0, "max": 10.0,"unit": ""},
    "activity_participation":    {"label": "Activity Participation",  "min": 0,   "max": 10,  "unit": ""},
}

# ── Attention rules ────────────────────────────────────────────────────────
ATTENTION_RULES = [
    ("attendance_pct",            "lt", 65,  "Attendance below 65% — strongly associated with at-risk performance."),
    ("attendance_pct",            "lt", 75,  "Attendance below 75% — consider improving regularity."),
    ("internal_marks",            "lt", 22,  "Internal marks below 22/50 — academic support may help."),
    ("internal_marks",            "lt", 30,  "Internal marks are in the lower-average range."),
    ("assignment_completion_pct", "lt", 50,  "Less than half of assignments completed — a key risk signal."),
    ("assignment_completion_pct", "lt", 70,  "Assignment completion below 70% — room for improvement."),
    ("study_hours_per_week",      "lt", 6,   "Very low study hours (< 6 hrs/week) — a common at-risk indicator."),
    ("study_hours_per_week",      "lt", 10,  "Study hours below 10/week — consider structured study sessions."),
    ("prev_sem_gpa",              "lt", 5.0, "Previous GPA below 5.0 — may indicate a persistent trend."),
    ("prev_sem_gpa",              "lt", 6.0, "Previous GPA is below average — monitoring recommended."),
    ("activity_participation",    "lt", 2,   "Very low activity participation — engagement may be lacking."),
]

# Minimum sklearn version required to load our models correctly
_MIN_SKLEARN = (1, 5, 0)


def _sklearn_version_tuple():
    parts = sklearn.__version__.split(".")
    return tuple(int(p) for p in parts[:3] if p.isdigit())


def _check_sklearn_version():
    current = _sklearn_version_tuple()
    if current < _MIN_SKLEARN:
        min_str = ".".join(str(v) for v in _MIN_SKLEARN)
        raise RuntimeError(
            f"scikit-learn {sklearn.__version__} is too old.\n"
            f"EduPredict requires scikit-learn >= {min_str}.\n\n"
            f"Fix (run in your activated venv):\n"
            f"  pip install \"scikit-learn>={min_str}\"\n"
            f"  python src/train.py\n"
            f"  streamlit run app/main.py"
        )


def load_model():
    """
    Load the best model pipeline and label encoder.
    Raises RuntimeError if sklearn version is incompatible.
    Raises FileNotFoundError with instructions if files are missing.
    """
    _check_sklearn_version()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            f"Run:  python src/train.py"
        )
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Label encoder not found at {PIPELINE_PATH}.\n"
            f"Run:  python src/train.py"
        )

    pipeline = joblib.load(MODEL_PATH)
    le       = joblib.load(PIPELINE_PATH)

    # Validate model metadata version match if available
    if MODEL_METADATA_PATH.exists():
        meta = json.loads(MODEL_METADATA_PATH.read_text())
        saved_ver = tuple(int(v) for v in meta.get("sklearn_version", "0.0.0").split(".")[:3] if v.isdigit())
        current   = _sklearn_version_tuple()
        # Major version mismatch is a hard error; minor mismatch is a warning
        if saved_ver[0] != current[0]:
            raise RuntimeError(
                f"Model was trained with sklearn {meta['sklearn_version']} but "
                f"you are running sklearn {sklearn.__version__}. "
                f"Retrain: python src/train.py"
            )

    return pipeline, le


def validate_features(features: dict) -> list:
    """Return a list of validation error strings (empty list = valid)."""
    errors = []
    for col in FEATURE_COLS:
        if col not in features:
            errors.append(f"Missing feature: {col}")
            continue
        val = features[col]
        meta = FEATURE_META[col]
        try:
            val = float(val)
        except (TypeError, ValueError):
            errors.append(f"{meta['label']}: must be a number (got {val!r})")
            continue
        if not (meta["min"] <= val <= meta["max"]):
            errors.append(
                f"{meta['label']}: {val} is outside valid range "
                f"[{meta['min']}, {meta['max']}]"
            )
    return errors


def predict(features: dict) -> dict:
    """
    Predict performance category for one student.

    Returns dict with keys:
      label         : str
      label_index   : int
      probabilities : dict {class_name: float}
      errors        : list[str]  — non-empty means prediction was skipped
    """
    errors = validate_features(features)
    if errors:
        return {"label": None, "label_index": None,
                "probabilities": {}, "errors": errors}

    try:
        pipeline, le = load_model()
    except (FileNotFoundError, RuntimeError) as exc:
        return {"label": None, "label_index": None,
                "probabilities": {}, "errors": [str(exc)]}

    X = pd.DataFrame([[features[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)

    try:
        idx   = int(pipeline.predict(X)[0])
        label = le.inverse_transform([idx])[0]
    except AttributeError as exc:
        if "multi_class" in str(exc):
            msg = (
                "sklearn version mismatch: your environment has an old sklearn that "
                "cannot load this model.\n"
                "Fix: pip install \"scikit-learn>=1.5.0\", then python src/train.py"
            )
            return {"label": None, "label_index": None,
                    "probabilities": {}, "errors": [msg]}
        raise

    try:
        proba_arr = pipeline.predict_proba(X)[0]
        probs = {
            le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba_arr)
        }
    except AttributeError as exc:
        if "multi_class" in str(exc):
            # predict() succeeded but predict_proba() failed — fall back gracefully
            probs = {cls: (1.0 if cls == label else 0.0) for cls in PERFORMANCE_CLASSES}
        else:
            raise

    return {
        "label":         label,
        "label_index":   idx,
        "probabilities": probs,
        "errors":        [],
    }


def get_feature_importance() -> pd.DataFrame:
    """
    Extract global feature importance from the saved model.
    LR  → mean |coefficient| across classes.
    RF/GB → feature_importances_.
    Returns DataFrame with columns [feature, label, importance], sorted desc.
    """
    pipeline, _ = load_model()
    clf = pipeline.named_steps["classifier"]

    if hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_).mean(axis=0)
        method = "Mean |coefficient| across classes (Logistic Regression)"
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
        method = "Gini impurity reduction (tree ensemble)"
    else:
        importance = np.ones(len(FEATURE_COLS)) / len(FEATURE_COLS)
        method = "Equal weights (method not available for this model type)"

    total = importance.sum()
    if total > 0:
        importance = importance / total

    labels = [FEATURE_META[c]["label"] for c in FEATURE_COLS]
    df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "label":      labels,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df.attrs["method"] = method
    return df


def get_indicator_summary(features: dict, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare a student's indicators against the full dataset.

    Parameters
    ----------
    features     : dict mapping FEATURE_COLS → float values
    reference_df : the full students DataFrame (for computing population stats)

    Returns
    -------
    DataFrame with one row per feature containing:
      feature, label, value, unit, avg, p25, p75,
      score_pct (0-100 normalised), avg_score_pct,
      vs_avg (signed difference), status ('above'/'below'/'at')

    NOTE: score_pct is a normalised representation only.
    Formula: (value - min) / (max - min) × 100
    It is NOT a GPA, official exam score, or validated success probability.
    """
    records = []
    for col in FEATURE_COLS:
        val  = float(features.get(col, 0))
        meta = FEATURE_META[col]
        lo, hi = meta["min"], meta["max"]

        col_data   = reference_df[col].dropna()
        avg        = col_data.mean()
        p25        = col_data.quantile(0.25)
        p75        = col_data.quantile(0.75)
        class_avgs = {
            cls: reference_df.loc[reference_df[TARGET_COL] == cls, col].mean()
            for cls in PERFORMANCE_CLASSES
            if cls in reference_df[TARGET_COL].values
        }

        score_pct     = round((val - lo) / (hi - lo) * 100, 1) if hi > lo else 0.0
        avg_score_pct = round((avg - lo) / (hi - lo) * 100, 1) if hi > lo else 0.0

        diff = val - avg
        if abs(diff) < 0.05:
            status = "at"
        elif diff > 0:
            status = "above"
        else:
            status = "below"

        records.append({
            "feature":       col,
            "label":         meta["label"],
            "value":         round(val, 1),
            "unit":          meta["unit"],
            "avg":           round(avg, 2),
            "p25":           round(p25, 2),
            "p75":           round(p75, 2),
            "class_avgs":    class_avgs,
            "score_pct":     score_pct,
            "avg_score_pct": avg_score_pct,
            "vs_avg":        round(diff, 2),
            "status":        status,
        })
    return pd.DataFrame(records)


def get_attention_flags(features: dict) -> list:
    """
    Return prioritised attention flags for a student.
    One flag per feature (the highest-priority rule that fires).
    """
    flags = []
    seen  = set()
    for feat, op, threshold, message in ATTENTION_RULES:
        if feat in seen:
            continue
        val = features.get(feat)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        triggered = (op == "lt" and val < threshold) or (op == "gt" and val > threshold)
        if triggered:
            severity = "high" if threshold in (65, 22, 50, 6, 5.0, 2) else "medium"
            flags.append({
                "feature":  feat,
                "label":    FEATURE_META[feat]["label"],
                "value":    val,
                "message":  message,
                "severity": severity,
            })
            seen.add(feat)
    return flags
