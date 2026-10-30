"""
EduPredict — central configuration.
All modules import constants from here to ensure consistency.
"""

import os
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
SRC_DIR      = PROJECT_ROOT / "src"
MODELS_DIR   = PROJECT_ROOT / "models"
APP_DIR      = PROJECT_ROOT / "app"
REPORTS_DIR  = PROJECT_ROOT / "reports"
TESTS_DIR    = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure src/ is on the Python path so sibling modules import cleanly
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Data ───────────────────────────────────────────────────────────────────
DATASET_PATH = DATA_DIR / "students.csv"

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH         = MODELS_DIR / "best_model.joblib"
PIPELINE_PATH      = MODELS_DIR / "pipeline.joblib"
METRICS_PATH       = REPORTS_DIR / "metrics.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Dataset parameters ─────────────────────────────────────────────────────
N_STUDENTS = 1000

# ── ML parameters ──────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
CV_FOLDS    = 5

# ── Feature / target names ─────────────────────────────────────────────────
FEATURE_COLS = [
    "attendance_pct",
    "internal_marks",
    "assignment_completion_pct",
    "study_hours_per_week",
    "prev_sem_gpa",
    "activity_participation",
]

TARGET_COL = "final_performance_label"

PERFORMANCE_CLASSES = ["at-risk", "average", "high"]

# ── Class colours (used in Plotly / matplotlib) ────────────────────────────
CLASS_COLORS = {
    "at-risk": "#EF4444",
    "average": "#F59E0B",
    "high":    "#10B981",
}

# ── Ensure output directories exist at import time ─────────────────────────
for _dir in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
