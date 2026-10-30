"""
EduPredict — Data preprocessing utilities.
Builds a fitted sklearn Pipeline (scaler only) for features.
Target encoding is handled separately to avoid leakage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import (
    DATASET_PATH, FEATURE_COLS, TARGET_COL,
    PERFORMANCE_CLASSES, TEST_SIZE, RANDOM_SEED,
)


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python src/generate_data.py` first."
        )
    df = pd.read_csv(path)
    missing = df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"Dataset has {missing} missing values — check generate_data.py.")
    unknown = set(df[TARGET_COL].unique()) - set(PERFORMANCE_CLASSES)
    if unknown:
        raise ValueError(f"Unknown class labels in dataset: {unknown}")
    return df


def make_label_encoder() -> LabelEncoder:
    """Return a LabelEncoder pre-fitted on the fixed class order."""
    le = LabelEncoder()
    le.fit(PERFORMANCE_CLASSES)   # at-risk=0, average=1, high=2
    return le


def split(df: pd.DataFrame):
    """
    Stratified 80/20 split.
    Returns X_train, X_test, y_train, y_test as DataFrames/Series,
    plus a fitted LabelEncoder for the target.
    """
    X = df[FEATURE_COLS].copy()
    y_raw = df[TARGET_COL].copy()

    le = make_label_encoder()
    y = pd.Series(le.transform(y_raw), name=TARGET_COL)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, le


def build_feature_pipeline() -> Pipeline:
    """Feature-only pipeline: StandardScaler."""
    return Pipeline([("scaler", StandardScaler())])
