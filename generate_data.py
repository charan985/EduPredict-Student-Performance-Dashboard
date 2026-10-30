"""
EduPredict — Synthetic student dataset generator.

IMPORTANT: All data produced by this script is SYNTHETIC.
No real student records are used, stored, or referenced.
Do not use this data to make decisions about actual students.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATASET_PATH, REPORTS_DIR, RANDOM_SEED,
    N_STUDENTS, FEATURE_COLS, TARGET_COL,
    PERFORMANCE_CLASSES, CLASS_COLORS,
)

# ── Per-class feature distributions ───────────────────────────────────────
# Each tuple: (mean, std, clip_low, clip_high)
CLASS_PARAMS = {
    "at-risk": {
        "attendance_pct":            (50.0, 10.0, 20.0,  65.0),
        "internal_marks":            (18.0,  5.0,  5.0,  28.0),
        "assignment_completion_pct": (44.0, 14.0, 10.0,  64.0),
        "study_hours_per_week":      ( 5.0,  2.0,  1.0,  11.0),
        "prev_sem_gpa":              ( 4.5,  1.0,  2.0,   6.2),
        "activity_participation":    ( 2.0,  1.0,  0.0,   5.0),
    },
    "average": {
        "attendance_pct":            (72.0,  8.0, 55.0,  85.0),
        "internal_marks":            (30.0,  5.0, 20.0,  40.0),
        "assignment_completion_pct": (70.0, 10.0, 48.0,  86.0),
        "study_hours_per_week":      (12.0,  3.0,  6.0,  20.0),
        "prev_sem_gpa":              ( 6.5,  0.8,  4.8,   8.0),
        "activity_participation":    ( 4.0,  1.5,  1.0,   7.0),
    },
    "high": {
        "attendance_pct":            (88.0,  5.0, 75.0, 100.0),
        "internal_marks":            (42.0,  4.0, 33.0,  50.0),
        "assignment_completion_pct": (92.0,  5.0, 80.0, 100.0),
        "study_hours_per_week":      (22.0,  4.0, 13.0,  35.0),
        "prev_sem_gpa":              ( 8.5,  0.7,  7.4,  10.0),
        "activity_participation":    ( 7.0,  1.5,  4.0,  10.0),
    },
}

# Class sizes must sum to N_STUDENTS
CLASS_SIZES = {"at-risk": 250, "average": 500, "high": 250}


def _generate_class(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    params = CLASS_PARAMS[label]
    records = {}
    for col in FEATURE_COLS:
        mean, std, lo, hi = params[col]
        values = rng.normal(mean, std, n)
        values = np.clip(values, lo, hi)
        # Round to 1 decimal; activity_participation is integer
        if col == "activity_participation":
            values = np.round(values).astype(int)
        else:
            values = np.round(values, 1)
        records[col] = values
    records[TARGET_COL] = label
    return pd.DataFrame(records)


def generate(n_students: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    frames = []
    for label in PERFORMANCE_CLASSES:
        n = CLASS_SIZES[label]
        frames.append(_generate_class(label, n, rng))

    df = pd.concat(frames, ignore_index=True)

    # Shuffle rows so classes are not in blocks
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add student_id as the first column
    df.insert(0, "student_id", [f"STU{str(i+1).zfill(4)}" for i in range(len(df))])

    return df


def validate(df: pd.DataFrame) -> None:
    expected_cols = ["student_id"] + FEATURE_COLS + [TARGET_COL]
    assert list(df.columns) == expected_cols, "Column mismatch"
    assert df[TARGET_COL].isin(PERFORMANCE_CLASSES).all(), "Unknown class label"
    assert (df["attendance_pct"].between(0, 100)).all()
    assert (df["internal_marks"].between(0, 50)).all()
    assert (df["assignment_completion_pct"].between(0, 100)).all()
    assert (df["study_hours_per_week"].between(0, 40)).all()
    assert (df["prev_sem_gpa"].between(0, 10)).all()
    assert (df["activity_participation"].between(0, 10)).all()
    assert df.isnull().sum().sum() == 0, "Unexpected missing values"


def save_class_distribution_chart(df: pd.DataFrame) -> Path:
    counts = df[TARGET_COL].value_counts().reindex(PERFORMANCE_CLASSES)
    colors = [CLASS_COLORS[c] for c in PERFORMANCE_CLASSES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "EduPredict — Synthetic Student Performance Distribution",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # Bar chart
    bars = axes[0].bar(PERFORMANCE_CLASSES, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Student Count per Category")
    axes[0].set_xlabel("Performance Category")
    axes[0].set_ylabel("Number of Students")
    axes[0].set_ylim(0, counts.max() * 1.2)
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            str(val), ha="center", va="bottom", fontweight="bold",
        )

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=PERFORMANCE_CLASSES,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Class Proportion")

    fig.text(
        0.5, -0.03,
        "Note: All data is 100% synthetic — generated for demonstration only.",
        ha="center", fontsize=9, color="gray", style="italic",
    )

    plt.tight_layout()
    out_path = REPORTS_DIR / "class_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_report(df: pd.DataFrame, chart_path: Path) -> None:
    total = len(df)
    counts = df[TARGET_COL].value_counts().reindex(PERFORMANCE_CLASSES)

    print("=" * 60)
    print("EduPredict — Synthetic Dataset Report")
    print("=" * 60)
    print(f"\n  Records generated : {total}")
    print(f"  Columns           : {len(df.columns)}  →  {list(df.columns)}")

    print("\n── First 10 rows ─────────────────────────────────────────")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df.head(10).to_string(index=False))

    print("\n── Class distribution ────────────────────────────────────")
    print(f"  {'Category':<10}  {'Count':>6}  {'Percentage':>10}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*10}")
    for label in PERFORMANCE_CLASSES:
        n = counts[label]
        pct = n / total * 100
        print(f"  {label:<10}  {n:>6}  {pct:>9.1f}%")
    print(f"  {'TOTAL':<10}  {total:>6}  {'100.0%':>10}")

    print("\n── Missing value counts ──────────────────────────────────")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values — dataset is complete.")
    else:
        print(missing[missing > 0])

    print("\n── Feature summary statistics ────────────────────────────")
    print(df[FEATURE_COLS].describe().round(2).to_string())

    print(f"\n  Chart saved → {chart_path}")
    print(f"  Dataset saved → {DATASET_PATH}")
    print("\n  ⚠  SYNTHETIC DATA: No real student records were used.")
    print("=" * 60)


def main() -> pd.DataFrame:
    print("Generating synthetic student dataset …")
    df = generate()
    validate(df)
    df.to_csv(DATASET_PATH, index=False)
    chart_path = save_class_distribution_chart(df)
    print_report(df, chart_path)
    return df


if __name__ == "__main__":
    main()
