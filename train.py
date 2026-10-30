"""
EduPredict — Model training and evaluation.

Trains Logistic Regression, Random Forest, and Gradient Boosting,
evaluates each on the held-out test set, and saves the best model.

Run:
    python src/train.py
"""

import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import sklearn

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config import (
    FEATURE_COLS, TARGET_COL, PERFORMANCE_CLASSES,
    MODEL_PATH, PIPELINE_PATH, METRICS_PATH, MODEL_METADATA_PATH,
    REPORTS_DIR, RANDOM_SEED, CV_FOLDS, CLASS_COLORS,
)
from preprocess import load_dataset, split, build_feature_pipeline

# ── Classifier definitions ─────────────────────────────────────────────────
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        C=1.0,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=RANDOM_SEED,
        subsample=0.8,
    ),
}

# ── Colour palette for charts ──────────────────────────────────────────────
MODEL_COLORS = {
    "Logistic Regression": "#6366f1",
    "Random Forest":       "#10B981",
    "Gradient Boosting":   "#F59E0B",
}


def build_full_pipeline(classifier) -> Pipeline:
    """Compose the feature scaler + classifier into one sklearn Pipeline."""
    return Pipeline([
        ("scaler",     build_feature_pipeline().named_steps["scaler"]),
        ("classifier", classifier),
    ])


def evaluate_model(pipeline, X_train, X_test, y_train, y_test,
                   le, cv: int = CV_FOLDS) -> dict:
    """Fit pipeline, run CV, evaluate on test set."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    cv_f1 = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_strategy, scoring="f1_macro", n_jobs=-1,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )

    atrisk_idx = list(le.classes_).index("at-risk")
    return {
        "accuracy":       round(accuracy_score(y_test, y_pred), 4),
        "macro_f1":       round(f1_score(y_test, y_pred, average="macro"), 4),
        "cv_f1_mean":     round(cv_f1.mean(), 4),
        "cv_f1_std":      round(cv_f1.std(), 4),
        "atrisk_recall":  round(report["at-risk"]["recall"], 4),
        "atrisk_precision": round(report["at-risk"]["precision"], 4),
        "report":         report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred":         y_pred.tolist(),
    }


def save_confusion_matrix(cm_data: list, model_name: str, le) -> Path:
    cm = np.array(cm_data)
    labels = le.classes_

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix — {model_name}",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold", fontsize=13)

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    out = REPORTS_DIR / f"confusion_{safe_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def save_comparison_chart(results: dict) -> Path:
    names   = list(results.keys())
    metrics = ["accuracy", "macro_f1", "cv_f1_mean", "atrisk_recall"]
    labels  = ["Accuracy", "Macro F1", "CV F1 (mean)", "At-Risk Recall"]

    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=label)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("EduPredict — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = REPORTS_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def select_best(results: dict) -> str:
    """
    Primary sort: macro F1 (balances all three classes).
    Tie-break: at-risk recall (most critical for intervention).
    """
    return max(
        results,
        key=lambda n: (results[n]["macro_f1"], results[n]["atrisk_recall"]),
    )


def print_results(results: dict, best_name: str, le) -> None:
    print("\n" + "=" * 70)
    print("EduPredict — Training & Evaluation Results")
    print("=" * 70)

    for name, res in results.items():
        marker = "  ★ BEST" if name == best_name else ""
        print(f"\n── {name}{marker} {'─'*(40-len(name))}")
        print(f"  Accuracy        : {res['accuracy']:.4f}")
        print(f"  Macro F1        : {res['macro_f1']:.4f}")
        print(f"  CV F1 (mean±std): {res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f}")
        print(f"  At-Risk Recall  : {res['atrisk_recall']:.4f}")
        print(f"  At-Risk Precision:{res['atrisk_precision']:.4f}")
        print()
        print("  Per-class report:")
        for cls in le.classes_:
            r = res["report"][cls]
            print(f"    {cls:<10}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")

    print("\n" + "=" * 70)
    print("Model Comparison Table")
    print("=" * 70)
    header = f"  {'Model':<22} {'Accuracy':>9} {'Macro F1':>9} {'CV F1':>9} {'AtRisk-R':>9}"
    print(header)
    print("  " + "-" * 60)
    for name, res in results.items():
        marker = " ★" if name == best_name else "  "
        print(
            f"{marker} {name:<22} "
            f"{res['accuracy']:>9.4f} "
            f"{res['macro_f1']:>9.4f} "
            f"{res['cv_f1_mean']:>9.4f} "
            f"{res['atrisk_recall']:>9.4f}"
        )
    print()
    print(f"  Selected model  : {best_name}")
    print(f"  Selection criteria:")
    print(f"    1. Highest macro F1  — balances performance across all three classes.")
    print(f"    2. Tie-break: at-risk recall — maximises detection of struggling students.")
    print("=" * 70)


def main():
    print("Loading dataset …")
    df = load_dataset()

    print("Splitting data (stratified 80/20) …")
    X_train, X_test, y_train, y_test, le = split(df)
    print(f"  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
    print(f"  Classes: {dict(zip(le.classes_, [int((y_train==i).sum()) for i in range(len(le.classes_))]))}")

    results   = {}
    pipelines = {}

    for name, clf in CLASSIFIERS.items():
        print(f"\nTraining {name} …")
        pipeline = build_full_pipeline(clf)
        res      = evaluate_model(pipeline, X_train, X_test, y_train, y_test, le)
        results[name]   = res
        pipelines[name] = pipeline

        cm_path = save_confusion_matrix(res["confusion_matrix"], name, le)
        print(f"  Macro F1={res['macro_f1']:.4f}  At-Risk Recall={res['atrisk_recall']:.4f}  → {cm_path.name}")

    best_name    = select_best(results)
    best_pipeline = pipelines[best_name]

    print_results(results, best_name, le)

    # Save best model and its pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    joblib.dump(le,            PIPELINE_PATH)

    # Save version metadata — used by explain.py to detect sklearn mismatches
    metadata = {
        "sklearn_version": sklearn.__version__,
        "python_version":  sys.version.split()[0],
        "best_model":      best_name,
        "trained_at":      datetime.now().isoformat(),
        "feature_cols":    FEATURE_COLS,
        "classes":         PERFORMANCE_CLASSES,
        "train_samples":   len(X_train),
        "test_samples":    len(X_test),
        "accuracy":        results[best_name]["accuracy"],
        "macro_f1":        results[best_name]["macro_f1"],
        "atrisk_recall":   results[best_name]["atrisk_recall"],
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model saved    → {MODEL_PATH}")
    print(f"  Encoder saved  → {PIPELINE_PATH}")
    print(f"  Metadata saved → {MODEL_METADATA_PATH}")
    print(f"  sklearn version: {sklearn.__version__}")

    # Save comparison chart
    chart_path = save_comparison_chart(results)
    print(f"  Comparison chart → {chart_path}")

    # Save metrics JSON (without large arrays for readability)
    metrics_out = {}
    for name, res in results.items():
        metrics_out[name] = {k: v for k, v in res.items() if k not in ("report", "y_pred")}
    metrics_out["best_model"] = best_name
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Metrics JSON   → {METRICS_PATH}")

    return results, best_name, le


if __name__ == "__main__":
    main()
