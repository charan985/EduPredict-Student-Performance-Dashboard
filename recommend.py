"""
EduPredict — Personalized recommendation engine.

Generates prioritized, actionable recommendations based on a student's
academic indicators and predicted performance category.

Priority levels:
  1 = Critical   — immediate action required
  2 = Important  — should be addressed this week
  3 = Suggested  — optimisation opportunity
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FEATURE_COLS, PERFORMANCE_CLASSES
from explain import FEATURE_META

# ── Recommendation rules ───────────────────────────────────────────────────
# Each rule: (feature, operator, threshold, priority, title, action, target_text, category)
#   operator: "lt" (less than) or "gte" (greater than or equal)
#   category: "attendance" | "academic" | "time" | "engagement"

_RULES = [

    # ── ATTENDANCE ─────────────────────────────────────────────────────────
    (
        "attendance_pct", "lt", 60, 1,
        "Critical attendance deficit",
        "Attendance is severely low. Meet your academic advisor this week, "
        "notify your subject teachers, and commit to attending every remaining class. "
        "Most institutions require ≥75% attendance to be eligible for exams.",
        "Reach ≥75% immediately",
        "attendance",
    ),
    (
        "attendance_pct", "lt", 75, 2,
        "Below-threshold attendance",
        "Attendance is below the recommended 75%. "
        "Limit absences to emergencies only. "
        "If unavoidable absences occur, arrange notes from a classmate the same day.",
        "Reach ≥80%",
        "attendance",
    ),
    (
        "attendance_pct", "lt", 85, 3,
        "Attendance can be strengthened",
        "Consistent attendance above 85% correlates with higher performance. "
        "Set a phone reminder 30 minutes before each class.",
        "Reach ≥85%",
        "attendance",
    ),

    # ── INTERNAL MARKS ─────────────────────────────────────────────────────
    (
        "internal_marks", "lt", 20, 1,
        "Marks critically low",
        "Internal marks are below 20/50. Request immediate academic support: "
        "attend office hours, form a small study group, and review the last two "
        "semesters of question papers. Focus on 2–3 core topics at a time rather "
        "than trying to cover everything at once.",
        "Target ≥28/50",
        "academic",
    ),
    (
        "internal_marks", "lt", 30, 2,
        "Internal marks below average",
        "Marks are below average. Review incorrect exam answers to identify "
        "recurring gaps. Use the Feynman technique: explain each concept aloud "
        "in simple language to test your understanding.",
        "Target ≥35/50",
        "academic",
    ),
    (
        "internal_marks", "lt", 38, 3,
        "Marks have room to grow",
        "You are performing at an average level. Targeting 2–3 weak topics per week "
        "and practising past questions under timed conditions can push marks above 40.",
        "Target ≥40/50",
        "academic",
    ),

    # ── ASSIGNMENT COMPLETION ──────────────────────────────────────────────
    (
        "assignment_completion_pct", "lt", 50, 1,
        "Less than half of assignments submitted",
        "Fewer than half your assignments have been completed. "
        "List every pending assignment, estimate hours needed, and block time in "
        "your calendar today. Ask teachers about late-submission policies. "
        "Submitting even partial work is better than zero.",
        "Complete all pending work",
        "academic",
    ),
    (
        "assignment_completion_pct", "lt", 70, 2,
        "Assignment completion below 70%",
        "A significant portion of assignments are incomplete. "
        "Use a task-tracking app (Notion, Trello, or a paper checklist) to capture "
        "every deadline. Break large assignments into 25-minute Pomodoro sessions.",
        "Reach ≥80%",
        "academic",
    ),
    (
        "assignment_completion_pct", "lt", 88, 3,
        "Aim for full assignment completion",
        "Completing 90%+ of assignments signals consistency to teachers and "
        "reinforces learning. Set a personal deadline 2 days before the official one.",
        "Reach ≥90%",
        "academic",
    ),

    # ── STUDY HOURS ────────────────────────────────────────────────────────
    (
        "study_hours_per_week", "lt", 6, 1,
        "Study time critically low",
        "Fewer than 6 hours of study per week is a strong predictor of at-risk "
        "performance. Build a fixed daily schedule: even 1 hour each morning "
        "and evening adds up to 14 hours a week. Use a study log to stay accountable.",
        "Reach ≥12 hrs/week",
        "time",
    ),
    (
        "study_hours_per_week", "lt", 12, 2,
        "Increase structured study time",
        "Study time is below recommended levels. Aim for 12–15 hours per week. "
        "Spaced repetition (e.g., Anki flashcards) makes each hour more effective "
        "than last-minute cramming.",
        "Reach ≥12 hrs/week",
        "time",
    ),
    (
        "study_hours_per_week", "lt", 18, 3,
        "More focused study hours recommended",
        "Increasing structured study to 18+ hours per week, with deliberate practice "
        "on difficult problems, is typical of high performers in this dataset.",
        "Reach ≥18 hrs/week",
        "time",
    ),

    # ── PREVIOUS GPA ───────────────────────────────────────────────────────
    (
        "prev_sem_gpa", "lt", 5.0, 1,
        "GPA indicates a persistent trend",
        "A GPA below 5.0 suggests a pattern that needs to be addressed now. "
        "Schedule a meeting with your faculty mentor or counsellor to create "
        "a structured recovery plan for this semester.",
        "Target ≥6.0 this semester",
        "academic",
    ),
    (
        "prev_sem_gpa", "lt", 6.5, 2,
        "GPA is in the lower range",
        "Previous GPA is below average. Identify which subject(s) dragged the "
        "overall score and build a focused improvement plan. A single subject "
        "improved by a full grade can shift the GPA significantly.",
        "Target ≥7.0 this semester",
        "academic",
    ),
    (
        "prev_sem_gpa", "lt", 7.5, 3,
        "GPA can cross into the high tier",
        "You are close to the high-performance threshold. Consistent work on "
        "your 1–2 weakest subjects this semester can move your GPA above 7.5.",
        "Target ≥8.0 this semester",
        "academic",
    ),

    # ── ACTIVITY PARTICIPATION ─────────────────────────────────────────────
    (
        "activity_participation", "lt", 2, 2,
        "Very low campus engagement",
        "Low extracurricular engagement is linked to lower motivation and weaker "
        "peer support networks. Join one club, study circle, or academic event. "
        "Even a single session per week can build useful connections.",
        "Participate in ≥2 activities",
        "engagement",
    ),
    (
        "activity_participation", "lt", 4, 3,
        "Moderate engagement — room to grow",
        "Increasing involvement in academic or co-curricular activities builds "
        "soft skills, improves motivation, and often opens mentorship opportunities.",
        "Participate in 4+ activities",
        "engagement",
    ),
]

# ── Category labels used in the dashboard ──────────────────────────────────
CATEGORY_ICONS = {
    "attendance": "🗓️",
    "academic":   "📚",
    "time":       "⏱️",
    "engagement": "🤝",
}

PRIORITY_LABELS = {
    1: ("Critical",  "#EF4444"),
    2: ("Important", "#F59E0B"),
    3: ("Suggested", "#6366f1"),
}


def get_recommendations(features: dict, predicted_label: str) -> list[dict]:
    """
    Return a prioritised list of recommendations for a student.

    Parameters
    ----------
    features       : dict  mapping FEATURE_COLS keys → float values
    predicted_label: str   one of PERFORMANCE_CLASSES

    Returns
    -------
    list of dicts, sorted by priority ascending (1 = most urgent first).
    Each dict has: priority, priority_label, priority_color, indicator,
    feature_label, current_value, target, title, action, category, icon.
    """
    recs  = []
    seen  = set()   # one recommendation per feature (the highest-priority one)

    for (feat, op, thresh, prio, title, action, target, cat) in _RULES:
        if feat in seen:
            continue
        val = features.get(feat)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue

        triggered = (op == "lt" and val < thresh) or (op == "gte" and val >= thresh)
        if not triggered:
            continue

        # At-risk students escalate priority-2 rules to priority-1
        effective_prio = prio
        if predicted_label == "at-risk" and prio == 2:
            effective_prio = 1

        p_label, p_color = PRIORITY_LABELS[effective_prio]

        recs.append({
            "priority":        effective_prio,
            "priority_label":  p_label,
            "priority_color":  p_color,
            "indicator":       feat,
            "feature_label":   FEATURE_META[feat]["label"],
            "current_value":   round(val, 1),
            "unit":            FEATURE_META[feat]["unit"],
            "target":          target,
            "title":           title,
            "action":          action,
            "category":        cat,
            "icon":            CATEGORY_ICONS[cat],
        })
        seen.add(feat)

    recs.sort(key=lambda r: r["priority"])
    return recs


def summarise(recs: list[dict]) -> dict:
    """Return counts and top action for a quick summary."""
    critical  = [r for r in recs if r["priority"] == 1]
    important = [r for r in recs if r["priority"] == 2]
    suggested = [r for r in recs if r["priority"] == 3]
    return {
        "total":     len(recs),
        "critical":  len(critical),
        "important": len(important),
        "suggested": len(suggested),
        "top_action": recs[0]["title"] if recs else "No issues detected",
    }


# ── CLI demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import DATASET_PATH, TARGET_COL
    from explain import predict

    df = pd.read_csv(DATASET_PATH)

    print("=" * 65)
    print("EduPredict — Recommendation Engine Demo")
    print("=" * 65)

    for true_label in PERFORMANCE_CLASSES:
        row      = df[df[TARGET_COL] == true_label].iloc[3]
        features = {c: float(row[c]) for c in FEATURE_COLS}
        result   = predict(features)
        recs     = get_recommendations(features, result["label"])
        summary  = summarise(recs)

        print(f"\nStudent {row['student_id']}  |  True: {true_label}  |  Predicted: {result['label']}")
        print(f"  Recommendations: {summary['total']} total "
              f"({summary['critical']} critical, {summary['important']} important, "
              f"{summary['suggested']} suggested)")
        for r in recs:
            print(f"\n  [{r['priority_label'].upper():9}] {r['icon']} {r['title']}")
            print(f"    Current: {r['feature_label']} = {r['current_value']}{r['unit']}")
            print(f"    Target : {r['target']}")
            print(f"    Action : {r['action'][:90]}…")
