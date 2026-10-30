"""
Quick setup verification — run once after installing requirements.
  python verify_setup.py
"""

import sys
import importlib
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

REQUIRED = [
    ("streamlit",    "streamlit"),
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("sklearn",      "scikit-learn"),
    ("plotly",       "plotly"),
    ("matplotlib",   "matplotlib"),
    ("seaborn",      "seaborn"),
    ("joblib",       "joblib"),
    ("pytest",       "pytest"),
]

print("=" * 50)
print("EduPredict — Setup Verification")
print("=" * 50)
print(f"Python : {sys.version}")
print()

all_ok = True
for module, pkg in REQUIRED:
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "unknown")
        print(f"  OK   {pkg:<25} {ver}")
    except ImportError:
        print(f"  FAIL {pkg:<25} NOT FOUND")
        all_ok = False

print()
from config import (
    PROJECT_ROOT, RANDOM_SEED, FEATURE_COLS,
    TARGET_COL, PERFORMANCE_CLASSES
)
print(f"config.py loaded   : OK")
print(f"Project root       : {PROJECT_ROOT}")
print(f"Random seed        : {RANDOM_SEED}")
print(f"Feature columns    : {FEATURE_COLS}")
print(f"Target column      : {TARGET_COL}")
print(f"Performance classes: {PERFORMANCE_CLASSES}")

print()
print("=" * 50)
if all_ok:
    print("All checks passed. Ready for Phase 2.")
else:
    print("Some packages are missing. Run:  pip install -r requirements.txt")
print("=" * 50)
