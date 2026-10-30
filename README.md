# EduPredict — AI-Driven Student Academic Performance Prediction System

> **Academic project using 100% synthetic data. No real student records are used or stored.**

## Overview

EduPredict predicts whether a student is likely to be **at-risk**, **average**, or **high** performing
based on six academic indicators, and provides explainable, actionable recommendations.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML | scikit-learn |
| Dashboard | Streamlit + Plotly |
| Data | Pandas / NumPy (synthetic) |

## Project Structure

```
EduPredict/
├── data/               # Generated synthetic dataset
├── src/                # Core modules (config, data gen, ML, explain)
├── models/             # Saved model pipeline
├── app/                # Streamlit dashboard
├── reports/            # Charts, metrics, summary
├── tests/              # pytest test suite
├── notebooks/          # Exploratory notebooks
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic dataset
python src/generate_data.py

# 4. Train models
python src/train.py

# 5. Launch dashboard
streamlit run app/main.py
```

## Phases

- [x] Phase 1 — Setup & project structure
- [x] Phase 2 — Synthetic dataset generation
- [x] Phase 3 — Early dashboard preview
- [x] Phase 4 — ML training & evaluation
- [x] Phase 5 — Prediction & explainability
- [x] Phase 6 — Personalized recommendations
- [x] Phase 7 — Faculty dashboard
- [ ] Phase 8 — Tests & integration
- [ ] Phase 9 — Final documentation

## Disclaimer

All student data in this project is **synthetically generated** for demonstration purposes.
It does not represent any real individual and must not be used to make decisions about actual students.
