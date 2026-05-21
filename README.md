# Fraud Detection Copilot

End-to-end fraud detection system built as a portfolio project targeting fraud/risk roles at LATAM fintechs.

## Status
🚧 Work in progress — Week 1 of 4

## Overview
Production-style ML system to detect fraudulent payment transactions, including:
- **ML model**: LightGBM trained on IEEE-CIS Fraud Detection dataset, with engineered velocity and aggregation features, class imbalance handling, and cost-based threshold tuning.
- **Explainability**: SHAP values for global feature importance and per-transaction waterfall plots.
- **API**: FastAPI scoring endpoint serving predictions in <200ms.
- **Dashboard**: Streamlit interface for fraud analysts to review flagged transactions.
- **Educational LLM agent**: Claude-powered explainer that generates both technical (analyst) and educational (learner) views of flagged transactions.

## Why this project
Most fraud detection portfolio projects stop at "I trained a model and got 0.95 AUC". This one explicitly:
- Reports precision-recall and dollar-cost metrics (not ROC-AUC, which is misleading at extreme imbalance).
- Tunes the decision threshold by business cost, not by F1.
- Ships as a deployable system, not a notebook.
- Documents the payments/fintech domain knowledge alongside the technical work (`domain_notes.md`).

## Stack
Python 3.11 · pandas · LightGBM · SHAP · FastAPI · Streamlit · Claude API

## Project Structure