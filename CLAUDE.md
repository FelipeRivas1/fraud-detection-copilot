Sobrescribí el archivo CLAUDE.md en la raíz del proyecto con el siguiente contenido exacto:

# Fraud Detection Copilot - Project Context

## Project Overview
End-to-end fraud detection system targeting fraud/risk roles at LATAM fintechs (Mercado Pago, Ualá, dLocal, Naranja X). Built as a portfolio project to demonstrate production-style ML work.

**Status**: Week 1 of 4 (setup + EDA)

## Tech Stack
- Python 3.11+ with uv for package management
- pandas, numpy, pyarrow for data
- LightGBM for the model (NOT deep learning — tabular fraud)
- SHAP for explainability
- FastAPI for the scoring API
- Streamlit for the analyst dashboard
- Anthropic SDK (Claude API) for the educational LLM agent
- Deploy targets: Streamlit Community Cloud, Railway/Render

## Project Structure
- `data/raw/` — Original IEEE-CIS CSVs (gitignored, regenerable from Kaggle)
- `data/processed/` — Parquet versions (gitignored)
- `notebooks/` — Exploration only (EDA, experiments). Numbered: 01_eda, 02_features, etc.
- `src/` — Production code, importable as modules
  - `src/data/` — Data loading and conversion
  - `src/features/` — Feature engineering (velocity, aggregations)
  - `src/models/` — Model training and evaluation
  - `src/api/` — FastAPI app
  - `src/dashboard/` — Streamlit app
  - `src/llm/` — Claude-powered explainer agent
- `models/` — Trained model artifacts (.pkl, gitignored)
- `tests/` — Unit tests
- `domain_notes.md` — User's learning notes on fintech/payments domain (CRITICAL: do not modify without explicit user request — see "Protected Files" section below)
- `v2_ideas.md` — Out-of-scope ideas parked for v2
- `PROGRESS.md` — Session-by-session log of work done, decisions made, and next steps (see "Session Workflow" section below)

## Working Principles
1. **Pedagogical mode**: For every non-trivial technical decision, briefly explain (a) what it is, (b) why we chose it over the obvious alternatives, (c) when the alternative would be better. The user is learning the domain from scratch and wants to defend each decision in an interview. Skip the explanation only for trivial preferences.
2. **No deep learning for tabular fraud**. LightGBM is the right tool. Do not suggest neural nets unless explicitly asked.
3. **Metrics**: precision-recall, AUC-PR, precision@K, expected cost. NEVER report ROC-AUC as the primary metric — it's misleading at ~3.5% fraud rate.
4. **Temporal split, not random**. Future data must not leak into training.
5. **No scope creep**. If the user mentions an idea outside the 4-week scope, suggest writing it to `v2_ideas.md` instead of building it.
6. **Anti-pattern to avoid**: starting many things at 60% completion. Finish before moving on.
7. **Go cell-by-cell / step-by-step**. The user wants to understand what's happening, not have a wall of code dumped. After writing each meaningful piece of code, pause and let the user run/review it before continuing.

## Protected Files
The following files belong to the user's personal learning process and MUST NOT be edited without explicit permission:
- `domain_notes.md`: the user writes this in their own words while learning the domain. Editing it would defeat its pedagogical purpose. You may READ it (e.g., as context for the LLM agent in week 4), but never write to it unless the user says "edit domain_notes.md" explicitly.

## Session Workflow

### PROGRESS.md
Maintain a session log at the repo root. At the END of every working session, append a new dated entry to `PROGRESS.md` with this structure:
YYYY-MM-DD - Session N: <short title>
Done:
- bullet list of concrete actions/files created/modified

Decisions:
- bullet list of non-trivial technical decisions made this session, with brief rationale

Open questions / Notes:
- bullet list of things to revisit, doubts, deferred decisions

Next:
- bullet list of what to tackle in the next session

Update `PROGRESS.md` PROACTIVELY at session end — do not wait for the user to ask. The user uses these entries to sync state with a separate planning conversation (the project chat in claude.ai).

### v2_ideas.md
If the user mentions an idea that is interesting but out of scope for the 4-week plan, suggest adding it to `v2_ideas.md` (one line per idea, dated). Do not silently expand scope.

## User Profile
- Felipe Rivas, final-year student at UTDT (Buenos Aires).
- Background: Python, SQL, ML supervised/unsupervised, deep learning with PyTorch.
- Domain knowledge of payments/fintech at project start: essentially zero. Learning in parallel via Fase 0 ramp-up with ChatGPT.
- Wants brutally honest, direct feedback. No flattery. No filler. No "great question!".
- Prefers Spanish for conversation, but technical terms in English (industry standard).
- Will spend ~10 hours/week on this for 4 weeks.

## Communication Style
- Concise, criterion-driven. No bullet lists when prose works.
- Explain trade-offs, not just pick winners.
- Reference the LATAM fintech context when relevant (Mercado Pago, Ualá, etc.).
- When the user makes a mistake or has a suboptimal idea, say so directly.
- Use real-world analogies (ideally fintech or Argentina-related), not generic "imagine a chef" ones.

## Domain Glossary (essentials for understanding code/comments)
- **CNP**: Card-not-present transaction (online, no physical card). Higher fraud risk.
- **Velocity**: Rate of transactions over a time window (e.g., 5 transactions in 10 minutes from same card).
- **Chargeback**: Cardholder-initiated reversal of a transaction; the cost the system aims to avoid.
- **SHAP**: Shapley Additive Explanations; per-prediction feature attribution method.
- **Precision @ K**: Of the top K most-suspicious transactions, what fraction are actually fraud. Important because manual review queues are finite.
- **MDR**: Merchant Discount Rate; what the merchant pays to process a card transaction.

## Current Phase
Week 1: Setup complete. Dataset (IEEE-CIS) downloaded and converted to parquet. Currently working on `notebooks/01_eda.ipynb` for initial exploration.
