# Fraud Detection Copilot - Project Context

## Project Overview
End-to-end fraud detection system targeting fraud/risk roles at LATAM fintechs (Mercado Pago, Ualá, dLocal, Naranja X). Built as a portfolio project to demonstrate production-style ML work.

**Status**: Week 2 of 4 (feature engineering done, starting baseline model)

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
- `notebooks/` — Exploration only (EDA, experiments). Numbered: 01_eda, 02_feature_validation, etc.
- `scripts/` — Orchestration scripts (e.g., build_features.py runs the full feature pipeline)
- `src/` — Production code, importable as modules
  - `src/data/` — Data loading and conversion
  - `src/features/` — Feature engineering (uid, time, velocity, split, encoding, aggregations)
  - `src/models/` — Model training and evaluation
  - `src/api/` — FastAPI app
  - `src/dashboard/` — Streamlit app
  - `src/llm/` — Claude-powered explainer agent
- `models/` — Trained model artifacts (.pkl, gitignored)
- `models/encoders/` — Serialized feature-engineering artifacts (frequency encoders, uid aggregations) as JSON, used at serving time
- `tests/` — Unit tests
- `domain_notes.md` — User's learning notes on fintech/payments domain (CRITICAL: do not modify without explicit user request — see "Protected Files" section below)
- `v2_ideas.md` — Out-of-scope ideas parked for v2
- `PROGRESS.md` — Session-by-session log of work done, decisions made, and next steps (see "Session Workflow" section below)

## Working Principles
1. **Pedagogical mode**: For every non-trivial technical decision, briefly explain (a) what it is, (b) why we chose it over the obvious alternatives, (c) when the alternative would be better, (d) how it connects to the Fraud Data Scientist role. The user is learning the domain and wants to defend each decision in an interview. Skip the explanation only for trivial preferences.
2. **No deep learning for tabular fraud**. LightGBM is the right tool. Do not suggest neural nets unless explicitly asked.
3. **Metrics**: precision-recall, AUC-PR, precision@K, expected cost. NEVER report ROC-AUC as the primary metric — it's misleading at ~3.5% fraud rate.
4. **Temporal split, not random**. Future data must not leak into training.
5. **No scope creep**. If the user mentions an idea outside the 4-week scope, suggest writing it to `v2_ideas.md` instead of building it.
6. **Anti-pattern to avoid**: starting many things at 60% completion. Finish before moving on.
7. **Go cell-by-cell / step-by-step**. The user wants to understand what's happening, not have a wall of code dumped. After writing each meaningful piece of code, pause and let the user run/review it before continuing.

## Working Modes

There are two valid modes for executing tasks. The default is "step-by-step" for this user. Switch only when explicitly asked.

### Default: Step-by-step mode
For each meaningful unit of work (a function, a notebook cell, a config block, a non-trivial refactor):
1. Briefly explain WHAT you're about to write and WHY (1-3 sentences).
2. Write the code for that unit only.
3. STOP. Wait for the user to confirm before continuing to the next unit.
4. After the user confirms (or asks for changes), repeat with the next unit.

Do not chain multiple units in one response. Do not auto-run scripts after generating them unless the user explicitly says "go ahead and run it" or equivalent.

This mode is the default because the user is learning the domain and wants to absorb decisions as they're made. Speed is secondary to understanding.

### Build-the-spec mode
Activate only when the user says something like "build the full module", "go ahead end-to-end", "no need to pause", or similar explicit signal.

In this mode:
1. Build the full deliverable as specified.
2. Show the result for review.
3. Wait for the user to run it themselves and report back.

Even in this mode, never auto-execute scripts that modify files outside the immediate task scope or that take significant time. Always pause before running.

### How to recognize which mode to use
- If the user's prompt contains phrases like "vamos paso a paso", "celda por celda", "esperá mi confirmación", "no avances sin que te diga" → step-by-step.
- If the user's prompt is a full spec of a deliverable with definitions, constraints, and entregables listed (like a ticket) → ask once: "¿lo construyo end-to-end o vamos pieza por pieza?". Default to step-by-step if no answer.
- If the user asks a conceptual question ("¿por qué X?", "¿qué es Y?") → answer first, do not write code until asked.

### File creation convention
The USER creates files (the chat tells them the name, location, and initial content). Claude Code then EDITS existing files. One file at a time — if a task needs a module plus its tests, that's two separate prompts. Claude Code may run scripts and tests directly to verify its own work. Exception: flag before any run that overwrites an existing trained model artifact or regenerates the full processed dataset — don't run those silently.

### Anti-pattern to avoid
Do NOT generate a large module, immediately run it, show the output, and ask "¿seguimos?". By that point the user has lost the chance to intervene at each decision. Pause earlier.

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

### PROGRESS.md Updates — Division of Labor
At the end of each task, PROGRESS.md gets a new entry. Two parties contribute:

**Claude Code writes the `Done:` block.** Files created/modified, functions added, scripts executed, outputs generated, artifacts saved. Mechanical, factual, code-level detail.

**The chat (Felipe + Claude) writes `Decisions:`, `Open questions / Notes:`, `Next:`.** The *why*: criterion behind choices, trade-offs, caveats, what was deferred, what's next and why. Claude Code did not participate in that discussion and cannot reconstruct it reliably.

Workflow: when Claude Code finishes a task, it writes the entry with `Done:` filled and the other three sections as `<pending: chat>` placeholders. The user pastes it into the chat; the chat fills in the three pending sections. The user commits the merged entry.

### v2_ideas.md
If the user mentions an idea that is interesting but out of scope for the 4-week plan, suggest adding it to `v2_ideas.md` (one line per idea, dated). Do not silently expand scope.

## User Profile
- Felipe Rivas, final-year student at UTDT (Buenos Aires).
- Background: Python, SQL, ML supervised/unsupervised, deep learning with PyTorch.
- Domain knowledge of payments/fintech: solid foundation built during the project (anatomy of a payment, who earns money, transaction types, chargebacks, fraud taxonomy). KYC/AML deferred to post-project.
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
Week 2: Baseline model. Feature engineering is COMPLETE. The full pipeline runs via `scripts/build_features.py` (~20s, one command) producing `data/processed/train_transaction_features.parquet` (590,540 rows × 460 cols).

Feature modules in `src/features/`, each exposing a pure `run(df) -> df` function plus individual `add_*`/fit/transform functions (so they're reusable in the API for single-transaction scoring without parquet round-trips):
- `uid.py`: two synthetic UIDs (uid1 = card1+addr1, dense, base for velocity; uid2 = card1+addr1+card-birthday via D1, precise, base for aggregations).
- `time.py`: hour, day_of_week, day_index (helper, NOT a feature — temporal leak).
- `velocity.py`: per-uid1 rolling windows (1h/24h: count, sum, mean amt, distinct ProductCD) + time_since_last. Strict past-only look-back, 6 passing tests.
- `split.py`: temporal 70/15/15 by TransactionDT (train 413k, val 88k, test 88k). Tests passing.
- `encoding.py`: frequency encoding of 6 high-cardinality categoricals (card1, card2, addr1, P_emaildomain, R_emaildomain, DeviceInfo). Fit on train, unseen → NaN. Artifacts in models/encoders/frequency_encoders.json.
- `aggregations.py`: per-uid2 count/mean/std amt + amt_ratio_to_uid2_mean + amt_zscore_uid2. Fit on train. Artifacts in models/encoders/uid2_aggregations.json.

KEY ANALYTICAL FINDING (validated two independent ways before training): fraud in IEEE-CIS is predominantly one-shot CNP with no prior uid history. Velocity correlates ~0 with isFraud; deviation features (amt_ratio, amt_zscore) have lift < 1 (amt_ratio>10 → 0.35x). "Out-of-pattern" flags established legitimate users, not fraudsters, because one-shot fraud has no prior pattern to deviate from. Implication: highest-importance features will likely be self-contained transaction attributes (TransactionAmt, ProductCD, Vesta C/D/V columns, card1_freq, identity), not history-based ones. To confirm with SHAP.

NEXT: baseline LightGBM (feature selection, low-cardinality categoricals as native categorical_feature, no imbalance handling initially, metrics = PR-AUC + precision@K + recall@precision on val), then cost-based threshold tuning, then SHAP.