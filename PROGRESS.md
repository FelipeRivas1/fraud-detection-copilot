# PROJECT PROGRESS LOG

---

## 2026-05-21 - Session 2: EDA notebook (01_eda.ipynb)

Done:
- Created `notebooks/01_eda.ipynb` with full initial EDA (9 cells)
- Converted raw CSVs to parquet on first run (`data/processed/`)
- Analyzed class imbalance: 3.5% fraud rate, ~28:1 ratio
- Analyzed `TransactionAmt` distribution by class (histograms + boxplot, log scale)
- Derived `hour` feature from `TransactionDT`; plotted fraud rate and volume by hour
- Plotted fraud rate by `ProductCD`, `card4`, `card6` vs global average
- Analyzed missing values: 174/394 features >50% NaN
- Analyzed identity join coverage: ~24% of transactions have identity data
- Added markdown summary table with key takeaways and modeling implications

Decisions:
- Load from parquet if available, convert from CSV otherwise — avoids re-parsing 500MB on every session
- `log1p` for amount histograms — handles zero-amounts and right-skewed distribution
- `isin()` for identity coverage check instead of full merge — sufficient for analysis, lower memory footprint
- Identity data has higher fraud rate (~selection bias) — will treat `has_identity` as a feature, not require it as model input

Open questions / Notes:
- Did not verify exact fraud rates for identity with/without (user ran cells but didn't report the ratio)
- 174 features >50% NaN: need to decide in week 2 which ones get `is_missing` indicators vs plain drop
- `TransactionDT` reference point is unknown — hour-of-day is recoverable but absolute dates are not

Next:
- `notebooks/02_features.ipynb`: feature engineering (velocity aggregations, cyclic hour encoding, log-amount, is_missing indicators)
- Start `src/features/` with production-ready versions of the engineered features
- Temporal train/val split using `TransactionDT`

---

## 2026-05-22 - Session 3: UID synthetic IDs (src/features/uid.py)

Done:
- Created `src/features/uid.py` with two UID definitions and diagnostics function.
- `uid1 = card1 + addr1`: 37,531 unique UIDs, 11.1% NaN, median group size 2, p90=23.
- `uid2 = card1 + addr1 + (TransactionDay - D1)`: 199,070 unique UIDs, 11.3% NaN, median group size 1, p75=3.
- Ran diagnostics on both: fraud rate per UID is bimodal (most UIDs at 0, tail at 1.0), confirms UIDs group coherently.
- Manual inspection of 5 random UIDs per type: consistent ProductCD and amount ranges within each UID.
- Saved enriched dataset to `data/processed/train_transaction_with_uid.parquet` (80.9 MB).

Decisions:
- Use uid1 as base for velocity features (higher density = more signal in short time windows).
- Use uid2 as base for static aggregations (higher precision = better per-user stats).
- Both UIDs go into the model; LightGBM picks which one is more useful per split.
- NaN handling: if any input column is NaN, UID is NaN (no imputation). Around 11% of rows will have no velocity features, which is acceptable.

Open questions / Notes:
- uid1 has max group size 5885 — almost certainly a masking collision, not a real user. Will need to be careful when computing velocity (large groups can dominate aggregations). Possible mitigation: clip aggregations or add a flag for "uid is suspiciously large".
- uid2 too sparse for velocity (p50=1), only useful for static aggregations. Confirmed.
- No tests yet — will add when we have velocity logic (more complex, more error-prone).

Next:
- Time features: hour, day_of_week, time_since_last from TransactionDT.
- Velocity features per uid1: count/sum/distinct in rolling windows (1h, 6h, 24h), time_since_last_uid.
- Strict past-only look-back; explicit unit test on a hand-built UID before trusting the full dataset.