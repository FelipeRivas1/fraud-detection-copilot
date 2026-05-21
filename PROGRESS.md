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