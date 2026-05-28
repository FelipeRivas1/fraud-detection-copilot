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

---

## 2026-05-23 - Session 4: Time features (src/features/time.py)

Done:
- Created `src/features/time.py` with add_hour, add_day_of_week, add_day_index, add_all_time_features, and time_diagnostics.
- Ran on the uid-enriched dataset. Saved as `data/processed/train_transaction_features.parquet`.
- Confirmed dataset span: 182 days (6 months), day_index 1→182.
- Hour distribution shows a deep valley between 06h-12h with peak at 19h. Not a typical local-time retail pattern — suggests TransactionDT is in UTC or arbitrary offset, not user-local time. Noted in domain_notes.md.
- day_of_week is mostly uniform (70k-98k per day). Mild signal at most.

Decisions:
- day_index kept as helper column (used by velocity and split), explicitly NOT a model feature (would leak temporal position).
- No cyclic encoding (sin/cos) of hour — LightGBM partitions ranges natively, cyclic encoding is a relic from linear/NN models. Added to v2_ideas if curious later.
- Features encapsulated in functions (not inline) to avoid training-serving skew when the same logic runs in the API endpoint in Week 3.

Open questions / Notes:
- Hour is relative, not local. Caveat noted in domain_notes.md; should also appear in README limitations section.
- Max day has 6852 transactions vs mean 3245 — probably a promo/seasonal day. Not anomalous enough to flag.

Next:
- Velocity features per uid1 (count/sum/distinct in rolling windows 1h, 6h, 24h; time_since_last_uid).
- Strict past-only look-back. Unit test on a hand-built UID before trusting full dataset.

---

## 2026-05-23 - Session 5: Velocity features + tests

Done:
- Created `src/features/velocity.py` with add_time_since_last, add_velocity_window, add_all_velocity_features, velocity_diagnostics.
- Implemented per-uid loop with np.searchsorted for strict past-only look-back: window = [t - W, t], inclusive on lower bound, exclusive on upper bound. Handles duplicate timestamps correctly (transactions at the same t do not see each other).
- 9 new features: time_since_last_uid1 + (count, sum, mean, n_distinct_productcd) × (1h, 24h).
- Ran on full dataset (~590k rows, ~37k uids) in ~30 seconds. Saved enriched dataset to data/processed/train_transaction_features.parquet.
- NaN handling: rows with uid1=NaN have all velocity features = NaN (~11% of rows).
- Created `tests/test_velocity.py` with 6 critical tests: no_leak_basic_window, no_leak_strict_upper_bound, time_since_last_first_transaction, time_since_last_duplicate_timestamps, uid_nan_propagates, n_distinct_productcd. All 6 pass.
- Added `[tool.pytest.ini_options]` block to pyproject.toml with `pythonpath = ["."]` and `testpaths = ["tests"]` to fix module resolution. Added pytest as dev dependency.

Decisions:
- Velocity only on uid1, not uid2. uid2 will be used later for static aggregations (different signal).
- Windows: 1h and 24h only. 6h skipped (interpolation, rarely adds signal). 7d skipped (that role belongs to static aggregations).
- Manual searchsorted loop chosen over pandas rolling. Reason: absolute control over look-back, no ambiguity with duplicate timestamps, easier to test.
- Window convention: [t - W, t] — inclusive on lower bound, strict on upper. Lower-bound inclusive matches the natural-language meaning of "in the last hour"; upper-bound strict is the non-negotiable invariant (no leak from present/future).
- Oversized uids (max group 5885) NOT clipped. Flagged for v2_ideas.md.

Open questions / Notes:
- [VALIDATE-IN-BASELINE] Pearson correlations of all velocity features with isFraud are very low (max +0.039 for n_distinct_productcd_uid1_1h, most <0.01). Implementation verified correct via tests, so this is a dataset property, not a bug. Two hypotheses to revisit when training the baseline: (a) IEEE-CIS fraud is mostly one-shot CNP, not card testing/ATO, which is what velocity captures best; (b) uid1 may not group fraudulent activity tightly. Pearson is linear and LightGBM is not, so velocity may still rank well in SHAP — check feature importance when baseline is trained. If velocity is low in SHAP, document in README as a dataset limitation, not a method limitation.
- Manual inspection: top bursts (count_uid1_1h >= 5) are all isFraud=0 — looks like legitimate recurring merchants/subscriptions, not attackers. Consistent with hypothesis (a).
- Initial pytest run failed with `ModuleNotFoundError: No module named 'src'`. Fixed by configuring pytest's pythonpath in pyproject.toml. This is one-time infrastructure setup, applies to all future test files.

Next:
- Frequency encoding for high-cardinality categoricals (card1, addr1, P_emaildomain, DeviceInfo).
- Then: static aggregations per uid2 (mean/std amt, count, ratio of current_amt vs uid_mean_amt).


---

## 2026-05-24 - Session 7: Frequency encoding (src/features/encoding.py)

Done:
- Created `src/features/encoding.py` with fit_frequency_encoders, save_encoders, load_encoders, transform_frequency_encoders, encoding_diagnostics.
- Encoded 6 high-cardinality columns: card1, card2, addr1, P_emaildomain, R_emaildomain, DeviceInfo. Added `{col}_freq` columns to the dataset.
- Encoders fit ONLY on train split (filtered internally). Persisted to models/encoders/frequency_encoders.json as {col: {value_str: count}}.
- Unseen values in val/test → NaN. NaN in original → NaN in encoded.
- Saved enriched dataset back to data/processed/train_transaction_features.parquet.

Decisions:
- JSON over pickle for encoders: inspectable, no security risk, portable across Python versions, sufficient for a dict.
- Unseen-in-train values → NaN (not 0 or 1). NaN is honest about "unknown"; LightGBM handles it natively in splits.
- Keys serialized as strings (JSON limitation) and re-cast to column dtype on transform.

Open questions / Notes:
- card1 unseen-in-train rate is only 0.37%. Train captures a stable card population; concept drift on card identity is mild in this dataset.
- R_emaildomain is 75-82% NaN — structural, not a bug. Recipient email only exists for some ProductCDs (P2P, billing-to-third-party). Kept the encoding because the 25% that have it likely carries signal. If SHAP later shows it's noise, will document and possibly drop.
- NaN % is very stable across train/val/test splits → temporal split is not separating wildly different distributions. Good.
- Cold-start scenario noted: a brand-new card will always have NaN in frequency encoding until next retraining. Velocity activates from the second transaction onward (if the system has an online feature store). For this project, the demo dashboard will document this as a limitation and pre-load real test-set transactions as canonical examples to show the model at its best.

Next:
- Static aggregations per uid2 (mean/std amt, count, ratio of current vs uid average). Last feature family before baseline.

---

## 2026-05-24 - Session 8: Identity merge fix + pipeline refactor

Done:
- Discovered missing identity merge: DeviceInfo_freq was being silently skipped because train_identity was never joined to train_transaction. The `column not in df, skipping` warning in encoding had been buried in earlier output.
- Created `src/data/load_data.py` with three functions: load_transactions(), load_identity(), load_transactions_with_identity(). LEFT JOIN on TransactionID; 23.8% of transactions get identity columns, rest remain NaN.
- Refactored 5 feature modules (uid, time, velocity, split, encoding) to expose a pure `run(df) -> df` function. Original `add_*` functions and `__main__` blocks preserved — modules remain individually runnable for debugging.
- encoding.run() supports training (fit + save + transform) and serving (load + transform) modes via `encoders_path` parameter.
- Created `scripts/build_features.py`: single-command orchestrator that runs the full pipeline in memory. Total time 16.7s on full dataset.
- Final df: 590,540 rows × 455 cols (394 raw transaction + 40 identity + 6 uid/time/velocity/encoding additions − 1 shared TransactionID = 455). 95.6 MB parquet.
- Verified all 6 frequency columns present including DeviceInfo_freq (80% NaN, consistent with identity coverage).

Decisions:
- Source of truth for data loading moved from notebook cell to src/data/load_data.py. Reason: API in Week 3 will need the same merge logic; keeping it in a notebook would force re-implementation (training-serving skew risk).
- Pipeline pattern: modules are pure (df → df), orchestrator handles IO. Means modules can be reused in the API to transform single transactions without parquet round-trips.
- train_transaction_features.parquet overwritten as the single output of the pipeline. train_transaction_with_uid.parquet (older intermediate) is now stale and can be deleted.
- encoding.run() takes an optional `encoders_path` parameter — explicit two-mode design (training vs serving) anticipating Week 3 API needs.

Open questions / Notes:
- Pipeline timing breakdown: load 0.9s, uid 0.6s, time 0.0s, velocity 12.3s, split 0.0s, encoding 0.2s, save 2.6s. Velocity dominates and is the obvious target if performance ever becomes an issue (it won't for this project size).
- Identity match rate (23.8%) matches the EDA finding. Selection bias still applies: identity rows have higher fraud rate, so `has_identity` (implicit via NaN in identity columns) is itself a feature LightGBM can learn.
- DeviceInfo_freq has 80% NaN — slightly higher than the 76% predicted from "identity covers 24%". The extra 4% are rows that have identity but no DeviceInfo specifically. Not a bug.

Next:
- Static aggregations per uid2 (mean/std amt, count, ratio current vs uid average). Last feature family before baseline.
- After that: baseline LightGBM with default params, PR-AUC + P@top-K + recall@precision metrics on val.

---

## 2026-05-26 - Session 9: Static aggregations + feature validation

Done:
- Created `src/features/aggregations.py` with fit_uid_aggregations, save_aggregations, load_aggregations, transform_uid_aggregations, run, aggregations_diagnostics. Same fit/save/load/transform/run pattern as encoding.py.
- 5 new features on uid2: count_uid2, mean_amt_uid2, std_amt_uid2, amt_ratio_to_uid2_mean, amt_zscore_uid2. Aggregations fit on train only, looked up on val/test (UIDs unseen in train → NaN).
- Safe division verified: ratio/zscore are NaN (not inf) when denominator is 0 or NaN, via .where() masks plus a belt-and-suspenders inf→NaN replace.
- Persisted to models/encoders/uid2_aggregations.json.
- Added Step 7 to scripts/build_features.py (aggregations, 2.5s). Pipeline now 8 steps, 19.6s total, final shape 590,540 × 460 cols, 102 MB.
- Created notebooks/02_feature_validation.ipynb for pre-modeling feature signal checks.
- Validated aggregation signal via lift analysis (notebook 02).

Decisions:
- Aggregations on uid2 (not uid1): uid2 is the more precise identifier; uid1 already carries velocity. Static aggregations need precision over density.
- Aggregations use fit-on-train / lookup-everywhere (option B), NOT per-row expanding windows (option A). Reasons: consistent with frequency encoding, matches how production user-profile stats are precomputed and refreshed, cheaper and more predictable. Documented in module docstring.
- Kept all 5 aggregation features despite inverted/weak univariate signal (see below). Removing features before seeing model-level importance (SHAP) is premature; univariate lift misses interactions. LightGBM and SHAP will decide.

Open questions / Notes:
- [HYPOTHESIS-VALIDATED] amt_ratio_to_uid2_mean and amt_zscore_uid2 have lift < 1 against fraud, consistently: amt_ratio>3 → 0.69x, amt_ratio>10 → 0.35x, |zscore|>3 → 0.67x, |zscore|>5 → 0.71x. "Out-of-pattern" transactions are LESS fraudulent than average, not more. Reason: deviation features require an established uid history (mean_amt_uid2 must exist). One-shot CNP fraud has no prior history — those uids have count=1, mean=NaN, and are excluded from the ratio filter. The rows that remain in "out-of-pattern" are established (legitimate) users making an occasional large purchase. This INVERTS the account-takeover intuition the features were built on.
- This is the SECOND independent piece of evidence (after velocity's ~0 correlation in Session 5) that IEEE-CIS fraud is predominantly one-shot CNP without prior uid history, not card-testing bursts or account takeover. The earlier [VALIDATE-IN-BASELINE] hypothesis is now considered validated by two independent routes BEFORE training.
- Implication for modeling: the highest-importance features will likely be self-contained transaction attributes (TransactionAmt, ProductCD, Vesta's C/D/V columns, card1 frequency, identity columns) rather than history-based features (velocity, aggregations). To be confirmed with SHAP. This is a genuine analytical finding worth putting in the README — it shows the dataset was understood, not just fed to a model.

Next:
- Baseline LightGBM: define feature set (exclude isFraud, TransactionID, TransactionDT, day_index, raw uid strings), default params, no imbalance handling. Metrics on val: PR-AUC, P@top-1%, P@top-5%, recall@precision. Inspect feature importance and confirm the one-shot hypothesis via SHAP later.