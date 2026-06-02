"""Baseline LightGBM training script.

Run from the repo root:
    python scripts/train_baseline.py

Produces:
    models/baseline_lgbm.pkl       — trained booster (joblib)
    models/baseline_metrics.json   — evaluation metrics + training metadata
    models/encoders/categorical_mappings.json — frozen category lists (fit here)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb

# Allow imports from src/ when run as a script from the repo root
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.dataset import (
    CATEGORICAL_COLS,
    fit_categorical_mappings,
    load_features,
    split_xy,
)
from src.models.evaluate import evaluate_all


# ── hyperparameters ───────────────────────────────────────────────────────────

PARAMS = {
    "objective": "binary",
    "metric": "average_precision",   # PR-AUC nativo, no ROC-AUC
    "learning_rate": 0.05,           # bajo + early stopping
    "num_leaves": 63,                # 2^6 - 1, conservador (LightGBM crece leaf-wise)
    "min_data_in_leaf": 100,         # evita hojas que memoricen fraudes individuales
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    # SIN scale_pos_weight, SIN is_unbalance: baseline limpio
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50
LOG_EVERY = 100

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parents[1]
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "train_transaction_features.parquet"
MAPPINGS_PATH = REPO_ROOT / "models" / "encoders" / "categorical_mappings.json"
MODEL_PATH = REPO_ROOT / "models" / "baseline_lgbm.pkl"
METRICS_PATH = REPO_ROOT / "models" / "baseline_metrics.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_recall_row(label: str, result: dict) -> str:
    if result["threshold"] is None:
        return f"  {label:<22}{'N/A':>6}  (threshold=N/A, 0 flagged)"
    return (
        f"  {label:<22}{result['recall']:>6.4f}"
        f"  (threshold={result['threshold']:.4f}, {result['n_flagged']} flagged)"
    )


def _print_split_report(name: str, metrics: dict) -> None:
    n = metrics["n_samples"]
    base_rate = metrics["base_rate"] * 100
    pak = metrics["precision_at_k"]
    rap = metrics["recall_at_precision"]

    print(f"\n{name} (n={n:,}, fraud rate={base_rate:.2f}%):")
    print(f"  {'PR-AUC:':<22}{metrics['pr_auc']:>6.4f}")
    for k in ("100", "500", "1000", "5000"):
        print(f"  {f'P@{k}:':<22}{pak[k]:>6.4f}")
    print(_fmt_recall_row("Recall@P=0.50:", rap["0.5"]))
    print(_fmt_recall_row("Recall@P=0.70:", rap["0.7"]))
    print(_fmt_recall_row("Recall@P=0.90:", rap["0.9"]))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load features ─────────────────────────────────────────────────
    print("\n=== Step 1: Load features ===")
    t0 = time.perf_counter()
    df = load_features(PARQUET_PATH)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Step 1 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 2: Fit categorical mappings ──────────────────────────────────────
    print("\n=== Step 2: Fit categorical mappings ===")
    t0 = time.perf_counter()
    train_df = df[df["split"] == "train"]
    fit_categorical_mappings(train_df, CATEGORICAL_COLS, MAPPINGS_PATH)
    print(f"Mappings saved → {MAPPINGS_PATH}")
    print(f"Step 2 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 3: Split X/y ─────────────────────────────────────────────────────
    print("\n=== Step 3: Split X/y ===")
    t0 = time.perf_counter()
    X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols = split_xy(
        df, MAPPINGS_PATH
    )
    print(f"Step 3 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 4: Train LightGBM ────────────────────────────────────────────────
    print("\n=== Step 4: Train LightGBM ===")
    t0 = time.perf_counter()

    train_set = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, categorical_feature=categorical_cols,
        reference=train_set, free_raw_data=False
    )

    booster = lgb.train(
        PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(LOG_EVERY),
        ],
    )
    print(f"Step 4 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 5: Predict ───────────────────────────────────────────────────────
    print("\n=== Step 5: Predict ===")
    t0 = time.perf_counter()
    y_val_score = booster.predict(X_val, num_iteration=booster.best_iteration)
    y_test_score = booster.predict(X_test, num_iteration=booster.best_iteration)
    print(f"Step 5 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 6: Evaluate ──────────────────────────────────────────────────────
    print("\n=== Step 6: Evaluate ===")
    t0 = time.perf_counter()
    val_metrics = evaluate_all(y_val, y_val_score)
    test_metrics = evaluate_all(y_test, y_test_score)

    print("\n" + "=" * 60)
    print("BASELINE LIGHTGBM — REPORT")
    print("=" * 60)
    print(f"Best iteration: {booster.best_iteration}")
    print(f"Features: {X_train.shape[1]} ({len(categorical_cols)} categorical)")
    _print_split_report("VAL SET", val_metrics)
    _print_split_report("TEST SET", test_metrics)
    print("=" * 60)

    print(f"Step 6 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 7: Save artifacts ────────────────────────────────────────────────
    print("\n=== Step 7: Save artifacts ===")
    t0 = time.perf_counter()

    joblib.dump(booster, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    metrics_doc = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_iteration": booster.best_iteration,
        "n_features": X_train.shape[1],
        "categorical_features": categorical_cols,
        "params": PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "val": val_metrics,
        "test": test_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_doc, f, indent=2, sort_keys=True)
    print(f"Metrics saved → {METRICS_PATH}")

    print(f"Step 7 done in {time.perf_counter() - t0:.1f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
