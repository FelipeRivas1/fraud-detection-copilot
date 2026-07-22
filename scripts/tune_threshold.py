"""Cost-based threshold tuning (Bahnsen, Aouada & Ottersten, 2014).

Run from the repo root, after scripts/train_baseline.py:
    python scripts/tune_threshold.py

Finds the score threshold that minimizes total expected cost on val, then
applies that SAME fixed threshold to test (no re-optimizing on test — that
would leak the test set into the decision).

Produces:
    models/threshold.json — chosen threshold + val/test cost reports.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib

# Allow imports from src/ when run as a script from the repo root
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.models.cost import evaluate_amount_aware, evaluate_at_threshold, find_optimal_threshold
from src.models.dataset import load_features, split_xy


# ── config ────────────────────────────────────────────────────────────────────

ADMIN_COST = 5.0   # fixed cost of one manual review, whether it's a TP or FP

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parents[1]
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "train_transaction_features.parquet"
MAPPINGS_PATH = REPO_ROOT / "models" / "encoders" / "categorical_mappings.json"
MODEL_PATH = REPO_ROOT / "models" / "baseline_lgbm.pkl"
THRESHOLD_PATH = REPO_ROOT / "models" / "threshold.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_report(name: str, result: dict) -> None:
    print(f"\n{name}:")
    print(f"  {'Threshold:':<16}{result['threshold']:>10.4f}")
    print(f"  {'Cost:':<16}{result['cost']:>10.2f}")
    print(f"  {'Baseline cost:':<16}{result['baseline_cost']:>10.2f}")
    print(f"  {'Savings:':<16}{result['savings_pct']:>10.2%}")
    print(f"  {'N flagged:':<16}{result['n_flagged']:>10,}")
    print(f"  {'Precision:':<16}{result['precision']:>10.4f}")
    print(f"  {'Recall:':<16}{result['recall']:>10.4f}")


def _print_comparison(name: str, global_result: dict, amount_aware_result: dict) -> None:
    print(f"\n{name} — global threshold vs. amount-aware:")
    print(f"  {'Metric':<12}{'Global':>14}{'Amount-aware':>16}")
    print(f"  {'Cost:':<12}{global_result['cost']:>14.2f}{amount_aware_result['cost']:>16.2f}")
    print(f"  {'Savings:':<12}{global_result['savings_pct']:>14.2%}{amount_aware_result['savings_pct']:>16.2%}")
    print(f"  {'N flagged:':<12}{global_result['n_flagged']:>14,}{amount_aware_result['n_flagged']:>16,}")
    print(f"  {'Precision:':<12}{global_result['precision']:>14.4f}{amount_aware_result['precision']:>16.4f}")
    print(f"  {'Recall:':<12}{global_result['recall']:>14.4f}{amount_aware_result['recall']:>16.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Step 1: Load model ────────────────────────────────────────────────────
    print("\n=== Step 1: Load model ===")
    t0 = time.perf_counter()
    booster = joblib.load(MODEL_PATH)
    print(f"Model loaded ← {MODEL_PATH}")
    print(f"Step 1 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 2: Load features + split ─────────────────────────────────────────
    print("\n=== Step 2: Load features + split ===")
    t0 = time.perf_counter()
    df = load_features(PARQUET_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols = split_xy(
        df, MAPPINGS_PATH
    )
    print(f"Step 2 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 3: Predict scores ────────────────────────────────────────────────
    print("\n=== Step 3: Predict scores ===")
    t0 = time.perf_counter()
    score_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    score_test = booster.predict(X_test, num_iteration=booster.best_iteration)
    print(f"Step 3 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 4: Extract amounts ────────────────────────────────────────────────
    print("\n=== Step 4: Extract amounts ===")
    t0 = time.perf_counter()
    amounts_val = X_val["TransactionAmt"]
    amounts_test = X_test["TransactionAmt"]
    print(f"Step 4 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 5: Find optimal threshold on val ─────────────────────────────────
    print("\n=== Step 5: Find optimal threshold on val ===")
    t0 = time.perf_counter()
    val_result = find_optimal_threshold(y_val, amounts_val, score_val, admin_cost=ADMIN_COST)
    threshold = val_result["threshold"]
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"Step 5 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 6: Apply fixed threshold to test ─────────────────────────────────
    print("\n=== Step 6: Apply fixed threshold to test ===")
    t0 = time.perf_counter()
    test_result = evaluate_at_threshold(
        y_test, amounts_test, score_test, threshold=threshold, admin_cost=ADMIN_COST
    )
    print(f"Step 6 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 7: Amount-aware rule ─────────────────────────────────────────────
    print("\n=== Step 7: Amount-aware rule ===")
    t0 = time.perf_counter()
    val_amount_aware = evaluate_amount_aware(y_val, amounts_val, score_val, admin_cost=ADMIN_COST)
    test_amount_aware = evaluate_amount_aware(y_test, amounts_test, score_test, admin_cost=ADMIN_COST)
    print(f"Step 7 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 8: Report ────────────────────────────────────────────────────────
    print("\n=== Step 8: Report ===")
    t0 = time.perf_counter()
    print("\n" + "=" * 60)
    print("COST-BASED THRESHOLD TUNING — REPORT")
    print("=" * 60)
    print(f"Admin cost per review: {ADMIN_COST}")
    _print_report("VAL SET (optimized)", val_result)
    _print_report("TEST SET (fixed threshold from val)", test_result)
    _print_comparison("VAL SET", val_result, val_amount_aware)
    _print_comparison("TEST SET", test_result, test_amount_aware)
    print("=" * 60)
    print(f"Step 8 done in {time.perf_counter() - t0:.1f}s")

    # ── Step 9: Save artifacts ─────────────────────────────────────────────────
    print("\n=== Step 9: Save artifacts ===")
    t0 = time.perf_counter()
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)

    threshold_doc = {
        "admin_cost": ADMIN_COST,
        "threshold": threshold,
        "val": val_result,
        "test": test_result,
        "amount_aware": {
            "val": val_amount_aware,
            "test": test_amount_aware,
        },
    }
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(threshold_doc, f, indent=2, sort_keys=True)
    print(f"Threshold doc saved → {THRESHOLD_PATH}")
    print(f"Step 9 done in {time.perf_counter() - t0:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
