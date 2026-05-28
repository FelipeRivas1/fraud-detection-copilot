"""End-to-end feature pipeline for the IEEE-CIS fraud detection dataset.

Loads transactions + identity, applies all feature modules in order, and saves
the enriched DataFrame to data/processed/train_transaction_features.parquet.

Usage (from repo root):
    python scripts/build_features.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make src/ importable when running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_data
from src.features import aggregations, encoding, split, uid
from src.features import time as time_features
from src.features import velocity

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_transaction_features.parquet"


def _step(n: int, name: str) -> float:
    print(f"\n{'=' * 60}")
    print(f"  Step {n}: {name}")
    print(f"{'=' * 60}")
    return time.perf_counter()


def _done(t0: float, df_shape: tuple) -> None:
    elapsed = time.perf_counter() - t0
    print(f"  → done in {elapsed:.1f}s  |  shape: {df_shape[0]:,} rows x {df_shape[1]} cols")


def main() -> None:
    pipeline_start = time.perf_counter()

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    t0 = _step(1, "Load transactions + identity")
    df = load_data.load_transactions_with_identity()
    _done(t0, df.shape)

    # ── Step 2: UIDs ─────────────────────────────────────────────────────────
    t0 = _step(2, "Synthetic UIDs (uid1, uid2)")
    df = uid.run(df)
    _done(t0, df.shape)

    # ── Step 3: Time features ─────────────────────────────────────────────────
    t0 = _step(3, "Time features (hour, day_of_week, day_index)")
    df = time_features.run(df)
    _done(t0, df.shape)

    # ── Step 4: Velocity features ─────────────────────────────────────────────
    t0 = _step(4, "Velocity features (time_since_last + 1h/24h windows)")
    df = velocity.run(df)
    _done(t0, df.shape)

    # ── Step 5: Temporal split ────────────────────────────────────────────────
    t0 = _step(5, "Temporal split (70/15/15 by TransactionDT)")
    df = split.run(df)
    split_counts = df["split"].value_counts()
    for label in ("train", "val", "test"):
        print(f"  {label}: {split_counts.get(label, 0):,} rows")
    _done(t0, df.shape)

    # ── Step 6: Frequency encoding ────────────────────────────────────────────
    t0 = _step(6, "Frequency encoding (fit on train, transform all)")
    df = encoding.run(df)
    _done(t0, df.shape)

    # ── Step 7: Static aggregations ───────────────────────────────────────────
    t0 = _step(7, "Static aggregations (uid2 count/mean/std + ratio/zscore)")
    df = aggregations.run(df)
    _done(t0, df.shape)

    # ── Step 8: Save ──────────────────────────────────────────────────────────
    t0 = _step(8, f"Save → {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"  File size: {size_mb:.1f} MB")
    _done(t0, df.shape)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete")
    print(f"  Total time : {total:.1f}s")
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Output     : {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
