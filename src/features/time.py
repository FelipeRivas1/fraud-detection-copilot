"""Time-based features derived from TransactionDT.

TransactionDT is an offset in seconds from an unknown base date (Vesta did not
disclose it). Absolute dates are not recoverable, but hour-of-day, day-of-week
(relative), and day-index from dataset start are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"DataFrame is missing required column: '{col}'")


def _seconds_to_int(series: pd.Series, divisor: int, modulo: int | None = None) -> pd.Series:
    """Divide a seconds series by divisor, optionally mod, returning int64 or nullable Int64."""
    result = series // divisor
    if modulo is not None:
        result = result % modulo
    if result.isna().any():
        return result.astype("Int64")
    return result.astype("int64")


# ── public API ────────────────────────────────────────────────────────────────

def add_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a new 'hour' column (int 0-23).

    Computed as (TransactionDT % 86400) // 3600.
    NaN rows in TransactionDT produce NaN in 'hour'.
    """
    _require_col(df, "TransactionDT")
    hour = _seconds_to_int(df["TransactionDT"] % 86400, divisor=3600)
    return df.assign(hour=hour)


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a new 'day_of_week' column (int 0-6).

    Computed as (TransactionDT // 86400) % 7. Because the dataset base date is
    unknown, day 0 is not necessarily Monday — this is a *relative* day-of-week
    from dataset start, useful as a cyclical pattern feature but not as a
    calendar weekday label.
    NaN rows in TransactionDT produce NaN in 'day_of_week'.
    """
    _require_col(df, "TransactionDT")
    dow = _seconds_to_int(df["TransactionDT"], divisor=86400, modulo=7)
    return df.assign(day_of_week=dow)


def add_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a new 'day_index' column (int, days since dataset start).

    Computed as TransactionDT // 86400.

    HELPER COLUMN — NOT A MODEL FEATURE. Use this for temporal train/test
    splits and as an input to velocity aggregations. Including it as a raw
    model feature would leak position-in-time information (the model would
    learn "later = less fraud" from the dataset construction, not from
    genuine signal).
    NaN rows in TransactionDT produce NaN in 'day_index'.
    """
    _require_col(df, "TransactionDT")
    day_idx = _seconds_to_int(df["TransactionDT"], divisor=86400)
    return df.assign(day_index=day_idx)


def add_all_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply add_hour, add_day_of_week, and add_day_index in order."""
    df = add_hour(df)
    df = add_day_of_week(df)
    df = add_day_index(df)
    return df


# ── diagnostics ───────────────────────────────────────────────────────────────

def time_diagnostics(df: pd.DataFrame) -> None:
    """Print temporal coverage and distribution stats for the time features."""
    _require_col(df, "TransactionDT")

    dt = df["TransactionDT"]
    span_days = (dt.max() - dt.min()) / 86400

    print(f"\n{'=' * 60}")
    print("  Time Feature Diagnostics")
    print(f"{'=' * 60}")
    print(f"  TransactionDT range : {dt.min():,.0f}  →  {dt.max():,.0f}")
    print(f"  Span                : {span_days:.1f} days")

    if "day_index" in df.columns:
        di = df["day_index"]
        print(f"\n  day_index range     : {di.min()}  →  {di.max()}")

    if "hour" in df.columns:
        print("\n  hour distribution (sorted by hour):")
        vc = df["hour"].value_counts().sort_index()
        for h, cnt in vc.items():
            bar = "█" * (cnt * 40 // vc.max())
            print(f"    {int(h):02d}h  {cnt:>7,}  {bar}")

    if "day_of_week" in df.columns:
        print("\n  day_of_week distribution (sorted by day):")
        vc = df["day_of_week"].value_counts().sort_index()
        for d, cnt in vc.items():
            bar = "█" * (cnt * 40 // vc.max())
            print(f"    day {int(d)}  {cnt:>7,}  {bar}")

    if "day_index" in df.columns:
        print("\n  Transactions per day (stats):")
        per_day = df.groupby("day_index").size()
        desc = per_day.describe()
        for stat, val in desc.items():
            print(f"    {stat:>6} : {val:.1f}")


# ── script entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROC_DIR = Path(__file__).parents[2] / "data" / "processed"
    src_path = PROC_DIR / "train_transaction_with_uid.parquet"
    dst_path = PROC_DIR / "train_transaction_features.parquet"

    print(f"Loading {src_path} ...")
    df = pd.read_parquet(src_path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df = add_all_time_features(df)

    time_diagnostics(df)

    df.to_parquet(dst_path, index=False)
    size_mb = dst_path.stat().st_size / 1_048_576
    print(f"\nSaved → {dst_path}")
    print(f"File size: {size_mb:.1f} MB")
    print("Done.")
