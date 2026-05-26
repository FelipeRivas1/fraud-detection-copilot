"""Temporal train/val/test split.

Fraud has concept drift: attack patterns change over time, and the model must
be evaluated on data strictly later than what it was trained on. A random split
would mix future and past, inflating metrics and not reflecting deployment.

This module produces a deterministic 70/15/15 split by TransactionDT, with no
overlap and chronological order preserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"DataFrame is missing required column: '{col}'")


# ── public API ────────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    time_col: str = "TransactionDT",
) -> pd.DataFrame:
    """Return df with a new 'split' column ('train', 'val', or 'test').

    Split boundaries are determined by quantiles of time_col, not row position,
    so the original row order is preserved:
      train : time_col <= quantile(train_frac)
      val   : quantile(train_frac) < time_col <= quantile(train_frac + val_frac)
      test  : time_col > quantile(train_frac + val_frac)

    No gap between splits. In production a gap of days/weeks between train end
    and val start would better reflect the delay between model training and
    deployment, but that is omitted here for simplicity.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0, got {train_frac + val_frac:.4f}"
        )
    _require_col(df, time_col)

    t = df[time_col].dropna()
    t_train = float(np.quantile(t, train_frac))
    t_val   = float(np.quantile(t, train_frac + val_frac))

    dt = df[time_col]
    split = pd.Series("test", index=df.index, dtype=object)
    split[dt <= t_train] = "train"
    split[(dt > t_train) & (dt <= t_val)] = "val"

    return df.assign(split=split)


def run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Apply temporal split with default 70/15/15 fractions."""
    return temporal_split(df)


# ── diagnostics ───────────────────────────────────────────────────────────────

def split_diagnostics(df: pd.DataFrame) -> None:
    """Print row counts, time ranges, fraud rates, and overlap check per split."""
    _require_col(df, "split")
    _require_col(df, "TransactionDT")

    n_total = len(df)
    print(f"\n{'=' * 60}")
    print("  Temporal Split Diagnostics")
    print(f"{'=' * 60}")

    for label in ("train", "val", "test"):
        sub = df[df["split"] == label]
        n = len(sub)
        dt_min = sub["TransactionDT"].min()
        dt_max = sub["TransactionDT"].max()
        span_days = (dt_max - dt_min) / 86400

        print(f"\n  [{label}]")
        print(f"    rows          : {n:,}  ({n / n_total * 100:.1f}%)")
        print(f"    TransactionDT : {dt_min:,.0f}  →  {dt_max:,.0f}")
        print(f"    span          : {span_days:.1f} days")

        if "day_index" in df.columns:
            print(f"    day_index     : {sub['day_index'].min()}  →  {sub['day_index'].max()}")

        if "isFraud" in df.columns:
            fraud_rate = sub["isFraud"].mean()
            n_fraud = int(sub["isFraud"].sum())
            print(f"    fraud rate    : {fraud_rate:.4f}  ({n_fraud:,} frauds)")

    # Chronological overlap check
    train_max = df.loc[df["split"] == "train", "TransactionDT"].max()
    val_min   = df.loc[df["split"] == "val",   "TransactionDT"].min()
    val_max   = df.loc[df["split"] == "val",   "TransactionDT"].max()
    test_min  = df.loc[df["split"] == "test",  "TransactionDT"].min()

    print(f"\n  Overlap check:")
    if train_max < val_min:
        print(f"    ✓ train/val  : train ends {train_max:,.0f}, val starts {val_min:,.0f}")
    else:
        print(f"    WARNING train/val: train_max ({train_max}) >= val_min ({val_min})")

    if val_max < test_min:
        print(f"    ✓ val/test   : val ends {val_max:,.0f}, test starts {test_min:,.0f}")
    else:
        print(f"    WARNING val/test: val_max ({val_max}) >= test_min ({test_min})")


# ── script entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROC_DIR = Path(__file__).parents[2] / "data" / "processed"
    src_path = PROC_DIR / "train_transaction_features.parquet"

    print(f"Loading {src_path} ...")
    df = pd.read_parquet(src_path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df = temporal_split(df)
    split_diagnostics(df)

    df.to_parquet(src_path, index=False)
    size_mb = src_path.stat().st_size / 1_048_576
    print(f"\nSaved → {src_path}")
    print(f"File size: {size_mb:.1f} MB")
    print("Done.")