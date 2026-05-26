"""
Synthetic user-identifier features for the IEEE-CIS fraud dataset.

uid1  = card1 + addr1
uid2  = card1 + addr1 + day_of_card  (where day_of_card = floor(TransactionDT/86400) - D1)

Both identifiers are NaN whenever any component is NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _safe_int_str(series: pd.Series) -> pd.Series:
    """Convert a numeric series to integer strings; NaN stays NaN."""
    result = pd.Series(np.nan, index=series.index, dtype=object)
    mask = series.notna()
    result[mask] = series[mask].astype(int).astype(str)
    return result


# ── public API ────────────────────────────────────────────────────────────────

def add_uid1(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a new 'uid1' column = card1 + addr1.

    NaN if either component is NaN.
    """
    _require_cols(df, ["card1", "addr1"])

    mask = df["card1"].notna() & df["addr1"].notna()
    uid1 = pd.Series(np.nan, index=df.index, dtype=object)
    uid1[mask] = (
        _safe_int_str(df["card1"])[mask]
        + "_"
        + _safe_int_str(df["addr1"])[mask]
    )
    return df.assign(uid1=uid1)


def add_uid2(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a new 'uid2' column = card1 + addr1 + day_of_card.

    day_of_card = floor(TransactionDT / 86400) - D1
    NaN if any component (card1, addr1, D1) is NaN.
    """
    _require_cols(df, ["card1", "addr1", "TransactionDT", "D1"])

    day_of_card = (df["TransactionDT"] // 86400) - df["D1"]
    mask = df["card1"].notna() & df["addr1"].notna() & day_of_card.notna()

    uid2 = pd.Series(np.nan, index=df.index, dtype=object)
    uid2[mask] = (
        _safe_int_str(df["card1"])[mask]
        + "_"
        + _safe_int_str(df["addr1"])[mask]
        + "_"
        + _safe_int_str(day_of_card)[mask]
    )
    return df.assign(uid2=uid2)


def run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Apply all uid transformations: add_uid1 then add_uid2."""
    df = add_uid1(df)
    df = add_uid2(df)
    return df


# ── diagnostics ───────────────────────────────────────────────────────────────

def uid_diagnostics(df: pd.DataFrame, uid_col: str) -> None:
    """Print coverage, cardinality, group-size distribution, and fraud stats for uid_col."""
    _require_cols(df, [uid_col])

    uid = df[uid_col]
    n_total  = len(uid)
    n_nan    = uid.isna().sum()
    pct_nan  = n_nan / n_total * 100
    n_unique = uid.dropna().nunique()

    print(f"\n{'=' * 60}")
    print(f"  Diagnostics for: {uid_col}")
    print(f"{'=' * 60}")
    print(f"  Total rows      : {n_total:,}")
    print(f"  NaN             : {n_nan:,}  ({pct_nan:.1f}%)")
    print(f"  Unique UIDs     : {n_unique:,}")

    # Group-size distribution
    group_sizes = df.dropna(subset=[uid_col]).groupby(uid_col).size()
    print(f"\n  Group size distribution (transactions per uid):")
    desc = group_sizes.describe(percentiles=[.25, .5, .75, .9, .95, .99])
    for stat, val in desc.items():
        print(f"    {stat:>6} : {val:.2f}")

    # Fraud rate analysis
    if "isFraud" in df.columns:
        global_rate = df["isFraud"].mean()
        print(f"\n  Fraud rate (global)         : {global_rate:.2%}")

        uid_fraud = (
            df.dropna(subset=[uid_col])
            .groupby(uid_col)
            .agg(n=("isFraud", "count"), fraud_rate=("isFraud", "mean"))
        )
        uid_fraud_multi = uid_fraud[uid_fraud["n"] >= 2]["fraud_rate"]
        print(f"  Fraud rate per uid (n>=2), describe:")
        desc_fr = uid_fraud_multi.describe(percentiles=[.25, .5, .75, .9, .95, .99])
        for stat, val in desc_fr.items():
            print(f"    {stat:>6} : {val:.4f}")

    # Random examples with >=3 transactions
    multi_uids = group_sizes[group_sizes >= 3].index
    if len(multi_uids) == 0:
        print("\n  No UIDs with >=3 transactions found.")
        return

    sample_uids = (
        pd.Series(multi_uids).sample(min(5, len(multi_uids)), random_state=42).tolist()
    )
    show_cols = [c for c in ["TransactionDT", "TransactionAmt", "ProductCD", "isFraud"] if c in df.columns]
    print(f"\n  5 random UIDs with >=3 transactions:")
    for uid_val in sample_uids:
        rows = df[df[uid_col] == uid_val][show_cols].sort_values("TransactionDT")
        print(f"\n  uid = {uid_val}  (n={len(rows)})")
        print(rows.to_string(index=False))


# ── script entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROC_DIR = Path(__file__).parents[2] / "data" / "processed"
    src_path = PROC_DIR / "train_transaction.parquet"
    dst_path = PROC_DIR / "train_transaction_with_uid.parquet"

    print(f"Loading {src_path} ...")
    df = pd.read_parquet(src_path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df = add_uid1(df)
    df = add_uid2(df)

    uid_diagnostics(df, "uid1")
    uid_diagnostics(df, "uid2")

    df.to_parquet(dst_path, index=False)
    size_mb = dst_path.stat().st_size / 1_048_576
    print(f"\nSaved → {dst_path}")
    print(f"File size: {size_mb:.1f} MB")
    print("Done.")
