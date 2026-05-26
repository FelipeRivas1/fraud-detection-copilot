"""Data loading and merging for the IEEE-CIS Fraud Detection dataset.

Single source of truth for how transactions and identity are loaded and joined.
Used by feature engineering scripts, the API, the dashboard, and tests — any
divergence in how data is loaded would create training-serving skew.

Transactions and identity are stored as separate parquet files. A LEFT JOIN on
TransactionID merges identity into transactions: ~24% of rows get identity
columns filled, the rest remain NaN (which is informative — `has_identity` is
itself a signal).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ── path constants ────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT / "data" / "raw" / "ieee-fraud-detection"
PROC_DIR = ROOT / "data" / "processed"

TRANS_CSV = RAW_DIR / "train_transaction.csv"
IDENT_CSV = RAW_DIR / "train_identity.csv"
TRANS_PAR = PROC_DIR / "train_transaction.parquet"
IDENT_PAR = PROC_DIR / "train_identity.parquet"


# ── public API ────────────────────────────────────────────────────────────────

def load_transactions() -> pd.DataFrame:
    """Load train_transaction as a DataFrame.

    Prefers the parquet version if it exists. Falls back to CSV, saving a
    parquet copy for future calls. Raises FileNotFoundError if neither exists.
    """
    if TRANS_PAR.exists():
        return pd.read_parquet(TRANS_PAR)

    if TRANS_CSV.exists():
        df = pd.read_csv(TRANS_CSV)
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(TRANS_PAR, index=False)
        return df

    raise FileNotFoundError(
        f"Transaction data not found. Expected parquet at:\n  {TRANS_PAR}\n"
        f"or CSV at:\n  {TRANS_CSV}\n"
        "Download the dataset from Kaggle: ieee-fraud-detection."
    )


def load_identity() -> pd.DataFrame:
    """Load train_identity as a DataFrame.

    Prefers the parquet version if it exists. Falls back to CSV, saving a
    parquet copy for future calls. Raises FileNotFoundError if neither exists.
    """
    if IDENT_PAR.exists():
        return pd.read_parquet(IDENT_PAR)

    if IDENT_CSV.exists():
        df = pd.read_csv(IDENT_CSV)
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(IDENT_PAR, index=False)
        return df

    raise FileNotFoundError(
        f"Identity data not found. Expected parquet at:\n  {IDENT_PAR}\n"
        f"or CSV at:\n  {IDENT_CSV}\n"
        "Download the dataset from Kaggle: ieee-fraud-detection."
    )


def load_transactions_with_identity() -> pd.DataFrame:
    """Return transactions LEFT JOINed with identity on TransactionID.

    Rows without a match in identity keep NaN for all identity columns.
    The row count equals the transaction table (no rows added or dropped).
    """
    trans = load_transactions()
    ident = load_identity()

    n_trans = len(trans)
    cols_before = trans.shape[1]

    merged = pd.merge(trans, ident, on="TransactionID", how="left")

    n_matched = merged["DeviceType"].notna().sum() if "DeviceType" in merged.columns else (
        merged[ident.columns.difference(["TransactionID"])].notna().any(axis=1).sum()
    )
    pct_matched = n_matched / n_trans * 100

    print(f"Transactions : {n_trans:,} rows")
    print(f"Identity match: {n_matched:,}  ({pct_matched:.1f}% of transactions)")
    print(f"Columns: {cols_before} → {merged.shape[1]}  (+{merged.shape[1] - cols_before} identity cols)")

    return merged
