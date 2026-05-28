"""Static aggregations per UID.

For each uid2, compute aggregate statistics over its full history (count, mean,
std of TransactionAmt). Then derive per-row features comparing the current
transaction against those stats (ratio and z-score of amount).

The amount-vs-history comparison is the actual signal: a transaction at 10x
the user's historical mean is suspicious in a way that a single number alone
cannot capture.

Critical invariant: aggregations are fit ONLY on the training set. Applying
them to val/test (and in production to new transactions) is a lookup, not a
recomputation. UIDs unseen in train get NaN. Same pattern as frequency
encoding — these are training artifacts that get serialized for serving.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ── path defaults ─────────────────────────────────────────────────────────────

_DEFAULT_AGGS_PATH = (
    Path(__file__).parents[2] / "models" / "encoders" / "uid2_aggregations.json"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _nan_to_none(val: float) -> float | None:
    """Convert NaN to None for JSON serialization (JSON null)."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return val


def _none_to_nan(val) -> float:
    """Convert None (JSON null) back to NaN on load."""
    return float("nan") if val is None else float(val)


# ── public API ────────────────────────────────────────────────────────────────

def fit_uid_aggregations(
    df: pd.DataFrame,
    uid_col: str = "uid2",
    amt_col: str = "TransactionAmt",
) -> dict[str, dict]:
    """Fit uid aggregations from training rows only.

    Returns {uid_value: {"count": int, "mean": float, "std": float}}.
    NaN uids are excluded. std uses ddof=1 (pandas default); count=1 → std=NaN.
    """
    _require_cols(df, ["split", uid_col, amt_col])

    train = df[df["split"] == "train"]
    grp = (
        train.dropna(subset=[uid_col])
        .groupby(uid_col)[amt_col]
        .agg(count="count", mean="mean", std="std")
    )

    return {
        str(uid): {
            "count": int(row["count"]),
            "mean":  float(row["mean"]),
            "std":   float(row["std"]) if not math.isnan(float(row["std"])) else float("nan"),
        }
        for uid, row in grp.iterrows()
    }


def save_aggregations(aggs: dict[str, dict], path: str | Path) -> None:
    """Serialize aggregations to JSON. NaN floats are stored as null (JSON null)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        uid: {
            "count": entry["count"],
            "mean":  _nan_to_none(entry["mean"]),
            "std":   _nan_to_none(entry["std"]),
        }
        for uid, entry in aggs.items()
    }
    with path.open("w") as f:
        json.dump(serializable, f)


def load_aggregations(path: str | Path) -> dict[str, dict]:
    """Load aggregations from JSON. null values are converted back to NaN."""
    with Path(path).open() as f:
        raw = json.load(f)

    return {
        uid: {
            "count": int(entry["count"]),
            "mean":  _none_to_nan(entry["mean"]),
            "std":   _none_to_nan(entry["std"]),
        }
        for uid, entry in raw.items()
    }


def transform_uid_aggregations(
    df: pd.DataFrame,
    aggs: dict[str, dict],
    uid_col: str = "uid2",
    amt_col: str = "TransactionAmt",
) -> pd.DataFrame:
    """Add 5 aggregation-derived columns for each row.

    Columns added:
      count_uid2, mean_amt_uid2, std_amt_uid2,
      amt_ratio_to_uid2_mean, amt_zscore_uid2

    uid not in aggs or uid NaN → NaN for all five columns.
    Ratio/zscore are NaN when the denominator is 0 or NaN (no inf produced).
    """
    _require_cols(df, [uid_col, amt_col])

    uid_series = df[uid_col].astype(object)  # uid2 is already str/object

    count_vals = uid_series.map(lambda u: aggs[u]["count"] if u in aggs else float("nan"))
    mean_vals  = uid_series.map(lambda u: aggs[u]["mean"]  if u in aggs else float("nan"))
    std_vals   = uid_series.map(lambda u: aggs[u]["std"]   if u in aggs else float("nan"))

    amt = df[amt_col].astype(float)

    # Safe ratio: NaN when mean is 0 or NaN
    safe_mean = mean_vals.where(mean_vals.notna() & (mean_vals != 0.0))
    amt_ratio = amt / safe_mean

    # Safe zscore: NaN when std is 0 or NaN
    safe_std = std_vals.where(std_vals.notna() & (std_vals != 0.0))
    amt_zscore = (amt - mean_vals) / safe_std

    # Guard: replace any stray inf (shouldn't happen with safe denominators, but belt-and-suspenders)
    amt_ratio  = amt_ratio.replace([np.inf, -np.inf], np.nan)
    amt_zscore = amt_zscore.replace([np.inf, -np.inf], np.nan)

    return df.assign(
        count_uid2             = count_vals.astype("float64"),
        mean_amt_uid2          = mean_vals.astype("float64"),
        std_amt_uid2           = std_vals.astype("float64"),
        amt_ratio_to_uid2_mean = amt_ratio.astype("float64"),
        amt_zscore_uid2        = amt_zscore.astype("float64"),
    )


def run(
    df: pd.DataFrame,
    aggs_path: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Apply uid aggregations.

    Training mode (aggs_path=None): fit on train split, save to
    models/encoders/uid2_aggregations.json, transform full df.
    Serving mode (aggs_path provided): load from disk, transform only.
    """
    if aggs_path is None:
        aggs = fit_uid_aggregations(df)
        save_aggregations(aggs, _DEFAULT_AGGS_PATH)
    else:
        aggs = load_aggregations(Path(aggs_path))

    return transform_uid_aggregations(df, aggs)


# ── diagnostics ───────────────────────────────────────────────────────────────

_AGG_FEATURES = [
    "count_uid2",
    "mean_amt_uid2",
    "std_amt_uid2",
    "amt_ratio_to_uid2_mean",
    "amt_zscore_uid2",
]


def aggregations_diagnostics(df: pd.DataFrame) -> None:
    """Print stats and anomaly counts for uid2 aggregation features."""
    present = [c for c in _AGG_FEATURES if c in df.columns]
    n_total = len(df)

    print(f"\n{'=' * 60}")
    print("  UID2 Aggregation Diagnostics")
    print(f"{'=' * 60}")

    for col in present:
        s = df[col]
        n_nan = s.isna().sum()
        n_inf = np.isinf(s.dropna()).sum()

        print(f"\n  {col}  (non-NaN: {n_total - n_nan:,} / {n_total:,},  NaN: {n_nan / n_total * 100:.1f}%)")

        if n_inf > 0:
            print(f"    WARNING: {n_inf} inf values detected!")

        s_clean = s.dropna()
        if len(s_clean) > 0:
            desc = s_clean.describe(percentiles=[0.25, 0.50, 0.75, 0.99])
            for stat, val in desc.items():
                print(f"    {stat:>6} : {val:.4f}")

    # Anomaly counts for ratio and zscore
    ratio_col  = "amt_ratio_to_uid2_mean"
    zscore_col = "amt_zscore_uid2"

    if ratio_col in df.columns:
        print(f"\n  {ratio_col} anomaly thresholds:")
        for threshold in (3, 10):
            mask = df[ratio_col] > threshold
            n = mask.sum()
            pct = n / n_total * 100
            print(f"    > {threshold:<3}: {n:,}  ({pct:.2f}%)", end="")
            if "isFraud" in df.columns and n > 0:
                fraud_rate = df.loc[mask, "isFraud"].mean()
                global_rate = df["isFraud"].mean()
                print(f"  |  fraud rate: {fraud_rate:.3f} vs global {global_rate:.3f}", end="")
            print()

    if zscore_col in df.columns:
        print(f"\n  |{zscore_col}| anomaly thresholds:")
        mask_any = df[zscore_col].abs() > 3
        n = mask_any.sum()
        pct = n / n_total * 100
        print(f"    |z| > 3: {n:,}  ({pct:.2f}%)", end="")
        if "isFraud" in df.columns and n > 0:
            fraud_rate = df.loc[mask_any, "isFraud"].mean()
            global_rate = df["isFraud"].mean()
            print(f"  |  fraud rate: {fraud_rate:.3f} vs global {global_rate:.3f}", end="")
        print()
