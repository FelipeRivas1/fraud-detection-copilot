"""Velocity features: rolling-window aggregations per UID.

Velocity captures the rate of activity for a synthetic user (uid) over a past
time window. Standard signals against card testing, account takeover, and
cashout bursts.

Critical invariant: look-back is STRICTLY past. For a transaction at time t,
the window includes events with timestamp in [t - window, t). Never t itself,
never anything after t. Violating this leaks the future and inflates metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _window_suffix(window_seconds: int) -> str:
    if window_seconds == 3600:
        return "1h"
    if window_seconds == 86400:
        return "24h"
    return f"{window_seconds}s"


# ── public API ────────────────────────────────────────────────────────────────

def add_time_since_last(df: pd.DataFrame, uid_col: str = "uid1") -> pd.DataFrame:
    """Return df with a new 'time_since_last_{uid_col}' column (float, seconds).

    NaN for the first transaction per uid and for rows where uid_col is NaN.
    Duplicate TransactionDT within the same uid yields 0, not NaN — two
    transactions happened simultaneously, which is valid and distinct from
    "no prior transaction".
    """
    _require_cols(df, ["TransactionDT", uid_col])
    out_col = f"time_since_last_{uid_col}"

    tsl = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = df[uid_col].notna()
    if valid.any():
        sorted_sub = df.loc[valid].sort_values([uid_col, "TransactionDT"])
        diff = sorted_sub.groupby(uid_col, sort=False)["TransactionDT"].diff()
        tsl.loc[diff.index] = diff.values

    return df.assign(**{out_col: tsl})


def add_velocity_window(
    df: pd.DataFrame,
    window_seconds: int,
    uid_col: str = "uid1",
) -> pd.DataFrame:
    """Return df with 4 velocity features for the given look-back window.

    Features added (W = window label "1h" / "24h" / "{N}s"):
      count_{uid_col}_{W}                  — number of prior transactions in window
      sum_amt_{uid_col}_{W}                — sum of TransactionAmt in window
      mean_amt_{uid_col}_{W}               — mean of TransactionAmt in window
      n_distinct_productcd_{uid_col}_{W}   — distinct ProductCD values in window

    The window [t - window_seconds, t) is strictly past. Duplicate timestamps
    within the same uid are excluded from each other (right boundary uses
    searchsorted side='left', which finds the first index >= t, so all rows
    with TransactionDT == t are excluded from each other's windows).
    """
    _require_cols(df, ["TransactionDT", "TransactionAmt", "ProductCD", uid_col])

    W = _window_suffix(window_seconds)
    col_count = f"count_{uid_col}_{W}"
    col_sum   = f"sum_amt_{uid_col}_{W}"
    col_mean  = f"mean_amt_{uid_col}_{W}"
    col_ndist = f"n_distinct_productcd_{uid_col}_{W}"

    n = len(df)
    out_count = np.full(n, np.nan)
    out_sum   = np.full(n, np.nan)
    out_mean  = np.full(n, np.nan)
    out_ndist = np.full(n, np.nan)

    valid_mask = df[uid_col].notna()
    if valid_mask.any():
        valid_pos = np.where(valid_mask.values)[0]
        sub = df.iloc[valid_pos]

        # np.lexsort: last key is primary sort key → uid first, then dt
        order = np.lexsort((sub["TransactionDT"].values, sub[uid_col].values))
        sorted_pos = valid_pos[order]

        dt_arr  = df["TransactionDT"].values[sorted_pos]
        amt_arr = df["TransactionAmt"].values[sorted_pos]
        pcd_arr = df["ProductCD"].values[sorted_pos]
        uid_arr = df[uid_col].values[sorted_pos]

        i = 0
        total = len(sorted_pos)
        while i < total:
            uid_val = uid_arr[i]
            j = i + 1
            while j < total and uid_arr[j] == uid_val:
                j += 1

            g_dt  = dt_arr[i:j]
            g_amt = amt_arr[i:j]
            g_pcd = pcd_arr[i:j]
            g_pos = sorted_pos[i:j]
            g_len = j - i

            for k in range(g_len):
                t = g_dt[k]
                # right: first index where dt >= t → excludes t and all duplicates of t
                right = int(np.searchsorted(g_dt, t, side="left"))
                left  = int(np.searchsorted(g_dt, t - window_seconds, side="left"))
                cnt   = right - left

                out_count[g_pos[k]] = cnt
                if cnt > 0:
                    w_amt = g_amt[left:right]
                    w_pcd = g_pcd[left:right]
                    out_sum[g_pos[k]]   = w_amt.sum()
                    out_mean[g_pos[k]]  = w_amt.mean()
                    nonnull = w_pcd[~pd.isna(w_pcd)]
                    out_ndist[g_pos[k]] = len(np.unique(nonnull))
                else:
                    out_sum[g_pos[k]]   = 0.0
                    out_mean[g_pos[k]]  = np.nan
                    out_ndist[g_pos[k]] = 0

            i = j

    return df.assign(**{
        col_count: out_count,
        col_sum:   out_sum,
        col_mean:  out_mean,
        col_ndist: out_ndist,
    })


def add_all_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply time_since_last + velocity windows (1h, 24h) for uid1.

    Adds 9 new columns total: 1 (time_since_last) + 4 per window × 2 windows.
    """
    df = add_time_since_last(df, uid_col="uid1")
    df = add_velocity_window(df, window_seconds=3600,  uid_col="uid1")
    df = add_velocity_window(df, window_seconds=86400, uid_col="uid1")
    return df


def run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Apply all velocity transformations."""
    return add_all_velocity_features(df)


# ── diagnostics ───────────────────────────────────────────────────────────────

_VELOCITY_FEATURES = [
    "time_since_last_uid1",
    "count_uid1_1h",
    "sum_amt_uid1_1h",
    "mean_amt_uid1_1h",
    "n_distinct_productcd_uid1_1h",
    "count_uid1_24h",
    "sum_amt_uid1_24h",
    "mean_amt_uid1_24h",
    "n_distinct_productcd_uid1_24h",
]


def velocity_diagnostics(df: pd.DataFrame) -> None:
    """Print stats, isFraud correlations, and burst examples for velocity features."""
    present = [c for c in _VELOCITY_FEATURES if c in df.columns]

    print(f"\n{'=' * 60}")
    print("  Velocity Feature Diagnostics")
    print(f"{'=' * 60}")

    for col in present:
        s = df[col].dropna()
        print(f"\n  {col}  (non-NaN: {len(s):,} / {len(df):,})")
        if len(s) > 0:
            desc = s.describe(percentiles=[0.50, 0.90, 0.99])
            for stat, val in desc.items():
                print(f"    {stat:>6} : {val:.4f}")

    if "isFraud" in df.columns:
        print(f"\n  Pearson correlation with isFraud:")
        for col in present:
            corr = df[[col, "isFraud"]].dropna().corr().loc[col, "isFraud"]
            print(f"    {col:<45s}: {corr:+.4f}")

    burst_col = "count_uid1_1h"
    if burst_col in df.columns:
        bursts = df[df[burst_col] >= 5]
        print(f"\n  Transactions with {burst_col} >= 5: {len(bursts):,}")
        if len(bursts) > 0:
            show = [c for c in [
                "uid1", "TransactionDT", "TransactionAmt", "ProductCD",
                "isFraud", burst_col, "sum_amt_uid1_1h",
            ] if c in df.columns]
            print(bursts.head(3)[show].to_string(index=False))


# ── script entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROC_DIR = Path(__file__).parents[2] / "data" / "processed"
    src_path = PROC_DIR / "train_transaction_features.parquet"

    print(f"Loading {src_path} ...")
    df = pd.read_parquet(src_path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    print("Computing velocity features (this may take 1-2 minutes)...")
    df = add_all_velocity_features(df)

    velocity_diagnostics(df)

    df.to_parquet(src_path, index=False)
    size_mb = src_path.stat().st_size / 1_048_576
    print(f"\nSaved → {src_path}")
    print(f"File size: {size_mb:.1f} MB")
    print("Done.")