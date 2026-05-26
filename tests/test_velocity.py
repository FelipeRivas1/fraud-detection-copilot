"""Tests for src/features/velocity.py.

The most important test is the no-leak invariant: for any transaction at time t,
no velocity feature includes information from events with timestamp >= t.
"""

import pandas as pd
import pytest

from src.features.velocity import (
    add_all_velocity_features,
    add_time_since_last,
    add_velocity_window,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_df(dt, amt, pcd, uid=None):
    """Build a minimal transaction DataFrame for testing."""
    n = len(dt)
    return pd.DataFrame({
        "TransactionDT":  dt,
        "TransactionAmt": amt,
        "ProductCD":      pcd,
        "uid1":           [uid] * n if isinstance(uid, (str, type(None))) else uid,
    })


# ── tests ─────────────────────────────────────────────────────────────────────

def test_no_leak_basic_window():
    """Window [t-W, t) is strictly past: the lower bound is inclusive, upper is exclusive.

    Four transactions at 0, 1800, 3600, 5400s with window=3600s (1h).
    At t=3600 the boundary is exactly t-W=0, which IS included (inclusive lower
    bound), so count should be 2 (t=0 and t=1800).
    At t=5400 the window [1800, 5400) includes t=1800 and t=3600 → count=2.
    """
    df = _make_df(
        dt  = [0, 1800, 3600, 5400],
        amt = [10.0, 20.0, 30.0, 40.0],
        pcd = ["W", "W", "W", "W"],
        uid = "A",
    )
    out = add_velocity_window(df, window_seconds=3600)

    counts = out["count_uid1_1h"].tolist()
    sums   = out["sum_amt_uid1_1h"].tolist()

    # t=0: no prior transactions
    assert counts[0] == 0
    assert sums[0] == 0.0
    assert pd.isna(out["mean_amt_uid1_1h"].iloc[0])

    # t=1800: window [−1800, 1800) → only t=0
    assert counts[1] == 1
    assert sums[1] == 10.0
    assert out["mean_amt_uid1_1h"].iloc[1] == pytest.approx(10.0)

    # t=3600: window [0, 3600) → t=0 (inclusive) and t=1800 → count=2
    assert counts[2] == 2
    assert sums[2] == pytest.approx(30.0)
    assert out["mean_amt_uid1_1h"].iloc[2] == pytest.approx(15.0)

    # t=5400: window [1800, 5400) → t=1800 and t=3600 → count=2
    assert counts[3] == 2
    assert sums[3] == pytest.approx(50.0)
    assert out["mean_amt_uid1_1h"].iloc[3] == pytest.approx(25.0)


def test_no_leak_strict_upper_bound():
    """Transactions with the same timestamp must not appear in each other's windows.

    Two rows for the same uid at TransactionDT=100. Because the window is [t-W, t)
    and both have t=100, neither sees the other. Both must have count=0.
    """
    df = _make_df(
        dt  = [100, 100],
        amt = [50.0, 60.0],
        pcd = ["W", "W"],
        uid = "A",
    )
    out = add_velocity_window(df, window_seconds=3600)

    assert out["count_uid1_1h"].iloc[0] == 0
    assert out["count_uid1_1h"].iloc[1] == 0
    assert out["sum_amt_uid1_1h"].iloc[0] == 0.0
    assert out["sum_amt_uid1_1h"].iloc[1] == 0.0
    assert pd.isna(out["mean_amt_uid1_1h"].iloc[0])
    assert pd.isna(out["mean_amt_uid1_1h"].iloc[1])


def test_time_since_last_first_transaction():
    """The first transaction of a uid has no prior event — time_since_last must be NaN."""
    df = _make_df(dt=[500], amt=[10.0], pcd=["W"], uid="A")
    out = add_time_since_last(df)
    assert pd.isna(out["time_since_last_uid1"].iloc[0])


def test_time_since_last_duplicate_timestamps():
    """Two simultaneous transactions for the same uid: first is NaN, second is 0.

    time_since_last=0 is correct here — the events are real and simultaneous,
    not "first ever". A 0 is distinguishable from NaN, which matters for the model.
    """
    df = _make_df(dt=[100, 100], amt=[10.0, 20.0], pcd=["W", "W"], uid="A")
    out = add_time_since_last(df)

    tsl = out["time_since_last_uid1"]
    # One row is NaN (first in group), the other is 0.
    # The sort order within tied timestamps is stable but arbitrary, so we check
    # the multiset of values rather than positional order.
    nan_count  = tsl.isna().sum()
    zero_count = (tsl == 0.0).sum()
    assert nan_count == 1
    assert zero_count == 1


def test_uid_nan_propagates():
    """Rows where uid1 is NaN must have NaN for every velocity feature."""
    df = _make_df(
        dt  = [100, 200, 300],
        amt = [10.0, 20.0, 30.0],
        pcd = ["W", "W", "W"],
        uid = ["A", None, "A"],
    )
    out = add_all_velocity_features(df)

    nan_row = out.iloc[1]
    velocity_cols = [
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
    for col in velocity_cols:
        assert pd.isna(nan_row[col]), f"Expected NaN for {col} when uid1 is NaN"


def test_n_distinct_productcd():
    """n_distinct_productcd counts unique ProductCD values in the look-back window."""
    df = _make_df(
        dt  = [0, 100, 200, 300],
        amt = [10.0, 20.0, 30.0, 40.0],
        pcd = ["W", "W", "C", "R"],
        uid = "A",
    )
    out = add_velocity_window(df, window_seconds=3600)
    ndist = out["n_distinct_productcd_uid1_1h"]

    # t=0: no prior transactions → 0
    assert ndist.iloc[0] == 0

    # t=100: window contains t=0 → ["W"] → 1 distinct
    assert ndist.iloc[1] == 1

    # t=200: window contains t=0, t=100 → ["W", "W"] → 1 distinct
    assert ndist.iloc[2] == 1

    # t=300: window contains t=0, t=100, t=200 → ["W", "W", "C"] → 2 distinct
    assert ndist.iloc[3] == 2
