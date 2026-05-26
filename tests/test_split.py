"""Tests for src/features/split.py."""

import pandas as pd
import pytest

from src.features.split import temporal_split


def _make_df(dt):
    """Minimal DataFrame with just TransactionDT."""
    return pd.DataFrame({"TransactionDT": dt})


def test_no_overlap_in_time():
    """Splits must be strictly chronological: max(train) < min(val) < min(test)."""
    dt = list(range(100))
    shuffled = pd.Series(dt).sample(frac=1, random_state=42).tolist()
    df = _make_df(shuffled)
    out = temporal_split(df)

    train_max = out.loc[out["split"] == "train", "TransactionDT"].max()
    val_min   = out.loc[out["split"] == "val",   "TransactionDT"].min()
    val_max   = out.loc[out["split"] == "val",   "TransactionDT"].max()
    test_min  = out.loc[out["split"] == "test",  "TransactionDT"].min()

    assert train_max < val_min, f"train_max={train_max} >= val_min={val_min}"
    assert val_max < test_min,  f"val_max={val_max} >= test_min={test_min}"


def test_proportions_approximately_correct():
    """70/15/15 split on range(1000) should yield near-exact counts."""
    df = _make_df(list(range(1000)))
    out = temporal_split(df)

    counts = out["split"].value_counts()
    assert abs(counts["train"] - 700) <= 5
    assert abs(counts["val"]   - 150) <= 5
    assert abs(counts["test"]  - 150) <= 5


def test_original_order_preserved():
    """temporal_split must not reorder rows; marker column order must be unchanged."""
    dt     = [50, 10, 90, 30, 70, 20, 80, 40, 60, 0]
    marker = list(range(len(dt)))
    df = pd.DataFrame({"TransactionDT": dt, "marker": marker})

    out = temporal_split(df)

    assert out["marker"].tolist() == marker


def test_invalid_fractions_raises():
    """ValueError when train_frac + val_frac >= 1.0."""
    df = _make_df(list(range(10)))

    with pytest.raises(ValueError):
        temporal_split(df, train_frac=0.7, val_frac=0.3)   # sum == 1.0

    with pytest.raises(ValueError):
        temporal_split(df, train_frac=0.8, val_frac=0.3)   # sum > 1.0


def test_missing_time_col_raises():
    """ValueError when the specified time_col is not in the DataFrame."""
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})

    with pytest.raises(ValueError, match="TransactionDT"):
        temporal_split(df, time_col="TransactionDT")
