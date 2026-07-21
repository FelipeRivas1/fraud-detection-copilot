"""Dataset contract: column definitions and categorical mappings for train/serve parity.

This module is the single source of truth for how raw feature columns are prepared
before being handed to LightGBM. Categorical mappings are frozen from training data
and reused at serving time so that pd.Categorical codes are identical regardless of
how many rows are in the DataFrame being scored.

Critical invariant: fit_categorical_mappings is called ONLY on the training split.
apply_categorical_mappings consumes the saved artifact and must be called on every
split and on every single-row scoring request at serving time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ── constants ─────────────────────────────────────────────────────────────────

CATEGORICAL_COLS: list[str] = [
    # transaction categoricals (low-cardinality, native LightGBM categorical)
    "ProductCD",
    "card4",
    "card6",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType",
    # identity categoricals (≤4 unique values each, from train_identity post-merge)
    "id_12", "id_15", "id_16", "id_23",
    "id_27", "id_28", "id_29",
    "id_34", "id_35", "id_36", "id_37", "id_38",
]
"""Low-cardinality columns passed to LightGBM as native categoricals.

These are label-encoded via pd.Categorical with frozen category lists so that
integer codes are reproducible across training splits and single-row serving calls.
High-cardinality columns (card1, card2, addr1, emaildomains, DeviceInfo, id_30,
id_31, id_33) are handled separately via frequency encoding in src/features/encoding.py.
"""

EXCLUDED_COLS: list[str] = [
    # raw transaction identifiers — no predictive signal
    "TransactionID",
    # raw event time: using it as a feature would leak the temporal split boundary
    # (val/test rows have systematically higher TransactionDT than train rows)
    "TransactionDT",
    # helper column created by src/features/time.py; encodes position in the
    # chronological sort used by the temporal split — pure leak, not a signal
    "day_index",
    # target and split metadata
    "isFraud",
    "split",
    # raw high-cardinality columns superseded by their _freq counterparts
    # produced by src/features/encoding.py; keeping both would be redundant
    # and the raw string values are not usable by LightGBM without encoding
    "card1",
    "card2",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",   # superseded by id_30_freq (OS + version, 75 categories)
    "id_31",   # superseded by id_31_freq (browser + version, 130 categories)
    "id_33",   # superseded by id_33_freq (screen resolution, 260 categories)
    # aggregation keys — signal already captured in derived features
    "uid1",    # key for velocity features (vel_*_uid1); string itself is not a feature
    "uid2",    # key for uid2 aggregations (*_uid2); string itself is not a feature
]
"""Columns dropped before building the feature matrix X.

Covers: identifiers, raw event time (temporal-split leak), split/target metadata,
raw high-cardinality strings replaced by frequency-encoded numerics, and UID
aggregation keys whose signal is fully captured in derived velocity/aggregation columns.
"""


# ── public API ────────────────────────────────────────────────────────────────

def fit_categorical_mappings(
    df: pd.DataFrame,
    columns: list[str],
    artifact_path: str | Path = "models/encoders/categorical_mappings.json",
) -> dict[str, list[str]]:
    """Extract and freeze category lists from training data.

    For each column, collects all non-NaN unique values, casts them to str,
    sorts alphabetically, and saves the result as a JSON artifact. The saved
    artifact is consumed at serving time so that pd.Categorical codes are
    identical whether the DataFrame has 400k rows or 1.

    Fit on training data only. Apply with apply_categorical_mappings to all
    splits and at serving time.

    Parameters
    ----------
    df:
        Training DataFrame. Must contain all columns listed in `columns`.
    columns:
        Column names to extract category lists from.
    artifact_path:
        Where to write the JSON artifact. Parent directories are created if
        they do not exist.

    Returns
    -------
    dict[str, list[str]]
        Mapping of {column_name: [sorted category strings]}.

    Raises
    ------
    ValueError
        If any column in `columns` is not present in `df`.
    """
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    mappings: dict[str, list[str]] = {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in columns
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w") as f:
        json.dump(mappings, f, indent=2, sort_keys=True)

    return mappings


def apply_categorical_mappings(
    df: pd.DataFrame,
    mappings: dict[str, list[str]] | str | Path = "models/encoders/categorical_mappings.json",
) -> pd.DataFrame:
    """Apply frozen category lists to the indicated columns.

    `mappings` can be the dict already in memory or a path to a JSON produced
    by fit_categorical_mappings. Returns a copy of df with the relevant columns
    converted to pd.Categorical with fixed category lists. Values present in df
    but absent from the frozen categories become NaN; LightGBM handles NaN
    natively in categorical features.

    Must be called on every split and on every single-row request at serving
    time — never call pd.Categorical without this function, because building
    categories from the DataFrame itself produces different integer codes.

    Parameters
    ----------
    df:
        DataFrame whose categorical columns will be converted.
    mappings:
        Frozen category lists as a dict, or a path to the JSON artifact.

    Returns
    -------
    pd.DataFrame
        Copy of df with categorical columns replaced by pd.Categorical series.

    Raises
    ------
    ValueError
        If any column in mappings is not present in df.
    """
    if isinstance(mappings, (str, Path)):
        with Path(mappings).open() as f:
            mappings = json.load(f)

    df = df.copy()

    for col, categories in mappings.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        # Cast only non-null values to str to avoid converting NaN → "nan"
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str)

        df[col] = pd.Categorical(df[col], categories=categories)

    return df


def load_features(
    parquet_path: str | Path = "/Users/feliperivas/Desktop/Code/Proyectos/fraud-detection-copilot/data/processed/train_transaction_features.parquet",
) -> pd.DataFrame:
    """Load the feature parquet produced by scripts/build_features.py.

    Parameters
    ----------
    parquet_path:
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame, unmodified.

    Raises
    ------
    FileNotFoundError
        If the parquet does not exist at parquet_path.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Feature parquet not found at '{parquet_path}'. "
            "Run `python scripts/build_features.py` to generate it."
        )
    return pd.read_parquet(parquet_path)


def split_xy(
    df: pd.DataFrame,
    mappings_path: str | Path = "models/encoders/categorical_mappings.json",
) -> tuple[
    pd.DataFrame, pd.Series,   # X_train, y_train
    pd.DataFrame, pd.Series,   # X_val,   y_val
    pd.DataFrame, pd.Series,   # X_test,  y_test
    list[str],                  # categorical_cols
]:
    """Split df into train/val/test feature matrices and target vectors.

    Drops EXCLUDED_COLS from X, applies frozen categorical mappings to
    CATEGORICAL_COLS, and returns X/y pairs per split plus the list of
    categorical column names to pass to LightGBM as `categorical_feature`.

    Assumes fit_categorical_mappings has already been run and the JSON artifact
    exists at mappings_path.

    Parameters
    ----------
    df:
        Full feature DataFrame with `split` and `isFraud` columns.
    mappings_path:
        Path to the JSON artifact produced by fit_categorical_mappings.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, categorical_cols

    Raises
    ------
    ValueError
        If `split` or `isFraud` columns are missing from df.
    FileNotFoundError
        If the mappings artifact does not exist at mappings_path.
    """
    if "split" not in df.columns:
        raise ValueError(
            "DataFrame has no 'split' column. Run src/features/split.py first."
        )
    if "isFraud" not in df.columns:
        raise ValueError("DataFrame has no 'isFraud' column.")

    mappings_path = Path(mappings_path)
    if not mappings_path.exists():
        raise FileNotFoundError(
            f"Categorical mappings not found at '{mappings_path}'. "
            "Run fit_categorical_mappings on the training split first."
        )

    splits: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for label in ("train", "val", "test"):
        subset = df[df["split"] == label]
        y = subset["isFraud"].astype(int)
        X = subset.drop(columns=EXCLUDED_COLS, errors="ignore")
        X = apply_categorical_mappings(X, mappings_path)
        splits[label] = (X, y)

    for label, (X, y) in splits.items():
        fraud_rate = y.mean() * 100
        print(f"  {label:<5}: X {str(X.shape):<16}  y fraud rate: {fraud_rate:.2f}%")

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    return X_train, y_train, X_val, y_val, X_test, y_test, CATEGORICAL_COLS