"""Frequency encoding for high-cardinality categorical columns.

Encodes each category by its count in the training set. Compresses thousands of
unique values into a single numeric column that captures rarity — a key signal
in fraud (new/rare cards are more likely to be fraudulent).

Critical invariant: encoders are fit ONLY on the training set. Applying the
same encoder to val/test prevents leakage. Values unseen in train become NaN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── public API ────────────────────────────────────────────────────────────────

def fit_frequency_encoders(
    df: pd.DataFrame,
    cols: list[str],
) -> dict[str, dict]:
    """Fit frequency encoders from the training rows only.

    Returns {col: {value: count}} for each column. NaN values are excluded from
    the counts. Columns not present in df are skipped with a warning.
    """
    if "split" not in df.columns:
        raise ValueError("DataFrame has no 'split' column. Run temporal_split first.")

    train = df[df["split"] == "train"]
    encoders: dict[str, dict] = {}

    for col in cols:
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found in DataFrame — skipping.")
            continue
        encoders[col] = train[col].value_counts(dropna=True).to_dict()

    return encoders


def save_encoders(encoders: dict[str, dict], path: str | Path) -> None:
    """Serialize encoders to JSON at path, creating parent directories if needed.

    All dict keys are serialized as strings (JSON does not support numeric keys).
    Values are int counts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # value_counts() returns numeric keys for numeric columns; JSON requires str keys
    serializable = {
        col: {str(k): int(v) for k, v in mapping.items()}
        for col, mapping in encoders.items()
    }
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)


def load_encoders(path: str | Path) -> dict[str, dict]:
    """Load encoders from JSON. Keys remain as strings — transform handles dtype conversion."""
    with Path(path).open() as f:
        return json.load(f)


def transform_frequency_encoders(
    df: pd.DataFrame,
    encoders: dict[str, dict],
) -> pd.DataFrame:
    """Add a '{col}_freq' column for each encoder, mapping values to training counts.

    Lookup keys are cast to match the dtype of each column in df, so numeric
    columns (card1, card2, addr1) work correctly even though JSON keys are strings.
    Values not seen in training and NaN in the original column both map to NaN.
    Columns not present in df are skipped with a warning.
    """
    new_cols: dict[str, pd.Series] = {}

    for col, mapping in encoders.items():
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found in DataFrame — skipping.")
            continue

        col_dtype = df[col].dtype

        # Convert string keys back to the column's dtype for correct lookup
        if pd.api.types.is_integer_dtype(col_dtype) or pd.api.types.is_float_dtype(col_dtype):
            try:
                typed_mapping = {col_dtype.type(k): v for k, v in mapping.items()}
            except (ValueError, TypeError):
                typed_mapping = mapping  # fall back to str keys if conversion fails
        else:
            typed_mapping = mapping  # string/object columns: keys already match

        freq = df[col].map(typed_mapping)  # unmapped values → NaN automatically
        new_cols[f"{col}_freq"] = freq.astype("float64")

    return df.assign(**new_cols)


_DEFAULT_COLS = [
    "card1", "card2", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo",
    "id_30", "id_31", "id_33",
]
_DEFAULT_ENC_PATH = Path(__file__).parents[2] / "models" / "encoders" / "frequency_encoders.json"


def run(
    df: pd.DataFrame,
    encoders_path: str | Path | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Apply frequency encoding to the default columns.

    Training mode (encoders_path=None): fit on the train split, save encoders to
    models/encoders/frequency_encoders.json, then transform the full df.

    Serving mode (encoders_path provided): load encoders from disk, transform only.
    """
    if encoders_path is None:
        encoders = fit_frequency_encoders(df, _DEFAULT_COLS)
        save_encoders(encoders, _DEFAULT_ENC_PATH)
    else:
        encoders = load_encoders(Path(encoders_path))

    return transform_frequency_encoders(df, encoders)


# ── diagnostics ───────────────────────────────────────────────────────────────

def encoding_diagnostics(df: pd.DataFrame, encoders: dict[str, dict]) -> None:
    """Print NaN rates, unseen-value counts, and distribution stats per encoded column."""
    present_cols = [col for col in encoders if col in df.columns]
    freq_cols = [f"{col}_freq" for col in present_cols if f"{col}_freq" in df.columns]

    n_total = len(df)
    print(f"\n{'=' * 60}")
    print("  Frequency Encoding Diagnostics")
    print(f"{'=' * 60}")

    for col in present_cols:
        freq_col = f"{col}_freq"
        if freq_col not in df.columns:
            continue

        encoded = df[freq_col]
        n_nan = encoded.isna().sum()

        # Unseen in train: original not NaN but encoded is NaN
        orig_not_nan = df[col].notna()
        n_unseen = (orig_not_nan & encoded.isna()).sum()

        print(f"\n  {col} → {freq_col}")
        print(f"    NaN total       : {n_nan:,} / {n_total:,}  ({n_nan / n_total * 100:.1f}%)")
        print(f"    unseen in train : {n_unseen:,}  ({n_unseen / n_total * 100:.2f}%)")

        s = encoded.dropna()
        if len(s) > 0:
            desc = s.describe(percentiles=[0.25, 0.50, 0.75, 0.99])
            for stat, val in desc.items():
                print(f"    {stat:>6} : {val:.1f}")

    # Summary table by split
    if "split" in df.columns and freq_cols:
        print(f"\n  NaN % per split:")
        header = f"  {'split':<8}" + "".join(f"  {c:<30}" for c in freq_cols)
        print(header)
        for label in ("train", "val", "test"):
            sub = df[df["split"] == label]
            if len(sub) == 0:
                continue
            row = f"  {label:<8}"
            for fc in freq_cols:
                pct = sub[fc].isna().mean() * 100
                row += f"  {pct:>5.1f}%{'':24}"
            print(row)


# ── script entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROC_DIR = Path(__file__).parents[2] / "data" / "processed"
    MODELS_DIR = Path(__file__).parents[2] / "models" / "encoders"
    src_path = PROC_DIR / "train_transaction_features.parquet"
    enc_path = MODELS_DIR / "frequency_encoders.json"

    COLS_TO_ENCODE = _DEFAULT_COLS

    print(f"Loading {src_path} ...")
    df = pd.read_parquet(src_path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    print("\nFitting encoders on train split ...")
    encoders = fit_frequency_encoders(df, COLS_TO_ENCODE)

    save_encoders(encoders, enc_path)
    print(f"Encoders saved → {enc_path}")

    print("\nUnique values per encoder:")
    for col, mapping in encoders.items():
        print(f"  {col:<20}: {len(mapping):,} unique values")

    print("\nApplying encoders ...")
    df = transform_frequency_encoders(df, encoders)

    encoding_diagnostics(df, encoders)

    df.to_parquet(src_path, index=False)
    size_mb = src_path.stat().st_size / 1_048_576
    print(f"\nSaved → {src_path}")
    print(f"File size: {size_mb:.1f} MB")
    print("Done.")
