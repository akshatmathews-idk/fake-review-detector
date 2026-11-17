"""Shared preprocessing and schema-validation helpers."""
from __future__ import annotations

import pandas as pd

TEXT_COLUMN = "text"
TARGET_COLUMN = "target"


def clean_review_text(value) -> str:
    """Strip surrounding whitespace; preserve the review's internal wording."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_text_series(series: pd.Series) -> pd.Series:
    """Clean a text series without converting missing values to the string 'nan'."""
    return series.map(clean_review_text).astype("string")


def normalize_label(value) -> int:
    """Map source labels OR/CG or numeric 0/1 to the binary target."""
    if pd.isna(value):
        raise ValueError("Label cannot be missing.")
    if isinstance(value, str):
        value = value.strip().upper()
        if value == "OR":
            return 0
        if value == "CG":
            return 1
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported label: {value!r}") from exc
    if numeric in (0, 1):
        return numeric
    raise ValueError(f"Unsupported label: {value!r}")


def validate_processed_frame(
    frame: pd.DataFrame,
    text_column: str = TEXT_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Validate the processed train/test schema and return a clean copy."""
    missing = {text_column, target_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Processed data is missing required columns: {sorted(missing)}")

    result = frame.copy()
    if result[target_column].isna().any():
        raise ValueError(f"Missing values found in {target_column!r}.")
    result[text_column] = clean_text_series(result[text_column])
    result = result.loc[result[text_column].str.len() > 0].copy()
    result[target_column] = result[target_column].astype(int)
    labels = set(result[target_column].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected binary targets 0 (OR) and 1 (CG); found {sorted(labels)}.")
    if result[text_column].duplicated().any():
        raise ValueError("Duplicate review texts found in the processed split.")
    return result.reset_index(drop=True)
