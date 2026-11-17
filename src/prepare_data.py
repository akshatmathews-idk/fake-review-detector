"""Clean the source dataset, remove duplicate review texts, and make a stratified split.

Run from the project root:
    python src/prepare_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import normalize_label

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "fake reviews dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}\n"
            "Download the Kaggle dataset and place its CSV at data/raw/fake reviews dataset.csv."
        )

    raw = pd.read_csv(RAW_PATH)
    required = {"text_", "label", "category", "rating"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {sorted(missing)}")

    if raw["text_"].isna().any() or raw["label"].isna().any():
        raise ValueError("Missing review text or label found; inspect the source dataset before proceeding.")

    cleaned = raw.copy()
    cleaned["text_"] = cleaned["text_"].astype("string").str.strip()
    if cleaned["text_"].eq("").any():
        raise ValueError("Empty review texts found after trimming whitespace.")

    duplicates_removed = int(cleaned["text_"].duplicated().sum())
    conflicting = cleaned.groupby("text_")["label"].nunique()
    conflicting_count = int((conflicting > 1).sum())
    if conflicting_count:
        raise ValueError(f"Found {conflicting_count} identical review texts with conflicting labels.")

    cleaned = cleaned.drop_duplicates(subset="text_", keep="first").reset_index(drop=True)
    cleaned["target"] = cleaned["label"].map(normalize_label).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned["text_"], cleaned["target"],
        test_size=0.20, random_state=42, stratify=cleaned["target"]
    )

    train_df = pd.DataFrame({"text": X_train, "target": y_train})
    test_df = pd.DataFrame({"text": X_test, "target": y_test})
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    train_texts = set(train_df["text"])
    test_texts = set(test_df["text"])
    if train_texts.intersection(test_texts):
        raise RuntimeError("Train/test text overlap detected after split.")

    metadata = {
        "source_file": str(RAW_PATH.relative_to(ROOT)),
        "original_rows": int(len(raw)),
        "duplicate_text_occurrences_removed": duplicates_removed,
        "cleaned_rows": int(len(cleaned)),
        "conflicting_duplicate_texts": conflicting_count,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "target_mapping": {"OR": 0, "CG": 1},
        "random_state": 42,
        "test_size": 0.20,
        "train_label_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_label_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
    }
    (PROCESSED_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Original rows: {len(raw):,}")
    print(f"Duplicate review texts removed: {duplicates_removed}")
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print("OR=0, CG=1. Stratification and no-overlap checks passed.")


if __name__ == "__main__":
    main()
