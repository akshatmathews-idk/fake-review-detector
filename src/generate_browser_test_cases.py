"""Generate six correctly classified examples for the browser demo."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import normalize_label, clean_text_series

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "fake reviews dataset.csv"
MODEL_PATH = ROOT / "models" / "final" / "final_logistic_regression_pipeline.joblib"
OUTPUT_PATH = ROOT / "website" / "data" / "browser_test_cases.json"


def main() -> None:
    for path in (RAW_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(RAW_PATH)
    if not {"text_", "label"}.issubset(df.columns):
        raise ValueError("Raw dataset must contain text_ and label columns.")
    df["text_"] = clean_text_series(df["text_"])
    df["target"] = df["label"].map(normalize_label)
    df = df.dropna(subset=["text_", "target"])
    df = df[df["text_"].str.len() > 0].drop_duplicates("text_", keep="first").reset_index(drop=True)

    model = joblib.load(MODEL_PATH)
    probability_cg = model.predict_proba(df["text_"].tolist())[:, 1]
    predictions = model.predict(df["text_"].tolist())
    df["probability_cg"] = probability_cg
    df["prediction"] = predictions

    cases = []
    for class_name, target in (("OR", 0), ("CG", 1)):
        subset = df[(df["target"] == target) & (df["prediction"] == target)].copy()
        subset["class_probability"] = subset["probability_cg"] if target == 1 else 1 - subset["probability_cg"]
        subset = subset.sort_values("class_probability", ascending=False).head(3)
        for index, (_, row) in enumerate(subset.iterrows(), start=1):
            cases.append({
                "id": f"{class_name}-{index}",
                "label": class_name,
                "review": str(row["text_"]),
                "actual_label": class_name,
                "reference_prediction": class_name,
                "reference_probability_cg": float(row["probability_cg"]),
            })
    if len(cases) != 6:
        raise RuntimeError(f"Expected six correctly classified examples, created {len(cases)}.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(cases)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
