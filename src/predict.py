"""Generate predictions for the project's processed review dataset.

Run from the project root:
    python src/predict.py
    python src/predict.py --split test
    python src/predict.py --split all --output results/predictions/my_predictions.csv

The default predicts both processed splits and records which split each row came from.
Training-set predictions are in-sample; test-set predictions are the held-out evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import validate_processed_frame  # noqa: E402

MODEL_PATH = ROOT / "models" / "final" / "final_logistic_regression_pipeline.joblib"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
DEFAULT_OUTPUT = ROOT / "results" / "predictions" / "dataset_predictions.csv"


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run `python src/prepare_data.py` first."
        )
    frame = validate_processed_frame(pd.read_csv(path))
    frame = frame.copy()
    frame.insert(0, "split", split_name)
    return frame


def predict_dataset(split: str, output_path: Path) -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run `python src/train.py` first."
        )

    model = joblib.load(MODEL_PATH)

    if split == "train":
        data = load_split(TRAIN_PATH, "train")
    elif split == "test":
        data = load_split(TEST_PATH, "test")
    else:
        train = load_split(TRAIN_PATH, "train")
        test = load_split(TEST_PATH, "test")
        data = pd.concat([train, test], ignore_index=True)

    texts = data["text"].astype(str)
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)

    classes = [int(value) for value in model.named_steps["classifier"].classes_]
    probability_by_class = {
        label: probabilities[:, index]
        for index, label in enumerate(classes)
    }

    output = pd.DataFrame({
        "split": data["split"],
        "text": texts,
        "actual_label": data["target"].astype(int),
        "predicted_label": predictions.astype(int),
        "actual_name": data["target"].map({0: "OR", 1: "CG"}),
        "predicted_name": pd.Series(predictions).map({0: "OR", 1: "CG"}),
        "probability_OR": probability_by_class[0],
        "probability_CG": probability_by_class[1],
    })
    output["correct"] = output["actual_label"] == output["predicted_label"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print(f"Rows predicted: {len(output):,}")
    print(f"Split selection: {split}")

    for split_name, group in output.groupby("split", sort=False):
        accuracy = group["correct"].mean()
        print(
            f"{split_name.title():>5}: {len(group):,} rows | "
            f"{group['correct'].sum():,} correct | "
            f"{(~group['correct']).sum():,} incorrect | "
            f"{accuracy:.2%} agreement with dataset labels"
        )

    if split in {"all", "train"}:
        print(
            "\nNote: train-split predictions are in-sample because the model "
            "was fitted using those reviews. Do not present their accuracy "
            "as held-out generalisation performance."
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Logistic Regression predictions for the processed review dataset."
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Which processed split to predict (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    predict_dataset(args.split, output_path)


if __name__ == "__main__":
    main()
