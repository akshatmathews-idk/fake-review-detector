"""Train the selected Logistic Regression + TF-IDF model.

Run from project root: python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from preprocessing import validate_processed_frame

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
MODEL_DIR = ROOT / "models" / "final"
RESULT_DIR = ROOT / "results" / "final_logistic_regression"
MODEL_PATH = MODEL_DIR / "final_logistic_regression_pipeline.joblib"
INFO_PATH = RESULT_DIR / "selected_model.json"

PIPELINE_CONFIG = {
    "tfidf": {
        "stop_words": None,
        "ngram_range": (1, 2),
        "min_df": 10,
        "max_df": 1.0,
        "max_features": 100000,
        "sublinear_tf": True,
        "lowercase": True,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "token_pattern": r"(?u)\b\w\w+\b",
    },
    "classifier": {
        "penalty": "l2",
        "C": 3.0,
        "class_weight": None,
        "solver": "liblinear",
        "max_iter": 5000,
        "random_state": 42,
    },
}

# Recorded from the selected no-stop-word Logistic Regression CV search.
# This is a validation estimate, not a held-out test metric.
CV_MACRO_F1 = 0.9397691912
CV_TRAIN_MACRO_F1 = 0.9794058675
CV_GENERALIZATION_GAP = CV_TRAIN_MACRO_F1 - CV_MACRO_F1


def build_pipeline() -> Pipeline:
    """Return a fresh pipeline using the final shared project configuration."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**PIPELINE_CONFIG["tfidf"])),
        ("classifier", LogisticRegression(**PIPELINE_CONFIG["classifier"])),
    ])


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}. Run src/prepare_data.py first.")
    train_df = validate_processed_frame(pd.read_csv(TRAIN_PATH))
    X_train, y_train = train_df["text"], train_df["target"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    train_macro_f1 = f1_score(y_train, pipeline.predict(X_train), average="macro")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    info = {
        "selected_model": "Logistic Regression (L2 / Ridge)",
        "model_type": "tfidf_logistic_regression",
        "training_rows": int(len(train_df)),
        "training_macro_f1": float(train_macro_f1),
        "cross_validation_macro_f1": CV_MACRO_F1,
        "cross_validation_training_macro_f1": CV_TRAIN_MACRO_F1,
        "generalization_gap": CV_GENERALIZATION_GAP,
        "target_mapping": {"OR": 0, "CG": 1},
        "best_parameters": {
            "tfidf__stop_words": None,
            "tfidf__ngram_range": [1, 2],
            "tfidf__min_df": 10,
            "tfidf__max_df": 1.0,
            "tfidf__max_features": 100000,
            "tfidf__sublinear_tf": True,
            "classifier__penalty": "l2",
            "classifier__C": 3.0,
            "classifier__solver": "liblinear",
            "classifier__class_weight": None,
        },
        "pipeline_path": str(MODEL_PATH.relative_to(ROOT)),
        "selection_note": (
            "Logistic Regression was retained for deployment. The tuned LinearSVC "
            "candidate had a slightly higher CV Macro F1 (about 94.38% versus 93.98%), "
            "but Logistic Regression had a smaller train-validation gap (about 3.96 versus "
            "5.27 percentage points) and native probability output."
        ),
    }
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Training Macro F1: {train_macro_f1:.4%}")
    print(f"Recorded CV Macro F1: {CV_MACRO_F1:.4%}")
    print(f"Metadata: {INFO_PATH}")


if __name__ == "__main__":
    main()
