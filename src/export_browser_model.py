"""Export the final Logistic Regression pipeline as a browser-readable JSON model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "final" / "final_logistic_regression_pipeline.joblib"
OUTPUT_PATH = ROOT / "website" / "data" / "browser_model.json"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the final model first: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    if classifier.__class__.__name__ != "LogisticRegression":
        raise TypeError("Expected LogisticRegression; refusing to export a different classifier.")

    payload = {
        "model_type": "TF-IDF + Logistic Regression (L2)",
        "model_family": "logistic_regression",
        "classes": [int(x) for x in classifier.classes_],
        "positive_class": 1,
        "intercept": float(classifier.intercept_[0]),
        "vocabulary": {str(term): int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf": [float(x) for x in vectorizer.idf_],
        "coefficients": [float(x) for x in classifier.coef_[0]],
        "probability_link": "sigmoid",
        "decision_score_description": "Log-odds for CG (class 1); sigmoid(score) gives P(CG).",
        "vectorizer": {
            "lowercase": bool(vectorizer.lowercase),
            "ngram_range": list(vectorizer.ngram_range),
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "max_features": vectorizer.max_features,
            "stop_words": None,
            "sublinear_tf": bool(vectorizer.sublinear_tf),
            "norm": vectorizer.norm,
            "use_idf": bool(vectorizer.use_idf),
            "smooth_idf": bool(vectorizer.smooth_idf),
            "token_pattern": vectorizer.token_pattern,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    size_mb = OUTPUT_PATH.stat().st_size / (1024 ** 2)
    print(f"Browser model written: {OUTPUT_PATH}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"Uncompressed JSON: {size_mb:.3f} MB")
    print("Browser inference applies the sigmoid to the Logistic Regression log-odds score.")


if __name__ == "__main__":
    main()
