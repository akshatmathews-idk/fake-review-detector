"""Build website/data/results.json using the final Logistic Regression results and model comparisons."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "fake reviews dataset.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
FINAL_DIR = ROOT / "results" / "final_logistic_regression"
MODEL_INFO = FINAL_DIR / "selected_model.json"
EVALUATION = FINAL_DIR / "final_evaluation.json"
METRICS_CSV = FINAL_DIR / "metrics" / "final_test_metrics.csv"
COMPARISON_CSV = ROOT / "results" / "final_model_comparison" / "final_comparison.csv"
VOCAB_JSON = ROOT / "results" / "linearsvc_vocab_experiment" / "linearsvc_vocabulary_comparison.json"
TUNING_SUMMARY = ROOT / "results" / "improvement_no_stopwords" / "model_improvement_summary.json"
BROWSER_MODEL = ROOT / "website" / "data" / "browser_model.json"
OUT = ROOT / "website" / "data" / "results.json"


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    required = [RAW_PATH, TRAIN_PATH, TEST_PATH, MODEL_INFO, EVALUATION, METRICS_CSV]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required project outputs:\n" + "\n".join(missing))

    raw = pd.read_csv(RAW_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    raw_clean = raw.dropna(subset=["text_", "label"]).copy()
    raw_clean["text_"] = raw_clean["text_"].astype("string").str.strip()
    raw_clean = raw_clean[raw_clean["text_"].str.len() > 0]
    raw_clean = raw_clean.drop_duplicates(subset="text_", keep="first")
    counts = raw_clean["label"].astype(str).str.strip().str.upper().value_counts()

    model_info = read_json(MODEL_INFO, {})
    evaluation = read_json(EVALUATION, {})
    metrics_frame = pd.read_csv(METRICS_CSV)
    metrics = {str(row["Metric"]): float(row["Score"]) for _, row in metrics_frame.iterrows()}

    comparison_rows = []
    if COMPARISON_CSV.exists():
        comp = pd.read_csv(COMPARISON_CSV)
        model_name_column = comp.columns[0]
        for _, row in comp.iterrows():
            comparison_rows.append({
                "name": str(row[model_name_column]),
                "cv_macro_f1": float(row["CV Macro F1"]),
                "cv_std": float(row["CV Macro F1 Std"]),
                "training_macro_f1": float(row["Training Macro F1"]),
                "generalization_gap": float(row["Generalisation Gap"]),
                "test_macro_f1": float(row["Macro F1"]),
                "test_accuracy": float(row["Accuracy"]),
                "test_roc_auc": float(row["ROC-AUC"]),
                "json_size_mb": float(row["JSON Size (MB)"]),
                "gzip_size_mb": float(row["Gzip Size (MB)"]),
                "average_inference_ms": float(row["Average Time per Review (ms)"]),
            })

    tuning_rows = []
    for row in read_json(TUNING_SUMMARY, []):
        tuning_rows.append({
            "name": row.get("Model", "Model"),
            "cv_macro_f1": float(row.get("CV Macro F1", 0)),
            "training_macro_f1": float(row.get("Training Macro F1", 0)),
            "generalization_gap": float(row.get("Generalization Gap", 0)),
            "cv_std": float(row.get("CV Macro F1 Std", 0)),
        })

    vocabulary_rows = []
    for row in read_json(VOCAB_JSON, []):
        vocabulary_rows.append({
            "feature_limit": int(row["max_features_limit"]),
            "actual_vocabulary_size": int(row["actual_vocabulary_size"]),
            "cv_macro_f1": float(row["mean_cv_macro_f1"]),
            "training_macro_f1": float(row["mean_train_macro_f1"]),
            "generalization_gap": float(row["generalization_gap"]),
            "cv_std": float(row["std_cv_macro_f1"]),
            "json_size_mb": float(row["browser_json_mb"]),
        })

    browser_bytes = BROWSER_MODEL.stat().st_size if BROWSER_MODEL.exists() else None
    browser_gzip_bytes = len(gzip.compress(BROWSER_MODEL.read_bytes())) if BROWSER_MODEL.exists() else None

    results = {
        "model": {
            "name": model_info.get("selected_model", "Logistic Regression (L2 / Ridge)"),
            "type": "TF-IDF unigrams + bigrams + Logistic Regression",
            "cv_macro_f1": float(model_info.get("cross_validation_macro_f1", 0)),
            "cv_training_macro_f1": float(model_info.get("cross_validation_training_macro_f1", 0)),
            "cv_generalization_gap": float(model_info.get("generalization_gap", 0)),
            "parameters": model_info.get("best_parameters", {}),
            "selection_note": model_info.get("selection_note", ""),
        },
        "dataset": {
            "source": "Fake Reviews Dataset — mexwell on Kaggle",
            "source_url": "https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset",
            "original_rows": int(len(raw)),
            "cleaned_rows": int(len(raw_clean)),
            "duplicates_removed": int(len(raw) - len(raw_clean)),
            "or_rows": int(counts.get("OR", 0)),
            "cg_rows": int(counts.get("CG", 0)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "categories": int(raw_clean["category"].nunique()) if "category" in raw_clean else None,
        },
        "held_out_test": {
            "metrics": metrics,
            "correct": int(evaluation.get("correct", 0)),
            "errors": int(evaluation.get("errors", 0)),
            "confusion_matrix": evaluation.get("confusion_matrix", [[0, 0], [0, 0]]),
            "confusion_matrix_labels": evaluation.get("confusion_matrix_labels", ["OR", "CG"]),
            "note": evaluation.get("evaluation_note", ""),
        },
        "final_model_comparison": comparison_rows,
        "tuned_model_comparison": tuning_rows,
        "linearsvc_vocabulary_tradeoff": vocabulary_rows,
        "browser_payload": {
            "json_bytes": browser_bytes,
            "json_mb": browser_bytes / (1024 ** 2) if browser_bytes is not None else None,
            "gzip_bytes_estimate": browser_gzip_bytes,
            "gzip_mb_estimate": browser_gzip_bytes / (1024 ** 2) if browser_gzip_bytes is not None else None,
        },
        "interpretation": {
            "or_label": "Original Review (OR)",
            "cg_label": "Computer-Generated (CG)",
            "warning": "The classifier predicts labels in this dataset. It does not verify authorship, deception, fraud, or review truthfulness.",
            "test_set_caveat": "The held-out test split was inspected during earlier development; it is not a pristine independent external validation.",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Website result data written: {OUT}")
    print(f"Selected model: {results['model']['name']}")
    print(f"Test Macro F1: {metrics.get('Macro F1-score', float('nan')):.4%}")
    if browser_bytes:
        print(f"Browser JSON: {browser_bytes / (1024 ** 2):.3f} MB; gzip estimate: {browser_gzip_bytes / (1024 ** 2):.3f} MB")


if __name__ == "__main__":
    main()
