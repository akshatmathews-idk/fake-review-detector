"""Evaluate the final Logistic Regression model on the existing test split."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from preprocessing import validate_processed_frame

ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "final" / "final_logistic_regression_pipeline.joblib"
RESULT_DIR = ROOT / "results" / "final_logistic_regression"
METRICS_DIR = RESULT_DIR / "metrics"
PREDICTIONS_DIR = RESULT_DIR / "predictions"


def main() -> None:
    for path in (TEST_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}. Run prepare_data.py and train.py first.")

    test_df = validate_processed_frame(pd.read_csv(TEST_PATH))
    X_test, y_test = test_df["text"], test_df["target"]
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    probability_cg = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (CG)": precision_score(y_test, y_pred, pos_label=1),
        "Recall (CG)": recall_score(y_test, y_pred, pos_label=1),
        "F1-score (CG)": f1_score(y_test, y_pred, pos_label=1),
        "Macro F1-score": f1_score(y_test, y_pred, average="macro"),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, probability_cg),
    }
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"Metric": k, "Score": float(v)} for k, v in metrics.items()]).to_csv(
        METRICS_DIR / "final_test_metrics.csv", index=False
    )
    pd.DataFrame({
        "text": X_test,
        "actual": y_test,
        "predicted": y_pred,
        "probability_cg": probability_cg,
        "correct": (y_test.to_numpy() == y_pred),
    }).to_csv(PREDICTIONS_DIR / "final_test_predictions.csv", index=False)

    evaluation = {
        "model": "Logistic Regression (L2 / Ridge)",
        "test_rows": int(len(test_df)),
        "correct": int((y_test.to_numpy() == y_pred).sum()),
        "errors": int((y_test.to_numpy() != y_pred).sum()),
        "confusion_matrix_labels": ["OR", "CG"],
        "confusion_matrix": cm.tolist(),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "evaluation_note": (
            "The test split was inspected during earlier development. These results are not "
            "independent external validation."
        ),
    }
    (RESULT_DIR / "final_evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    print("TEST CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["OR", "CG"], digits=4))
    print("Confusion matrix (rows=actual OR, CG; columns=predicted OR, CG):")
    print(cm)
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4%}")
    print(f"\nSaved evaluation under: {RESULT_DIR}")


if __name__ == "__main__":
    main()
