import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .constants import (
    DEFAULT_ANOMALY_METRICS_PATH,
    DEFAULT_ANOMALY_MODEL_PATH,
    NORMAL_CLASS_ID,
    TEST_DATA_PATH,
)
from .data_loader import load_dataset
from .preprocess import split_features_target


def evaluate_anomaly_model(
    model_path: str | Path = DEFAULT_ANOMALY_MODEL_PATH,
    test_path: str | Path = TEST_DATA_PATH,
    metrics_path: str | Path = DEFAULT_ANOMALY_METRICS_PATH,
) -> dict:
    model = joblib.load(model_path)
    test_df = load_dataset(test_path)
    X_test, y_test = split_features_target(test_df)

    y_true = (y_test != NORMAL_CLASS_ID).astype(int)
    raw_predictions = model.predict(X_test)
    y_pred = (raw_predictions == -1).astype(int)

    anomaly_scores = -model.decision_function(X_test)
    metrics = {
        "roc_auc": roc_auc_score(y_true, anomaly_scores),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["normal", "anomaly"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "labels": {"0": "normal", "1": "anomaly"},
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_frame = pd.DataFrame(metrics["classification_report"]).T
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(report_frame.to_string())
    print(f"Saved anomaly metrics to {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved anomaly detector.")
    parser.add_argument("--model-path", default=DEFAULT_ANOMALY_MODEL_PATH, type=Path)
    parser.add_argument("--test-path", default=TEST_DATA_PATH, type=Path)
    parser.add_argument("--metrics-path", default=DEFAULT_ANOMALY_METRICS_PATH, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_anomaly_model(
        model_path=args.model_path,
        test_path=args.test_path,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
