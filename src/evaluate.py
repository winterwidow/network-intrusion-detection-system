"""reports multiclass performance metrics"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .constants import (
    ATTACK_ID_TO_CLASS,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    REPORTS_DIR,
    TEST_DATA_PATH,
)
from .data_loader import load_dataset
from .preprocess import split_features_target


def evaluate_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_path: str | Path = TEST_DATA_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict:
    model = joblib.load(model_path)
    test_df = load_dataset(test_path)
    X_test, y_test = split_features_target(test_df)

    y_pred = model.predict(X_test)
    labels = sorted(ATTACK_ID_TO_CLASS)
    target_names = [ATTACK_ID_TO_CLASS[label] for label in labels]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": {str(label): ATTACK_ID_TO_CLASS[label] for label in labels},
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_frame = pd.DataFrame(metrics["classification_report"]).T
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(report_frame.to_string())
    print(f"Saved metrics to {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved NSL-KDD model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument("--test-path", default=TEST_DATA_PATH, type=Path)
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        test_path=args.test_path,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
