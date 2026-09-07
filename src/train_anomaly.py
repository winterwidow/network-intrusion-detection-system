"""trains an Isolation Forest on normal traffic only to detect anomalies"""

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from .constants import DEFAULT_ANOMALY_MODEL_PATH, NORMAL_CLASS_ID, TRAIN_DATA_PATH
from .data_loader import load_dataset
from .preprocess import build_preprocessor, split_features_target


def train_anomaly_model(
    train_path: str | Path = TRAIN_DATA_PATH,
    model_path: str | Path = DEFAULT_ANOMALY_MODEL_PATH,
    contamination: float = 0.02,
) -> Pipeline:
    train_df = load_dataset(train_path)
    X_train, y_train = split_features_target(train_df)
    normal_X_train = X_train.loc[y_train == NORMAL_CLASS_ID]

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(list(X_train.columns))),
            (
                "detector",
                IsolationForest(
                    contamination=contamination,
                    n_estimators=200,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(normal_X_train)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an anomaly detector on normal NSL-KDD traffic."
    )
    parser.add_argument("--train-path", default=TRAIN_DATA_PATH, type=Path)
    parser.add_argument("--model-path", default=DEFAULT_ANOMALY_MODEL_PATH, type=Path)
    parser.add_argument(
        "--contamination",
        default=0.02,
        type=float,
        help="Expected fraction of unusual records inside the normal-only training data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_anomaly_model(
        train_path=args.train_path,
        model_path=args.model_path,
        contamination=args.contamination,
    )
    print(f"Saved anomaly detection pipeline to {args.model_path}")


if __name__ == "__main__":
    main()
