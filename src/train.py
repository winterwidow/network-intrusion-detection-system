import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .constants import DEFAULT_MODEL_PATH, MODELS_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH
from .data_loader import load_data
from .preprocess import build_preprocessor, split_features_target


def build_classifier(model_name: str):
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=8,
            max_features=8,
            random_state=42,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=100,
            max_features=7,
            n_jobs=-1,
            random_state=42,
        )
    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            max_features=7,
            n_estimators=100,
            n_jobs=-1,
            random_state=42,
        )

    supported = "decision_tree, extra_trees, logistic_regression, random_forest"
    raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}")


def train_model(
    train_path: str | Path = TRAIN_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    model_name: str = "random_forest",
) -> Pipeline:
    train_df, _ = load_data(train_path, test_path)
    X_train, y_train = split_features_target(train_df)

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(list(X_train.columns))),
            ("classifier", build_classifier(model_name)),
        ]
    )
    model.fit(X_train, y_train)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NSL-KDD intrusion classifier.")
    parser.add_argument("--train-path", default=TRAIN_DATA_PATH, type=Path)
    parser.add_argument("--test-path", default=TEST_DATA_PATH, type=Path)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument(
        "--model",
        choices=["decision_tree", "extra_trees", "logistic_regression", "random_forest"],
        default="random_forest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = train_model(
        train_path=args.train_path,
        test_path=args.test_path,
        model_path=args.model_path,
        model_name=args.model,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saved {args.model} pipeline to {args.model_path}")
    print(f"Pipeline steps: {', '.join(model.named_steps)}")


if __name__ == "__main__":
    main()
