"""trains the multiclass intrusion model"""

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


def _parse_max_features(value: str | None) -> str | float | int | None:
    if value in (None, "", "none", "None"):
        return None
    if value in {"sqrt", "log2"}:
        return value
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError("max_features must be sqrt, log2, none, an int, or a float") from exc
    if number.is_integer():
        if number < 1:
            raise ValueError("max_features int must be >= 1")
        return int(number)
    if not (0.0 < number <= 1.0):
        raise ValueError("max_features float must be in (0, 1]")
    return number


def build_classifier(
    model_name: str,
    n_estimators: int = 100,
    max_depth: int | None = None,
    max_features: str | float | int | None = 7,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced_subsample",
    random_state: int = 42,
):
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced" if class_weight == "balanced_subsample" else class_weight,
            random_state=random_state,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced" if class_weight == "balanced_subsample" else class_weight,
            max_iter=1000,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight=class_weight,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        )

    supported = "decision_tree, extra_trees, logistic_regression, random_forest"
    raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}")


def train_model(
    train_path: str | Path = TRAIN_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    model_name: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int | None = None,
    max_features: str | float | int | None = 7,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced_subsample",
    random_state: int = 42,
) -> Pipeline:
    train_df, _ = load_data(train_path, test_path)
    X_train, y_train = split_features_target(train_df)

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(list(X_train.columns))),
            (
                "classifier",
                build_classifier(
                    model_name=model_name,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
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
    parser.add_argument("--n-estimators", default=100, type=int)
    parser.add_argument("--max-depth", default=None, type=int)
    parser.add_argument("--max-features", default="7")
    parser.add_argument("--min-samples-split", default=2, type=int)
    parser.add_argument("--min-samples-leaf", default=1, type=int)
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced", "balanced_subsample"],
        default="balanced_subsample",
    )
    parser.add_argument("--random-state", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = train_model(
        train_path=args.train_path,
        test_path=args.test_path,
        model_path=args.model_path,
        model_name=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=_parse_max_features(args.max_features),
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=args.random_state,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saved {args.model} pipeline to {args.model_path}")
    print(f"Pipeline steps: {', '.join(model.named_steps)}")


if __name__ == "__main__":
    main()
