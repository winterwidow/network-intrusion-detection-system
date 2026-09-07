"""trains, evaluates, and logs one every run"""

import argparse
from datetime import datetime
from pathlib import Path

from .constants import MODELS_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH
from .evaluate import evaluate_model
from .experiment_tracker import EXPERIMENTS_PATH, log_experiment
from .train import _parse_max_features, train_model


def build_run_id(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{model_name}"


def run_experiment(
    model_name: str,
    n_estimators: int,
    max_depth: int | None,
    max_features: str | float | int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    class_weight: str | None,
    random_state: int,
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    experiments_path: Path = EXPERIMENTS_PATH,
) -> dict:
    run_id = build_run_id(model_name)
    model_path = MODELS_DIR / f"{run_id}.joblib"

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight,
        "random_state": random_state,
    }

    train_model(
        train_path=train_path,
        test_path=test_path,
        model_path=model_path,
        model_name=model_name,
        **params,
    )
    metrics = evaluate_model(
        model_path=model_path,
        test_path=test_path,
        metrics_path=None,
    )
    row = log_experiment(
        run_id=run_id,
        model_name=model_name,
        params=params,
        metrics=metrics,
        model_path=model_path,
        metrics_path=None,
        experiments_path=experiments_path,
    )

    print(f"Logged experiment to {experiments_path}")
    print(
        "Summary: "
        f"accuracy={row['accuracy']:.4f}, "
        f"macro_f1={row['macro_f1']:.4f}, "
        f"weighted_f1={row['weighted_f1']:.4f}"
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, and log one experiment.")
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
    parser.add_argument("--train-path", default=TRAIN_DATA_PATH, type=Path)
    parser.add_argument("--test-path", default=TEST_DATA_PATH, type=Path)
    parser.add_argument("--experiments-path", default=EXPERIMENTS_PATH, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        model_name=args.model,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=_parse_max_features(args.max_features),
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=args.random_state,
        train_path=args.train_path,
        test_path=args.test_path,
        experiments_path=args.experiments_path,
    )


if __name__ == "__main__":
    main()
