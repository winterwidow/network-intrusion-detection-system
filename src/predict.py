import argparse
from pathlib import Path

import pandas as pd
import joblib

from .constants import DEFAULT_MODEL_PATH, MODEL_FEATURE_COLUMNS, NSL_KDD_COLUMNS
from .preprocess import decode_attack_class


def _load_prediction_frame(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    if set(MODEL_FEATURE_COLUMNS).issubset(frame.columns):
        return frame[MODEL_FEATURE_COLUMNS]

    raw_frame = pd.read_csv(input_path, header=None)
    if raw_frame.shape[1] == len(NSL_KDD_COLUMNS):
        raw_frame.columns = NSL_KDD_COLUMNS
        return raw_frame[MODEL_FEATURE_COLUMNS]
    if raw_frame.shape[1] == len(MODEL_FEATURE_COLUMNS):
        raw_frame.columns = MODEL_FEATURE_COLUMNS
        return raw_frame

    raise ValueError(
        "Prediction input must be a header CSV with model feature columns, "
        "a raw NSL-KDD file, or a headerless feature-only CSV."
    )


def predict_file(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    model = joblib.load(model_path)
    input_path = Path(input_path)
    output_path = Path(output_path)

    frame = _load_prediction_frame(input_path)
    predictions = model.predict(frame)

    output = frame.copy()
    output["predicted_attack_class_id"] = predictions
    output["predicted_attack_class"] = [decode_attack_class(value) for value in predictions]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict attack classes for a CSV file.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = predict_file(args.input_path, args.output_path, args.model_path)
    print(f"Wrote {len(output)} predictions to {args.output_path}")


if __name__ == "__main__":
    main()
