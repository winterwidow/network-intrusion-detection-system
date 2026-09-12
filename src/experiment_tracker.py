"""appends experiment results to a CSV tracker"""

import csv
import json
from pathlib import Path
from typing import Any

from .constants import ATTACK_ID_TO_CLASS, REPORTS_DIR


EXPERIMENTS_PATH = REPORTS_DIR / "experiments.csv"


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    report = metrics["classification_report"]
    row = {
        "accuracy": metrics["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    for class_name in ATTACK_ID_TO_CLASS.values():
        class_metrics = report.get(class_name, {})
        row[f"{class_name}_precision"] = class_metrics.get("precision", 0)
        row[f"{class_name}_recall"] = class_metrics.get("recall", 0)
        row[f"{class_name}_f1"] = class_metrics.get("f1-score", 0)

    return row


def log_experiment(
    run_id: str,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    model_path: str | Path,
    metrics_path: str | Path | None = None,
    experiments_path: str | Path = EXPERIMENTS_PATH,
) -> dict[str, Any]:
    experiments_path = Path(experiments_path)
    experiments_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "run_id": run_id,
        "model_name": model_name,
        "params": json.dumps(params, sort_keys=True),
        "model_path": str(model_path),
        "metrics_path": "" if metrics_path is None else str(metrics_path),
    }
    row.update(_flatten_metrics(metrics))

    file_exists = experiments_path.exists() and experiments_path.stat().st_size > 0
    with experiments_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row
