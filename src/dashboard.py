"""Interactive monitoring dashboard for the NSL-KDD intrusion models.

The dashboard currently replays batches from the test dataset.  It uses the
same scoring path that a future packet-to-flow collector will use, so the UI
can be developed and demonstrated before live capture is connected.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# Streamlit executes this file from the src directory, so expose the package root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    CATEGORICAL_FEATURES,
    DEFAULT_ANOMALY_METRICS_PATH,
    DEFAULT_ANOMALY_MODEL_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    TEST_DATA_PATH,
)
from src.data_loader import load_dataset
from src.preprocess import decode_attack_class, split_features_target


st.set_page_config(page_title="Intrusion Monitoring Console", page_icon="IDS", layout="wide")


@st.cache_resource(show_spinner="Loading trained models...")
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner="Loading NSL-KDD replay data...")
def load_replay_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = load_dataset(path)
    return split_features_target(frame)


@st.cache_data(show_spinner="Scoring replay batch...")
def score_replay_batch(
    model_path: str,
    anomaly_model_path: str,
    start_index: int,
    batch_size: int,
) -> pd.DataFrame:
    model = load_model(model_path)
    anomaly_model = load_model(anomaly_model_path)
    features, actual_classes = load_replay_data(str(TEST_DATA_PATH))

    indices = [(start_index + offset) % len(features) for offset in range(batch_size)]
    batch = features.iloc[indices].copy().reset_index(drop=True)
    actual = actual_classes.iloc[indices].reset_index(drop=True)
    predictions = model.predict(batch)
    probabilities = model.predict_proba(batch)
    anomaly_predictions = anomaly_model.predict(batch)
    anomaly_scores = -anomaly_model.decision_function(batch)

    alerts = pd.DataFrame(
        {
            "alert_id": [f"replay-{start_index + offset:06d}" for offset in range(batch_size)],
            "replay_record": [start_index + offset for offset in range(batch_size)],
            "predicted_attack_class": [decode_attack_class(value) for value in predictions],
            "actual_attack_class": [decode_attack_class(value) for value in actual],
            "confidence": probabilities.max(axis=1),
            "is_anomaly": anomaly_predictions == -1,
            "anomaly_score": anomaly_scores,
        }
    )
    alerts = pd.concat([alerts, batch], axis=1)
    alerts["risk_level"] = alerts.apply(_risk_level, axis=1)
    return alerts.sort_values(["risk_level", "confidence"], ascending=[True, False])


def _risk_level(alert: pd.Series) -> str:
    is_attack = alert["predicted_attack_class"] != "normal"
    if is_attack and alert["is_anomaly"] and alert["confidence"] >= 0.85:
        return "critical"
    if is_attack or alert["is_anomaly"]:
        return "high"
    return "normal"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _available_models() -> list[Path]:
    return sorted(MODELS_DIR.glob("*.joblib"))


def _feature_importance(model_path: str) -> pd.DataFrame:
    model = load_model(model_path)
    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocess")
    if classifier is None or preprocessor is None or not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
    importance["feature"] = importance["feature"].str.removeprefix("numeric__").str.removeprefix("categorical__")
    for feature in CATEGORICAL_FEATURES:
        importance.loc[importance["feature"].str.startswith(f"{feature}_"), "feature"] = feature
    return importance.groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)


def _show_overview(alerts: pd.DataFrame) -> None:
    alert_mask = (alerts["predicted_attack_class"] != "normal") | alerts["is_anomaly"]
    metrics = st.columns(4)
    metrics[0].metric("Connections replayed", len(alerts))
    metrics[1].metric("Security alerts", int(alert_mask.sum()))
    metrics[2].metric("Anomalies", int(alerts["is_anomaly"].sum()))
    metrics[3].metric("Average confidence", f"{alerts['confidence'].mean():.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted traffic classes")
        class_counts = alerts["predicted_attack_class"].value_counts().rename_axis("class").to_frame("connections")
        st.dataframe(class_counts, width="stretch")
    with right:
        st.subheader("Alert severity")
        risk_counts = alerts["risk_level"].value_counts().reindex(["critical", "high", "normal"], fill_value=0)
        st.dataframe(risk_counts.rename_axis("risk_level").to_frame("connections"), width="stretch")

    st.subheader("Latest flagged connections")
    flagged = alerts.loc[alert_mask, _alert_columns()].head(12)
    if flagged.empty:
        st.success("No alerts in this replay batch.")
    else:
        st.dataframe(flagged, hide_index=True, width="stretch")


def _alert_columns() -> list[str]:
    return [
        "alert_id",
        "risk_level",
        "predicted_attack_class",
        "actual_attack_class",
        "confidence",
        "is_anomaly",
        "anomaly_score",
        "protocol_type",
        "service",
        "src_bytes",
        "dst_bytes",
    ]


def _show_alert_queue(alerts: pd.DataFrame) -> None:
    st.subheader("Alert queue")
    filter_columns = st.columns(3)
    selected_classes = filter_columns[0].multiselect(
        "Predicted class", sorted(alerts["predicted_attack_class"].unique()), default=sorted(alerts["predicted_attack_class"].unique())
    )
    minimum_confidence = filter_columns[1].slider("Minimum confidence", 0.0, 1.0, 0.50, 0.05)
    anomalies_only = filter_columns[2].toggle("Anomalies only")

    filtered = alerts[
        alerts["predicted_attack_class"].isin(selected_classes)
        & (alerts["confidence"] >= minimum_confidence)
        & (alerts["is_anomaly"] if anomalies_only else True)
    ]
    st.dataframe(filtered[_alert_columns()], hide_index=True, width="stretch")
    st.download_button(
        "Download filtered alerts",
        data=filtered.to_csv(index=False),
        file_name="nids_alerts.csv",
        mime="text/csv",
    )

    if filtered.empty:
        return

    selected_id = st.selectbox("Inspect a connection", filtered["alert_id"].tolist())
    selected = filtered.loc[filtered["alert_id"] == selected_id].iloc[0]
    details, flow = st.columns([1, 2])
    with details:
        st.subheader("Alert details")
        st.write(f"Prediction: `{selected['predicted_attack_class']}`")
        st.write(f"Confidence: `{selected['confidence']:.1%}`")
        st.write(f"Risk: `{selected['risk_level']}`")
        st.write(f"Anomaly score: `{selected['anomaly_score']:.4f}`")
        st.caption("Actual class is visible only because this is a labelled NSL-KDD replay.")
    with flow:
        st.subheader("Connection features")
        feature_names = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
            "count", "srv_count", "serror_rate", "rerror_rate", "dst_host_count",
            "dst_host_srv_count",
        ]
        feature_values = pd.DataFrame(
            {"feature": feature_names, "value": selected[feature_names].astype(str).tolist()}
        )
        st.dataframe(feature_values, hide_index=True, width="stretch")


def _show_model_health(model_path: str) -> None:
    st.subheader("Saved model evaluation")
    metrics = _read_json(DEFAULT_METRICS_PATH)
    anomaly_metrics = _read_json(DEFAULT_ANOMALY_METRICS_PATH)
    if metrics:
        evaluation = metrics.get("classification_report", {})
        summary = st.columns(3)
        summary[0].metric("Classifier accuracy", f"{metrics.get('accuracy', 0):.1%}")
        summary[1].metric("Classifier macro F1", f"{evaluation.get('macro avg', {}).get('f1-score', 0):.3f}")
        summary[2].metric("Anomaly ROC-AUC", f"{anomaly_metrics.get('roc_auc', 0):.3f}")

        rows = []
        for name in ["normal", "dos", "probe", "r2l", "u2r"]:
            item = evaluation.get(name, {})
            rows.append({"class": name, "precision": item.get("precision", 0), "recall": item.get("recall", 0), "f1": item.get("f1-score", 0)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("Run `python -m src.evaluate` to add saved evaluation metrics.")

    experiments_path = REPORTS_DIR / "experiments.csv"
    if experiments_path.exists():
        st.subheader("Experiment tracker")
        experiments = pd.read_csv(experiments_path)
        columns = [column for column in ["run_id", "model_name", "accuracy", "macro_f1", "weighted_f1"] if column in experiments]
        st.dataframe(experiments.sort_values("macro_f1", ascending=False)[columns], hide_index=True, width="stretch")

    st.caption(f"Currently selected model: {Path(model_path).name}")


def _show_feature_signals(model_path: str) -> None:
    st.subheader("Global model signals")
    importance = _feature_importance(model_path)
    if importance.empty:
        st.info("Global importance is available for tree-based models.")
        return
    st.dataframe(importance.head(15), hide_index=True, width="stretch")
    st.caption("These are built-in tree importances. They show model reliance, not causal proof.")


def main() -> None:
    st.title("Intrusion Monitoring Console")
    st.caption("Simulated live replay using NSL-KDD test traffic, the intrusion classifier, and the anomaly detector.")

    model_paths = _available_models()
    if not model_paths or not DEFAULT_ANOMALY_MODEL_PATH.exists():
        st.error("Train the classifier and anomaly models before starting the dashboard.")
        return

    default_model_index = model_paths.index(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH in model_paths else 0
    with st.sidebar:
        st.header("Replay controls")
        selected_model = st.selectbox("Classifier model", model_paths, index=default_model_index, format_func=lambda path: path.name)
        batch_size = st.select_slider("Connections per replay", options=[25, 50, 100, 200, 500], value=100)
        if "replay_offset" not in st.session_state:
            st.session_state.replay_offset = 0
        controls = st.columns(2)
        if controls[0].button("Replay next batch", type="primary", width="stretch"):
            st.session_state.replay_offset += batch_size
        if controls[1].button("Reset", width="stretch"):
            st.session_state.replay_offset = 0
        st.caption("A real capture worker can later write to the same alert format shown here.")

    alerts = score_replay_batch(
        str(selected_model),
        str(DEFAULT_ANOMALY_MODEL_PATH),
        st.session_state.replay_offset,
        batch_size,
    )
    overview, queue, health, signals = st.tabs(["Overview", "Alert queue", "Model health", "Feature signals"])
    with overview:
        _show_overview(alerts)
    with queue:
        _show_alert_queue(alerts)
    with health:
        _show_model_health(str(selected_model))
    with signals:
        _show_feature_signals(str(selected_model))


if __name__ == "__main__":
    main()
