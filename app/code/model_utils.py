"""Load the fitted A3 pipeline and expose small helpers for the Dash page."""

from pathlib import Path
import hashlib
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from dash import dcc, html

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "car_price_classifier.joblib"
MODEL_NAME = "st127314-a3-model"
LOGGER = logging.getLogger(__name__)


def load_registry_model(tracking_uri):
    """Load the current packaged model from one MLflow registry."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "production")
        if version.tags.get("artifact_sha256") == digest:
            return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    except mlflow.exceptions.RestException as exc:
        if exc.error_code not in {"RESOURCE_DOES_NOT_EXIST", "ENDPOINT_NOT_FOUND"}:
            raise

    packaged_model = joblib.load(MODEL_PATH)
    mlflow.set_experiment("st127314-a3")
    with mlflow.start_run(run_name="production-model") as run:
        mlflow.sklearn.log_model(packaged_model, artifact_path="model", serialization_format="cloudpickle")
        registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, registered.version, "artifact_sha256", digest)
    client.set_registered_model_alias(MODEL_NAME, "production", registered.version)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")


def load_serving_model():
    """Prefer configured remote MLflow, then local MLflow, then the packaged model."""
    remote_uri = os.environ.get("MLFLOW_REMOTE_TRACKING_URI")
    local_uri = os.environ.get("MLFLOW_TRACKING_URI")
    # Avoid a long app startup when an optional remote server is offline.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    for tracking_uri in dict.fromkeys(uri for uri in (remote_uri, local_uri) if uri):
        try:
            return load_registry_model(tracking_uri)
        except (mlflow.exceptions.MlflowException, OSError) as exc:
            LOGGER.warning("MLflow at %s is unavailable: %s", tracking_uri, exc)
    return joblib.load(MODEL_PATH)


model = load_serving_model()

preprocessor = model.named_steps["preprocessor"]
categorical_transformer = preprocessor.named_transformers_["categorical"]
category_values = categorical_transformer.named_steps["onehot"].categories_
categorical_columns = preprocessor.transformers_[1][2]
category_options = {
    column: sorted(values.tolist())
    for column, values in zip(categorical_columns, category_values)
}

OWNER_OPTIONS = [
    {"label": "First Owner", "value": 1},
    {"label": "Second Owner", "value": 2},
    {"label": "Third Owner", "value": 3},
    {"label": "Fourth & Above Owner", "value": 4},
]


def class_description(class_id):
    edges = model.price_bin_edges_
    low, high = edges[class_id], edges[class_id + 1]
    if not np.isfinite(low):
        return f"Budget (up to {high:,.0f})"
    if not np.isfinite(high):
        return f"Premium (above {low:,.0f})"
    return f"{low:,.0f} to {high:,.0f}"


def field_label(text):
    return html.Label(text, className="field-label")


def dropdown_component(component_id, label, options, placeholder):
    return html.Div([
        field_label(label),
        dcc.Dropdown(
            id=component_id,
            options=[{"label": option, "value": option} for option in options],
            placeholder=placeholder,
            clearable=True,
        ),
    ], className="form-field")


def number_component(component_id, label, placeholder="Optional", **kwargs):
    return html.Div([
        field_label(label),
        dcc.Input(id=component_id, type="number", placeholder=placeholder, **kwargs),
    ], className="form-field")
