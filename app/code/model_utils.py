"""Load the fitted A3 pipeline and expose small helpers for the Dash page."""

from pathlib import Path

import joblib
import numpy as np
from dash import dcc, html

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "car_price_classifier.joblib"
model = joblib.load(MODEL_PATH)

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
