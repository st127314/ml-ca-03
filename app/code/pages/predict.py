import dash
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from data_prep import normalise_missing
from model_utils import (
    OWNER_OPTIONS,
    category_options,
    class_description,
    dropdown_component,
    field_label,
    model,
    number_component,
)

dash.register_page(__name__, path="/predict", name="Predict")

layout = html.Div([
    html.Div([
        html.P("CAR PRICE CLASSIFIER", className="eyebrow"),
        html.H1("Find the likely selling-price band"),
        html.P(
            "Year and max power are required. Missing optional fields are imputed by the "
            "same fitted preprocessing pipeline used during training.",
            className="lede",
        ),
    ], className="hero"),
    html.Div([
        html.Div([
            html.H2("Vehicle details"),
            html.Div([
                dropdown_component("brand", "Brand", category_options["brand"], "Select a brand"),
                number_component("year", "Year (required)", "e.g. 2017", min=1980, max=2030, required=True),
                number_component("km-driven", "Kilometres driven", min=0),
                dropdown_component("fuel", "Fuel", category_options["fuel"], "Select fuel"),
                dropdown_component("seller-type", "Seller type", category_options["seller_type"], "Select seller"),
                dropdown_component("transmission", "Transmission", category_options["transmission"], "Select transmission"),
                html.Div([
                    field_label("Owner history"),
                    dcc.Dropdown(id="owner", options=OWNER_OPTIONS, placeholder="Select owner history", clearable=True),
                ], className="form-field"),
                number_component("mileage", "Mileage (kmpl)", min=0),
                number_component("engine", "Engine (CC)", min=0),
                number_component("max-power", "Max power (bhp, required)", min=0, required=True),
                number_component("seats", "Seats", min=1, max=20),
            ], className="form-grid"),
            html.Button("Classify price", id="predict-button", n_clicks=0, className="predict-button"),
        ], className="card form-card"),
        html.Div([
            html.P("PREDICTION", className="eyebrow"),
            html.Div("Complete the form and click classify.", id="prediction-output", className="prediction-output"),
            html.P("Probabilities show the model's confidence across all four price bands.", className="helper-text"),
        ], className="card result-card"),
    ], className="content-grid"),
], className="page")


@callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("brand", "value"), State("year", "value"), State("km-driven", "value"),
    State("fuel", "value"), State("seller-type", "value"), State("transmission", "value"),
    State("owner", "value"), State("mileage", "value"), State("engine", "value"),
    State("max-power", "value"), State("seats", "value"),
)
def classify_price(n_clicks, brand, year, km_driven, fuel, seller_type,
                   transmission, owner, mileage, engine, max_power, seats):
    if not n_clicks:
        raise PreventUpdate
    if year is None or max_power is None:
        return html.Div("Please enter year and max power.", className="result-warning")

    row = normalise_missing(pd.DataFrame([{
        "brand": brand, "year": year, "km_driven": km_driven, "fuel": fuel,
        "seller_type": seller_type, "transmission": transmission, "owner": owner,
        "mileage": mileage, "engine": engine, "max_power": max_power, "seats": seats,
    }]))
    class_id = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    return html.Div([
        html.P(f"Class {class_id}", className="price-value"),
        html.P(class_description(class_id), className="result-label"),
        html.Ul([
            html.Li(f"Class {index}: {probability:.1%}")
            for index, probability in enumerate(probabilities)
        ], className="coef-list"),
    ])
