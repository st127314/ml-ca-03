import dash
from dash import dcc, html

from model_utils import class_description, model

dash.register_page(__name__, path="/", name="Home")

metrics = model.test_report_

layout = html.Div([
    html.Div([
        html.P("A3 - MULTINOMIAL LOGISTIC REGRESSION", className="eyebrow"),
        html.H1("Which price band does this used car belong to?"),
        html.P(
            "A from-scratch softmax classifier places a car into one of four price bands. "
            "The bands are quartiles learned only from the training prices, keeping the "
            "held-out test set out of preprocessing decisions.",
            className="lede",
        ),
        dcc.Link(html.Button("Classify a car", className="predict-button"), href="/predict"),
    ], className="hero"),
    html.Div([
        html.Div([
            html.H2("Four prediction classes"),
            html.Div([
                html.Div([
                    html.P(f"Class {class_id}", className="stat-value"),
                    html.P(class_description(class_id), className="stat-label"),
                ], className="stat-card")
                for class_id in range(4)
            ], className="stat-grid"),
        ], className="card"),
        html.Div([
            html.H2("How the model was built"),
            html.P(
                "The data cleaning is carried over from A1 and A2. Numeric fields are "
                "median-imputed and standardised; categorical fields are most-frequent "
                "imputed and one-hot encoded. The classifier itself uses only NumPy: "
                "stable softmax probabilities, cross-entropy loss, mini-batch gradient "
                "descent, and an optional L2 penalty that excludes the intercept."
            ),
            html.P(
                "The experiment compares ordinary and ridge logistic regression. "
                "Selection uses macro F1 so each price class counts equally. On the "
                f"held-out set the selected model reaches {metrics['accuracy']:.3f} "
                f"accuracy and {metrics['macro avg']['f1-score']:.3f} macro F1."
            ),
        ], className="card"),
        html.Div([
            html.H2("What support means"),
            html.P(
                "Support is the number of true examples of a class in the evaluated "
                "dataset. It is not a score. Scikit-learn uses these counts as weights "
                "for its weighted-average precision, recall, and F1."
            ),
        ], className="card"),
    ], className="content-stack"),
], className="page")

