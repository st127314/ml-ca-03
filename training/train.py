"""Train, compare, log, register, and save the A3 classifier.

Run from the repository root.  Remote MLflow writes happen only when ``--log-mlflow``
is supplied, so tests and local model builds never mutate the course server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
APP_CODE = ROOT / "app" / "code"
sys.path.insert(0, str(APP_CODE))

from data_prep import bucket_prices, load_clean_data, price_bin_edges  # noqa: E402
from logistic_regression import LogisticRegression, classification_report_from_scratch  # noqa: E402

STUDENT_ID = "st127314"
TRACKING_URI = "http://mlflow.ml.brain.cs.ait.ac.th/"
EXPERIMENT_NAME = f"{STUDENT_ID}-a3"
MODEL_NAME = f"{STUDENT_ID}-a3-model"
NUMERIC = ["year", "km_driven", "owner", "mileage", "engine", "max_power", "seats"]
CATEGORICAL = ["brand", "fuel", "seller_type", "transmission"]


def build_pipeline(l2_lambda=0.0, learning_rate=0.05, num_epochs=600):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=256,
            l2_lambda=l2_lambda,
            random_state=42,
        )),
    ])


def load_split(dataset_path):
    df = load_clean_data(dataset_path, verbose=False)
    X = df.drop(columns="selling_price")
    prices = df["selling_price"]
    X_train, X_test, price_train, price_test = train_test_split(
        X, prices, test_size=0.20, random_state=42
    )
    edges = price_bin_edges(price_train)
    y_train = bucket_prices(price_train, edges).to_numpy()
    y_test = bucket_prices(price_test, edges).to_numpy()
    return X_train, X_test, y_train, y_test, edges


def load_partitions(dataset_path):
    """Outer test split plus an inner validation split used for model selection."""
    X_train, X_test, y_train, y_test, edges = load_split(dataset_path)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    return X_fit, X_val, X_train, X_test, y_fit, y_val, y_train, y_test, edges


def register_run(run_id, model_name=MODEL_NAME):
    """Register a logged run and move it to Staging as requested in the brief."""
    import mlflow

    registered = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    return registered.version


def train(args):
    X_fit, X_val, X_train, X_test, y_fit, y_val, y_train, y_test, edges = load_partitions(args.dataset)
    candidates = [
        {"l2_lambda": penalty, "learning_rate": rate}
        for penalty in (0.0, 1e-5, 1e-4, 1e-3)
        for rate in (0.03, 0.05)
    ]
    mlflow = None
    if args.log_mlflow:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

    results = []
    best = None
    for params in candidates:
        pipeline = build_pipeline(**params, num_epochs=args.epochs)
        run_context = mlflow.start_run(run_name=f"l2={params['l2_lambda']}-lr={params['learning_rate']}") if mlflow else None
        try:
            pipeline.fit(X_fit, y_fit)
            predictions = pipeline.predict(X_val)
            report = classification_report_from_scratch(y_val, predictions, labels=[0, 1, 2, 3])
            row = {
                **params,
                "validation_accuracy": report["accuracy"],
                "validation_macro_f1": report["macro avg"]["f1-score"],
                "validation_weighted_f1": report["weighted avg"]["f1-score"],
            }
            results.append(row)
            if mlflow:
                mlflow.log_params(params)
                mlflow.log_metrics({key: value for key, value in row.items() if key.startswith("validation_")})
            if best is None or row["validation_macro_f1"] > best[0]["validation_macro_f1"]:
                best = (row, pipeline)
        finally:
            if run_context is not None:
                mlflow.end_run()

    best_row, _ = best
    best_pipeline = build_pipeline(
        l2_lambda=best_row["l2_lambda"],
        learning_rate=best_row["learning_rate"],
        num_epochs=args.epochs,
    )
    best_pipeline.fit(X_train, y_train)
    test_predictions = best_pipeline.predict(X_test)
    scratch_report = classification_report_from_scratch(y_test, test_predictions, labels=[0, 1, 2, 3])
    test_metrics = {
        "test_accuracy": scratch_report["accuracy"],
        "test_macro_f1": scratch_report["macro avg"]["f1-score"],
        "test_weighted_f1": scratch_report["weighted avg"]["f1-score"],
    }
    # Attach serving metadata before either MLflow or joblib serialises the pipeline.
    best_pipeline.price_bin_edges_ = np.asarray(edges)
    best_pipeline.test_report_ = scratch_report
    best_pipeline.training_config_ = {**best_row, **test_metrics}
    best_run_id = None
    if mlflow:
        with mlflow.start_run(run_name="best-model-refit") as final_run:
            mlflow.log_params({key: best_row[key] for key in ("l2_lambda", "learning_rate")})
            mlflow.log_metrics(test_metrics)
            mlflow.sklearn.log_model(best_pipeline, "model")
            best_run_id = final_run.info.run_id
    # Metadata stays on the Pipeline so the deployed app needs only one artifact.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, args.output, compress=3)

    sklearn_report = classification_report(
        y_test, best_pipeline.predict(X_test), labels=[0, 1, 2, 3], output_dict=True, zero_division=0
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "registered_model": MODEL_NAME,
        "price_bin_edges": [None if not np.isfinite(value) else float(value) for value in edges],
        "best": {**best_row, **test_metrics},
        "all_runs": results,
        "scratch_matches_sklearn": bool(
            np.isclose(scratch_report["macro avg"]["f1-score"], sklearn_report["macro avg"]["f1-score"])
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if args.register:
        if not best_run_id:
            raise ValueError("--register requires --log-mlflow")
        version = register_run(best_run_id)
        print(f"registered {MODEL_NAME} version {version} at Staging")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "Cars.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "app" / "models" / "car_price_classifier.joblib")
    parser.add_argument("--summary", type=Path, default=ROOT / "figures" / "experiment_results.json")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
