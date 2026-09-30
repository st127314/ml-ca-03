from pathlib import Path

import numpy as np

from train import build_pipeline, load_split

ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline_accepts_raw_car_columns_and_predicts_four_classes():
    X_train, X_test, y_train, _, _ = load_split(ROOT / "datasets" / "Cars.csv")
    model = build_pipeline(l2_lambda=1e-4, num_epochs=10)
    model.fit(X_train.iloc[:400], y_train[:400])
    predictions = model.predict(X_test.iloc[:7])
    assert predictions.shape == (7,)
    assert set(np.unique(predictions)).issubset({0, 1, 2, 3})


def test_a1_a2_missing_indicators_are_preserved():
    X_train, _, _, _, _ = load_split(ROOT / "datasets" / "Cars.csv")
    preprocessor = build_pipeline(num_epochs=1).named_steps["preprocessor"]
    preprocessor.fit(X_train)
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    assert numeric_imputer.add_indicator
    assert len(numeric_imputer.indicator_.features_) > 0
    assert "name" not in X_train and "torque" not in X_train
    assert "brand" in X_train
