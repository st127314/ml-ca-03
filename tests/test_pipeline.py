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

