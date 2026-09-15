import numpy as np
from sklearn.metrics import classification_report

from logistic_regression import LogisticRegression, classification_report_from_scratch


def synthetic_data(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(240, 3))
    scores = np.c_[X[:, 0], X[:, 1], -X[:, 0], -X[:, 1]]
    y = np.argmax(scores + rng.normal(scale=0.08, size=scores.shape), axis=1)
    return X, y


def test_model_takes_the_expected_input():
    """Required test 1: a two-dimensional numeric feature matrix can be fitted."""
    X, y = synthetic_data()
    model = LogisticRegression(num_epochs=250, learning_rate=0.1, batch_size=64)
    returned = model.fit(X, y)
    assert returned is model
    assert model.n_features_in_ == X.shape[1]
    assert np.isfinite(model.weights_).all()


def test_model_output_has_the_expected_shape():
    """Required test 2: one class and four probabilities are returned per row."""
    X, y = synthetic_data()
    model = LogisticRegression(num_epochs=250, learning_rate=0.1, batch_size=64).fit(X, y)
    assert model.predict(X[:9]).shape == (9,)
    assert model.predict_proba(X[:9]).shape == (9, 4)
    assert np.allclose(model.predict_proba(X[:9]).sum(axis=1), 1.0)


def test_scratch_report_matches_sklearn():
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 0, 3])
    ours = classification_report_from_scratch(y_true, y_pred, labels=[0, 1, 2, 3])
    theirs = classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0
    )
    for key in ["0", "1", "2", "3", "macro avg", "weighted avg"]:
        for metric in ["precision", "recall", "f1-score"]:
            assert np.isclose(ours[key][metric], theirs[key][metric])
    assert np.isclose(ours["accuracy"], theirs["accuracy"])


def test_l2_penalty_shrinks_non_intercept_weights():
    X, y = synthetic_data()
    plain = LogisticRegression(num_epochs=300, learning_rate=0.08, l2_lambda=0).fit(X, y)
    ridge = LogisticRegression(num_epochs=300, learning_rate=0.08, l2_lambda=0.02).fit(X, y)
    assert np.linalg.norm(ridge.weights_[1:]) < np.linalg.norm(plain.weights_[1:])

