"""Multinomial logistic regression and classification metrics from scratch.

Only NumPy is used for the optimisation and metric calculations.  The estimator
follows the small part of scikit-learn's API needed by ``Pipeline`` so the exact
same fitted object can be evaluated, logged to MLflow, and served by Dash.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _as_1d(values):
    return np.asarray(values).reshape(-1)


def accuracy(y_true, y_pred):
    """Fraction of predictions equal to the true class."""
    true, pred = _as_1d(y_true), _as_1d(y_pred)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and have the same shape")
    return float(np.mean(true == pred))


def precision(y_true, y_pred, labels=None):
    """Per-class TP / (TP + FP), returning zero when the denominator is zero."""
    true, pred = _as_1d(y_true), _as_1d(y_pred)
    labels = np.asarray(np.unique(np.r_[true, pred]) if labels is None else labels)
    scores = []
    for label in labels:
        tp = np.sum((true == label) & (pred == label))
        fp = np.sum((true != label) & (pred == label))
        scores.append(tp / (tp + fp) if tp + fp else 0.0)
    return np.asarray(scores, dtype=float)


def recall(y_true, y_pred, labels=None):
    """Per-class TP / (TP + FN), returning zero when the denominator is zero."""
    true, pred = _as_1d(y_true), _as_1d(y_pred)
    labels = np.asarray(np.unique(np.r_[true, pred]) if labels is None else labels)
    scores = []
    for label in labels:
        tp = np.sum((true == label) & (pred == label))
        fn = np.sum((true == label) & (pred != label))
        scores.append(tp / (tp + fn) if tp + fn else 0.0)
    return np.asarray(scores, dtype=float)


def f1_score(y_true, y_pred, labels=None):
    """Per-class harmonic mean of precision and recall."""
    p = precision(y_true, y_pred, labels)
    r = recall(y_true, y_pred, labels)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) != 0)


def _support(y_true, labels):
    true = _as_1d(y_true)
    return np.asarray([np.sum(true == label) for label in labels], dtype=int)


def macro_precision(y_true, y_pred, labels=None):
    return float(np.mean(precision(y_true, y_pred, labels)))


def macro_recall(y_true, y_pred, labels=None):
    return float(np.mean(recall(y_true, y_pred, labels)))


def macro_f1(y_true, y_pred, labels=None):
    return float(np.mean(f1_score(y_true, y_pred, labels)))


def _weighted(values, y_true, labels):
    supports = _support(y_true, labels)
    return float(np.average(values, weights=supports)) if supports.sum() else 0.0


def weighted_precision(y_true, y_pred, labels=None):
    labels = np.asarray(np.unique(np.r_[_as_1d(y_true), _as_1d(y_pred)]) if labels is None else labels)
    return _weighted(precision(y_true, y_pred, labels), y_true, labels)


def weighted_recall(y_true, y_pred, labels=None):
    labels = np.asarray(np.unique(np.r_[_as_1d(y_true), _as_1d(y_pred)]) if labels is None else labels)
    return _weighted(recall(y_true, y_pred, labels), y_true, labels)


def weighted_f1(y_true, y_pred, labels=None):
    labels = np.asarray(np.unique(np.r_[_as_1d(y_true), _as_1d(y_pred)]) if labels is None else labels)
    return _weighted(f1_score(y_true, y_pred, labels), y_true, labels)


def classification_report_from_scratch(y_true, y_pred, labels=None):
    """A compact dictionary with the same values as sklearn's report."""
    true, pred = _as_1d(y_true), _as_1d(y_pred)
    labels = np.asarray(np.unique(np.r_[true, pred]) if labels is None else labels)
    p, r, f = precision(true, pred, labels), recall(true, pred, labels), f1_score(true, pred, labels)
    support = _support(true, labels)
    report = {
        str(label): {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1-score": float(f[i]), "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    report["accuracy"] = accuracy(true, pred)
    report["macro avg"] = {
        "precision": float(p.mean()), "recall": float(r.mean()),
        "f1-score": float(f.mean()), "support": int(support.sum()),
    }
    report["weighted avg"] = {
        "precision": _weighted(p, true, labels),
        "recall": _weighted(r, true, labels),
        "f1-score": _weighted(f, true, labels),
        "support": int(support.sum()),
    }
    return report


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """Softmax regression trained with mini-batch gradient descent.

    ``l2_lambda=0`` gives ordinary multinomial logistic regression.  A positive
    value adds ``lambda * sum(weights**2)`` to the mean cross-entropy.  The
    intercept row is deliberately not penalised.
    """

    def __init__(
        self,
        learning_rate=0.05,
        num_epochs=600,
        batch_size=256,
        l2_lambda=0.0,
        fit_intercept=True,
        tolerance=1e-7,
        patience=20,
        random_state=42,
        verbose=False,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _softmax(scores):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def _add_intercept(self, X):
        X = np.asarray(X, dtype=float)
        return np.c_[np.ones(X.shape[0]), X] if self.fit_intercept else X

    def _loss(self, X, y_index):
        probabilities = self._softmax(X @ self.weights_)
        data_loss = -np.mean(np.log(np.clip(probabilities[np.arange(len(y_index)), y_index], 1e-15, 1)))
        regularised = self.weights_[1:] if self.fit_intercept else self.weights_
        return float(data_loss + self.l2_lambda * np.sum(regularised**2))

    def fit(self, X, y):
        X = self._add_intercept(X)
        y = _as_1d(y)
        if X.shape[0] != y.size:
            raise ValueError("X and y must contain the same number of rows")
        if self.l2_lambda < 0:
            raise ValueError("l2_lambda must be non-negative")

        self.classes_, y_index = np.unique(y, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("at least two classes are required")

        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 0.01, size=(X.shape[1], self.classes_.size))
        best_loss, stale_epochs = np.inf, 0
        self.loss_history_ = []
        eye = np.eye(self.classes_.size)

        for epoch in range(self.num_epochs):
            permutation = rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], self.batch_size):
                indices = permutation[start : start + self.batch_size]
                xb, yb = X[indices], y_index[indices]
                error = self._softmax(xb @ self.weights_) - eye[yb]
                gradient = xb.T @ error / len(indices)
                penalty_gradient = 2 * self.l2_lambda * self.weights_
                if self.fit_intercept:
                    penalty_gradient[0] = 0
                self.weights_ -= self.learning_rate * (gradient + penalty_gradient)

            loss = self._loss(X, y_index)
            self.loss_history_.append(loss)
            improvement = best_loss - loss
            if improvement > self.tolerance:
                best_loss, stale_epochs = loss, 0
            else:
                stale_epochs += 1
            if self.verbose and (epoch % 100 == 0 or epoch == self.num_epochs - 1):
                print(f"epoch={epoch:4d} loss={loss:.6f}")
            if stale_epochs >= self.patience:
                break

        self.n_features_in_ = X.shape[1] - int(self.fit_intercept)
        self.n_iter_ = len(self.loss_history_)
        return self

    def predict_proba(self, X):
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before predict_proba")
        return self._softmax(self._add_intercept(X) @ self.weights_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return accuracy(y, self.predict(X))

    # Metric methods keep the assignment's requested API on the model class.
    accuracy = staticmethod(accuracy)
    precision = staticmethod(precision)
    recall = staticmethod(recall)
    f1_score = staticmethod(f1_score)
    macro_precision = staticmethod(macro_precision)
    macro_recall = staticmethod(macro_recall)
    macro_f1 = staticmethod(macro_f1)
    weighted_precision = staticmethod(weighted_precision)
    weighted_recall = staticmethod(weighted_recall)
    weighted_f1 = staticmethod(weighted_f1)
