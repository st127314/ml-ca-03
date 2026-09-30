import mlflow

import model_utils


def test_remote_registry_is_preferred(monkeypatch):
    monkeypatch.setenv("MLFLOW_REMOTE_TRACKING_URI", "http://remote:5000")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://local:5000")
    seen = []
    sentinel = object()
    monkeypatch.setattr(model_utils, "load_registry_model", lambda uri: seen.append(uri) or sentinel)

    assert model_utils.load_serving_model() is sentinel
    assert seen == ["http://remote:5000"]


def test_local_registry_is_used_when_remote_is_unavailable(monkeypatch):
    monkeypatch.setenv("MLFLOW_REMOTE_TRACKING_URI", "http://remote:5000")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://local:5000")
    seen = []
    sentinel = object()

    def load(uri):
        seen.append(uri)
        if uri == "http://remote:5000":
            raise mlflow.exceptions.MlflowException("unreachable")
        return sentinel

    monkeypatch.setattr(model_utils, "load_registry_model", load)
    assert model_utils.load_serving_model() is sentinel
    assert seen == ["http://remote:5000", "http://local:5000"]
