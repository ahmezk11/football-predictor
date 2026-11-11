"""The smoke check must fail on exactly the break it exists to catch."""
from __future__ import annotations

import pytest

from src import smoke
from src.features import FEATURE_VERSION


def _usable(predictor, **meta):
    predictor.meta.update({"feature_version": FEATURE_VERSION, "model_name": "logistic",
                           "trained_at": "2026-09-15T00:00:00+00:00", "train_rows": 100, **meta})
    return predictor


def test_passes_on_a_model_this_code_can_use(predictor, monkeypatch, capsys):
    monkeypatch.setattr(smoke.Predictor, "load", classmethod(lambda cls, *a, **k: _usable(predictor)))
    assert smoke.main() == 0
    assert "smoke check passed" in capsys.readouterr().out


def test_fails_on_a_model_built_for_another_feature_version(predictor, monkeypatch, capsys):
    """A saved model the running code cannot use - the bug that reached a user."""
    monkeypatch.setattr(smoke.Predictor, "load",
                        classmethod(lambda cls, *a, **k: _usable(predictor, feature_version=None)))
    assert smoke.main() == 1
    assert "different feature version" in capsys.readouterr().out


def test_fails_when_probabilities_do_not_add_up(predictor, monkeypatch, capsys):
    good = _usable(predictor)
    real = good.predict

    def broken(*a, **k):
        out = real(*a, **k)
        out["probabilities"]["draw"] += 0.5
        return out

    monkeypatch.setattr(good, "predict", broken)
    monkeypatch.setattr(smoke.Predictor, "load", classmethod(lambda cls, *a, **k: good))
    assert smoke.main() == 1
    assert "sum to" in capsys.readouterr().out


def test_fails_when_no_model_can_be_loaded(monkeypatch):
    def missing(cls, *a, **k):
        raise FileNotFoundError("no usable model")

    monkeypatch.setattr(smoke.Predictor, "load", classmethod(missing))
    with pytest.raises(FileNotFoundError):
        smoke.main()
