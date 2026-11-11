"""Which saved model gets loaded when an installed app holds more than one."""
from __future__ import annotations

import joblib
import pandas as pd
import pytest

from src import predict as predict_mod
from src.features import FEATURE_VERSION
from src.predict import Predictor
from src.refresh import unseen_matches


def bundle(predictor, *, trained_at, version=FEATURE_VERSION, **extra):
    b = {"pipeline": predictor.pipeline, "builder": predictor.builder,
         "features": predictor.features, "classes": predictor.classes,
         "model_name": "logistic", "trained_at": trained_at,
         "data_range": ["2015-08-08", "2021-05-01"], **extra}
    if version is not None:
        b["feature_version"] = version
    return b


@pytest.fixture
def candidates(monkeypatch, tmp_path):
    """Point Predictor.load at files the test writes, in a chosen order."""
    paths = []
    monkeypatch.setattr(predict_mod, "model_candidates", lambda name="model.joblib": paths)
    monkeypatch.setattr(predict_mod, "model_path", lambda name="model.joblib": tmp_path / name)

    def add(name, content):
        path = tmp_path / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            joblib.dump(content, path)
        paths.append(path)
        return path
    return add


def test_newest_compatible_model_wins_regardless_of_location(candidates, predictor):
    candidates("local.joblib", bundle(predictor, trained_at="2026-09-05T06:04:50+00:00"))
    shipped = candidates("shipped.joblib", bundle(predictor, trained_at="2026-09-15T12:38:38+00:00"))
    assert Predictor.load().path == shipped


def test_a_local_retrain_newer_than_the_shipped_model_is_used(candidates, predictor):
    local = candidates("local.joblib", bundle(predictor, trained_at="2026-10-01T00:00:00+00:00"))
    candidates("shipped.joblib", bundle(predictor, trained_at="2026-09-15T12:38:38+00:00"))
    assert Predictor.load().path == local


def test_the_bug_a_leftover_from_older_code_is_ignored(candidates, predictor):
    """What broke the installed app: an unversioned model saved by an earlier build."""
    candidates("leftover.joblib", bundle(predictor, trained_at="2027-01-01T00:00:00+00:00",
                                         version=None))
    shipped = candidates("shipped.joblib", bundle(predictor, trained_at="2026-09-15T12:38:38+00:00"))
    assert Predictor.load().path == shipped, "newer by date, but unusable by this code"


def test_a_model_from_a_different_feature_version_is_ignored(candidates, predictor):
    candidates("future.joblib", bundle(predictor, trained_at="2027-01-01T00:00:00+00:00",
                                       version=FEATURE_VERSION + 1))
    shipped = candidates("shipped.joblib", bundle(predictor, trained_at="2026-09-15T00:00:00+00:00"))
    assert Predictor.load().path == shipped


def test_an_unreadable_file_does_not_stop_loading(candidates, predictor):
    candidates("corrupt.joblib", b"not a pickle at all")
    good = candidates("good.joblib", bundle(predictor, trained_at="2026-09-15T00:00:00+00:00"))
    assert Predictor.load().path == good


def test_no_usable_model_explains_why(candidates, predictor):
    candidates("old.joblib", bundle(predictor, trained_at="2026-09-05T00:00:00+00:00", version=1))
    with pytest.raises(FileNotFoundError, match="feature version 1"):
        Predictor.load()


def test_a_freshly_trained_bundle_is_stamped():
    """Guards the other half: train.py must write the version it was built with."""
    import inspect

    from src import train
    assert '"feature_version": FEATURE_VERSION' in inspect.getsource(train.run)


# --- what counts as new data --------------------------------------------------
def test_unseen_matches_counts_only_matches_after_training_data_ends(predictor):
    matches = pd.DataFrame({"date": pd.to_datetime(
        ["2021-04-01", "2021-05-01", "2021-05-02", "2021-08-15"])})
    predictor.meta["data_range"] = ["2015-08-08", "2021-05-01"]
    assert unseen_matches(matches, predictor) == 2


def test_an_empty_data_folder_is_not_mistaken_for_new_data(predictor):
    """On first launch all history is re-downloaded; none of it is new to the model."""
    history = pd.DataFrame({"date": pd.date_range("2005-08-01", "2021-05-01", periods=38000)})
    predictor.meta["data_range"] = ["2005-08-01", "2021-05-01"]
    assert unseen_matches(history, predictor) == 0
