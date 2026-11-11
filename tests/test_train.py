"""End-to-end training tests on synthetic data with known signal."""
from __future__ import annotations

import numpy as np
import pytest

from src import evaluate as ev
from src.features import build_features, feature_columns
from src.model import MODEL_NAMES, build_model
from src.train import chronological_split, cross_validate, train_final


@pytest.fixture(scope="module")
def prepared(matches):
    feats, builder = build_features(matches, min_history=5)
    features = [c for c in feature_columns(False) if c in feats.columns]
    return feats, builder, features


def test_split_is_chronological(prepared):
    feats, _, _ = prepared
    train, test = chronological_split(feats, 0.2)
    assert train.date.max() < test.date.min(), "test set must start after training ends"
    assert len(train) + len(test) == len(feats)
    assert 0.15 < len(test) / len(feats) < 0.25


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_model_beats_the_prior_baseline(prepared, name):
    feats, _, features = prepared
    train, test = chronological_split(feats, 0.2)
    pipe = build_model(name, features)
    pipe.fit(train[features], train["result"])
    proba = pipe.predict_proba(test[features])
    classes = list(pipe.named_steps["clf"].classes_)

    model = ev.evaluate(test["result"], proba, classes)
    baseline = ev.baseline_prior(train["result"], test["result"])
    assert model["rps"] < baseline["rps"], f"{name} adds nothing over the base rate"
    assert model["log_loss"] < baseline["log_loss"]


def test_probabilities_are_valid(prepared):
    feats, _, features = prepared
    train, test = chronological_split(feats, 0.2)
    pipe = build_model("logistic", features)
    pipe.fit(train[features], train["result"])
    proba = pipe.predict_proba(test[features])
    assert proba.shape == (len(test), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_cross_validation_trains_only_on_the_past(prepared):
    feats, _, features = prepared
    train, _ = chronological_split(feats, 0.2)
    res = cross_validate(train, "logistic", features, n_splits=3)
    assert len(res["folds"]) == 3
    ends = [f["train_end"] for f in res["folds"]]
    assert ends == sorted(ends), "each fold must extend the training window forward"
    assert 0 < res["mean"]["rps"] < 0.35


def test_calibration_wrapper_still_produces_valid_probabilities(prepared):
    feats, _, features = prepared
    train, test = chronological_split(feats, 0.2)
    pipe = train_final(train, "logistic", features, calibrate=True)
    proba = pipe.predict_proba(test[features])
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(pipe.classes_) == {"H", "D", "A"}


def test_shuffled_labels_destroy_the_signal(prepared):
    """A sanity check on the whole pipeline: no signal in, no skill out."""
    feats, _, features = prepared
    train, test = chronological_split(feats, 0.2)
    rng = np.random.default_rng(3)
    scrambled = train.copy()
    scrambled["result"] = rng.permutation(scrambled["result"].to_numpy())

    pipe = build_model("logistic", features)
    pipe.fit(scrambled[features], scrambled["result"])
    proba = pipe.predict_proba(test[features])
    classes = list(pipe.named_steps["clf"].classes_)
    noise = ev.evaluate(test["result"], proba, classes)

    pipe2 = build_model("logistic", features)
    pipe2.fit(train[features], train["result"])
    real = ev.evaluate(test["result"], pipe2.predict_proba(test[features]), classes)
    assert real["rps"] < noise["rps"], "a model on scrambled labels should not compete"
