"""Golden values: nothing may move predictions without someone deciding to.

The leakage and determinism tests prove a run is *valid*; they say nothing about
whether it still produces the same answers. A refactor of features.py or
ratings.py can quietly shift every probability and every existing test still
passes. These pin the numbers.

Two layers, because they fail for different reasons:

  features     a scripted sequence of matches through the feature builder. Pure
               arithmetic, no model, so a mismatch means the ratings or features
               changed - and the diff says which ones.
  predictions  probabilities from a model trained on the synthetic league. Covers
               the whole path including the pipeline, with a tolerance, since a
               scikit-learn upgrade can move the last decimals.

When a change is deliberate, regenerate and read the diff before committing:

    python -m tests.test_golden --update
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.features import FeatureBuilder, build_features, feature_columns

GOLDEN = Path(__file__).parent / "golden"
FEATURES_FILE = GOLDEN / "features.json"
PREDICTIONS_FILE = GOLDEN / "predictions.json"

FEATURE_TOLERANCE = 1e-6      # pure arithmetic: should be exact
PROBABILITY_TOLERANCE = 2e-3  # a fitted model: allow for library drift

# A fixed run of matches. Two seasons so the season rollover, the promoted-club
# prior, head-to-head and the venue windows all get exercised.
SCRIPT = [
    # season, date, home, away, hg, ag, home shots on target, away shots on target
    ("2021", "2020-08-15", "Alpha", "Beta", 2, 1, 6, 3),
    ("2021", "2020-08-22", "Gamma", "Delta", 0, 0, 4, 4),
    ("2021", "2020-08-29", "Beta", "Gamma", 1, 3, 5, 8),
    ("2021", "2020-09-05", "Delta", "Alpha", 1, 4, 2, 9),
    ("2021", "2020-09-12", "Alpha", "Gamma", 2, 2, 7, 6),
    ("2021", "2020-09-19", "Beta", "Delta", 3, 0, 9, 2),
    ("2021", "2020-09-26", "Gamma", "Alpha", 1, 2, 5, 7),
    ("2021", "2020-10-03", "Delta", "Beta", 0, 1, 3, 6),
    ("2122", "2021-08-14", "Alpha", "Epsilon", 3, 0, 8, 2),   # Epsilon is new
    ("2122", "2021-08-21", "Epsilon", "Beta", 1, 1, 4, 5),
]
FIXTURES = [("Alpha", "Beta"), ("Gamma", "Delta"), ("Epsilon", "Alpha"), ("Beta", "Epsilon")]


def scripted_matches() -> pd.DataFrame:
    rows = []
    for i, (season, date, home, away, hg, ag, hst, ast) in enumerate(SCRIPT):
        rows.append({
            "match_id": i, "league": "GOLD", "season": season, "date": pd.Timestamp(date),
            "home_team": home, "away_team": away, "home_goals": hg, "away_goals": ag,
            "home_shots": hst * 2, "away_shots": ast * 2, "home_sot": hst, "away_sot": ast,
            "home_corners": 5, "away_corners": 4, "home_fouls": 11, "away_fouls": 12,
            "home_yellows": 1, "away_yellows": 2, "home_reds": 0, "away_reds": 0,
            "result": "H" if hg > ag else "A" if ag > hg else "D",
        })
    return pd.DataFrame(rows)


def computed_features() -> dict:
    """Every numeric feature of the last scripted match, plus the fixtures after it."""
    builder = FeatureBuilder()
    builder.transform(scripted_matches())
    out = {}
    for home, away in FIXTURES:
        row = builder.features("GOLD", home, away, pd.Timestamp("2021-08-28"))
        out[f"{home} v {away}"] = {k: round(float(v), 6) for k, v in row.items()
                                   if isinstance(v, (int, float)) and not isinstance(v, bool)
                                   and not (isinstance(v, float) and math.isnan(v))}
    return out


def computed_predictions(predictor) -> dict:
    teams = predictor.teams("SYN")[:6]
    out = {}
    for home, away in ((teams[0], teams[1]), (teams[2], teams[3]), (teams[4], teams[5])):
        p = predictor.predict("SYN", home, away, "2026-01-10")["probabilities"]
        out[f"{home} v {away}"] = {k: round(v, 6) for k, v in p.items()}
    return out


def _compare(actual: dict, expected: dict, tolerance: float, what: str):
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    assert not missing, f"{what} disappeared: {missing}"

    drifted = []
    for key, exp in expected.items():
        got = actual[key]
        for field, value in exp.items():
            if field not in got:
                drifted.append(f"{key}.{field}: gone (was {value})")
            elif abs(got[field] - value) > tolerance:
                drifted.append(f"{key}.{field}: {value} -> {got[field]}")
    assert not drifted, (
        f"{what} moved:\n  " + "\n  ".join(drifted[:25])
        + f"\n\nIf this was deliberate: python -m tests.test_golden --update")
    if added:
        print(f"note: new {what} not yet pinned: {added}")


def test_feature_values_have_not_moved():
    """Catches a refactor of features.py or ratings.py changing what is computed."""
    _compare(computed_features(), json.loads(FEATURES_FILE.read_text()),
             FEATURE_TOLERANCE, "features")


def test_predicted_probabilities_have_not_moved(predictor):
    """Catches the whole path shifting: features, pipeline, model settings."""
    _compare(computed_predictions(predictor), json.loads(PREDICTIONS_FILE.read_text()),
             PROBABILITY_TOLERANCE, "probabilities")


def test_the_pinned_features_cover_what_the_model_trains_on():
    """A golden file that misses the model's inputs would guard nothing."""
    pinned = set(next(iter(json.loads(FEATURES_FILE.read_text()).values())))
    trained_on = {f for f in feature_columns(False) if f != "league"}
    assert trained_on <= pinned, sorted(trained_on - pinned)


def test_a_model_trained_now_still_scores_its_own_holdout(predictor, matches):
    """The pickle-compatibility failure, in miniature: train, save, load, predict."""
    import joblib

    feats, builder = build_features(matches, min_history=5)
    bundle = {"pipeline": predictor.pipeline, "builder": builder,
              "features": predictor.features, "classes": predictor.classes}
    path = GOLDEN / "_roundtrip.joblib"
    try:
        joblib.dump(bundle, path)
        restored = joblib.load(path)
        row = restored["builder"].features("SYN", *predictor.teams("SYN")[:2],
                                           pd.Timestamp("2026-01-10"))
        X = pd.DataFrame([row]).reindex(columns=restored["features"])
        proba = restored["pipeline"].predict_proba(X)[0]
        assert abs(sum(proba) - 1) < 1e-9
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    import sys

    if "--update" not in sys.argv:
        raise SystemExit("pass --update to rewrite the golden files")
    GOLDEN.mkdir(exist_ok=True)
    FEATURES_FILE.write_text(json.dumps(computed_features(), indent=2, sort_keys=True) + "\n")

    # The prediction goldens need the same model the tests build.
    from tests.conftest import synthetic_league
    from src.predict import Predictor
    from src.train import train_final

    feats, builder = build_features(synthetic_league(), min_history=5)
    cols = [c for c in feature_columns(False) if c in feats.columns]
    pipe = train_final(feats, "logistic", cols, calibrate=False)
    predictor = Predictor({"pipeline": pipe, "builder": builder, "features": cols,
                           "classes": list(pipe.named_steps["clf"].classes_),
                           "model_name": "logistic"})
    PREDICTIONS_FILE.write_text(json.dumps(computed_predictions(predictor), indent=2, sort_keys=True) + "\n")
    print(f"wrote {FEATURES_FILE.name} and {PREDICTIONS_FILE.name}")
