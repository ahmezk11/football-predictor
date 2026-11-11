"""End-to-end smoke check: can a freshly trained model actually be used?

Training and predicting live in different processes in the real app - one writes
a model, the other loads it and scores fixtures - and the failure that bit a user
sat exactly in that gap: a saved model whose pickled feature builder the running
code could no longer use. Everything passed; the app broke on launch.

This trains nothing. It loads whatever model is on disk the way the app does, and
scores a fixture, so a break shows up in CI rather than at first launch.

    python -m src.smoke
"""
from __future__ import annotations

import sys

from .features import FEATURE_VERSION
from .predict import Predictor


def main() -> int:
    predictor = Predictor.load()
    meta = predictor.meta
    print(f"loaded {meta.get('model_name')} from {getattr(predictor, 'path', '?')}")
    print(f"  feature version {meta.get('feature_version')} (code expects {FEATURE_VERSION})")
    print(f"  trained {meta.get('trained_at')} on {meta.get('train_rows')} matches")

    if meta.get("feature_version") != FEATURE_VERSION:
        print("FAIL: the saved model was built for a different feature version")
        return 1

    leagues = predictor.leagues()
    if not leagues:
        print("FAIL: the model knows no leagues")
        return 1

    league = leagues[0]["code"]
    teams = predictor.teams(league)
    if len(teams) < 2:
        print(f"FAIL: {league} has fewer than two teams")
        return 1

    home, away = teams[0], teams[1]
    result = predictor.predict(league, home, away)
    probs = result["probabilities"]
    total = sum(probs.values())
    print(f"  {home} v {away}: {result['prediction_label']} "
          f"(H {probs['home_win']:.3f} D {probs['draw']:.3f} A {probs['away_win']:.3f})")

    # The API rounds each probability to four decimals, so a correct result can
    # sum to 1.0001. This catches a distribution that is actually wrong, not the
    # rounding: three rounded values cannot drift further than about 1.5e-4.
    if abs(total - 1.0) > 1e-3:
        print(f"FAIL: probabilities sum to {total}")
        return 1
    if not result.get("context", {}).get("elo"):
        print("FAIL: no reasoning returned with the prediction")
        return 1

    # The batch path the app uses to forecast a whole season.
    from types import SimpleNamespace
    fixtures = [SimpleNamespace(league=league, home_team=home, away_team=away,
                                date="2026-01-10", odds=None)]
    batch = predictor.forecast_batch(fixtures)
    if not batch or batch[0] is None:
        print("FAIL: batch forecasting returned nothing")
        return 1

    print("smoke check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
