"""Market comparison. The odds source is often unreachable, so the pieces are
tested against synthetic odds rather than a live download."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmark import (
    disagreements,
    market_probabilities,
    paired_bootstrap,
    per_match_rps,
)


def frame(rows):
    return pd.DataFrame([{
        "date": pd.Timestamp("2026-01-01"), "league": "E0",
        "home_team": h, "away_team": a, "result": r,
        "p_market_home": ph, "p_market_draw": pd_, "p_market_away": pa,
    } for h, a, r, ph, pd_, pa in rows])


def test_market_probabilities_are_normalised():
    df = frame([("A", "B", "H", 0.5, 0.3, 0.3)])      # sums to 1.1, an overround
    p = market_probabilities(df)
    assert p.sum() == pytest.approx(1.0)
    assert p[0][2] == pytest.approx(0.5 / 1.1)        # column order is A, D, H


def test_missing_odds_are_reported_not_guessed():
    df = frame([("A", "B", "H", 0.5, 0.3, 0.2)])
    df.loc[0, "p_market_draw"] = np.nan
    assert market_probabilities(df) is None
    assert market_probabilities(df.drop(columns=["p_market_home"])) is None


def test_per_match_rps_matches_the_metric_used_everywhere():
    from src.evaluate import ranked_probability_score

    proba = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    y = ["H", "A"]
    assert per_match_rps(proba, y).mean() == pytest.approx(
        ranked_probability_score(y, proba, ["A", "D", "H"]))


def test_a_perfect_forecast_scores_zero():
    assert per_match_rps(np.array([[0.0, 0.0, 1.0]]), ["H"])[0] == pytest.approx(0.0)


def test_bootstrap_detects_a_real_gap():
    model = np.full(500, 0.18)
    market = np.full(500, 0.20)
    out = paired_bootstrap(model, market, rounds=500)
    assert out["model_minus_market"] == pytest.approx(0.02)
    assert out["model_better_share"] == 1.0
    assert out["ci_low"] > 0


def test_bootstrap_stays_agnostic_when_there_is_no_gap():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.2, 0.05, 800)
    out = paired_bootstrap(noise, noise + rng.normal(0, 0.05, 800), rounds=500)
    assert out["ci_low"] < 0 < out["ci_high"]


def test_disagreements_report_who_was_right():
    df = frame([
        ("A", "B", "H", 0.30, 0.30, 0.40),    # market favours away, model will favour home
        ("C", "D", "A", 0.30, 0.30, 0.40),    # market favours away and is right
        ("E", "F", "H", 0.60, 0.25, 0.15),    # both favour home: not a disagreement
    ])
    market = market_probabilities(df)
    model = np.array([[0.2, 0.2, 0.6], [0.2, 0.2, 0.6], [0.15, 0.25, 0.60]])
    out = disagreements(df, model, market)
    assert out["matches"] == 2
    assert out["model_right"] == pytest.approx(0.5)
    assert out["market_right"] == pytest.approx(0.5)
    assert out["biggest"][0]["match"] in {"A v B", "C v D"}


def test_no_disagreements_is_handled():
    df = frame([("A", "B", "H", 0.60, 0.25, 0.15)])
    market = market_probabilities(df)
    out = disagreements(df, market.copy(), market)
    assert out["matches"] == 0 and "model_right" not in out


def test_the_benchmark_refuses_a_dataset_without_odds(tmp_path, monkeypatch):
    """The common case: the odds source blocked the download."""
    from types import SimpleNamespace

    from src import benchmark

    df = frame([("A", "B", "H", 0.5, 0.3, 0.2)])
    df["odds_home"] = np.nan
    monkeypatch.setattr(benchmark.data_mod, "load", lambda name: df)
    with pytest.raises(SystemExit, match="closing odds"):
        benchmark.run(SimpleNamespace(dataset="matches_odds.csv", test_fraction=0.2))
