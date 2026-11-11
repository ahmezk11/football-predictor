"""Feature-engineering tests, with an emphasis on leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ELO_START,
    FeatureBuilder,
    build_features,
    feature_columns,
)


def make_matches(n_rounds: int = 8) -> pd.DataFrame:
    """A tiny synthetic league: A is strong, D is weak, results are deterministic."""
    teams = ["A", "B", "C", "D"]
    strength = {"A": 3.0, "B": 2.0, "C": 1.5, "D": 0.5}
    rows, day = [], pd.Timestamp("2020-08-01")
    for rnd in range(n_rounds):
        pairs = [(teams[0], teams[1]), (teams[2], teams[3])] if rnd % 2 == 0 \
            else [(teams[1], teams[2]), (teams[3], teams[0])]
        for home, away in pairs:
            hg = int(round(strength[home]))
            ag = int(round(strength[away] - 0.5))
            rows.append({
                "date": day, "league": "T1", "season": "2021",
                "home_team": home, "away_team": away,
                "home_goals": hg, "away_goals": max(ag, 0),
                "home_shots": 12, "away_shots": 9, "home_sot": 5, "away_sot": 3,
                "home_corners": 6, "away_corners": 4, "home_fouls": 10, "away_fouls": 11,
                "home_yellows": 1, "away_yellows": 2, "home_reds": 0, "away_reds": 0,
                "odds_home": 1.8, "odds_draw": 3.5, "odds_away": 4.2,
            })
        day += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    df["result"] = np.where(df.home_goals > df.away_goals, "H",
                            np.where(df.home_goals < df.away_goals, "A", "D"))
    df["match_id"] = range(len(df))
    return df


def test_first_match_has_no_history():
    """The very first row must be blank: nothing has happened yet."""
    matches = make_matches()
    feats = FeatureBuilder().transform(matches)
    first = feats.iloc[0]
    assert first.elo_home == ELO_START and first.elo_away == ELO_START
    assert np.isnan(first.ppg5_home) and np.isnan(first.ppg5_away)
    assert first.h2h_played == 0
    assert np.isnan(first.rest_days_home)


def test_features_never_see_their_own_result():
    """Feature values must be identical whether or not later matches exist."""
    matches = make_matches(10)
    full = FeatureBuilder().transform(matches)
    truncated = FeatureBuilder().transform(matches.iloc[:12])

    cols = [c for c in full.columns
            if c not in {"date", "home_team", "away_team", "result", "season", "match_id",
                         "league"}]
    a = full.iloc[:12][cols].astype(float).to_numpy()
    b = truncated[cols].astype(float).to_numpy()
    assert np.allclose(a, b, equal_nan=True), "future matches changed past features"


def test_state_updates_after_a_win():
    builder = FeatureBuilder()
    matches = make_matches(2)
    builder.transform(matches)
    a = builder.state("T1", "A")
    assert builder.elo.rating("T1", "A") > ELO_START > builder.elo.rating("T1", "D"), \
        "the stronger side should gain rating"
    assert a.played == 2 and a.last_date is not None


def test_form_reflects_recent_results():
    matches = make_matches(8)
    feats = FeatureBuilder().transform(matches)
    late = feats[(feats.home_team == "A") | (feats.away_team == "A")].iloc[-1]
    ppg = late.ppg5_home if late.home_team == "A" else late.ppg5_away
    assert ppg > 2.0, "A wins nearly everything, so its points-per-game must be high"


def test_head_to_head_accumulates():
    feats = FeatureBuilder().transform(make_matches(8))
    ab = feats[(feats.home_team == "A") & (feats.away_team == "B")]
    assert ab.h2h_played.iloc[0] == 0
    assert ab.h2h_played.iloc[-1] > 0
    assert ab.h2h_ppg_home.iloc[-1] == pytest.approx(3.0)


def test_rest_days_are_computed():
    feats = FeatureBuilder().transform(make_matches(4))
    later = feats[feats.matches_played_home > 0]
    assert (later.rest_days_home.dropna() == 7).all()


def test_market_features_remove_the_overround():
    f = FeatureBuilder.market_features({"home": 2.0, "draw": 4.0, "away": 4.0})
    total = f["p_market_home"] + f["p_market_draw"] + f["p_market_away"]
    assert total == pytest.approx(1.0)
    assert f["market_overround"] == pytest.approx(0.0)
    assert f["p_market_home"] == pytest.approx(0.5)


def test_market_features_tolerate_junk():
    for bad in (None, {}, {"home": 0, "draw": 3, "away": 3},
                {"home": "x", "draw": 3, "away": 3}, {"home": float("nan")}):
        assert all(np.isnan(v) for v in FeatureBuilder.market_features(bad).values())


def test_build_features_drops_cold_start_rows():
    matches = make_matches(10)
    feats, builder = build_features(matches, min_history=4)
    assert (feats.matches_played_home >= 4).all()
    assert (feats.matches_played_away >= 4).all()
    assert len(feats) < len(matches)
    assert set(builder.known_teams("T1")) == {"A", "B", "C", "D"}


def test_feature_columns_toggle_odds():
    with_odds = feature_columns(True)
    without = feature_columns(False)
    assert "p_market_home" in with_odds and "p_market_home" not in without
    assert set(without).issubset(set(with_odds))


def test_prediction_features_match_training_features():
    """A future fixture must produce exactly the columns the model expects."""
    matches = make_matches(10)
    builder = FeatureBuilder()
    train_rows = builder.transform(matches)
    fixture = builder.features("T1", "A", "B", pd.Timestamp("2021-01-01"))
    expected = set(feature_columns(True))
    assert expected.issubset(set(fixture)), expected - set(fixture)
    assert expected.issubset(set(train_rows.columns))


def test_season_rollover_regresses_elo():
    """Same results, one run crossing into a new season: that run must end lower.

    Comparing against an identical run that stays in one season isolates the
    regression itself, whatever the tuned K and regression strength happen to be.
    """
    def run(new_season: bool) -> float:
        builder = FeatureBuilder()
        builder.transform(make_matches(8))
        later = make_matches(2)
        later["season"] = "2122" if new_season else "2021"
        later["date"] = later["date"] + pd.Timedelta(days=400)
        builder.transform(later)
        return builder.elo.rating("T1", "A")

    assert run(new_season=True) < run(new_season=False)
