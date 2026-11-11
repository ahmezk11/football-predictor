"""Rating systems: directions of movement, priors, and read-only predictions."""
from __future__ import annotations

import math

import pytest

from src.ratings import (
    ESTABLISHED_LEAGUE,
    Elo,
    EloParams,
    LeagueContext,
    OnlinePoisson,
    PiParams,
    PiRatings,
    PoissonParams,
    poisson_outcomes,
)


def establish(system, league="L", n=ESTABLISHED_LEAGUE):
    """Play enough matches between two clubs that the league counts as established."""
    for i in range(n):
        system.update(league, "A", "B", 1, 1, season="s1")


# --- Elo ----------------------------------------------------------------------
def test_elo_winner_gains_and_it_is_zero_sum():
    elo = Elo(EloParams(k=20))
    elo.update("L", "A", "B", 3, 0)
    a, b = elo.rating("L", "A"), elo.rating("L", "B")
    assert a > 1500 > b
    assert a + b == pytest.approx(3000)


def test_elo_bigger_margin_moves_ratings_further():
    small, big = Elo(), Elo()
    small.update("L", "A", "B", 1, 0)
    big.update("L", "A", "B", 4, 0)
    assert big.rating("L", "A") > small.rating("L", "A")


def test_elo_newcomer_to_an_established_league_starts_below_average():
    elo = Elo(EloParams(promoted_offset=75))
    assert elo.rating("L", "Newcomer") == 1500, "no prior before the league has history"
    establish(elo)
    assert elo.rating("L", "Newcomer") == pytest.approx(1425)


def test_elo_regresses_towards_the_mean_between_seasons():
    elo = Elo(EloParams(k=30, season_regression=0.5))
    for _ in range(10):
        elo.update("L", "A", "B", 3, 0, season="s1")
    before = elo.rating("L", "A")
    elo.update("L", "A", "B", 1, 1, season="s2")
    assert elo.rating("L", "A") < before


def test_elo_prediction_does_not_create_state():
    elo = Elo()
    elo.expected_home("L", "A", "B")
    assert elo.r == {}


# --- pi-ratings ---------------------------------------------------------------
def test_pi_home_win_raises_home_side_and_lowers_away_side():
    pi = PiRatings(PiParams())
    pi.update("L", "A", "B", 3, 0)
    a_home, a_away = pi.ratings("L", "A")
    b_home, b_away = pi.ratings("L", "B")
    assert a_home > 0 and b_away < 0
    assert 0 < a_away < a_home, "the cross-update moves the other venue's rating less"


def test_pi_expected_goal_difference_follows_ratings():
    pi = PiRatings()
    for _ in range(30):
        pi.update("L", "Strong", "Weak", 3, 0)
    assert pi.expected_gd("L", "Strong", "Weak") > 0.5


def test_pi_newcomer_prior():
    pi = PiRatings(PiParams(promoted_offset=0.5))
    establish(pi)
    assert pi.ratings("L", "Newcomer") == [-0.5, -0.5]


# --- online Poisson -----------------------------------------------------------
def test_poisson_outcomes_are_a_distribution():
    for lam_h, lam_a in ((0.3, 3.0), (1.4, 1.1), (2.5, 0.4)):
        h, d, a = poisson_outcomes(lam_h, lam_a)
        assert h + d + a == pytest.approx(1.0)
        assert min(h, d, a) > 0


def test_poisson_stronger_side_is_favoured():
    h, _, a = poisson_outcomes(2.0, 0.8)
    assert h > a


def test_dixon_coles_correction_adds_draw_probability():
    _, plain, _ = poisson_outcomes(1.3, 1.1, rho=0.0)
    _, corrected, _ = poisson_outcomes(1.3, 1.1, rho=-0.1)
    assert corrected > plain


def test_poisson_scoring_more_than_expected_raises_attack_and_lowers_their_defence():
    model = OnlinePoisson(PoissonParams(lr=0.05))
    model.update("L", "A", "B", 5, 0)
    att_a, _ = model.team("L", "A")
    _, def_b = model.team("L", "B")
    assert att_a > 0 and def_b < 0


def test_poisson_learns_that_a_strong_side_scores_more():
    model = OnlinePoisson(PoissonParams(lr=0.05))
    for _ in range(40):
        model.update("L", "Strong", "Weak", 3, 0)
        model.update("L", "Weak", "Strong", 0, 3)
    lam_h, lam_a = model.rates("L", "Strong", "Weak")
    assert lam_h > 2 * lam_a


def test_poisson_ignores_missing_values():
    model = OnlinePoisson()
    model.update("L", "A", "B", float("nan"), 1.0)
    assert model.att == {}


def test_poisson_newcomer_scores_less():
    model = OnlinePoisson(PoissonParams(promoted_attack=0.3))
    establish(model)
    lam_new, _ = model.rates("L", "Newcomer", "A")
    lam_old, _ = model.rates("L", "B", "A")
    assert lam_new < lam_old


# --- league context -----------------------------------------------------------
def test_league_context_tracks_a_shift_in_home_advantage():
    ctx = LeagueContext(halflife=50)
    start = ctx.get("L")["home"]
    for _ in range(300):
        ctx.update("L", 0, 1)            # a run of away wins
    assert ctx.get("L")["home"] < start
    assert ctx.get("L")["away"] > 0.8
    assert math.isclose(sum(ctx.get("L")[k] for k in ("home", "draw", "away")), 1.0,
                        rel_tol=1e-6)


# --- integration with the feature builder ------------------------------------
def test_feature_rows_carry_every_rating_feature():
    from src.features import RATING_FEATURES, FeatureBuilder
    import pandas as pd

    fb = FeatureBuilder()
    fb.update({"league": "L", "home_team": "A", "away_team": "B", "home_goals": 2,
               "away_goals": 1, "date": pd.Timestamp("2020-01-01"), "season": "1920",
               "home_sot": 6, "away_sot": 3})
    row = fb.features("L", "A", "B", pd.Timestamp("2020-01-08"))
    missing = [f for f in RATING_FEATURES if f not in row]
    assert not missing, missing
    assert row["pois_p_home"] + row["pois_p_draw"] + row["pois_p_away"] == pytest.approx(1.0)
