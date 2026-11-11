"""Automatic forecasting: what gets forecast, what gets locked in, and when."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.autopredict import LOCK_IN_DAYS, forecast_all, forecast_league, status_of
from src.fixtures import Fixture, FixtureFeed
from src.store import Store

TODAY = date(2026, 9, 15)


def d(n: int) -> str:
    return (TODAY + timedelta(days=n)).isoformat()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "auto.db")


def feed_of(fixtures, league="SYN"):
    feed = FixtureFeed(None)
    feed._cache[league] = (float("inf"), fixtures)
    return feed


def fx(home, away, when, rnd="Matchday 1", **kw):
    return Fixture("SYN", when, "15:00", home, away, round=rnd, **kw)


# --- status ------------------------------------------------------------------
def test_status_of_each_kind():
    assert status_of(fx("a", "b", d(-3), played=True, home_goals=1, away_goals=0), TODAY) == "played"
    assert status_of(fx("a", "b", d(-3)), TODAY) == "awaiting"
    assert status_of(fx("a", "b", d(0)), TODAY) == "upcoming"
    assert status_of(fx("a", "b", d(5)), TODAY) == "upcoming"
    assert status_of(fx("a", "b", d(5), predictable=False), TODAY) == "unpredictable"


# --- forecasting -------------------------------------------------------------
def test_batch_forecast_matches_single_predictions(predictor):
    t = predictor.teams("SYN")
    fixtures = [fx(t[0], t[1], d(2)), fx(t[2], t[3], d(9)), fx(t[4], t[5], d(30))]
    for f, batch in zip(fixtures, predictor.forecast_batch(fixtures)):
        single = predictor.predict(f.league, f.home_team, f.away_team, f.date)["probabilities"]
        for k in single:
            assert batch["probabilities"][k] == pytest.approx(single[k], abs=1e-4)


def test_batch_forecast_skips_what_it_cannot_score(predictor):
    t = predictor.teams("SYN")
    out = predictor.forecast_batch([fx(t[0], "Nobody FC", d(2)), fx(t[0], t[0], d(2)),
                                    fx(t[0], t[1], d(2))])
    assert out[0] is None and out[1] is None and out[2] is not None


def test_every_upcoming_fixture_is_forecast(predictor, store):
    t = predictor.teams("SYN")
    fixtures = [fx(t[0], t[1], d(1)), fx(t[2], t[3], d(60), rnd="Matchday 9")]
    data = forecast_league(predictor, feed_of(fixtures), store, "SYN", today=TODAY)
    rows = [m for g in data["matchdays"] for m in g["matches"]]
    assert all(m.get("prediction") for m in rows)
    assert data["counts"]["forecast"] == 2


# --- locking in --------------------------------------------------------------
def test_lock_in_window_boundary(predictor, store):
    t = predictor.teams("SYN")
    fixtures = [fx(t[0], t[1], d(LOCK_IN_DAYS)), fx(t[2], t[3], d(LOCK_IN_DAYS + 1))]
    data = forecast_league(predictor, feed_of(fixtures), store, "SYN", today=TODAY)
    rows = {m["home_team"]: m for g in data["matchdays"] for m in g["matches"]}
    assert rows[t[0]]["prediction"]["locked_in"] is True
    assert rows[t[2]]["prediction"]["locked_in"] is False
    assert {(r["home_team"]) for r in store.history()} == {t[0]}


def test_a_match_already_kicked_off_is_never_newly_locked_in(predictor, store):
    """Otherwise a slow results feed would let forecasts be made after the match."""
    t = predictor.teams("SYN")
    data = forecast_league(predictor, feed_of([fx(t[0], t[1], d(-2))]), store, "SYN",
                           today=TODAY)
    row = data["matchdays"][0]["matches"][0]
    assert row["status"] == "awaiting"
    assert row["prediction"]["locked_in"] is False, "shown, but not saved"
    assert store.history() == []


def test_awaiting_match_shows_the_forecast_locked_in_before_kickoff(predictor, store):
    t = predictor.teams("SYN")
    before = forecast_league(predictor, feed_of([fx(t[0], t[1], d(2))]), store, "SYN",
                             today=TODAY)
    locked = before["matchdays"][0]["matches"][0]["prediction"]["probabilities"]

    later = TODAY + timedelta(days=5)    # match date has passed, no result posted yet
    after = forecast_league(predictor, feed_of([fx(t[0], t[1], d(2))]), store, "SYN",
                            today=later)
    row = after["matchdays"][0]["matches"][0]
    assert row["status"] == "awaiting" and row["prediction"]["locked_in"] is True
    for k in locked:
        assert row["prediction"]["probabilities"][k] == pytest.approx(locked[k], abs=1e-4)


def test_locking_in_never_overwrites_a_graded_prediction(predictor, store):
    t = predictor.teams("SYN")
    forecast_league(predictor, feed_of([fx(t[0], t[1], d(2))]), store, "SYN", today=TODAY)
    store.settle([{"league": "SYN", "home_team": t[0], "away_team": t[1],
                   "result": "D", "home_goals": 1, "away_goals": 1}])
    graded = store.history()[0]

    store.save_predictions([{"league": "SYN", "home_team": t[0], "away_team": t[1],
                             "date": d(2), "prediction": "A", "model": {},
                             "probabilities": {"home_win": 0.1, "draw": 0.1, "away_win": 0.8}}])
    after = store.history()[0]
    assert (after["predicted"], after["p_away"], after["result"]) == \
           (graded["predicted"], graded["p_away"], "D")


def test_played_match_shows_whether_the_locked_in_call_was_right(predictor, store):
    t = predictor.teams("SYN")
    forecast_league(predictor, feed_of([fx(t[0], t[1], d(2))]), store, "SYN", today=TODAY)
    call = store.history()[0]["predicted"]
    outcome = "A" if call != "A" else "H"
    store.settle([{"league": "SYN", "home_team": t[0], "away_team": t[1],
                   "result": outcome, "home_goals": 0, "away_goals": 1}])
    played = fx(t[0], t[1], d(2), played=True, home_goals=0, away_goals=1)
    data = forecast_league(predictor, feed_of([played]), store, "SYN",
                           today=TODAY + timedelta(days=4))
    assert data["matchdays"][0]["matches"][0]["prediction"]["correct"] is False


# --- where the view opens ----------------------------------------------------
def test_opens_on_the_first_round_with_a_match_still_to_come(predictor, store):
    """Not on a round the feed simply has not posted results for yet."""
    t = predictor.teams("SYN")
    fixtures = [fx(t[0], t[1], d(-3), rnd="Matchday 4"),       # kicked off, no result
                fx(t[2], t[3], d(4), rnd="Matchday 5")]
    data = forecast_league(predictor, feed_of(fixtures), store, "SYN", today=TODAY)
    assert data["current_round"] == "Matchday 5"


def test_one_league_failing_does_not_stop_the_rest(predictor, store, monkeypatch):
    monkeypatch.setattr(predictor, "leagues",
                        lambda: [{"code": "SYN", "name": "s"}, {"code": "BAD", "name": "b"}])
    feed = feed_of([fx(*predictor.teams("SYN")[:2], d(2))])
    real = feed.matchdays

    def flaky(league, refresh=False):
        if league == "BAD":
            raise RuntimeError("feed down")
        return real(league, refresh=refresh)

    monkeypatch.setattr(feed, "matchdays", flaky)
    out = forecast_all(predictor, feed, store, today=TODAY)
    assert "error" in out["BAD"]
    assert out["SYN"]["forecast"] == 1
