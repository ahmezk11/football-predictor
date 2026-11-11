"""API tests against a model trained on synthetic data in a temp directory."""
from __future__ import annotations

import joblib
import pytest

from src import api as api_mod
from src.predict import UnknownTeamError


# --- Predictor ---------------------------------------------------------------
def test_bundle_survives_a_round_trip(predictor, tmp_path):
    """The fitted feature-state store has to pickle, or deployment is impossible."""
    path = tmp_path / "m.joblib"
    joblib.dump({"pipeline": predictor.pipeline, "builder": predictor.builder,
                 "features": predictor.features, "classes": predictor.classes}, path)
    restored = joblib.load(path)
    assert restored["builder"].known_teams("SYN") == predictor.builder.known_teams("SYN")


def test_predict_returns_a_normalised_distribution(predictor):
    teams = predictor.teams("SYN")
    r = predictor.predict("SYN", teams[0], teams[1], "2025-01-01")
    p = r["probabilities"]
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)
    assert r["prediction"] in {"H", "D", "A"}
    assert r["confidence"] == pytest.approx(max(p.values()), abs=1e-4)


def test_fair_odds_invert_the_probabilities(predictor):
    teams = predictor.teams("SYN")
    r = predictor.predict("SYN", teams[0], teams[1])
    assert r["fair_odds"]["home_win"] == pytest.approx(
        1 / r["probabilities"]["home_win"], rel=1e-2)


def test_home_advantage_shows_up_when_sides_are_swapped(predictor):
    """The same fixture reversed should shift probability toward the new host."""
    a, b = predictor.teams("SYN")[:2]
    fwd = predictor.predict("SYN", a, b)["probabilities"]
    rev = predictor.predict("SYN", b, a)["probabilities"]
    assert fwd["home_win"] > rev["away_win"]


def test_prediction_does_not_mutate_state(predictor):
    """Scoring a fixture must leave the builder untouched, or repeats would drift."""
    a, b = predictor.teams("SYN")[:2]
    before = predictor.builder.elo.rating("SYN", a)
    first = predictor.predict("SYN", a, b)["probabilities"]
    second = predictor.predict("SYN", a, b)["probabilities"]
    assert predictor.builder.elo.rating("SYN", a) == before
    assert first == second


def test_unknown_team_is_rejected(predictor):
    with pytest.raises(UnknownTeamError):
        predictor.predict("SYN", "Atlantis FC", predictor.teams("SYN")[0])


def test_a_team_cannot_play_itself(predictor):
    a = predictor.teams("SYN")[0]
    with pytest.raises(ValueError):
        predictor.predict("SYN", a, a)


# --- HTTP --------------------------------------------------------------------
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "logistic"


def test_reference_endpoints(client):
    assert client.get("/leagues").json()[0]["code"] == "SYN"
    teams = client.get("/teams?league=SYN").json()["teams"]
    assert len(teams) == 12
    form = client.get(f"/teams/SYN/{teams[0]}/form").json()
    assert form["matches_recorded"] > 0 and "elo" in form


def test_predict_endpoint(client):
    teams = client.get("/teams?league=SYN").json()["teams"]
    r = client.post("/predict", json={"league": "SYN", "home_team": teams[0],
                                      "away_team": teams[1], "match_date": "2025-03-01"})
    assert r.status_code == 200
    body = r.json()
    assert set(body["probabilities"]) == {"home_win", "draw", "away_win"}
    assert body["context"]["elo"][teams[0]] is not None


def test_predict_rejects_a_self_fixture(client):
    teams = client.get("/teams?league=SYN").json()["teams"]
    r = client.post("/predict", json={"league": "SYN", "home_team": teams[0],
                                      "away_team": teams[0]})
    assert r.status_code == 422


def test_predict_reports_unknown_teams(client):
    r = client.post("/predict", json={"league": "SYN", "home_team": "Nowhere United",
                                      "away_team": "Somewhere City"})
    assert r.status_code == 404
    assert "unknown team" in r.json()["detail"].lower()


def test_bad_odds_are_rejected(client):
    teams = client.get("/teams?league=SYN").json()["teams"]
    r = client.post("/predict", json={"league": "SYN", "home_team": teams[0],
                                      "away_team": teams[1],
                                      "odds": {"home": 0.5, "draw": 3.0, "away": 4.0}})
    assert r.status_code == 422


def test_batch_endpoint_isolates_failures(client):
    teams = client.get("/teams?league=SYN").json()["teams"]
    r = client.post("/predict/batch", json={"matches": [
        {"league": "SYN", "home_team": teams[0], "away_team": teams[1]},
        {"league": "SYN", "home_team": "Ghost FC", "away_team": teams[2]},
    ]})
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert "probabilities" in preds[0]
    assert "error" in preds[1], "one bad fixture must not sink the whole batch"


def test_model_metadata_endpoint(client):
    body = client.get("/model").json()
    assert body["classes"] == {"A": "Away win", "D": "Draw", "H": "Home win"}
    from src.features import feature_columns
    assert body["features"] == feature_columns(False)


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "<title>Match Predictor</title>" in r.text
    # The shell must offer every view the API backs.
    for view in ("upcoming", "predict", "history", "teams"):
        assert f'data-view="{view}"' in r.text


# --- the matchday schedule ---------------------------------------------------
@pytest.fixture
def scheduled(client, predictor, monkeypatch):
    """A season stubbed into the feed, dated relative to today so it never rots."""
    from datetime import date, timedelta

    from src.fixtures import Fixture, FixtureFeed

    today = date.today()
    day = lambda n: (today + timedelta(days=n)).isoformat()   # noqa: E731
    teams = predictor.teams("SYN")
    fixtures = [
        Fixture("SYN", day(-10), "15:00", teams[0], teams[1], round="Matchday 1",
                played=True, home_goals=2, away_goals=1),
        Fixture("SYN", day(-10), "17:30", teams[2], teams[3], round="Matchday 1",
                played=True, home_goals=0, away_goals=0),
        Fixture("SYN", day(3), "15:00", teams[0], teams[2], round="Matchday 2"),
        Fixture("SYN", day(3), "15:00", teams[1], "Promoted FC", round="Matchday 2",
                predictable=False, unknown_teams=("Promoted FC",)),
        Fixture("SYN", day(40), "15:00", teams[3], teams[4], round="Matchday 3"),
    ]
    feed = FixtureFeed(None)
    monkeypatch.setattr(feed, "season", lambda league, refresh=False: fixtures)
    api_mod.state["feed"] = feed
    yield client
    api_mod.state["feed"] = None


def _match(body, home, away):
    for g in body["matchdays"]:
        for m in g["matches"]:
            if (m["home_team"], m["away_team"]) == (home, away):
                return m
    raise AssertionError(f"{home} v {away} not in schedule")


def test_schedule_groups_by_matchday(scheduled):
    body = scheduled.get("/schedule?league=SYN").json()
    assert [r["round"] for r in body["rounds"]] == ["Matchday 1", "Matchday 2", "Matchday 3"]
    assert body["current_round"] == "Matchday 2"


def test_schedule_forecasts_every_upcoming_match_without_being_asked(scheduled, predictor):
    body = scheduled.get("/schedule?league=SYN").json()
    t = predictor.teams("SYN")
    for home, away in ((t[0], t[2]), (t[3], t[4])):
        pred = _match(body, home, away)["prediction"]
        assert sum(pred["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["counts"]["forecast"] == 2


def test_only_near_fixtures_are_locked_in(scheduled, predictor):
    body = scheduled.get("/schedule?league=SYN").json()
    t = predictor.teams("SYN")
    assert _match(body, t[0], t[2])["prediction"]["locked_in"] is True     # in 3 days
    assert _match(body, t[3], t[4])["prediction"]["locked_in"] is False    # in 40 days
    saved = {(p["home_team"], p["away_team"])
             for p in scheduled.get("/predictions").json()["predictions"]}
    assert (t[0], t[2]) in saved and (t[3], t[4]) not in saved


def test_schedule_reports_results_for_played_matches(scheduled):
    day = scheduled.get("/schedule?league=SYN&round=Matchday 1").json()["matchdays"][0]
    assert day["played"] == 2
    first = day["matches"][0]
    assert first["status"] == "played"
    assert (first["home_goals"], first["away_goals"], first["result"]) == (2, 1, "H")


def test_schedule_lists_unpredictable_matches_but_flags_them(scheduled):
    day = scheduled.get("/schedule?league=SYN&round=Matchday 2").json()["matchdays"][0]
    blocked = [m for m in day["matches"] if m["status"] == "unpredictable"]
    assert len(blocked) == 1 and blocked[0]["unknown_teams"] == ["Promoted FC"]
    assert "prediction" not in blocked[0]


def test_round_filter_keeps_the_full_round_list(scheduled):
    body = scheduled.get("/schedule?league=SYN&round=Matchday 3").json()
    assert len(body["matchdays"]) == 1
    assert len(body["rounds"]) == 3, "the matchday picker needs every round"


def test_forecast_summary_covers_every_league(scheduled):
    body = scheduled.get("/forecast/summary").json()
    assert body["leagues"]["SYN"]["forecast"] == 2


def test_schedule_rejects_an_unknown_league(scheduled):
    assert scheduled.get("/schedule?league=ZZ").status_code == 404


def test_schedule_rejects_an_unknown_matchday(scheduled):
    assert scheduled.get("/schedule?league=SYN&round=Matchday 99").status_code == 404


# --- drift, served to the app -------------------------------------------------
def test_drift_endpoint_returns_the_season_table(scheduled, monkeypatch):
    from src import api as api_mod

    seasons = {"seasons": [{"season": "2024/25", "start": 2024, "matches": 380,
                            "rps": 0.2, "accuracy": 0.53, "home_win_rate": 0.44,
                            "draw_rate": 0.25, "away_win_rate": 0.31,
                            "goals_per_game": 2.8}],
               "holdout_rps": 0.1995, "note": "not out-of-sample"}
    calls = []

    def fake(matches, predictor):
        calls.append(1)
        return seasons

    monkeypatch.setattr("src.monitor.season_drift", fake)
    monkeypatch.setattr("src.data.load", lambda name="matches.csv": "frame")
    api_mod.state.pop("drift", None)

    first = scheduled.get("/drift").json()
    assert first["seasons"][0]["season"] == "2024/25"
    scheduled.get("/drift")
    assert len(calls) == 1, "a feature pass over every match must not run twice"
    api_mod.state.pop("drift", None)


def test_schedule_reports_an_unreachable_fixture_source(scheduled):
    """The app must be able to tell 'no schedule' from 'no internet'."""
    from src import api as api_mod

    api_mod.state["feed"].errors["SYN"] = "connection reset"
    try:
        body = scheduled.get("/schedule?league=SYN").json()
        assert body["feed_error"] == "connection reset"
    finally:
        api_mod.state["feed"].errors.clear()


def test_schedule_says_nothing_when_the_feed_is_fine(scheduled):
    assert scheduled.get("/schedule?league=SYN").json()["feed_error"] is None
