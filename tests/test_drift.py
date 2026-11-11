"""Season drift: the table the monitor writes to JSON, served to the app."""
from __future__ import annotations

import pytest

from src.monitor import season_drift


@pytest.fixture(scope="module")
def drift(matches, predictor):
    return season_drift(matches, predictor)


def test_every_season_is_covered(drift, matches):
    assert len(drift["seasons"]) == matches["season"].nunique()
    assert [r["start"] for r in drift["seasons"]] == sorted(r["start"] for r in drift["seasons"])


def test_each_season_carries_the_sport_and_the_model(drift):
    for row in drift["seasons"]:
        assert row["home_win_rate"] is not None, f"{row['season']} has no outcome mix"
        total = row["home_win_rate"] + row["draw_rate"] + row["away_win_rate"]
        assert total == pytest.approx(1.0, abs=0.01)
        assert 0 < row["rps"] < 1
        assert row["matches"] > 0


def test_season_labels_survive_the_csv_round_trip(matches, predictor):
    """Reading the dataset turns the season '0506' into the integer 506."""
    as_ints = matches.assign(season=matches["season"].astype(int))
    rows = season_drift(as_ints, predictor)["seasons"]
    assert all("/" in r["season"] for r in rows)
    assert all(2000 <= r["start"] <= 2100 for r in rows), [r["season"] for r in rows]
    # And the mix must still be found, not dropped by a key mismatch.
    assert all(r["home_win_rate"] is not None for r in rows)


def test_the_in_sample_caveat_travels_with_the_numbers(drift):
    """These scores are not out-of-sample; the app must be able to say so."""
    assert "not out-of-sample" in drift["note"]


def test_holdout_score_is_offered_as_a_baseline(drift, predictor):
    assert drift["holdout_rps"] == predictor.meta["holdout_metrics"]["rps"]


# --- the feed going down is not the same as having no fixtures ---------------
def test_an_unreachable_feed_keeps_the_last_schedule_and_says_why(monkeypatch):
    from src import fixtures as fixtures_mod
    from src.fixtures import FeedUnavailable, FixtureFeed, TeamResolver

    feed = FixtureFeed(TeamResolver({"E0": ["Arsenal", "Chelsea"]}))
    monkeypatch.setattr(fixtures_mod, "from_openfootball",
                        lambda lg, r, season=None, timeout=30: ["a fixture"])
    assert feed.season("E0") == ["a fixture"]
    assert feed.errors == {}

    def down(lg, r, season=None, timeout=30):
        raise FeedUnavailable("connection reset")

    monkeypatch.setattr(fixtures_mod, "from_openfootball", down)
    assert feed.season("E0", refresh=True) == ["a fixture"], "the last schedule should stand"
    assert "connection reset" in feed.errors["E0"]


def test_a_feed_that_was_never_reachable_reports_empty(monkeypatch):
    from src import fixtures as fixtures_mod
    from src.fixtures import FeedUnavailable, FixtureFeed, TeamResolver

    def down(lg, r, season=None, timeout=30):
        raise FeedUnavailable("dns failure")

    monkeypatch.setattr(fixtures_mod, "from_openfootball", down)
    feed = FixtureFeed(TeamResolver({"E0": []}))
    assert feed.season("E0") == []
    assert feed.errors["E0"] == "dns failure"


def test_recovery_clears_the_error(monkeypatch):
    from src import fixtures as fixtures_mod
    from src.fixtures import FeedUnavailable, FixtureFeed, TeamResolver

    feed = FixtureFeed(TeamResolver({"E0": []}))

    def down(lg, r, season=None, timeout=30):
        raise FeedUnavailable("offline")

    monkeypatch.setattr(fixtures_mod, "from_openfootball", down)
    feed.season("E0")
    monkeypatch.setattr(fixtures_mod, "from_openfootball",
                        lambda lg, r, season=None, timeout=30: ["back"])
    assert feed.season("E0", refresh=True) == ["back"]
    assert "E0" not in feed.errors
