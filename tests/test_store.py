"""Store tests: favourites, saved predictions, settling and the scorecard."""
from __future__ import annotations

import pytest

from src.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def prediction(home="Arsenal", away="Chelsea", date="2026-09-06",
               probs=(0.6, 0.25, 0.15), league="E0"):
    h, d, a = probs
    call = max(zip("HDA", probs), key=lambda t: t[1])[0]
    return {"league": league, "home_team": home, "away_team": away, "date": date,
            "prediction": call, "probabilities": {"home_win": h, "draw": d, "away_win": a},
            "model": {"name": "logistic", "uses_market_odds": False}}


# --- favourites --------------------------------------------------------------
def test_follow_and_unfollow(store):
    store.follow("E0", "Arsenal")
    store.follow("SP1", "Real Madrid")
    assert store.favourite_teams() == ["Arsenal", "Real Madrid"]
    store.unfollow("E0", "Arsenal")
    assert store.favourite_teams() == ["Real Madrid"]


def test_following_twice_is_harmless(store):
    store.follow("E0", "Arsenal")
    store.follow("E0", "Arsenal")
    assert len(store.favourites()) == 1


# --- predictions -------------------------------------------------------------
def test_saving_the_same_fixture_updates_rather_than_duplicates(store):
    store.save_prediction(prediction())
    store.save_prediction(prediction(probs=(0.4, 0.3, 0.3)))
    rows = store.history()
    assert len(rows) == 1
    assert rows[0]["p_home"] == pytest.approx(0.4)


def test_settling_attaches_the_real_result(store):
    store.save_prediction(prediction())
    settled = store.settle([{"league": "E0", "home_team": "Arsenal",
                             "away_team": "Chelsea", "result": "H",
                             "home_goals": 2, "away_goals": 0}])
    assert settled == 1
    row = store.history()[0]
    assert row["result"] == "H" and row["home_goals"] == 2
    assert store.open_predictions() == []


def test_settling_survives_a_moved_fixture(store):
    """Matches get rescheduled; the prediction still refers to the same game."""
    store.save_prediction(prediction(date="2026-09-06"))
    assert store.settle([{"league": "E0", "home_team": "Arsenal", "away_team": "Chelsea",
                          "result": "D", "home_goals": 1, "away_goals": 1}]) == 1
    assert store.history()[0]["result"] == "D"


def test_settling_is_idempotent(store):
    store.save_prediction(prediction())
    result = [{"league": "E0", "home_team": "Arsenal", "away_team": "Chelsea",
               "result": "H", "home_goals": 2, "away_goals": 0}]
    assert store.settle(result) == 1
    assert store.settle(result) == 0, "an already-settled prediction must not resettle"


def test_unrelated_results_are_ignored(store):
    store.save_prediction(prediction())
    assert store.settle([{"league": "E0", "home_team": "Everton",
                          "away_team": "Fulham", "result": "H"}]) == 0
    assert len(store.open_predictions()) == 1


def test_scorecard_is_empty_before_anything_settles(store):
    store.save_prediction(prediction())
    card = store.scorecard()
    assert card["settled"] == 0 and card["pending"] == 1


def test_scorecard_scores_the_settled_predictions(store):
    calls = [
        (prediction("Arsenal", "Chelsea", "2026-09-06", (0.7, 0.2, 0.1)), "H"),   # hit
        (prediction("Leeds", "Everton", "2026-09-07", (0.2, 0.3, 0.5)), "A"),     # hit
        (prediction("Fulham", "Brentford", "2026-09-08", (0.6, 0.25, 0.15)), "A"),  # miss
    ]
    for pred, _ in calls:
        store.save_prediction(pred)
    store.settle([{"league": p["league"], "home_team": p["home_team"],
                   "away_team": p["away_team"], "result": r, "home_goals": 1,
                   "away_goals": 0} for p, r in calls])

    card = store.scorecard()
    assert card["settled"] == 3
    assert card["correct"] == 2
    assert card["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert 0 < card["rps"] < 1
    assert card["by_outcome"]["H"]["predicted"] == 2
    by_team = {b["team"]: b for b in card["by_team"]}
    assert by_team["Arsenal"]["correct"] == 1


def test_forget_removes_a_prediction(store):
    pid = store.save_prediction(prediction())
    assert store.forget(pid) is True
    assert store.history() == []
    assert store.forget(pid) is False


def test_meta_round_trip(store):
    assert store.get_meta("last_retrain") is None
    store.set_meta("last_retrain", "2026-09-03")
    store.set_meta("last_retrain", "2026-09-04")     # upsert, not duplicate
    assert store.get_meta("last_retrain") == "2026-09-04"


def test_store_reopens_existing_database(tmp_path):
    path = tmp_path / "persist.db"
    Store(path).follow("E0", "Arsenal")
    assert Store(path).favourite_teams() == ["Arsenal"]


# --- concurrency -------------------------------------------------------------
def test_database_uses_wal_so_reads_and_writes_do_not_block(store):
    with store._connect() as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


def test_a_background_write_does_not_block_a_read(store):
    """The forecast thread writes while the UI reads; that must not deadlock."""
    import threading

    store.save_prediction(prediction())
    errors = []

    def writer():
        try:
            for i in range(25):
                store.save_predictions([prediction(home=f"Club {i}", away="Rivals")])
        except Exception as exc:                      # pragma: no cover - failure path
            errors.append(exc)

    def reader():
        try:
            for _ in range(25):
                store.history(limit=50)
                store.scorecard()
        except Exception as exc:                      # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in threads), "a thread was still blocked"
    assert not errors, errors


# --- self-benchmarking scorecard ---------------------------------------------
def _settled(store, calls):
    """calls: (home, away, probs, result, home_goals, away_goals, markets)"""
    for home, away, probs, result, hg, ag, markets in calls:
        pred = prediction(home=home, away=away, probs=probs)
        if markets:
            pred["markets"] = markets
        store.save_prediction(pred)
        store.settle([{"league": "E0", "home_team": home, "away_team": away,
                       "result": result, "home_goals": hg, "away_goals": ag}])


def test_scorecard_reports_baselines_over_the_same_matches(store):
    """56% means nothing without knowing what guessing would have scored."""
    _settled(store, [
        ("A", "B", (0.6, 0.25, 0.15), "H", 2, 0, None),
        ("C", "D", (0.5, 0.3, 0.2), "H", 1, 0, None),
        ("E", "F", (0.2, 0.3, 0.5), "A", 0, 2, None),
        ("G", "H", (0.55, 0.25, 0.2), "D", 1, 1, None),
    ])
    card = store.scorecard()
    base = card["baselines"]
    assert set(base) == {"base_rates", "always_home"}
    # Two of four finished as home wins, so always-home scores 0.5 here.
    assert base["always_home"]["accuracy"] == pytest.approx(0.5)
    assert 0 < base["base_rates"]["rps"] < 1
    assert card["accuracy"] >= 0


def test_baselines_use_the_graded_set_not_the_whole_league(store):
    """An away-heavy sample must move the baseline with it."""
    _settled(store, [(f"H{i}", f"A{i}", (0.4, 0.3, 0.3), "A", 0, 1, None) for i in range(5)])
    assert store.scorecard()["baselines"]["always_home"]["accuracy"] == 0.0


# --- goals and exact score, graded separately --------------------------------
def markets(over, btts, score):
    return {"over_2_5": over, "both_score": btts,
            "scorelines": [{"home": score[0], "away": score[1], "probability": 0.12}]}


def test_goal_markets_are_graded_on_their_own(store):
    _settled(store, [
        ("A", "B", (0.6, 0.25, 0.15), "H", 3, 1, markets(0.8, 0.7, (2, 1))),   # over, btts
        ("C", "D", (0.5, 0.3, 0.2), "H", 1, 0, markets(0.3, 0.2, (1, 0))),     # under, no btts
        ("E", "F", (0.4, 0.3, 0.3), "D", 0, 0, markets(0.2, 0.3, (1, 1))),     # under, no btts
        ("G", "H", (0.4, 0.3, 0.3), "A", 1, 3, markets(0.7, 0.6, (1, 2))),     # over, btts
        ("I", "J", (0.4, 0.3, 0.3), "H", 2, 0, markets(0.4, 0.4, (2, 0))),     # under, no btts
    ])
    m = store.scorecard()["markets"]
    assert m["over_2_5"]["graded"] == 5
    assert m["over_2_5"]["accuracy"] == 1.0, "every over/under call was right"
    assert m["both_score"]["accuracy"] == 1.0
    assert m["over_2_5"]["actual_rate"] == pytest.approx(0.4)
    assert 0 <= m["over_2_5"]["brier"] <= 0.25


def test_exact_score_is_graded_too(store):
    _settled(store, [
        ("A", "B", (0.6, 0.25, 0.15), "H", 2, 1, markets(0.5, 0.5, (2, 1))),   # exact hit
        ("C", "D", (0.6, 0.25, 0.15), "H", 3, 0, markets(0.5, 0.5, (1, 0))),   # miss
        ("E", "F", (0.6, 0.25, 0.15), "H", 1, 0, markets(0.5, 0.5, (1, 0))),   # exact hit
        ("G", "H", (0.4, 0.3, 0.3), "A", 0, 1, markets(0.5, 0.5, (1, 1))),     # miss
        ("I", "J", (0.4, 0.3, 0.3), "D", 1, 1, markets(0.5, 0.5, (1, 1))),     # exact hit
    ])
    exact = store.scorecard()["markets"]["exact_score"]
    assert exact["graded"] == 5 and exact["correct"] == 3


def test_markets_are_absent_until_there_is_something_to_grade(store):
    _settled(store, [("A", "B", (0.6, 0.25, 0.15), "H", 2, 0, None)])
    assert store.scorecard()["markets"] == {}


def test_an_older_database_gains_the_new_columns(tmp_path):
    """Existing installs must keep their history, not start over."""
    import sqlite3

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, league TEXT NOT NULL,
            home_team TEXT NOT NULL, away_team TEXT NOT NULL, match_date TEXT NOT NULL,
            created_at TEXT NOT NULL, model_name TEXT, p_home REAL NOT NULL,
            p_draw REAL NOT NULL, p_away REAL NOT NULL, predicted TEXT NOT NULL,
            used_odds INTEGER NOT NULL DEFAULT 0, result TEXT, home_goals INTEGER,
            away_goals INTEGER, settled_at TEXT,
            UNIQUE (league, home_team, away_team, match_date));
        INSERT INTO predictions (league, home_team, away_team, match_date, created_at,
                                 p_home, p_draw, p_away, predicted)
        VALUES ('E0','Arsenal','Chelsea','2026-01-01','2026-01-01T00:00:00+00:00',
                0.6,0.25,0.15,'H');""")
    con.commit()
    con.close()

    store = Store(path)
    rows = store.history()
    assert len(rows) == 1 and rows[0]["home_team"] == "Arsenal", "history was lost"
    assert rows[0]["p_over25"] is None
    store.save_prediction({**prediction(home="Arsenal", away="Chelsea", date="2026-01-01"),
                           "markets": markets(0.6, 0.5, (2, 1))})
    assert store.history()[0]["p_over25"] == pytest.approx(0.6)
