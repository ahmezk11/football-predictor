"""Local SQLite store: followed teams and a record of every prediction made.

This is what turns a model into a personal app. Predictions are saved when they
are made, then settled against real results once the matches are played, so the
app can show how it actually did on *your* fixtures rather than quoting a
held-out score from training time.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import evaluate as ev
from .config import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS favourites (
    league     TEXT NOT NULL,
    team       TEXT NOT NULL,
    added_at   TEXT NOT NULL,
    PRIMARY KEY (league, team)
);

CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    league       TEXT NOT NULL,
    home_team    TEXT NOT NULL,
    away_team    TEXT NOT NULL,
    match_date   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    model_name   TEXT,
    p_home       REAL NOT NULL,
    p_draw       REAL NOT NULL,
    p_away       REAL NOT NULL,
    predicted    TEXT NOT NULL,
    used_odds    INTEGER NOT NULL DEFAULT 0,
    -- from the goal model's score distribution, graded separately from the result
    p_over25     REAL,
    p_btts       REAL,
    top_home     INTEGER,
    top_away     INTEGER,
    -- filled in once the match has been played
    result       TEXT,
    home_goals   INTEGER,
    away_goals   INTEGER,
    settled_at   TEXT,
    UNIQUE (league, home_team, away_team, match_date)
);

CREATE INDEX IF NOT EXISTS idx_pred_open ON predictions (result, match_date);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def default_db_path() -> Path:
    """Where the app keeps its data.

    Overridable so a packaged desktop build can write to the user's Application
    Support directory instead of inside the app bundle, which is read-only.
    """
    override = os.environ.get("PREDICTOR_DATA_DIR")
    base = Path(override) if override else ROOT / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base / "app.db"


def _market_fields(pred: dict) -> tuple:
    """(over 2.5, both score, most likely home goals, away goals) or blanks."""
    m = pred.get("markets") or {}
    top = (m.get("scorelines") or [{}])[0]
    return (m.get("over_2_5"), m.get("both_score"), top.get("home"), top.get("away"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)
            self._add_missing_columns(con)

    @staticmethod
    def _add_missing_columns(con) -> None:
        """Bring an older database up to date without touching its contents."""
        have = {row["name"] for row in con.execute("PRAGMA table_info(predictions)")}
        for name, kind in (("p_over25", "REAL"), ("p_btts", "REAL"),
                           ("top_home", "INTEGER"), ("top_away", "INTEGER")):
            if name not in have:
                con.execute(f"ALTER TABLE predictions ADD COLUMN {name} {kind}")

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        # Background forecasting writes while the UI reads. Under the default
        # rollback journal those serialise; WAL lets them run concurrently, and
        # busy_timeout waits out a writer instead of failing immediately.
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=5000")
        con.execute("PRAGMA synchronous=NORMAL")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # -- favourites ----------------------------------------------------------
    def follow(self, league: str, team: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO favourites (league, team, added_at) VALUES (?,?,?)",
                (league, team, _now()))

    def unfollow(self, league: str, team: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM favourites WHERE league=? AND team=?", (league, team))

    def favourites(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT league, team, added_at FROM favourites ORDER BY league, team")
            return [dict(r) for r in rows]

    def favourite_teams(self) -> list[str]:
        return [f["team"] for f in self.favourites()]

    # -- predictions ---------------------------------------------------------
    def save_predictions(self, preds: list[dict]) -> int:
        """Record many predictions over one connection. Returns how many were written."""
        if not preds:
            return 0
        rows = [(pr["league"], pr["home_team"], pr["away_team"], pr["date"], _now(),
                 (pr.get("model") or {}).get("name"),
                 pr["probabilities"]["home_win"], pr["probabilities"]["draw"],
                 pr["probabilities"]["away_win"], pr["prediction"],
                 int(bool((pr.get("model") or {}).get("uses_market_odds"))),
                 *_market_fields(pr))
                for pr in preds]
        with self._connect() as con:
            con.executemany(
                """INSERT INTO predictions
                     (league, home_team, away_team, match_date, created_at, model_name,
                      p_home, p_draw, p_away, predicted, used_odds,
                      p_over25, p_btts, top_home, top_away)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (league, home_team, away_team, match_date)
                   DO UPDATE SET created_at=excluded.created_at,
                                 model_name=excluded.model_name,
                                 p_home=excluded.p_home, p_draw=excluded.p_draw,
                                 p_away=excluded.p_away, predicted=excluded.predicted,
                                 used_odds=excluded.used_odds,
                                 p_over25=excluded.p_over25, p_btts=excluded.p_btts,
                                 top_home=excluded.top_home, top_away=excluded.top_away
                   WHERE predictions.result IS NULL""", rows)
        return len(rows)

    def save_prediction(self, pred: dict) -> int:
        """Record a prediction. Re-predicting the same fixture updates it."""
        p = pred["probabilities"]
        with self._connect() as con:
            cur = con.execute(
                """INSERT INTO predictions
                     (league, home_team, away_team, match_date, created_at, model_name,
                      p_home, p_draw, p_away, predicted, used_odds,
                      p_over25, p_btts, top_home, top_away)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (league, home_team, away_team, match_date)
                   DO UPDATE SET created_at=excluded.created_at,
                                 model_name=excluded.model_name,
                                 p_home=excluded.p_home, p_draw=excluded.p_draw,
                                 p_away=excluded.p_away, predicted=excluded.predicted,
                                 used_odds=excluded.used_odds,
                                 p_over25=excluded.p_over25, p_btts=excluded.p_btts,
                                 top_home=excluded.top_home, top_away=excluded.top_away""",
                (pred["league"], pred["home_team"], pred["away_team"], pred["date"],
                 _now(), (pred.get("model") or {}).get("name"),
                 p["home_win"], p["draw"], p["away_win"], pred["prediction"],
                 int(bool((pred.get("model") or {}).get("uses_market_odds"))),
                 *_market_fields(pred)))
            if cur.lastrowid:
                return cur.lastrowid
            row = con.execute(
                """SELECT id FROM predictions
                   WHERE league=? AND home_team=? AND away_team=? AND match_date=?""",
                (pred["league"], pred["home_team"], pred["away_team"],
                 pred["date"])).fetchone()
            return row["id"] if row else -1

    def open_predictions(self) -> list[dict]:
        """Saved predictions for matches that have not been settled yet."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM predictions WHERE result IS NULL ORDER BY match_date")
            return [dict(r) for r in rows]

    def settle(self, results: list) -> int:
        """Attach real outcomes to open predictions. Returns how many settled.

        Matched on league + both teams, not on date: fixtures get moved for
        television, and a prediction made for a postponed match is still a
        prediction about that match.
        """
        settled = 0
        with self._connect() as con:
            for fx in results:
                outcome = fx.result if hasattr(fx, "result") else fx.get("result")
                if outcome is None:
                    continue
                home = fx.home_team if hasattr(fx, "home_team") else fx["home_team"]
                away = fx.away_team if hasattr(fx, "away_team") else fx["away_team"]
                league = fx.league if hasattr(fx, "league") else fx["league"]
                hg = fx.home_goals if hasattr(fx, "home_goals") else fx.get("home_goals")
                ag = fx.away_goals if hasattr(fx, "away_goals") else fx.get("away_goals")
                cur = con.execute(
                    """UPDATE predictions
                          SET result=?, home_goals=?, away_goals=?, settled_at=?
                        WHERE league=? AND home_team=? AND away_team=? AND result IS NULL""",
                    (outcome, hg, ag, _now(), league, home, away))
                settled += cur.rowcount
        return settled

    def history(self, limit: int = 200, settled_only: bool = False) -> list[dict]:
        where = "WHERE result IS NOT NULL" if settled_only else ""
        with self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM predictions {where} "
                "ORDER BY match_date DESC, id DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

    def predictions_for(self, league: str) -> dict[tuple[str, str], dict]:
        """Saved predictions in one league, keyed by (home, away).

        Keyed on the teams rather than the date so a rescheduled fixture still
        shows the call that was already made for it.
        """
        with self._connect() as con:
            rows = con.execute("SELECT * FROM predictions WHERE league=?", (league,))
            return {(r["home_team"], r["away_team"]): dict(r) for r in rows}

    def scorecard(self) -> dict:
        """How the forecasts did - next to what guessing would have done.

        A bare "56% correct" is not interpretable on its own: home wins alone are
        about 45% of matches, and these are the fixtures the user happened to ask
        about, not a balanced sample. The same baselines the training report uses
        are computed over exactly this graded set.
        """
        settled = self.history(limit=100000, settled_only=True)
        if not settled:
            return {"settled": 0, "pending": len(self.open_predictions()),
                    "message": "no settled predictions yet"}

        classes = ["A", "D", "H"]
        proba = np.array([[r["p_away"], r["p_draw"], r["p_home"]] for r in settled])
        truth = [r["result"] for r in settled]
        metrics = ev.evaluate(truth, proba, classes)

        # Same matches, same metrics, no model.
        mix = [truth.count(c) / len(truth) for c in classes]
        base_rates = ev.evaluate(truth, np.tile(mix, (len(truth), 1)), classes)
        always_home = ev.evaluate(truth, np.tile([0.0005, 0.0005, 0.999], (len(truth), 1)),
                                  classes)

        by_team: dict[str, dict] = {}
        for r in settled:
            hit = int(r["predicted"] == r["result"])
            for team in (r["home_team"], r["away_team"]):
                b = by_team.setdefault(team, {"team": team, "n": 0, "correct": 0})
                b["n"] += 1
                b["correct"] += hit
        for b in by_team.values():
            b["accuracy"] = round(b["correct"] / b["n"], 3)

        return {
            "settled": len(settled),
            "pending": len(self.open_predictions()),
            "correct": int(sum(r["predicted"] == r["result"] for r in settled)),
            "accuracy": round(metrics["accuracy"], 4),
            "rps": round(metrics["rps"], 4),
            "log_loss": round(metrics["log_loss"], 4),
            "brier": round(metrics["brier"], 4),
            "baselines": {
                "base_rates": {"accuracy": round(base_rates["accuracy"], 4),
                               "rps": round(base_rates["rps"], 4)},
                "always_home": {"accuracy": round(always_home["accuracy"], 4),
                                "rps": round(always_home["rps"], 4)},
            },
            "markets": self._market_scorecard(settled),
            "by_outcome": {
                cls: {"predicted": int(sum(r["predicted"] == cls for r in settled)),
                      "actual": int(sum(r["result"] == cls for r in settled))}
                for cls in ("H", "D", "A")},
            "by_team": sorted(by_team.values(), key=lambda b: -b["n"])[:20],
            "first_prediction": min(r["match_date"] for r in settled),
            "last_prediction": max(r["match_date"] for r in settled),
        }

    @staticmethod
    def _market_scorecard(settled: list[dict]) -> dict:
        """Goals and exact score, graded on their own.

        Each is a separate claim from the same score distribution, so each is an
        independent check on whether it is calibrated - the outcome probabilities
        can look fine while the goal model is off.
        """
        def graded(field):
            return [r for r in settled
                    if r.get(field) is not None
                    and r["home_goals"] is not None and r["away_goals"] is not None]

        out: dict[str, dict] = {}
        for field, key, actual_of in (
            ("p_over25", "over_2_5", lambda r: (r["home_goals"] + r["away_goals"]) > 2.5),
            ("p_btts", "both_score", lambda r: r["home_goals"] > 0 and r["away_goals"] > 0),
        ):
            rows = graded(field)
            if len(rows) < 5:
                continue
            probs = np.array([r[field] for r in rows])
            hits = np.array([float(actual_of(r)) for r in rows])
            out[key] = {
                "graded": len(rows),
                "accuracy": round(float(((probs > 0.5) == (hits > 0.5)).mean()), 4),
                "brier": round(float(np.mean((probs - hits) ** 2)), 4),
                "predicted_rate": round(float(probs.mean()), 4),
                "actual_rate": round(float(hits.mean()), 4),
            }

        exact = [r for r in settled if r.get("top_home") is not None
                 and r["home_goals"] is not None]
        if len(exact) >= 5:
            right = sum(r["top_home"] == r["home_goals"] and r["top_away"] == r["away_goals"]
                        for r in exact)
            out["exact_score"] = {
                "graded": len(exact),
                "correct": int(right),
                "accuracy": round(right / len(exact), 4),
                # For scale: naming the single most likely scoreline is hard, and
                # about one in nine matches finishes on the most common one.
                "note": "the most likely scoreline, called exactly",
            }
        return out

    def forget(self, prediction_id: int) -> bool:
        with self._connect() as con:
            return con.execute("DELETE FROM predictions WHERE id=?",
                               (prediction_id,)).rowcount > 0

    # -- meta ----------------------------------------------------------------
    def get_meta(self, key: str, default=None):
        with self._connect() as con:
            row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO meta (key, value) VALUES (?,?) "
                "ON CONFLICT (key) DO UPDATE SET value=excluded.value", (key, value))

    def stats(self) -> dict:
        with self._connect() as con:
            n_pred = con.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
            n_fav = con.execute("SELECT COUNT(*) c FROM favourites").fetchone()["c"]
        return {"database": str(self.path), "predictions": n_pred, "favourites": n_fav,
                "last_data_refresh": self.get_meta("last_data_refresh"),
                "last_retrain": self.get_meta("last_retrain")}
