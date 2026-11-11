"""Scoring upcoming fixtures with a trained model bundle."""
from __future__ import annotations

import logging
from datetime import date as date_cls
from datetime import datetime, timezone

import joblib
import pandas as pd

from .config import CLASS_LABELS, LEAGUES, model_candidates, model_path
from .features import FEATURE_VERSION, FeatureBuilder

log = logging.getLogger(__name__)


class UnknownTeamError(ValueError):
    pass


class Predictor:
    """Loads a trained bundle and scores fixtures that have not been played."""

    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.builder: FeatureBuilder = bundle["builder"]
        self.features: list[str] = bundle["features"]
        self.classes: list[str] = list(bundle["classes"])
        self.meta = {k: v for k, v in bundle.items()
                     if k not in {"pipeline", "builder"}}

    @classmethod
    def load(cls, name: str = "model.joblib") -> "Predictor":
        """The most recently trained model this code can actually use.

        An installed app can hold two: the one shipped inside it, and one it
        retrained into its own data folder. Preferring the local copy by location
        meant a leftover from an older version shadowed every model shipped
        after it - and crashed on first use, because its pickled feature builder
        predates the current one. So each candidate is checked for a matching
        FEATURE_VERSION, and the newest compatible one wins.
        """
        usable, rejected = [], []
        for path in model_candidates(name):
            try:
                bundle = joblib.load(path)
            except Exception as exc:                       # truncated, corrupt, ...
                rejected.append(f"{path}: unreadable ({exc.__class__.__name__})")
                continue
            version = bundle.get("feature_version")
            if version != FEATURE_VERSION:
                rejected.append(f"{path}: built for feature version {version}, "
                                f"this code needs {FEATURE_VERSION}")
                continue
            usable.append((str(bundle.get("trained_at") or ""), path, bundle))

        for reason in rejected:
            log.warning("ignoring model %s", reason)
        if not usable:
            detail = "; ".join(rejected) or f"{model_path(name)} missing"
            raise FileNotFoundError(f"no usable model ({detail}) - run `python -m src.train`")

        usable.sort(key=lambda item: item[0], reverse=True)
        _, path, bundle = usable[0]
        predictor = cls(bundle)
        predictor.path = path
        return predictor

    # -- introspection -------------------------------------------------------
    def leagues(self) -> list[dict]:
        return [{"code": c, "name": LEAGUES.get(c, c)} for c in self.builder.leagues()]

    def teams(self, league: str | None = None) -> list[str]:
        return self.builder.known_teams(league)

    def team_form(self, league: str, team: str) -> dict:
        st = self.builder.state(league, team)
        recent = list(st.recent)[-5:]
        return {
            "team": team,
            "league": league,
            "elo": round(self.builder.elo.rating(league, team), 1),
            "matches_recorded": st.played,
            "last_match": st.last_date.date().isoformat() if st.last_date is not None else None,
            "last5_points": [m["points"] for m in recent],
            "last5_ppg": round(sum(m["points"] for m in recent) / len(recent), 2) if recent else None,
            "last5_goals_for": sum(m["gf"] for m in recent) if recent else None,
            "last5_goals_against": sum(m["ga"] for m in recent) if recent else None,
        }

    # -- prediction ----------------------------------------------------------
    def predict(self, league: str, home_team: str, away_team: str,
                match_date: str | date_cls | None = None,
                odds: dict | None = None) -> dict:
        known = set(self.builder.known_teams(league))
        missing = [t for t in (home_team, away_team) if t not in known]
        if missing:
            raise UnknownTeamError(
                f"unknown team(s) for {league}: {', '.join(missing)}")
        if home_team == away_team:
            raise ValueError("a team cannot play itself")

        when = pd.Timestamp(match_date) if match_date else pd.Timestamp(datetime.now(timezone.utc).date())
        row = self.builder.features(league, home_team, away_team, when, odds)
        X = pd.DataFrame([row]).reindex(columns=self.features)
        proba = self.pipeline.predict_proba(X)[0]

        probs = {c: float(p) for c, p in zip(self.classes, proba)}
        best = max(probs, key=probs.get)
        return {
            "league": league,
            "home_team": home_team,
            "away_team": away_team,
            "date": when.date().isoformat(),
            "prediction": best,
            "prediction_label": CLASS_LABELS[best],
            "confidence": round(probs[best], 4),
            "probabilities": {
                "home_win": round(probs.get("H", 0.0), 4),
                "draw": round(probs.get("D", 0.0), 4),
                "away_win": round(probs.get("A", 0.0), 4),
            },
            "fair_odds": {
                k: (round(1 / v, 2) if v > 1e-9 else None)
                for k, v in {
                    "home_win": probs.get("H", 0.0),
                    "draw": probs.get("D", 0.0),
                    "away_win": probs.get("A", 0.0),
                }.items()
            },
            "markets": self.builder.score_forecast(
                league, home_team, away_team,
                (probs.get("H", 0.0), probs.get("D", 0.0), probs.get("A", 0.0))),
            "context": self._context(row, home_team, away_team),
            "model": {"name": self.meta.get("model_name"),
                      "uses_market_odds": self.meta.get("use_odds", False),
                      "trained_at": self.meta.get("trained_at")},
        }

    @staticmethod
    def _context(row: dict, home: str, away: str) -> dict:
        def r(key, nd=2):
            v = row.get(key)
            return None if v is None or pd.isna(v) else round(float(v), nd)

        return {
            "elo": {home: r("elo_home", 1), away: r("elo_away", 1),
                    "difference": r("elo_diff", 1)},
            "form_ppg_last5": {home: r("ppg5_home"), away: r("ppg5_away")},
            "goals_for_last5": {home: r("gf5_home"), away: r("gf5_away")},
            "goals_against_last5": {home: r("ga5_home"), away: r("ga5_away")},
            "venue_ppg_last5": {f"{home} at home": r("venue_ppg_home"),
                                f"{away} away": r("venue_ppg_away")},
            "head_to_head": {"meetings": r("h2h_played", 0),
                             f"{home} ppg": r("h2h_ppg_home"),
                             f"{home} goal diff": r("h2h_gd_home")},
            "rest_days": {home: r("rest_days_home", 0), away: r("rest_days_away", 0)},
        }

    def forecast_batch(self, fixtures: list) -> list[dict | None]:
        """Probabilities for many fixtures in one model call.

        The fixture list is ~1,500 unplayed matches across five leagues. Scoring
        them one at a time costs a DataFrame and a predict_proba per match
        (~1.6ms each); building every feature row first and calling the model
        once is what makes predicting the whole season on the fly affordable.

        Entries that cannot be scored - unknown club, a side playing itself -
        come back as None in the same position, so callers can zip results.
        """
        rows, positions = [], []
        for i, fx in enumerate(fixtures):
            known = set(self.builder.known_teams(fx.league))
            if (fx.home_team not in known or fx.away_team not in known
                    or fx.home_team == fx.away_team):
                continue
            rows.append(self.builder.features(fx.league, fx.home_team, fx.away_team,
                                              pd.Timestamp(fx.date), fx.odds))
            positions.append(i)

        out: list[dict | None] = [None] * len(fixtures)
        if not rows:
            return out
        proba = self.pipeline.predict_proba(
            pd.DataFrame(rows).reindex(columns=self.features))
        for i, row in zip(positions, proba):
            probs = {c: float(v) for c, v in zip(self.classes, row)}
            best = max(probs, key=probs.get)
            fx = fixtures[i]
            out[i] = {
                "prediction": best,
                "prediction_label": CLASS_LABELS[best],
                "confidence": round(probs[best], 4),
                "probabilities": {"home_win": round(probs.get("H", 0.0), 4),
                                  "draw": round(probs.get("D", 0.0), 4),
                                  "away_win": round(probs.get("A", 0.0), 4)},
                "markets": self.builder.score_forecast(
                    fx.league, fx.home_team, fx.away_team,
                    (probs.get("H", 0.0), probs.get("D", 0.0), probs.get("A", 0.0))),
            }
        return out

    def predict_many(self, fixtures: list[dict]) -> list[dict]:
        out = []
        for fx in fixtures:
            try:
                out.append(self.predict(fx["league"], fx["home_team"], fx["away_team"],
                                        fx.get("date"), fx.get("odds")))
            except (UnknownTeamError, ValueError) as exc:
                out.append({**fx, "error": str(exc)})
        return out


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Predict a match outcome.")
    ap.add_argument("league")
    ap.add_argument("home_team")
    ap.add_argument("away_team")
    ap.add_argument("--date", default=None)
    ap.add_argument("--odds", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"))
    ap.add_argument("--model", default="model.joblib")
    args = ap.parse_args()

    odds = dict(zip(("home", "draw", "away"), args.odds)) if args.odds else None
    p = Predictor.load(args.model)
    print(json.dumps(p.predict(args.league, args.home_team, args.away_team,
                               args.date, odds), indent=2))


if __name__ == "__main__":
    main()
