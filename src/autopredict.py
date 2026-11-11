"""Automatic season forecasts: every unplayed fixture, every league, no buttons.

Two different things happen to a fixture here, on purpose:

  forecast   computed fresh on every view, for the whole season. Cheap - one
             batched model call covers all ~1,500 unplayed matches - and never
             stale, because it always reflects the current model and form.

  lock in    saved to the prediction history, so it is graded once the result
             arrives. Only fixtures in the next LOCK_IN_DAYS are locked in.
             Saving the whole season would bury the history under forecasts for
             matches months away, and freeze them long before kick-off. A
             locked-in forecast keeps updating until the match date passes;
             after that it is left alone, so a result can never be graded
             against a forecast made after the match was played.
"""
from __future__ import annotations

import logging
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone

from .config import CLASS_LABELS

log = logging.getLogger(__name__)

LOCK_IN_DAYS = 7


def _today() -> date_cls:
    return datetime.now(timezone.utc).date()


def status_of(fx, today: date_cls) -> str:
    """Where a fixture stands: played, awaiting a result, upcoming, or unscorable."""
    if not fx.predictable:
        return "unpredictable"
    if fx.played:
        return "played"
    try:
        when = date_cls.fromisoformat(fx.date)
    except ValueError:
        return "upcoming"
    # The match date has passed but the feed has not posted the result yet.
    return "awaiting" if when < today else "upcoming"


def _saved_view(prior: dict) -> dict:
    return {
        "prediction": prior["predicted"],
        "prediction_label": CLASS_LABELS[prior["predicted"]],
        "probabilities": {"home_win": prior["p_home"], "draw": prior["p_draw"],
                          "away_win": prior["p_away"]},
        "locked_in": True,
        "correct": None if not prior["result"] else prior["predicted"] == prior["result"],
    }


def forecast_league(predictor, feed, store, league: str, *,
                    today: date_cls | None = None,
                    lock_in_days: int = LOCK_IN_DAYS) -> dict:
    """The league's season by matchday, with every unplayed fixture forecast."""
    today = today or _today()
    horizon = today + timedelta(days=lock_in_days)
    groups = feed.matchdays(league)
    saved = store.predictions_for(league)

    fixtures = [fx for g in groups for fx in g["matches"]]
    statuses = [status_of(fx, today) for fx in fixtures]
    to_score = [fx for fx, st in zip(fixtures, statuses) if st in ("upcoming", "awaiting")]
    scores = dict(zip(map(id, to_score), predictor.forecast_batch(to_score)))

    model_meta = {"name": predictor.meta.get("model_name"),
                  "uses_market_odds": predictor.meta.get("use_odds", False)}
    lock_in, counts = [], {"forecast": 0, "locked_in": 0, "awaiting": 0,
                           "played": 0, "unpredictable": 0}
    rows_by_fixture = {}

    for fx, st in zip(fixtures, statuses):
        row = {**fx.as_dict(), "status": st}
        prior = saved.get((fx.home_team, fx.away_team))
        forecast = scores.get(id(fx))

        if st == "played":
            counts["played"] += 1
            row.update({"home_goals": fx.home_goals, "away_goals": fx.away_goals,
                        "result": fx.result})
            if prior:
                row["prediction"] = _saved_view(prior)

        elif st == "awaiting":
            counts["awaiting"] += 1
            # Prefer what was locked in before the match; never lock in afterwards.
            if prior:
                row["prediction"] = _saved_view(prior)
            elif forecast:
                row["prediction"] = {**forecast, "locked_in": False, "correct": None}

        elif st == "upcoming" and forecast:
            counts["forecast"] += 1
            when = date_cls.fromisoformat(fx.date) if fx.date else today
            locked = when <= horizon
            row["prediction"] = {**forecast, "locked_in": locked, "correct": None}
            if locked:
                counts["locked_in"] += 1
                lock_in.append({"league": fx.league, "home_team": fx.home_team,
                                "away_team": fx.away_team, "date": fx.date,
                                "model": model_meta, **forecast})

        elif st == "unpredictable":
            counts["unpredictable"] += 1

        rows_by_fixture[id(fx)] = row

    store.save_predictions(lock_in)

    matchdays = [{**{k: v for k, v in g.items() if k != "matches"},
                  "matches": [rows_by_fixture[id(fx)] for fx in g["matches"]]}
                 for g in groups]
    # Open on the first round with a match still to come. "First unfinished"
    # is not enough: when the feed is slow to post results, that lands on a
    # round of matches that have already kicked off.
    current = (next((g["round"] for g in matchdays
                     if any(m["status"] == "upcoming" for m in g["matches"])), None)
               or next((g["round"] for g in groups if g["played"] < g["total"]), None)
               or (groups[-1]["round"] if groups else None))
    return {"league": league, "current_round": current, "lock_in_days": lock_in_days,
            "counts": counts, "matchdays": matchdays,
            "feed_error": feed.errors.get(league)}


def forecast_all(predictor, feed, store, **kw) -> dict[str, dict]:
    """Run every league. Used at startup and after a data refresh."""
    out = {}
    for lg in predictor.leagues():
        try:
            out[lg["code"]] = forecast_league(predictor, feed, store, lg["code"], **kw)["counts"]
        except Exception as exc:  # one league's feed failing must not stop the rest
            log.warning("forecast failed for %s: %s", lg["code"], exc)
            out[lg["code"]] = {"error": str(exc)}
    return out
